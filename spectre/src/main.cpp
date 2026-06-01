#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "llama-cpp.h"

static inline const char *log_level_to_string(enum ggml_log_level level) {
  switch (level) {
  case GGML_LOG_LEVEL_DEBUG:
    return "[DEBUG] ";
  case GGML_LOG_LEVEL_CONT:
  case GGML_LOG_LEVEL_INFO:
    return "[INFO] ";
  case GGML_LOG_LEVEL_WARN:
    return "[WARN] ";
  case GGML_LOG_LEVEL_ERROR:
    return "[ERROR] ";
  case GGML_LOG_LEVEL_NONE:
  default:
    return "";
  }
}

template <typename... Args>
static inline void print(enum ggml_log_level level, std::string_view fmt, Args &&...args) {
  try {
    auto message = std::vformat(fmt, std::make_format_args(args...));
    std::cout << log_level_to_string(level) << message << '\n';
  } catch (const std::format_error &e) {
    std::cout << log_level_to_string(GGML_LOG_LEVEL_ERROR)
              << "print(): std::format_error: " << e.what()
              << "  fmt=\"" << fmt << "\"\n";
  }
}

struct Parameters {
  // the number of layers to store in VRAM (<0 means all layers)
  int32_t ngl = -1;

  // text context, 0 = from model (TODO if speculative we use the same context window for both models)
  int32_t ctx = 0;

  // Updates logit_i' = logit_i / temp.
  // When temp <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
  float temp = 0.8f;

  // greedy sampler (select the token with the highest prob)
  bool greedy = false;

  // the n-gram cache implementation maintains statistics about short n-gram sequences.
  bool ngram = false;
  // length of the lookup pattern (n-gram). lower values fire more often on short prompts,
  // higher values reduce false positives on long prompts. llama.cpp defaults to 12, which only
  // hits in long contexts; 2-4 is a good range for benchmarks under a few hundred tokens.
  int32_t n_gram_size = 2;
  // maximum length of the proposed draft (m-gram) following an n-gram hit.
  int32_t m_gram_size = 12;

  // "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
  float top_p = 0.90f;
  int32_t top_k = 40;

  std::string prompt = "Write a Python class called Record with 20 properties: id, name, email, phone, address, city, state, zip_code, country, age, salary, department, role, manager, status, created_at, updated_at, is_active, score, notes. For each property implement a getter and setter using exactly this pattern: def get_X(self): return self._X and def set_X(self, value): self._X = value";

  // speculative decoding parameters
  int64_t n_min = 0;     // minimum number of draft tokens to use for speculative decoding
  int64_t n_max = 8;     // maximum number of tokens to draft during speculative decoding
  int64_t n_accept = 0;  // number of tokens accepted by the target model.
  int64_t n_drafted = 0; // number of tokens drafted by the draft model.

  // used to determine end of generation
  bool has_eos = false;

  std::string draft_model_path;
  std::string target_model_path;

  // reproducibility / structured output
  uint32_t seed = 1234;
  std::string results_dir = "results/spectre";
  std::string run_id; // auto-generated if empty

  // hard cap on generated tokens (0 = unlimited, stop only on EOS / KV exhaustion)
  int64_t n_predict = 0;

  bool draft_speculative_decoding_is_enabled() {
    if (!this->draft_model_path.empty()) {
      return true;
    }
    return false;
  }
};

class TeeBuf : public std::streambuf {
public:
  TeeBuf(std::streambuf *sb1, std::streambuf *sb2) : sb1(sb1), sb2(sb2) {}

protected:
  virtual int overflow(int c) override {
    if (c == EOF) return !EOF;
    if (sb1->sputc(c) == EOF || sb2->sputc(c) == EOF) return EOF;
    return c;
  }

  virtual int sync() override {
    return (sb1->pubsync() == 0 && sb2->pubsync() == 0) ? 0 : -1;
  }

private:
  std::streambuf *sb1;
  std::streambuf *sb2;
};

struct RoundSummary {
  int n_drafted;
  int n_accepted_drafts;
  int rejected_pos; // -1 if all draft tokens accepted
};

class RunRecorder {
public:
  RunRecorder(const std::string &results_dir, const std::string &run_id,
              const std::string &started_at_iso, const Parameters &p)
      : run_dir(std::filesystem::path(results_dir) / run_id),
        run_id_(run_id),
        started_at_(started_at_iso),
        params_snapshot_(p) {
    std::filesystem::create_directories(this->run_dir);

    this->tokens.open(this->run_dir / "tokens.csv");
    if (!this->tokens) {
      throw std::runtime_error(std::format("failed to open {}", (this->run_dir / "tokens.csv").string()));
    }
    this->tokens << "step,call,source,pos_in_draft,token_id,p_target,p_draft,logit,logprob\n";

    this->write_meta(/*complete=*/false, 0, 0, 0, 0, 0.0, 0.0);
  }

  void record_token(
      int call,
      const char *source,
      int pos_in_draft,
      int token_id,
      double p_target,
      double p_draft,
      double logit,
      double logprob) {
    this->tokens << this->step << ',' << call << ',' << source << ','
                 << pos_in_draft << ',' << token_id << ','
                 << fmt_double(p_target) << ','
                 << fmt_double(p_draft) << ','
                 << fmt_double(logit) << ','
                 << fmt_double(logprob) << '\n'
                 << std::flush;
    this->step += 1;
  }

  void record_round(int n_drafted, int n_accepted_drafts, int rejected_pos) {
    this->rounds.push_back({n_drafted, n_accepted_drafts, rejected_pos});
  }

  void finalize(
      int64_t n_decoded_tokens,
      int64_t n_drafted,
      int64_t n_accepted_drafts,
      int64_t n_bonus_samples,
      double prompt_ms,
      double decode_ms) {
    this->tokens.flush();
    this->tokens.close();
    this->write_meta(/*complete=*/true, n_decoded_tokens, n_drafted,
                     n_accepted_drafts, n_bonus_samples, prompt_ms, decode_ms);
  }

  const std::filesystem::path &dir() const { return this->run_dir; }

  static std::string iso_timestamp(bool compact = false) {
    using clock = std::chrono::system_clock;
    auto now = clock::now();
    std::time_t t = clock::to_time_t(now);
    std::tm tm_local{};
    localtime_r(&t, &tm_local);
    std::ostringstream os;
    if (compact) {
      os << std::put_time(&tm_local, "%Y%m%d-%H%M%S");
    } else {
      os << std::put_time(&tm_local, "%Y-%m-%dT%H:%M:%S");
    }
    return os.str();
  }

private:
  void write_meta(bool complete,
                  int64_t n_decoded_tokens,
                  int64_t n_drafted,
                  int64_t n_accepted_drafts,
                  int64_t n_bonus_samples,
                  double prompt_ms,
                  double decode_ms) {
    const std::filesystem::path final_path = this->run_dir / "meta.json";
    const std::filesystem::path tmp_path = this->run_dir / "meta.json.tmp";

    std::ofstream m(tmp_path);
    if (!m) {
      throw std::runtime_error(std::format("failed to open {}", tmp_path.string()));
    }

    const Parameters &p = this->params_snapshot_;
    const double total_ms = prompt_ms + decode_ms;
    const double tok_per_s = decode_ms > 0.0 ? (1000.0 * (double)n_decoded_tokens / decode_ms) : 0.0;
    const double accept_rate = n_drafted > 0
                                   ? (double)n_accepted_drafts / (double)n_drafted
                                   : 0.0;

    m << "{\n";
    m << "  \"run_id\": \"" << json_escape(this->run_id_) << "\",\n";
    m << "  \"started_at\": \"" << json_escape(this->started_at_) << "\",\n";
    m << "  \"complete\": " << (complete ? "true" : "false") << ",\n";

    m << "  \"config\": {\n";
    m << "    \"target_model_path\": \"" << json_escape(p.target_model_path) << "\",\n";
    m << "    \"draft_model_path\": " << (p.draft_model_path.empty() ? "null" : ("\"" + json_escape(p.draft_model_path) + "\""))
      << ",\n";
    m << "    \"speculative\": " << (p.draft_model_path.empty() ? "false" : "true") << ",\n";
    m << "    \"ctx\": " << p.ctx << ",\n";
    m << "    \"ngl\": " << p.ngl << ",\n";
    m << "    \"n_min\": " << p.n_min << ",\n";
    m << "    \"n_max\": " << p.n_max << ",\n";
    m << "    \"ngram\": " << (p.ngram ? "true" : "false") << ",\n";
    m << "    \"n_gram_size\": " << p.n_gram_size << ",\n";
    m << "    \"m_gram_size\": " << p.m_gram_size << ",\n";
    m << "    \"temp\": " << p.temp << ",\n";
    m << "    \"top_p\": " << p.top_p << ",\n";
    m << "    \"top_k\": " << p.top_k << ",\n";
    m << "    \"greedy\": " << (p.greedy ? "true" : "false") << ",\n";
    m << "    \"seed\": " << p.seed << ",\n";
    m << "    \"n_predict\": " << p.n_predict << ",\n";
    m << "    \"prompt_n_chars\": " << p.prompt.size() << ",\n";
    m << "    \"prompt\": \"" << json_escape(p.prompt) << "\"\n";
    m << "  },\n";

    m << "  \"totals\": {\n";
    m << "    \"n_decoded_tokens\": " << n_decoded_tokens << ",\n";
    m << "    \"n_drafted\": " << n_drafted << ",\n";
    m << "    \"n_accepted_drafts\": " << n_accepted_drafts << ",\n";
    m << "    \"n_bonus_samples\": " << n_bonus_samples << ",\n";
    m << "    \"accept_rate\": " << accept_rate << ",\n";
    m << "    \"prompt_ms\": " << prompt_ms << ",\n";
    m << "    \"decode_ms\": " << decode_ms << ",\n";
    m << "    \"total_ms\": " << total_ms << ",\n";
    m << "    \"tok_per_s\": " << tok_per_s << "\n";
    m << "  },\n";

    m << "  \"rounds\": [";
    for (std::size_t i = 0; i < this->rounds.size(); ++i) {
      const auto &r = this->rounds[i];
      if (i > 0) m << ",";
      m << "\n    {\"n_drafted\": " << r.n_drafted
        << ", \"n_accepted_drafts\": " << r.n_accepted_drafts
        << ", \"rejected_pos\": " << r.rejected_pos << "}";
    }
    if (!this->rounds.empty()) m << "\n  ";
    m << "]\n";
    m << "}\n";
    m.flush();
    m.close();

    std::error_code ec;
    std::filesystem::rename(tmp_path, final_path, ec);
    if (ec) {
      throw std::runtime_error(std::format("failed to rename {} -> {}: {}",
                                           tmp_path.string(), final_path.string(),
                                           ec.message()));
    }
  }

  static std::string fmt_double(double v) {
    if (std::isnan(v)) return std::string{};
    return std::format("{:.8g}", v);
  }

  static std::string json_escape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
      switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          out += std::format("\\u{:04x}", (unsigned)c);
        } else {
          out += c;
        }
      }
    }
    return out;
  }

  std::filesystem::path run_dir;
  std::string run_id_;
  std::string started_at_;
  Parameters params_snapshot_;
  std::ofstream tokens;
  std::vector<RoundSummary> rounds;
  int step = 0;
};

class Application {
public:
  Application(int argc, char **argv) {
    this->parse_cli_args(argc, argv);
  };

  Application(const Application &) = delete;
  Application &operator=(const Application &) = delete;

  Application(Application &&) = delete;
  Application &operator=(Application &&) = delete;

  ~Application() {
    llama_sampler_free(sampler_target);
    if (params.draft_speculative_decoding_is_enabled()) {
      llama_sampler_free(sampler_draft);
      llama_batch_free(speculation_batch_target);
      llama_batch_free(speculation_batch_draft);
      llama_free(ctx_draft);
      llama_model_free(model_draft);
    }
    llama_free(ctx_target);
    llama_model_free(model_target);
    llama_backend_free();

    if (this->cout_saved) {
      std::cout.rdbuf(this->cout_saved);
      this->cout_saved = nullptr;
    }
    this->tee_buf.reset();
  }

  void start() {
    this->initialize();
    this->tokenize();
    this->decode();
    this->run();
  }

private:
  Parameters params;

  std::ofstream log_file;
  std::unique_ptr<TeeBuf> tee_buf;
  std::streambuf *cout_saved = nullptr;

  llama_token last_token;

  llama_model *model_target = nullptr;
  llama_model *model_draft = nullptr;

  llama_context *ctx_target = nullptr;
  llama_context *ctx_draft = nullptr;

  std::vector<llama_token> prompt_target;
  std::vector<llama_token> prompt_draft;

  llama_batch batch{};
  llama_batch speculation_batch_target{};
  llama_batch speculation_batch_draft{};

  llama_sampler *sampler_target = nullptr;
  llama_sampler *sampler_draft = nullptr;

  const struct llama_vocab *vocab_target = nullptr;
  const struct llama_vocab *vocab_draft = nullptr;

  // p_draft for each token returned by the last call to draft(), parallel to the returned vector.
  std::vector<double> last_draft_probs;

  void print_usage(char **argv) {
    const char *name = argv[0];

    // truncate the default prompt for display
    constexpr std::size_t PROMPT_PREVIEW_MAX = 60;
    std::string prompt_preview = this->params.prompt;
    if (prompt_preview.size() > PROMPT_PREVIEW_MAX) {
      prompt_preview = prompt_preview.substr(0, PROMPT_PREVIEW_MAX) + "...";
    }

    print(GGML_LOG_LEVEL_NONE, "Usage: {} --target-model <file.gguf> [--draft-model <file.gguf>] [OPTIONS]", name);
    print(GGML_LOG_LEVEL_NONE, "");
    print(GGML_LOG_LEVEL_NONE, "Speculative decoding runs whenever --draft-model is given; otherwise the");
    print(GGML_LOG_LEVEL_NONE, "binary runs vanilla autoregressive decoding against the target model only.");
    print(GGML_LOG_LEVEL_NONE, "");
    print(GGML_LOG_LEVEL_NONE, "Models:");
    print(GGML_LOG_LEVEL_NONE, "  --target-model <file>    gguf target model file (required)");
    print(GGML_LOG_LEVEL_NONE, "  --draft-model <file>     gguf draft model file (enables speculative decoding)");
    print(GGML_LOG_LEVEL_NONE, "");
    print(GGML_LOG_LEVEL_NONE, "Runtime:");
    print(GGML_LOG_LEVEL_NONE, "  --ctx-size <n>           context size in tokens (0 = from model) (default: {})", this->params.ctx);
    print(GGML_LOG_LEVEL_NONE, "  --n-gpu-layers <n>       layers in VRAM (<0 = all) (default: {})", this->params.ngl);
    print(GGML_LOG_LEVEL_NONE, "  --n-predict <n>          hard cap on generated tokens (0 = unlimited) (default: {})", this->params.n_predict);
    print(GGML_LOG_LEVEL_NONE, "");
    print(GGML_LOG_LEVEL_NONE, "Sampling:");
    print(GGML_LOG_LEVEL_NONE, "  --temp <n>               temperature (default: {})", this->params.temp);
    print(GGML_LOG_LEVEL_NONE, "  --top-p <n>              top-p sampling (default: {})", this->params.top_p);
    print(GGML_LOG_LEVEL_NONE, "  --top-k <n>              top-k sampling (default: {})", this->params.top_k);
    print(GGML_LOG_LEVEL_NONE, "  --greedy                 greedy sampler; overrides temp/top-p/top-k (default: {})", this->params.greedy ? "true" : "false");
    print(GGML_LOG_LEVEL_NONE, "  --prompt <text>          initial prompt (default: \"{}\")", prompt_preview);
    print(GGML_LOG_LEVEL_NONE, "");
    print(GGML_LOG_LEVEL_NONE, "Speculation (only effective when --draft-model is set):");
    print(GGML_LOG_LEVEL_NONE, "  --ngram                  enable n-gram drafter (hybrid: ngram first, draft model on miss) (default: {})", this->params.ngram);
    print(GGML_LOG_LEVEL_NONE, "  --n-gram-size <n>        lookup pattern length for the n-gram drafter (default: {})", this->params.n_gram_size);
    print(GGML_LOG_LEVEL_NONE, "  --m-gram-size <n>        max draft length proposed after an n-gram hit (default: {})", this->params.m_gram_size);
    print(GGML_LOG_LEVEL_NONE, "  --n-max <n>              max tokens to draft per speculative call (default: {})", this->params.n_max);
    print(GGML_LOG_LEVEL_NONE, "  --n-min <n>              min draft length; below this the draft is discarded (default: {})", this->params.n_min);
    print(GGML_LOG_LEVEL_NONE, "");
    print(GGML_LOG_LEVEL_NONE, "Output / reproducibility:");
    print(GGML_LOG_LEVEL_NONE, "  --seed <n>               sampler seed (default: {})", this->params.seed);
    print(GGML_LOG_LEVEL_NONE, "  --run-id <id>            unique run identifier (default: auto-generated as YYYYMMDD-HHMMSS_<mode>_seed<N>)");
    print(GGML_LOG_LEVEL_NONE, "  --results-dir <path>     where to write <run-id>/{{meta.json,tokens.csv}} (default: \"{}\")", this->params.results_dir);
    print(GGML_LOG_LEVEL_NONE, "");
    print(GGML_LOG_LEVEL_NONE, "Misc:");
    print(GGML_LOG_LEVEL_NONE, "  -h, --help               print this message and exit");
    print(GGML_LOG_LEVEL_NONE, "");
    print(GGML_LOG_LEVEL_NONE, "Examples:");
    print(GGML_LOG_LEVEL_NONE, "  # vanilla autoregressive baseline");
    print(GGML_LOG_LEVEL_NONE, "  {} --target-model Qwen2.5-Coder-3B-Instruct-IQ2_M.gguf \\", name);
    print(GGML_LOG_LEVEL_NONE, "      --prompt \"Tell me a joke\" --ctx-size 8192 --n-gpu-layers -1 \\");
    print(GGML_LOG_LEVEL_NONE, "      --n-predict 256 --seed 42 --run-id ar-baseline_seed42");
    print(GGML_LOG_LEVEL_NONE, "");
    print(GGML_LOG_LEVEL_NONE, "  # speculative decoding with a smaller draft model");
    print(GGML_LOG_LEVEL_NONE, "  {} --target-model Nemotron-3-Nano-4B-BF16.gguf \\", name);
    print(GGML_LOG_LEVEL_NONE, "      --draft-model Nemotron-3-Nano-4B-Q8_0.gguf \\");
    print(GGML_LOG_LEVEL_NONE, "      --n-max 8 --n-predict 256 --seed 42 --run-id spec-nmax8_seed42");
  }

  void parse_cli_args(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
      try {
        if (std::strcmp(argv[i], "-h") == 0 ||
            std::strcmp(argv[i], "--help") == 0) {
          print_usage(argv);
          std::exit(0);
        } else if (std::strcmp(argv[i], "--target-model") == 0) {
          if (i + 1 < argc) {
            this->params.target_model_path = argv[++i];
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for target model");
          }
        } else if (std::strcmp(argv[i], "--draft-model") == 0) {
          if (i + 1 < argc) {
            this->params.draft_model_path = argv[++i];
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for draft model");
          }
        } else if (std::strcmp(argv[i], "--ctx-size") == 0) {
          if (i + 1 < argc) {
            this->params.ctx = std::stoi(argv[++i]);
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for context size");
          }
        } else if (std::strcmp(argv[i], "--n-gpu-layers") == 0) {
          if (i + 1 < argc) {
            this->params.ngl = std::stoi(argv[++i]);
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for n-gpu-layers");
          }
        } else if (std::strcmp(argv[i], "--prompt") == 0) {
          if (i + 1 < argc) {
            this->params.prompt = argv[++i];
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for prompt");
          }
        } else if (std::strcmp(argv[i], "--temp") == 0) {
          if (i + 1 < argc) {
            this->params.temp = std::stof(argv[++i]);
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for temperature");
          }
        } else if (std::strcmp(argv[i], "--top-p") == 0) {
          if (i + 1 < argc) {
            this->params.top_p = std::stof(argv[++i]);
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for top-p");
          }
        } else if (std::strcmp(argv[i], "--greedy") == 0) {
          this->params.greedy = true;
        } else if (std::strcmp(argv[i], "--ngram") == 0) {
          this->params.ngram = true;
        } else if (std::strcmp(argv[i], "--n-gram-size") == 0) {
          if (i + 1 < argc) {
            this->params.n_gram_size = std::stoi(argv[++i]);
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for --n-gram-size");
          }
        } else if (std::strcmp(argv[i], "--m-gram-size") == 0) {
          if (i + 1 < argc) {
            this->params.m_gram_size = std::stoi(argv[++i]);
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for --m-gram-size");
          }
        } else if (std::strcmp(argv[i], "--top-k") == 0) {
          if (i + 1 < argc) {
            this->params.top_k = std::stoi(argv[++i]);
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for top-k");
          }
        } else if (std::strcmp(argv[i], "--seed") == 0) {
          if (i + 1 < argc) {
            this->params.seed = (uint32_t)std::stoul(argv[++i]);
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for --seed");
          }
        } else if (std::strcmp(argv[i], "--results-dir") == 0) {
          if (i + 1 < argc) {
            this->params.results_dir = argv[++i];
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for --results-dir");
          }
        } else if (std::strcmp(argv[i], "--run-id") == 0) {
          if (i + 1 < argc) {
            this->params.run_id = argv[++i];
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for --run-id");
          }
        } else if (std::strcmp(argv[i], "--n-predict") == 0) {
          if (i + 1 < argc) {
            this->params.n_predict = std::stoll(argv[++i]);
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for --n-predict");
          }
        } else if (std::strcmp(argv[i], "--n-max") == 0) {
          if (i + 1 < argc) {
            this->params.n_max = std::stoll(argv[++i]);
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for --n-max");
          }
        } else if (std::strcmp(argv[i], "--n-min") == 0) {
          if (i + 1 < argc) {
            this->params.n_min = std::stoll(argv[++i]);
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for --n-min");
          }
        } else {
          print_usage(argv);
          throw std::runtime_error(std::format("Unknown argument: {}", argv[i]));
        }
      } catch (const std::invalid_argument &e) {
        throw std::runtime_error(
            std::format("Invalid numeric value '{}': {}", argv[i], e.what()));
      } catch (const std::out_of_range &e) {
        throw std::runtime_error(
            std::format("Numeric value out of range '{}': {}", argv[i], e.what()));
      }
    }

    if (this->params.target_model_path.empty()) {
      print_usage(argv);
      throw std::runtime_error("Error: --target-model argument is required");
    }
  }

  std::tuple<double, double> softmax(
      const float *logits_row,
      const llama_token token,
      const struct llama_vocab *vocab = nullptr) {
    if (vocab == nullptr) vocab = this->vocab_target;
    const int n_vocab = llama_vocab_n_tokens(vocab);

    double max_logit = logits_row[0];
    for (int j = 1; j < n_vocab; ++j) {
      if (logits_row[j] > max_logit) {
        max_logit = logits_row[j];
      }
    }

    double denom = 0.0;
    for (int j = 0; j < n_vocab; ++j) {
      denom += std::exp((double)logits_row[j] - max_logit);
    }

    const llama_token tid = token;
    const double logit = logits_row[tid];
    const double prob = std::exp((double)logit - max_logit) / denom;

    return std::make_tuple(logit, prob);
  }

  std::tuple<double, double> stable_sigmoid(
      const float *logits_row,
      const llama_token token,
      const struct llama_vocab *vocab = nullptr) {
    if (vocab == nullptr) vocab = this->vocab_target;

    const llama_token tid = token;
    const double logit = logits_row[tid];

    float prob;
    if (logit >= 0.0f) {
      prob = 1.0f / (1.0f + std::exp(-logit));
    } else {
      const float exp_logit = std::exp(logit);
      prob = exp_logit / (1.0f + exp_logit);
    }

    return std::make_tuple(logit, prob);
  }

  void initialize(void) {
#if 0
    this->log_file.open("log.txt", std::ios::out | std::ios::trunc);
    if (!this->log_file) {
      throw std::runtime_error("failed to open log.txt");
    }
    this->cout_saved = std::cout.rdbuf();
    this->tee_buf = std::make_unique<TeeBuf>(this->cout_saved, this->log_file.rdbuf());
    std::cout.rdbuf(this->tee_buf.get());
#endif

    try {
      llama_log_set(
          [](ggml_log_level level, const char *text, void * /*user_data*/) {
            std::cout << log_level_to_string(level) << text << std::flush;
          },
          nullptr);

      llama_backend_init();

      print(GGML_LOG_LEVEL_INFO, "llama_print_system_info:       {}", llama_print_system_info());
      print(GGML_LOG_LEVEL_INFO, "llama_supports_mmap:           {}", llama_supports_mmap());
      print(GGML_LOG_LEVEL_INFO, "llama_supports_mlock:          {}", llama_supports_mlock());
      print(GGML_LOG_LEVEL_INFO, "llama_supports_gpu_offload:    {}", llama_supports_gpu_offload());

      struct llama_model_params params = llama_model_default_params();
      // ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);

      // params.devices = &device;
      params.n_gpu_layers = this->params.ngl;
      // params.use_mmap = llama_supports_mmap();
      // params.use_mlock = llama_supports_mlock();

      this->model_target = llama_model_load_from_file(this->params.target_model_path.c_str(), params);
      if (!this->model_target) {
        throw std::runtime_error("failed to load target model");
      }

      if (this->params.draft_speculative_decoding_is_enabled()) {
        this->model_draft = llama_model_load_from_file(this->params.draft_model_path.c_str(), params);
        if (!this->model_draft) {
          throw std::runtime_error("failed to load draft model");
        }
      }

      print(GGML_LOG_LEVEL_INFO, "target_llama_model_n_params:    {}", llama_model_n_params(this->model_target));
      if (this->params.draft_speculative_decoding_is_enabled()) {
        print(GGML_LOG_LEVEL_INFO, "draft_llama_model_n_params:    {}", llama_model_n_params(this->model_draft));
      }

      struct llama_context_params ctx_params = llama_context_default_params();

      ctx_params.no_perf = false;
      ctx_params.n_ctx = this->params.ctx;

      this->ctx_target = llama_init_from_model(this->model_target, ctx_params);
      if (!this->ctx_target) {
        throw std::runtime_error("failed to create the llama_context for target");
      }

      if (this->params.draft_speculative_decoding_is_enabled()) {
        this->ctx_draft = llama_init_from_model(this->model_draft, ctx_params);
        if (!this->ctx_draft) {
          throw std::runtime_error("failed to create the llama_context for draft");
        }
      }

      print(GGML_LOG_LEVEL_INFO, "target_llama_n_ctx:        {}", llama_n_ctx(this->ctx_target));
      print(GGML_LOG_LEVEL_INFO, "target_llama_n_ctx_seq:    {}", llama_n_ctx_seq(this->ctx_target));
      print(GGML_LOG_LEVEL_INFO, "target_llama_n_batch:      {}", llama_n_batch(this->ctx_target));
      print(GGML_LOG_LEVEL_INFO, "target_llama_n_ubatch:     {}", llama_n_ubatch(this->ctx_target));
      print(GGML_LOG_LEVEL_INFO, "target_llama_n_seq_max:    {}", llama_n_seq_max(this->ctx_target));

      if (this->params.draft_speculative_decoding_is_enabled()) {
        print(GGML_LOG_LEVEL_INFO, "draft_llama_n_ctx:        {}", llama_n_ctx(this->ctx_draft));
        print(GGML_LOG_LEVEL_INFO, "draft_llama_n_ctx_seq:    {}", llama_n_ctx_seq(this->ctx_draft));
        print(GGML_LOG_LEVEL_INFO, "draft_llama_n_batch:      {}", llama_n_batch(this->ctx_draft));
        print(GGML_LOG_LEVEL_INFO, "draft_llama_n_ubatch:     {}", llama_n_ubatch(this->ctx_draft));
        print(GGML_LOG_LEVEL_INFO, "draft_llama_n_seq_max:    {}", llama_n_seq_max(this->ctx_draft));
      }
    } catch (...) {
      if (this->cout_saved) {
        std::cout.rdbuf(this->cout_saved);
        this->cout_saved = nullptr;
      }
      this->tee_buf.reset();
      throw;
    }
  }

  void tokenize(void) {
    this->vocab_target = llama_model_get_vocab(this->model_target);

    if (this->params.draft_speculative_decoding_is_enabled()) {
      this->vocab_draft = llama_model_get_vocab(this->model_draft);
    }

    const int prompt_target_len = -llama_tokenize(
        this->vocab_target,
        this->params.prompt.c_str(),
        this->params.prompt.size(),
        NULL, 0, true, true);

    prompt_target.resize(prompt_target_len);

    int n = llama_tokenize(
        this->vocab_target,
        this->params.prompt.c_str(), this->params.prompt.size(),
        prompt_target.data(), prompt_target.size(),
        true, true);

    if (n < 0) {
      throw std::runtime_error(std::format("failed to tokenize prompt (n = {})", n));
    }

    print(GGML_LOG_LEVEL_INFO, "\"{}\" ({} tokens)", this->params.prompt.c_str(), prompt_target_len);

    if (llama_n_ctx(this->ctx_target) < (uint32_t)prompt_target.size()) {
      throw std::runtime_error(std::format("the prompt exceeds the context size ({} tokens, ctx {})", prompt_target.size(), llama_n_ctx(this->ctx_target)));
    }

    if (llama_n_batch(this->ctx_target) < (uint32_t)prompt_target.size()) {
      throw std::runtime_error(std::format("the prompt exceeds the batch size ({} tokens, batch {})", prompt_target.size(), llama_n_batch(this->ctx_target)));
    }

    if (this->params.draft_speculative_decoding_is_enabled()) {
      if (llama_vocab_get_add_bos(vocab_target) != llama_vocab_get_add_bos(vocab_draft) ||
          (llama_vocab_get_add_bos(vocab_target) && llama_vocab_bos(vocab_target) != llama_vocab_bos(vocab_draft))) {
        throw std::runtime_error(std::format(
            "%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
            __func__,
            llama_vocab_get_add_bos(vocab_target), llama_vocab_get_add_bos(vocab_draft),
            llama_vocab_bos(vocab_target), llama_vocab_bos(vocab_draft)));
      }

      if (llama_vocab_get_add_eos(vocab_target) != llama_vocab_get_add_eos(vocab_draft) ||
          (llama_vocab_get_add_eos(vocab_target) && llama_vocab_eos(vocab_target) != llama_vocab_eos(vocab_draft))) {
        throw std::runtime_error(std::format(
            "%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
            __func__,
            llama_vocab_get_add_eos(vocab_target), llama_vocab_get_add_eos(vocab_draft),
            llama_vocab_eos(vocab_target), llama_vocab_eos(vocab_draft)));
      }

      const int n_vocab_target = llama_vocab_n_tokens(vocab_target);
      const int n_vocab_draft = llama_vocab_n_tokens(vocab_draft);
      const int vocab_diff = n_vocab_target > n_vocab_draft
                                 ? n_vocab_target - n_vocab_draft
                                 : n_vocab_draft - n_vocab_target;

      if (vocab_diff > 128) {
        throw std::runtime_error(std::format(
            "%s: draft model vocab must closely match target model to use speculation but "
            "target vocab size %d does not match draft vocab size %d - difference %d, max allowed 128\n",
            __func__, n_vocab_target, llama_vocab_n_tokens(vocab_draft), vocab_diff));
      }

      for (int i = 128; i < std::min(n_vocab_target, n_vocab_draft); ++i) {
        const char *token_text_target = llama_vocab_get_text(vocab_target, i);
        const char *token_text_draft = llama_vocab_get_text(vocab_draft, i);

        if (std::strcmp(token_text_target, token_text_draft) != 0) {
          throw std::runtime_error(std::format(
              "%s: draft model vocab must match target model to use speculation but "
              "token %d content differs\n",
              __func__, i));
        }
      }
    }

    for (auto id : prompt_target) {
      char buf[128] = {0};
      int n = llama_token_to_piece(this->vocab_target, id, buf, sizeof(buf), 0, true);
      if (n < 0) {
        throw std::runtime_error("failed to convert token to piece");
      }
      std::size_t pos = 0;
      std::string token(buf, n);
      while ((pos = token.find('\n', pos)) != std::string::npos) {
        token.replace(pos, 1, "\\n");
        pos += 2;
      }
      print(GGML_LOG_LEVEL_INFO, "|{}|", token.c_str());
    }

    print(GGML_LOG_LEVEL_INFO, "llama_vocab_n_tokens:    {}", llama_vocab_n_tokens(this->vocab_target));
    print(GGML_LOG_LEVEL_INFO, "llama_vocab_type:        {}", (int)llama_vocab_type(this->vocab_target));
  }

  void decode(void) {
    struct llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();

    sampler_params.no_perf = false;

    this->sampler_target = llama_sampler_chain_init(sampler_params);
    if (!this->sampler_target) {
      throw std::runtime_error("failed to create the llama_sampler_chain_params");
    }

    if (this->params.greedy) {
      // greedy sampler. select the token with the highest probability (logit)
      // at each step of text generation, leading to deterministic and generally more focused outputs
      llama_sampler_chain_add(this->sampler_target, llama_sampler_init_greedy());
    } else {
      llama_sampler_chain_add(this->sampler_target, llama_sampler_init_top_k(this->params.top_k));
      llama_sampler_chain_add(this->sampler_target, llama_sampler_init_top_p(this->params.top_p, 1));
      llama_sampler_chain_add(this->sampler_target, llama_sampler_init_temp(this->params.temp));
      llama_sampler_chain_add(this->sampler_target, llama_sampler_init_dist(this->params.seed));
    }

    if (this->params.draft_speculative_decoding_is_enabled()) {
      // context holds the size of batch
      this->speculation_batch_target = llama_batch_init(llama_n_batch(this->ctx_target), 0, 1);
      this->speculation_batch_draft = llama_batch_init(llama_n_batch(this->ctx_draft), 0, 1);

      this->sampler_draft = llama_sampler_chain_init(sampler_params);
      if (!this->sampler_draft) {
        throw std::runtime_error("failed to create draft sampler chain");
      }

      if (this->params.greedy) {
        // greedy sampler. select the token with the highest probability (logit)
        // at each step of text generation, leading to deterministic and generally more focused outputs
        llama_sampler_chain_add(this->sampler_draft, llama_sampler_init_greedy());
      } else {
        llama_sampler_chain_add(this->sampler_draft, llama_sampler_init_top_k(this->params.top_k));
        llama_sampler_chain_add(this->sampler_draft, llama_sampler_init_top_p(this->params.top_p, 1));
        llama_sampler_chain_add(this->sampler_draft, llama_sampler_init_temp(this->params.temp));
        llama_sampler_chain_add(this->sampler_draft, llama_sampler_init_dist(this->params.seed));
      }
    }

    // TODO remove; this is not required since most models are decoder only
    // if (llama_model_has_encoder(this->model_target)) {
    //   if (this->params.speculative_decoding_enabled()) {
    //     print(GGML_LOG_LEVEL_WARN,
    //           "Speculative decoding is not implemented for encoder–decoder models; "
    //           "draft model is loaded but generation uses the target only.");
    //   }
    //   // eval the prompt
    //   if (llama_encode(this->ctx_target, current_batch)) {
    //     throw std::runtime_error("failed to eval");
    //   }
    //
    //   llama_token decoder_start_token_id = llama_model_decoder_start_token(this->model_target);
    //   if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
    //     decoder_start_token_id = llama_vocab_bos(this->vocab_target);
    //   }
    //
    //   current_batch = llama_batch_get_one(&decoder_start_token_id, 1);
    // }
  }

  void run(void) {
    // current accepted token each pass
    llama_token current_token = -1;
    std::size_t tokens_decoded = 0;

    int64_t n_bonus_samples = 0;

    print(GGML_LOG_LEVEL_INFO, "llama_model_chat_template:\n{}", llama_model_chat_template(this->model_target, NULL));

    // auto-generate a run-id if none was supplied.
    if (this->params.run_id.empty()) {
      const std::string ts = RunRecorder::iso_timestamp(/*compact=*/true);
      const std::string mode = this->params.draft_speculative_decoding_is_enabled() ? "spec" : "ar";
      this->params.run_id = std::format("{}_{}_seed{}", ts, mode, this->params.seed);
    }

    const std::string started_at = RunRecorder::iso_timestamp();
    RunRecorder recorder(this->params.results_dir, this->params.run_id, started_at, this->params);

    print(GGML_LOG_LEVEL_INFO, "writing structured run output to: {}", recorder.dir().string());

    const int64_t t_start_total = ggml_time_us();
    int64_t t_after_prompt = 0;

    this->params.n_accept = 0;
    this->params.n_drafted = 0;
    this->params.has_eos = false; // end-of-sentence

    if (this->prompt_target.empty()) {
      throw std::runtime_error("prompt_target is empty");
    }

    // get prompt batch
    this->batch = llama_batch_get_one(this->prompt_target.data(), (int32_t)this->prompt_target.size() - 1);

    // evaluate prompt (update KV cache and compute logits for the prompt)
    if (llama_decode(this->ctx_target, this->batch)) {
      throw std::runtime_error("failed to eval prompt prefix on target");
    }
    t_after_prompt = ggml_time_us();

    // sample starting from the last token of the prompt
    this->last_token = this->prompt_target.back();
    this->prompt_target.pop_back();

    llama_memory_t mem_target = llama_get_memory(this->ctx_target);

    if (this->params.draft_speculative_decoding_is_enabled()) {
      print(GGML_LOG_LEVEL_INFO, "Speculative Decoding using a draft model is enabled");

      {
        char buf[1024] = {0};
        llama_model_desc(this->model_target, buf, sizeof(buf));
        print(GGML_LOG_LEVEL_INFO, "target_model :    {}", buf);

        llama_model_desc(this->model_draft, buf, sizeof(buf));
        print(GGML_LOG_LEVEL_INFO, "draft_model  :    {}", buf);
      }

      // TODO remove; this is not required since most models are decoder only
      // if (llama_model_has_encoder(this->model_target)) {
      //   throw std::runtime_error("speculative decoding is only implemented for decoder-only models");
      // }

      int call_idx = 0;
      while (!this->params.has_eos) {
        // M-RoPE (Qwen3): batch positions must satisfy Y > X (where X = memory.seq_pos_max).
        // Tracking "pos" as prompt_target.size() drifts from real KV (e.g. X=21 vs Y=7).
        // Fix: ask the cache where does the sequence end then +1 for the next real decode step.
        // Note:
        //     X = What's the last position index still in the KV cache?
        //     Y = What position does this new batch claim its _first_ token uses?
        const llama_pos pmax = llama_memory_seq_pos_max(mem_target, 0);
        llama_pos n_past = (pmax < 0) ? 0 : (pmax + 1);

        // get drafted tokens: n-gram first if --ngram is set, otherwise (or on miss) the draft model
        // n-gram takes precedence over draft based Speculative Decoding
        std::vector<llama_token> draft;
        if (this->params.ngram) {
          draft = this->draft_ngram();
        }
        if (draft.empty()) {
          draft = this->draft();
        }

        if (draft.size() > (std::size_t)this->params.n_max) {
          draft.resize((std::size_t)this->params.n_max);
        } else if (draft.size() < (std::size_t)this->params.n_min) {
          draft.clear();
        }

        // reset target batch
        this->reset_batch(this->speculation_batch_target);

        // add prompt batch to target
        this->create_new_batch(
            this->speculation_batch_target,
            (int32_t)llama_n_batch(this->ctx_target),
            this->last_token, n_past, true);
        n_past += 1;

        // add drafted tokens to the target
        for (std::size_t i = 0; i < draft.size(); ++i) {
          this->create_new_batch(
              this->speculation_batch_target, (int32_t)llama_n_batch(this->ctx_target),
              draft[i], n_past + (llama_pos)i, true);
        }

        // evaluate the batch (update KV cache and compute logits for the batch)
        if (llama_decode(this->ctx_target, this->speculation_batch_target)) {
          throw std::runtime_error("target speculative verification decode failed");
        }

        // do the actual verification of the sampled tokens
        const std::vector<llama_token> accepted = sample_and_accept(this->sampler_target, this->ctx_target, draft);
        if (accepted.empty()) {
          throw std::runtime_error("speculative accept produced no tokens");
        }

        n_past += (llama_pos)((int)accepted.size() - 1);
        this->params.n_drafted += (int64_t)draft.size();
        this->params.n_accept += (int64_t)accepted.size() - 1;

        // per-round summary: how many drafts matched before first rejection (or -1 if all accepted).
        // accepted.size() == draft.size() + 1  if every draft matched (the last entry is the bonus sample).
        // otherwise accepted.back() is the resampled token at the mismatch position.
        const int n_accepted_drafts_in_round = (int)accepted.size() - 1;
        const int rejected_pos = (accepted.size() == draft.size() + 1) ? -1 : n_accepted_drafts_in_round;
        recorder.record_round((int)draft.size(), n_accepted_drafts_in_round, rejected_pos);
        if (rejected_pos == -1) {
          n_bonus_samples += 1;
        }

        for (std::size_t i = 0; i < accepted.size(); ++i) {
          auto [logit, prob] = softmax(llama_get_logits_ith(this->ctx_target, (int32_t)i), accepted[i]);
          const double logprob = prob > 0.0 ? std::log(prob) : -std::numeric_limits<double>::infinity();

          // Source labelling:
          //   - draft   : positions [0, n_accepted_drafts_in_round) matched a draft token
          //   - bonus   : the final entry when ALL drafts matched (target's free bonus sample)
          //   - draft   : when there was a rejection, the last entry replaces the rejected draft
          //               token at the mismatch position. We still attribute it to "draft" because
          //               that position was drafted (just not accepted as-is). p_draft for that
          //               position is the draft's probability on the token IT proposed there.
          const bool is_bonus = (rejected_pos == -1) && (i + 1 == accepted.size());
          const char *source = is_bonus ? "bonus" : "draft";
          const int pos_in_draft = is_bonus ? -1 : (int)i;
          const double p_draft = (!is_bonus && pos_in_draft >= 0 &&
                                  pos_in_draft < (int)this->last_draft_probs.size())
                                     ? this->last_draft_probs[(std::size_t)pos_in_draft]
                                     : std::numeric_limits<double>::quiet_NaN();

          recorder.record_token(
              call_idx, source, pos_in_draft, (int)accepted[i],
              prob, p_draft, logit, logprob);

          this->prompt_target.push_back(this->last_token);
          this->last_token = accepted[i];

          // is last_token end-of-generation
          if (llama_vocab_is_eog(this->vocab_target, this->last_token)) {
            this->params.has_eos = true;
            break;
          }

          // hard cap on tokens (treated like EOS for the purposes of clean finalize)
          if (this->params.n_predict > 0 && (int64_t)(tokens_decoded + 1) >= this->params.n_predict) {
            this->params.has_eos = true;
          }

          char buf[128] = {0};
          int n = llama_token_to_piece(this->vocab_target, this->last_token, buf, sizeof(buf), 0, true);
          if (n < 0) {
            throw std::runtime_error("failed to convert token to piece");
          }
          std::string token(buf, n);
          std::size_t pos = 0;
          while ((pos = token.find('\n', pos)) != std::string::npos) {
            token.replace(pos, 1, "\\n");
            pos += 2;
          }

          // rotating ANSI color for the token: cyan -> magenta -> blue -> yellow -> green -> red
          static constexpr int kColors[] = {36, 35, 34, 33, 32, 31};
          static constexpr std::size_t kCol = 24; // pad token+brackets to this column

          const int color = kColors[i % std::size(kColors)];
          const std::size_t visible = token.size() + 2; // 2 chars for the surrounding '|'
          const std::string pad(visible < kCol ? kCol - visible : 1, ' ');

          print(GGML_LOG_LEVEL_INFO,
                "\x1b[{}m|{}|\x1b[0m{}(accepted {} out of {} draft tokens, last_token = {})",
                color, token, pad,
                (int)accepted.size() - 1, (int)draft.size(), this->last_token);

          tokens_decoded += 1;
        }

        llama_memory_seq_rm(mem_target, 0, n_past, -1);
        call_idx += 1;
      }

      std::cout << std::endl;

      const int64_t t_end = ggml_time_us();
      const double prompt_ms = (t_after_prompt - t_start_total) / 1000.0;
      const double decode_ms = (t_end - t_after_prompt) / 1000.0;
      const float speed = tokens_decoded / std::max((float)(decode_ms / 1000.0), 1e-6f);

      print(GGML_LOG_LEVEL_INFO,
            "decoded {} tokens in {:.3f} s, speed: {:.2f} t/s (prompt {:.1f} ms, decode {:.1f} ms)",
            tokens_decoded, decode_ms / 1000.0, speed, prompt_ms, decode_ms);

      if (this->params.n_drafted > 0) {
        print(GGML_LOG_LEVEL_INFO,
              "speculative: n_drafted = {}, n_accept = {}, accept = {:.2f}%",
              this->params.n_drafted, this->params.n_accept,
              100.0 * (double)this->params.n_accept / (double)this->params.n_drafted);
      }

      recorder.finalize((int64_t)tokens_decoded,
                        this->params.n_drafted,
                        this->params.n_accept,
                        n_bonus_samples,
                        prompt_ms, decode_ms);
      print(GGML_LOG_LEVEL_INFO, "wrote {} and {}",
            (recorder.dir() / "meta.json").string(),
            (recorder.dir() / "tokens.csv").string());

      llama_perf_sampler_print(this->sampler_target);
      llama_perf_context_print(this->ctx_target);
      llama_perf_context_print(this->ctx_draft);

      return;
    }

    //
    // autoregressive
    //

    bool first_iteration = true;
    for (;;) {
      // evaluate the batch (update KV cache and compute logits for the batch)
      if (llama_decode(this->ctx_target, batch)) {
        throw std::runtime_error("failed to eval");
      }
      if (first_iteration) {
        t_after_prompt = ggml_time_us();
        first_iteration = false;
      }

      // sample and accept the last token of the last evaluation (the next token)
      current_token = llama_sampler_sample(this->sampler_target, this->ctx_target, -1);

      auto [logit, prob] = softmax(llama_get_logits_ith(this->ctx_target, -1), current_token);
      const double logprob = prob > 0.0 ? std::log(prob) : -std::numeric_limits<double>::infinity();
      recorder.record_token(
          (int)tokens_decoded, "ar", -1, (int)current_token,
          prob, std::numeric_limits<double>::quiet_NaN(),
          logit, logprob);

      // is it an end of generation?
      if (llama_vocab_is_eog(this->vocab_target, current_token)) {
        break;
      }

      char buf[128] = {0};
      int n = llama_token_to_piece(this->vocab_target, current_token, buf, sizeof(buf), 0, true);
      if (n < 0) {
        throw std::runtime_error("failed to convert token to piece");
      }
      std::cout.write(buf, n);
      std::cout.flush();

      // prepare the next batch with the sampled token
      batch = llama_batch_get_one(&current_token, 1);

      tokens_decoded += 1;

      // hard cap on tokens
      if (this->params.n_predict > 0 && (int64_t)tokens_decoded >= this->params.n_predict) {
        break;
      }
    }

    std::cout << std::endl;

    const int64_t t_end = ggml_time_us();
    const double prompt_ms = t_after_prompt > 0 ? (t_after_prompt - t_start_total) / 1000.0 : 0.0;
    const double decode_ms = t_after_prompt > 0 ? (t_end - t_after_prompt) / 1000.0
                                                : (t_end - t_start_total) / 1000.0;
    const float speed = tokens_decoded / std::max((float)(decode_ms / 1000.0), 1e-6f);

    print(GGML_LOG_LEVEL_INFO,
          "decoded {} tokens in {:.3f} s, speed: {:.2f} t/s (prompt {:.1f} ms, decode {:.1f} ms)",
          tokens_decoded, decode_ms / 1000.0, speed, prompt_ms, decode_ms);

    recorder.finalize((int64_t)tokens_decoded,
                      /*n_drafted=*/0,
                      /*n_accepted_drafts=*/0,
                      /*n_bonus_samples=*/0,
                      prompt_ms, decode_ms);

    print(GGML_LOG_LEVEL_INFO, "wrote {} and {}",
          (recorder.dir() / "meta.json").string(),
          (recorder.dir() / "tokens.csv").string());

    // __asm volatile("int3");
    llama_perf_sampler_print(this->sampler_target);
    llama_perf_context_print(this->ctx_target);
  }

  void reset_batch(llama_batch &batch) {
    batch.n_tokens = 0;
  }

  void create_new_batch(
      llama_batch &batch,
      int32_t max_tokens,
      llama_token id,
      llama_pos pos,
      bool output) {
    //
    // llama_decode does not take a string but a llama_batch which is a small array of slots,
    // each describing one token we want to process in this forward pass.
    // create_new_batch basically is fill the next slot with (token, pos, logits, seq) then n_tokens++
    //
    assert(batch.n_tokens < max_tokens && "llama_batch capacity exceeded");
    assert(batch.seq_id[batch.n_tokens] && "llama_batch seq_id slot missing");
    batch.token[batch.n_tokens] = id;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id[batch.n_tokens][0] = 0;
    batch.logits[batch.n_tokens] = output;
    batch.n_tokens++;
  }

  std::vector<llama_token> sample_and_accept(
      llama_sampler *sampler,
      llama_context *ctx,
      const std::vector<llama_token> &draft) {
    llama_synchronize(ctx); // wait until all computations are finished

    std::vector<llama_token> accepted;
    accepted.reserve(draft.size() + 1);

    for (std::size_t index = 0; index < draft.size(); ++index) {
      // given the current logits, pick a token
      const llama_token accepted_token = llama_sampler_sample(sampler, ctx, (int32_t)index);
      // that token is now part of the sequence, update internal state
      llama_sampler_accept(sampler, accepted_token);
      accepted.push_back(accepted_token);
      // stop at first mismatch
      // last element is the correction
      if (draft[index] != accepted_token) {
        return accepted;
      }
    }

    // all draft tokens matched the target's samples at each step
    const llama_token id = llama_sampler_sample(sampler, ctx, (int32_t)draft.size());
    llama_sampler_accept(sampler, id);
    accepted.push_back(id);

    return accepted;
  }

  // READ: https://web.stanford.edu/~jurafsky/slp3/3.pdf
  //
  //                  <---
  // |                                                |
  // |                    tokens                      |
  // |                                                |
  // |                                    | | pattern |
  // |                                    | |         |
  // |                                    | |         |
  // 0                                    | |         49                 if (x != m) then { match = false }
  // v                                    | |         v                  x == m for the (match_pos)th token
  // -------------------------------x-------------m----
  //                                ^      ^
  //                               31      38
  //                          match_pos  n-gram
  //
  std::vector<llama_token> draft_ngram() {
    const auto &tokens = this->prompt_target;
    const llama_token sampled = this->last_token;
    const std::size_t N = (std::size_t)this->params.n_gram_size;
    const std::size_t M = (std::size_t)this->params.m_gram_size;
    const std::size_t cur_len = tokens.size();
    std::vector<llama_token> result;
    if (cur_len <= N + M + 1) return result;
    std::vector<llama_token> pattern;
    pattern.reserve(N);
    for (std::size_t j = cur_len - N + 1; j < cur_len; ++j) {
      pattern.push_back(tokens[j]);
    }
    pattern.push_back(sampled);
    std::size_t match_pos = 0;
    for (std::size_t j = cur_len - N - 1; j > 0; --j) {
      bool match = true;
      for (std::size_t k = 0; k < pattern.size(); ++k) {
        if (tokens[j + k] != pattern[k]) {
          match = false;
          break;
        }
      }
      if (match) {
        match_pos = j;
        break;
      }
    }
    if (match_pos == 0) return result;
    const std::size_t copy_max = std::min(M, cur_len - (match_pos + N));
    if (copy_max < N) return result;
    result.reserve(copy_max);
    for (std::size_t j = 0; j < copy_max; ++j) {
      result.push_back(tokens[match_pos + N + j]);
    }
    this->last_draft_probs.assign(result.size(), std::numeric_limits<double>::quiet_NaN());
    return result;
  }

  std::vector<llama_token> draft(void) {
    llama_memory_t mem_draft = llama_get_memory(this->ctx_draft);

    int reuse_i = 0; // the index of the first token to be reused
    int reuse_n = 0; // how much tokens can we reuse

    const std::vector<llama_token> &current_prompt = this->prompt_target;

    //   context size of draft model   [48]
    // - max tokens to draft at a time [16]
    // ____________________________________
    //
    //   tokens waiting to be drafted  [32]
    //
    const uint32_t draft_n_ctx_u = llama_n_ctx(ctx_draft);
    if (this->params.n_max >= (int64_t)draft_n_ctx_u) {
      throw std::runtime_error(std::format(
          "draft n_max ({}) must be less than draft model context size ({})",
          this->params.n_max, draft_n_ctx_u));
    }
    const int n_ctx = (int)draft_n_ctx_u - (int)this->params.n_max;

    // the index of the first token waiting to be drafted
    const int i_start = std::max(0, (int)current_prompt.size() - n_ctx);

    // reuse as much as possible from the old draft context
    // ideally, the draft context should be as big as the target context
    // and we will always reuse the entire prompt
    for (int i = 0; i < (int)this->prompt_draft.size(); ++i) {
      int cur = 0;
      while (i_start + cur < (int)current_prompt.size() && i + cur < (int)this->prompt_draft.size() &&
             current_prompt[(std::size_t)(i_start + cur)] == this->prompt_draft[(std::size_t)(i + cur)]) {
        cur++;
      }
      if ((cur >= 256 || n_ctx >= (int)current_prompt.size()) && cur > reuse_n) {
        reuse_i = i;
        reuse_n = cur;
      }
    }

    std::vector<llama_token> result;
    result.reserve((std::size_t)this->params.n_max); // n_max tokens to be drafted at a time

    if (reuse_n == 0) {
      // nothing to be reused
      llama_memory_clear(mem_draft, false);
      this->prompt_draft.clear();
    } else {
      // this happens when a previous draft has been discarded (for example, due to being too small),
      // but the target model agreed with it. in this case, we simply pass back the previous results
      // to save compute
      if (reuse_i + reuse_n < (int)this->prompt_draft.size() &&
          this->prompt_draft[(std::size_t)(reuse_i + reuse_n)] == this->last_token) {
        for (int i = reuse_i + reuse_n + 1; i < (int)this->prompt_draft.size(); ++i) {
          result.push_back(this->prompt_draft[(std::size_t)i]);
          if (this->params.n_max <= (int)result.size()) {
            break;
          }
        }
        return result;
      }

      if (reuse_i > 0) {
        llama_memory_seq_rm(mem_draft, 0, 0, reuse_i);
        llama_memory_seq_add(mem_draft, 0, reuse_i, -1, -reuse_i);
        this->prompt_draft.erase(this->prompt_draft.begin(), this->prompt_draft.begin() + reuse_i);
      }

      if (reuse_n < (int)this->prompt_draft.size()) {
        llama_memory_seq_rm(mem_draft, 0, reuse_n, -1);
        this->prompt_draft.erase(this->prompt_draft.begin() + reuse_n, this->prompt_draft.end());
      }
    }

    // clean slate
    this->reset_batch(this->speculation_batch_draft);

    const int32_t draft_batch_cap = (int32_t)llama_n_batch(this->ctx_draft);
    for (std::size_t i = (std::size_t)i_start + (std::size_t)reuse_n; i < current_prompt.size(); ++i) {
      this->create_new_batch(
          this->speculation_batch_draft, draft_batch_cap,
          current_prompt[i],
          (llama_pos)(i - (std::size_t)i_start), false);
      // update the draft prefix
      this->prompt_draft.push_back(current_prompt[i]);
    }

    //
    // TODO is this needed?
    // we can just llama_decode the speculation_batch_draft after adding the last_token batch
    // normally our batch is one new token each time (after the first full-prompt decode)
    //
    {
      if (this->speculation_batch_draft.n_tokens > 0) {
        // evaluate the batch (update KV cache and compute logits for the batch)
        if (llama_decode(this->ctx_draft, this->speculation_batch_draft)) {
          throw std::runtime_error("draft model: failed to decode prompt window");
        }
      }

      // clean slate again
      this->reset_batch(this->speculation_batch_draft);
    }

    // position of last_token equals the draft prefix
    this->create_new_batch(
        this->speculation_batch_draft, draft_batch_cap,
        this->last_token,
        (llama_pos)this->prompt_draft.size(),
        true);

    // update the draft prefix with the last_token
    this->prompt_draft.push_back(this->last_token);

    // evaluate the batch (update KV cache and compute logits for the batch)
    if (llama_decode(this->ctx_draft, this->speculation_batch_draft)) {
      throw std::runtime_error("draft model: failed to decode last context token");
    }

    // clean up
    llama_sampler_reset(this->sampler_draft);

    this->last_draft_probs.clear();
    this->last_draft_probs.reserve((std::size_t)this->params.n_max);

    for (int i = 0; i < this->params.n_max; ++i) {
      // just like the sample_and_accept method
      // only this time we need to be careful to not surpass n_max

      this->reset_batch(this->speculation_batch_draft);

      // turn logits into one chosen token
      // given the current logits, pick a token
      const llama_token accepted_token = llama_sampler_sample(this->sampler_draft, this->ctx_draft, 0);

      //
      // capture the draft's probability mass on the token it just sampled (the p_draft in tokens.csv)
      //
      {
        const float *draft_logits = llama_get_logits_ith(this->ctx_draft, 0);
        auto [_logit, p_d] = this->softmax(draft_logits, accepted_token, this->vocab_draft);
        (void)_logit;
        this->last_draft_probs.push_back(p_d);
      }

      // that token is now part of the sequence, update internal state
      llama_sampler_accept(this->sampler_draft, accepted_token);
      result.push_back(accepted_token);

      // make sure we don't surpass the max number of tokens to draft during speculative decoding
      if (this->params.n_max <= (int)result.size()) {
        break;
      }

      // next position is always current prompt_draft.size()
      this->create_new_batch(
          this->speculation_batch_draft, draft_batch_cap,
          accepted_token,
          (llama_pos)this->prompt_draft.size(),
          true);

      // evaluate the batch (update KV cache and compute logits for the batch)
      if (llama_decode(this->ctx_draft, this->speculation_batch_draft)) {
        break;
      }

      this->prompt_draft.push_back(accepted_token);
    }

    return result;
  }
};

int main(int argc, char **argv) {
  try {
    Application app(argc, argv);
    app.start();
  } catch (const std::exception &e) {
    print(GGML_LOG_LEVEL_ERROR, e.what());
    return 1;
  }
  return 0;
}
