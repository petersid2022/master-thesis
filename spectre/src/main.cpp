#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "llama-cpp.h"

static inline const char *log_level_to_string(enum ggml_log_level level) { // {{{
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
} // }}}

template <typename... Args> // {{{
static inline void print(enum ggml_log_level level, std::string_view fmt, Args &&...args) {
  auto message = std::vformat(fmt, std::make_format_args(args...));
  std::cout << log_level_to_string(level) << message << '\n';
} // }}}

struct Parameters // {{{
{
  int32_t ngl = 2048; // the number of layers to store in VRAM (<0 means all layers)
  int32_t ctx = 2048; // text context, 0 = from model (TODO if speculative we use the same context window for both models)

  // Updates logit_i' = logit_i / temp.
  // When temp <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
  float temp = 0.8f;

  // "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
  float top_p = 0.95f;
  int32_t top_k = 40;

  std::string prompt = "How old is the universe?";

  // speculative decoding parameters
  int64_t n_min = 0;     // minimum number of draft tokens to use for speculative decoding
  int64_t n_max = 16;    // maximum number of tokens to draft during speculative decoding
  int64_t n_accept = 0;  // number of tokens accepted by the target model.
  int64_t n_drafted = 0; // number of tokens drafted by the draft model.

  // used to determine end of generation
  bool has_eos = false;

  std::string dft_model_path;
  std::string tgt_model_path;

  bool speculative_decoding_is_enabled() {
    if (!this->dft_model_path.empty()) {
      return true;
    }
    return false;
  }
}; // }}}

class TeeBuf : public std::streambuf { // {{{
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
}; // }}}

class Application // {{{
{
public:
  Application(int argc, char **argv) {
    this->parse_cli_args(argc, argv);
  };

  Application(const Application &) = delete;
  Application &operator=(const Application &) = delete;

  Application(Application &&) = delete;
  Application &operator=(Application &&) = delete;

  ~Application() {
    llama_sampler_free(sampler_tgt);
    if (params.speculative_decoding_is_enabled()) {
      llama_sampler_free(sampler_dft);
      llama_batch_free(speculation_batch_tgt);
      llama_batch_free(speculation_batch_dft);
      llama_free(ctx_dft);
      llama_model_free(model_dft);
    }
    llama_free(ctx_tgt);
    llama_model_free(model_tgt);
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

  llama_model *model_tgt = nullptr;
  llama_model *model_dft = nullptr;

  llama_context *ctx_tgt = nullptr;
  llama_context *ctx_dft = nullptr;

  std::vector<llama_token> prompt_tgt;
  std::vector<llama_token> prompt_dft;

  llama_batch batch{};
  llama_batch speculation_batch_tgt{};
  llama_batch speculation_batch_dft{};

  llama_sampler *sampler_tgt = nullptr;
  llama_sampler *sampler_dft = nullptr;

  const struct llama_vocab *vocab_tgt = nullptr;
  const struct llama_vocab *vocab_dft = nullptr;

  void print_usage(char **argv) { // {{{
    const char *name = argv[0];

    print(GGML_LOG_LEVEL_WARN, "Usage: {} -m model.gguf [OPTIONS]", name);
    print(GGML_LOG_LEVEL_WARN, "");
    print(GGML_LOG_LEVEL_WARN, "Options:");
    print(GGML_LOG_LEVEL_WARN, "  -t,   --temp <n>               [NO-OP] temperature (default: {})", this->params.temp);
    print(GGML_LOG_LEVEL_WARN, "  -p,   --top-p <n>              [NO-OP] top-p sampling (default: {})", this->params.top_p);
    print(GGML_LOG_LEVEL_WARN, "  -k,   --top-k <n>              [NO-OP] top-k sampling (default: {})", this->params.top_k);
    print(GGML_LOG_LEVEL_WARN, "  -pro, --prompt <text>          initial prompt (default: \"{}\")", this->params.prompt);
    print(GGML_LOG_LEVEL_WARN, "  -tgt, --target-model <file>    gguf target model file (required)");
    print(GGML_LOG_LEVEL_WARN, "  -dft, --draft-model <file>     gguf draft model file (required for speculative decoding)");
    print(GGML_LOG_LEVEL_WARN, "  -ctx, --ctx-size <n>           context size in tokens (0 = from model) (default: {})", this->params.ctx);
    print(GGML_LOG_LEVEL_WARN, "  -ngl, --n-gpu-layers <n>       layers in VRAM (<0 = all) (default: {})", this->params.ngl);
    print(GGML_LOG_LEVEL_WARN, "");
    print(GGML_LOG_LEVEL_WARN, "Example:");
    print(GGML_LOG_LEVEL_WARN, "  {} --target-model Qwen2.5-Coder-3B-Instruct-IQ2_M.gguf --prompt \"Tell me a joke\" --ctx-size 8192 --n-gpu-layers 40", name);
  } // }}}

  void parse_cli_args(int argc, char **argv) { // {{{
    for (int i = 1; i < argc; i++) {
      try {
        if (std::strcmp(argv[i], "-tgt") == 0 || std::strcmp(argv[i], "--target-model") == 0) {
          if (i + 1 < argc) {
            this->params.tgt_model_path = argv[++i];
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for target model");
          }
        } else if (std::strcmp(argv[i], "-dft") == 0 || std::strcmp(argv[i], "--draft-model") == 0) {
          if (i + 1 < argc) {
            this->params.dft_model_path = argv[++i];
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for draft model");
          }
        } else if (std::strcmp(argv[i], "-ctx") == 0 || std::strcmp(argv[i], "--ctx-size") == 0) {
          if (i + 1 < argc) {
            this->params.ctx = std::stoi(argv[++i]);
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for context size");
          }
        } else if (std::strcmp(argv[i], "-ngl") == 0 || std::strcmp(argv[i], "--n-gpu-layers") == 0) {
          if (i + 1 < argc) {
            this->params.ngl = std::stoi(argv[++i]);
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for n-gpu-layers");
          }
        } else if (std::strcmp(argv[i], "-pro") == 0 || std::strcmp(argv[i], "--prompt") == 0) {
          if (i + 1 < argc) {
            this->params.prompt = argv[++i];
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for prompt");
          }
        } else if (std::strcmp(argv[i], "-t") == 0 || std::strcmp(argv[i], "--temp") == 0) {
          if (i + 1 < argc) {
            this->params.temp = std::stof(argv[++i]);
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for temperature");
          }
        } else if (std::strcmp(argv[i], "-p") == 0 || std::strcmp(argv[i], "--top-p") == 0) {
          if (i + 1 < argc) {
            this->params.top_p = std::stof(argv[++i]);
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for top-p");
          }
        } else if (std::strcmp(argv[i], "-k") == 0 || std::strcmp(argv[i], "--top-k") == 0) {
          if (i + 1 < argc) {
            this->params.top_k = std::stoi(argv[++i]);
          } else {
            print_usage(argv);
            throw std::runtime_error("Missing argument for top-k");
          }
        } else {
          print_usage(argv);
          throw std::runtime_error(std::format("Unknown argument: {}", argv[i]));
        }
      } catch (const std::exception &e) {
        throw std::runtime_error(std::format("Error parsing argument '{}': {}", argv[i], e.what()));
      }
    }

    if (this->params.tgt_model_path.empty()) {
      print_usage(argv);
      throw std::runtime_error("Error: --target-model (-tgt) argument is required");
    }
  } // }}}

  std::tuple<double, double> softmax( // {{{
      const float *logits_row,
      const llama_token accepted) {
    const int n_vocab = llama_vocab_n_tokens(this->vocab_tgt);

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

    const llama_token tid = accepted;
    const double logit = logits_row[tid];
    const double prob = std::exp((double)logit - max_logit) / denom;

    return std::make_tuple(logit, prob);
  } // }}}

  void initialize(void) { // {{{
    this->log_file.open("log.txt", std::ios::out | std::ios::trunc);
    if (!this->log_file) {
      throw std::runtime_error("failed to open log.txt");
    }
    this->cout_saved = std::cout.rdbuf();
    this->tee_buf = std::make_unique<TeeBuf>(this->cout_saved, this->log_file.rdbuf());
    std::cout.rdbuf(this->tee_buf.get());

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
      ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);

      params.devices = &device;
      params.n_gpu_layers = this->params.ngl;
      params.use_mmap = llama_supports_mmap();
      params.use_mlock = llama_supports_mlock();

      this->model_tgt = llama_model_load_from_file(this->params.tgt_model_path.c_str(), params);
      if (!this->model_tgt) {
        throw std::runtime_error("failed to load target model");
      }

      if (this->params.speculative_decoding_is_enabled()) {
        this->model_dft = llama_model_load_from_file(this->params.dft_model_path.c_str(), params);
        if (!this->model_dft) {
          throw std::runtime_error("failed to load draft model");
        }
      }

      print(GGML_LOG_LEVEL_INFO, "tgt_llama_model_n_params:    {}", llama_model_n_params(this->model_tgt));
      if (this->params.speculative_decoding_is_enabled()) {
        print(GGML_LOG_LEVEL_INFO, "dft_llama_model_n_params:    {}", llama_model_n_params(this->model_dft));
      }

      struct llama_context_params ctx_params = llama_context_default_params();

      ctx_params.no_perf = false;
      ctx_params.n_ctx = this->params.ctx;

      this->ctx_tgt = llama_init_from_model(this->model_tgt, ctx_params);
      if (!this->ctx_tgt) {
        throw std::runtime_error("failed to create the llama_context for target");
      }

      if (this->params.speculative_decoding_is_enabled()) {
        this->ctx_dft = llama_init_from_model(this->model_dft, ctx_params);
        if (!this->ctx_dft) {
          throw std::runtime_error("failed to create the llama_context for draft");
        }
      }

      print(GGML_LOG_LEVEL_INFO, "tgt_llama_n_ctx:        {}", llama_n_ctx(this->ctx_tgt));
      print(GGML_LOG_LEVEL_INFO, "tgt_llama_n_ctx_seq:    {}", llama_n_ctx_seq(this->ctx_tgt));
      print(GGML_LOG_LEVEL_INFO, "tgt_llama_n_batch:      {}", llama_n_batch(this->ctx_tgt));
      print(GGML_LOG_LEVEL_INFO, "tgt_llama_n_ubatch:     {}", llama_n_ubatch(this->ctx_tgt));
      print(GGML_LOG_LEVEL_INFO, "tgt_llama_n_seq_max:    {}", llama_n_seq_max(this->ctx_tgt));

      if (this->params.speculative_decoding_is_enabled()) {
        print(GGML_LOG_LEVEL_INFO, "dft_llama_n_ctx:        {}", llama_n_ctx(this->ctx_dft));
        print(GGML_LOG_LEVEL_INFO, "dft_llama_n_ctx_seq:    {}", llama_n_ctx_seq(this->ctx_dft));
        print(GGML_LOG_LEVEL_INFO, "dft_llama_n_batch:      {}", llama_n_batch(this->ctx_dft));
        print(GGML_LOG_LEVEL_INFO, "dft_llama_n_ubatch:     {}", llama_n_ubatch(this->ctx_dft));
        print(GGML_LOG_LEVEL_INFO, "dft_llama_n_seq_max:    {}", llama_n_seq_max(this->ctx_dft));
      }
    } catch (...) {
      if (this->cout_saved) {
        std::cout.rdbuf(this->cout_saved);
        this->cout_saved = nullptr;
      }
      this->tee_buf.reset();
      throw;
    }
  } // }}}

  void tokenize(void) { // {{{
    this->vocab_tgt = llama_model_get_vocab(this->model_tgt);

    if (this->params.speculative_decoding_is_enabled()) {
      this->vocab_dft = llama_model_get_vocab(this->model_dft);
    }

    const int prompt_tgt_len = -llama_tokenize(
        this->vocab_tgt,
        this->params.prompt.c_str(),
        this->params.prompt.size(),
        NULL, 0, true, true);

    prompt_tgt.resize(prompt_tgt_len);

    int n = llama_tokenize(
        this->vocab_tgt,
        this->params.prompt.c_str(), this->params.prompt.size(),
        prompt_tgt.data(), prompt_tgt.size(),
        true, true);

    if (n < 0) {
      throw std::runtime_error(std::format("failed to tokenize prompt (n = {})", n));
    }

    print(GGML_LOG_LEVEL_INFO, "\"{}\" ({} tokens)", this->params.prompt.c_str(), prompt_tgt_len);

    if (llama_n_ctx(this->ctx_tgt) < (uint32_t)prompt_tgt.size()) {
      throw std::runtime_error(std::format("the prompt exceeds the context size ({} tokens, ctx {})", prompt_tgt.size(), llama_n_ctx(this->ctx_tgt)));
    }

    if (llama_n_batch(this->ctx_tgt) < (uint32_t)prompt_tgt.size()) {
      throw std::runtime_error(std::format("the prompt exceeds the batch size ({} tokens, batch {})", prompt_tgt.size(), llama_n_batch(this->ctx_tgt)));
    }

    if (this->params.speculative_decoding_is_enabled()) {

      if (
          llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
          llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
          llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
          llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)) {
        throw std::runtime_error("draft model special tokens must match target model to use speculation");
      }

      {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_difference = n_vocab_tgt > n_vocab_dft
                                         ? n_vocab_tgt - n_vocab_dft
                                         : n_vocab_dft - n_vocab_tgt;

        if (vocab_difference > 128) {
          throw std::runtime_error(std::format(
              "draft model vocab must closely match target model to use speculation but "
              "target vocab size {} does not match draft vocab size {} - difference {}, max allowed 128",
              n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_difference));
        }
      }
    }

    for (auto id : prompt_tgt) {
      char buf[128] = {0};
      int n = llama_token_to_piece(this->vocab_tgt, id, buf, sizeof(buf), 0, true);
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

    print(GGML_LOG_LEVEL_INFO, "llama_vocab_n_tokens:    {}", llama_vocab_n_tokens(this->vocab_tgt));
    print(GGML_LOG_LEVEL_INFO, "llama_vocab_type:        {}", (int)llama_vocab_type(this->vocab_tgt));
  } // }}}

  void decode(void) { // {{{
    struct llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();

    sampler_params.no_perf = false;

    this->sampler_tgt = llama_sampler_chain_init(sampler_params);
    if (!this->sampler_tgt) {
      throw std::runtime_error("failed to create the llama_sampler_chain_params");
    }

    // llama_sampler_chain_add(this->sampler_tgt, llama_sampler_init_top_k(this->params.top_k));
    // llama_sampler_chain_add(this->sampler_tgt, llama_sampler_init_top_p(this->params.top_p, 1));
    // llama_sampler_chain_add(this->sampler_tgt, llama_sampler_init_temp(this->params.temp));
    // llama_sampler_chain_add(this->sampler_tgt, llama_sampler_init_dist(std::time(nullptr)));

    // greedy sampler. select the token with the highest probability (logit)
    // at each step of text generation, leading to deterministic and generally more focused outputs
    llama_sampler_chain_add(this->sampler_tgt, llama_sampler_init_greedy());

    if (this->params.speculative_decoding_is_enabled()) {
      // context holds the size of batch
      this->speculation_batch_tgt = llama_batch_init(llama_n_batch(this->ctx_tgt), 0, 1);
      this->speculation_batch_dft = llama_batch_init(llama_n_batch(this->ctx_dft), 0, 1);

      this->sampler_dft = llama_sampler_chain_init(sampler_params);
      if (!this->sampler_dft) {
        throw std::runtime_error("failed to create draft sampler chain");
      }

      // llama_sampler_chain_add(this->sampler_dft, llama_sampler_init_top_k(this->params.top_k));
      // llama_sampler_chain_add(this->sampler_dft, llama_sampler_init_top_p(this->params.top_p, 1));
      // llama_sampler_chain_add(this->sampler_dft, llama_sampler_init_temp(this->params.temp));
      // llama_sampler_chain_add(this->sampler_dft, llama_sampler_init_dist(std::time(nullptr)));

      // greedy sampler. select the token with the highest probability (logit)
      // at each step of text generation, leading to deterministic and generally more focused outputs
      llama_sampler_chain_add(this->sampler_dft, llama_sampler_init_greedy());
    }

    // prepare first batch of tokens aka the prompt
    batch = llama_batch_get_one(prompt_tgt.data(), prompt_tgt.size());

    // TODO remove; this is not required since most models are decoder only
    // if (llama_model_has_encoder(this->model_tgt)) {
    //   if (this->params.speculative_decoding_enabled()) {
    //     print(GGML_LOG_LEVEL_WARN,
    //           "Speculative decoding is not implemented for encoder–decoder models; "
    //           "draft model is loaded but generation uses the target only.");
    //   }
    //   // eval the prompt
    //   if (llama_encode(this->ctx_tgt, current_batch)) {
    //     throw std::runtime_error("failed to eval");
    //   }
    //
    //   llama_token decoder_start_token_id = llama_model_decoder_start_token(this->model_tgt);
    //   if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
    //     decoder_start_token_id = llama_vocab_bos(this->vocab_tgt);
    //   }
    //
    //   current_batch = llama_batch_get_one(&decoder_start_token_id, 1);
    // }
  } // }}}

  void run(void) { // {{{
    // current accepted token each pass
    llama_token current_token = -1;
    std::size_t tokens_decoded = 0;

    const int64_t start = ggml_time_us();

    print(GGML_LOG_LEVEL_INFO, "llama_model_chat_template:    \n{}", llama_model_chat_template(this->model_tgt, NULL));

    std::ofstream file("metrics.csv");
    if (!file) {
      throw std::runtime_error("failed to open file");
    }

    file << "step,logit,prob,logprob\n";

    if (this->params.speculative_decoding_is_enabled()) {
      print(GGML_LOG_LEVEL_INFO, "Speculative Decoding is enabled");

      char buf[1024] = {0};
      llama_model_desc(this->model_tgt, buf, sizeof(buf));
      print(GGML_LOG_LEVEL_INFO, "target_model :    {}", buf);
      llama_model_desc(this->model_dft, buf, sizeof(buf));
      print(GGML_LOG_LEVEL_INFO, "draft_model  :    {}", buf);

      // TODO remove; this is not required since most models are decoder only
      // if (llama_model_has_encoder(this->model_tgt)) {
      //   throw std::runtime_error("speculative decoding is only implemented for decoder-only models");
      // }

      this->params.n_accept = 0;
      this->params.n_drafted = 0;
      this->params.has_eos = false; // end-of-sentence

      if (this->prompt_tgt.empty()) {
        throw std::runtime_error("speculative decoding needs a non-empty prompt");
      }

      // get prompt batch
      this->batch = llama_batch_get_one(this->prompt_tgt.data(), (int32_t)this->prompt_tgt.size() - 1);

      // evaluate prompt (aka update KV cache and compute logits for the prompt)
      if (llama_decode(this->ctx_tgt, this->batch)) {
        throw std::runtime_error("failed to eval prompt prefix on target");
      }

      // sample starting from the last token of the prompt
      this->last_token = this->prompt_tgt.back();
      this->prompt_tgt.pop_back();

      llama_memory_t mem_tgt = llama_get_memory(this->ctx_tgt);

      while (!this->params.has_eos) {
        // M-RoPE (Qwen3): batch positions must satisfy Y > X (where X = memory.seq_pos_max).
        // Tracking "pos" as prompt_tgt.size() drifts from real KV (e.g. X=21 vs Y=7).
        // Fix: ask the cache where does the sequence end then +1 for the next real decode step.
        // Note:
        //     X = What's the last position index still in the KV cache?
        //     Y = What position does this new batch claim its _first_ token uses?
        const llama_pos pmax = llama_memory_seq_pos_max(mem_tgt, 0);
        llama_pos n_past = (pmax < 0) ? 0 : (pmax + 1);

        // get drafted tokens, resize if too many, reset if too few
        std::vector<llama_token> draft = this->draft();
        if (draft.size() > (std::size_t)this->params.n_max) {
          draft.resize((std::size_t)this->params.n_max);
        } else if (draft.size() < (std::size_t)this->params.n_min) {
          draft.clear();
        }

        // reset target batch
        this->reset_batch(this->speculation_batch_tgt);

        // add prompt batch to target
        this->create_new_batch(this->speculation_batch_tgt, this->last_token, n_past, true);
        n_past += 1;

        // add drafted tokens to the target
        for (std::size_t i = 0; i < draft.size(); ++i) {
          this->create_new_batch(this->speculation_batch_tgt, draft[i], n_past + (llama_pos)i, true);
        }

        // evaluate the batch (aka update KV cache and compute logits for the batch)
        if (llama_decode(this->ctx_tgt, this->speculation_batch_tgt)) {
          throw std::runtime_error("target speculative verification decode failed");
        }

        // do the actual verification of the sampled tokens
        const std::vector<llama_token> accepted = sample_and_accept(this->sampler_tgt, this->ctx_tgt, draft);
        if (accepted.empty()) {
          throw std::runtime_error("speculative accept produced no tokens");
        }

        n_past += (llama_pos)((int)accepted.size() - 1);
        this->params.n_drafted += (int64_t)draft.size();
        this->params.n_accept += (int64_t)accepted.size() - 1;

        for (std::size_t i = 0; i < accepted.size(); ++i) {
          auto [logit, prob] = softmax(llama_get_logits_ith(this->ctx_tgt, (int32_t)i), accepted[i]);
          file << tokens_decoded << ',' << logit << ',' << prob << ',' << std::log(prob) << '\n';

          this->prompt_tgt.push_back(this->last_token);
          this->last_token = accepted[i];

          // is last_token end-of-generation
          if (llama_vocab_is_eog(this->vocab_tgt, this->last_token)) {
            this->params.has_eos = true;
            break;
          }

          char buf[128] = {0};
          int n = llama_token_to_piece(this->vocab_tgt, this->last_token, buf, sizeof(buf), 0, true);
          if (n < 0) {
            throw std::runtime_error("failed to convert token to piece");
          }
          std::string token(buf, n);
          std::size_t pos = 0;
          while ((pos = token.find('\n', pos)) != std::string::npos) {
            token.replace(pos, 1, "\\n");
            pos += 2;
          }
          // TODO don't write ansi escape code sequences to log.txt
          print(GGML_LOG_LEVEL_INFO,
                // "\x1b[{}m|{}|\x1b[0m{:<{}}"
                "|{}|{:<{}}"
                "(accepted {} out of {} draft tokens, last_token = {})",
                // 36 - (int)(i % 6),
                token.c_str(),
                "", 24 - (int)(token.size() + 2), // 24 variable len spacer
                (int)accepted.size() - 1,
                (int)draft.size(),
                this->last_token);
          tokens_decoded += 1;
        }

        llama_memory_seq_rm(mem_tgt, 0, n_past, -1);
      }

      std::cout << std::endl;

      const int64_t end = ggml_time_us();
      const float delta = (end - start) / 1000000.0f;
      const float speed = tokens_decoded / std::max(delta, 1e-6f);

      print(GGML_LOG_LEVEL_INFO, "decoded {} tokens in {} s, speed: {} t/s", tokens_decoded, delta, speed);

      if (this->params.n_drafted > 0) {
        print(GGML_LOG_LEVEL_INFO,
              "speculative: n_drafted = {}, n_accept = {}, accept = {:.2f}%",
              this->params.n_drafted, this->params.n_accept,
              100.0 * (double)this->params.n_accept / (double)this->params.n_drafted);
      }

      llama_perf_sampler_print(this->sampler_tgt);
      llama_perf_context_print(this->ctx_tgt);
      llama_perf_context_print(this->ctx_dft);
      return;
    }

    for (;;) {
      // evaluate the batch (aka update KV cache and compute logits for the batch)
      if (llama_decode(this->ctx_tgt, batch)) {
        throw std::runtime_error("failed to eval");
      }

      // sample and accept the last token of the last evaluation (the next token)
      current_token = llama_sampler_sample(this->sampler_tgt, this->ctx_tgt, -1);

      auto [logit, prob] = softmax(llama_get_logits_ith(this->ctx_tgt, -1), current_token);
      file << tokens_decoded << ',' << logit << ',' << prob << ',' << std::log(prob) << '\n';

      // is it an end of generation?
      if (llama_vocab_is_eog(this->vocab_tgt, current_token)) {
        break;
      }

      char buf[128] = {0};
      int n = llama_token_to_piece(this->vocab_tgt, current_token, buf, sizeof(buf), 0, true);
      if (n < 0) {
        throw std::runtime_error("failed to convert token to piece");
      }
      std::cout.write(buf, n);
      std::cout.flush();

      // prepare the next batch with the sampled token
      batch = llama_batch_get_one(&current_token, 1);

      tokens_decoded += 1;
    }

    std::cout << std::endl;

    const int64_t end = ggml_time_us();

    float delta = (end - start) / 1000000.0f;
    float speed = tokens_decoded / ((end - start) / 1000000.0f);

    print(GGML_LOG_LEVEL_INFO, "decoded {} tokens in {} s, speed: {} t/s", tokens_decoded, delta, speed);

    // __asm volatile("int3");
    llama_perf_sampler_print(this->sampler_tgt);
    llama_perf_context_print(this->ctx_tgt);
  } // }}}

  void reset_batch(llama_batch &batch) { // {{{
    batch.n_tokens = 0;
  } // }}}

  void create_new_batch( // {{{
      llama_batch &batch,
      llama_token id,
      llama_pos pos,
      bool output) {
    // llama_decode does not take a string but a llama_batch which is a small array of slots,
    // each describing one token we want to process in this forward pass.
    // create_new_batch basically is fill the next slot with (token, pos, logits, seq) then n_tokens++
    assert(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");
    batch.token[batch.n_tokens] = id;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id[batch.n_tokens][0] = 0;
    batch.logits[batch.n_tokens] = output;
    batch.n_tokens++;
  } // }}}

  std::vector<llama_token> sample_and_accept( // {{{
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
  } // }}}

  std::vector<llama_token> draft(void) { // {{{
    llama_memory_t mem_dft = llama_get_memory(this->ctx_dft);

    // TODO implement reuse context window mechanism
    // int reuse_i = 0; // the index of the first token to be reused
    // int reuse_n = 0; // how much tokens can we reuse

    llama_memory_clear(mem_dft, false);
    this->prompt_dft.clear();

    const std::vector<llama_token> &current_prompt = this->prompt_tgt;

    //   context size of draft model   [48]
    // - max tokens to draft at a time [16]
    // ____________________________________
    //
    //   tokens waiting to be drafted  [32]
    //
    const int n_ctx = (int)llama_n_ctx(ctx_dft) - this->params.n_max;

    // the index of the first token waiting to be drafted
    const int i_start = std::max(0, (int)current_prompt.size() - n_ctx);

    // reuse as much as possible from the old draft context
    // ideally, the draft context should be as big as the target context
    // and we will always reuse the entire prompt
    // for (int i = 0; i < (int)this->prompt_dft.size(); ++i) {
    //   int cur = 0;
    //   while (i_start + cur < (int)current_prompt.size() && i + cur < (int)this->prompt_dft.size() &&
    //          current_prompt[(std::size_t)(i_start + cur)] == this->prompt_dft[(std::size_t)(i + cur)]) {
    //     cur++;
    //   }
    //   if ((cur >= 256 || n_ctx >= (int)current_prompt.size()) && cur > reuse_n) {
    //     reuse_i = i;
    //     reuse_n = cur;
    //   }
    // }

    std::vector<llama_token> result;
    result.reserve((std::size_t)this->params.n_max); // n_max tokens to be drafted at a time

    // TODO implement reuse context window mechanism
    // if (reuse_n == 0) {
    //   // nothing to be reused
    //   llama_memory_clear(mem_dft, false);
    //   this->prompt_dft.clear();
    // } else {
    //   // this happens when a previous draft has been discarded (for example, due to being too small),
    //   // but the target model agreed with it. in this case, we simply pass back the previous results
    //   // to save compute
    //   if (reuse_i + reuse_n < (int)this->prompt_dft.size() &&
    //       this->prompt_dft[(std::size_t)(reuse_i + reuse_n)] == this->last_token) {
    //     for (int i = reuse_i + reuse_n + 1; i < (int)this->prompt_dft.size(); ++i) {
    //       result.push_back(this->prompt_dft[(std::size_t)i]);
    //       if (this->params.n_max <= (int)result.size()) {
    //         break;
    //       }
    //     }
    //     return result;
    //   }
    //
    //   if (reuse_i > 0) {
    //     llama_memory_seq_rm(mem_dft, 0, 0, reuse_i);
    //     llama_memory_seq_add(mem_dft, 0, reuse_i, -1, -reuse_i);
    //     this->prompt_dft.erase(this->prompt_dft.begin(), this->prompt_dft.begin() + reuse_i);
    //   }
    //
    //   if (reuse_n < (int)this->prompt_dft.size()) {
    //     llama_memory_seq_rm(mem_dft, 0, reuse_n, -1);
    //     this->prompt_dft.erase(this->prompt_dft.begin() + reuse_n, this->prompt_dft.end());
    //   }
    // }

    // clean slate
    this->reset_batch(this->speculation_batch_dft);

    // for (std::size_t i = (std::size_t)i_start + (std::size_t)reuse_n; i < current_prompt.size(); ++i) {
    for (std::size_t i = i_start; i < current_prompt.size(); ++i) {
      this->create_new_batch(
          this->speculation_batch_dft,
          current_prompt[i],
          (llama_pos)(i - (std::size_t)i_start), false);
      // update the draft prefix
      this->prompt_dft.push_back(current_prompt[i]);
    }

    //
    // TODO is this needed?
    // we can just llama_decode the speculation_batch_dft after adding the last_token batch
    // normally our batch is one new token each time (after the first full-prompt decode)
    //
    {
      if (this->speculation_batch_dft.n_tokens > 0) {
        // evaluate the batch (aka update KV cache and compute logits for the batch)
        if (llama_decode(this->ctx_dft, this->speculation_batch_dft)) {
          throw std::runtime_error("draft model: failed to decode prompt window");
        }
      }

      // clean slate again
      this->reset_batch(this->speculation_batch_dft);
    }

    // position of last_token equals the draft prefix
    this->create_new_batch(
        this->speculation_batch_dft,
        this->last_token,
        (llama_pos)this->prompt_dft.size(),
        true);

    // update the draft prefix with the last_token
    this->prompt_dft.push_back(this->last_token);

    // evaluate the batch (aka update KV cache and compute logits for the batch)
    if (llama_decode(this->ctx_dft, this->speculation_batch_dft)) {
      throw std::runtime_error("draft model: failed to decode last context token");
    }

    // clean up
    llama_sampler_reset(this->sampler_dft);

    for (int i = 0; i < this->params.n_max; ++i) {
      // just like the sample_and_accept method
      // only this time we need to be careful to not surpass n_max

      this->reset_batch(this->speculation_batch_dft);

      // turn logits into one chosen token
      // given the current logits, pick a token
      const llama_token accepted_token = llama_sampler_sample(this->sampler_dft, this->ctx_dft, 0);
      // that token is now part of the sequence, update internal state
      llama_sampler_accept(this->sampler_dft, accepted_token);
      result.push_back(accepted_token);

      // make sure we don't surpass the max number of tokens to draft during speculative decoding
      if (this->params.n_max <= (int)result.size()) {
        break;
      }

      // next position is always current prompt_dft.size()
      this->create_new_batch(
          this->speculation_batch_dft,
          accepted_token,
          (llama_pos)this->prompt_dft.size(),
          true);

      // evaluate the batch (aka update KV cache and compute logits for the batch)
      if (llama_decode(this->ctx_dft, this->speculation_batch_dft)) {
        break;
      }

      this->prompt_dft.push_back(accepted_token);
    }

    return result;
  } // }}}

}; // }}}

int main(int argc, char **argv) { // {{{
  try {
    Application app(argc, argv);
    app.start();
  } catch (const std::exception &e) {
    print(GGML_LOG_LEVEL_ERROR, e.what());
    return 1;
  }
  return 0;
} // }}}
