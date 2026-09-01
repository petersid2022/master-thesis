#include <array>
#include <atomic>
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
#include <mutex>
#include <nvml.h>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <vector>

#include "llama-cpp.h"

/// ===========
///  Utilities
/// ===========

class SpectreError : public std::runtime_error {
public:
  template <typename... Args>
  explicit SpectreError(std::format_string<Args...> fmt, Args &&...args)
      : std::runtime_error(std::format(fmt, std::forward<Args>(args)...)) {}
};

class NvmlGpu {
public:
  enum class Mode { Unavailable,
                    EnergyCounter,
                    PowerDraw };

  NvmlGpu() {
    if (nvmlInit_v2() != NVML_SUCCESS) {
      return;
    }
    initialized_ = true;

    if (nvmlDeviceGetHandleByIndex_v2(0, &device_) != NVML_SUCCESS) {
      return;
    }

    char buf[NVML_DEVICE_NAME_BUFFER_SIZE]{};
    if (nvmlDeviceGetName(device_, buf, NVML_DEVICE_NAME_BUFFER_SIZE) == NVML_SUCCESS) {
      name_ = buf;
    }

    unsigned long long millijoules{};
    if (nvmlDeviceGetTotalEnergyConsumption(device_, &millijoules) == NVML_SUCCESS) {
      mode_ = Mode::EnergyCounter;
      return;
    }

    unsigned int milliwatts{};
    if (nvmlDeviceGetPowerUsage(device_, &milliwatts) == NVML_SUCCESS) {
      mode_ = Mode::PowerDraw;
    }
  }

  NvmlGpu(const NvmlGpu &) = delete;
  NvmlGpu &operator=(const NvmlGpu &) = delete;

  ~NvmlGpu() {
    stop_sampling();
    if (initialized_) {
      nvmlShutdown();
    }
  }

  bool available() const { return mode_ != Mode::Unavailable; }

  Mode mode() const { return mode_; }

  const std::string &name() const { return name_; }

  const char *source_tag() const {
    switch (mode_) {
    case Mode::EnergyCounter:
      return "nvml:0";
    case Mode::PowerDraw:
      return "nvml:0:power.draw";
    case Mode::Unavailable:
      return "nvml:0=unavailable";
    }
    return "nvml:0=unavailable";
  }

  void begin_window() {
    last_joules_.reset();
    last_samples_ = 0;
    switch (mode_) {
    case Mode::EnergyCounter:
      energy_start_mj_ = read_mj();
      break;
    case Mode::PowerDraw:
      start_sampling();
      break;
    case Mode::Unavailable:
      break;
    }
  }

  void end_window() {
    switch (mode_) {
    case Mode::EnergyCounter: {
      const auto end_mj = read_mj();
      if (energy_start_mj_ && end_mj) {
        last_joules_ = static_cast<double>(unsigned_delta(*energy_start_mj_, *end_mj)) / 1000.0;
      }
      break;
    }
    case Mode::PowerDraw:
      last_joules_ = stop_sampling();
      break;
    case Mode::Unavailable:
      break;
    }
  }

  std::optional<double> last_joules() const { return last_joules_; }
  std::size_t last_samples() const { return last_samples_; }

private:
  struct Sample {
    std::chrono::steady_clock::time_point t{};
    unsigned int milliwatts{};
  };

  static constexpr auto kSamplePeriod = std::chrono::milliseconds(50);

  std::optional<std::uint64_t> read_mj() const {
    unsigned long long millijoules{};
    if (nvmlDeviceGetTotalEnergyConsumption(device_, &millijoules) != NVML_SUCCESS) {
      return std::nullopt;
    }
    return static_cast<std::uint64_t>(millijoules);
  }

  void push_sample() {
    unsigned int milliwatts{};
    if (nvmlDeviceGetPowerUsage(device_, &milliwatts) != NVML_SUCCESS) {
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mu_);
    samples_.push_back({now, milliwatts});
  }

  void start_sampling() {
    stop_sampling();
    {
      std::lock_guard<std::mutex> lock(mu_);
      samples_.clear();
    }
    running_.store(true, std::memory_order_relaxed);
    push_sample();
    sampler_ = std::thread([this] {
      while (running_.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(kSamplePeriod);
        if (!running_.load(std::memory_order_relaxed)) {
          break;
        }
        push_sample();
      }
    });
  }

  std::optional<double> stop_sampling() {
    running_.store(false, std::memory_order_relaxed);
    if (sampler_.joinable()) {
      sampler_.join();
    }
    if (mode_ != Mode::PowerDraw) {
      return std::nullopt;
    }
    push_sample();
    std::lock_guard<std::mutex> lock(mu_);
    last_samples_ = samples_.size();
    return integrate_joules(samples_);
  }

  static std::optional<double> integrate_joules(const std::vector<Sample> &samples) {
    if (samples.size() < 2) {
      return std::nullopt;
    }
    double joules = 0.0;
    for (std::size_t i = 1; i < samples.size(); ++i) {
      const double dt = std::chrono::duration<double>(samples[i].t - samples[i - 1].t).count();
      const double w0 = static_cast<double>(samples[i - 1].milliwatts) / 1000.0;
      const double w1 = static_cast<double>(samples[i].milliwatts) / 1000.0;
      joules += 0.5 * (w0 + w1) * dt;
    }
    return joules;
  }

  static std::uint64_t unsigned_delta(std::uint64_t start, std::uint64_t end) {
    if (end >= start) {
      return end - start;
    }
    return (std::numeric_limits<std::uint64_t>::max() - start) + end + 1;
  }

  bool initialized_{};
  Mode mode_{Mode::Unavailable};
  nvmlDevice_t device_{};
  std::string name_;

  std::optional<std::uint64_t> energy_start_mj_;
  std::optional<double> last_joules_;
  std::size_t last_samples_{};

  std::atomic<bool> running_{false};
  std::mutex mu_;
  std::vector<Sample> samples_;
  std::thread sampler_;
};

class InferenceTelemetry {
public:
  struct Telemetry {
    struct llama_perf_context_data perf {};
    std::chrono::steady_clock::time_point start_time{};
    std::chrono::steady_clock::time_point end_time{};

    std::optional<std::uint64_t> start_ujoules;
    std::optional<std::uint64_t> end_ujoules;

    double total_seconds{};
    std::optional<double> cpu_joules;
    std::optional<double> gpu_joules;
    std::optional<double> total_joules;
    std::optional<double> average_wattage;
    std::size_t gpu_samples{};

    double milliseconds() const { return total_seconds * 1000.0; }
  };

  Telemetry prompt{};
  Telemetry decode{};

  InferenceTelemetry() {
    if (read_energy_uj()) {
      rapl_available = true;
      std::ifstream range(kRaplMaxRangePath);
      if (range) {
        range >> rapl_max_range_uj;
      }
    }
  }

  bool energy_available() const { return rapl_available || nvml_.available(); }

  // primary meter for energy_j
  // NVML GPU if it works, else RAPL
  std::string energy_source() const {
    if (nvml_.available()) {
      return nvml_.source_tag();
    }
    if (rapl_available) {
      return "intel-rapl:0";
    }
    return "unavailable";
  }

  std::string energy_detail() const {
    std::string detail;
    detail += rapl_available ? "intel-rapl:0" : "intel-rapl:0=unavailable";
    detail += "  ";
    detail += nvml_.source_tag();
    if (nvml_.available() && !nvml_.name().empty()) {
      detail += " (";
      detail += nvml_.name();
      detail += ")";
    }
    return detail;
  }

  const std::string &gpu_name() const { return nvml_.name(); }

  void begin_measuring(Telemetry &s) {
    s = Telemetry{};
    s.start_ujoules = read_energy_uj();
    nvml_.begin_window();
    s.start_time = std::chrono::steady_clock::now();
  }

  void end_measuring(Telemetry &s) {
    s.end_time = std::chrono::steady_clock::now();
    nvml_.end_window();
    s.end_ujoules = read_energy_uj();

    s.total_seconds = std::chrono::duration<double>(s.end_time - s.start_time).count();
    s.gpu_joules = nvml_.last_joules();
    s.gpu_samples = nvml_.last_samples();

    if (s.start_ujoules && s.end_ujoules) {
      if (auto delta_uj = energy_delta_uj(*s.start_ujoules, *s.end_ujoules)) {
        s.cpu_joules = static_cast<double>(*delta_uj) / 1e6;
      }
    }

    s.total_joules = s.gpu_joules ? s.gpu_joules : s.cpu_joules;
    if (s.total_joules && s.total_seconds > 0.0) {
      s.average_wattage = *s.total_joules / s.total_seconds;
    }
  }

  std::optional<double> decode_joules_per_token(int64_t n_tokens) const {
    if (!decode.total_joules || n_tokens <= 0) {
      return std::nullopt;
    }
    return *decode.total_joules / static_cast<double>(n_tokens);
  }

private:
  static constexpr const char *kRaplEnergyPath = "/sys/class/powercap/intel-rapl:0/energy_uj";
  static constexpr const char *kRaplMaxRangePath = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj";

  NvmlGpu nvml_{};
  bool rapl_available{};
  std::uint64_t rapl_max_range_uj{};

  std::optional<std::uint64_t> read_energy_uj() const {
    std::ifstream file(kRaplEnergyPath);
    if (!file) {
      return std::nullopt;
    }
    std::uint64_t ujoules{};
    if (!(file >> ujoules)) {
      return std::nullopt;
    }
    return ujoules;
  }

  std::optional<std::uint64_t> energy_delta_uj(std::uint64_t start, std::uint64_t end) const {
    if (end >= start) {
      return end - start;
    }
    if (rapl_max_range_uj == 0) {
      return std::nullopt;
    }
    return (rapl_max_range_uj - start) + end;
  }
};

class TerminalColor {
  const char *code;

  static bool should_color() {
    static const bool tty = isatty(fileno(stdout));
    return tty;
  }

public:
  constexpr explicit TerminalColor(const char *ansi) : code(ansi) {}

  friend std::ostream &operator<<(std::ostream &os, const TerminalColor &tc) {
    if (should_color()) {
      os << tc.code;
    }
    return os;
  }

  template <typename... Args>
  std::string operator()(Args &&...args) const {
    std::ostringstream out;

    if (should_color()) {
      out << code;
      ((out << std::forward<Args>(args)), ...);
      out << "\033[0m";
    } else {
      ((out << std::forward<Args>(args)), ...);
    }

    return out.str();
  }
};

namespace Color {
inline constexpr TerminalColor Red("\033[31m");
inline constexpr TerminalColor Green("\033[32m");
inline constexpr TerminalColor Yellow("\033[33m");
inline constexpr TerminalColor Blue("\033[34m");
inline constexpr TerminalColor Reset("\033[0m");
} // namespace Color

static inline std::string log_level_to_string(enum ggml_log_level level) {
  using namespace Color;

  switch (level) {
  case GGML_LOG_LEVEL_DEBUG:
    return Green("[DEBUG] ");
  case GGML_LOG_LEVEL_CONT:
  case GGML_LOG_LEVEL_INFO:
    return Blue("[INFO] ");
  case GGML_LOG_LEVEL_WARN:
    return Yellow("[WARN] ");
  case GGML_LOG_LEVEL_ERROR:
    return Red("[ERROR] ");
  case GGML_LOG_LEVEL_NONE:
  default:
    return Reset("");
  }
}

template <typename... Args>
static inline void print(enum ggml_log_level level, std::string_view fmt, const Args &...args) {
  try {
    auto message = std::vformat(fmt, std::make_format_args(args...));
    std::cout << log_level_to_string(level) << message << '\n';
  } catch (const std::format_error &e) {
    std::cout << log_level_to_string(GGML_LOG_LEVEL_ERROR)
              << "print(): std::format_error: " << e.what()
              << "  fmt=\"" << fmt << "\"\n";
  }
}

template <typename... Args>
static inline void print(std::string_view fmt, const Args &...args) {
  print(GGML_LOG_LEVEL_INFO, fmt, args...);
}

struct LlamaModelDeleter {
  void operator()(llama_model *m) {
    if (m) {
      llama_model_free(m);
    }
  }
};

struct LlamaContextDeleter {
  void operator()(llama_context *c) {
    if (c) {
      llama_free(c);
    }
  }
};

struct LlamaSamplerDeleter {
  void operator()(llama_sampler *s) {
    if (s) {
      llama_sampler_free(s);
    }
  }
};

/// =================
///  Data Structures
/// =================

struct InferenceParameters {
  // the number of layers to store in VRAM (<0 means all layers)
  int32_t gpu_layers = -1;

  // text context size, 0 = from model
  uint32_t context_size = 0;

  // hard cap on generated tokens (0 = unlimited, stop only on EOS / KV exhaustion)
  int64_t max_generated_tokens = 0;

  /// =================================
  ///  stochastic speculative sampling
  /// =================================

  // updates logit_i' = logit_i / temp
  float temperature = 0.8f;

  // https://arxiv.org/abs/1904.09751
  float top_p = 0.90f;
  int32_t top_k = 40;

  uint32_t seed = 1234;

  /// ================================
  ///  greedy exact-match speculation
  /// ================================

  // select the token with the highest prob
  bool greedy = false;

  /// ==================================
  ///  n-gram implementation parameters
  /// ==================================

  // the n-gram cache implementation maintains statistics about short n-gram sequences
  bool ngram = false;

  // length of the lookup pattern (n-gram)
  // lower values fire more often on short prompts,
  // higher values reduce false positives on long prompts
  int32_t n_gram_size = 12;

  // maximum length of the proposed draft (m-gram) following an n-gram hit
  int32_t m_gram_size = 64;

  std::string prompt = "Write a Python class called Record with 20 properties: id, name, email,"
                       "phone, address, city, state, zip_code, country, age, salary, department,"
                       "role, manager, status, created_at, updated_at, is_active, score, notes."
                       "For each property implement a getter and setter using exactly this pattern:"
                       "def get_X(self): return self._X and def set_X(self, value): self._X = value";

  /// =====================================
  ///  reproducibility / structured output
  /// =====================================

  std::string run_id; // auto-generated if empty
  std::string results_dir = "results/spectre";
  bool verbose = false; // per-round recap on stdout (off = generated text only)

  /// =================================
  ///  speculative decoding parameters
  /// =================================

  int64_t min_tokens_to_draft = 0;   // minimum number of draft tokens to use for speculative decoding
  int64_t max_tokens_to_draft = 8;   // maximum number of tokens to draft during speculative decoding
  int64_t tokens_accepted_count = 0; // number of tokens accepted by the target model
  int64_t tokens_drafted_count = 0;  // number of tokens drafted by the draft model

  // used to determine end of generation
  bool has_encountered_eos = false;

  std::string draft_model_path;
  std::string target_model_path;

  bool draft_speculative_decoding_is_enabled() const {
    return !draft_model_path.empty();
  }

  bool ngram_speculative_decoding_is_enabled() const {
    return ngram;
  }
};

enum SpeculationAlgorithm {
  NgramSimple,
  NgramMod,
  NgramCache,
  DraftBased,
  Invalid
};

static constexpr std::string_view speculation_algorithm_name(SpeculationAlgorithm algorithm) {
  switch (algorithm) {
  case SpeculationAlgorithm::NgramSimple:
    return "ngram";
  case SpeculationAlgorithm::NgramMod:
    return "ngram-mod";
  case SpeculationAlgorithm::NgramCache:
    return "ngram-cache";
  case SpeculationAlgorithm::DraftBased:
    return "draft";
  case SpeculationAlgorithm::Invalid:
    return "none";
  }
  return "none";
}

enum VerificationKind {
  Correction,
  Bonus,
  Draft,
  Autoregressive
};

static constexpr std::string_view verification_kind_name(VerificationKind kind) {
  switch (kind) {
  case VerificationKind::Bonus:
    return "bonus";
  case VerificationKind::Correction:
    return "correction";
  case VerificationKind::Draft:
    return "draft";
  case VerificationKind::Autoregressive:
    return "ar";
  }
  return "ar";
}

struct VerificationResults {
  llama_token target_token;
  std::vector<llama_token> accepted_drafts;
  VerificationKind kind = VerificationKind::Autoregressive;
  std::optional<std::size_t> rejected_proposal_index = std::nullopt;
};

struct InferenceRoundSummary {
  int tokens_drafted_this_round = 0;
  int drafts_accepted_this_round = 0;
  std::optional<std::size_t> rejected_proposal_index = std::nullopt;
};

class InferenceRunRecorder {
public:
  InferenceRunRecorder(const std::string &results_dir,
                       const std::string &run_id,
                       const std::string &started_at_iso,
                       const InferenceParameters &p)
      : run_dir(std::filesystem::path(results_dir) / run_id),
        run_id_(run_id),
        started_at_(started_at_iso),
        params_snapshot_(p) {
    std::filesystem::create_directories(run_dir);

    tokens.open(run_dir / "tokens.csv");
    if (!tokens) {
      throw SpectreError("failed to open {}", (run_dir / "tokens.csv").string());
    }
    tokens << "step,call,source,pos_in_draft,token_id,p_target,rejected_token_id,p_draft,logit,logprob\n";

    write_metadata(false, /* complete */
                   0,     /* tokens_decoded_count */
                   0,     /* tokens_drafted_count */
                   0,     /* drafts_accepted_count */
                   0,     /* bonus_tokens_drafted_in_round */
                   0.0,   /* prompt_ms */
                   0.0    /* decode_ms */
    );
  }

  void record_token(int call,
                    std::string_view source,
                    std::optional<std::size_t> position_in_draft,
                    int token_id,
                    double p_target,
                    std::optional<int> rejected_token_id,
                    double p_draft,
                    double logit,
                    double logprob) {

    tokens << step << ',' << call << ',' << source << ','
           << position_in_draft.value_or(-1) << ',' << token_id << ','
           << fmt_double(p_target) << ','
           << (rejected_token_id.has_value() ? std::to_string(*rejected_token_id) : std::string{}) << ','
           << fmt_double(p_draft) << ','
           << fmt_double(logit) << ','
           << fmt_double(logprob) << '\n'
           << std::flush;

    step += 1;
  }

  void record_round(int tokens_drafted_count,
                    int drafts_accepted_count,
                    std::optional<std::size_t> rejected_proposal_index) {
    rounds.push_back({tokens_drafted_count, drafts_accepted_count, rejected_proposal_index});
  }

  void finalize(int64_t tokens_generated_in_round,
                int64_t tokens_drafted_count,
                int64_t drafts_accepted_count,
                int64_t bonus_tokens_drafted_in_round,
                double prompt_ms,
                double decode_ms,
                std::optional<double> energy_j = std::nullopt,
                std::optional<double> j_per_token = std::nullopt,
                std::optional<double> average_wattage = std::nullopt,
                std::string_view energy_source = "unavailable",
                std::optional<double> cpu_energy_j = std::nullopt,
                std::optional<double> gpu_energy_j = std::nullopt,
                std::string_view gpu_name = "",
                std::size_t gpu_sample_count = 0) {
    tokens.flush();
    tokens.close();

    write_metadata(true,                          /* complete */
                   tokens_generated_in_round,     /* tokens_decoded_count */
                   tokens_drafted_count,          /* tokens_drafted_count */
                   drafts_accepted_count,         /* drafts_accepted_count */
                   bonus_tokens_drafted_in_round, /* bonus_tokens_drafted_in_round */
                   prompt_ms,                     /* prompt_ms */
                   decode_ms,                     /* decode_ms */
                   energy_j,
                   j_per_token,
                   average_wattage,
                   energy_source,
                   cpu_energy_j,
                   gpu_energy_j,
                   gpu_name,
                   gpu_sample_count);
  }

  const std::filesystem::path &dir() const { return run_dir; }

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
  void write_metadata(bool complete,
                      int64_t tokens_decoded_count,
                      int64_t tokens_drafted_count,
                      int64_t drafts_accepted_count,
                      int64_t bonus_tokens_drafted_in_round,
                      double prompt_ms,
                      double decode_ms,
                      std::optional<double> energy_j = std::nullopt,
                      std::optional<double> j_per_token = std::nullopt,
                      std::optional<double> average_wattage = std::nullopt,
                      std::string_view energy_source = "unavailable",
                      std::optional<double> cpu_energy_j = std::nullopt,
                      std::optional<double> gpu_energy_j = std::nullopt,
                      std::string_view gpu_name = "",
                      std::size_t gpu_sample_count = 0) {

    const std::filesystem::path final_path = run_dir / "meta.json";
    const std::filesystem::path tmp_path = run_dir / "meta.json.tmp";

    std::ofstream m(tmp_path);
    if (!m) {
      throw SpectreError("failed to open {}", tmp_path.string());
    }

    const InferenceParameters &p = params_snapshot_;

    const double total_ms = prompt_ms + decode_ms;
    const double accept_rate = tokens_drafted_count > 0 ? static_cast<double>(drafts_accepted_count) / static_cast<double>(tokens_drafted_count) : 0.0;
    const double tok_per_s = decode_ms > 0.0 ? (1000.0 * static_cast<double>(tokens_decoded_count) / decode_ms) : 0.0;

    m << "{\n";
    m << "  \"run_id\": \"" << json_escape(run_id_) << "\",\n";
    m << "  \"started_at\": \"" << json_escape(started_at_) << "\",\n";
    m << "  \"complete\": " << (complete ? "true" : "false") << ",\n";
    m << "  \"config\": {\n";
    m << "    \"target_model_path\": \"" << json_escape(p.target_model_path) << "\",\n";
    m << "    \"draft_model_path\": " << (p.draft_model_path.empty() ? "null" : ("\"" + json_escape(p.draft_model_path) + "\"")) << ",\n";
    m << "    \"speculative\": " << ((!p.draft_model_path.empty() || p.ngram) ? "true" : "false") << ",\n";
    m << "    \"ctx\": " << p.context_size << ",\n";
    m << "    \"ngl\": " << p.gpu_layers << ",\n";
    m << "    \"n_min\": " << p.min_tokens_to_draft << ",\n";
    m << "    \"n_max\": " << p.max_tokens_to_draft << ",\n";
    m << "    \"ngram\": " << (p.ngram ? "true" : "false") << ",\n";
    m << "    \"n_gram_size\": " << p.n_gram_size << ",\n";
    m << "    \"m_gram_size\": " << p.m_gram_size << ",\n";
    m << "    \"temp\": " << p.temperature << ",\n";
    m << "    \"top_p\": " << p.top_p << ",\n";
    m << "    \"top_k\": " << p.top_k << ",\n";
    m << "    \"greedy\": " << (p.greedy ? "true" : "false") << ",\n";
    m << "    \"seed\": " << p.seed << ",\n";
    m << "    \"n_predict\": " << p.max_generated_tokens << ",\n";
    m << "    \"prompt_n_chars\": " << p.prompt.size() << ",\n";
    m << "    \"prompt\": \"" << json_escape(p.prompt) << "\"\n";
    m << "  },\n";
    m << "  \"totals\": {\n";
    m << "    \"n_decoded_tokens\": " << tokens_decoded_count << ",\n";
    m << "    \"n_drafted\": " << tokens_drafted_count << ",\n";
    m << "    \"n_accepted_drafts\": " << drafts_accepted_count << ",\n";
    m << "    \"n_bonus_samples\": " << bonus_tokens_drafted_in_round << ",\n";
    m << "    \"accept_rate\": " << accept_rate << ",\n";
    m << "    \"prompt_ms\": " << prompt_ms << ",\n";
    m << "    \"decode_ms\": " << decode_ms << ",\n";
    m << "    \"total_ms\": " << total_ms << ",\n";
    m << "    \"tok_per_s\": " << tok_per_s << ",\n";
    m << "    \"energy_j\": ";
    write_json_optional_double(m, energy_j);
    m << ",\n";
    m << "    \"j_per_token\": ";
    write_json_optional_double(m, j_per_token);
    m << ",\n";
    m << "    \"average_wattage\": ";
    write_json_optional_double(m, average_wattage);
    m << ",\n";
    m << "    \"energy_source\": \"" << json_escape(energy_source) << "\",\n";
    m << "    \"cpu_energy_j\": ";
    write_json_optional_double(m, cpu_energy_j);
    m << ",\n";
    m << "    \"gpu_energy_j\": ";
    write_json_optional_double(m, gpu_energy_j);
    m << ",\n";
    m << "    \"gpu_name\": ";
    if (gpu_name.empty()) {
      m << "null";
    } else {
      m << "\"" << json_escape(gpu_name) << "\"";
    }
    m << ",\n";
    m << "    \"gpu_sample_count\": ";
    if (gpu_sample_count > 0) {
      m << gpu_sample_count;
    } else {
      m << "null";
    }
    m << "\n";
    m << "  },\n";

    m << "  \"rounds\": [";
    for (std::size_t i = 0; i < rounds.size(); ++i) {
      const auto &r = rounds[i];
      if (i > 0) m << ",";
      m << "\n    {\"n_drafted\": " << r.tokens_drafted_this_round
        << ", \"n_accepted_drafts\": " << r.drafts_accepted_this_round
        << ", \"rejected_proposal_index\": ";
      if (r.rejected_proposal_index.has_value()) {
        m << *r.rejected_proposal_index;
      } else {
        m << "null";
      }
      m << "}";
    }
    if (!rounds.empty()) m << "\n  ";
    m << "]\n";
    m << "}\n";
    m.flush();
    m.close();

    std::error_code ec;
    std::filesystem::rename(tmp_path, final_path, ec);
    if (ec) {
      throw SpectreError("failed to rename {} -> {}: {}", tmp_path.string(), final_path.string(), ec.message());
    }
  }

  static std::string fmt_double(double v) {
    if (std::isnan(v)) return std::string{};
    return std::format("{:.8g}", v);
  }

  static void write_json_optional_double(std::ostream &os, const std::optional<double> &v) {
    if (v.has_value() && std::isfinite(*v)) {
      os << *v;
    } else {
      os << "null";
    }
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
  InferenceParameters params_snapshot_;
  std::ofstream tokens;
  std::vector<InferenceRoundSummary> rounds;
  int step = 0;
};

class SpectreConfig {
private:
  InferenceParameters params;

  void print_usage(char *argv[]) const;

public:
  static SpectreConfig from_args(int argc, char *argv[]);

  const InferenceParameters &parameters() const { return params; }
};

void SpectreConfig::print_usage(char *argv[]) const {
  const char *name = argv[0];

  // truncate the default prompt for display
  constexpr std::size_t PROMPT_PREVIEW_MAX = 60;
  std::string prompt_preview = params.prompt;

  if (prompt_preview.size() > PROMPT_PREVIEW_MAX) {
    prompt_preview = prompt_preview.substr(0, PROMPT_PREVIEW_MAX) + "...";
  }

  print("Usage: {} --target-model <file.gguf> [--draft-model <file.gguf>] [OPTIONS]", name);
  print("");
  print("Models:");
  print("  --target-model <file>    gguf target model file (required)");
  print("  --draft-model <file>     gguf draft model file (enables draft based speculative decoding)");
  print("");
  print("Runtime:");
  print("  --ctx-size <n>           context size in tokens (0 = from model) (default: {})", params.context_size);
  print("  --n-gpu-layers <n>       layers in VRAM (<0 = all) (default: {})", params.gpu_layers);
  print("  --n-predict <n>          hard cap on generated tokens (0 = unlimited) (default: {})", params.max_generated_tokens);
  print("");
  print("Sampling:");
  print("  --temp <n>               temperature (default: {})", params.temperature);
  print("  --top-p <n>              top-p sampling (default: {})", params.top_p);
  print("  --top-k <n>              top-k sampling (default: {})", params.top_k);
  print("  --greedy                 greedy sampler; overrides temp/top-p/top-k (default: {})", params.greedy ? "true" : "false");
  print("  --prompt <text>          initial prompt (default: \"{}\")", prompt_preview);
  print("");
  print("Speculation (only effective when --draft-model is set):");
  print("  --ngram                  enable n-gram drafter (hybrid: ngram first, draft model on miss) (default: {})", params.ngram);
  print("  --n-gram-size <n>        lookup pattern length for the n-gram drafter (default: {})", params.n_gram_size);
  print("  --m-gram-size <n>        max draft length proposed after an n-gram hit (default: {})", params.m_gram_size);
  print("  --n-max <n>              max tokens to draft per speculative call (default: {})", params.max_tokens_to_draft);
  print("  --n-min <n>              min draft length; below this the draft is discarded (default: {})", params.min_tokens_to_draft);
  print("");
  print("Output / reproducibility:");
  print("  --seed <n>               sampler seed (default: {})", params.seed);
  print("  --run-id <id>            unique run identifier (default: auto-generated as YYYYMMDD-HHMMSS_<mode>_seed<N>)");
  print("  --results-dir <path>     where to write <run-id>/{{meta.json,tokens.csv}} (default: \"{}\")", params.results_dir);
  print("  --verbose                per-round recap: drafter, accepted n/k, draft vs target, token ids (default: {})", params.verbose ? "true" : "false");
  print("");
  print("Misc:");
  print("  -h, --help               print this message and exit");
}

SpectreConfig SpectreConfig::from_args(int argc, char *argv[]) {
  SpectreConfig config{};
  InferenceParameters &params = config.params;

  for (int i = 1; i < argc; i++) {
    try {
      if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
        config.print_usage(argv);
        std::exit(0);
      } else if (std::strcmp(argv[i], "--target-model") == 0) {
        if (i + 1 < argc) {
          params.target_model_path = argv[++i];
        } else {
          config.print_usage(argv);
          throw SpectreError("Missing argument for target model");
        }
      } else if (std::strcmp(argv[i], "--draft-model") == 0) {
        if (i + 1 < argc) {
          params.draft_model_path = argv[++i];
        } else {
          config.print_usage(argv);
          throw SpectreError("Missing argument for draft model");
        }
      } else if (std::strcmp(argv[i], "--ctx-size") == 0) {
        if (i + 1 < argc) {
          params.context_size = (uint32_t)std::stoi(argv[++i]);
        } else {
          config.print_usage(argv);
          throw SpectreError("Missing argument for context size");
        }
      } else if (std::strcmp(argv[i], "--n-gpu-layers") == 0) {
        if (i + 1 < argc) {
          params.gpu_layers = std::stoi(argv[++i]);
        } else {
          config.print_usage(argv);
          throw SpectreError("Missing argument for n-gpu-layers");
        }
      } else if (std::strcmp(argv[i], "--prompt") == 0) {
        if (i + 1 < argc) {
          params.prompt = argv[++i];
        } else {
          config.print_usage(argv);
          throw SpectreError("Missing argument for prompt");
        }
      } else if (std::strcmp(argv[i], "--temp") == 0) {
        if (i + 1 < argc) {
          params.temperature = std::stof(argv[++i]);
        } else {
          config.print_usage(argv);
          throw SpectreError("Missing argument for temperature");
        }
      } else if (std::strcmp(argv[i], "--top-p") == 0) {
        if (i + 1 < argc) {
          params.top_p = std::stof(argv[++i]);
        } else {
          config.print_usage(argv);
          throw SpectreError("Missing argument for top-p");
        }
      } else if (std::strcmp(argv[i], "--greedy") == 0) {
        params.greedy = true;
      } else if (std::strcmp(argv[i], "--verbose") == 0) {
        params.verbose = true;
      } else if (std::strcmp(argv[i], "--ngram") == 0) {
        params.ngram = true;
      } else if (std::strcmp(argv[i], "--n-gram-size") == 0) {
        if (i + 1 < argc) {
          params.n_gram_size = std::stoi(argv[++i]);
        } else {
          config.print_usage(argv);
          throw SpectreError("Missing argument for --n-gram-size");
        }
      } else if (std::strcmp(argv[i], "--m-gram-size") == 0) {
        if (i + 1 < argc) {
          params.m_gram_size = std::stoi(argv[++i]);
        } else {
          config.print_usage(argv);
          throw SpectreError("Missing argument for --m-gram-size");
        }
      } else if (std::strcmp(argv[i], "--top-k") == 0) {
        if (i + 1 < argc) {
          params.top_k = std::stoi(argv[++i]);
        } else {
          config.print_usage(argv);
          throw SpectreError("Missing argument for top-k");
        }
      } else if (std::strcmp(argv[i], "--seed") == 0) {
        if (i + 1 < argc) {
          params.seed = (uint32_t)std::stoul(argv[++i]);
        } else {
          config.print_usage(argv);
          throw SpectreError("Missing argument for --seed");
        }
      } else if (std::strcmp(argv[i], "--results-dir") == 0) {
        if (i + 1 < argc) {
          params.results_dir = argv[++i];
        } else {
          config.print_usage(argv);
          throw SpectreError("Missing argument for --results-dir");
        }
      } else if (std::strcmp(argv[i], "--run-id") == 0) {
        if (i + 1 < argc) {
          params.run_id = argv[++i];
        } else {
          config.print_usage(argv);
          throw SpectreError("Missing argument for --run-id");
        }
      } else if (std::strcmp(argv[i], "--n-predict") == 0) {
        if (i + 1 < argc) {
          params.max_generated_tokens = std::stoll(argv[++i]);
        } else {
          config.print_usage(argv);
          throw SpectreError("Missing argument for --n-predict");
        }
      } else if (std::strcmp(argv[i], "--n-max") == 0) {
        if (i + 1 < argc) {
          params.max_tokens_to_draft = std::stoll(argv[++i]);
        } else {
          config.print_usage(argv);
          throw SpectreError("Missing argument for --n-max");
        }
      } else if (std::strcmp(argv[i], "--n-min") == 0) {
        if (i + 1 < argc) {
          params.min_tokens_to_draft = std::stoll(argv[++i]);
        } else {
          config.print_usage(argv);
          throw SpectreError("Missing argument for --n-min");
        }
      } else {
        config.print_usage(argv);
        throw SpectreError("Unknown argument: {}", argv[i]);
      }
    } catch (const std::invalid_argument &e) {
      throw SpectreError("Invalid numeric value '{}': {}", argv[i], e.what());
    } catch (const std::out_of_range &e) {
      throw SpectreError("Numeric value out of range '{}': {}", argv[i], e.what());
    }
  }

  if (params.target_model_path.empty()) {
    config.print_usage(argv);
    throw SpectreError("Error: --target-model argument is required");
  }

  if (params.run_id.empty()) {
    const std::string ts = InferenceRunRecorder::iso_timestamp(true);

    const std::string_view mode = (params.draft_speculative_decoding_is_enabled() ||
                                   params.ngram_speculative_decoding_is_enabled())
                                      ? "spec"
                                      : "ar";

    params.run_id = params.greedy
                        ? std::format("{}_{}_greedy", ts, mode)
                        : std::format("{}_{}_seed{}", ts, mode, params.seed);
  }

  return config;
}

class Spectre {
private:
  InferenceParameters params;
  InferenceTelemetry telemetry;

  /// ======================================================================
  ///  decode token -> token enters KV cache and produces next-token logits
  ///  sample token -> selects from logits but does not enter KV cache
  /// ======================================================================

  // token that's accepted but not yet in the target kv cache
  llama_token pending_token = 0;

  std::vector<llama_token> tokens_already_in_target_kv;

  // what tokens currently sit in the draft KV cache
  std::vector<llama_token> tokens_in_draft_kv;

  std::unique_ptr<llama_model, LlamaModelDeleter> model_weights_target;
  std::unique_ptr<llama_model, LlamaModelDeleter> model_weights_draft;

  // execution state including kv cache (target and draft therefore have separate kv caches)
  std::unique_ptr<llama_context, LlamaContextDeleter> ctx_target;
  std::unique_ptr<llama_context, LlamaContextDeleter> ctx_draft;

  // tokens ids and positions
  llama_batch batch{};
  llama_batch speculative_batch_draft{};
  llama_batch speculative_batch_target{};

  // converts one logits row into a selected token
  std::unique_ptr<llama_sampler, LlamaSamplerDeleter> sampler_target;
  std::unique_ptr<llama_sampler, LlamaSamplerDeleter> sampler_draft;

  // token id metadata and token-to-text conversion
  const struct llama_vocab *vocabulary_draft = nullptr;
  const struct llama_vocab *vocabulary_target = nullptr;

  // for this proposal, what did the drafter think q(token) was?
  std::vector<double> last_draft_probabilities;

  std::optional<InferenceRunRecorder> recorder;
  int tokens_generated_in_round = 0;
  int bonus_tokens_drafted_in_round = 0;

  enum SpeculationAlgorithm algorithm = SpeculationAlgorithm::Invalid;

  const bool ds = params.draft_speculative_decoding_is_enabled(); // draft based speculative decoding
  const bool ns = params.ngram_speculative_decoding_is_enabled(); // ngram based speculative decoding

  const struct llama_sampler_chain_params default_sampler_params = []() {
    struct llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();

    sampler_params.no_perf = false;

    return sampler_params;
  }();

  const struct llama_context_params default_ctx_params = [this]() {
    struct llama_context_params ctx_params = llama_context_default_params();

    ctx_params.no_perf = false;
    ctx_params.n_ctx = params.context_size;

    return ctx_params;
  }();

  const struct llama_model_params default_model_params = [this]() {
    struct llama_model_params model_params = llama_model_default_params();

#if 0
      ggml_backend_dev_t model_backend = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
      model_params.devices = &model_backend;
#endif

    model_params.n_gpu_layers = params.gpu_layers;
    model_params.load_mode = llama_supports_mmap()
                                 ? LLAMA_LOAD_MODE_MMAP
                                 : LLAMA_LOAD_MODE_NONE;

    return model_params;
  }();

  void escape_newlines_in_place(std::string &str) {
    size_t pos = 0;
    while ((pos = str.find('\n', pos)) != std::string::npos) {
      str.replace(pos, 1, "\\n");
      pos += 2;
    }
  }

  std::string token_to_string(const struct llama_vocab *vocab, llama_token token, bool special = true,
                              bool escape_newlines = true) {
    std::string piece;
    piece.resize(piece.capacity());

    const int count = llama_token_to_piece(vocab, token, &piece[0], static_cast<int>(piece.size()), 0, special);
    if (count < 0) {
      piece.resize(static_cast<std::size_t>(-count));
      const int check = llama_token_to_piece(vocab, token, &piece[0], static_cast<int>(piece.size()), 0, special);
      if (check != -count) {
        throw SpectreError("failed to convert token {} to piece", token);
      }
    } else {
      piece.resize(static_cast<std::size_t>(count));
    }

    if (escape_newlines) {
      escape_newlines_in_place(piece);
    }

    return piece;
  }

  void emit_generated_token(llama_token id) {
    const std::string piece = token_to_string(vocabulary_target, id, true, false);
    std::cout.write(piece.data(), static_cast<std::streamsize>(piece.size()));
    std::cout.flush();
  }

  void print_round_recap(int round,
                         SpeculationAlgorithm algo,
                         VerificationKind kind,
                         const std::vector<llama_token> &proposed,
                         const std::vector<llama_token> &accepted,
                         llama_token kept,
                         std::optional<std::size_t> rejected_index) {
    using namespace Color;

    const auto bar = [this](llama_token id) {
      return std::format("|{}|", token_to_string(vocabulary_target, id));
    };

    if (params.verbose) {
      std::cout << '\n';
      print("round {}  drafter={}  accepted {}/{}",
            round, speculation_algorithm_name(algo),
            static_cast<int>(accepted.size()),
            static_cast<int>(proposed.size()));

      if (kind == VerificationKind::Correction && rejected_index &&
          *rejected_index < proposed.size()) {
        print("  draft   {}", Blue(bar(proposed[*rejected_index])));
        print("  target  {}", Yellow(bar(kept)));
      } else if (kind == VerificationKind::Bonus) {
        print("  bonus   {}", bar(kept));
      }

      std::string rejected = "-";
      if (rejected_index && *rejected_index < proposed.size()) {
        rejected = std::to_string(static_cast<int>(proposed[*rejected_index]));
      }
      print("  kind={}  kept_id={}  rejected_id={}",
            verification_kind_name(kind), kept, rejected);
    }
  }

  std::tuple<double, double> softmax(const float *logits_row,
                                     const llama_token token,
                                     const struct llama_vocab *vocab = nullptr) {

    if (vocab == nullptr) vocab = vocabulary_target;
    const int count = llama_vocab_n_tokens(vocab);

    double max_logit = logits_row[0];
    for (int j = 1; j < count; ++j) {
      if (logits_row[j] > max_logit) {
        max_logit = logits_row[j];
      }
    }

    double denom = 0.0;
    for (int j = 0; j < count; ++j) {
      denom += std::exp(static_cast<double>(logits_row[j]) - max_logit);
    }

    const llama_token tid = token;
    const double logit = logits_row[tid];
    const double prob = std::exp(static_cast<double>(logit) - max_logit) / denom;

    return std::make_tuple(logit, prob);
  }

  // TODO
  std::tuple<double, double> stable_sigmoid(const float *logits_row,
                                            const llama_token token,
                                            const struct llama_vocab *vocab = nullptr) {

    if (vocab == nullptr) vocab = vocabulary_target;

    const llama_token tid = token;
    const double logit = logits_row[tid];

    float prob;
    if (logit >= 0.0f) {
      prob = 1.0f / (1.0f + static_cast<float>(std::exp(-logit)));
    } else {
      const float exp_logit = static_cast<float>(std::exp(logit));
      prob = exp_logit / (1.0f + exp_logit);
    }

    return std::make_tuple(logit, prob);
  }

  void init_backend() {
    auto logger = [](ggml_log_level level, const char *text, void *) {
      if (level == GGML_LOG_LEVEL_DEBUG) {
        return;
      }
      std::cout << log_level_to_string(level) << text << std::flush;
    };

    llama_log_set(logger, nullptr);

    llama_backend_init();

    print("llama_print_system_info:       {}", llama_print_system_info());
    print("llama_supports_mmap:           {}", llama_supports_mmap());
    print("llama_supports_mlock:          {}", llama_supports_mlock());
    print("llama_supports_gpu_offload:    {}", llama_supports_gpu_offload());
  }

  void load_draft() {
    model_weights_draft.reset(llama_model_load_from_file(params.draft_model_path.c_str(), default_model_params));
    if (!model_weights_draft) {
      throw SpectreError("failed to load draft model");
    }

    print("draft_llama_model_n_params:    {}", llama_model_n_params(model_weights_draft.get()));

    ctx_draft.reset(llama_init_from_model(model_weights_draft.get(), default_ctx_params));
    if (!ctx_draft) {
      throw SpectreError("failed to create the llama_context for draft");
    }

    auto d = ctx_draft.get();

    print("draft_llama_n_ctx:        {}", llama_n_ctx(d));
    print("draft_llama_n_ctx_seq:    {}", llama_n_ctx_seq(d));
    print("draft_llama_n_batch:      {}", llama_n_batch(d));
    print("draft_llama_n_ubatch:     {}", llama_n_ubatch(d));
    print("draft_llama_n_seq_max:    {}", llama_n_seq_max(d));
    print("draft_llama_model_chat_template:\n{}", llama_model_chat_template(model_weights_draft.get(), nullptr));
  }

  void load_target() {
    model_weights_target.reset(llama_model_load_from_file(params.target_model_path.c_str(), default_model_params));
    if (!model_weights_target) {
      throw SpectreError("failed to load target model");
    }

    print("target_llama_model_n_params:    {}", llama_model_n_params(model_weights_target.get()));

    ctx_target.reset(llama_init_from_model(model_weights_target.get(), default_ctx_params));
    if (!ctx_target) {
      throw SpectreError("failed to create the llama_context for target");
    }

    auto t = ctx_target.get();

    print("target_llama_n_ctx:        {}", llama_n_ctx(t));
    print("target_llama_n_ctx_seq:    {}", llama_n_ctx_seq(t));
    print("target_llama_n_batch:      {}", llama_n_batch(t));
    print("target_llama_n_ubatch:     {}", llama_n_ubatch(t));
    print("target_llama_n_seq_max:    {}", llama_n_seq_max(t));
    print("target_llama_model_chat_template:\n{}", llama_model_chat_template(model_weights_target.get(), nullptr));
  }

  void validate_vocab_compat() {
    const auto validate_token_match = [&](const std::string &token_type,
                                          bool add_target, bool add_draft,
                                          int id_target, int id_draft) {
      bool add_mismatch = (add_target != add_draft);
      bool id_mismatch = (add_target && (id_target != id_draft));

      if (add_mismatch || id_mismatch) {
        throw SpectreError("{}: draft model {} tokens must match target model to use speculation. "
                           "add: {} - {}, id: {} - {}",
                           __func__, token_type, add_target, add_draft, id_target, id_draft);
      }
    };

    // validate beginning-of-sentence (BOS) tokens
    validate_token_match("bos",
                         llama_vocab_get_add_bos(vocabulary_target), llama_vocab_get_add_bos(vocabulary_draft),
                         llama_vocab_bos(vocabulary_target), llama_vocab_bos(vocabulary_draft));

    // validate end-of-sentence (EOS) tokens
    validate_token_match("eos",
                         llama_vocab_get_add_eos(vocabulary_target), llama_vocab_get_add_eos(vocabulary_draft),
                         llama_vocab_eos(vocabulary_target), llama_vocab_eos(vocabulary_draft));

    const int32_t vocab_target_count = llama_vocab_n_tokens(vocabulary_target);
    const int32_t vocab_draft_count = llama_vocab_n_tokens(vocabulary_draft);
    const int32_t vocab_diff = std::abs(vocab_target_count - vocab_draft_count);

    constexpr int delta = 128;

    // check vocabulary size delta
    if (vocab_diff > delta) {
      throw SpectreError("{}: draft model vocab must closely match target model to use speculation but "
                         "target vocab size {} does not match draft vocab size {} - difference {}, max allowed {}",
                         __func__, vocab_target_count, vocab_draft_count, vocab_diff, delta);
    }

    // validate token content matches
    const int32_t check_limit = std::min(vocab_target_count, vocab_draft_count);
    for (int32_t i = delta; i < check_limit; ++i) {
      std::string_view token_target = llama_vocab_get_text(vocabulary_target, i);
      std::string_view token_draft = llama_vocab_get_text(vocabulary_draft, i);

      if (token_target != token_draft) {
        throw SpectreError("{}: draft model vocab must match target model to"
                           " use speculation but token {} content differs",
                           __func__, i);
      }
    }
  }

  void prepare_prompt() {
    auto wt = model_weights_target.get();
    auto wd = model_weights_draft.get();

    llama_chat_message msg{
        .role = "user",                  /* role */
        .content = params.prompt.c_str() /* content */
    };

    const char *tmpl = llama_model_chat_template(wt, nullptr);
    if (tmpl == nullptr) {
      throw SpectreError("llama_model_chat_template failed");
    }

    const int32_t need = llama_chat_apply_template(tmpl, &msg, 1, true, nullptr, 0);
    if (need < 0) {
      throw SpectreError("this custom template is not supported");
    }

    const std::size_t len = static_cast<std::size_t>(need);
    std::string rendered(len, '\0');

    const int32_t got = llama_chat_apply_template(tmpl,                                 /* tmpl */
                                                  &msg,                                 /* chat */
                                                  1,                                    /* n_msg */
                                                  true,                                 /* add_ass */
                                                  rendered.data(),                      /* buf */
                                                  static_cast<int32_t>(rendered.size()) /* length */
    );

    if (got != static_cast<int32_t>(rendered.size())) {
      throw SpectreError("this custom template is not supported");
    }

    vocabulary_target = llama_model_get_vocab(wt);

    if (params.draft_speculative_decoding_is_enabled()) {
      vocabulary_draft = llama_model_get_vocab(wd);
    }

    // tokenize the rendered chat template
    const int32_t prompt_target_len = -llama_tokenize(vocabulary_target,                     /* vocab */
                                                      rendered.c_str(),                      /* text */
                                                      static_cast<int32_t>(rendered.size()), /* text_len */
                                                      nullptr,                               /* tokens */
                                                      0,                                     /* n_tokens_max */
                                                      true,                                  /* add_special */
                                                      true                                   /* parse_special */
    );

    if (prompt_target_len <= 0) {
      throw SpectreError("failed to tokenize rendered chat prompt (n = {})", prompt_target_len);
    }

    tokens_already_in_target_kv.resize(static_cast<uint32_t>(prompt_target_len));

    int32_t n = llama_tokenize(vocabulary_target,                                        /* vocab */
                               rendered.c_str(),                                         /* text */
                               static_cast<int32_t>(rendered.size()),                    /* text_len */
                               tokens_already_in_target_kv.data(),                       /* tokens */
                               static_cast<int32_t>(tokens_already_in_target_kv.size()), /* n_tokens_max */
                               true,                                                     /* add_special */
                               true                                                      /* parse_special */
    );

    if (n < 0) {
      throw SpectreError("failed to tokenize rendered chat prompt (n = {})", n);
    }

    tokens_already_in_target_kv.resize(static_cast<uint32_t>(n));

    print("\"{}\" ({} tokens)", rendered.c_str(), n);

    auto target_kv_size = static_cast<uint32_t>(tokens_already_in_target_kv.size());

    // if context size < kv cache size then we've got a problem
    if (llama_n_ctx(ctx_target.get()) < target_kv_size) {
      throw SpectreError("the prompt exceeds the context size ({} tokens, ctx {})", target_kv_size, llama_n_ctx(ctx_target.get()));
    }

    // if capacity < kv cache size then we've got a problem
    if (llama_n_batch(ctx_target.get()) < target_kv_size) {
      throw SpectreError("the prompt exceeds the batch size ({} tokens, batch {})", target_kv_size, llama_n_batch(ctx_target.get()));
    }

    for (auto id : tokens_already_in_target_kv) {
      print("|{}|", token_to_string(vocabulary_target, id).c_str());
    }

    print("llama_vocab_n_tokens:    {}", llama_vocab_n_tokens(vocabulary_target));
    print("llama_vocab_type:        {}", static_cast<int>(llama_vocab_type(vocabulary_target)));
  }

  void add_default_sample_chains(struct llama_sampler *chain) {
    if (params.greedy) {
      // greedy sampler. select the token with the highest probability (logit)
      // at each step of text generation, leading to deterministic and generally more focused outputs
      llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
      llama_sampler_chain_add(chain, llama_sampler_init_temp(params.temperature));
      llama_sampler_chain_add(chain, llama_sampler_init_top_k(params.top_k));
      llama_sampler_chain_add(chain, llama_sampler_init_top_p(params.top_p, 1));
      llama_sampler_chain_add(chain, llama_sampler_init_dist(params.seed));
    }
  }

  void init_target_sampler() {
    sampler_target.reset(llama_sampler_chain_init(default_sampler_params));
    if (!sampler_target) {
      throw SpectreError("failed to create the sampler_params");
    }

    add_default_sample_chains(sampler_target.get());
  }

  void init_draft_sampler() {
    sampler_draft.reset(llama_sampler_chain_init(default_sampler_params));
    if (!sampler_draft) {
      throw SpectreError("failed to create draft sampler chain");
    }

    add_default_sample_chains(sampler_draft.get());
  }

  void init_llama_batches() {
    // context holds the size of batch
    speculative_batch_target = llama_batch_init(static_cast<int32_t>(llama_n_batch(ctx_target.get())), 0, 1);

    if (ds) {
      speculative_batch_draft = llama_batch_init(static_cast<int32_t>(llama_n_batch(ctx_draft.get())), 0, 1);
    }
  }

  bool are_ngram_proposed_tokens_usable(const std::vector<llama_token> &proposals) {
    auto n = static_cast<std::size_t>(params.n_gram_size);
    if (proposals.empty() || n == 0) return false;

    auto proposal_len = proposals.size();
    auto kv_cache_len = tokens_already_in_target_kv.size();

    if (proposal_len >= n && kv_cache_len >= n) {
      bool echoes = true;

      for (std::size_t k = 0; k < n; ++k) {
        if (proposals[k] != tokens_already_in_target_kv[kv_cache_len - n + k]) {
          echoes = false;
          break;
        }
      }
      if (echoes) return false;
    }

    // reject period-2 / period-3 loops: A A A... or A B A B...
    auto periodic = [&](std::size_t p) {
      if (p == 0 || proposals.size() < 2 * p) return false;

      for (std::size_t i = p; i < proposals.size(); ++i) {
        if (proposals[i] != proposals[i - p]) return false;
      }
      return true;
    };

    if (periodic(1) || periodic(2) || periodic(3)) return false;

    return true;
  }

  void prefill_target_prefix() {
    llama_synchronize(ctx_target.get());
    if (params.draft_speculative_decoding_is_enabled()) {
      llama_synchronize(ctx_draft.get());
    }

    recorder.emplace(params.results_dir, params.run_id, InferenceRunRecorder::iso_timestamp(), params);
    print("writing structured run output to: {}", recorder->dir().string());

    if (!telemetry.energy_available()) {
      print(GGML_LOG_LEVEL_ERROR,
            "no energy meter: RAPL energy_uj unreadable and NVML libnvidia-ml.so unavailable ({})",
            telemetry.energy_detail());
    } else {
      print("energy meter: {}", telemetry.energy_detail());
    }

    params.tokens_accepted_count = 0;
    params.tokens_drafted_count = 0;
    params.has_encountered_eos = false;

    tokens_generated_in_round = 0;
    bonus_tokens_drafted_in_round = 0;

    // we've already filled target's kv cache vector with the prompt
    if (tokens_already_in_target_kv.empty()) {
      throw SpectreError("tokens_already_in_target_kv is empty");
    }

    //
    // pack all tokens from prompt BUT the last one into a single batch
    //
    // TODO: redundant for autoregressive runs but simplifies speculative verification
    //
    auto t = tokens_already_in_target_kv.data();
    auto n = static_cast<int32_t>(tokens_already_in_target_kv.size()) - 1;

    batch = llama_batch_get_one(t, n);

    telemetry.begin_measuring(telemetry.prompt);

    // evaluate prompt => update KV cache and compute logits for the prompt
    if (llama_decode(ctx_target.get(), batch)) {
      throw SpectreError("failed to eval prompt prefix on target");
    }

    llama_synchronize(ctx_target.get());
    if (ds) {
      llama_synchronize(ctx_draft.get());
    }
    telemetry.end_measuring(telemetry.prompt);
    telemetry.begin_measuring(telemetry.decode);

    // sample starting from the last token of the prompt
    pending_token = tokens_already_in_target_kv.back();

    // don't forget to remove the token from the target kv cache
    tokens_already_in_target_kv.pop_back();

    // place the very last token to a new batch
    batch = llama_batch_get_one(&pending_token, 1);
  }

  void generate_speculative() {
    /* ====================== */
    /*  speculative decoding  */
    /* ====================== */

    if (ds && ns) {
      print("speculative decoding using draft model and ngram cache is enabled");
    } else if (ds) {
      print("speculative decoding using draft model is enabled");
    } else if (ns) {
      print("speculative decoding using ngram cache is enabled");
    }

    int speculative_round = 0;

    auto print_model_desc = [](std::string_view label, const auto &weights) {
      std::array<char, 1024> buf{};
      int32_t len = llama_model_desc(weights.get(), buf.data(), buf.size());

      if (len > 0) {
        size_t l = std::min(static_cast<size_t>(len), buf.size() - 1);
        std::string_view desc(buf.data(), l);
        print("{:<13}:    {}", label, desc);
      }
    };

    print_model_desc("target_model", model_weights_target);

    if (params.draft_speculative_decoding_is_enabled()) {
      print_model_desc("draft_model", model_weights_draft);
    }

    // get a pointer to target's context
    llama_memory_t mem_target = llama_get_memory(ctx_target.get());

    while (!params.has_encountered_eos) /* decoding event loop that only stops when we encounter end-of-sentence */
    {

      //
      // append the pending target token immediately after sequence already cached prefix
      // we use the KV cache as the position source of truth after rollback operations
      //
      const llama_pos max_cached_position = llama_memory_seq_pos_max(mem_target, 0);
      llama_pos next_target_token_position = (max_cached_position < 0) ? 0 : (max_cached_position + 1);

      //
      // sample proposed tokens
      //
      // they may come from a draft model or n-gram and are not yet accepted
      //
      std::vector<llama_token> proposed_tokens;

      // n-gram first if --ngram is set, otherwise (or on miss) the draft model
      if (params.ngram) {
        proposed_tokens = draft_using_ngram_simple();
      }

      // fallback (hybrid)
      if (!are_ngram_proposed_tokens_usable(proposed_tokens)) {
        proposed_tokens.clear();
        algorithm = SpeculationAlgorithm::Invalid;
        if (ds) {
          proposed_tokens = draft_using_draft_model();
        }
      }

      auto max = static_cast<std::size_t>(params.max_tokens_to_draft);
      auto min = static_cast<std::size_t>(params.min_tokens_to_draft);

      if (proposed_tokens.size() > max) {
        proposed_tokens.resize(max);
      } else if (proposed_tokens.size() < min) {
        proposed_tokens.clear();
        algorithm = SpeculationAlgorithm::Invalid;
      }

      // reset target batch so we can reuse it on later iterations
      reset_batch(speculative_batch_target);

      // add pending_token to the batch we send to target
      create_new_batch(speculative_batch_target,
                       static_cast<int32_t>(llama_n_batch(ctx_target.get())), /* batch capacity */
                       pending_token,
                       next_target_token_position);

      next_target_token_position += 1;

      // add drafted tokens to the target
      for (std::size_t i = 0; i < proposed_tokens.size(); ++i) {
        create_new_batch(speculative_batch_target,
                         static_cast<int32_t>(llama_n_batch(ctx_target.get())), /* batch capacity */
                         proposed_tokens[i],
                         next_target_token_position + (llama_pos)i);
      }

      //
      // evaluate the batch => update KV cache and compute logits for the batch
      //
      if (llama_decode(ctx_target.get(), speculative_batch_target)) {
        throw SpectreError("target speculative verification decode failed");
      }

      //
      // do the actual verification of the sampled tokens
      //
      auto verifications = verify_draft_proposals(proposed_tokens);

      auto &kind = verifications.kind;
      auto &target_token = verifications.target_token;
      auto &accepted_drafts = verifications.accepted_drafts;
      auto &rejected_proposal_index = verifications.rejected_proposal_index;

      params.tokens_accepted_count += static_cast<int64_t>(accepted_drafts.size());
      params.tokens_drafted_count += static_cast<int64_t>(proposed_tokens.size());

      next_target_token_position += static_cast<llama_pos>(accepted_drafts.size());

      if (kind == VerificationKind::Bonus) {
        bonus_tokens_drafted_in_round += 1;
      }

      for (std::size_t i = 0; i < accepted_drafts.size(); ++i) {

        auto [logit, prob] = softmax(llama_get_logits_ith(ctx_target.get(), (int32_t)i), accepted_drafts[i]);
        const double logprob = prob > 0.0 ? std::log(prob) : -std::numeric_limits<double>::infinity();

        std::optional<std::size_t> position_in_draft = std::optional<std::size_t>{i};

        double draft_probability = std::numeric_limits<double>::quiet_NaN();

        if (position_in_draft && *position_in_draft < last_draft_probabilities.size()) {
          draft_probability = last_draft_probabilities[*position_in_draft];
        }

        recorder->record_token(speculative_round,                    /* call */
                               "draft",                              /* source */
                               position_in_draft,                    /* pos_in_draft */
                               static_cast<int>(accepted_drafts[i]), /* token_id */
                               prob,                                 /* p_target */
                               std::nullopt,                         /* rejected_token_id */
                               draft_probability,                    /* p_draft */
                               logit,                                /* logit */
                               logprob                               /* logprob */
        );

        //
        // we decoded the pending_token so now we add it to the target's KV cache
        //
        tokens_already_in_target_kv.push_back(pending_token);

        //
        // for now temporary, pending_token is the accepted token we are currently on
        //
        pending_token = accepted_drafts[i];

        // first increment overall generated tokens then check
        tokens_generated_in_round += 1;

        // is pending_token end-of-generation?
        if (llama_vocab_is_eog(vocabulary_target, pending_token)) {
          params.has_encountered_eos = true;
          break;
        }

        // hard cap on tokens (treated like EOS for the purposes of clean finalize)
        if (params.max_generated_tokens > 0 &&
            static_cast<int64_t>(tokens_generated_in_round) >= params.max_generated_tokens) {
          params.has_encountered_eos = true;
          break;
        }

        emit_generated_token(pending_token);
      }

      //
      // commit the last pending token into the KV mirror (the last accepted draft, or
      // the round's original pending if no draft was kept)
      //
      tokens_already_in_target_kv.push_back(pending_token);

      // drafts already hit EOS / n_predict: do not emit the correction/bonus
      if (!params.has_encountered_eos) {
        pending_token = target_token;
        tokens_generated_in_round += 1;

        const std::string_view source = [](VerificationKind k) {
          switch (k) {
          case Bonus:
            return "bonus";
          case Correction:
            return "correction";
          case Autoregressive:
            return "ar";
          case Draft:
          default:
            return "draft";
          }
        }(kind);

        auto [logit, prob] = softmax(llama_get_logits_ith(ctx_target.get(), (int32_t)accepted_drafts.size()), target_token);
        const double logprob = prob > 0.0 ? std::log(prob) : -std::numeric_limits<double>::infinity();

        std::optional<std::size_t> position_in_draft;
        std::optional<int> rejected_token_id;
        double draft_probability = std::numeric_limits<double>::quiet_NaN();

        // correction: token_id is X, rejected_token_id is H, p_draft is q(H)
        if (kind == VerificationKind::Correction && rejected_proposal_index.has_value() &&
            *rejected_proposal_index < proposed_tokens.size()) {
          position_in_draft = rejected_proposal_index;
          rejected_token_id = static_cast<int>(proposed_tokens[*rejected_proposal_index]);
          if (*position_in_draft < last_draft_probabilities.size()) {
            draft_probability = last_draft_probabilities[*position_in_draft];
          }
        }

        recorder->record_token(speculative_round,              /* call */
                               source,                         /* source */
                               position_in_draft,              /* pos_in_draft */
                               static_cast<int>(target_token), /* token_id */
                               prob,                           /* p_target */
                               rejected_token_id,              /* rejected_token_id */
                               draft_probability,              /* p_draft */
                               logit,                          /* logit */
                               logprob                         /* logprob */
        );

        if (llama_vocab_is_eog(vocabulary_target, pending_token)) {
          params.has_encountered_eos = true;
        } else if (params.max_generated_tokens > 0 &&
                   static_cast<int64_t>(tokens_generated_in_round) >= params.max_generated_tokens) {
          params.has_encountered_eos = true;
        } else {
          emit_generated_token(pending_token);
        }
      }

      llama_memory_seq_rm(mem_target, 0, next_target_token_position, -1);

      recorder->record_round(static_cast<int>(proposed_tokens.size()),
                             static_cast<int>(accepted_drafts.size()),
                             rejected_proposal_index);

      if (!proposed_tokens.empty()) {
        print_round_recap(speculative_round,
                          algorithm,
                          kind,
                          proposed_tokens,
                          accepted_drafts,
                          target_token,
                          rejected_proposal_index);
      }

      speculative_round += 1;
    }

    std::cout << std::endl;
  }

  void generate_autoregressive() {
    /* ========================= */
    /*  autoregressive decoding  */
    /* ========================= */

    for (;;) {
      //
      // evaluate the batch => update KV cache and compute logits for the batch
      //
      if (llama_decode(ctx_target.get(), batch)) {
        throw SpectreError("failed to eval");
      }

      // sample and accept the last token of the last evaluation (the next token)
      pending_token = llama_sampler_sample(sampler_target.get(), ctx_target.get(), -1);

      auto [logit, prob] = softmax(llama_get_logits_ith(ctx_target.get(), -1), pending_token);
      const double logprob = prob > 0.0 ? std::log(prob) : -std::numeric_limits<double>::infinity();

      recorder->record_token(static_cast<int>(tokens_generated_in_round), /* call */
                             "ar",                                        /* source */
                             std::nullopt,                                /* pos_in_draft */
                             static_cast<int>(pending_token),             /* token_id */
                             prob,                                        /* p_target */
                             std::nullopt,                                /* rejected_token_id */
                             std::numeric_limits<double>::quiet_NaN(),    /* p_draft */
                             logit,                                       /* logit */
                             logprob                                      /* logprob */
      );

      tokens_generated_in_round += 1;

      // is it an end of generation?
      if (llama_vocab_is_eog(vocabulary_target, pending_token)) {
        break;
      }

      // hard cap on tokens
      if (params.max_generated_tokens > 0 && static_cast<int64_t>(tokens_generated_in_round) >= params.max_generated_tokens) {
        break;
      }

      emit_generated_token(pending_token);

      // prepare the next batch with the sampled token
      batch = llama_batch_get_one(&pending_token, 1);
    }

    std::cout << std::endl;
  }

  void finalize_run() {
    llama_synchronize(ctx_target.get());
    if (ds) {
      llama_synchronize(ctx_draft.get());
    }
    telemetry.end_measuring(telemetry.decode);

    const double prompt_ms = telemetry.prompt.milliseconds();
    const double decode_ms = telemetry.decode.milliseconds();

    const float speed = static_cast<float>(tokens_generated_in_round) /
                        std::max(static_cast<float>(decode_ms / 1000.0), 1e-6f);

    print("decoded {} tokens in {:.3f} s, speed: {:.2f} t/s (prompt {:.1f} ms, decode {:.1f} ms)",
          tokens_generated_in_round,
          decode_ms / 1000.0,
          speed,
          prompt_ms,
          decode_ms);

    if (params.tokens_drafted_count > 0) {
      print("speculative: n_drafted = {}, n_accept = {}, accept = {:.2f}%",
            params.tokens_drafted_count, params.tokens_accepted_count,
            100.0 * static_cast<double>(params.tokens_accepted_count) /
                static_cast<double>(params.tokens_drafted_count));
    } else if (ns) {
      print(GGML_LOG_LEVEL_WARN,
            "ngram produced no drafts (n-gram-size={}); decoded target-only. "
            "repeated n-grams are uncommon until the model starts copying the prompt or its own output",
            params.n_gram_size);
    }

    const auto energy_j = telemetry.decode.total_joules;
    const auto j_per_token = telemetry.decode_joules_per_token(tokens_generated_in_round);
    const auto average_wattage = telemetry.decode.average_wattage;
    const auto cpu_energy_j = telemetry.decode.cpu_joules;
    const auto gpu_energy_j = telemetry.decode.gpu_joules;

    if (energy_j) {
      if (telemetry.decode.gpu_samples > 0) {
        print("energy (decode window, {}, {} samples): {:.3f} J, {:.4f} J/token, {:.2f} W avg",
              telemetry.energy_source(),
              telemetry.decode.gpu_samples,
              *energy_j,
              j_per_token.value_or(0.0),
              average_wattage.value_or(0.0));
      } else {
        print("energy (decode window, {}): {:.3f} J, {:.4f} J/token, {:.2f} W avg",
              telemetry.energy_source(),
              *energy_j,
              j_per_token.value_or(0.0),
              average_wattage.value_or(0.0));
      }
    }
    if (cpu_energy_j && gpu_energy_j) {
      print("  cpu intel-rapl:0: {:.3f} J    gpu nvml:0: {:.3f} J", *cpu_energy_j, *gpu_energy_j);
    }

    recorder->finalize(static_cast<int64_t>(tokens_generated_in_round),
                       params.tokens_drafted_count,
                       params.tokens_accepted_count,
                       static_cast<int64_t>(bonus_tokens_drafted_in_round),
                       prompt_ms,
                       decode_ms,
                       energy_j,
                       j_per_token,
                       average_wattage,
                       telemetry.energy_source(),
                       cpu_energy_j,
                       gpu_energy_j,
                       telemetry.gpu_name(),
                       telemetry.decode.gpu_samples);

    print("wrote {} and {}",
          (recorder->dir() / "meta.json").string(),
          (recorder->dir() / "tokens.csv").string());

    llama_perf_sampler_print(sampler_target.get());
    llama_perf_context_print(ctx_target.get());
    if (ds) {
      llama_perf_context_print(ctx_draft.get());
    }
  }

  void reset_batch(llama_batch &batch) {
    batch.n_tokens = 0;
  }

  void create_new_batch(llama_batch &batch, int32_t max_tokens, llama_token id, llama_pos pos, bool output = true) {
    //
    // llama_decode does not take a string but a llama_batch which is a small array of slots,
    // each describing one token we want to process in this forward pass
    // create_new_batch basically is fill the next slot with (token, pos, logits, seq) then n_tokens++
    //
    if (batch.n_tokens >= max_tokens) {
      throw SpectreError("llama_batch capacity exceeded ({}/{})", batch.n_tokens, max_tokens);
    }
    if (batch.seq_id[batch.n_tokens] == nullptr) {
      throw SpectreError("llama_batch seq_id slot missing");
    }
    batch.token[batch.n_tokens] = id;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id[batch.n_tokens][0] = 0; // TODO should we use multiple sequences?
    batch.logits[batch.n_tokens] = output;
    batch.n_tokens++;
  }

  VerificationResults verify_draft_proposals(const std::vector<llama_token> &proposes) {

    auto ctx = ctx_target.get();
    auto sampler = sampler_target.get();

    llama_synchronize(ctx); // wait until all computations are finished

    VerificationResults results;

    auto &kind = results.kind;
    auto &accepted = results.accepted_drafts;
    auto &target_token = results.target_token;
    auto &rejection_position = results.rejected_proposal_index;

    accepted.reserve(proposes.size() + 1);

    for (std::size_t index = 0; index < proposes.size(); ++index) {
      // given the current logits, pick a token using the sampler (greedy or stochastic)
      const llama_token token = llama_sampler_sample(sampler, ctx, static_cast<int32_t>(index));

      // stop at first mismatch
      if (proposes[index] != token) {
        target_token = token;
        rejection_position = index;
        kind = VerificationKind::Correction;
        return results;
      }

      accepted.push_back(token);
    }

    //
    // * after all N proposals match, logits at index N predict the token after the final proposal
    //
    // * sampling it gives the free (no additional target-model forward pass) bonus token from the
    //   same target-model decode
    //
    const llama_token bonus = llama_sampler_sample(sampler, ctx, static_cast<int32_t>(proposes.size()));

    if (proposes.empty()) {
      kind = VerificationKind::Autoregressive;
    } else {
      kind = VerificationKind::Bonus;
    }

    target_token = bonus;
    rejection_position = std::nullopt;

    return results;
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

  // TODO
  std::vector<llama_token> draft_using_ngram_cache() {
    return {};
  }

  // TODO
  std::vector<llama_token> draft_using_ngram_mod() {
    return {};
  }

  std::vector<llama_token> draft_using_ngram_simple() {
    algorithm = SpeculationAlgorithm::NgramSimple;

    const auto &tokens = tokens_already_in_target_kv;
    const llama_token sampled = pending_token; // sampled but not decoded

    const std::size_t length = tokens.size(); // target kv cache length

    const std::size_t N = static_cast<std::size_t>(params.n_gram_size); // how many tokens we match
    const std::size_t M = static_cast<std::size_t>(params.m_gram_size); // max how many tokens we copy after a hit

    std::vector<llama_token> result;

    // need the current n-gram plus at least one earlier position to search
    if (N == 0 || length <= N + 1) {
      return result;
    }

    std::vector<llama_token> pattern;
    pattern.reserve(N);

    // get the first pattern occurence (starting from the end)
    for (std::size_t j = length - N + 1; j < length; ++j) {
      pattern.push_back(tokens[j]);
    }

    // add to that our pending token
    pattern.push_back(sampled);

    std::size_t match_pos = 0;
    for (std::size_t j = length - N - 1; j > 0; --j) {
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

    if (match_pos == 0) {
      return result;
    }

    if (match_pos + N >= length) {
      return result;
    }

    const std::size_t copy_max = std::min(M, length - (match_pos + N));
    if (copy_max == 0) {
      return result;
    }

    result.reserve(copy_max);

    for (std::size_t j = 0; j < copy_max; ++j) {
      result.push_back(tokens[match_pos + N + j]);
    }

    last_draft_probabilities.assign(result.size(), std::numeric_limits<double>::quiet_NaN());

    return result;
  }

  llama_memory_t draft_memory() {
    return llama_get_memory(ctx_draft.get());
  }

  std::size_t draft_kv_len() {
    const llama_pos max_cached_pos = llama_memory_seq_pos_max(draft_memory(), 0);
    return max_cached_pos < 0 ? 0 : static_cast<std::size_t>(max_cached_pos + 1);
  }

  void draft_kv_reset() {
    llama_memory_clear(draft_memory(), false);
    tokens_in_draft_kv.clear();
  }

  //
  // llama KV is the source of truth
  //
  bool draft_kv_keep_first(std::size_t n) {
    if (n >= tokens_in_draft_kv.size()) {
      return true;
    }

    if (!llama_memory_seq_rm(draft_memory(), 0, static_cast<llama_pos>(n), -1)) {
      draft_kv_reset();
      return false;
    }

    tokens_in_draft_kv.resize(n);

    return true;
  }

  bool draft_kv_drop_first(std::size_t n) {
    if (n == 0) {
      return true;
    }

    llama_memory_t mem = draft_memory();

    if (n >= tokens_in_draft_kv.size() ||
        !llama_memory_can_shift(mem) ||
        !llama_memory_seq_rm(mem, 0, 0, static_cast<llama_pos>(n))) {

      draft_kv_reset();
      return false;
    }

    llama_memory_seq_add(mem, 0, static_cast<llama_pos>(n), -1, -static_cast<llama_pos>(n));

    tokens_in_draft_kv.erase(tokens_in_draft_kv.begin(),
                             tokens_in_draft_kv.begin() + static_cast<std::ptrdiff_t>(n));

    return true;
  }

  //
  // propose new tokens using a secondary smaller model
  //
  std::vector<llama_token> draft_using_draft_model() {
    algorithm = SpeculationAlgorithm::DraftBased;

    // the draft-side token mirror must track KV exactly
    if (draft_kv_len() != tokens_in_draft_kv.size()) {
      print(GGML_LOG_LEVEL_WARN,
            "draft(): KV/token mirror drift detected"
            "(kv_len={}, prompt_draft={})",
            draft_kv_len(),
            tokens_in_draft_kv.size());

      print(GGML_LOG_LEVEL_WARN, "resyncing draft state");
      draft_kv_reset();
    }

    int reuse_starting_from = 0; // the index of the first token to be reused
    int reuse_count = 0;         // how much tokens can we reuse

    const std::vector<llama_token> &current_prompt = tokens_already_in_target_kv;

    //
    //   context size of draft model   [48]
    // - max tokens to draft at a time [16]
    // ____________________________________
    //
    //   tokens waiting to be drafted  [32]
    //
    const uint32_t draft_context_size_capacity = llama_n_ctx(ctx_draft.get());

    if (params.max_tokens_to_draft >= static_cast<int64_t>(draft_context_size_capacity)) {
      throw SpectreError("draft n_max ({}) must be less "
                         "than draft model context size ({})",
                         params.max_tokens_to_draft,
                         draft_context_size_capacity);
    }

    const int tokens_queued_to_be_drafted = static_cast<int>(draft_context_size_capacity - params.max_tokens_to_draft);

    const int current_prompt_len = static_cast<int>(current_prompt.size());
    const int prompt_draft_len = static_cast<int>(tokens_in_draft_kv.size());

    // the index of the first token waiting to be drafted
    const int first_token = std::max(0, current_prompt_len - tokens_queued_to_be_drafted);

    // reuse as much as possible from the old draft context
    // ideally, the draft context should be as big as the target context
    // and we will always reuse the entire prompt
    for (int i = 0; i < prompt_draft_len; ++i) {
      int cursor = 0;

      const int max_draft_cursor = prompt_draft_len - i;
      const int max_prompt_cursor = current_prompt_len - first_token;
      const int max_cursor = std::min(max_prompt_cursor, max_draft_cursor);

      while (cursor < max_cursor &&
             current_prompt[static_cast<size_t>(first_token + cursor)] == tokens_in_draft_kv[static_cast<size_t>(i + cursor)]) {
        cursor++;
      }

      if ((cursor >= 256 || tokens_queued_to_be_drafted >= current_prompt_len) && cursor > reuse_count) {
        reuse_starting_from = i;
        reuse_count = cursor;
      }
    }

    std::vector<llama_token> result;
    result.reserve(static_cast<std::size_t>(params.max_tokens_to_draft)); // n_max tokens to be drafted at a time

    if (reuse_count == 0) {
      draft_kv_reset();
    } else {
      // this happens when a previous draft has been discarded (for example, due to being too small),
      // but the target model agreed with it. in this case, we simply pass back the previous results
      // to save compute
      if (reuse_starting_from + reuse_count < prompt_draft_len &&
          tokens_in_draft_kv[(std::size_t)(reuse_starting_from + reuse_count)] == pending_token) {

        for (int i = reuse_starting_from + reuse_count + 1; i < prompt_draft_len; ++i) {
          result.push_back(tokens_in_draft_kv[static_cast<std::size_t>(i)]);

          if (params.max_tokens_to_draft <= static_cast<int>(result.size())) {
            break;
          }
        }

        return result;
      }

      // skip re-evaluating a prefix the draft already computed
      // seq_rm is allowed to fail so on failure both mirrors are wiped and this
      // round prefills from first_token (reuse_count = 0)
      if (reuse_starting_from > 0 &&
          !draft_kv_drop_first(static_cast<std::size_t>(reuse_starting_from))) {
        reuse_count = 0;
      }

      if (reuse_count > 0 &&
          static_cast<std::size_t>(reuse_count) < tokens_in_draft_kv.size() &&
          !draft_kv_keep_first(static_cast<std::size_t>(reuse_count))) {
        reuse_count = 0;
      }

      if (draft_kv_len() != tokens_in_draft_kv.size()) {
        draft_kv_reset();
        reuse_count = 0;
      }
    }

    // clean slate
    reset_batch(speculative_batch_draft);

    const int32_t draft_batch_capacity = static_cast<int32_t>(llama_n_batch(ctx_draft.get()));
    llama_pos next_position = static_cast<llama_pos>(draft_kv_len());

    for (std::size_t i = static_cast<std::size_t>(first_token + reuse_count); i < current_prompt.size(); ++i) {
      create_new_batch(speculative_batch_draft, draft_batch_capacity, current_prompt[i], next_position, false);

      // update the draft prefix
      tokens_in_draft_kv.push_back(current_prompt[i]);
      next_position += 1;
    }

    //
    // TODO is this needed?
    // we can just llama_decode the speculation_batch_draft after adding the pending_token batch
    // normally our batch is one new token each time (after the first full-prompt decode)
    //
    {
      if (speculative_batch_draft.n_tokens > 0) {
        //
        // evaluate the batch => update KV cache and compute logits for the batch
        //
        if (llama_decode(ctx_draft.get(), speculative_batch_draft)) {
          throw SpectreError("draft model: failed to decode prompt window");
        }
      }

      // clean slate again
      reset_batch(speculative_batch_draft);
    }

    // position must come from KV
    const llama_pos last_token_pos = static_cast<llama_pos>(draft_kv_len());

    create_new_batch(speculative_batch_draft, /* batch */
                     draft_batch_capacity,    /* max_tokens */
                     pending_token,           /* id */
                     last_token_pos,          /* pos */
                     true                     /* output */
    );

    //
    // update the draft prefix with the pending_token
    //
    tokens_in_draft_kv.push_back(pending_token);

    //
    // evaluate the batch => update KV cache and compute logits for the batch
    //
    if (llama_decode(ctx_draft.get(), speculative_batch_draft)) {
      throw SpectreError("draft model: failed to decode last context token");
    }

    //
    // clean up
    //
    llama_sampler_reset(sampler_draft.get());

    last_draft_probabilities.clear();
    last_draft_probabilities.reserve((std::size_t)params.max_tokens_to_draft);

    for (int i = 0; i < params.max_tokens_to_draft; ++i) {
      // just like the sample_and_accept method
      // only this time we need to be careful to not surpass n_max

      reset_batch(speculative_batch_draft);

      // turn logits into one chosen token
      // given the current logits, pick a token
      const llama_token proposed_token = llama_sampler_sample(sampler_draft.get(), ctx_draft.get(), 0);

      //
      // capture the draft's probability mass on the token it just sampled (the p_draft in tokens.csv)
      //
      {
        const float *draft_logits = llama_get_logits_ith(ctx_draft.get(), 0);
        auto [_logit, p_d] = softmax(draft_logits, proposed_token, vocabulary_draft);
        (void)_logit;
        last_draft_probabilities.push_back(p_d);
      }

      // make sure we don't surpass the max number of tokens to draft during speculative decoding
      if (params.max_tokens_to_draft <= static_cast<int>(result.size())) {
        break;
      }

      const llama_pos draft_next_pos = static_cast<llama_pos>(draft_kv_len());

      create_new_batch(speculative_batch_draft, /* batch */
                       draft_batch_capacity,    /* max_tokens */
                       proposed_token,          /* id */
                       draft_next_pos,          /* pos */
                       true                     /* output */
      );

      //
      // evaluate the batch => update KV cache and compute logits for the batch
      //
      if (llama_decode(ctx_draft.get(), speculative_batch_draft)) {
        break;
      }

      tokens_in_draft_kv.push_back(proposed_token);
    }

    return result;
  }

public:
  Spectre(const SpectreConfig &config) : params{config.parameters()} {}

  Spectre(const Spectre &) = delete;
  Spectre &operator=(const Spectre &) = delete;

  Spectre(Spectre &&) = delete;
  Spectre &operator=(Spectre &&) = delete;

  ~Spectre() {
    llama_batch_free(speculative_batch_target);
    llama_batch_free(speculative_batch_draft);
    llama_backend_free();
  }

  int run() {

#if !defined(NDEBUG)
    print(GGML_LOG_LEVEL_WARN, "asserts enabled, performance may be affected");
#endif

#if (defined(_MSC_VER) && defined(_DEBUG)) || (!defined(_MSC_VER) && !defined(__OPTIMIZE__))
    print(GGML_LOG_LEVEL_WARN, "debug build, performance may be affected");
#endif

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    print(GGML_LOG_LEVEL_WARN, "sanitizer enabled, performance may be affected");
#endif

    init_backend();

    load_target();
    if (ds) {
      load_draft();
    }

    prepare_prompt();

    if (ds) {
      validate_vocab_compat();
    }

    init_target_sampler();
    if (ds || ns) {
      init_llama_batches();
    }
    if (ds) {
      init_draft_sampler();
    }

    prefill_target_prefix();

    if (ds || ns) {
      generate_speculative();
    } else {
      generate_autoregressive();
    }

    finalize_run();

    return 0;
  }
};

int main(int argc, char *argv[]) {
  try {
    auto config = SpectreConfig::from_args(argc, argv);
    Spectre spectre{config};
    return spectre.run();
  } catch (const std::exception &e) {
    print(GGML_LOG_LEVEL_ERROR, e.what());
    return 1;
  }
  return 0;
}
