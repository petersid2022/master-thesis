#include <cmath>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

#include "llama-cpp.h"

typedef enum
{
  INFO,
  ERROR,
  WARNING,
} log_level;

static inline void print(log_level level, const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);

  switch (level)
  {
  case INFO:
    fprintf(stderr, "\x1b[36m[INFO]\x1b[0m ");
    break;
  case WARNING:
    fprintf(stderr, "\x1b[33m[WARNING]\x1b[0m ");
    break;
  case ERROR:
    fprintf(stderr, "\x1b[31m[ERROR]\x1b[0m ");
    break;
  default:
    abort();
  }

  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
  va_end(args);
}

static inline void print_usage(char **argv)
{
  print(WARNING, "Usage: %s -m model.gguf [OPTIONS]", argv[0]);
  print(WARNING, "");
  print(WARNING, "Options:");
  print(WARNING, "    -m,   --model <file>        gguf model file (required)");
  print(WARNING, "    -p,   --prompt <text>       initial prompt");
  print(WARNING, "    -ctx, --ctx-size <n>        context size in tokens");
  print(WARNING, "    -ngl, --n-gpu-layers <n>    the number of layers to store in VRAM (<0 means all layers)");
  print(WARNING, "");
  print(WARNING, "Example:");
  print(WARNING, "  %s -m Qwen2.5-Coder-3B-Instruct-IQ2_M.gguf -p \"Tell me a joke\" -c 8192 -ngl 40", argv[0]);
}

struct Config
{
  int32_t ngl = 2048; // the number of layers to store in VRAM (<0 means all layers)
  int32_t ctx = 2048; // text context, 0 = from model

  // Updates logit_i` = logit_i / temp.
  // When temp <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
  float temp = 0.8f;

  // "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
  float p = 0.95f;
  int32_t k = 40;

  std::string prompt = "How old is the universe?";
  std::string model_path = "../../models/Nemotron-3-Nano-4B-Q8_0.gguf";
};

struct Spectre
{
  llama_model *model = nullptr;
  llama_context *context = nullptr;
  llama_sampler *sampler = nullptr;

  Config config;

  Spectre() = default;

  Spectre(const Spectre &) = delete;
  Spectre &operator=(const Spectre &) = delete;

  Spectre(Spectre &&) = delete;
  Spectre &operator=(Spectre &&) = delete;

  ~Spectre()
  {
    llama_free(context);
    llama_sampler_free(sampler);
    llama_model_free(model);
    llama_backend_free();
  }
};

int main(int argc, char **argv)
{
  Spectre spectre;

  // --------------------------------------------------
  // Parse command line arguments
  // --------------------------------------------------
  for (int i = 1; i < argc; i++)
  {
    try
    {
      if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0)
      {
        if (i + 1 < argc)
        {
          spectre.config.model_path = argv[++i];
        }
        else
        {
          print_usage(argv);
          return 1;
        }
      }
      else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0)
      {
        if (i + 1 < argc)
        {
          spectre.config.prompt = argv[++i];
        }
        else
        {
          print_usage(argv);
          return 1;
        }
      }
      else if (strcmp(argv[i], "-ctx") == 0 || strcmp(argv[i], "--ctx-size") == 0)
      {
        if (i + 1 < argc)
        {
          spectre.config.ctx = std::stoi(argv[++i]);
        }
        else
        {
          print_usage(argv);
          return 1;
        }
      }
      else if (strcmp(argv[i], "-ngl") == 0 || strcmp(argv[i], "--n-gpu-layers") == 0)
      {
        if (i + 1 < argc)
        {
          spectre.config.ngl = std::stoi(argv[++i]);
        }
        else
        {
          print_usage(argv);
          return 1;
        }
      }
      else
      {
        print_usage(argv);
        return 1;
      }
    }
    catch (const std::exception &e)
    {
      print(ERROR, e.what());
      print_usage(argv);
      return 1;
    }
  }

  if (spectre.config.model_path.empty())
  {
    print_usage(argv);
    return 1;
  }

  // --------------------------------------------------
  // Initialization
  // --------------------------------------------------
  llama_backend_init();

  print(INFO, "llama_print_system_info:       %s", llama_print_system_info());
  print(INFO, "llama_supports_mmap:           %d", llama_supports_mmap());
  print(INFO, "llama_supports_mlock:          %d", llama_supports_mlock());
  print(INFO, "llama_supports_gpu_offload:    %d", llama_supports_gpu_offload());

  struct llama_model_params params = llama_model_default_params();
  ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);

  params.devices = &device;
  params.n_gpu_layers = spectre.config.ngl;
  params.use_mmap = llama_supports_mmap();
  params.use_mlock = llama_supports_mlock();

  if (!(spectre.model = llama_model_load_from_file(spectre.config.model_path.c_str(), params)))
  {
    print(ERROR, "failed to load model");
    return 1;
  }

  print(INFO, "llama_model_n_params:    %d", -llama_model_n_params(spectre.model));

  // --------------------------------------------------
  // Tokenizer
  // --------------------------------------------------
  const struct llama_vocab *vocab = llama_model_get_vocab(spectre.model);

  const int n_prompt = -llama_tokenize(vocab, spectre.config.prompt.c_str(), spectre.config.prompt.size(), NULL, 0, true, true);

  std::vector<llama_token> tokens(n_prompt);

  if (llama_tokenize(vocab, spectre.config.prompt.c_str(), spectre.config.prompt.size(), tokens.data(), tokens.size(), true, true) < 0)
  {
    print(ERROR, "failed to tokenize prompt");
    return 1;
  }

  print(INFO, "\"%s\" (%d tokens)", spectre.config.prompt.c_str(), n_prompt);

  for (auto id : tokens)
  {
    char token[128];
    int n = llama_token_to_piece(vocab, id, token, sizeof(token), 0, true);
    if (n < 0)
    {
      print(ERROR, "failed to convert token to piece");
      return 1;
    }
    print(INFO, "|%.*s|", n, token);
  }

  print(INFO, "llama_vocab_n_tokens:    %d", llama_vocab_n_tokens(vocab));
  print(INFO, "llama_vocab_type:        %d", llama_vocab_type(vocab));

  // --------------------------------------------------
  // Context
  // --------------------------------------------------
  struct llama_context_params ctx_params = llama_context_default_params();

  ctx_params.no_perf = false;
  ctx_params.n_ctx = spectre.config.ctx;
  ctx_params.n_batch = n_prompt;

  if (!(spectre.context = llama_init_from_model(spectre.model, ctx_params)))
  {
    print(ERROR, "failed to create the llama_context");
    return 1;
  }

  print(INFO, "llama_n_ctx:        %d", llama_n_ctx(spectre.context));
  print(INFO, "llama_n_ctx_seq:    %d", llama_n_ctx_seq(spectre.context));
  print(INFO, "llama_n_batch:      %d", llama_n_batch(spectre.context));
  print(INFO, "llama_n_ubatch:     %d", llama_n_ubatch(spectre.context));
  print(INFO, "llama_n_seq_max:    %d", llama_n_seq_max(spectre.context));

  // --------------------------------------------------
  // Sampler
  // --------------------------------------------------
  struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();

  sparams.no_perf = false;

  if (!(spectre.sampler = llama_sampler_chain_init(sparams)))
  {
    print(ERROR, "failed to create the llama_sampler_chain_params");
    return 1;
  }

  llama_sampler_chain_add(spectre.sampler, llama_sampler_init_top_k(spectre.config.k));
  llama_sampler_chain_add(spectre.sampler, llama_sampler_init_top_p(spectre.config.p, 1));
  llama_sampler_chain_add(spectre.sampler, llama_sampler_init_temp(spectre.config.temp));
  llama_sampler_chain_add(spectre.sampler, llama_sampler_init_dist(std::time(nullptr)));

  // prepare first batch
  llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

  if (llama_model_has_encoder(spectre.model))
  {
    if (llama_encode(spectre.context, batch))
    {
      print(ERROR, "failed to eval");
      return 1;
    }

    llama_token decoder_start_token_id = llama_model_decoder_start_token(spectre.model);
    if (decoder_start_token_id == LLAMA_TOKEN_NULL)
    {
      decoder_start_token_id = llama_vocab_bos(vocab);
    }

    batch = llama_batch_get_one(&decoder_start_token_id, 1);
  }

  llama_token cursor;
  std::size_t decoded_tokens = 0;
  const int64_t start = ggml_time_us();

  print(INFO, "llama_model_chat_template:    \n%s", llama_model_chat_template(spectre.model, NULL));

  std::ofstream file("metrics.csv");
  if (!file)
  {
    print(ERROR, "failed to open file");
    return 1;
  }

  file << "step,logit,prob,logprob\n";

  // --------------------------------------------------
  // Main loop
  // --------------------------------------------------
  for (;;)
  {
    // evaluate the current batch with the transformer model
    if (llama_decode(spectre.context, batch))
    {
      print(ERROR, "failed to eval, return code %d", 1);
      return 1;
    }

    const float *logits = llama_get_logits_ith(spectre.context, -1);
    cursor = llama_sampler_sample(spectre.sampler, spectre.context, -1);

    const int n_vocab = llama_vocab_n_tokens(vocab);

    // numerically stable softmax
    double max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i)
    {
      if (logits[i] > max_logit) max_logit = logits[i];
    }

    double denom = 0.0;
    for (int i = 0; i < n_vocab; ++i)
    {
      denom += std::exp((double)logits[i] - max_logit);
    }

    const double logit = logits[cursor];
    const double prob = std::exp((double)logit - max_logit) / denom;
    const double logprob = std::log(prob);

    file << decoded_tokens << ','
         << logit << ','
         << prob << ','
         << logprob << '\n';

    // is it an end of generation?
    if (llama_vocab_is_eog(vocab, cursor))
    {
      break;
    }

    char token[128];
    int n = llama_token_to_piece(vocab, cursor, token, sizeof(token), 0, true);
    if (n < 0)
    {
      print(ERROR, "failed to convert token to piece");
      return 1;
    }
    printf("%.*s", n, token);
    fflush(stdout);

    // prepare the next batch with the sampled token
    batch = llama_batch_get_one(&cursor, 1);

    decoded_tokens += 1;
  }

  const int64_t end = ggml_time_us();

  printf("\n");

  print(INFO, "decoded %zu tokens in %.2f s, speed: %.2f t/s",
        decoded_tokens, (end - start) / 1000000.0f,
        decoded_tokens / ((end - start) / 1000000.0f));

  llama_perf_sampler_print(spectre.sampler);
  llama_perf_context_print(spectre.context);

  return 0;
}
