#define _GNU_SOURCE
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "llama.h"

#define NOB_IMPLEMENTATION
#include "nob.h"

#ifndef PROMPT
#define PROMPT "how old is the earth?"
#endif // PROMPT

typedef struct arena_t
{
  uint8_t *data;
  size_t count;
  size_t capacity;
} arena_t;

static inline void *arena_alloc(arena_t *arena, size_t size)
{
  assert(arena);
  assert(size > 0);
  assert(arena->data);

  size_t space_left = arena->count - arena->capacity;
  assert(size < space_left);

  void *pointer = arena->data + arena->capacity;
  arena->capacity += size;

  return pointer;
}

static inline void arena_init(arena_t *arena, size_t size)
{
  assert(arena);
  assert(size > 0);

  arena->data = malloc(size);
  assert(arena->data);

  arena->count = size;
  arena->capacity = 0;
}

static inline void arena_deinit(arena_t *arena)
{
  assert(arena);
  assert(arena->data);

  free(arena->data);

  arena->data = NULL;
  arena->count = 0;
  arena->capacity = 0;
}

int main(int argc, char **argv)
{
  bool success;
  arena_t arena;

  struct llama_sampler *smp = NULL;
  struct llama_model *model = NULL;
  struct llama_context *ctx = NULL;

  char *name = shift_args(&argc, &argv);

  if (argc != 1)
  {
    nob_log(NOB_ERROR, "Usage: %s <gguf model>", name);
    nob_log(NOB_ERROR, argc < 1 ? "No input file is provided" : "Too many arguments");
    return 1;
  }

  arena_init(&arena, 8 * 1024 * 1024 * sizeof(uint8_t));

  llama_backend_init();

  ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);

  struct llama_model_params params = llama_model_default_params();

  params.use_mmap = true;
  params.use_mlock = true;
  params.devices = &device;

  if (!(model = llama_model_load_from_file(*argv, params)))
  {
    nob_log(NOB_ERROR, "failed to load model");
    success = false;
    goto cleanup;
  }

  printf("\n");

  // --------------------------------------------------
  // Tokenizer
  // --------------------------------------------------
  const struct llama_vocab *vocab = llama_model_get_vocab(model);

  const int n_prompt = -llama_tokenize(vocab, PROMPT, strlen(PROMPT), NULL, 0, true, true);

  llama_token *tokens = (llama_token *)arena_alloc(&arena, n_prompt * sizeof(llama_token));
  if (!tokens)
  {
    nob_log(NOB_ERROR, "failed to allocate memory");
    success = false;
    goto cleanup;
  }

  if (llama_tokenize(vocab, PROMPT, strlen(PROMPT), tokens, n_prompt, true, true) < n_prompt)
  {
    nob_log(NOB_ERROR, "failed to tokenize prompt");
    success = false;
    goto cleanup;
  }

  nob_log(NOB_INFO, "user prompt was \"%s\" => number of tokens is %d", PROMPT, n_prompt);

  for (int i = 0; i < n_prompt; ++i)
  {
    int id = tokens[i];
    char buf[128] = {0};
    llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    nob_log(NOB_INFO, "\t|%d|\t-\t|%s|", id, buf);
  }

  printf("\n");

  // --------------------------------------------------
  // Context
  // --------------------------------------------------
  struct llama_context_params ctx_params = llama_context_default_params();

  // TODO play around with these
  ctx_params.n_ctx = 0;
  ctx_params.no_perf = false;
  ctx_params.n_batch = n_prompt;

  if (!(ctx = llama_init_from_model(model, ctx_params)))
  {
    nob_log(NOB_ERROR, "failed to create the llama_context");
    success = false;
    goto cleanup;
  }

  // --------------------------------------------------
  // Sampler
  // --------------------------------------------------
  struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
  sparams.no_perf = false;

  if (!(smp = llama_sampler_chain_init(sparams)))
  {
    nob_log(NOB_ERROR, "failed to create the llama_sampler_chain_params");
    success = false;
    goto cleanup;
  }

  // llama_sampler_chain_add(smp, llama_sampler_init_greedy());

  // TODO this throws this assertion:
  // llama.cpp/src/llama-sampler.cpp:866: GGML_ASSERT(cur_p.selected >= 0 && cur_p.selected < (int32_t) cur_p.size) failed
  // llama_sampler_chain_add(smp, llama_sampler_init_top_k(40));

  llama_sampler_chain_add(smp, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

  // prepare batch
  llama_batch batch = llama_batch_get_one(tokens, n_prompt);

  if (llama_model_has_encoder(model))
  {
    if (llama_encode(ctx, batch))
    {
      nob_log(NOB_ERROR, "failed to eval");
      success = false;
      goto cleanup;
    }

    llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
    if (decoder_start_token_id == LLAMA_TOKEN_NULL)
    {
      decoder_start_token_id = llama_vocab_bos(vocab);
    }

    batch = llama_batch_get_one(&decoder_start_token_id, 1);
  }

  const int64_t t_main_start = ggml_time_us();
  int n_decode = 0;
  llama_token new_token_id;

  // --------------------------------------------------
  // Main loop
  // --------------------------------------------------
  for (;;)
  {
    // evaluate the current batch with the transformer model
    if (llama_decode(ctx, batch))
    {
      nob_log(NOB_ERROR, "failed to eval, return code %d", 1);
      return 1;
    }

    new_token_id = llama_sampler_sample(smp, ctx, -1);

    // is it an end of generation?
    if (llama_vocab_is_eog(vocab, new_token_id))
    {
      break;
    }

    char buf[128] = {0};
    int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
    if (n < 0)
    {
      nob_log(NOB_ERROR, "failed to convert token to piece");
      return 1;
    }
    printf("%.*s", n, buf);
    fflush(stdout);

    // prepare the next batch with the sampled token
    batch = llama_batch_get_one(&new_token_id, 1);

    n_decode += 1;
  }

  const int64_t t_main_end = ggml_time_us();

  printf("\n");

  nob_log(NOB_INFO,
          "decoded %d tokens in %.2f s, speed: %.2f t/s",
          n_decode,
          (t_main_end - t_main_start) / 1000000.0f,
          n_decode / ((t_main_end - t_main_start) / 1000000.0f));

  printf("\n");

  llama_perf_sampler_print(smp);
  llama_perf_context_print(ctx);

  printf("\n");

  success = true;

cleanup:
  arena_deinit(&arena);
  llama_free(ctx);
  llama_sampler_free(smp);
  llama_model_free(model);
  llama_backend_free();

  return success ? 0 : 1;
}
