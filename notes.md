# Intro

Speculative decoding provides speedups when the draft model is fast and accurate enough that most speculative tokens are accepted. On GPU this works well because the cost of evaluating multiple tokens in parallel is low.

On CPU, draft evaluation is expensive and small differences between model logits cause many rejections.

Speculative sampling was invented for GPUs, where:
* draft model runs on tensor cores
* target model runs on tensor cores
* parallel kernels overlap efficiently
* memory bandwidth is massive

Speculative sampling only helps if:
* Draft model is much faster per token than target
* Hardware can evaluate both models in parallel efficiently
* Draft acceptance rate is high enough (ideally >60%)

Speculative sampling is GPU-optimized
This research addresses the inference scheduling problem, drawing from compiler optimization theory (e.g., branch prediction, profile-guided optimization, instruction scheduling) to minimize both latency and energy usage. These parallels arise from the fact that LLM inference can be viewed as a computational pipeline, similar to how compilers treat code as a sequence or graph to optimize (see [https://en.wikipedia.org/wiki/Abstract_syntax_tree](Abstract Syntax Tree)). Previous research shows speedups of 2-3x from speculative decoding alone, and layering compiler-inspired optimizations could yield even greater improvements, especially for edge cases like long sequences or low-acceptance-rate drafts.

## The idea:
1. The small model proposes several next tokens (a "draft" sequence).
2. The large model verifies them in parallel.
3. If the proposals are likely under the large model, they are accepted, saving time because the large model doesn’t need to compute them token-by-token

## Speculative decoding functions as follows:
1. The small LLM generates a series of sequential tokens, after which the large LLM evaluates all of these in a single pass.
2. For each token, if the probability in the large LLM is higher than that of the small one, it is accepted directly (therefore not affecting the large LLM's statistics). If the probability is lower, the likelihood of acceptance is proportional to the difference in probabilities. This method makes it likely that the token will not be accepted, in which case the computation is wasted. If the small model performs well, we gain a speedup without changing the output, but if it performs poorly, we waste significant compute resources, slowing down the process.    See: [https://en.wikipedia.org/wiki/Branch_predictor](Branch predictor)
3. Speculative decoding variants allow tokens to exit the model early if the model is confident, similar to compiler optimizations like function inlining or dead code elimination, which skip unnecessary computations.
4. Speculative sampling often builds a tree of possible token paths during text generation. We can apply compiler-style graph transformations (e.g., pruning redundant branches via static analysis or common subexpression elimination on token probabilities) to optimize the tree exploration.
5. Just as compilers schedule operations for parallel execution, speculative sampling could "schedule" token generation to prioritize high-probability paths, inspired by compiler techniques for out-of-order execution.

These are softmax probabilities over the vocabulary, the same kind of probabilities used during standard next-token sampling in an LLM.

# [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
> [!TIP]
> auto-regressive sampling vs speculative sampling

## The problem
Transformer decoding remains a highly costly and inefficient process in this regime. Since each new token depends on the past, many such transformer calls are required to sample a new sequence. Whilst transformers can be trained efficiently and in parallel on TPUs and GPUs, samples are typically drawn auto-regressively. For most applications, auto-regressive sampling (ArS) is highly memory bandwidth bound and thus cannot make effective use of modern accelerator hardware (Shazeer, 2019). A memory bound model call only generates a single token for every sequence in the batch, hence generating multiple tokens introduces a large amount of latency in any system which makes use of it.

## Solution
Speculative Sampling (SpS): an algorithm for accelerating transformer decoding by enabling the generation of multiple tokens from each transformer call.

## Algorithm
1. Generate a short draft of length K (draft model)
2. Score the draft using the lager model (i.e. the model from we wish to sample from - target model)
3. Accept a subset of K from left to right (via a rejection sampling scheme) recovering the distribution of the target model in the process.

## Notes
1. The next token might sometimes be “obvious” therefore if there is strong agreement between the draft and target model’s distributions on a given token or sub-sequence of tokens, this setup permits the generation of multiple tokens each time the target model is called.
2. The latency of parallel scoring of short continuations, generated by a faster but less powerful draft model, is comparable to that of sampling a single token from the larger target mode

## Choice of Draft Models
1. Incorporating draft generation into the target model and train the model from the start.
2. Using sequence level distillation to generate a second model which predicts K tokens in parallel.
3. Set a portion of the activations of the target model as an input to the draft model, and train the draft model with this input.

## Conclusion
We show that the expected acceptance rate of draft tokens is sufficient to offset the overhead of the drafting process for large language models (LLMs), resulting in an effective and practical method for reducing sampling latency without the need for modifying the target model or biasing the sample distribution.

# [A Hitchhiker’s Guide to Speculative Decoding](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)
> [!TIP]
> By Team PyTorch. May 2, 2024

## Key takeaways
1. Speculative decoding : an optimization technique for inference that makes educated guesses about future tokens while generating the current token.
2. All within a single forward pass. No [backpatching](https://www.geeksforgeeks.org/compiler-design/backpatching-in-compiler-design/)
3. Based on the premise that the model is powerful enough to predict multiple tokens in a single forward pass.
4. It incorporates a verification mechanism to ensure the correctness of these speculated tokens
5. Thereby guaranteeing that the overall output of speculative decoding is identical to that of vanilla decoding.

# [Speculative Sampling Explained](https://saibo-creator.github.io/post/2024_03_08_speculative_sampling)
> [!TIP]
> Use a draft sampling to achieve the same sampling result as the target sampling

# [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
> [!TIP]
> Inference from large autoregressive models like Transformers is slow - decoding K tokens takes K serial runs of the model

## Key takeaways
1. Accelerate existing off-the-shelf models without retraining or architecture changes
2. Sample from autoregressive models faster by computing several tokens in parallel. Why this works:
    - hard language-modelling tasks often include easier subtasks that can be approximated well by more efficient models
    - using speculative execution and a novel sampling method, we can run large models in parallel on the outputs of the approximation models and thus generate several tokens concurrently without changing the distribution
3. A single decode step from large autoregressive models (notably transformers) is significantly slower than a step from their smaller counterpart
4. Several approaches were developed to make inference from them faster.
    - Reduce the inference cost for *all* inputs equally
    - Adaptive Computation Method --> not all inference steps are equal, use the large models where it makes sense
5. Some inference steps are "harder" and some are "easier"
6. Inference from large models is often not bottlenecked on arithmetic operations, but rather on memory bandwidth and communication --> thus additional computation resources might be available

# Resources
 1. [Must-read papers and blogs on Speculative Decoding](https://github.com/hemingkx/SpeculativeDecodingPapers)
 2. [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
 3. [Accelerating LLM Inference with Staged Speculative Decoding](https://arxiv.org/abs/2308.04623)
 4. [Looking Back at Speculative Decoding](https://news.ycombinator.com/item?id=43216518)
 5. [Instantaneous Grammatical Error Correction with Shallow Aggressive Decoding](https://arxiv.org/abs/2106.04970)
 6. [A Hitchhiker’s Guide to Speculative Decoding](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding)
 7. [Looking back at speculative decoding](https://research.google/blog/looking-back-at-speculative-decoding)
 8. [Learning Harmonized Representations for Speculative Sampling](https://arxiv.org/abs/2408.15766)
 9. [Llama.cpp speculative sampling: 2x faster inference for large models](https://news.ycombinator.com/item?id=37390024)
10. [Speculative: PoC for speeding-up inference via speculative sampling by ggerganov](https://news.ycombinator.com/item?id=37357783)
11. [Speculative Sampling Explained](https://saibo-creator.github.io/post/2024_03_08_speculative_sampling)
12. [llama : add example for speculative sampling #2030](https://github.com/ggml-org/llama.cpp/issues/2030)
13. [speculative : PoC for speeding-up inference via speculative sampling #2926](https://github.com/ggml-org/llama.cpp/pull/2926)
14. [Enable speculative decoding #5800](https://github.com/ollama/ollama/issues/5800)
15. [Understanding LLM System with 3-layer Abstraction](https://ralphmao.github.io/ML-software-system)
16. [Speculative Decoding for 2x Faster Whisper Inference](https://huggingface.co/blog/whisper-speculative-decoding)
17. [speculative : refactor and add a simpler example #10362](https://github.com/ggml-org/llama.cpp/pull/10362)
18. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
19. [speculative_decoding.ipynb](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/speculative_decoding.ipynb#scrollTo=af0b3757-72dc-48a8-9d9d-fc135386cae5)
20. [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)

# Footnote
1. `logit(p) = log(p/(1-p))` is the raw, denormalized predictions generated by a model before applying any activation function
