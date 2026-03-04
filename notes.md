# Speculative Sampling

<!--toc:start-->
- [Speculative Sampling](#speculative-sampling)
  - [Intro](#intro)
    - [How Speculative Sampling works in general](#how-speculative-sampling-works-in-general)
  - [Concrete Directions](#concrete-directions)
    - [Adaptive Draft Length for llama.cpp](#adaptive-draft-length-for-llamacpp)
    - [Tree-Structured Drafting in llama.cpp](#tree-structured-drafting-in-llamacpp)
    - [Heterogeneous CPU/GPU Scheduling for Edge](#heterogeneous-cpugpu-scheduling-for-edge)
    - [EAGLE-3 Support in llama.cpp](#eagle-3-support-in-llamacpp)
    - [Energy-Aware Speculative Decoding](#energy-aware-speculative-decoding)
    - [Relative Papers](#relative-papers)
  - [Accelerating Large Language Model Decoding with Speculative Sampling](#accelerating-large-language-model-decoding-with-speculative-sampling)
    - [The problem](#the-problem)
    - [Solution](#solution)
    - [Algorithm](#algorithm)
    - [Notes](#notes)
    - [Choice of Draft Models](#choice-of-draft-models)
    - [Conclusion](#conclusion)
  - [A Hitchhiker's Guide to Speculative Decoding](#a-hitchhikers-guide-to-speculative-decoding)
    - [Key takeaways](#key-takeaways)
  - [Speculative Sampling Explained](#speculative-sampling-explained)
  - [Fast Inference from Transformers via Speculative Decoding](#fast-inference-from-transformers-via-speculative-decoding)
    - [Key takeaways](#key-takeaways-1)
  - [Reward-Guided Speculative Decoding for Efficient LLM Reasoning](#reward-guided-speculative-decoding-for-efficient-llm-reasoning)
    - [Key takeaways](#key-takeaways-2)
    - [Reward-Guided Speculative Decoding (RSD)](#reward-guided-speculative-decoding-rsd)
  - [Speculative Speculative Decoding (SSD)](#speculative-speculative-decoding-ssd)
    - [Key takeaways](#key-takeaways-3)
  - [Resources](#resources)
- [Footnote](#footnote)
- [vim: nonu cole=3](#vim-nonu-cole3)
<!--toc:end-->

> This research addresses the inference scheduling problem, drawing from compiler optimization theory (e.g., branch prediction, profile-guided optimization, instruction scheduling) to minimize both latency and energy usage. These parallels arise from the fact that LLM inference can be viewed as a computational pipeline, similar to how compilers treat code as a sequence or graph to optimize ([Abstract Syntax Tree](https://en.wikipedia.org/wiki/abstract_syntax_tree)). Previous research shows speedups of 2-3x from speculative decoding alone, and layering compiler-inspired optimizations could yield even greater improvements, especially for edge cases like long sequences or low-acceptance-rate drafts.

## Intro
Speculative decoding provides speedups when the draft model is fast and accurate enough that most speculative tokens are accepted. On GPU this works well because the cost of evaluating multiple tokens in parallel is low.

On CPU, draft evaluation is expensive and small differences between model logits cause many rejections.

### How Speculative Sampling works in general
*from: [Reward-Guided Speculative Decoding for Efficient LLM Reasoning](#resources-reward-guided-speculative-decoding-for-efficient-llm-reasoning)*
> The smaller model serves as a guide, proposing the overall sequences that the larger model can confirm or adjust, leading to faster inference without compromising quality.

Speculative sampling is GPU-optimized, where:
* draft model runs on tensor cores
* target model runs on tensor cores
* parallel kernels overlap efficiently
* memory bandwidth is massive

Speculative sampling only helps if:
* Draft model is much faster per token than target
* Hardware can evaluate both models in parallel efficiently
* Draft acceptance rate is high enough (ideally >60%)

In simple terms:
1. The small model proposes several next tokens (a "draft" sequence).
2. The large model verifies them in parallel.
3. If the proposals are likely under the large model, they are accepted, saving time because the large model doesn't need to compute them token-by-token

In other words:
1. The small LLM generates a series of sequential tokens, after which the large LLM evaluates all of these in a single pass.
2. For each token, if the probability in the large LLM is higher than that of the small one, it is accepted directly (therefore not affecting the large LLM's statistics). If the probability is lower, the likelihood of acceptance is proportional to the difference in probabilities. This method makes it likely that the token will not be accepted, in which case the computation is wasted. If the small model performs well, we gain a speedup without changing the output, but if it performs poorly, we waste significant compute resources, slowing down the process. See: [Branch predictor](https://en.wikipedia.org/wiki/Branch_predictor)
3. Speculative decoding variants allow tokens to exit the model early if the model is confident, similar to compiler optimizations like function inlining or dead code elimination, which skip unnecessary computations.
4. Speculative sampling often builds a tree of possible token paths during text generation. We can apply compiler-style graph transformations (e.g., pruning redundant branches via static analysis or common sub expression elimination on token probabilities) to optimize the tree exploration.
5. Just as compilers schedule operations for parallel execution, speculative sampling could "schedule" token generation to prioritize high-probability paths, inspired by compiler techniques for out-of-order execution.

These are soft-max probabilities over the vocabulary, the same kind of probabilities used during standard next-token sampling in an LLM.

## Concrete Directions

### Adaptive Draft Length for llama.cpp

llama.cpp uses **fixed** `--draft-max` and `--draft-min` parameters. This is a known weakness. The optimal draft length depends on context difficulty -- easy tokens (boilerplate, code patterns) should draft long, hard tokens (reasoning, rare words) should draft short or not at all.

* Implement an entropy-based or lightweight prediction-head approach (inspired by [SpecDec++](https://arxiv.org/abs/2405.19715) or [AdaEDL](https://proceedings.mlr.press/v262/agrawal24a.html)) directly in llama.cpp's speculative pipeline. AdaEDL is particularly attractive because it's **training-free** -- it uses an entropy-based lower bound on acceptance probability to decide when to stop drafting. This means you don't need to train anything, just instrument the existing draft loop.

* 10-57% improvement over static draft lengths according to the AdaEDL paper.

* This is **profile-guided optimization** -- you're using runtime statistics (token entropy) to make scheduling decisions, exactly like a PGO-enabled compiler uses branch frequency data.

### Tree-Structured Drafting in llama.cpp

llama.cpp currently does **linear** speculative drafting -- one sequence of draft tokens. Research shows that **tree-structured** drafting (where you branch at uncertain positions) dramatically improves acceptance rates. [TALON](https://arxiv.org/abs/2601.07353) achieves up to 5.16x speedup by adaptively constructing deep-and-narrow trees for deterministic contexts and shallow-and-wide trees for uncertain ones.

* Implement a basic tree-structured draft in llama.cpp with adaptive branching. The key challenge is implementing **tree attention** for verification efficiently in GGML (llama.cpp's tensor library). This is non-trivial but it's exactly the kind of systems work that makes a strong ECE thesis.

* This is **speculative execution with branch fan-out** -- instead of predicting one path, you execute multiple paths and discard the wrong ones, just like a superscalar processor.

### Heterogeneous CPU/GPU Scheduling for Edge

There's a very recent paper (February 2026) -- [Compiler-Assisted Speculative Sampling for Accelerated LLM Inference on Heterogeneous Edge Devices](https://arxiv.org/abs/2602.08060) -- that does exactly what the thesis introduction describes: using compiler-style cost models to partition draft/target model execution across CPU and GPU on edge SoCs. They got 1.68x speedup on ARM Cortex-A + Mali GPU.

* Build an analytical cost model for llama.cpp that decides at runtime whether to use speculative decoding at all, and if so, how to partition work across available compute (CPU threads, GPU if available, even NPU on newer hardware). llama.cpp already supports hybrid CPU/GPU execution via layer offloading -- we'd extend this with speculation-aware scheduling.

* This is underexplored. Most speculative decoding research assumes beefy GPUs. The edge/CPU story is wide open.

### EAGLE-3 Support in llama.cpp

llama.cpp **does not support EAGLE models** ([issue #15305](https://github.com/ggml-org/llama.cpp/issues/15305)). vLLM, TRT-LLM, and SGLang all do. EAGLE-3 delivers 2-2.5x speedup over autoregressive decoding. This is the single most requested speculative decoding feature in the llama.cpp community.

* Implement EAGLE-3 model loading and inference in llama.cpp. This requires understanding the EAGLE architecture (a lightweight draft head that reuses the target model's hidden states as features) and integrating it into GGML's graph execution.

* This would likely be merged upstream and would be your most visible open-source contribution. It's also a natural comparison point for your thesis experiments.

### Energy-Aware Speculative Decoding

Your abstract mentions energy consumption, which almost nobody benchmarks rigorously. Most papers only report tokens/second or latency.

* Instrument llama.cpp to measure actual energy consumption (via RAPL on Intel, or power sensors on ARM) under different speculative decoding configurations. Then build a scheduler that optimizes for an **energy-latency Pareto frontier** rather than pure speed. Sometimes the fastest configuration wastes energy on rejected drafts -- a smarter policy might accept slightly higher latency for significantly lower energy.

* Edge deployment, battery-powered devices, sustainability arguments. This is a differentiator that no other speculative decoding paper does well.

### Relative Papers

- [SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths](https://arxiv.org/abs/2405.19715)
- [TALON: Confidence-Aware Speculative Decoding with Adaptive Token Trees](https://arxiv.org/abs/2601.07353)
- [Compiler-Assisted Speculative Sampling on Heterogeneous Edge Devices](https://arxiv.org/abs/2602.08060)
- [Efficient Speculative Decoding for Llama at Scale](https://arxiv.org/abs/2508.08192)
- [Speculative Speculative Decoding (Saguaro)](https://arxiv.org/abs/2603.03251)

## Accelerating Large Language Model Decoding with Speculative Sampling
> [!TIP]
> auto-regressive sampling vs speculative sampling
> [Source](#resources-accelerating-large-language-model-decoding-with-speculative-sampling) 

### The problem
Transformer decoding remains a highly costly and inefficient process in this regime. Since each new token depends on the past, many such transformer calls are required to sample a new sequence. Whilst transformers can be trained efficiently and in parallel on TPUs and GPUs, samples are typically drawn auto-regressively. For most applications, auto-regressive sampling (ArS) is highly memory bandwidth bound and thus cannot make effective use of modern accelerator hardware (Shazeer, 2019). A memory bound model call only generates a single token for every sequence in the batch, hence generating multiple tokens introduces a large amount of latency in any system which makes use of it.

### Solution
Speculative Sampling (SpS): an algorithm for accelerating transformer decoding by enabling the generation of multiple tokens from each transformer call.

### Algorithm
1. Generate a short draft of length K (draft model)
2. Score the draft using the lager model (i.e. the model from we wish to sample from - target model)
3. Accept a subset of K from left to right (via a rejection sampling scheme) recovering the distribution of the target model in the process.

### Notes
1. The next token might sometimes be "obvious" therefore if there is strong agreement between the draft and target model's distributions on a given token or sub-sequence of tokens, this setup permits the generation of multiple tokens each time the target model is called.
2. The latency of parallel scoring of short continuations, generated by a faster but less powerful draft model, is comparable to that of sampling a single token from the larger target mode

### Choice of Draft Models
1. Incorporating draft generation into the target model and train the model from the start.
2. Using sequence level distillation to generate a second model which predicts K tokens in parallel.
3. Set a portion of the activations of the target model as an input to the draft model, and train the draft model with this input.

### Conclusion
We show that the expected acceptance rate of draft tokens is sufficient to offset the overhead of the drafting process for large language models (LLMs), resulting in an effective and practical method for reducing sampling latency without the need for modifying the target model or biasing the sample distribution.

## A Hitchhiker's Guide to Speculative Decoding
> [!TIP]
> By Team PyTorch. May 2, 2024
> [Source](#resources-a-hitchhikers-guide-to-speculative-decoding)

### Key takeaways
1. Speculative decoding : an optimization technique for inference that makes educated guesses about future tokens while generating the current token.
2. All within a single forward pass. No [backpatching](https://www.geeksforgeeks.org/compiler-design/backpatching-in-compiler-design/)
3. Based on the premise that the model is powerful enough to predict multiple tokens in a single forward pass.
4. It incorporates a verification mechanism to ensure the correctness of these speculated tokens
5. Thereby guaranteeing that the overall output of speculative decoding is identical to that of vanilla decoding.

## Speculative Sampling Explained
> [!TIP]
> Use a draft sampling to achieve the same sampling result as the target sampling
> [Source](#resources-speculative-sampling-explained)

## Fast Inference from Transformers via Speculative Decoding
> [!TIP]
> Inference from large autoregressive models like Transformers is slow - decoding K tokens takes K serial runs of the model
> [Source](#resources-fast-inference-from-transformers-via-speculative-decoding)

### Key takeaways
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

## Reward-Guided Speculative Decoding for Efficient LLM Reasoning
> [!TIP]
> Unlike normal speculative sampling we incorporate controlled bias to prioritize high-reward outputs.
> [Source](#resources-reward-guided-speculative-decoding-for-efficient-llm-reasoning)

### Key takeaways
1. Evaluate intermediate decoding steps and dynamically decide (threshold based) to invoke the target model.
2. Unbiasedness maintains theoretical fidelity but often reduces efficiency (especially when the draft diverges from the target)
3. Allowing controlled bias (where the final distribution deviates slightly from the large model) can improve performance
4. If a draft token is correct but does not match the large model's distribution exactly, strict rejection is counterproductive.
5. Reward-guided acceptance:
    * retains valuable partial solutions,
    * reduces unnecessary queries,
    * can even surpass the large model's performance

### Reward-Guided Speculative Decoding (RSD)
- Balances efficiency and accuracy by integrating computationally lightweight "draft" evaluations with reward-driven refinements from a more capable "target" model
- Adaptively select high-value draft outputs rather than discarding mismatched tokens outright

## Speculative Speculative Decoding (SSD)
> [!TIP]
> Speculative Decoding relies on a sequential dependence between speculation and verification, so we parallelize these operations.
> [Source](#resources-speculative-speculative-decoding)

### Key takeaways
1. While a verification is ongoing, the draft model *predicts* likely verification outcomes and prepares speculations pre-emptively for them.
2. If the actual verification outcome is then in the predicted set, a speculation can be returned immediately, eliminating drafting overhead entirely.
3. The result is SAGUARO, an optimized SSD algorithm.

## Resources
1. <span id="resources-speculative-sampling"></span> [Speculative Sampling](https://github.com/hemingkx/SpeculativeDecodingPapers)
2. <span id="resources-llama-cpp-docs-speculative-md"></span> [llama.cpp: docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
3. <span id="resources-must-read-papers-and-blogs-on-speculative-decoding"></span> [Must-read papers and blogs on Speculative Decoding](https://github.com/hemingkx/SpeculativeDecodingPapers)
4. <span id="resources-accelerating-large-language-model-decoding-with-speculative-sampling"></span> [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
5. <span id="resources-accelerating-llm-inference-with-staged-speculative-decoding"></span> [Accelerating LLM Inference with Staged Speculative Decoding](https://arxiv.org/abs/2308.04623)
6. <span id="resources-looking-back-at-speculative-decoding-hacker-news"></span> [Looking Back at Speculative Decoding HN](https://news.ycombinator.com/item?id=43216518)
7. <span id="resources-instantaneous-grammatical-error-correction-with-shallow-aggressive-decoding"></span> [Instantaneous Grammatical Error Correction with Shallow Aggressive Decoding](https://arxiv.org/abs/2106.04970)
8. <span id="resources-a-hitchhikers-guide-to-speculative-decoding"></span> [A Hitchhiker's Guide to Speculative Decoding](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding)
9. <span id="resources-looking-back-at-speculative-decoding"></span> [Looking back at speculative decoding](https://research.google/blog/looking-back-at-speculative-decoding)
10. <span id="resources-learning-harmonized-representations-for-speculative-sampling"></span> [Learning Harmonized Representations for Speculative Sampling](https://arxiv.org/abs/2408.15766)
20. <span id="resources-llama-cpp-speculative-sampling-2x-faster-inference-for-large-models"></span> [Llama.cpp speculative sampling: 2x faster inference for large models](https://news.ycombinator.com/item?id=37390024)
12. <span id="resources-speculative-poc-for-speeding-up-inference-via-speculative-sampling-by-ggerganov"></span> [Speculative: PoC for speeding-up inference via speculative sampling by ggerganov](https://news.ycombinator.com/item?id=37357783)
13. <span id="resources-speculative-sampling-explained"></span> [Speculative Sampling Explained](https://saibo-creator.github.io/post/2024_03_08_speculative_sampling)
14. <span id="resources-llama-add-example-for-speculative-sampling-2030"></span> [llama : add example for speculative sampling #2030](https://github.com/ggml-org/llama.cpp/issues/2030)
15. <span id="resources-speculative-poc-for-speeding-up-inference-via-speculative-sampling-#2926"></span> [speculative : PoC for speeding-up inference via speculative sampling #2926](https://github.com/ggml-org/llama.cpp/pull/2926)
16. <span id="resources-enable-speculative-decoding-5800"></span> [Enable speculative decoding #5800](https://github.com/ollama/ollama/issues/5800)
17. <span id="resources-understanding-llm-system-with-3-layer-abstraction"></span> [Understanding LLM System with 3-layer Abstraction](https://ralphmao.github.io/ML-software-system)
18. <span id="resources-speculative-decoding-for-2x-faster-whisper-inference"></span> [Speculative Decoding for 2x Faster Whisper Inference](https://huggingface.co/blog/whisper-speculative-decoding)
19. <span id="resources-speculative-refactor-and-add-a-simpler-example-10362"></span> [speculative : refactor and add a simpler example #10362](https://github.com/ggml-org/llama.cpp/pull/10362)
20. <span id="resources-fast-inference-from-transformers-via-speculative-decoding"></span> [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
21. <span id="resources-speculative_decoding-ipynb"></span> [speculative_decoding.ipynb](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/speculative_decoding.ipynb#scrollTo=af0b3757-72dc-48a8-9d9d-fc135386cae5)
22. <span id="resources-medusa:-simple-llm-inference-acceleration-framework-with-multiple-decoding-heads"></span> [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
23. <span id="resources-reward-guided-speculative-decoding-for-efficient-llm-reasoning"></span> [Reward-Guided Speculative Decoding for Efficient LLM Reasoning](https://arxiv.org/abs/2501.19324)
24. <span id="resources-speculative-speculative-decoding"></span> [Speculative Speculative Decoding](https://arxiv.org/pdf/2603.03251)
25. <span id="talon-confidence-aware-speculative-decoding-with-adaptive-token-trees"></span>[TALON: Confidence-Aware Speculative Decoding with Adaptive Token Trees](https://arxiv.org/abs/2601.07353)

# Footnote
1. `logit(p) = log(p/(1-p))` is the raw, denormalized predictions generated by a model before applying any activation function

# vim: nonu cole=3
