## Table of Contents

<!--toc:start-->
- [Table of Contents](#table-of-contents)
- [Intro](#intro)
  - [Wikipedia entry for Speculative Decoding](#wikipedia-entry-for-speculative-decoding)
  - [How Speculative Sampling works in general](#how-speculative-sampling-works-in-general)
- [Model naming conventions](#model-naming-conventions)
    - [Reading a filename](#reading-a-filename)
    - [Parameter count](#parameter-count)
    - [Fine-tuning suffixes](#fine-tuning-suffixes)
    - [Quantization (GGUF / llama.cpp)](#quantization-gguf--llamacpp)
    - [Other ecosystems](#other-ecosystems)
    - [Picking SD pairs](#picking-sd-pairs)
- [Quantization](#quantization)
  - [What it is](#what-it-is)
  - [Why we do it - memory bandwidth is the bottleneck](#why-we-do-it--memory-bandwidth-is-the-bottleneck)
  - [Does it hurt quality?](#does-it-hurt-quality)
  - [Do other inference engines use it?](#do-other-inference-engines-use-it)
  - [How far to push it](#how-far-to-push-it)
  - [How quantization interacts with speculative decoding](#how-quantization-interacts-with-speculative-decoding)
- [Concrete Directions](#concrete-directions)
  - [Adaptive Draft Length for llama.cpp](#adaptive-draft-length-for-llamacpp)
  - [Tree-Structured Drafting in llama.cpp](#tree-structured-drafting-in-llamacpp)
  - [Heterogeneous CPU/GPU Scheduling for Edge](#heterogeneous-cpugpu-scheduling-for-edge)
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
- [Compiler-Assisted Speculative Sampling for Accelerated LLM Inference on Heterogeneous Edge Devices](#compiler-assisted-speculative-sampling-for-accelerated-llm-inference-on-heterogeneous-edge-devices)
  - [Key takeaways](#key-takeaways-4)
- [Adaptive Draft length (see: AdaEDL, SpecDec++)](#adaptive-draft-length-see-adaedl-specdec)
  - [Profile-Guided Optimization for Speculative Decoding](#profile-guided-optimization-for-speculative-decoding)
- [Trace-Based Speculation](#trace-based-speculation)
  - [Lightweight JIT for token prediction that runs alongside the draft model](#lightweight-jit-for-token-prediction-that-runs-alongside-the-draft-model)
- [Implementations](#implementations)
- [Quality Evaluation](#quality-evaluation)
  - [PPL of two samples is not a comparison](#ppl-of-two-samples-is-not-a-comparison)
  - [The acceptance rule and why it preserves the distribution](#the-acceptance-rule-and-why-it-preserves-the-distribution)
  - [The TV-distance bound on acceptance](#the-tv-distance-bound-on-acceptance)
  - [Perplexity as a sanity check](#perplexity-as-a-sanity-check)
  - [Metrics](#metrics)
  - [Data convention](#data-convention)
  - [How the literature evaluates "quality is preserved"](#how-the-literature-evaluates-quality-is-preserved)
  - [When PPL actually matters (lossy variants)](#when-ppl-actually-matters-lossy-variants)
- [Resources](#resources)
- [Footnote](#footnote)
<!--toc:end-->

## Intro
> Η εργασία αυτή μελετά την τεχνική speculative sampling, με σκοπό την επιλογή ενός μικρού γλωσσικού μοντέλου (SLM) αντί ενός μεγάλου (LLM), όταν το SLM μπορεί να αποδώσει εξίσου καλά, ενεργοποιώντας το LLM μόνο όταν κρίνεται απαραίτητο.
> 
> Θα διεξαχθούν πειράματα για τη σύγκριση διαφορετικών προσεγγίσεων, με μετρήσεις χρόνου απόκρισης και κατανάλωσης ενέργειας σε σχέση με τα αρχικά μεγάλα μοντέλα.
> 
> Ιδιαίτερη έμφαση θα δοθεί (α) στην επιλογή των κατάλληλων μοντέλων και (β) στην επίδρασή τους στην ακρίβεια και στην αποδοτικότητα των αποτελεσμάτων.

> Inference scheduling problem. Compilers already have language for this: branch prediction, PGO, instruction scheduling. LLM decode is a pipeline; an AST is a pipeline too ([Abstract Syntax Tree](https://en.wikipedia.org/wiki/abstract_syntax_tree)).
>
> Speculative decoding by itself is 2-3x in the papers. Adaptive k / tree drafts / CPU-GPU split might help on long sequences or when the draft is often wrong. Might not. That's the experiment.

* SD pays off when the draft is cheap and most of its tokens get accepted.
* GPU: verifying several tokens in one pass is cheap, so this works.
* CPU: drafting itself is expensive, and tiny logit mismatches reject a lot.
* Speculative decoding in LLMs is speculative execution in CPUs.

### Wikipedia entry for Speculative Decoding
> [!TIP]
>
> [Source](https://en.wikipedia.org/wiki/Transformer_(deep_learning)#Speculative_decoding)

* Speculative decoding is a method to accelerate token decoding.
* Similarly to speculative execution in CPUs, future tokens are computed quickly, then verified.
* If the quickly computed tokens are incorrect, they are discarded and computed slowly.
* The key factor in speculative decoding is that a transformer decoder can verify faster than it can decode, in the following sense.

### How Speculative Sampling works in general
from: [Reward-Guided Speculative Decoding for Efficient LLM Reasoning](#resources-reward-guided-speculative-decoding-for-efficient-llm-reasoning)
> The smaller model serves as a guide, proposing the overall sequences that the larger model can confirm or adjust, leading to faster inference without compromising quality.

On GPU this is a good fit:
* draft on tensor cores
* target on tensor cores
* kernels overlap
* bandwidth is huge

Doesn't help unless:
* draft is much cheaper per token than the target
* running both doesn't eat the savings
* acceptance is high enough (ballpark >60%)

What actually happens:
* Small model proposes a few next tokens (the draft).
* Large model scores them in one pass.
* Keep a proposal if it's likely under the large model. That's the speedup: the large model didn't decode those tokens one by one.
* Leviathan rule: if p_target ≥ p_draft, accept (doesn't change the target's distribution). If lower, accept with probability p/q. A miss wastes the draft work. Good draft → faster, same output. Bad draft → slower than AR. See: [Branch predictor](https://en.wikipedia.org/wiki/Branch_predictor)
* Some papers exit early when the model is confident. Loose analogue of skipping work in a compiler.
* Tree drafts: several paths instead of one sequence, prune the rest. Not in spectre (parked).
* "Schedule the high-probability path first" is the OOO analogy. Also not this month.

Softmax over the vocab. Same probabilities used in normal decoding.

## Model naming conventions

How to read a filename like Qwen3-30B-A3B-Instruct-Q4_K_M.gguf. Four or five unrelated things jammed into one string.

### Reading a filename

| Slot | Example | What it means |
|---|---|---|
| Family / version | Qwen3 | Model family and major version |
| Size | 30B-A3B | Parameter counts (see below) |
| Fine-tuning | Instruct | Training objective |
| Quantization | Q4_K_M | How weights are stored (see below) |
| Format | .gguf | File format (here, llama.cpp's GGUF) |

Decoding the local models in this repo:

```
Qwen2.5-Coder-3B-Instruct-IQ2_M.gguf
↑       ↑    ↑   ↑       ↑    ↑
fam     v    spec size  tune  quant (i-quant, 2-bit medium)

Nemotron-3-Nano-4B-BF16.gguf
↑        ↑ ↑    ↑  ↑
fam      v size size precision (no quant - full bf16)
```

Nano here is NVIDIA's size-class label (Nano < Mini < Small < Medium < Large)
rather than a separate slot.

### Parameter count

Dense models (the simple case):

```
8B    = 8 billion parameters, every one used per token
70B   = 70 billion parameters, every one used per token
1.5B  = 1.5 billion parameters
```

Memory ≈ params × bytes_per_param. A 70B model at FP16 is ~140 GB.

MoE models - the XB-AYB form. X is total, A is active per token:

```
30B-A3B            = 30 B total parameters, ~3 B active per token
Mixtral 8x7B       ≈ 47 B total, ~13 B active (top-2 of 8 experts)
DeepSeek-V3 671B-A37B = 671 B total, 37 B active
```

A Mixture-of-Experts layer holds N small "expert" sub-networks plus a router.
The router picks the top-k experts per token, and only those experts run.
Memory is paid for all 30 B parameters; compute is only ~3 B per token.

Why this shows up in the thesis:

- Throughput tracks active params, not total. A 30B-A3B that fits in VRAM decodes closer to a dense 3B than to a dense 30B.
- SD ratios should use active params. Usual rule of thumb is draft ≈ target / 10..20. A 30B-A3B target is asking a 3B draft to keep up, which is harder than drafting for a dense 30B.
- Memory is still needed for all 30B unless experts are offloaded.

MoE models that show up a lot:

| Model | Total | Active | Notes |
|---|---:|---:|---|
| Mixtral 8x7B | 47 B | ~13 B | 8 experts, top-2 routed |
| Mixtral 8x22B | 141 B | ~39 B | larger Mixtral |
| Qwen3-30B-A3B | 30 B | 3 B | dense-draft-friendly active size |
| Qwen3-235B-A22B | 235 B | 22 B | flagship Qwen3 |
| DeepSeek-V3 | 671 B | 37 B | extreme MoE ratio (~5%) |

### Fine-tuning suffixes

| Suffix | Meaning |
|---|---|
| (none) or -Base | Pretrained on text completion only - no instruction following |
| -Instruct | SFT'd on instruction-following pairs |
| -Chat | Same idea, often with multi-turn formatting |
| -Coder | Continued pre-training on code |
| -Math, -Reasoning, -Thinking | Specialized variants |
| -DPO, -RLHF, -RLAIF | Indicates the preference-training method used |
| -it | Some labs use -it for "instruction-tuned" (Gemma) |

For SD, draft and target should be the same kind of fine-tune.
-Instruct with -Instruct. Mixing a base model with a chat model tanks
acceptance: same prompt template, different distributions.

### Quantization (GGUF / llama.cpp)

Bit-width prefix:

```
F32 / F16 / BF16  = full or half precision floats
Q8_0 / Q6_K       = 8 or 6 bits per weight
Q5_K / Q4_K       = 5 or 4 bits
Q3_K / Q2_K       = 3 or 2 bits
IQ4_XS / IQ3_M    = importance-matrix quants (i-quants)
IQ2_XXS / IQ1_S   = ultra-low-bit (1.5-2.5 bits effective)
```

Suffix tier. _S, _M, _L after a K-quant means
Small / Medium / Large mixed-precision variant:

```
Q4_K_S = mostly 4-bit; smallest size, lowest quality
Q4_K_M = 4-bit with critical layers (attention, output) bumped higher - best-balanced
Q4_K_L = even more critical layers kept higher
```

M is the usual pick. S is smaller and worse; L rarely worth it.

Q_K vs IQ_:

- Q_K (K-quants): groups of weights share a scale/min. Fast, simple. ~mid-2023.
- IQ_ (I-quants / importance-matrix): calibration set (the "imatrix") decides
  which weights keep more bits. Same bit budget, usually better quality, a bit
  slower on some hardware.

At 4 bits and below I prefer IQ if it exists. At 5+ bits Q_K_M is fine.

Legacy / non-K formats:

```
Q4_0, Q4_1, Q5_0, Q5_1   = older single-precision-per-block formats
Q8_0                      = 8-bit, used as the high-fidelity reference; "essentially lossless"
```

Q8_0 is worth remembering: half the size of FP16, effectively lossless.
Usually the right draft quant. Fast, small, and any quality drop is the
target's problem because the target verifies.

### Other ecosystems

| Format | Where |
|---|---|
| GPTQ | HuggingFace, AutoGPTQ, ExLlama - GPU-focused 4-bit |
| AWQ | activation-aware quantization, GPU |
| EXL2 | ExLlamaV2; per-layer mixed bit-width |
| MLX | Apple Silicon |
| safetensors (uncompressed) | the FP16/BF16 source weights from HF |

For llama.cpp work, only GGUF + the Q_K/IQ family is relevant.

Filenames sometimes also carry:

```
-128k      context window in tokens (e.g. Qwen2.5-Coder-7B-Instruct-128k)
-YaRN      rope-scaling extension method (rare in filenames)
-RAG       fine-tuned for retrieval-augmented generation
-Function  fine-tuned for tool-use / function calling
```

### Picking SD pairs

What actually moves SD:

1. Same family, same fine-tune. Vocab has to match. spectre checks this at startup.
2. Active-param ratio. Aim for draft active ≈ target active ÷ 8..20.
   - Dense 26B: draft 1.5B-3B.
   - MoE 30B-A3B: the ideal draft is 0.2B-0.4B, which almost never exists. So SD on MoE is harder, not easier.
3. Quant pair. Target Q5_K_M or higher (a noisy target wastes the verification). Draft Q4_K_M or Q8_0.

The Nemotron BF16 (target) + Nemotron Q8_0 (draft) pair in this repo is
self-speculation: same model, two precisions. Q8 is ~50% faster to evaluate
and close enough that acceptance is high. Useful baseline. Not what most
papers do (they train a smaller draft).

## Quantization

The Q4_K_M / IQ2_M / BF16 bits in filenames. Quantization and SD both
try to spend less time waiting on memory. Quantization: smaller weights. SD:
fewer target weight-reads per output token.

### What it is

Store each weight in fewer bits than it was trained at.

Training is usually float32 (4 bytes) or bfloat16 (2 bytes). Quantization
packs that into 8, 4, sometimes 2 bits. Some precision is lost.

Simplest scheme (uniform int8):

```
original  weight   ∈ [-W_max, +W_max]                   (float, full range)
quantized weight   = round(weight × (127 / W_max))      → int8 ∈ [-128, +127]
restored  weight   ≈ quantized / (127 / W_max)          (back to float, lossy)
```

Store the int8 and the scale W_max/127. At inference, dequantize on the fly
for the matmul.

Actual formats (Q4_K_M, IQ3_M, ...) do more than that:

- Block-wise scaling: ~32 weights share a scale, so a few outliers don't wreck the tensor.
- Mixed precision: attention output / embeddings stay at higher bit-width than FFN.
- Importance weighting (the "I" in I-quants): a calibration set decides which weights get more bits.

That's it. Fewer bits, smaller file.

### Why we do it - memory bandwidth is the bottleneck

"Doesn't fit in VRAM" is the symptom. Decode is bandwidth-bound. Compute is not.

One token from a 7B model on an A100:

| Operation | Numbers |
|---|---|
| Weights read from VRAM -> cache | ~7 B x 2 bytes = 14 GB |
| Math performed | ~7 B x 2 = 14 G FLOPs |
| A100 memory bandwidth | ~2 TB/s |
| A100 compute throughput (FP16) | ~312 TFLOP/s |
| Time spent moving weights | 14 GB / 2 TB/s = 7 ms |
| Time spent computing | 14 G / 312 T = 0.045 ms |

Compute: 45 μs. Memory: 7 ms. The GPU is waiting on weights. That's decode.

Same 7B, quantized:

| Quant | Bytes/weight | Read time | Implied tok/s |
|---|---:|---:|---:|
| F16 (baseline) | 2.0 | 7.0 ms | ~143 |
| Q8_0 | 1.0 | 3.5 ms | ~286 |
| Q4_K_M | 0.55 | 1.9 ms | ~526 |

This does not speed up the math. After dequant the GPU still computes in
higher precision. It speeds up the transfer. Since that's the wait,
tok/s scales almost with bytes/weight. Q4_K_M is ~3.7× vs F16 because the
weights are ~3.7× smaller.

### Does it hurt quality?

Yes. How much depends how far the quant goes. Usual measure: PPL gap vs FP16 on a
held-out corpus. Rough curve (shape is consistent, numbers move by model):

| Quant | Quality loss vs F16 | Size vs F16 |
|---|---|---:|
| F16 | 0 (reference) | 100 % |
| Q8_0 | ~0.0 % PPL Δ | 50 % |
| Q6_K | ~0.1 % | 38 % |
| Q5_K_M | ~0.5 % | 33 % |
| Q4_K_M | ~1-2 % | 28 %  ← practical sweet spot |
| Q3_K_M | ~3-6 % | 23 % |
| Q2_K | ~10-20 % | 19 %  ← starts to matter |
| IQ1_S | 30-50 %+ | 14 %  ← only useful in extremis |

Above ~4 bits the loss is often smaller than changing the seed. Below ~3 bits
it shows up, and long chains of reasoning start falling apart.

Bigger models survive this better. A 70B at 4-bit often beats a 13B at 16-bit
on both quality and speed, at similar memory. That's the local-inference
advice: go bigger, then quantize.

### Do other inference engines use it?

Yes. Everyone:

| Engine | Quantization support |
|---|---|
| llama.cpp / GGUF | K-quants and I-quants (this thesis) |
| vLLM (Berkeley/UC, dominant server) | AWQ, GPTQ, FP8, INT8 KV cache, FP8 KV cache |
| TGI (HuggingFace) | bitsandbytes (8/4-bit), GPTQ, AWQ, EETQ |
| TensorRT-LLM (NVIDIA, production) | FP8 (heavily), INT8, INT4 weight-only |
| ExLlamaV2 | EXL2 - bespoke per-layer mixed bit-width |
| MLX (Apple) | INT4, INT8 quantization for Apple Silicon |
| MLC-LLM (mobile) | INT3, INT4 |
| Ollama, LM Studio, Jan | all wrap llama.cpp |

Hosted APIs (Anthropic, OpenAI, Google, DeepInfra, Together, Fireworks, ...)
quantize internally too. Latency/cost tiers are probably different quants of
the same backbone.

Research training is not quantized. Gradients blow up. (FP8 training
is a thing now at large scale.)

### How far to push it

- Q8_0: 50% memory, ~0 quality loss. Worth using. Q6_K / Q5_K_M almost the same.
- Q4_K_M: ~1-2% on standard benches, ~3-4× faster. Fine for a chatbot. Maybe not if the task is actually high-stakes.
- IQ2 / IQ1: real quality hit, because otherwise the model doesn't load. This is how a 70B runs on a 16 GB card.

### How quantization interacts with speculative decoding

1. Same bottleneck, different axis. Quant: fewer bytes per weight. SD: fewer
   target weight-reads per output token (verify N drafts in one pass). They
   multiply: 4× from Q4 × 2× from SD ≈ 8× vs F16 AR. quality-speed-vs-accept.png
   is the SD half. The quant sets the baseline those plots sit on.

2. Self-spec (Nemotron BF16 target + Q8 draft) is quant-as-draft. The Q8 is a
   noisy copy of the target, so acceptance stays high. LayerSkip / self-distillation
   papers treat this as its own family.

3. Quant choice moves acceptance even inside one family. Both at Q4_K_M → more
   correlated than FP16 target + Q4 draft, because they made similar rounding
   errors. That's usable, not a bug.

## Concrete Directions
### Adaptive Draft Length for llama.cpp

llama.cpp uses fixed --draft-max and --draft-min. Optimal k depends on how hard the next tokens are.
> Easy stuff (boilerplate, code patterns) → draft long. Hard stuff (reasoning, rare words) → draft short or don't.

* Entropy or a small prediction head in llama.cpp's speculative loop. SpecDec++ / AdaEDL. AdaEDL is training-free.

* Entropy lower bound on acceptance → stop drafting. No training, just instrument the draft loop.

* AdaEDL paper: 10-57% over static k.

* This is PGO if profiles are collected offline. AdaEDL itself is an online heuristic, not PGO. Mixing the two in the thesis is a mistake.

### Tree-Structured Drafting in llama.cpp

llama.cpp currently does linear speculative drafting (one sequence). Tree drafts (branch at uncertain positions) raise acceptance. [TALON](https://arxiv.org/abs/2601.07353) claims up to 5.16x with deep-narrow trees on easy context and shallow-wide on uncertain context.

* Tree draft in llama.cpp with adaptive branching. Hard part: tree attention for verification in GGML.

* Same idea as speculative execution with fan-out: several paths, retire in order, drop the rest. Parked (see todo.txt).

### Heterogeneous CPU/GPU Scheduling for Edge

There's a Feb 2026 paper ([Compiler-Assisted Speculative Sampling for Accelerated LLM Inference on Heterogeneous Edge Devices](https://arxiv.org/abs/2602.08060)) that does the intro pitch: cost model for draft/target on CPU+GPU edge SoCs. 1.68x on ARM Cortex-A + Mali.

* Cost model for llama.cpp: should we speculate at all, and where (CPU threads, GPU, NPU).
> llama.cpp already does hybrid CPU/GPU via layer offload. Speculation-aware scheduling would sit on top of that. Out of scope for September.

### Relative Papers

- [SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths](https://arxiv.org/abs/2405.19715)
- [TALON: Confidence-Aware Speculative Decoding with Adaptive Token Trees](https://arxiv.org/abs/2601.07353)
- [Compiler-Assisted Speculative Sampling on Heterogeneous Edge Devices](https://arxiv.org/abs/2602.08060)
- [Efficient Speculative Decoding for Llama at Scale](https://arxiv.org/abs/2508.08192)
- [Speculative Speculative Decoding (Saguaro)](https://arxiv.org/abs/2603.03251)
- Eagle-3, currently the SOTA spec decoder
    - https://github.com/ggml-org/llama.cpp/discussions/15902
    - https://github.com/ggml-org/llama.cpp/pull/18039
    - https://github.com/ggml-org/llama.cpp/pull/18471

## Accelerating Large Language Model Decoding with Speculative Sampling
> [!TIP]
>
> auto-regressive sampling vs speculative sampling
>
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
>
> By Team PyTorch. May 2, 2024
>
> [Source](#resources-a-hitchhikers-guide-to-speculative-decoding)

### Key takeaways
1. Guess future tokens while generating the current one
2. One forward pass. No [backpatching](https://www.geeksforgeeks.org/compiler-design/backpatching-in-compiler-design/)
3. Assumes the model can predict more than one token per pass
4. Then verify
5. Output = vanilla decoding (same distribution)

## Speculative Sampling Explained
> [!TIP]
>
> Use a draft sampling to achieve the same sampling result as the target sampling
>
> [Source](#resources-speculative-sampling-explained)

## Fast Inference from Transformers via Speculative Decoding
> [!TIP]
>
> Inference from large autoregressive models like Transformers is slow - decoding K tokens takes K serial runs of the model
>
> [Source](#resources-fast-inference-from-transformers-via-speculative-decoding)

### Key takeaways
1. Accelerate existing off-the-shelf models without retraining or architecture changes
2. Sample from autoregressive models faster by computing several tokens in parallel. Why this works:
    - hard language-modelling tasks often include easier subtasks that can be approximated well by more efficient models
    - using speculative execution and a novel sampling method, we can run large models in parallel on the outputs of the approximation models and thus generate several tokens concurrently without changing the distribution
3. A single decode step from large autoregressive models (notably transformers) is significantly slower than a step from their smaller counterpart
4. Several approaches were developed to make inference from them faster.
    - Reduce the inference cost for all inputs equally
    - Adaptive Computation Method --> not all inference steps are equal, use the large models where it makes sense
5. Some inference steps are "harder" and some are "easier"
6. Inference from large models is often not bottlenecked on arithmetic operations, but rather on memory bandwidth and communication --> thus additional computation resources might be available

## Reward-Guided Speculative Decoding for Efficient LLM Reasoning
> [!TIP]
>
> Unlike normal speculative sampling we incorporate controlled bias to prioritize high-reward outputs.
>
> [Source](#resources-reward-guided-speculative-decoding-for-efficient-llm-reasoning)

### Key takeaways
1. Evaluate intermediate decoding steps and dynamically decide (threshold based) to invoke the target model.
2. Unbiasedness maintains theoretical fidelity but often reduces efficiency (especially when the draft diverges from the target)
3. Allowing controlled bias (where the final distribution deviates slightly from the large model) can improve performance
4. If a draft token is correct but does not match the large model's distribution exactly, strict rejection is counterproductive.
5. Reward-guided acceptance: keep partial solutions, fewer target calls, sometimes beats the large model on the task.

### Reward-Guided Speculative Decoding (RSD)
- Cheap draft evals + reward from the target
- Keep high-value drafts instead of dumping them on mismatch

## Speculative Speculative Decoding (SSD)
> [!TIP]
>
> Speculative Decoding relies on a sequential dependence between speculation and verification, so we parallelize these operations.
>
> [Source](#resources-speculative-speculative-decoding)

### Key takeaways
1. While a verification is ongoing, the draft model predicts likely verification outcomes and prepares speculations pre-emptively for them.
2. If the actual verification outcome is then in the predicted set, a speculation can be returned immediately, eliminating drafting overhead entirely.
3. The result is SAGUARO, an optimized SSD algorithm.

## Compiler-Assisted Speculative Sampling for Accelerated LLM Inference on Heterogeneous Edge Devices
> [!TIP]
>
> When speculative sampling and heterogeneous execution are jointly beneficial and is validated on an edge devic
>
> [Source](#compiler-assisted-speculative-sampling-for-accelerated-llm-inference-on-heterogeneous-edge-devices)

### Key takeaways
1. SD at the edge is hindered by two major challenges:
    * Integrating SD into a compiler-based workflow without sacrificing performance or programmability
    * Exploiting the heterogeneous compute resources of modern SoCs through carefully designed partitioning strategies
2. Use an analytical cost model that explores heterogeneous hardware configurations and guides coarse-grained partitioning of LLM subgraphs, particularly with edge-typical short input sequence lengths.

## Adaptive Draft length (see: AdaEDL, SpecDec++)
### Profile-Guided Optimization for Speculative Decoding

1. PGO is a compiler technique:
    * instrument code,
    * run it on representative inputs,
    * collect profiles,
    * recompile with the profile data to make better decisions.

2. AdaEDL, SpecDec++ don't use PGO, they use online heuristics.

> This is not machine learning.
>
> Not training a network. Collecting execution profiles and using them to set a policy.
> Same job as GCC -fprofile-generate / -fprofile-use, different domain.

1. Instrumentation
    * Run the draft+target model pair on representative inputs. At each speculative step, log:
        - The draft model's entropy/confidence at each position
        - Whether the token was accepted or rejected
        - The context features (token type, position in sequence, preceding pattern)

2. Profile analysis
    * From the collected data, a cheap predictor: context features → expected acceptance.
    * Same role as a compiler branch profile ("this call site is taken 94% of the time")

3. Policy
    * High expected acceptance → draft long
    * Low → draft short or skip
    * Very uncertain → tree (fan-out). Parked.

## Trace-Based Speculation
### Lightweight JIT for token prediction that runs alongside the draft model

> trace-based compilation, but for tokens.

LuaJIT / V8 / PyPy: record hot paths, compile those, interpreter for the rest.

Same idea here:
    * After def, the next tokens are almost always name(args): that's a hot trace
    * Hot → long drafts
    * Cold / unfamiliar → short draft or AR
    * Store traces as n-grams, a trie, whatever. This is basically the n-gram drafter with extra steps.

## Implementations
* Speculative decoding: generate draft tokens cheaply, verify them with the target in one batch. Speedup if the draft is usually right.

* GPU: parallel verify is cheap, so this works.
* CPU: drafting is expensive and small logit diffs reject a lot.

* Families (a draft-model method can be mixed with a no-draft-model method)
    * Draft model (most common)
    * n-gram {Cache, Map, Mod} - pattern match on the prompt / previous output
    * EAGLE-{1,2,3} - extra module on the target's internals, currently the fastest
    * Speculative Speculative Decoding - while verify is in flight, the draft predicts likely outcomes and pre-prepares the next speculation. Saguaro.

* What spectre actually does
    * start()
        * initialize();
            * initialize libllama backend
            * load target/draft models + CTXs
        * tokenize();
            * get model VOCABs
            * tokenize prompt
        * decode();
            * initialize sampler/decoder (temp, top-k/p, dist) for target/draft
            * prepare first batch of tokens (aka the prompt)
        * run();
            * if speculative:
                * assert prompt -> get prompt batch
                * evaluate the prompt batch with the transformer
                * sample starting from the last token of the prompt
                * while !end_of_sentence:
                    * draft() tokens, resize if too many, reset if too few
                    * do the actual verification of the sampled tokens
                    * get last token
                    * get logit of last token -> softmax to normalize
                    * is eog:
                        * done
                    * else:
                        * get token string representation
            * else:
                * evaluate the prompt batch with the transformer
                * get last token
                * get logit of last token -> softmax to normalize
                * is eog:
                    * done
                * else:
                    * get token string representation
                    * prepare the next batch with the sampled token

```
  flowchart TB
    subgraph start["start()"]
      direction TB
      I[initialize]
      T[tokenize]
      D[decode]
      R[run]
      I --> T --> D --> R
    end
    subgraph initialize["initialize()"]
      direction TB
      I1[Initialize libllama backend]
      I2[Load target and draft models plus contexts]
      I1 --> I2
    end
    subgraph tokenize["tokenize()"]
      direction TB
      T1[Get model vocabs]
      T2[Tokenize prompt]
      T1 --> T2
    end
    subgraph decode["decode()"]
      direction TB
      D1["Sampler and decoder setup<br/>temp, top-k, top-p, dist"]
      D2[First batch equals prompt tokens]
      D1 --> D2
    end
    subgraph run["run()"]
      direction TB
      Q{Speculative decoding?}
      Q -->|yes| S
      Q -->|no| N
      subgraph S["Speculative path"]
        direction TB
        S0[Assert non-empty prompt]
        S1[Prompt batch through transformer]
        S2[Sample from last prompt token]
        S3{End of sentence?}
        S4[Draft tokens, resize or reset count]
        S5[Verify with target model]
        S6[Last token, logit, softmax]
        S7{EOG token?}
        S8[Token to string and metrics]
        S0 --> S1 --> S2 --> S3
        S3 -->|no| S4 --> S5 --> S6 --> S7
        S7 -->|yes| SDONE[Done]
        S7 -->|no| S8 --> S3
      end
      subgraph N["Non-speculative path"]
        direction TB
        N1[Prompt batch through transformer]
        N2{EOG token?}
        N3[Last token, logit, softmax]
        N4[Token to string and metrics]
        N5[Next batch is sampled token]
        N1 --> N2
        N2 -->|yes| NDONE[Done]
        N2 -->|no| N3 --> N4 --> N5 --> N1
      end
    end
```

## DFlash
* https://github.com/z-lab/dflash
* https://arxiv.org/abs/2602.06036
* https://arxiv.org/pdf/2602.06036

## server: fix checkpoints creation by jacekpoplawski · Pull Request #22929 · ggml-org/llama.cpp
* https://www.reddit.com/r/LocalLLaMA/comments/1tn0jyp/server_fix_checkpoints_creation_by_jacekpoplawski/
* https://github.com/ggml-org/llama.cpp/pull/22929

## Quality Evaluation

> [!TIP]
>
> How do we know SD text is "as good as" what the target would have written alone?
> Vanilla SD: identical in distribution, by construction. Details below.

### PPL of two samples is not a comparison

First idea: run target alone, run SD, compare PPL of the two texts. Skip that.

1. They sample different sequences. PPL of a generated sequence under the model that generated it is self-evaluation. Two valid completions of the same prompt can have very different per-token PPL and both be fine.
2. For vanilla SD the comparison is empty. Correct SD is a sample from the target. PPL(SD | target) and PPL(AR | target) have the same distribution. "They match" proves nothing. "They differ" means the implementation is wrong.

So: PPL as a sanity check (did I implement acceptance correctly) and as a drift metric (lossy variants). Not as a quality bake-off between two lossless algorithms.

### The acceptance rule and why it preserves the distribution

Let p(x | ctx) be the target distribution and q(x | ctx) the draft distribution at the current position. Draft proposes x ~ q. Speculative sampling accepts x with probability

```
α(x) = min(1, p(x) / q(x))
```

and, if rejected, resamples from the residual distribution

```
p̃(x) ∝ max(0, p(x) − q(x))     (normalized so it sums to 1)
```

Theorem (Leviathan et al. 2022; Chen et al. 2023): the accepted token is distributed as p(·|ctx). Exactly.

Proof sketch. Mass on a particular token x* after one step:
- accepted from draft: q(x*) * min(1, p(x*)/q(x*)) = min(q(x*), p(x*))
- residual after rejection: P(reject) * p_res(x*) where P(reject) = 1 - sum min(p(x), q(x)) and p_res(x*) = max(0, p(x*) - q(x*)) / P(reject)
- sum: min(p(x*), q(x*)) + max(0, p(x*) - q(x*)) = p(x*).

So vanilla SD output is statistically the same as target-only sampling. "Is the draft garbage?" reduces to "is the verifier correct." Spectre's verifier is sampler-agreement, not this p/q rule. Leviathan lives in Chapter 2.

### The TV-distance bound on acceptance

The only dial that affects how much gets accepted is how close q is to p. Expected per-token acceptance probability:

```
E[α] = Σₓ q(x) · min(1, p(x)/q(x))
     = Σₓ min(p(x), q(x))
     = 1 − ½ Σₓ |p(x) − q(x)|
     = 1 − TV(p, q)
```

So TV(p, q) is the upper bound on per-token acceptance. Pinsker: TV ≤ √(½ KL(p ‖ q)), so high KL ⇒ low acceptance. The reverse is weaker.

For n-gram speculators, q is basically a Dirac on the predicted token:
- TV(p, q) = 1 − p(x_predicted)
- Acceptance at that position = p(x_predicted)
- Over many calls, A / G ≈ Bernoulli with parameter E[p(x_predicted)]

That's why A / G is a decent summary for n-gram drafting specifically.

### Perplexity as a sanity check

Per-token target probability of accepted tokens is written by RunRecorder::record_token
in spectre/src/main.cpp, into results/spectre/<run-id>/tokens.csv (schema below).

Corpus PPL of a generated sequence:

```
PPL = exp( -mean( log p_target(x_t | x_<t) ) )
```

For vanilla SD this should match (within sampling noise) the PPL of a target-only
run with the same prompt and seed. If they disagree, the acceptance code is
wrong. --seed pins the sampler so the comparison is reproducible.

### Metrics

What I'll report, in this order:

1. Speed
   - decode throughput tok/s (from eval time in llama-server logs)
   - per-token latency ms/token
   - prefill tok/s (mostly invariant to SD, still useful as a sanity check)

2. Efficiency (is the drafter doing anything)
   - acceptance rate = A / G (A accepted draft tokens, G generated draft tokens)
   - block efficiency τ = E[accepted per call] (Leviathan 2022)
   - mean accepted length per speculative round
   - per-position acceptance P(i-th accepted) = p̂ⁱ under an independent-Bernoulli model

3. Quality (only for lossy variants; sanity check for lossless)
   - per-token log p_target(accepted)
   - corpus PPL of generated text
   - histogram of p_target(x) (where is it hard)
   - KL(p || q) per position if both distributions are available

Scripts (see spectre/scripts/README.md for the full reference):
- spectre/scripts/benchmark.sh sweeps the spectre binary across (seed × n_max)
  configs and writes one results/spectre/<run-id>/ directory per run.
  Default pair is Qwen2.5-1.5B drafting for Qwen2.5-3B (fast smoke). Override with
  TGT_MODEL=...gemma-4-26B-A4B... DFT_MODEL=...gemma-4-E2B... ./spectre/scripts/benchmark.sh
  for the realistic Gemma showcase.
- spectre/scripts/quality_eval.py reads every results/spectre/<run-id>/ directory
  and produces 7 PNGs into spectre/presentation/png/quality-*.png.
- spectre/presentation/html/quality.html for the live HTML view.

Reproduce end-to-end:
```bash
./spectre/scripts/benchmark.sh               # ~30-90s on a small GPU
python3 spectre/scripts/quality_eval.py      # regenerate figures
```

### Data convention

Each spectre run writes:

```
results/spectre/<run-id>/
  meta.json       config + totals + per-round summaries
  tokens.csv      one row per accepted token
```

Run-id is YYYYMMDD-HHMMSS_<mode>_seed<N> or --run-id. meta.json starts with
"complete": false and gets rewritten at the end with totals. tokens.csv is
flushed per row. Both survive SIGTERM / SIGINT.

meta.json schema (annotated):

```jsonc
{
  "run_id": "20260531-141523_spec_seed42",
  "started_at": "2026-05-31T14:15:23",
  "complete": true,                 // false iff process was killed pre-finalize
  "config": {
    "tgt_model_path": "...gguf",
    "dft_model_path": "...gguf"|null,
    "speculative": true,            // = (dft_model_path != null)
    "ctx": 4096, "ngl": -1,
    "n_min": 0, "n_max": 8,
    "temp": 0.8, "top_p": 0.9, "top_k": 40, "greedy": false,
    "seed": 42,
    "prompt_n_chars": 432,
    "prompt": "..."
  },
  "totals": {
    "n_decoded_tokens": 173,
    "n_drafted": 280,
    "n_accepted_drafts": 100,
    "n_bonus_samples": 50,
    "accept_rate": 0.357,
    "prompt_ms": 117.9,
    "decode_ms": 37862.7,
    "total_ms": 37980.6,
    "tok_per_s": 4.57
  },
  "rounds": [                       // one entry per speculative call
    {"n_drafted": 4, "n_accepted_drafts": 2, "rejected_pos": 2},
    {"n_drafted": 4, "n_accepted_drafts": 4, "rejected_pos": -1}
  ]
}
```

rejected_pos = -1 means all draft tokens in that call matched (the target then
contributed a bonus sample, counted in n_bonus_samples).

tokens.csv schema:

| column          | type    | meaning |
|---|---|---|
| step          | int     | global accepted-token index (0-based) |
| call          | int     | speculative call index (0-based); equals step in AR mode |
| source        | str     | "draft", "bonus", or "ar" |
| pos_in_draft  | int     | 0..k-1 if source=="draft", else -1 |
| token_id      | int     | token id of the accepted token |
| p_target      | float   | target's probability mass on the accepted token |
| p_draft       | float   | draft's probability mass on the same token (empty for bonus/ar) |
| logit         | float   | target's logit value of the accepted token |
| logprob       | float   | log p_target |

Empty p_draft cells are NaN. The script quality_eval.py handles both empty and NaN.

Reproducibility for the thesis: full prompt, --seed, absolute model paths,
started_at to match system logs. To regenerate: spectre/build/spectre --run-id <same> ...
overwrites the directory. (binary is spectre, not main - benchmark.sh still
looks for main, that's a todo.)

### How the literature evaluates "quality is preserved"

| Family | Examples | Quality battery |
|---|---|---|
| Vanilla SD (lossless) | Leviathan 2022, Chen 2023 | None required; PPL parity as sanity check |
| Self-speculation / n-gram | llama.cpp ngram_*, our PR #22055 | None required (still uses standard rejection rule) |
| Tree drafting (lossless) | SpecInfer, EAGLE, TALON | None required; ablations on tree topology |
| Soft / lossy acceptance | Medusa, Lookahead | HumanEval pass@1, MT-Bench, GSM8K, MMLU |
| Reward-guided | RSD (Liao 2025) | MATH, GSM8K, AIME - task-specific |
| Quantized draft | various | Corpus PPL on WikiText-103 / C4 + downstream |
| Biased acceptance (τ < 1) | various ablations | KL drift + downstream evals |

Standard datasets when downstream evaluation is required:
- Generation quality: MT-Bench, AlpacaEval, Arena-Hard
- Reasoning: GSM8K, MATH, AIME, MMLU
- Code: HumanEval, MBPP, LiveCodeBench
- PPL: WikiText-103, C4 validation, The Pile validation slice

What papers don't use for SD:
- BLEU / ROUGE / chrF - they punish valid alternative completions
- One-prompt qualitative "looks good" - not reproducible

### When PPL actually matters (lossy variants)

Lossless SD is the boring case (PPL preserved by construction). The interesting
stuff is lossy methods that trade exactness for speed:

1. Biased acceptance. α(x) = min(1, p(x)/q(x)) → α'(x) = min(1, τ · p(x)/q(x)) with τ > 1. More accepts, drift away from p. Plot speedup against KL(p ‖ p').
2. Medusa-style soft acceptance. Multiple heads vote, rejection is relaxed. They report PPL drift vs speedup.
3. Reward-guided (RSD). Acceptance is a reward threshold. Can beat the target on reasoning, at the cost of unbiasedness.
4. Aggressive draft quant (Q2/Q3 draft of a Q8 target). Raises TV(p, q). Either quality drift (strict rule) or more speed (relaxed rule).

Possible follow-up: p_target and p_draft are already in tokens.csv. Sweep
τ ∈ [1.0, 2.5] and plot extra-speedup vs mean |p_target − p_draft| (TV proxy)
vs HumanEval pass@1. Turns lossless/lossy into a continuous knob. Not September.

Figures produced by spectre/scripts/quality_eval.py (run after collecting fresh data):

| File | What it shows |
|---|---|
| quality-ppl-trace.png | per-token logprob + cumulative PPL; baseline AR overlay if a complete AR run exists |
| quality-target-prob-hist.png | distribution of p_target(accepted) split by source (draft / bonus / ar) |
| quality-acceptance-by-run.png | bar chart of acceptance rate per run |
| quality-speed-vs-accept.png | efficiency frontier (with baseline reference line) |
| quality-per-position-empirical.png | empirical per-position acceptance + Bernoulli model |
| quality-draft-vs-target-prob.png | p_target vs p_draft scatter on accepted tokens |
| quality-rounds-histogram.png | distribution of accepted-per-call vs. geometric model |


### n-gram Language Models

The intuition of the n-gram model is that instead of computing the probability of a
word given its entire history, we can approximate the history by just the last few
words

The bigram model, for example, approximates the probability of a word given
all the previous words P(w_n|w_1:n−1) by using only the conditional probability given
the preceding word P(w_n|w_n−1)

The assumption that the probability of a word depends only on the previous word is
called a Markov assumption. Markov models are the class of probabilistic models
that assume we can predict the probability of some future unit without looking too
far into the past

n-gram --> looks n−1 words into the past

Sampling from a distribution means to choose random points
according to their likelihood. Thus sampling from a language model -- which rep
resents a distribution over sentences -- means to generate some sentences, choosing
each sentence according to its likelihood as defined by the model.

• Language models offer a way to assign a probability to a sentence or other
  sequence of words or tokens, and to predict a word or token from preceding
  words or tokens.
• N-grams are perhaps the simplest kind of language model. They are Markov
  models that estimate words from a fixed window of previous words. N-gram
  models can be trained by counting in a training corpus and normalizing the
  counts (the maximum likelihood estimate).
• N-gram language models can be evaluated on a test set using perplexity.
• The perplexity of a test set according to a language model is a function of
  the probability of the test set: the inverse test set probability according to the
  model, normalized by the length.
• Sampling from a language model means to generate some sentences, choos
  ing each sentence according to its likelihood as defined by the model.
• Smoothing algorithms provide a way to estimate probabilities for events that
  were unseen in training. Commonly used smoothing algorithms for n-grams
  include add-1 smoothing, or rely on lower-order n-gram counts through inter
  polation

• Quantization Pareto (spectre/presentation/png/quant-pareto.png already exists).
  Bandwidth-bound on this hardware, so Q4_K/Q5_K is the operating point, not a
  compromise. Model selection is (architecture, quant), not architecture alone.
• Hybrid-cache metaphor: SD = speculative execution, n-gram = BTB, draft model = L2.
  The brief already uses this.

Claim I wanted to make: on bandwidth-bound consumer HW, n-gram → small model →
target verify should recover most of GPU SD's speedup and stay lossless. Measure
latency, accept, energy over (pair × quant × strategy). Report a frontier, not
one number.

"Multiple choice / reasoning benchmarks for SD" - those evaluate the target, not SD.

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
26. <span id="compiler-assisted-speculative-sampling-for-accelerated-llm-inference-on-heterogeneous-edge-devices"></span>[Compiler-Assisted Speculative Sampling for Accelerated LLM Inference on Heterogeneous Edge Devices](https://arxiv.org/pdf/2602.08060)
27. <span id="an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference"></span>[An Introduction to Speculative Decoding for Reducing Latency in AI Inference](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)

## Footnote
1. logit(p) = log(p/(1-p)) is the raw, denormalized predictions generated by a model before applying any activation function
2.
```
Ο Leviathan εγγυάται ότι:

\[
X_{\mathrm{SD}}\sim P_{\mathrm{target}}
\quad\text{και}\quad
X_{\mathrm{AR}}\sim P_{\mathrm{target}}.
\]

Άρα τα perplexities έχουν την ίδια κατανομή, όχι υποχρεωτικά την ίδια τιμή:

\[
\operatorname{PPL}(X_{\mathrm{SD}})
\overset{d}{=}
\operatorname{PPL}(X_{\mathrm{AR}}),
\]

αλλά δύο ανεξάρτητα παραγόμενα κείμενα μπορούν να έχουν διαφορετικό PPL, όπως δύο ρίψεις του ίδιου δίκαιου νομίσματος δεν
δίνουν αναγκαστικά το ίδιο αποτέλεσμα.
```
3. 
```
* Positional information, such as RoPE, is applied so that the model distinguishes:

    `A` at position 0

    from:

    `A` at position 4

* The embedding matrix can be viewed as:

    embedding_table[vocabulary_size][model_dimension]

* A Transformer repeatedly applies two main operations:

    1. Attention                    :   exchange information between tokens
    2. FFN (Feed-Forward Network)   :   transform the information inside each token

    This distinction is important:

        Attention:
            token E collects information from A, B, C, D, E

        FFN:
            transforms E’s resulting feature vector
            without directly reading the other positions

* What is a decoder-only Transformer? (used by GPT, Llama, Qwen, Mistral, Gemma)

    A decoder-only Transformer is a model designed to predict the next token from the tokens preceding it.

    Given: A B C D E

    it models: P(F | A B C D E)

    A decoder-only model uses causal self-attention.
    "Causal" here means that a token cannot inspect future tokens:

        ┌───────┬─────────────┐
        │ Token │ May inspect │
        ├───────┼─────────────┤
        │ A     │ A           │
        ├───────┼─────────────┤
        │ B     │ A B         │
        ├───────┼─────────────┤
        │ C     │ A B C       │
        ├───────┼─────────────┤
        │ D     │ A B C D     │
        ├───────┼─────────────┤
        │ E     │ A B C D E   │
        └───────┴─────────────┘

    The original Transformer had two separate parts:

        >       source text → encoder → internal representation
        >                                   ↓
        >       generated text ← decoder ← representation

    GPT-style language models remove the separate encoder. Everything is supplied as one prefix:

        >       Question + instructions + previous answer tokens
        >                              ↓
        >                       decoder-only model
        >                              ↓
        >                         next token

    In general:
        
        ┌─────────────────┬───────────────────────────────────────┬─────────────────────────────────────────┐
        │ Architecture    │ Visibility                            │ Typical purpose                         │
        ├─────────────────┼───────────────────────────────────────┼─────────────────────────────────────────┤
        │ Encoder-only    │ Both directions                       │ Understanding and embeddings            │
        ├─────────────────┼───────────────────────────────────────┼─────────────────────────────────────────┤
        │ Decoder-only    │ Current and previous tokens           │ Text generation                         │
        ├─────────────────┼───────────────────────────────────────┼─────────────────────────────────────────┤
        │ Encoder-decoder │ Encoder bidirectional, decoder causal │ Translation and sequence transformation │
        └─────────────────┴───────────────────────────────────────┴─────────────────────────────────────────┘

* What is attention?

    Attention allows one token to retrieve relevant information from other tokens.
    It is not a manually programmed grammatical rule.
    The model learns useful relationships from training data.

    Consider:
    The animal did not cross the street because it was tired.
    
    When processing it, the model needs information from animal.
    
    Attention provides a learned mechanism for forming connections such as:
    it → animal
    
* Query, key, and value

    For every token, the model creates three vectors:
    
        • Query (Q): what information is this token looking for?
        • Key (K): what kind of information does this token contain?
        • Value (V): what information should this token contribute?

    A useful analogy is a search system:

        • Query: search request
        • Key: document description
        • Value: document contents

    For tokens A B C D E, each one produces:

        A → q_A, k_A, v_A
        B → q_B, k_B, v_B
        C → q_C, k_C, v_C
        D → q_D, k_D, v_D
        E → q_E, k_E, v_E

    Comparing queries and keys

        Suppose the model is processing E.
        It compares q_E with every allowed key:

            q_E · k_A
            q_E · k_B
            q_E · k_C
            q_E · k_D
            q_E · k_E

        Larger dot product means that the corresponding token may be more relevant.

        The scores are scaled: score(E,j) = q_E · k_j / sqrt(head_dimension)

        The causal mask sets future-token scores to negative infinity.
        For E, there are no future prompt tokens. For C, the scores for D and E would be masked.

        Softmax converts the scores into weights:

            A: 0.05
            B: 0.10
            C: 0.15
            D: 0.60
            E: 0.10
        
        The attention result is a weighted sum of values:

            output_E =
                0.05 × v_A +
                0.10 × v_B +
                0.15 × v_C +
                0.60 × v_D +
                0.10 × v_E

* Activation functions

    Activation functions allow neural networks to represent nonlinear relationships.
    Without nonlinear activations, stacking linear matrix multiplications would still
    behave like one large linear transformation.

    SiLU is an activation function:
    
        SiLU(x) = x × sigmoid(x)

    SiLU is a smooth alternative to activations such as ReLU.

* RoPE: Rotary Positional Embedding

    Attention alone does not inherently know token order.

    Without positional information, these could appear too similar:
        dog bites man
        man bites dog

    RoPE encodes positions by rotating query and key vector components according to token position:

        A uses position 0
        B uses position 1
        C uses position 2
        ...

    The resulting query-key dot products contain information about relative token distance.

    KV-cache entries and newly decoded tokens must have correct positions because RoPE is applied using those positions.

    Incorrect KV positions can break generation, especially for models with more complex RoPE variants.

* Logits

    Logits are the model’s unnormalized vocabulary scores:
    
        token F : 14.2
        token G : 11.7
        token H :  8.4

    Greedy decoding selects the largest logit
    Stochastic sampling may transform them into probabilities using softmax.

* Softmax

    Softmax converts arbitrary scores into nonnegative values that sum to one

        logits        : 14.2   11.7   8.4
        probabilities : 0.92   0.075  0.005

    Appears in two places:

        1. attention scores become attention weights;
        2. vocabulary logits become token probabilities.

* KV cache

    The KV cache stores keys and values from previous token positions at every attention layer.

    When generating F after A B C D E, the model reuses cached:

        K/V for A B C D E

    It only calculates the new vectors for F.

* A compact mental model

    Attention   =   tokens communicate with previous tokens
    FFN         =   each token transforms its own features
    KV cache    =   remember previous attention keys and values
    LM head     =   turn the final token representation into vocabulary scores
    Sampler     =   choose the next token from those scores

```
4.
```
    * Let the cheap draft model guess several future tokens.
    * Run the expensive target model once over those guesses.
    * Keep guesses that exactly match what the target model would have produced.
    * At the first mismatch, discard the remaining guesses and emit the target’s token instead.
```
5.
```
* Remember the next-token shift:

    logits after E predict the token after E
    logits after F predict the token after F
    logits after G predict the token after G
    logits after H predict the token after H
```
6.
```
    The target KV cache currently contains:

        A B C D
    
    Spectre constructs:
    
        target batch = [E, F, G, H]
    
    The complete assumed sequence is therefore:
    
    cached prefix: A B C D
    new batch:             E F G H
    
    The target model performs one forward pass over the complete batch.
    
    Because attention is causal, each position sees only the history preceding it:
    
    ┌────────────────┬───────────────────────┬───────────────────────┐
    │ Batch position │ Token being processed │ History visible there │
    ├────────────────┼───────────────────────┼───────────────────────┤
    │ 0              │ E                     │ A B C D E             │
    ├────────────────┼───────────────────────┼───────────────────────┤
    │ 1              │ F                     │ A B C D E F           │
    ├────────────────┼───────────────────────┼───────────────────────┤
    │ 2              │ G                     │ A B C D E F G         │
    ├────────────────┼───────────────────────┼───────────────────────┤
    │ 3              │ H                     │ A B C D E F G H       │
    └────────────────┴───────────────────────┴───────────────────────┘

    The model produces one logits row after each batch token.

    Therefore:

    ┌────────────┬─────────────────┬─────────────────────────────────┐
    │ Logits row │ Comes after     │ Used for                        │
    ├────────────┼─────────────────┼─────────────────────────────────┤
    │ 0          │ A B C D E       │ verify proposed F               │
    ├────────────┼─────────────────┼─────────────────────────────────┤
    │ 1          │ A B C D E F     │ verify proposed G               │
    ├────────────┼─────────────────┼─────────────────────────────────┤
    │ 2          │ A B C D E F G   │ verify proposed H               │
    ├────────────┼─────────────────┼─────────────────────────────────┤
    │ 3          │ A B C D E F G H │ generate a possible bonus token │
    └────────────┴─────────────────┴─────────────────────────────────┘

    The main performance opportunity → one target call can potentially produce several output tokens.

```
7. Verification happens from left to right and stops at the first mismatch.
8.
```
    sample_and_accept()

        for each draft position:
          select target token from corresponding target logits;
          if target token matches draft token:
              continue;
          else:
              return immediately;

    For our draft:

        draft[0] = F
        draft[1] = G
        draft[2] = H
    
    the function does:
    
        target row 0 versus F
        target row 1 versus G
        target row 2 versus H
    
    There are three important outcomes.

    [A] rejection at the first token
        The draft proposes:

            F G H
            
        But the target says:
            
            row 0 → X
            
        So:
            
            target X != draft F
            
        The verifier stops immediately.
            
        The output of this round is:
            
            X
            
        The other draft tokens are unusable:
            
            F rejected
            G invalid because it assumed F
            H invalid because it assumed F and G

        Thus:
            accepted draft tokens: none
            target correction:     X

        The ordinary target-only AR model would also have generated X after A B C D E, so correctness is preserved.

    [B] some drafts match, then rejection
        Suppose:

            draft proposal: F G H
            target choices: F G X

        Verification proceeds:

            position 0:
                draft = F
                target = F
                match

            position 1:
                draft = G
                target = G
                match

            position 2:
                draft = H
                target = X
                MISMATCH!!!

        The output of this round is:

            F G X

        Thus:
            F = accepted draft
            G = accepted draft
            X = target correction replacing H
        
    [C] every draft matches
        Suppose:

            draft proposal: F G H
            target choices: F G H

        All three proposals are accepted.

        The target forward pass also produced row 3:

            logits after H

        Spectre can use that row to select the next target token:

            row 3 → I

        The complete output of the round becomes:

            F G H I

        F = accepted draft
        G = accepted draft
        H = accepted draft
        I = target bonus token

        `I` is called a bonus because its logits were already calculated by the verification
        forward pass. No additional target forward pass is needed to select it.
```
9.
```
    Why `accepted.size() - 1` ?

        The return vector always has this structure:

            zero or more accepted drafts
                        +
            one final target token

        The final target token is either:

            a correction

                or:

            a bonus
```
10.
```
The draft model sequentially proposes k tokens.
The target model evaluates the pending token and all k proposals in one batched forward pass.
The program compares each proposal with the target’s corresponding choice.
It retains the matching prefix, and at the first mismatch uses the target’s choice instead.
```
11.
```
    Assume:
    
        accepted history    :   A B C D E
        draft length        :       3
    
    Draft-model work

        the ordinary draft model performs approximately three sequential one-token forward passes.

        decode E → propose F
        decode F → propose G
        decode G → propose H

    Target-model work

        it does NOT perform three separate target calls

        It receives one batch:

            [E, F, G, H]

        and performs one batched forward pass:

        That one call produces four logits rows:

            after E → target choice for F's position
            after F → target choice for G's position
            after G → target choice for H's position
            after H → possible bonus token

    So the distinction is:

        Draft:
            several sequential calls to a cheap model

        Target:
            one batched call evaluating several positions
        
    The performance advantage comes from batching
```
12.
```
    A simplified cost comparison is:
    
    ordinary AR:
        4 sequential expensive target calls

    speculation, if all three drafts match:
        3 sequential cheap draft calls
        + 1 batched expensive target call
```
13.
```
    The speculative round can emit:
    
        F G H + one bonus token

```
14. If proposals are frequently rejected, some batched target computation is wasted and the benefit decreases.
15.
```
    There are two executions being compared.

    Ordinary target-only AR

        decode E
        target selects F
        decode F
        target selects G
        decode G
        target selects X

    Batched speculative verification

        target evaluates [E, F, G, H] once
        row after E selects F
        row after F selects G
        row after G selects X
```
16. Why do the batched rows match ordinary AR? Because of causal attention.
17.
```
    This resembles speculative CPU execution:

        predict a path
        execute work on that path
        retire valid results in order
        discard work after a misprediction
```
18.
```
    The general correctness argument
    
    Start with the same accepted prefix for ordinary AR and speculation.
    
    For every draft proposal:
    
        1. If it equals the target’s next token, appending it produces the same prefix as ordinary target-only generation.
        2. At the first mismatch, emit the target’s next token instead, again producing the same prefix as ordinary target-only
           generation.
        3. Start the next round from that identical prefix.
    
    By repeating this argument, greedy speculative decoding produces the same token sequence as greedy target-only AR.
```
19.
```
    There are two related but distinct guarantees.

    Greedy exact-match speculation

        draft token == target argmax → accept
        otherwise → emit target argmax

        Guarantee:

            the speculative token sequence is identical to target-only greedy decoding

    Canonical stochastic speculative sampling

        Leviathan’s stochastic algorithm works with:

            p(x) = target probability
            q(x) = draft probability

        A proposal is accepted with probability:

            min(1, p(x) / q(x))

        If rejected, the replacement is sampled from a corrected residual distribution based on:

            max(0, p(x) - q(x))

        Its guarantee is different:

            the speculative output has the same probability distribution as target-only sampling

        It does not necessarily produce the exact same token sequence as a separate target-only
        run with an informally “same” seed, because the two procedures consume randomness differently.
```
20. speculative decoding performs roughly fixed work per round, while acceptance determines how many useful tokens that work yields.
21.
```
    Modern LLMs use an algorithm called Byte Pair Encoding (BPE).

    This splits text into tiny fragments (prefixes, suffixes, syllables, and single letters)
    so that the model can handle any sequence of characters ever written, including typos,
    made-up words, code, and slurs, without needing an infinitely large dictionary.
```

