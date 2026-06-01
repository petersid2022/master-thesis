## Table of Contents

<!--toc:start-->
- [Table of Contents](#table-of-contents)
- [Intro](#intro)
  - [Wikipedia entry for Speculative Decoding](#wikipedia-entry-for-speculative-decoding)
  - [How Speculative Sampling works in general](#how-speculative-sampling-works-in-general)
- [Model naming conventions](#model-naming-conventions)
  - [The five-slot decoder](#the-five-slot-decoder)
  - [Parameter count](#parameter-count)
  - [Fine-tuning suffixes](#fine-tuning-suffixes)
  - [Quantization (GGUF / llama.cpp)](#quantization-gguf--llamacpp)
  - [Other ecosystems](#other-ecosystems)
  - [Practical takeaways for picking SD pairs](#practical-takeaways-for-picking-sd-pairs)
- [Quantization](#quantization)
  - [What it is](#what-it-is)
  - [Why we do it - memory bandwidth is the bottleneck](#why-we-do-it--memory-bandwidth-is-the-bottleneck)
  - [Does it hurt quality?](#does-it-hurt-quality)
  - [Do other inference engines use it?](#do-other-inference-engines-use-it)
  - [Pareto-frontier tool, not a necessary evil](#pareto-frontier-tool-not-a-necessary-evil)
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
- [A Comprehensive Analysis of SD Implementations and Techniques](#a-comprehensive-analysis-of-sd-implementations-and-techniques)
- [Quality Evaluation](#quality-evaluation)
  - [Why naive perplexity is a trap](#why-naive-perplexity-is-a-trap)
  - [The acceptance rule and why it preserves the distribution](#the-acceptance-rule-and-why-it-preserves-the-distribution)
  - [The TV-distance bound on acceptance](#the-tv-distance-bound-on-acceptance)
  - [Perplexity as a sanity check](#perplexity-as-a-sanity-check)
  - [Empirical metrics we actually report](#empirical-metrics-we-actually-report)
  - [Data convention](#data-convention)
  - [How the literature evaluates "quality is preserved"](#how-the-literature-evaluates-quality-is-preserved)
  - [When perplexity becomes a real lever (lossy variants)](#when-perplexity-becomes-a-real-lever-lossy-variants)
- [Resources](#resources)
- [Footnote](#footnote)
<!--toc:end-->

## Intro
> Η εργασία αυτή μελετά την τεχνική speculative sampling, με σκοπό την επιλογή ενός μικρού γλωσσικού μοντέλου (SLM) αντί ενός μεγάλου (LLM), όταν το SLM μπορεί να αποδώσει εξίσου καλά, ενεργοποιώντας το LLM μόνο όταν κρίνεται απαραίτητο.
> 
> Θα διεξαχθούν πειράματα για τη σύγκριση διαφορετικών προσεγγίσεων, με μετρήσεις χρόνου απόκρισης και κατανάλωσης ενέργειας σε σχέση με τα αρχικά μεγάλα μοντέλα.
> 
> Ιδιαίτερη έμφαση θα δοθεί (α) στην επιλογή των κατάλληλων μοντέλων και (β) στην επίδρασή τους στην ακρίβεια και στην αποδοτικότητα των αποτελεσμάτων.

> This research addresses the inference scheduling problem, drawing from compiler optimization theory (e.g., branch prediction, profile-guided optimization, instruction scheduling) to minimize both latency and energy usage.
> 
> These parallels arise from the fact that LLM inference can be viewed as a computational pipeline, similar to how compilers treat code as a sequence or graph to optimize ([Abstract Syntax Tree](https://en.wikipedia.org/wiki/abstract_syntax_tree)).
> 
> Previous research shows speedups of 2-3x from speculative decoding alone, and layering compiler-inspired optimizations could yield even greater improvements, especially for edge cases like long sequences or low-acceptance-rate drafts.

* Speculative decoding provides speedups when the draft model is fast and accurate enough that most speculative tokens are accepted. 
* On GPU this works well because the cost of evaluating multiple tokens in parallel is low.
* On CPU, draft evaluation is expensive and small differences between model logits cause many rejections.
* Speculative decoding essentially in LLMs **is** speculative execution in CPUs.

### Wikipedia entry for Speculative Decoding
> [!TIP]
>
> [Source](https://en.wikipedia.org/wiki/Transformer_(deep_learning)#Speculative_decoding)

* Speculative decoding is a method to accelerate token decoding.
* Similarly to speculative execution in CPUs, future tokens are computed quickly, then verified.
* If the quickly computed tokens are incorrect, they are discarded and computed slowly.
* The key factor in speculative decoding is that a transformer decoder can verify faster than it can decode, in the following sense.

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
* The small model proposes several next tokens (a "draft" sequence).
* The large model verifies them in parallel.
* If the proposals are likely under the large model, they are accepted, saving time because the large model doesn't need to compute them token-by-token

In other words:
* The small LLM generates a series of sequential tokens, after which the large LLM evaluates all of these in a single pass.
* For each token, if the probability in the large LLM is higher than that of the small one, it is accepted directly (therefore not affecting the large LLM's statistics). If the probability is lower, the likelihood of acceptance is proportional to the difference in probabilities. This method makes it likely that the token will not be accepted, in which case the computation is wasted. If the small model performs well, we gain a speedup without changing the output, but if it performs poorly, we waste significant compute resources, slowing down the process. See: [Branch predictor](https://en.wikipedia.org/wiki/Branch_predictor)
* Speculative decoding variants allow tokens to exit the model early if the model is confident, similar to compiler optimizations like function inlining or dead code elimination, which skip unnecessary computations.
* Speculative sampling often builds a tree of possible token paths during text generation. We can apply compiler-style graph transformations (e.g., pruning redundant branches via static analysis or common sub expression elimination on token probabilities) to optimize the tree exploration.
* Just as compilers schedule operations for parallel execution, speculative sampling could "schedule" token generation to prioritize high-probability paths, inspired by compiler techniques for out-of-order execution.

These are soft-max probabilities over the vocabulary, the same kind of probabilities used during standard next-token sampling in an LLM.

## Model naming conventions

A reference for reading model filenames in the wild. Filenames like
`Qwen3-30B-A3B-Instruct-Q4_K_M.gguf` compress 4-5 orthogonal things into one
string; this section decodes each slot.

### The five-slot decoder

| Slot | Example | What it means |
|---|---|---|
| Family / version | `Qwen3` | Model family and major version |
| **Size** | `30B-A3B` | **Parameter counts** (see below) |
| Fine-tuning | `Instruct` | Training objective |
| **Quantization** | `Q4_K_M` | **How weights are stored** (see below) |
| Format | `.gguf` | File format (here, llama.cpp's GGUF) |

Decoding the local models in this repo:

```
Qwen2.5-Coder-3B-Instruct-IQ2_M.gguf
↑       ↑    ↑   ↑       ↑    ↑
fam     v    spec size  tune  quant (i-quant, 2-bit medium)

Nemotron-3-Nano-4B-BF16.gguf
↑        ↑ ↑    ↑  ↑
fam      v size size precision (no quant - full bf16)
```

`Nano` here is NVIDIA's size-class label (Nano < Mini < Small < Medium < Large)
rather than a separate slot.

### Parameter count

**Dense models** (the simple case):

```
8B    = 8 billion parameters, every one used per token
70B   = 70 billion parameters, every one used per token
1.5B  = 1.5 billion parameters
```

Memory ≈ `params × bytes_per_param`. A 70B model at FP16 is ~140 GB.

**MoE models - the `XB-AYB` form.** `X` is total, `A` is active per token:

```
30B-A3B            = 30 B total parameters, ~3 B active per token
Mixtral 8x7B       ≈ 47 B total, ~13 B active (top-2 of 8 experts)
DeepSeek-V3 671B-A37B = 671 B total, 37 B active
```

A Mixture-of-Experts layer holds N small "expert" sub-networks plus a router.
The router picks the top-k experts per token, and only those experts run. So
you pay **memory** for all 30 B parameters but **compute** only for ~3 B per token.

Why this matters for the thesis:

- **Throughput is bound by active params**, not total. A 30B-A3B model decodes
  nearly as fast as a dense 3B model when everything fits in VRAM.
- **Speculative-decoding economics change.** The target/draft size ratio that
  gives a good speedup is usually `draft ≈ target / 10..20`. For MoE,
  compare *active* params, not totals. A `30B-A3B` target effectively asks a
  3B draft to keep up - harder than drafting for a dense 30B model.
- **Memory pressure is still high.** You need ~30 B worth of VRAM regardless,
  unless you use expert offloading.

Real MoE models you'll encounter:

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
| (none) or `-Base` | Pretrained on text completion only - no instruction following |
| `-Instruct` | SFT'd on instruction-following pairs |
| `-Chat` | Same idea, often with multi-turn formatting |
| `-Coder` | Continued pre-training on code |
| `-Math`, `-Reasoning`, `-Thinking` | Specialized variants |
| `-DPO`, `-RLHF`, `-RLAIF` | Indicates the preference-training method used |
| `-it` | Some labs use `-it` for "instruction-tuned" (Gemma) |

For SD experiments, **the draft and target should share fine-tuning style** -
pair `-Instruct` with `-Instruct`. Mixing base with chat models lowers
acceptance rates because their distributions on chat-formatted input diverge.

### Quantization (GGUF / llama.cpp)

**Bit-width prefix:**

```
F32 / F16 / BF16  = full or half precision floats
Q8_0 / Q6_K       = 8 or 6 bits per weight
Q5_K / Q4_K       = 5 or 4 bits
Q3_K / Q2_K       = 3 or 2 bits
IQ4_XS / IQ3_M    = importance-matrix quants (i-quants)
IQ2_XXS / IQ1_S   = ultra-low-bit (1.5-2.5 bits effective)
```

**Suffix tier.** `_S`, `_M`, `_L` after a K-quant means
**Small / Medium / Large mixed-precision variant**:

```
Q4_K_S = mostly 4-bit; smallest size, lowest quality
Q4_K_M = 4-bit with critical layers (attention, output) bumped higher - best-balanced
Q4_K_L = even more critical layers kept higher
```

`M` is usually the sweet spot. `S` saves a bit of disk/VRAM at noticeable
quality loss; `L` is rarely worth it.

**`Q_K` vs `IQ_`:**

- **`Q_K`** ("K-quants"): groups of weights with shared scale/min values.
  Fast on most hardware, simple. Released ~mid-2023.
- **`IQ_`** ("I-quants" / importance-matrix quants): uses a calibration dataset
  (the "imatrix") to assign *which* weights need more precision.
  **Same bit budget → higher quality**, slightly slower on some CPUs/GPUs.

Rule of thumb: at 4 bits and below prefer `IQ` when available;
at 5+ bits, `Q_K_M` is fine.

**Legacy / non-K formats:**

```
Q4_0, Q4_1, Q5_0, Q5_1   = older single-precision-per-block formats
Q8_0                      = 8-bit, used as the high-fidelity reference; "essentially lossless"
```

`Q8_0` deserves a callout: it's effectively a lossless 50% size reduction from
FP16. **Almost always the right choice for a draft model** - fast, tiny, and the
draft's slight quality loss doesn't matter because the target verifies everything.

### Other ecosystems

| Format | Where |
|---|---|
| `GPTQ` | HuggingFace, AutoGPTQ, ExLlama - GPU-focused 4-bit |
| `AWQ` | activation-aware quantization, GPU |
| `EXL2` | ExLlamaV2; per-layer mixed bit-width |
| `MLX` | Apple Silicon |
| `safetensors` (uncompressed) | the FP16/BF16 source weights from HF |

For llama.cpp work, only `GGUF` + the `Q_K`/`IQ` family is relevant.

You'll occasionally see context-related extras in filenames:

```
-128k      context window in tokens (e.g. Qwen2.5-Coder-7B-Instruct-128k)
-YaRN      rope-scaling extension method (rare in filenames)
-RAG       fine-tuned for retrieval-augmented generation
-Function  fine-tuned for tool-use / function calling
```

### Practical takeaways for picking SD pairs

Three things determine speculative-decoding efficiency:

1. **Same family + same fine-tune.** Vocabulary and distribution must match;
   the spectre binary checks vocab equivalence at startup.
2. **Active-param ratio.** Aim for `draft active ≈ target active ÷ 8..20`.
   - For dense `26B`: draft 1.5B-3B.
   - For MoE `30B-A3B`: ideal draft is 0.2B-0.4B (rarely available) - which is
     why **SD on MoE is harder, not easier**.
3. **Quant pair.** Target `Q5_K_M` or higher (don't waste verification budget
   on a noisy target); draft `Q4_K_M` or `Q8_0`.

The Nemotron BF16 (target) + Nemotron Q8_0 (draft) pair used in this repo is a
**self-speculative decoding** setup: same model, different precisions. The Q8
version is ~50% faster to evaluate but produces nearly identical distributions,
so acceptance is very high. It's a useful baseline but not what most papers do
(they use a separately-trained smaller model).

## Quantization

The "Q4_K_M / IQ2_M / BF16" suffixes in model filenames are *quantization
formats*; this section explains what quantization actually does and why it
matters for the speculative-decoding story.

Quantization and Speculative Decoding both attack the same bottleneck (memory bandwidth) from different angles.

### What it is

**Storing each weight in fewer bits than the precision it was trained at.**

Models are trained in `float32` (32 bits = 4 bytes per weight) or, more often
today, `bfloat16` (16 bits = 2 bytes). Quantization converts those weights
into a lower-bit representation - typically 8, 4, or even 2 bits - at the cost
of some precision loss.

The simplest scheme (uniform integer quantization):

```
original  weight   ∈ [-W_max, +W_max]                   (float, full range)
quantized weight   = round(weight × (127 / W_max))      → int8 ∈ [-128, +127]
restored  weight   ≈ quantized / (127 / W_max)          (back to float, lossy)
```

You store the `int8` value and the scaling factor `W_max/127`. At inference
time, you decode back to float on the fly to do the matrix multiply.

Real quantization is much fancier than this - modern formats (`Q4_K_M`,
`IQ3_M`, etc.) use:

- **Block-wise scaling**: each group of ~32 weights gets its own scale, so a
  few outliers don't blow up the whole tensor.
- **Mixed precision**: critical layers (attention output projection, embedding)
  are kept at higher precision than feedforward.
- **Importance weighting** (the "I" in I-quants): a calibration dataset
  identifies which weights matter most and assigns them more bits.

But the core idea is unchanged: trade precision for storage.

### Why we do it - memory bandwidth is the bottleneck

The widely-quoted reason is "models don't fit in VRAM." That's true but it's
the symptom, not the cause. The real reason is more important:

**LLM inference is memory-bandwidth-bound, not compute-bound.**

When you generate one token from a 7 B model on an A100:

| Operation | Numbers |
|---|---|
| Weights you must read from VRAM → cache | ~7 B × 2 bytes = **14 GB** |
| Math you must perform | ~7 B × 2 = **14 G** FLOPs |
| A100 memory bandwidth | ~2 TB/s |
| A100 compute throughput (FP16) | ~312 TFLOP/s |
| Time spent moving weights | 14 GB / 2 TB/s = **7 ms** |
| Time spent computing | 14 G / 312 T = **0.045 ms** |

The compute finishes in 45 μs. The memory transfer takes 7 ms. **The GPU is
idle ~99% of the time waiting on memory.** This is the central fact of LLM
inference, and it dictates everything downstream.

Now apply quantization to the same 7 B model:

| Quant | Bytes/weight | Read time | Implied tok/s |
|---|---:|---:|---:|
| F16 (baseline) | 2.0 | 7.0 ms | ~143 |
| Q8_0 | 1.0 | 3.5 ms | ~286 |
| Q4_K_M | 0.55 | 1.9 ms | ~526 |

You don't speed up the math - the GPU still computes in higher precision
internally after dequantizing. You speed up the **memory transfer**, and since
memory is the bottleneck, you get a near-linear speedup. Q4_K_M is ~3.7× faster
than F16 simply because each weight is 3.7× smaller.

### Does it hurt quality?

Yes, but how much depends on how aggressive. The standard way to measure this
is the perplexity gap between the quantized model and its FP16 reference on a
held-out corpus. Rough empirical curve (varies by model; the shape is universal):

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

Above ~4 bits the quality loss is **smaller than what a different random seed
gives you** on most benchmarks. Below ~3 bits, the loss becomes user-visible
and chains of reasoning start to break.

There's also a model-size interaction: **bigger models tolerate quantization
better.** A 70B at 4 bits often outperforms a 13B at 16 bits in both quality
and speed and uses similar memory. This is the standard "go bigger, then
quantize" advice for local inference.

### Do other inference engines use it?

Universally. It's table stakes:

| Engine | Quantization support |
|---|---|
| **llama.cpp / GGUF** | K-quants and I-quants (this thesis) |
| **vLLM** (Berkeley/UC, dominant server) | AWQ, GPTQ, FP8, INT8 KV cache, FP8 KV cache |
| **TGI** (HuggingFace) | bitsandbytes (8/4-bit), GPTQ, AWQ, EETQ |
| **TensorRT-LLM** (NVIDIA, production) | FP8 (heavily), INT8, INT4 weight-only |
| **ExLlamaV2** | EXL2 - bespoke per-layer mixed bit-width |
| **MLX** (Apple) | INT4, INT8 quantization for Apple Silicon |
| **MLC-LLM** (mobile) | INT3, INT4 |
| **Ollama, LM Studio, Jan** | all wrap llama.cpp |

The hosted-API providers (Anthropic, OpenAI, Google, DeepInfra, Together,
Fireworks…) also quantize internally; "latency-vs-cost tiers" almost certainly
correspond to different quantization levels of the same backbone.

The only place you *don't* see quantization is research training (you train at
BF16/FP16 because gradient noise blows up at lower precision - though FP8
training is now production-grade at large scale).

### Pareto-frontier tool, not a necessary evil

The honest framing: it's not "good" or "evil" - it's a Pareto-frontier knob.

- **At the right operating points it's free.** `Q8_0` is a 50 % memory and
  bandwidth saving for ~0 % quality loss. There is no good reason *not* to use
  it. `Q6_K` and `Q5_K_M` are nearly the same story.
- **At aggressive operating points it's a trade.** `Q4_K_M` costs ~1-2 % on
  standard benchmarks but cuts inference time ~3-4×. For an interactive
  chatbot wildly worth it; for a medical diagnostic system maybe not.
- **At extreme operating points it's a necessary evil.** `IQ2`, `IQ1` - you
  accept measurable quality loss because the alternative is "the model doesn't
  load." This is what makes a 70 B model run on a 16 GB consumer GPU at all.

Same logic as JPEG vs PNG: nobody calls JPEG "evil" because it's lossy; it's
just the right answer for most use cases and the wrong answer for some.

### How quantization interacts with speculative decoding

Three threads worth pulling on, all relevant to the thesis:

1. **Quantization and SD attack the same bottleneck from different angles.**
   Quant reduces *bytes per weight*. SD reduces *token-equivalent weight-reads
   per output token* - by verifying N drafted tokens in parallel against the
   target, the target reads its weights once and outputs ≤ N tokens. The two
   multiply: a 4× speedup from Q4 × a 2× speedup from SD ≈ 8× faster than an
   F16 AR baseline. `quality-speed-vs-accept.png` measures the SD half; the
   quant choice sets the baseline against which everything is plotted.

2. **Self-speculative decoding (the Nemotron BF16 + Q8 setup) is essentially
   quant-as-draft.** The Q8 draft is a "noisy approximation of the target" -
   exactly the regime where SD shines, because the noise is small enough that
   acceptance stays high. There's a cluster of recent papers (LayerSkip,
   SpecDec via Self-Distillation) formalising this as its own SD family.

3. **Quant choice affects acceptance rate even within the same model family.**
   If target and draft are *both* `Q4_K_M`, the draft is closer to the target
   than if the target were FP16 - both made the same quantization "errors", so
   their distributions are more correlated. This is a deliberately exploitable
   design choice, not a bug.

## Concrete Directions
### Adaptive Draft Length for llama.cpp

llama.cpp uses **fixed** `--draft-max` and `--draft-min` parameters. The optimal draft length depends on context difficulty.
> Easy tokens (boilerplate, code patterns) should draft long, hard tokens (reasoning, rare words) should draft short or not at all.

* Implement an entropy-based or lightweight prediction-head approach (inspired by [SpecDec++](https://arxiv.org/abs/2405.19715) or [AdaEDL](https://proceedings.mlr.press/v262/agrawal24a.html)) directly in llama.cpp's speculative pipeline. AdaEDL is particularly attractive because it's **training-free**

* It uses an entropy-based lower bound on acceptance probability to decide when to stop drafting. This means we don't need to train anything, just instrument the existing draft loop.

* 10-57% improvement over static draft lengths according to the AdaEDL paper.

* This is **profile-guided optimization**, using runtime statistics (token entropy) to make scheduling decisions, exactly like a PGO-enabled compiler uses branch frequency data.

### Tree-Structured Drafting in llama.cpp

llama.cpp currently does **linear** speculative drafting (one sequence of draft tokens). Research shows that **tree-structured** drafting (where you branch at uncertain positions) dramatically improves acceptance rates. [TALON](https://arxiv.org/abs/2601.07353) achieves up to 5.16x speedup by adaptively constructing deep-and-narrow trees for deterministic contexts and shallow-and-wide trees for uncertain ones.

* Implement a basic tree-structured draft in llama.cpp with adaptive branching. The key challenge is implementing **tree attention** for verification efficiently in GGML (llama.cpp's tensor library).

* This is **speculative execution with branch fan-out**. Instead of predicting one path, we execute multiple paths and discard the wrong ones, just like a superscalar processor.

### Heterogeneous CPU/GPU Scheduling for Edge

There's a very recent paper (February 2026) ([Compiler-Assisted Speculative Sampling for Accelerated LLM Inference on Heterogeneous Edge Devices](https://arxiv.org/abs/2602.08060)) that does exactly what the thesis introduction describes: using compiler-style cost models to partition draft/target model execution across CPU and GPU on edge SoCs. They got 1.68x speedup on ARM Cortex-A + Mali GPU.

* Build an analytical cost model for llama.cpp that decides at runtime whether to use speculative decoding at all, and if so, how to partition work across available compute (CPU threads, GPU if available, even NPU on newer hardware).
> llama.cpp already supports hybrid CPU/GPU execution via layer offloading -> extend this with speculation-aware scheduling.

### Relative Papers

- [SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths](https://arxiv.org/abs/2405.19715)
- [TALON: Confidence-Aware Speculative Decoding with Adaptive Token Trees](https://arxiv.org/abs/2601.07353)
- [Compiler-Assisted Speculative Sampling on Heterogeneous Edge Devices](https://arxiv.org/abs/2602.08060)
- [Efficient Speculative Decoding for Llama at Scale](https://arxiv.org/abs/2508.08192)
- [Speculative Speculative Decoding (Saguaro)](https://arxiv.org/abs/2603.03251)
- Eagle-3 Speculative Decoding the SOTA algorithm for speculative decoding
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
1. Speculative decoding : an optimization technique for inference that makes educated guesses about future tokens while generating the current token.
2. All within a single forward pass. No [backpatching](https://www.geeksforgeeks.org/compiler-design/backpatching-in-compiler-design/)
3. Based on the premise that the model is powerful enough to predict multiple tokens in a single forward pass.
4. It incorporates a verification mechanism to ensure the correctness of these speculated tokens
5. Thereby guaranteeing that the overall output of speculative decoding is identical to that of vanilla decoding.

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
    - Reduce the inference cost for *all* inputs equally
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
5. Reward-guided acceptance:
    * retains valuable partial solutions,
    * reduces unnecessary queries,
    * can even surpass the large model's performance

### Reward-Guided Speculative Decoding (RSD)
- Balances efficiency and accuracy by integrating computationally lightweight "draft" evaluations with reward-driven refinements from a more capable "target" model
- Adaptively select high-value draft outputs rather than discarding mismatched tokens outright

## Speculative Speculative Decoding (SSD)
> [!TIP]
>
> Speculative Decoding relies on a sequential dependence between speculation and verification, so we parallelize these operations.
>
> [Source](#resources-speculative-speculative-decoding)

### Key takeaways
1. While a verification is ongoing, the draft model *predicts* likely verification outcomes and prepares speculations pre-emptively for them.
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

> **This is not machine learning.**
>
> We are not training a neural network. We are collecting execution profiles and using them to parameterize a policy
>
> Exactly what GCC's `-fprofile-generate` / `-fprofile-use` does. It's a compiler technique applied to a new domain.

1. Instrumentation
    * Run the draft+target model pair on representative inputs. At each speculative step, log:
        - The draft model's entropy/confidence at each position
        - Whether the token was accepted or rejected
        - The context features (token type, position in sequence, preceding pattern)

2. Profile analysis
    * From the collected data:
        - build a lightweight predictor that maps context features to expected acceptance probability.
    * This is analogous to a compiler's branch prediction profile (at this call site, the branch is taken 94% of the time)

3. Optimized policy generation
    * Use the profile to generate a speculation policy:
        - High expected acceptance → draft long (like a predicted-taken branch)
        - Low expected acceptance → draft short or skip (like a predicted-not-taken branch)
        - Very uncertain → branch (tree-structured draft, like a superscalar processor exploring both paths)

## Trace-Based Speculation
### Lightweight JIT for token prediction that runs alongside the draft model

> **trace-based compilation for speculative decoding.**

In JIT compilers (LuaJIT, V8, PyPy), the runtime records "hot traces" (frequently executed paths through the program) and compiles them to optimized native code. Cold paths fall back to the interpreter.

1. Apply this to speculative decoding:
    * Record frequently occurring token generation patterns (e.g., "after generating `def`, the next tokens are almost always `function_name(args):`")
    * For these "hot traces," use aggressive speculation (long drafts, high confidence)
    * For "cold" / unfamiliar patterns, fall back to conservative speculation or no speculation
    * The "traces" could be stored as n-gram patterns, trie structures, or even finite automata

## A Comprehensive Analysis of SD Implementations and Techniques
* Speculative decoding is an inference-time optimization method to accelerate token decoding.

* It does that by generating draft tokens quickly and then verifying them with the target model in a single batch, this approach can achieve substantial speedups when the draft predictions are frequently correct.

* Provides speedups when the draft model is fast and accurate enough that most speculative tokens are accepted. 
* On GPU this works well because the cost of evaluating multiple tokens in parallel is low.
* On CPU, draft evaluation is expensive and small differences between model logits cause many rejections.

* Several implementations (Note: an implementation with draft model can be mixed with an implementation without draft model)
    * Draft Model (most popular)
    * n-gram {Cache, Map, Mod} (uses pattern matching to guess future tokens based on the prompt or previous output)
    * EAGLE-{1,2,3} (the fastest, attaches a separate module to the target model's internal layers)
    * Speculative Speculative Decoding (the SOTA algorithm, while a verification is ongoing, the draft model predicts likely verification outcomes and prepares speculations pre-emptively for them)

* High level overview of my implementation
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
> How do we know that the text speculative decoding produces is "as good as" what the target model would have produced on its own?
> Short answer for vanilla SD: by construction, it is *identical in distribution*. Long answer below.

### Why naive perplexity is a trap

The first instinct is: "run target alone, run SD, compute perplexity of each output, compare." This is wrong for two reasons:

1. **They sample different sequences.** Perplexity of a *generated* sequence under the model that generated it is a self-evaluation, not a comparison. Two different valid completions of the same prompt can have very different per-token perplexities and both be high-quality.
2. **For vanilla SD the comparison is theoretically vacuous.** The output of correct SD is a *sample* from the target - exactly. So `PPL(SD output | target)` and `PPL(target output | target)` have the same distribution. Reporting "they match" doesn't prove much; reporting "they differ" proves your SD implementation is buggy.

So perplexity is useful here as a **sanity check** (the implementation is correct) and as a **drift metric** (for lossy variants), not as a quality comparison between losslessly-equivalent algorithms.

### The acceptance rule and why it preserves the distribution

Let `p(x | ctx)` be the target distribution and `q(x | ctx)` the draft distribution at the current position. Draft proposes `x ~ q`. Speculative sampling accepts `x` with probability

```
α(x) = min(1, p(x) / q(x))
```

and, if rejected, resamples from the residual distribution

```
p̃(x) ∝ max(0, p(x) − q(x))     (normalized so it sums to 1)
```

**Theorem** (Leviathan et al. 2022; Chen et al. 2023): the marginal distribution of the accepted token equals `p(·|ctx)` exactly.

*Proof sketch.* Mass on a particular token `x*` after one step:
- accepted from draft: `q(x*) · min(1, p(x*)/q(x*)) = min(q(x*), p(x*))`
- contributed via residual after rejection: `P(reject) · p̃(x*)` where `P(reject) = 1 − Σₓ min(p(x), q(x))` and `p̃(x*) = max(0, p(x*) − q(x*)) / P(reject)`
- sum: `min(p(x*), q(x*)) + max(0, p(x*) − q(x*)) = p(x*)`.  ∎

Implication: **the output text from vanilla SD is statistically indistinguishable from target-only sampling.** Any "is the draft bonkers" question collapses to "is your implementation correct."

### The TV-distance bound on acceptance

The *only* dial that affects how much you accept is how close `q` is to `p`. Expected per-token acceptance probability:

```
E[α] = Σₓ q(x) · min(1, p(x)/q(x))
     = Σₓ min(p(x), q(x))
     = 1 − ½ Σₓ |p(x) − q(x)|
     = 1 − TV(p, q)
```

So `TV(p, q)` (total variation distance) is the **theoretical upper bound** on the marginal speedup you can get from any drafter. Equivalent statements with KL: `TV ≤ √(½ KL(p ‖ q))` (Pinsker), so high KL ⇒ low acceptance, but the reverse doesn't hold tightly.

For our n-gram speculators, `q` is essentially a Dirac delta on the predicted token, so:
- `TV(p, q) = 1 − p(x_predicted)`
- Acceptance probability per position = `p(x_predicted)` exactly.
- Acceptance count over many calls ≈ Bernoulli with parameter `E[p(x_predicted)]`.

This is why the empirical `A / G` ratio is a useful summary statistic for n-gram drafting specifically.

### Perplexity as a sanity check

Per-token target probability of accepted tokens is written by `RunRecorder::record_token`
in `spectre/src/main.cpp`, into `results/spectre/<run-id>/tokens.csv` (schema below).

Corpus PPL of a generated sequence:

```
PPL = exp( -mean( log p_target(x_t | x_<t) ) )
```

For vanilla SD this should equal (within sampling noise) the PPL of a target-only run
with the same prompt and seed. Disagreement = a bug in the acceptance code. The new
`--seed` flag pins the sampler so this comparison is reproducible across runs.

### Empirical metrics we actually report

Three families, in order of importance:

1. **Speed**
   - decode throughput `tok/s` (from `eval time` line in llama-server logs)
   - per-token latency `ms/token`
   - prefill `tok/s` (mostly invariant to SD, but useful to confirm)

2. **Efficiency** (drafter performance)
   - `acceptance rate = A / G` where `A` = accepted draft tokens, `G` = generated draft tokens
   - `block efficiency τ = E[accepted per call]` (Leviathan 2022)
   - `mean accepted length` per speculative round
   - per-position acceptance `P(i-th accepted) = p̂ⁱ` under independent-Bernoulli model

3. **Quality** (only matters for lossy variants; sanity check for lossless)
   - per-token `log p_target(accepted)` trace
   - corpus PPL of generated text
   - histogram of `p_target(x)` (where does difficulty hide?)
   - KL(p ‖ q) per position if both distributions are available

Scripts (see `spectre/scripts/README.md` for the full reference):
- `spectre/scripts/benchmark.sh` sweeps the spectre binary across `(seed × n_max)`
  configs and writes one `results/spectre/<run-id>/` directory per run.
  Default pair is Qwen2.5-1.5B drafting for Qwen2.5-3B (fast smoke). Override with
  `TGT_MODEL=...gemma-4-26B-A4B... DFT_MODEL=...gemma-4-E2B... ./spectre/scripts/benchmark.sh`
  for the realistic Gemma showcase.
- `spectre/scripts/quality_eval.py` reads every `results/spectre/<run-id>/` directory
  and produces 7 PNGs into `spectre/presentation/png/quality-*.png`.
- See `spectre/presentation/html/quality.html` for the live HTML view.

Reproduce end-to-end:
```bash
./spectre/scripts/benchmark.sh               # ~30-90s on a small GPU
python3 spectre/scripts/quality_eval.py      # regenerate figures
```

### Data convention

Every invocation of the spectre binary produces a self-contained, reproducible run directory:

```
results/spectre/<run-id>/
  meta.json       config snapshot + per-run aggregates + per-round summaries
  tokens.csv      per-accepted-token observations (one row per accepted token)
```

A run-id is auto-generated as `YYYYMMDD-HHMMSS_<mode>_seed<N>` or set explicitly via
`--run-id`. The run directory is written incrementally - `meta.json` is created at
startup with `"complete": false` and overwritten at the end with `"complete": true`
plus totals. `tokens.csv` is flushed per row. Both files survive `SIGTERM` / `SIGINT`.

**`meta.json` schema** (annotated):

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

`rejected_pos = -1` means all draft tokens in that call matched (the target then
contributed a bonus sample, counted in `n_bonus_samples`).

**`tokens.csv` schema:**

| column          | type    | meaning |
|---|---|---|
| `step`          | int     | global accepted-token index (0-based) |
| `call`          | int     | speculative call index (0-based); equals `step` in AR mode |
| `source`        | str     | `"draft"`, `"bonus"`, or `"ar"` |
| `pos_in_draft`  | int     | 0..k-1 if `source=="draft"`, else -1 |
| `token_id`      | int     | token id of the accepted token |
| `p_target`      | float   | target's probability mass on the accepted token |
| `p_draft`       | float   | draft's probability mass on the same token (empty for `bonus`/`ar`) |
| `logit`         | float   | target's logit value of the accepted token |
| `logprob`       | float   | `log p_target` |

Empty `p_draft` cells are NaN. The script `quality_eval.py` handles both empty and NaN.

**Reproducibility checklist** for the thesis: prompt is recorded in full, sampler is
seeded via `--seed`, model paths are absolute, and `started_at` lets you correlate
with system logs. To regenerate any result: `spectre/build/main --run-id <same> ...`
will overwrite the directory deterministically.

### How the literature evaluates "quality is preserved"

| Family | Examples | Quality battery |
|---|---|---|
| Vanilla SD (lossless) | Leviathan 2022, Chen 2023 | None required; PPL parity as sanity check |
| Self-speculation / n-gram | llama.cpp ngram_*, our PR #22055 | None required (still uses standard rejection rule) |
| Tree drafting (lossless) | SpecInfer, EAGLE, TALON | None required; ablations on tree topology |
| **Soft / lossy acceptance** | Medusa, Lookahead | HumanEval pass@1, MT-Bench, GSM8K, MMLU |
| **Reward-guided** | RSD (Liao 2025) | MATH, GSM8K, AIME - task-specific |
| **Quantized draft** | various | Corpus PPL on WikiText-103 / C4 + downstream |
| **Biased acceptance** (`τ < 1`) | various ablations | KL drift + downstream evals |

Standard datasets when downstream evaluation is required:
- **Generation quality:** MT-Bench, AlpacaEval, Arena-Hard
- **Reasoning:** GSM8K, MATH, AIME, MMLU
- **Code:** HumanEval, MBPP, LiveCodeBench
- **PPL:** WikiText-103, C4 validation, The Pile validation slice

What papers **don't** typically use for SD evaluation:
- BLEU / ROUGE / chrF - penalize valid alternative completions; bad fit for sampling.
- Single-prompt qualitative comparison - anecdotal, not reproducible.

### When perplexity becomes a real lever (lossy variants)

Lossless SD is the boring case (PPL preserved by construction). The interesting frontier is **lossy** variants that deliberately trade exactness for speed:

1. **Biased acceptance.** Replace `α(x) = min(1, p(x)/q(x))` with `α'(x) = min(1, τ · p(x)/q(x))` for `τ > 1`. Accepts more tokens, drifts away from `p`. The Pareto frontier `speedup × KL(p ‖ p')` is the right thing to plot.
2. **Medusa-style soft acceptance.** Multiple heads vote; the rejection rule is relaxed. PPL drift vs. speedup is reported in the Medusa paper.
3. **Reward-guided (RSD).** Replace acceptance criterion with a reward threshold. Can actually *improve* over target on reasoning tasks at the cost of theoretical unbiasedness.
4. **Aggressive draft quantization** (Q2/Q3 draft of a Q8 target). Increases `TV(p, q)` - quality drift if you keep the acceptance rule strict, or speed gain if you relax it.

Plausible thesis follow-up: with `p_target` and `p_draft` now in `tokens.csv` per
accepted token, sweep a bias parameter `τ ∈ [1.0, 2.5]` in the acceptance rule
and plot the Pareto `extra-speedup` vs `mean |p_target − p_draft|` (TV proxy)
vs `HumanEval pass@1`. Converts the "lossless / lossy" boolean into a continuous
design space.

Figures produced by `spectre/scripts/quality_eval.py` (run after collecting fresh data):

| File | What it shows |
|---|---|
| `quality-ppl-trace.png` | per-token logprob + cumulative PPL; baseline AR overlay if a complete AR run exists |
| `quality-target-prob-hist.png` | distribution of `p_target(accepted)` split by source (draft / bonus / ar) |
| `quality-acceptance-by-run.png` | bar chart of acceptance rate per run |
| `quality-speed-vs-accept.png` | efficiency frontier (with baseline reference line) |
| `quality-per-position-empirical.png` | empirical per-position acceptance + Bernoulli model |
| `quality-draft-vs-target-prob.png` | `p_target` vs `p_draft` scatter on accepted tokens |
| `quality-rounds-histogram.png` | distribution of accepted-per-call vs. geometric model |


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
according to their likelihood. Thus sampling from a language model—which rep
resents a distribution over sentences—means to generate some sentences, choosing
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
1. `logit(p) = log(p/(1-p))` is the raw, denormalized predictions generated by a model before applying any activation function
