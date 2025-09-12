#import "style.typ": *

#align(center, text(17pt)[
  *Optimizing Text Generation with Large language Models via Speculative Sampling.*
])

= Table of contents
<table-of-contents>
+ #link(<abstract>)[Abstract]
+ #link(<introduction>)[Introduction]
+ #link(<keywords>)[Keywords]
+ #link(<resources>)[Resources]

= Abstract
<abstract>
= Introduction
<introduction>
#emph["compiler for speculative inference"]

This paper studies the speculative sampling technique, with the aim of
choosing a small language model (SLM) instead of a large one (LLM), when
the SLM can perform equally well, activating the LLM only when
necessary. Experiments will be conducted to compare different
approaches, with response time and energy consumption measurements in
relation to the initial large models. Particular emphasis will be given
to (a) the selection of appropriate models and (b) their impact on the
accuracy and efficiency of the results.

An inference scheduling problem, drawing from compiler optimization
theory (e.g., branch prediction, PGO, instruction scheduling) to
minimize latency and energy usage.

#blockquote[
These overlaps stem from the fact that LLM inference can be viewed as a
computational pipeline, similar to how compilers treat code as a graph
or sequence to transform.
]

Research shows speedups of 2-3x from speculative decoding alone, and
layering compiler-inspired opts could push this further, especially for
edge cases like long sequences or low-acceptance-rate drafts.

=== Speculative decoding/sampling works like this:
<speculative-decodingsampling-works-like-this>
+ The Small LLM generates a bunch of sequential tokens. Then, the big
  LLM runs all these in one go.
+ For each token, if the probability in the large LLM is higher than the
  probability of the small, it’s taken directly (therefore, it’s not
  messing with the large LLM’s statistics). If the probability is lower,
  the chances of it being taken is proportional to the difference in
  probabilities.

This makes it likely that the token is not taken, and all the effort is
wasted.

i.e. if the small model is pretty good, we get a speed up, and we don’t
change the output, but if its bad, we are wasting lots of compute for
nothing, and its overall slower. #emph[See
#link("https://en.wikipedia.org/wiki/Branch_predictor")[Branch\_predictor]]

- These speculative decoding variants allow tokens to exit the model
  early if confident, analogous to compiler optimizations like function
  inlining or dead code elimination to skip unnecessary computations.
- Speculative sampling often builds a tree of possible token paths
  during drafting. we could apply compiler-style graph transformations
  (e.g., pruning redundant branches via static analysis or common
  subexpression elimination on token probabilities) to make the tree
  exploration more efficient.
- In compilers, scheduling reorders operations for parallelism. In
  speculative sampling, we could "schedule" draft token generation to
  prioritize high-probability paths, inspired by compiler techniques for
  out-of-order execution.

== Keywords
<keywords>
- #link("https://en.wikipedia.org/wiki/Scheduling_(computing)")[Scheduling (computing)]
- Compiler-inspired runtime design for speculative text generation with
  multi-model pipelines
- A prediction-and-scheduling runtime with compiler-like heuristics
- Code generation for inference graphs
- SLM–LLM pipeline as an IR (intermediate representation)
- Adaptive runtime scheduler (like a JIT) for inference.
- Profiling framework that drives decisions.
- Performance/energy trade-off curves similar to compiler optimization
  trade-offs.

== Resources
<resources>
+ #link("https://arxiv.org/abs/2302.01318")[Accelerating Large Language Model Decoding with Speculative Sampling (DeepMind)]
+ #link("https://arxiv.org/abs/2308.04623")[Accelerating LLM Inference with Staged Speculative Decoding]
+ #link("https://news.ycombinator.com/item?id=43216518")[Looking Back at Speculative Decoding]
+ #link("https://arxiv.org/abs/2106.04970")[Instantaneous Grammatical Error Correction with Shallow Aggressive Decoding]
+ #link("https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/")[A Hitchhiker’s Guide to Speculative Decoding - By Team PyTorch at IBM]
+ #link("https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/speculative_decoding.ipynb#scrollTo=baf87589-b7fe-45dd-a6f6-9b9223581562")[Speculative Decoding for 2x Faster Whisper Inference]
+ #link("https://research.google/blog/looking-back-at-speculative-decoding/")[Looking back at speculative decoding]
+ #link("https://arxiv.org/abs/2408.15766")[Learning Harmonized Representations for Speculative Sampling]
+ #link("https://news.ycombinator.com/item?id=37390024")[Llama.cpp speculative sampling: 2x faster inference for large models]
+ #link("https://news.ycombinator.com/item?id=37357783")[Speculative: PoC for speeding-up inference via speculative sampling by ggerganov]
+ #link("https://saibo-creator.github.io/post/2024_03_08_speculative_sampling/")[Speculative Sampling Explained]
+ #link("https://github.com/ggml-org/llama.cpp/issues/2030")[llama.cpp: add example for speculative sampling \#2030]
+ #link("https://github.com/ggml-org/llama.cpp/pull/2926")[llama.cpp: PoC for speeding-up inference via speculative sampling \#2926]
+ #link("https://github.com/ollama/ollama/issues/5800")[ollama: Enable speculative decoding \#5800]
+ #link("https://ralphmao.github.io/ML-software-system/")[Understanding LLM System with 3-layer Abstraction]
+ #link("https://huggingface.co/blog/whisper-speculative-decoding")[Speculative Decoding for 2x Faster Whisper Inference]
