# Optimizing Text Generation with Large language Models via Speculative Sampling.

## Table of contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
    - [Keywords](#keywords)
    - [Resources](#resources)

## Abstract

## Introduction

Speculative decoding works like this:

The Small LLM generates a bunch of sequential tokens. Then, the big LLM runs all these in one go.

For each token, if the probability in the large LLM is higher than the probability of the small, it's taken directly (therefore, it's not messing with the large LLM's statistics). If the probability is lower, the chances of it being taken is proportional to the difference in probabilities.

This makes it likely that the token is not taken, and all the effort is wasted.

i.e. if the small model is pretty good, you get a speed up, and you don't change the output, but if its bad, you are wasting lots of compute for nothing, and its overall slower. *See [Branch\_predictor](https://en.wikipedia.org/wiki/Branch_predictor)*

### Optimizing Text Production from Large Language Models (LLM) through the Speculative Sampling Technique

This paper studies the speculative sampling technique, with the aim of choosing a small language model (SLM) instead of a large one (LLM), when the SLM can perform equally well, activating the LLM only when necessary. Experiments will be conducted to compare different approaches, with response time and energy consumption measurements in relation to the initial large models. Particular emphasis will be given to (a) the selection of appropriate models and (b) their impact on the accuracy and efficiency of the results.

An inference scheduling problem, drawing from compiler optimization theory (e.g., branch prediction, PGO, instruction scheduling) to minimize latency and energy usage.

### Keywords:

* [Scheduling (computing)](https://en.wikipedia.org/wiki/Scheduling_(computing))
* Compiler-inspired runtime design for speculative text generation with multi-model pipelines
* A prediction-and-scheduling runtime with compiler-like heuristics
* Code generation for inference graphs
* SLM–LLM pipeline as an IR (intermediate representation)
* Adaptive runtime scheduler (like a JIT) for inference.
* Profiling framework that drives decisions.
* Performance/energy trade-off curves similar to compiler optimization trade-offs.

### Resources:
1. [Accelerating Large Language Model Decoding with Speculative Sampling (DeepMind)](https://arxiv.org/abs/2302.01318)
2. [Accelerating LLM Inference with Staged Speculative Decoding](https://arxiv.org/abs/2308.04623)
3. [Looking Back at Speculative Decoding](https://news.ycombinator.com/item?id=43216518)
4. [Instantaneous Grammatical Error Correction with Shallow Aggressive Decoding](https://arxiv.org/abs/2106.04970)
5. [A Hitchhiker’s Guide to Speculative Decoding - By Team PyTorch at IBM](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)
6. [Speculative Decoding for 2x Faster Whisper Inference](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/speculative_decoding.ipynb#scrollTo=baf87589-b7fe-45dd-a6f6-9b9223581562)
7. [Learning Harmonized Representations for Speculative Sampling](https://arxiv.org/abs/2408.15766)
8. [Llama.cpp speculative sampling: 2x faster inference for large models ](https://news.ycombinator.com/item?id=37390024)
9. [Speculative: PoC for speeding-up inference via speculative sampling by ggerganov](https://news.ycombinator.com/item?id=37357783)
10. [Speculative Sampling Explained](https://saibo-creator.github.io/post/2024_03_08_speculative_sampling/)
11. [llama.cpp: add example for speculative sampling #2030](https://github.com/ggml-org/llama.cpp/issues/2030)
12. [llama.cpp: PoC for speeding-up inference via speculative sampling #2926](https://github.com/ggml-org/llama.cpp/pull/2926)
13. [ollama: Enable speculative decoding #5800](https://github.com/ollama/ollama/issues/5800)

// vim: wrap nonu cole=3
