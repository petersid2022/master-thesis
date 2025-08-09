# Optimizing Text Generation with Large language Models via Speculative Sampling.

## Table of contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
    - [Keywords](#keywords)
    - [Resources](#resources)

## Abstract

## Introduction

### Βελτιστοποίηση Παραγωγής Κειμένου από Μεγάλα Γλωσσικά Μοντέλα (LLM) μέσω της Τεχνικής Speculative Sampling

Η εργασία αυτή μελετά την τεχνική speculative sampling, με σκοπό την επιλογή ενός μικρού γλωσσικού μοντέλου (SLM) αντί ενός μεγάλου (LLM), όταν το SLM μπορεί να αποδώσει εξίσου καλά, ενεργοποιώντας το LLM μόνο όταν κρίνεται απαραίτητο. Θα διεξαχθούν πειράματα για τη σύγκριση διαφορετικών προσεγγίσεων, με μετρήσεις χρόνου απόκρισης και κατανάλωσης ενέργειας σε σχέση με τα αρχικά μεγάλα μοντέλα. Ιδιαίτερη έμφαση θα δοθεί (α) στην επιλογή των κατάλληλων μοντέλων και (β) στην επίδρασή τους στην ακρίβεια και στην αποδοτικότητα των αποτελεσμάτων.

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

// vim: wrap cursorline scrolloff=0
