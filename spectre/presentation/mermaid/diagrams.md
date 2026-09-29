# Mermaid sources (Spectre / speculative decoding)

Copy any block into [Mermaid Live Editor](https://mermaid.live) or into Markdown (GitHub, GitLab, Obsidian).

Diagrams mirror `spectre/src/main.cpp` and `html/atlas.html`.

---

## AR vs speculative

```mermaid
flowchart LR
  subgraph ar["Vanilla AR (main.cpp non-spec branch)"]
    direction TB
    A1["llama_decode(ctx_tgt, batch of 1)"] --> A2["llama_sampler_sample(sampler_tgt, …)"]
    A2 --> A3["current_batch = llama_batch_get_one(pointer to token, 1)"]
    A3 --> A1
  end
  subgraph sp["Speculative (draft path set)"]
    direction TB
    S1["draft() on ctx_dft"] --> S2["Build speculation_batch_tgt"]
    S2 --> S3["llama_decode(ctx_tgt, batch)"]
    S3 --> S4["sample_and_accept"]
    S4 --> S5["Update prompt_tgt / last_token"]
    S5 --> S6["llama_memory_seq_rm"]
    S6 --> S1
  end
```

## Target prefix init

```mermaid
flowchart TD
  P["prompt_tgt - full tokenized prompt, N tokens"] --> B["llama_batch_get_one(prompt_tgt, N minus 1)"]
  B --> D["llama_decode(ctx_tgt, batch)"]
  D --> KV["KV cache: positions 0 … N−2"]
  P --> L["last_token = prompt_tgt.back()"]
  L --> POP["prompt_tgt.pop_back()"]
  POP --> INV["Invariant: full prefix = prompt_tgt then last_token"]
```

## One round pipeline

```mermaid
flowchart LR
  T1["pmax → n_past"] --> T2["draft()"]
  T2 --> T3["trim n_max / n_min"]
  T3 --> T4["reset_batch tgt"]
  T4 --> T5["add last_token + draft[]"]
  T5 --> T6["llama_decode tgt"]
  T6 --> T7["sample_and_accept"]
  T7 --> T8["push_back loop"]
  T8 --> T9["memory_seq_rm"]
```

## Sample propagation (sequence)

```mermaid
sequenceDiagram
  participant PT as prompt_tgt
  participant L as last_token
  participant D as draft
  participant T as ctx_tgt
  Note over L,D: Round start: L holds seed
  D->>D: draft() uses PT + L
  D->>T: speculation batch: L, d0, d1, ...
  T->>T: llama_decode
  T-->>L: accepted[] from sample_and_accept
  loop For each accepted[i]
    PT->>PT: push_back(previous last_token)
    L->>L: last_token = accepted[i]
  end
  T->>T: llama_memory_seq_rm
  Note over PT,L: Next round: extended PT, new L
```

## sample_and_accept

```mermaid
flowchart TD
  SYNC["llama_synchronize(ctx)"] --> IDX{"index < draft.size ?"}
  IDX -->|yes| SAM["sample at logits row index"]
  SAM --> ACC["sampler_accept(id)"]
  ACC --> PUSH["accepted.push_back(id)"]
  PUSH --> EQ{"draft[index] == id ?"}
  EQ -->|no| RET1["return accepted"]
  EQ -->|yes| INC["index++"]
  INC --> IDX
  IDX -->|no| BON["sample at row draft.size"]
  BON --> FIN["push, return accepted"]
```

## Future work (mind map)

```mermaid
mindmap
  root((Future work))
    Adaptive draft
      AdaEDL / SpecDec++
      entropy bounds
    Tree drafting
      TALON
      GGML attention cost
    Heterogeneous
      CPU GPU NPU split
      cost model
    Trace speculation
      hot n-grams
      JIT analogy
```

---

# GGUF file format

## On-disk layout (memory-mappable)

```mermaid
flowchart TB
  classDef hdr  fill:#fef3c7,stroke:#b45309,color:#0f172a
  classDef meta fill:#dbeafe,stroke:#1e40af,color:#0f172a
  classDef ti   fill:#e0f2fe,stroke:#0369a1,color:#0f172a
  classDef td   fill:#dcfce7,stroke:#166534,color:#0f172a

  subgraph FILE[".gguf file (little-endian, mmap'd at load)"]
    direction TB
    H["Header<br/>magic 'GGUF' | version | n_tensors | n_meta_kv"]:::hdr
    KV["Metadata KV map<br/>general.architecture = llama<br/>llama.embedding_length = 4096<br/>llama.context_length = 8192<br/>tokenizer.ggml.tokens = [...]<br/>quantization_version = 2"]:::meta
    TI["Tensor info table (per tensor)<br/>name | n_dims | shape | ggml_type | data_offset"]:::ti
    PAD["alignment padding (to 32-byte multiple)"]:::meta
    TD["Tensor data blob<br/>raw block-quantized weights, contiguous<br/>mmap'd directly into GPU/CPU pages"]:::td

    H --> KV --> TI --> PAD --> TD
  end

  LOAD["llama_model_load_from_file<br/>parses header + KV + tensor info,<br/>mmaps tensor data (no copy)"]
  FILE --> LOAD
```

## Conversion pipeline (HF -> GGUF -> quantized)

```mermaid
flowchart LR
  HF["HuggingFace repo<br/>model.safetensors (BF16/FP16)<br/>config.json, tokenizer.*"]
  CV["convert_hf_to_gguf.py<br/>(reads HF layout,<br/>emits GGUF metadata + FP16 tensors)"]
  F16[".gguf F16<br/>(reference, ~2 bytes/weight)"]
  QZ["llama-quantize in.gguf out.gguf Q4_K_M<br/>(per-tensor block quantization)"]
  OUT[".gguf Q4_K_M<br/>(~0.55 bytes/weight, ~28% size)"]

  HF --> CV --> F16 --> QZ --> OUT

  IMAT["llama-imatrix --calibration-corpus C4<br/>-> imatrix.dat (importance per weight col)"]
  IMAT -. required for IQ_* quants .-> QZ
```

## Inside one Q4_K super-block (256 weights, 144 bytes)

```mermaid
flowchart TB
  subgraph SB["Super-block (256 weights, 144 bytes total)"]
    direction TB
    SS["d : FP16 (2 B) - super scale"]
    SM["dmin : FP16 (2 B) - super min"]
    SC["8x 6-bit sub-scales + 8x 6-bit sub-mins (12 B)"]
    QS["8 sub-blocks x 32 weights x 4 bits = 128 B"]
  end

  subgraph SUB["One sub-block (32 weights)"]
    direction TB
    QQ["each weight: q_i in [0..15] (4 bits)"]
    REC["reconstruct: w_i ≈ d · scale_j · q_i + dmin · min_j<br/>(j = sub-block index)"]
    QQ --> REC
  end

  QS --> SUB

  NOTE["effective ≈ 144 x 8 / 256 = 4.5 bits/weight<br/>FP16 = 16 bits -> 3.6x smaller"]
  SB --> NOTE
```

## K-quant vs I-quant

```mermaid
flowchart LR
  classDef k fill:#dbeafe,stroke:#1e40af,color:#0f172a
  classDef i fill:#fee2e2,stroke:#b91c1c,color:#0f172a

  subgraph KQ["K-quant"]
    K1["block of 32 weights"]:::k
    K2["uniform: all 32 share<br/>same bit budget"]:::k
    K3["one scale + one min per block"]:::k
    K4["simple SIMD dequant (fast on CPU/GPU)"]:::k
    K1 --> K2 --> K3 --> K4
  end

  subgraph IQ["I-quant (imatrix-aware)"]
    I0["offline: imatrix records<br/>sum activation^2 per column"]:::i
    I1["block of 32 weights"]:::i
    I2["important cols -> finer lattice<br/>unimportant cols -> coarser lattice"]:::i
    I3["LUT-based dequant (slower kernel)"]:::i
    I0 --> I1 --> I2 --> I3
  end

  KQ -. same bit budget .-> IQ
  IQ -. lower PPL at <=4 bits .-> WIN["IQ wins quality at low bits<br/>K wins raw decode speed"]
```

