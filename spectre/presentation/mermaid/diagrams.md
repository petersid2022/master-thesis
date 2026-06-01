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
