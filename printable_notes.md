---
title: "Speculative Decoding"
subtitle: "A printable study guide from transformer inference to the Spectre implementation"
date: "11 August 2026"
lang: en
toc: true
toc-depth: 3
papersize: a4
geometry: margin=25mm
fontsize: 11pt
---

# Reader's guide

These notes are a curated version of `notes.md`. They are arranged as a learning
path rather than as a chronological research notebook:

```text
decoder inference
    -> autoregressive bottleneck
    -> speculative decoding
    -> correctness guarantees
    -> llama.cpp implementation
    -> experimental evaluation
```

The guide uses one running example:

```text
accepted history: A B C D E
draft proposal:       F G H
```

Letters represent token IDs, not necessarily individual characters or words.
A real tokenizer may encode the text as different pieces and may add a beginning
of sequence token.

## Scope and terminology

This guide distinguishes three algorithms that must not be conflated:

1. **Target-only autoregressive decoding:** the target model produces one token
   at a time.
2. **Greedy exact-match speculation:** a draft token is retained only when it is
   the same token as the target model's greedy choice.
3. **Canonical stochastic speculative sampling:** draft proposals are accepted
   using the probability ratio $p/q$, with residual sampling after rejection.

Canonical speculative decoding still invokes the target model to verify every
round. A system that sometimes skips the target entirely is a routing or cascade
system and has a different correctness argument.

## Notation

| Symbol | Meaning |
|---|---|
| $p(x\mid h)$ | target-model distribution for token $x$ after history $h$ |
| $q(x\mid h)$ | draft-model distribution |
| $k$ | maximum number of proposed draft tokens in one round |
| AR | autoregressive |
| KV cache | cached attention keys and values |
| EOG/EOS | end-of-generation/end-of-sequence token |
| logit | unnormalized vocabulary score |

When this guide discusses canonical stochastic sampling, $p$ and $q$ mean
the distributions actually used for sampling after relevant transformations
such as temperature, top-k, top-p and grammar constraints. A softmax over raw
model logits is not necessarily that distribution.

# Part I — Transformer inference foundations

# 1. Transformer architectures

## 1.1 Encoder-only models

Encoder-only models, such as BERT, build representations of an existing input.
Their self-attention is normally bidirectional: every token can attend to tokens
on both its left and right.

Typical uses include:

- classification;
- semantic embeddings;
- retrieval;
- named-entity recognition.

## 1.2 Decoder-only models

Decoder-only models, such as GPT, Llama, Qwen, Mistral and Gemma, predict the
next token from the accepted prefix.

Given:

```text
A B C D E
```

the model represents:

$$
P(F\mid A,B,C,D,E).
$$

It uses **causal self-attention**:

| Token | Tokens visible to it |
|---|---|
| `A` | `A` |
| `B` | `A B` |
| `C` | `A B C` |
| `D` | `A B C D` |
| `E` | `A B C D E` |

The causal mask prevents an earlier position from using future information.

## 1.3 Encoder-decoder models

Encoder-decoder models, such as the original Transformer and T5, have two
components:

```text
source input -> encoder -> source representation
                                  |
previous output -> decoder -------+-> next output token
```

The encoder normally uses bidirectional self-attention. The decoder uses causal
self-attention over its generated prefix and cross-attention over the encoder
representation.

| Architecture | Visibility | Typical purpose |
|---|---|---|
| Encoder-only | Both directions | Understanding and embeddings |
| Decoder-only | Current and previous tokens | Text generation |
| Encoder-decoder | Bidirectional source, causal output | Translation and transformation |

# 2. What happens after tokenization?

Assume the tokenizer returns:

```text
[A, B, C, D, E]
```

Internally these are integer token IDs.

## 2.1 Embeddings

The embedding table maps every token ID to a vector:

```text
embedding_table[vocabulary_size][model_dimension]
```

For example:

```text
A -> embedding(A)
B -> embedding(B)
...
E -> embedding(E)
```

The resulting prompt representation has shape:

```text
[number_of_tokens, model_dimension]
```

## 2.2 The two operations repeated by a Transformer

A decoder layer primarily performs:

```text
1. causal self-attention: exchange information between token positions
2. feed-forward network: transform each token's features independently
```

A simplified Llama-style block is:

```text
hidden state
    -> normalization
    -> causal self-attention
    -> residual addition
    -> normalization
    -> feed-forward network
    -> residual addition
```

This block is repeated for every model layer.

# 3. Attention

Attention lets a token retrieve information from relevant earlier tokens.

Consider:

```text
The animal did not cross the street because it was tired.
```

When processing `it`, the model may need information represented at `animal`.
This relationship is learned from data rather than manually programmed.

## 3.1 Query, key and value

At every layer, each token representation is projected into three vectors:

- **Query (Q):** what information is this position looking for?
- **Key (K):** what information does this position advertise?
- **Value (V):** what information can this position contribute?

For the running example:

```text
A -> q_A, k_A, v_A
B -> q_B, k_B, v_B
C -> q_C, k_C, v_C
D -> q_D, k_D, v_D
E -> q_E, k_E, v_E
```

These vectors are created using learned matrices:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

## 3.2 Attention scores

The query for `E` is compared with every permitted key:

```text
q_E dot k_A
q_E dot k_B
q_E dot k_C
q_E dot k_D
q_E dot k_E
```

The scaled score is:

$$
s_{ij}=\frac{q_i k_j^\mathsf{T}}{\sqrt{d_{\text{head}}}}.
$$

The causal mask sets scores for future positions to negative infinity.
Softmax converts the remaining scores into weights that sum to one.

An illustrative result might be:

```text
A: 0.05
B: 0.10
C: 0.15
D: 0.60
E: 0.10
```

The attention output for `E` is the weighted sum:

$$
o_E=0.05v_A+0.10v_B+0.15v_C+0.60v_D+0.10v_E.
$$

Multi-head attention repeats this process with several learned projections and
combines their outputs. Individual heads can represent different relationships,
although they are not guaranteed to have a simple human-readable role.

# 4. Feed-forward networks and common acronyms

Attention mixes information between positions. The feed-forward network (FFN)
then transforms the features of each position independently.

## 4.1 FFN

The same FFN is applied separately to every token:

```text
FFN(hidden_A)
FFN(hidden_B)
...
FFN(hidden_E)
```

It commonly expands the model dimension and then projects it back:

```text
model dimension
    -> larger intermediate dimension
    -> model dimension
```

## 4.2 SiLU and SwiGLU

SiLU is a nonlinear activation:

$$
\operatorname{SiLU}(x)=x\,\sigma(x).
$$

SwiGLU is a gated FFN used by many Llama-style models. A simplified expression
is:

$$
\operatorname{FFN}(x)=
W_{\text{down}}\left(
\operatorname{SiLU}(W_{\text{gate}}x)\odot W_{\text{up}}x
\right).
$$

- $W_{\text{up}}$ expands the representation.
- $W_{\text{gate}}$ decides which transformed features should pass.
- $W_{\text{down}}$ projects back to the model dimension.
- $\odot$ is element-wise multiplication.

These details matter for understanding model computation, but they do not
directly determine speculative acceptance.

## 4.3 RoPE

Attention alone does not encode token order. Rotary Positional Embedding (RoPE)
rotates query and key components according to position:

```text
A at position 0
B at position 1
...
E at position 4
```

The resulting query-key relationship carries relative-position information.
Incorrect positions therefore change attention and can corrupt generation.

## 4.4 RMSNorm and residual connections

RMSNorm controls the scale of hidden vectors. A residual connection adds a
block's input back to its output:

$$
\text{result}=x+\text{transformation}(x).
$$

Residual paths help deep models preserve and refine information.

# 5. Logits, softmax and sampling

After the last Transformer layer, the final hidden state is projected into
vocabulary space by the language-model head.

For position `E`, the model produces one logit per vocabulary token:

```text
token F: 14.2
token G: 11.7
token H:  8.4
...
```

These logits represent scores for the token following `E`.

## 5.1 The next-token shift

The indexing rule is:

```text
logits after A predict the token after A
logits after B predict the token after B
...
logits after E predict the token after E
```

During ordinary prompt generation, only the final prompt row is needed.

## 5.2 Greedy and stochastic selection

Greedy decoding chooses:

$$
x=\arg\max_x \operatorname{logit}(x).
$$

Stochastic decoding converts or filters the scores using operations such as:

- temperature;
- top-k;
- top-p;
- a random draw.

Vocabulary softmax and attention softmax are separate operations serving
different purposes.

# 6. Prefill, decode and the KV cache

## 6.1 Prompt prefill

The inference engine can submit:

```text
[A, B, C, D, E]
```

as one prompt batch. Causal attention lets prompt positions be calculated in a
batched forward pass while preserving left-to-right visibility.

The final logits row predicts the first generated token `F`.

## 6.2 KV cache

At each layer, the engine stores the keys and values for processed positions:

```text
KV[layer][kv_head][position][head_dimension]
```

Queries and most intermediate hidden states are not normally retained. Future
tokens need the old keys and values.

This description applies to ordinary attention-based Transformers. Hybrid or
recurrent architectures can maintain additional recurrent state besides a
conventional KV cache.

After prompt prefill:

```text
KV cache: A B C D E
logits:   prediction after E
```

## 6.3 Autoregressive decode

Suppose the sampler selects `F`. Sampling does not add `F` to the KV cache.
The engine must submit `F` as the next decoder input.

| Call | Input | KV cache after the call | Logits predict |
|---|---|---|---|
| Prefill | `A B C D E` | `A B C D E` | `F` |
| Decode | `F` | `A B C D E F` | `G` |
| Decode | `G` | `A B C D E F G` | `H` |

A minimal AR loop is:

```text
decode prompt

repeat:
    sample from latest logits
    record and emit the token
    stop on EOG or budget
    decode the sampled token
```

## Mental model

```text
Attention = tokens communicate with previous tokens
FFN       = each token transforms its features
KV cache  = retained attention keys and values
LM head   = hidden state to vocabulary scores
Sampler   = scores to next token
```

## Core inference invariant

Before sampling the next token:

> The KV cache must represent the accepted history exactly once at the correct
> positions, and the logits must come from the final token of that history.

# Part II — Why decoding is slow

# 7. The autoregressive bottleneck

Generation is sequential:

```text
produce F
    -> decode F
    -> produce G
    -> decode G
    -> produce H
```

The next generated token is unknown until the previous token has been selected.
A target-only decoder therefore requires one expensive target step per output
token.

## 7.1 Memory bandwidth

Single-token decode often has low arithmetic intensity: large model weights must
be accessed to process only one new token. On many devices, moving weights and
KV data is more limiting than peak floating-point throughput.

This is a workload-dependent claim, not a universal constant. The balance
changes with:

- batch size;
- model architecture;
- CPU/GPU backend;
- quantization;
- offloading;
- context length;
- Flash Attention and kernel implementation;
- dense versus mixture-of-experts execution.

## 7.2 Why batched verification can help

If several candidate token IDs are already known, the target can evaluate them
as a causal batch. Larger matrix operations can use the hardware more efficiently
and amortize part of the model-weight traffic.

One batched target call is not the same cost as one single-token call. The useful
comparison is measured latency:

$$
T_{\text{target-batch}}(k+1)
\quad\text{versus}\quad
(k+1)T_{\text{target-single}}.
$$

# 8. Quantization

Quantization stores model weights using fewer bits than the training precision.
A basic conceptual example is:

$$
q=\operatorname{round}(w/s),\qquad \hat{w}=sq,
$$

where $s$ is a scale, $q$ is a low-bit integer and $\hat{w}$ approximates
the original weight.

Practical formats use block-wise scales, mixed precision and sometimes
importance calibration.

Quantization can:

- reduce model size;
- reduce weight-memory traffic;
- allow more layers to remain on an accelerator;
- alter model logits and therefore draft acceptance.

The speed and quality effects must be measured for the chosen model and backend.
Labels such as “lossless,” “3x faster,” or “the optimal quant” should not be
treated as universal properties.

Quantization and speculative decoding attack related costs:

```text
quantization          -> fewer bytes per weight
speculative decoding -> fewer sequential target calls per useful output token
```

# Part III — Speculative decoding

# 9. Greedy exact-match speculation

Assume:

```text
accepted history: A B C D E
draft proposal:       F G H
```

For clarity, assume deterministic greedy sampling.

## 9.1 One-sentence algorithm

The cheap draft model sequentially proposes $k$ tokens. The target evaluates
the pending token and all proposals in one causal batch. Proposals are checked
from left to right. A matching prefix is retained; the first mismatch is replaced
by the target's token.

## 9.2 Draft work

For $k=3$, a model-based drafter approximately performs:

```text
decode E -> propose F
decode F -> propose G
decode G -> propose H
```

These steps are sequential, but the draft model is intended to be much cheaper
than the target. An n-gram drafter can propose tokens without neural-model
forward passes.

## 9.3 Target work

Suppose the target KV cache contains:

```text
A B C D
```

and `E` is the accepted token held outside the cache. The target receives:

```text
[E, F, G, H]
```

One target call produces:

| Target row | Prefix represented by the row | Purpose |
|---:|---|---|
| 0, after `E` | `A B C D E` | verify `F` |
| 1, after `F` | `A B C D E F` | verify `G` |
| 2, after `G` | `A B C D E F G` | verify `H` |
| 3, after `H` | `A B C D E F G H` | possible bonus token |

The target is not generating unknown tokens in parallel. The drafter supplied
candidate token IDs, allowing the target to score a known hypothetical branch.
One `llama_decode` API call can still be divided into internal micro-batches by
the backend; “one call” does not mean one indivisible hardware operation.

## 9.4 Why verification stops at the first mismatch

Rows after a rejected proposal were calculated under the wrong history.

If `H` is rejected and replaced with `X`, a row calculated after `H` represents:

```text
A B C D E F G H
```

but the accepted history is:

```text
A B C D E F G X
```

The later row is therefore invalid for the real generation path.

# 10. The three outcomes of a greedy round

## 10.1 Immediate rejection

```text
draft:  F G H
target: X ...
```

`F` does not match `X`. The round emits:

```text
X
```

All later proposal work is discarded.

## 10.2 Matching prefix followed by correction

```text
draft:  F G H
target: F G X
```

The round emits:

```text
F G X
```

- `F` and `G` are accepted draft tokens.
- `H` is rejected.
- `X` is the target correction.

## 10.3 All proposals match

```text
draft:  F G H
target: F G H
```

The target also has logits after `H`, so it selects a bonus token `I`.
The round emits:

```text
F G H I
```

`I` is “bonus” only in the sense that no additional target forward pass is
needed to obtain its logits.

# 11. Calls, work and useful output

Ignoring prefix synchronization overhead, a three-token proposal performs
roughly:

```text
3 sequential cheap draft calls
+ 1 batched target call
```

The amount of useful output depends on the verification result:

| Outcome | Model-call structure | Output tokens |
|---|---|---:|
| First proposal fails | 3 draft + 1 batched target | 1 |
| Second proposal fails | same | 2 |
| Third proposal fails | same | 3 |
| All proposals match | same | 4 |

The first-proposal failure is the worst case for amortization. All proposals
matching and producing a bonus is the best case. The target batch has already
been evaluated before the program knows where the first mismatch occurs.

Under the simplified assumption that every proposal is accepted independently
with the same probability $\alpha$, the expected number of emitted tokens per
round is:

$$
\mathbb E[L]
=1+\alpha+\alpha^2+\cdots+\alpha^k
=\frac{1-\alpha^{k+1}}{1-\alpha},
$$

where $L=A+1$: $A$ accepted draft tokens plus one correction or bonus.
Real acceptance events are context-dependent and censored by earlier rejection,
so this is a model for interpretation rather than a measurement identity.

# 12. Why greedy correctness is preserved

Define the greedy target choice:

$$
T(h)=\arg\max_x p(x\mid h).
$$

Suppose:

```text
T(A B C D E)     = F
T(A B C D E F)   = G
T(A B C D E F G) = X
```

Target-only AR emits:

```text
F G X
```

If the draft proposes `F G H`, batched verification yields:

```text
row after E -> F
row after F -> G
row after G -> X
```

The speculative round accepts `F G`, rejects `H` and emits correction `X`.
Its output is also:

```text
F G X
```

The row after `G` cannot see later batch token `H` because of the causal mask.
It is therefore the same logical target decision as a standalone AR step after
`A B C D E F G`.

By induction:

1. Both algorithms begin a round with the same accepted prefix.
2. Every retained proposal equals the target's next greedy token.
3. At the first mismatch, speculation emits the target's token.
4. Both algorithms end the round with the same prefix.

Under equivalent state and deterministic execution, the complete greedy token
sequences are identical.

# 13. “Proposed,” “committed” and cache rollback

“Commit” is an analogy, not a llama.cpp operation.

Before verification, proposal tokens exist in temporary memory. Spectre does not
need to display them and later retract them. It displays them only after target
verification.

The target KV cache does temporarily contain the hypothetical batch. For:

```text
cached:   A B C D
batch:            E F G H
```

the cache temporarily represents:

```text
A B C D E F G H
```

If `H` is rejected and replaced by `X`, the cache suffix containing `H` is
removed:

```text
kept in target cache: A B C D E F G
held for next round:  X
```

`X` was sampled from logits after `G`; it has not yet been submitted as model
input. It is decoded at the beginning of the next round.

This is closer to speculative CPU execution than to version control:

```text
predict a path
execute the path provisionally
retire valid results in order
discard work after a misprediction
```

# 14. Canonical stochastic speculative sampling

Greedy exact matching is not the same algorithm as stochastic speculative
sampling from Leviathan et al. and Chen et al.

Let:

$$
p(x)=P_{\text{target}}(x\mid h),\qquad
q(x)=P_{\text{draft}}(x\mid h).
$$

Both distributions must correspond to the actual sampling policy, including
temperature and any truncation or grammar filters.

The draft proposes $x\sim q$. Canonical speculative sampling accepts it with:

$$
\alpha(x)=\min\left(1,\frac{p(x)}{q(x)}\right).
$$

If the proposal is rejected, the replacement is sampled from:

$$
\tilde p(x)=
\frac{\max(0,p(x)-q(x))}
{\sum_y\max(0,p(y)-q(y))}.
$$

The probability mass for a token $x^{*}$ is:

$$
\min(p(x^{*}),q(x^{*}))
+\max(0,p(x^{*})-q(x^{*}))
=p(x^{*}).
$$

Thus a correct stochastic speculative sampler produces the target distribution
exactly.

## 14.1 Sequence equality versus distribution equality

Greedy exact matching aims for:

```text
the same deterministic token sequence
```

Canonical stochastic sampling guarantees:

```text
the same probability distribution over sequences
```

Two independent samples from the same distribution need not be identical. A
same-seed claim also requires careful accounting of random-number consumption.

## 14.2 Acceptance and total variation distance

The expected one-position acceptance probability is:

$$
\begin{aligned}
\mathbb E[\alpha]
&=\sum_x q(x)\min\left(1,\frac{p(x)}{q(x)}\right)\\
&=\sum_x\min(p(x),q(x))\\
&=1-\operatorname{TV}(p,q).
\end{aligned}
$$

This is an acceptance identity, not a direct speedup formula. Runtime also
depends on draft cost, verification cost, block length and hardware.

For a deterministic n-gram proposal, $q$ is concentrated on one token. Under
the canonical stochastic rule, its acceptance probability is the post-sampler
target probability of that proposed token. Under greedy exact matching, the
result is instead binary: the proposal either equals the target argmax or it
does not.

# Part IV — Mapping the ideas to Spectre and llama.cpp

# 15. Runtime objects

The main runtime objects are:

- backend;
- target and draft models;
- target and draft contexts;
- vocabularies;
- sampler chains;
- token batches;
- logits;
- target and draft memory/KV caches;
- run configuration and telemetry.

A clear lifecycle is:

```text
initialize backend and load models
    -> create contexts
    -> tokenize prompt
    -> initialize samplers and reusable batches
    -> prefill prompt prefix
    -> run AR or speculative loop
    -> finalize telemetry
```

# 16. Baseline state and prompt handling

Spectre's shared speculative representation is:

```text
prompt_target = tokens already represented by target KV
last_token    = accepted token pending target decode
```

For `A B C D E`:

```text
target KV / prompt_target: A B C D
last_token:                E
```

The AR loop must decode `E`, sample `F`, then decode `F`. Re-decoding
`A B C D` would duplicate the prompt and omit `E`.

A conventional standalone AR implementation can instead prefill all of
`A B C D E` at once. If AR and speculation use different prompt boundaries,
their timing definitions must remain explicit and comparable.

# 17. Target positions and KV state

For a single text sequence with contiguous positions:

```text
cached positions: 0 1 2 3
maximum position: 3
next position:    4
```

`llama_memory_seq_pos_max(memory, 0)` queries cache bookkeeping for sequence
zero. It does not run a forward pass or copy the key/value tensors.

Using the KV cache as the position source of truth is robust after rollback.
However, it does not repair a logical mismatch between token history and cache
state. In a text-only engine without context shifting, this should remain true:

$$
|\texttt{prompt\_target}|=\texttt{max\_cached\_position}+1.
$$

Unexpected disagreement should be logged or asserted during development.

# 18. One target verification round in code terms

The round performs:

```text
1. query the next target position
2. obtain draft proposals
3. clamp proposal length to n_min/n_max
4. build [last_token, draft_0, ..., draft_k-1]
5. call target llama_decode once
6. verify proposals from left to right
7. classify emitted tokens
8. update logical history
9. remove rejected target-cache suffix
```

## 18.1 The confusing `accepted` vector

The current vector named `accepted` actually has this form:

```text
zero or more accepted draft tokens
+ exactly one final target token
```

The final target token is either:

- a correction after the first mismatch; or
- a bonus after all proposals match.

Examples:

| Outcome | Returned vector | Accepted proposal count |
|---|---|---:|
| Immediate rejection | `[X]` | 0 |
| Two matches, correction | `[F,G,X]` | 2 |
| All match, bonus | `[F,G,H,I]` | 3 |

This explains why the existing code uses `accepted.size() - 1`, but an explicit
result type is clearer.

## 18.2 Recommended emission categories

The output event needs three categories:

```text
accepted_draft
target_correction
target_bonus
```

Proposal source is a separate dimension:

```text
model
ngram
none
```

A correction row must not associate emitted correction `X` with the probability
of rejected proposal `H`.

A useful event representation is:

```cpp
enum class EmissionKind {
  accepted_draft,
  target_correction,
  target_bonus,
};

struct VerifiedToken {
  llama_token emitted_token;
  EmissionKind kind;
  std::optional<std::size_t> draft_position;
  std::optional<llama_token> proposed_token;
};
```

For correction `X` replacing `H`, telemetry should distinguish:

```text
emitted_token       = X
proposed_token      = H
target probability  = p(X)
proposal probability = q(H)
```

If the schema cannot represent both tokens, proposal probability should be
missing for correction rows rather than attached to the wrong token ID.

# 19. Implementation invariants and tests

High-value invariants include:

1. Target and draft use compatible token IDs and special-token behavior.
2. The target KV cache represents the accepted prefix exactly.
3. The pending `last_token` is not already duplicated in target KV.
4. Batch row $i$ is mapped to proposal $i$.
5. No rejected proposal remains in target KV after rollback.
6. A sampled correction or bonus is held pending until its next decode.
7. Generated-token and EOG accounting is consistent across CSV and metadata.
8. Token limits cannot be exceeded by the remainder of a speculative block.
9. Sampler state is advanced exactly once per selected token.

Suggested tests:

- AR output against a known target-only reference;
- greedy AR/speculative token parity;
- immediate rejection;
- rejection at every possible proposal position;
- all proposals accepted plus bonus;
- EOG at a proposal, correction and bonus position;
- `k=0` and `k=1`;
- n-gram hit and miss;
- exact token-budget enforcement;
- cache positions after rollback;
- deterministic configuration parsing.

llama.cpp APIs evolve. In the version currently vendored by this workspace,
`llama_sampler_sample()` already accepts the selected token into sampler state.
Calling `llama_sampler_accept()` again is redundant and can be wrong for a
stateful sampler.

# 20. Timing and observability

Separate timing scopes should include:

- prompt-prefix evaluation;
- draft-prefix synchronization;
- draft generation;
- target verification;
- sampling and bookkeeping;
- telemetry/output;
- end-to-end generation.

Before reading a timestamp intended to delimit accelerator work, synchronize
the relevant context. Avoid synchronization inside hot loops unless the result
is required immediately.

Per-round telemetry should include:

```text
round index
draft source
number proposed
number accepted
rejection position or no rejection
bonus produced
draft time
target verification time
```

Per-token telemetry should include:

```text
emitted token ID
emission kind
proposal token ID when applicable
draft position
target probability with clearly defined semantics
draft probability with clearly defined semantics
EOG flag
```

Spectre currently calculates its logged probabilities by applying softmax to
raw logits. When temperature, top-k, top-p or other filters are active, these
values are diagnostic raw-model probabilities rather than the actual $p$ and
$q$ required by canonical acceptance, KL or TV calculations. Full transformed
distributions are required for those analyses.

Do not include per-token file flushing or verbose console output in a metric
labelled “model decode time” unless the intent is explicitly end-to-end latency.

# Part V — Drafters and method taxonomy

# 21. Model-based drafting

A separate smaller model generates proposals autoregressively. Good pairings
normally require:

- compatible vocabulary/token-ID semantics;
- compatible special-token behavior and chat formatting;
- sufficiently similar output distributions;
- a draft that is much cheaper than the target on the selected hardware.

There is no universal optimal parameter-count ratio. Dense/MoE architecture,
quantization, memory placement and backend efficiency all affect cost.

# 22. N-gram drafting

An n-gram model approximates future behavior using a bounded history. In a
traditional word-level model:

$$
P(w_t\mid w_1,\ldots,w_{t-1})
\approx
P(w_t\mid w_{t-n+1},\ldots,w_{t-1}).
$$

Spectre's n-gram drafter is better understood as deterministic history lookup:

1. form a short pattern ending at the current token;
2. search earlier accepted history for that pattern;
3. copy the continuation after a match;
4. fall back to the neural draft model on a miss.

This can be very effective for repetitive code and structured text. It should
not be described as a trained probabilistic n-gram model unless probabilities
are actually estimated from counts.

For multi-token blocks, an overall accepted/proposed ratio hides that later
positions are observed only when earlier positions survive. More informative
n-gram measurements include hit rate, lookup time, mean accepted prefix length,
emitted tokens per verification and per-position survival probabilities.

# 23. Taxonomy

| Design axis | Alternatives |
|---|---|
| Draft source | Small model, n-gram/cache, self-speculation, learned heads |
| Proposal shape | Linear sequence, tree |
| Draft length | Fixed, adaptive |
| Scheduling | Sequential, asynchronous, heterogeneous |
| Guarantee | Exact greedy, distribution-preserving, deliberately biased |
| Training | Training-free, distilled, trained auxiliary module |

Examples of advanced families include EAGLE-style learned proposals, Medusa
heads, tree verification, adaptive-length methods and asynchronous speculative
pipelines.

# Part VI — Experimental methodology

# 24. Questions the experiment should answer

Useful research questions include:

1. Does greedy speculation produce exactly the same tokens as greedy AR?
2. How do draft strategy and draft length affect accepted tokens per round?
3. When does target batching offset draft overhead?
4. How do model pair, quantization and hardware placement affect the frontier?
5. What portion of latency belongs to drafting, verification and bookkeeping?

# 25. Controlled factors

Record and control:

- target and draft model identifiers or hashes;
- quantization;
- prompt or dataset;
- chat-template treatment;
- sampler policy and seed;
- context and batch sizes;
- draft minimum and maximum;
- n-gram parameters;
- CPU/GPU placement and layer offload;
- warm-up policy;
- compiler flags;
- Spectre and llama.cpp revisions;
- repetition count.

# 26. Metrics

## 26.1 Performance

- end-to-end latency;
- time to first token;
- output tokens per second;
- milliseconds per output token;
- target invocations per output token;
- energy per output token, when measurable.

## 26.2 Speculation efficiency

- proposal acceptance rate:

$$
\frac{\text{accepted draft tokens}}{\text{proposed draft tokens}};
$$

- accepted prefix length per round;
- emitted tokens per target verification;
- bonus-token rate;
- rejection position distribution.

Acceptance rate is not itself speedup. A high-acceptance drafter can still be
unprofitable if it is too expensive.

## 26.3 Correctness and quality

For greedy exact-match decoding, use token-by-token parity.

For canonical stochastic sampling, test distributional agreement over many
trials. Perplexity can be a sanity check, but two independent valid samples need
not have equal perplexity.

For deliberately lossy or reward-guided variants, report distribution drift and
task-specific quality in addition to speed.

# 27. Model names and pairing reference

A filename such as:

```text
Qwen3-30B-A3B-Instruct-Q4_K_M.gguf
```

often encodes:

| Slot | Example | Meaning |
|---|---|---|
| Family/version | `Qwen3` | architecture family |
| Size | `30B-A3B` | total and approximately active parameters |
| Fine-tuning | `Instruct` | instruction-tuned variant |
| Quantization | `Q4_K_M` | weight-storage format |
| Container | `.gguf` | llama.cpp model format |

Common suffixes:

| Suffix | Meaning |
|---|---|
| Base or none | pretrained completion model |
| Instruct / Chat / `it` | instruction or chat tuned |
| Coder | code specialization |
| Math / Reasoning / Thinking | task specialization |
| DPO / RLHF / RLAIF | preference-training method |

Fine-tuning and prompt-template mismatches can reduce draft agreement even when
the vocabulary is compatible.

## 27.1 Quantization labels

| Label family | General meaning |
|---|---|
| F32, F16, BF16 | floating-point storage |
| Q8, Q6, Q5, Q4, Q3, Q2 | progressively lower-bit quantization families |
| K-quants | block quantization used by GGUF/llama.cpp |
| I-quants | importance-calibrated quantization families |

Suffixes such as `_S`, `_M` and `_L` distinguish format-specific variants. Their
quality and speed ordering is model- and backend-dependent and should be measured
rather than inferred solely from the filename.

## 27.2 Dense and mixture-of-experts models

For a dense `8B` model, all major layers participate for every token.

For a name such as `30B-A3B`, approximately 30 billion parameters are stored,
while routing activates a smaller subset for a token. Total parameters still
affect memory capacity and potentially movement; active parameters influence
compute, but throughput is not determined by active count alone.

During multi-token verification, different positions can route to different
experts, increasing the union of expert weights that must be accessed.

# Part VII — Research directions

# 28. Adaptive draft length

A fixed $k$ cannot be optimal for every context:

```text
predictable/repetitive context -> longer draft may be profitable
uncertain context              -> shorter draft or AR fallback
```

Possible signals include entropy, draft confidence and observed acceptance
history. Online heuristics should be distinguished from true offline
profile-guided optimization.

# 29. Tiered drafting

A tiered strategy can try:

```text
n-gram lookup
    -> neural draft model on miss
    -> target verification
```

The cache-hierarchy analogy is useful but limited: the target still verifies
speculative output, so a target verification is not equivalent to a conventional
main-memory miss.

# 30. Tree proposals

Linear drafting bets on one path. Tree drafting explores multiple continuations
around uncertain positions. Its main systems challenge is efficient tree
attention and target verification without making the candidate structure too
expensive.

# 31. Heterogeneous and asynchronous scheduling

CPU/GPU/NPU placement introduces:

- transfer cost;
- backend-specific kernel efficiency;
- synchronization;
- memory-capacity constraints;
- opportunities for overlap.

Canonical draft-then-verify execution is sequential between phases. Overlapping
future drafting with current verification belongs to asynchronous speculative
variants and requires additional rollback logic.

# 32. Lossy and reward-guided variants

Distribution-preserving speculation optimizes execution without changing the
target distribution. Other methods intentionally relax that guarantee to improve
acceptance or task reward.

These methods require a quality-speed evaluation rather than only correctness
parity.

# Appendix A — Compact glossary

| Term | Meaning |
|---|---|
| Acceptance | Retaining a proposal under the chosen verification rule |
| Bonus token | Target token available after all proposals in a block are accepted |
| Correction | Target token emitted at the first rejected position |
| Causal mask | Prevents a position from attending to future positions |
| Draft | Cheap mechanism proposing future tokens |
| FFN | Per-position feed-forward network |
| KV cache | Per-layer cached attention keys and values |
| Logit | Unnormalized vocabulary score |
| LM head | Projection from hidden state to vocabulary logits |
| Prefill | Batched processing of the initial prompt |
| RoPE | Rotary positional encoding applied to queries and keys |
| Softmax | Normalizes scores into a probability distribution |
| Target | Model whose greedy sequence or distribution must be preserved |
| Verification | Target evaluation and acceptance/rejection of proposals |

# Appendix B — Essential formulas

## Greedy next token

$$
x_t=\arg\max_x p(x\mid x_{<t}).
$$

## Canonical acceptance

$$
\alpha(x)=\min\left(1,\frac{p(x)}{q(x)}\right).
$$

## Residual distribution

$$
\tilde p(x)\propto\max(0,p(x)-q(x)).
$$

## Acceptance and total variation

$$
\mathbb E[\alpha]=\sum_x\min(p(x),q(x))=1-\operatorname{TV}(p,q).
$$

## Sequence perplexity

$$
\operatorname{PPL}
=
\exp\left(
-\frac{1}{T}\sum_{t=1}^{T}
\log p(x_t\mid x_{<t})
\right).
$$

# Appendix C — Selected reading

1. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
2. [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
3. [llama.cpp speculative decoding documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
4. [A Hitchhiker's Guide to Speculative Decoding](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding)
5. [Looking Back at Speculative Decoding](https://research.google/blog/looking-back-at-speculative-decoding)
6. [SpecDec++: Adaptive Candidate Lengths](https://arxiv.org/abs/2405.19715)
7. [Medusa](https://arxiv.org/abs/2401.10774)
8. [Reward-Guided Speculative Decoding](https://arxiv.org/abs/2501.19324)

# Appendix D — Review questions

1. Why do logits after `E` predict the token following `E`?
2. What is stored in the KV cache, and why are queries normally not stored?
3. Why can the target evaluate `[E,F,G,H]` in one call?
4. Why are target rows after the first mismatch unusable?
5. What distinguishes an accepted draft, correction and bonus?
6. Why can high acceptance still fail to produce speedup?
7. What does greedy parity prove?
8. What does stochastic distribution preservation prove?
9. Why is $1-\operatorname{TV}(p,q)$ not itself a speedup?
10. Which state must be rolled back after a rejected proposal?

# Appendix E — Printing

The file can be printed directly from a Markdown preview. With Pandoc and a
LaTeX PDF engine installed:

```bash
pandoc printable_notes.md \
  --toc \
  --pdf-engine=xelatex \
  -o printable_notes.pdf
```

The original `notes.md` remains the unfiltered research notebook. This document
is the stable learning and discussion copy.
