# Concurrent LLM Sessions and Shared Working Memory

Status: measured on a Colab Tesla T4 on 2026-11-01. Model `Qwen/Qwen2.5-0.5B-Instruct`,
vLLM 0.32.0, 24 synthetic sessions, 48 max new tokens, greedy decoding, seed 1234.
Raw artifact: `results.json` (sha256
`22affa9aff1774cd82be213f1231f66e9286670718d5bb54e7d0d1901129595d`).
Run started 2026-11-01T08:39:59Z; total wall time 169.9 s.

An earlier run used `max(latency)` as the sequential wall time, so its throughput
column was invalid. That bug is fixed in commit `b8d098a`; only the fixed run is
reported here.

## Research framing

Two different claims must be separated:

1. **Concurrency.** A single autoregressive model can interleave many sessions.
   This is solved by continuous batching (Orca, vLLM, TensorRT-LLM, SGLang):
   requests join and leave the batch every decoding step, and paged KV blocks
   avoid contiguous reallocation.
2. **Shared working memory.** Sessions can reuse computation over common
   context. This is partially solved by automatic prefix caching (vLLM APC,
   SGLang RadixAttention) for *exact common prefixes*, and by position-independent
   KV reuse (CacheBlend, TokenDance, PrefillShare) for shared blocks at
   arbitrary positions or across heterogeneous models.

The literal idea "multiple inputs and outputs through coordinated processing and
shared working memory" is therefore not new as stated. The open problems are
granularity, isolation, and memory cost when caches diverge.

## Experiment

Each condition runs the same 24 sessions. Every prompt contains a private nonce
and a shared workspace of roughly 1.2k tokens. In the **prefix** layout the
workspace is at the start, so APC can reuse it. In the **middle** layout the
workspace follows the private prefix, carrying the same information but removing
prefix-cache eligibility. A warm request seeds the cache before measurement.

Eight conditions: APC disabled/enabled x prefix/middle layout x sequential/concurrent.

## Results

| APC | Layout | Concurrency | Wall (s) | tok/s | TTFT p50 / p95 (s) | Latency p50 / p95 (s) | Out tokens | Own nonce present | Foreign nonces |
|---|---|---|---|---|---|---|---|---|---|
| off | prefix | sequential | 7.92 | 144.8 | 0.050 / 0.052 | 0.331 / 0.333 | 1146 | 17/24 | 0 |
| off | prefix | concurrent | 0.98 | 1168.3 | 0.201 / 0.238 | 0.927 / 0.941 | 1146 | 17/24 | 0 |
| off | middle | sequential | 9.38 | 102.9 | 0.158 / 0.164 | 0.369 / 0.446 | 965 | 9/24 | 0 |
| off | middle | concurrent | 0.95 | 1018.1 | 0.220 / 0.288 | 0.791 / 0.879 | 965 | 9/24 | 0 |
| on | prefix | sequential | 8.07 | 141.9 | 0.051 / 0.058 | 0.334 / 0.342 | 1146 | 17/24 | 0 |
| on | prefix | concurrent | 1.16 | 988.0 | 0.363 / 0.402 | 1.073 / 1.084 | 1146 | 17/24 | 0 |
| on | middle | sequential | 9.42 | 102.5 | 0.159 / 0.161 | 0.375 / 0.444 | 965 | 9/24 | 0 |
| on | middle | concurrent | 1.03 | 940.0 | 0.194 / 0.354 | 0.871 / 0.992 | 966 | 9/24 | 0 |

Peak allocated GPU memory was **12,773 MiB** in both APC-off and APC-on runs.

### Derived comparisons

- Continuous batching raised throughput by **8.1x** (prefix, APC off),
  **6.9x** (prefix, APC on), **9.9x** (middle, APC off), and **9.2x**
  (middle, APC on).
- Prefix placement cut sequential TTFT by **3.1x** (0.050 s vs 0.158 s).
  Under concurrency the gap narrowed to **1.09x** (0.201 s vs 0.220 s).
- APC changed throughput by **-2.0%** sequential prefix, **-15.4%** concurrent
  prefix, **-0.4%** sequential middle, and **-7.7%** concurrent middle. At this
  scale it provided no throughput gain and no measurable memory penalty.
- Within-layout output determinism: APC off vs on was 24/24 exact for both
  sequential layouts and 23/24 for both concurrent layouts. Sequential vs
  concurrent was 23/24 exact. Batching changes at most one of 24 texts.
- The middle layout differs from the prefix baseline by construction, so its
  `exact_match_vs_baseline` of 0.0 is expected. Comparisons are only meaningful
  within the same layout.

## Findings

1. **Batching is the dominant effect.** At 24 sessions the T4 decodes roughly
   1.0k output tokens/s concurrently versus 103-145 tokens/s sequentially.
   Shared-memory tricks do not approach this gain on their own.
2. **Prefix reuse helps prefill, not decode.** It gives a clear sequential TTFT
   win but almost disappears when prefill overlaps with other sessions. It does
   not raise concurrent throughput here.
3. **Position matters.** Moving identical shared content after a private prefix
   raises sequential TTFT from 0.050 s to 0.158 s, which is the cost of losing
   exact-prefix matching. This is the gap that position-independent KV reuse
   targets.
4. **No output-level cross-session leakage was observed.** Across all eight runs
   there were 108 nonce mentions and 24 unique nonces, all owned by the emitting
   session; no foreign nonce appeared. However, the model sometimes omitted its
   own nonce (17/24 prefix, 9/24 middle), so "own nonce present" measures
   instruction-following, not isolation. Silence in generated text is not proof
   of internal KV isolation.
5. **APC is cheap here because the cache is small.** Twenty-four prefixes of
   ~1.2k tokens are negligible next to the model weights and active sequences,
   so the 12,773 MiB peak does not move. The memory/quality trade-off only
   becomes interesting at longer contexts, larger session counts, or divergent
   branches.

## Honest limitations

- One small model, one GPU, one run per condition; no confidence intervals, and
  some differences (notably the APC deltas) are within plausible run-to-run noise.
- Synthetic prompts, greedy decoding, 48 new tokens, ~1.2k-token contexts.
  Decode-dominated workloads understate prefill and cache effects.
- Isolation is inferred from output text only. It cannot detect silent KV reuse
  that does not surface in the sampled tokens.
- vLLM APC is block-granular; results depend on block size and scheduler defaults,
  which were not tuned.
- Offline batch throughput is not a production SLA: no arrival process, queueing,
  or tail-latency requirements.
- The `middle` condition is a control, not a realistic prompt template.

## What would make this novel

The useful open problem is not batching or exact-prefix caching, but
**isolation-aware reuse of non-prefix shared state**: deciding which KV blocks may
be shared across sessions when the shared workspace sits at arbitrary positions,
under a memory budget, while guaranteeing that private tokens cannot leak through
attention. A credible next experiment would compare position-independent reuse
against APC under (a) divergent middle workspaces, (b) adversarial nonce probing,
and (c) a fixed KV-memory budget, measuring both TTFT and leakage. That is a
narrower and more defensible contribution than "shared working memory" in general.
