# Shared-Context Lab

A reproducible benchmark of concurrency and shared-context KV reuse for
autoregressive LLM sessions, built for a single Colab T4 GPU.

Measured on 2026-11-01 (Tesla T4, vLLM 0.32.0, Qwen2.5-0.5B-Instruct):
continuous batching reached 940-1168 output tokens/s at 24 sessions, 6.9-9.9x
the sequential rate. Prefix reuse cut sequential TTFT 3.1x but gave no
concurrent throughput gain. See `REPORT.md` for the full table and limitations.

## Question

Can one ordinary autoregressive model serve multiple concurrent sessions with
coordinated processing and shared working memory? Where is genuine headroom
beyond existing inference engines?

## Modes

| Mode | What changes |
|---|---|
| sequential | one request at a time, no batching |
| concurrent | all sessions submitted at once (vLLM continuous batching) |
| prefix layout | shared workspace placed at the start of every prompt |
| middle layout | shared workspace placed after a private session prefix |
| APC off/on | vLLM automatic prefix caching disabled/enabled |

The prefix layout is cache-friendly. The middle layout carries identical
information but defeats prefix matching, isolating the cost of position.
A warm request seeds the cache before each measurement.

## Metrics

- wall time and output throughput
- TTFT p50/p95, end-to-end latency p50/p95
- peak GPU memory
- session isolation (own nonce present, no foreign nonces)
- exact-match rate against the sequential prefix baseline

## Run

Open `experiment.ipynb` in Colab with a T4 runtime and run all cells.
`bench.py` can also run directly on any CUDA host with vLLM installed.

## Files

- `bench.py` - benchmark harness
- `experiment.ipynb` - Colab entry point
- `REPORT.md` - findings and design implications
- `results.json` - raw output of the fixed 2026-11-01 run
  (sha256 `22affa9aff1774cd82be213f1231f66e9286670718d5bb54e7d0d1901129595d`)

Synthetic text only; no private data.
