# Shared-Context Lab

A reproducible benchmark of concurrency and shared-context KV reuse for
autoregressive LLM sessions, built for a single Colab T4 GPU.

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
- `results.json` - produced by a run (not committed until verified)

Synthetic text only; no private data.
