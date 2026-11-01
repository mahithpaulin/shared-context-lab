# Concurrent LLM Sessions and Shared Working Memory

Status: experiment pending. This report is filled in only from measured runs.

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

See `README.md`. The middle-layout condition is the important control: it holds
the shared content constant while removing prefix-cache eligibility.

## Findings

To be completed from `results.json` with the run URL and timestamps.

## Honest limitations

- One small model, one GPU, synthetic prompts, greedy decoding.
- vLLM prefix caching is block-granular; results depend on block size.
- Offline batch throughput is not the same as a production SLA.
