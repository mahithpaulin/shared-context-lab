"""Concurrent LLM session benchmark: batching vs shared-context KV reuse.

Runs on a single GPU with vLLM's OpenAI-compatible server.
Compares:
  - sequential independent sessions
  - concurrent sessions (continuous batching)
  - shared context placed at the prompt prefix (prefix-cacheable)
  - shared context placed in the middle (prefix cache cannot help)

Outputs results.json and prints a summary. Designed for a Colab T4.
"""
import asyncio
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import httpx
from transformers import AutoTokenizer

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PORT = int(os.environ.get("PORT", "8000"))
BASE = f"http://127.0.0.1:{PORT}"
N_SESSIONS = int(os.environ.get("N_SESSIONS", "24"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "48"))
SEED = 1234


def log(*a):
    print(*a, flush=True)


def build_prompts(tokenizer):
    """Build two prompt layouts with identical content, different ordering."""
    filler = (
        "The archive records measurements from the coastal observatory. "
        "Each entry notes the tide, wind direction, instrument status, and a short remark. "
    )
    workspace = (
        "SHARED WORKSPACE. This block is common to every session. "
        + filler * 40
    )
    ws_ids = tokenizer(workspace, add_special_tokens=False).input_ids

    prompts = {"prefix": [], "middle": []}
    meta = []
    for i in range(N_SESSIONS):
        nonce = f"SESSION-{i:02d}-{hashlib.sha1(str(i + SEED).encode()).hexdigest()[:8].upper()}"
        private = (
            f"Private session code: {nonce}. "
            + filler * 3
        )
        query = (
            " First output the private session code exactly as given above, "
            "then answer in one short sentence: what is 2 + 2?"
        )
        ptr = tokenizer(private + query, add_special_tokens=False).input_ids
        for layout, text in (
            ("prefix", workspace + private + query),
            ("middle", private + workspace + query),
        ):
            ids = tokenizer(text, add_special_tokens=False).input_ids
            prompts[layout].append(text)
            meta.append(
                {
                    "layout": layout,
                    "session": i,
                    "nonce": nonce,
                    "prompt_tokens": len(ids),
                    "shared_tokens": len(ws_ids),
                    "private_tokens": len(ptr),
                }
            )
    return prompts, meta


def start_server(apc: bool):
    args = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(PORT),
        "--max-model-len", "4096",
        "--max-num-seqs", "64",
        "--gpu-memory-utilization", "0.85",
        "--disable-log-requests",
    ]
    if apc:
        args.append("--enable-prefix-caching")
    logf = open(f"/content/server_apc_{int(apc)}.log", "w")
    proc = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT)
    deadline = time.time() + 420
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode}; see server log")
        try:
            r = httpx.get(f"{BASE}/health", timeout=2)
            if r.status_code == 200:
                log(f"server ready apc={apc}")
                return proc, logf
        except Exception:
            pass
        time.sleep(2)
    proc.terminate()
    raise TimeoutError("server did not become healthy")


def stop_server(proc, logf):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    logf.close()
    time.sleep(2)


class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.samples = 0
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True, timeout=5,
                ).strip().splitlines()[0]
                self.peak = max(self.peak, int(out))
                self.samples += 1
            except Exception:
                pass
            time.sleep(0.2)


async def one_request(client, prompt, idx, rec):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": SEED,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    parts = []
    usage = None
    async with client.stream("POST", f"{BASE}/v1/completions", json=payload, timeout=300) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                txt = ch.get("text") or ""
                if txt and ttft is None:
                    ttft = time.perf_counter() - t0
                parts.append(txt)
    total = time.perf_counter() - t0
    rec[idx] = {
        "session": idx,
        "ttft_s": ttft,
        "latency_s": total,
        "output_tokens": (usage or {}).get("completion_tokens"),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "text": "".join(parts),
    }


async def run_concurrent(prompts, concurrency):
    rec = {}
    limits = httpx.Limits(max_connections=128, max_keepalive_connections=64)
    async with httpx.AsyncClient(limits=limits) as client:
        if concurrency == "sequential":
            for i, p in enumerate(prompts):
                await one_request(client, p, i, rec)
        else:
            await asyncio.gather(*[one_request(client, p, i, rec) for i, p in enumerate(prompts)])
    return [rec[i] for i in range(len(prompts))]


def aggregate(rows):
    ttfts = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
    lats = [r["latency_s"] for r in rows]
    toks = sum(r["output_tokens"] or 0 for r in rows)
    wall = max(lats)
    return {
        "n": len(rows),
        "wall_s": wall,
        "throughput_tok_s": toks / wall if wall else None,
        "ttft_p50_s": statistics.median(ttfts) if ttfts else None,
        "ttft_p95_s": sorted(ttfts)[int(0.95 * (len(ttfts) - 1))] if ttfts else None,
        "latency_p50_s": statistics.median(lats),
        "latency_p95_s": sorted(lats)[int(0.95 * (len(lats) - 1))],
        "output_tokens": toks,
    }


def isolation(rows, meta):
    ok = 0
    leaks = []
    for r, m in zip(rows, meta):
        own = m["nonce"] in r["text"]
        others = [x["nonce"] for x in meta if x["nonce"] != m["nonce"] and x["nonce"] in r["text"]]
        if own and not others:
            ok += 1
        if not own or others:
            leaks.append({"session": r["session"], "own_nonce": own, "foreign_nonces": others})
    return {"sessions_with_own_code_only": ok, "total": len(rows), "failures": leaks}


def main():
    t_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts, meta = build_prompts(tokenizer)
    lengths = {k: [m["prompt_tokens"] for m in meta if m["layout"] == k] for k in prompts}
    log("prompt token ranges:", {k: (min(v), max(v)) for k, v in lengths.items()})

    results = {
        "meta": {
            "model": MODEL,
            "n_sessions": N_SESSIONS,
            "max_tokens": MAX_TOKENS,
            "seed": SEED,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "vllm": None,
            "gpu": None,
        },
        "prompt_meta": meta,
        "runs": [],
        "baseline_text": {},
    }
    try:
        import vllm
        results["meta"]["vllm"] = vllm.__version__
    except Exception:
        pass
    try:
        results["meta"]["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip()
    except Exception:
        pass

    for apc in (False, True):
        proc, logf = start_server(apc)
        sampler = GpuSampler()
        sampler.start()
        try:
            for layout in ("prefix", "middle"):
                ps = prompts[layout]
                mt = [m for m in meta if m["layout"] == layout]
                # warm one request so prefix cache has a chance to hold the shared block
                warm = {}
                with httpx.Client(timeout=300) as c:
                    async def _warm():
                        async with httpx.AsyncClient() as ac:
                            await one_request(ac, ps[0], 0, warm)
                    asyncio.run(_warm())
                for conc in ("sequential", "concurrent"):
                    rows = asyncio.run(run_concurrent(ps, conc))
                    agg = aggregate(rows)
                    iso = isolation(rows, mt)
                    run = {
                        "apc": apc,
                        "layout": layout,
                        "concurrency": conc,
                        "aggregate": agg,
                        "isolation": iso,
                        "requests": rows,
                    }
                    results["runs"].append(run)
                    if layout == "prefix" and conc == "sequential":
                        results["baseline_text"] = {r["session"]: r["text"] for r in rows}
                    log(
                        f"apc={apc} layout={layout} {conc}: "
                        f"wall={agg['wall_s']:.2f}s tput={agg['throughput_tok_s']:.1f} tok/s "
                        f"ttft_p50={agg['ttft_p50_s']} ttft_p95={agg['ttft_p95_s']} "
                        f"iso={iso['sessions_with_own_code_only']}/{iso['total']}"
                    )
        finally:
            sampler.stop_flag = True
            sampler.join(timeout=3)
            results.setdefault("gpu_peak_mib", {})[f"apc_{int(apc)}"] = sampler.peak
            stop_server(proc, logf)

    # exact-match rate vs sequential prefix baseline
    base = results.get("baseline_text", {})
    for run in results["runs"]:
        matches = 0
        for r in run["requests"]:
            b = base.get(r["session"])
            if b is not None and b == r["text"]:
                matches += 1
        run["exact_match_vs_baseline"] = matches / len(run["requests"])

    results["meta"]["wall_s"] = time.time() - t_start
    with open("/content/results.json", "w") as f:
        json.dump(results, f, indent=2)

    log("\n=== SUMMARY ===")
    for run in results["runs"]:
        a = run["aggregate"]
        log(
            f"apc={str(run['apc']):5} layout={run['layout']:6} {run['concurrency']:10} "
            f"wall={a['wall_s']:6.2f}s tput={a['throughput_tok_s']:6.1f} "
            f"ttft50={a['ttft_p50_s']:.3f} ttft95={a['ttft_p95_s']:.3f} "
            f"match={run['exact_match_vs_baseline']:.2f}"
        )
    log("gpu_peak_mib", results.get("gpu_peak_mib"))

    # upload artifacts for retrieval outside the notebook UI
    for path in ("/content/results.json", "/content/bench.log"):
        if os.path.exists(path):
            try:
                out = subprocess.check_output(
                    ["curl", "-sS", "-F", f"file=@{path}", "https://0x0.st"],
                    text=True, timeout=120,
                ).strip()
                log(f"ARTIFACT {path} -> {out}")
            except Exception as e:
                log(f"upload failed {path}: {e}")


if __name__ == "__main__":
    main()
