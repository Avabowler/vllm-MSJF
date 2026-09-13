# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark for output-length-prediction-aware scheduling (MSJF vs FCFS).

Drives a running ``vllm serve`` instance with a ShareGPT/Alpaca-style JSONL
trace at a Poisson arrival rate and reports TTFT / JCT (e2e) / throughput,
plus preemption counts scraped from the server's Prometheus endpoint.

The trace's "output" field determines the true output length. In the default
replay mode (``--ignore-eos``) every request generates exactly its true
length, so different scheduling policies see an identical workload and the
comparison is exact. Client-side predictions are injected through the
``kv_transfer_params`` field of the OpenAI-compatible API:

  --prediction none     no predictions (baseline / mean backend on server)
  --prediction oracle   prediction = true length
  --prediction noisy    prediction = true length * N(1, sigma)  (--noise-sigma)

See EXPERIMENTS.md in this directory for the full experiment matrix.

Example:
    python run_benchmark.py --host localhost --port 8000 \
        --dataset ShareGPT.jsonl --num-prompts 512 --request-rate 10 \
        --prediction oracle --output result_fcfs.json
"""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import aiohttp


async def resolve_true_lengths(server, dataset, num_prompts, session) -> list[int]:
    """Tokenize the trace outputs with the server's /tokenize endpoint."""
    lengths = []
    for record in dataset[:num_prompts]:
        # 本节点部署的 vllm 暴露 /tokenize（请求体用 prompt 字段），非 /v1/tokenize+text
        body = {"model": server["model"], "prompt": record["output"],
                "add_special_tokens": False}
        async with session.post(
            f"{server['url']}/tokenize", json=body
        ) as resp:
            resp.raise_for_status()
            lengths.append((await resp.json())["count"])
    return lengths


async def send_request(session, server, record, true_len, args, rng, send_time):
    body = {
        "model": server["model"],
        "prompt": record["prompt"],
        "max_tokens": true_len if args.ignore_eos else args.max_tokens,
        "ignore_eos": args.ignore_eos,
        "temperature": 0.0,
        "stream": True,
    }
    prediction = None
    if args.prediction == "oracle":
        prediction = true_len
    elif args.prediction == "noisy":
        prediction = int(true_len * rng.gauss(1.0, args.noise_sigma))
    if prediction is not None:
        body["kv_transfer_params"] = {"output_len_prediction": prediction}

    ttft = None
    async with session.post(
        f"{server['url']}/v1/completions", json=body
    ) as resp:
        resp.raise_for_status()
        async for chunk in resp.content:
            if chunk.strip() and ttft is None:
                ttft = time.perf_counter() - send_time
    e2e = time.perf_counter() - send_time
    return {"ttft": ttft, "e2e": e2e, "true_len": true_len}


async def scrape_counter(session, server, metric) -> float:
    try:
        async with session.get(f"{server['url']}/metrics") as resp:
            text = await resp.text()
    except aiohttp.ClientError:
        return 0.0
    for line in text.splitlines():
        if line.startswith(metric):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


async def run(args):
    rng = random.Random(args.seed)
    dataset = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset = dataset[: args.num_prompts]
    server = {"url": f"http://{args.host}:{args.port}", "model": args.model}
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    preconnect = aiohttp.TCPConnector(limit=args.num_prompts)

    async with aiohttp.ClientSession(timeout=timeout, connector=preconnect) as session:
        true_lens = await resolve_true_lengths(server, dataset, len(dataset), session)
        preemptions_before = await scrape_counter(
            session, server, "vllm:num_preemptions_total"
        )
        start = time.perf_counter()

        async def paced_send(record, true_len):
            # Poisson process: exponential inter-arrival times.
            await asyncio.sleep(rng.expovariate(args.request_rate))
            send_time = time.perf_counter()
            return await send_request(session, server, record, true_len, args, rng,
                                      send_time)

        results = await asyncio.gather(
            *(paced_send(rec, tl) for rec, tl in zip(dataset, true_lens))
        )
        wall = time.perf_counter() - start
        preemptions_after = await scrape_counter(
            session, server, "vllm:num_preemptions_total"
        )

    e2es = sorted(r["e2e"] for r in results)
    ttfts = sorted(r["ttft"] for r in results if r["ttft"] is not None)
    output_tokens = sum(r["true_len"] for r in results)

    def pct(sorted_vals, p):
        return sorted_vals[min(int(p * len(sorted_vals)), len(sorted_vals) - 1)]

    report = {
        "config": vars(args),
        "completed": len(results),
        "request_rate": args.request_rate,
        "duration_s": round(wall, 2),
        "output_throughput_tps": round(output_tokens / wall, 1),
        "jct_mean_s": round(sum(e2es) / len(e2es), 3),
        "jct_p50_s": round(pct(e2es, 0.50), 3),
        "jct_p99_s": round(pct(e2es, 0.99), 3),
        "ttft_mean_s": round(sum(ttfts) / len(ttfts), 3),
        "ttft_p50_s": round(pct(ttfts, 0.50), 3),
        "ttft_p99_s": round(pct(ttfts, 0.99), 3),
        "preemptions": int(preemptions_after - preemptions_before),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True,
                        help="JSONL with 'prompt' and 'output' fields")
    parser.add_argument("--num-prompts", type=int, default=512)
    parser.add_argument("--request-rate", type=float, default=10.0,
                        help="Poisson arrival rate, requests/second")
    parser.add_argument("--prediction",
                        choices=["none", "oracle", "noisy"], default="none")
    parser.add_argument("--noise-sigma", type=float, default=0.25,
                        help="Relative std-dev for --prediction noisy")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Cap when not using --ignore-eos")
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument("--no-ignore-eos", dest="ignore_eos",
                        action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="Write the JSON report to this file")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
