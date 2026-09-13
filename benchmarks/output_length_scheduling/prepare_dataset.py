# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert common instruction datasets to the benchmark JSONL format.

Produces ``{"prompt": str, "output": str}`` lines consumed by
``run_benchmark.py`` (the "output" field provides the true output length for
replay) and by the predictor training script.

Supported --format values:
  alpaca     {"instruction", "input", "output"}
             (alpaca-gpt4-data-zh, Alpaca-GPT4, ...)
  ultrachat  {"prompt", "messages": [{role, content}, ...]} or messages only
             (HuggingFaceH4/ultrachat_200k, ...)
  sharegpt   {"conversations": [{"from": human/gpt, "value": ...}, ...]}
             (ShareGPT mirrors, BelleGroup exports, ...)
  longalign  {"conversations": [{"from": "human"/"gpt", "value": ...}]}
             (THUDM/LongAlign-10k — long inputs)
  longwriter {"messages": [{role, content}, ...], "length": N}
             (THUDM/LongWriter-6k — long outputs)

Examples:
    python prepare_dataset.py --format ultrachat \
        --input ultrachat_200k.jsonl --output uc.jsonl --max-samples 2000
    python prepare_dataset.py --format alpaca \
        --input alpaca_gpt4_data_zh.json --output alpaca_zh.jsonl
"""

import argparse
import json
import random
from pathlib import Path


def first_human(messages: list[dict], role_key: str, content_key: str) -> str:
    for m in messages:
        if m.get(role_key) in ("human", "user"):
            return m.get(content_key, "")
    return ""


def first_assistant(messages: list[dict], role_key: str, content_key: str) -> str:
    for m in messages:
        if m.get(role_key) in ("assistant", "gpt"):
            return m.get(content_key, "")
    return ""


def to_pairs(record: dict, fmt: str) -> tuple[str, str] | None:
    if fmt == "alpaca":
        prompt = record.get("instruction", "")
        if record.get("input"):
            prompt = f"{prompt}\n{record['input']}"
        return prompt or None, record.get("output", "")
    if fmt in ("ultrachat", "longwriter"):
        messages = record.get("messages") or []
        return (
            record.get("prompt") or first_human(messages, "role", "content") or None,
            first_assistant(messages, "role", "content"),
        )
    if fmt in ("sharegpt", "longalign"):
        conversations = record.get("conversations") or []
        return (
            first_human(conversations, "from", "value") or None,
            first_assistant(conversations, "from", "value"),
        )
    raise ValueError(f"Unknown format: {fmt}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        required=True,
        choices=["alpaca", "ultrachat", "sharegpt", "longalign", "longwriter"],
    )
    parser.add_argument(
        "--input", required=True, help="Source file (.jsonl or .json array)"
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--seed", type=int, default=0, help="Shuffle seed before --max-sampling"
    )
    args = parser.parse_args()

    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    records: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        loaded = json.loads(line)
        records.extend(loaded if isinstance(loaded, list) else [loaded])

    random.seed(args.seed)
    random.shuffle(records)

    written = 0
    with Path(args.output).open("w", encoding="utf-8") as out:
        for record in records:
            if args.max_samples is not None and written >= args.max_samples:
                break
            pair = to_pairs(record, args.format)
            if not pair or not pair[0] or not pair[1]:
                continue
            prompt, output = pair
            out.write(
                json.dumps({"prompt": prompt, "output": output}, ensure_ascii=False)
                + "\n"
            )
            written += 1
    print(f"Wrote {written} samples to {args.output}")


if __name__ == "__main__":
    main()
