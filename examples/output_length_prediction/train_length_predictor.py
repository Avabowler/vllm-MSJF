# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Train the PiLLM-style output-length predictor head.

Extracts last-layer hidden states of prompt tokens from a (frozen) causal
LLM via HuggingFace ``transformers``, then trains a weighted-pooling FC
layer + MLP trunk with two heads:

- a bucket classifier (cross-entropy over ``--num-buckets`` output-length
  buckets), and
- a scalar ranker trained with the ListMLE listwise loss over requests in
  the same bucket (ideal order: descending true output length).

Only the head is trained; the LLM is inference-only.

Example:
    python train_length_predictor.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset ShareGPT.jsonl \
        --output-dir ./length_predictor_ckpt \
        --num-buckets 10 --max-output-len 2048

Dataset format: JSONL with a "prompt" and an "output" field (ShareGPT /
Alpaca exports are easy to convert; see --prompt-key/--output-key).
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class WeightedPoolingHead(nn.Module):
    """Same architecture as vllm.v1.worker.length_predictor_head."""

    def __init__(self, hidden_size: int, mlp_hidden_size: int, num_buckets: int):
        super().__init__()
        self.pool_scorer = nn.Linear(hidden_size, 1, bias=False)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(mlp_hidden_size, num_buckets)
        self.ranker = nn.Linear(mlp_hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = self.pool_scorer(hidden_states).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * hidden_states).sum(dim=1)
        return self.trunk(pooled)


def listmle(scores: torch.Tensor) -> torch.Tensor:
    """ListMLE loss for one group; `scores` is a 1-D tensor whose ideal
    order is descending. Returns the listwise negative log-likelihood."""
    max_val = scores.max()
    exp = torch.exp(scores - max_val)
    # Cumulative sums from the tail: denominator of each rank's softmax.
    denom_rev = torch.flip(torch.cumsum(torch.flip(exp, [0]), 0), [0])
    return -(scores - max_val - torch.log(denom_rev)).sum()


def load_dataset(path: str, prompt_key: str, output_key: str, tokenizer, args):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            prompt = str(record[prompt_key])
            output = str(record[output_key])
            prompt_ids = tokenizer(
                prompt, truncation=True, max_length=args.max_prompt_len
            )["input_ids"]
            output_len = len(tokenizer(output)["input_ids"])
            if not 0 < output_len <= args.max_output_len or not prompt_ids:
                continue
            samples.append((prompt_ids, output_len))
    return samples


@torch.inference_mode()
def extract_hidden_states(model, tokenizer, batch, device):
    """Last-layer hidden state of every prompt token: [B, L, H] (padded)."""
    max_len = max(len(ids) for ids in batch)
    pad_id = tokenizer.pad_token_id or 0
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ids in enumerate(batch):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attention_mask[i, : len(ids)] = 1
    # fix: 直接取 base transformer 的 last_hidden_state，避免物化全部层隐状态
    # 和计算 LM head logits（数值等价于 output_hidden_states=True 的最后一层）
    out = model.model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
    )
    return out.last_hidden_state, attention_mask.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Target LLM (HF id or path)")
    parser.add_argument("--dataset", required=True, help="JSONL train file")
    parser.add_argument("--eval-dataset", help="JSONL eval file (optional)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--output-key", default="output")
    parser.add_argument("--num-buckets", type=int, default=10)
    parser.add_argument("--max-output-len", type=int, default=2048)
    parser.add_argument("--max-prompt-len", type=int, default=2048)
    parser.add_argument("--mlp-hidden-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hidden_size = model.config.hidden_size
    head = WeightedPoolingHead(
        hidden_size, args.mlp_hidden_size, args.num_buckets
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)

    bucket_width = args.max_output_len / args.num_buckets

    def bucket_of(output_len: int) -> int:
        return min(int(output_len / bucket_width), args.num_buckets - 1)

    train = load_dataset(
        args.dataset, args.prompt_key, args.output_key, tokenizer, args
    )
    print(f"Loaded {len(train)} training samples.")
    train.sort(key=lambda s: s[1])  # group similar lengths for ListMLE

    step = 0
    for epoch in range(args.epochs):
        random.shuffle(train)
        for i in range(0, len(train) - args.batch_size + 1, args.batch_size):
            batch = train[i : i + args.batch_size]
            hidden, _ = extract_hidden_states(
                model, tokenizer, [b[0] for b in batch], device
            )
            labels = torch.tensor(
                [bucket_of(b[1]) for b in batch], device=device
            )
            # fix: 补 device=device，否则排序分组里 CPU/CUDA 索引不匹配
            true_lens = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=device)

            trunk = head(hidden.float())
            ce = F.cross_entropy(head.classifier(trunk), labels)

            # ListMLE over same-bucket groups (ideal: descending true length).
            rank_loss = hidden.new_zeros(())
            preds = head.classifier(trunk).argmax(dim=-1)
            for b in preds.unique():
                idx = (preds == b).nonzero(as_tuple=True)[0]
                if idx.numel() < 2:
                    continue
                order = idx[true_lens[idx].argsort(descending=True)]
                rank_loss = rank_loss + listmle(head.ranker(trunk[order]).squeeze(-1))

            loss = ce + args.rank_weight * rank_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 50 == 0:
                acc = (preds == labels).float().mean().item()
                print(f"epoch {epoch} step {step}: ce={ce.item():.4f} "
                      f"rank={rank_loss.item():.4f} bucket_acc={acc:.3f}")

    if args.eval_dataset:
        evaluate(model, tokenizer, head, args, device, bucket_of)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w") as f:
        json.dump(
            {
                "model": args.model,
                "hidden_size": hidden_size,
                "mlp_hidden_size": args.mlp_hidden_size,
                "num_buckets": args.num_buckets,
                "max_output_len": args.max_output_len,
            },
            f,
            indent=2,
        )
    from safetensors.torch import save_file

    save_file(
        {k: v.contiguous() for k, v in head.state_dict().items()},
        str(out_dir / "model.safetensors"),
    )
    print(f"Saved checkpoint to {out_dir}")


@torch.inference_mode()
def evaluate(model, tokenizer, head, args, device, bucket_of):
    eval_samples = load_dataset(
        args.eval_dataset, args.prompt_key, args.output_key, tokenizer, args
    )
    bucket_width = args.max_output_len / args.num_buckets
    correct = abs_err = n = 0
    for i in range(0, len(eval_samples), args.batch_size):
        batch = eval_samples[i : i + args.batch_size]
        hidden, _ = extract_hidden_states(
            model, tokenizer, [b[0] for b in batch], device
        )
        trunk = head(hidden.float())
        preds = head.classifier(trunk).argmax(dim=-1)
        for row, (_, true_len) in enumerate(batch):
            pred_bucket = int(preds[row].item())
            pred_len = int((pred_bucket + 0.5) * bucket_width)
            correct += int(pred_bucket == bucket_of(true_len))
            abs_err += abs(pred_len - true_len)
            n += 1
    print(f"Eval: bucket_acc={correct / n:.4f} MAE={abs_err / n:.1f} tokens (n={n})")


if __name__ == "__main__":
    main()
