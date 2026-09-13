# 输出长度预测 + MSJF 调度：实验设计与执行手册

本目录包含 PiLLM（HiPC 2025, *An LLM Inference System with Integrated Output
Length Prediction and Memory Adaptive Scheduling*）方法在 vLLM v1 上的实现，
以及完整的对比 / 消融实验设计。

- 实现：`--scheduling-policy msjf` + `--length-predictor-config`（见下文）
- 训练预测头：`examples/output_length_prediction/train_length_predictor.py`
- 负载回放与测量：`benchmarks/output_length_scheduling/run_benchmark.py`

---

## 0. 快速起步（象征性实验，推荐从这里开始）

不必完整复现论文。用下面的"最小实验集"约 **1~2 小时**（不含模型下载）即可
看到 MSJF 的核心效果。

### 0.1 模型选择（仓库已原生支持 Qwen3.5）

| 模型 | 显卡要求（FP16/BF16） | 说明 |
|---|---|---|
| **Qwen/Qwen3.5-9B**（主推） | 1× 24GB（4090/A30/3090 等） | 与论文 7B 规模对齐；KV 空间紧，调度差异明显 |
| Qwen/Qwen3.5-2B | 1× 12GB | 快速迭代/功能验证；`--gpu-memory-utilization` 调低制造压力 |
| Qwen/Qwen3.5-35B-A3B（MoE） | 2× 48GB 或 4× 24GB（TP） | 可选；解码快，KV 压力主导 |

本仓库 model registry 已注册 `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM`，
无需任何额外适配。国内下载权重二选一：

```bash
# 方式 A：ModelScope（推荐，国内直连）
VLLM_USE_MODELSCOPE=1 vllm serve Qwen/Qwen3.5-9B ...

# 方式 B：HF 镜像
export HF_ENDPOINT=https://hf-mirror.com
vllm serve Qwen/Qwen3.5-9B ...
```

> 注意：`mlp` 预测头与模型的 hidden_size 绑定，换模型需用训练脚本重训；
> `oracle` / `client` / `mean` 三种后端与模型无关，开箱即用。
> 因此**象征性实验直接用 oracle/mean，不必训练任何模型**。

### 0.2 数据集（替代论文的 ShareGPT/Alpaca，均已转为统一格式）

用 `prepare_dataset.py` 把任意常见指令数据集转成 benchmark 需要的
`{"prompt", "output"}` JSONL（`output` 提供"真实输出长度"用于回放）：

| 数据集 | 角色（对应论文） | 下载 | 转换 |
|---|---|---|---|
| **ultrachat_200k**（HuggingFaceH4） | 通用真实负载（替代 ShareGPT） | `hf-mirror.com` | `--format ultrachat` |
| **alpaca-gpt4-data-zh**（中文） | 短输出负载（替代 Alpaca） | ModelScope `AI-ModelScope/alpaca-gpt4-data-zh` | `--format alpaca` |
| **LongAlign-10k**（THUDM） | **长输入**（4k~32k prompt），直接检验 MSJF 的"输入内存感知" | `hf-mirror.com` | `--format longalign` |
| **LongWriter-6k**（THUDM） | **长输出**（2k~32k），压测低估修正与高水位 | `hf-mirror.com` | `--format longwriter` |

```bash
# 示例：处理 ultrachat 取 2000 条
python prepare_dataset.py --format ultrachat \
    --input ultrachat_200k_test.jsonl --output uc.jsonl --max-samples 2000
```

### 0.3 最小实验集（象征性，单卡即可）

统一负载参数：`--num-prompts 512 --ignore-eos --seed 0`，
请求率从 `--request-rate 5` 爬到 KV 接近打满（日志里
`GPU KV cache usage` 峰值 > 90% 即进入有效区间）。

| 编号 | 内容 | 服务端 | 客户端 |
|---|---|---|---|
| E1 | 功能跑通 + 基线 | 默认（fcfs） | `--prediction none` |
| E2 | MSJF 收益 | `--scheduling-policy msjf --length-predictor-config '{"backend":"oracle"}'` | `--prediction oracle` |
| E3 | 真实可部署形态 | 同 E2 但 `backend:"mean"` | `--prediction none` |
| E4 | 论文式准入消融 | E2 + `--msjf-full-fit-mode` | `--prediction oracle` |
| E5 | 预测误差敏感性 | E2 | `--prediction noisy --noise-sigma 0.5` |

每个实验在 2~3 个请求率下各跑一次即可出趋势。判读方式：

- E2 vs E1：`jct_mean_s` / `jct_p99_s` / `ttft_mean_s` 下降、
  `preemptions` 大幅减少 → MSJF 生效；
- E3 vs E2：`mean` 后端无先验信息，收益会小一些但应为正；
- E4 vs E2：抢占最少但吞吐可能下降（论文式硬准入的代价）；
- E5 vs E2：`--noise-sigma` 越大退化越缓则说明预约制鲁棒（对比
  `--msjf-reservation-factor 0` 的 E2 变体更明显）。

换 LongAlign（长输入）再跑一遍 E1/E2：长短输入混排时 MSJF 相对
"纯按输出长度 SJF"的优势即论文的核心卖点（输入也占 KV）。

---

## 1. 实现回顾（各组件对应的开关）

| 组件 | 论文对应 | 开关 |
|---|---|---|
| 输出长度预测（客户端/真值/统计均值） | SLM-P / 预测器 | `--length-predictor-config '{"backend":"client"/"oracle"/"mean"}'` |
| 输出长度预测（隐状态 MLP 头，论文忠实实现） | 加权池化 + MLP 分类 + ListMLE 排序 | `backend:"mlp"` + `checkpoint`（用训练脚本产出） |
| MSJF 排序（按估计 KV 占用升序准入） | MSJF / Best-Fit | `--scheduling-policy msjf` |
| 预约制聚合准入（改进项，见 §4.1） | 论文为"预测全序列放得下才准入" | `--msjf-reservation-factor 0.8`（默认）；论文式用 `--msjf-full-fit-mode` |
| Best-fit backfill | Best-Fit 打包 | `--msjf-max-backfill-skips 4`（默认） |
| 低估修正（超预测后上调并重排） | 动态调整优先级 | `--msjf-overrun-factor 1.25`（默认） |
| 高水位排空背压 | 内存自适应 | `--msjf-high-watermark 0.95`（默认关闭） |
| 老化防饥饿 | 论文未涉及 | `--msjf-aging-factor`（默认关闭） |
| P/D 分离：P 节点预测随 `kv_transfer_params` 传给 D 节点 | 系统集成 | KV connector 现有通道，无需额外开关 |

关键机制说明（与论文的差异）：

- **排队级 SJF 需要排队前可得的预测**。vLLM 请求一旦准入即运行到底，
  因此隐状态预测头（prefill 完成时出结果）在单机模式下主要用于
  内存准入/抢占决策与低估修正；排队排序收益要靠 client/oracle/mean
  等 prompt-only 预测来源。**PD 分离是本实现的推荐部署**：预测在
  P 节点 prefill 完成时计算，随 `kv_transfer_params` 到达 D 节点，
  D 节点排队前即可用（见 docs 的 disagg_prefill 拓扑）。
- **预约制准入**：对 running 请求按
  `blocks(prompt + effective_pred) − 已持有块` 记账，乘以
  `msjf_reservation_factor(λ)` 后作为新准入必须保留的空闲块。
  请求接近完成时预约自动缩水 → 准入自动放宽；预测偏大只浪费少量
  余量而非压死并发。论文式严格准入保留为 `--msjf-full-fit-mode`
  供消融对比。

新增监控指标（日志 + Prometheus）：
`vllm:msjf_reserved_blocks`、`vllm:msjf_gate_deferrals_total`、
`vllm:msjf_underestimated_requests_total`、
`vllm:length_prediction_mae_tokens`、
`vllm:length_prediction_bucket_accuracy`。
日志行会额外输出 `MSJF reserved blocks / gate deferrals / underestimations /
Pred len MAE / Pred bucket acc` 字段。

---

## 2. 环境准备

> 象征性实验直接看 §0（Qwen3.5 + 现成数据集）。本节是论文级完整复现的配置。

Linux GPU 环境（论文用 1×A30 24GB，FP16；任何 ≥24GB 的卡均可）：

```bash
# 1) 安装（本仓库工作副本）
uv venv --python 3.12 && source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# 2) 数据集：用 prepare_dataset.py 转换（见 §0.2）。
#    论文原版数据集：ShareGPT（长输入）
#    https://huggingface.co/datasets/anon8231489123/ShareGPT52K
#    Alpaca（短输入）https://huggingface.co/datasets/tatsu-lab/alpaca
#    转为每行 {"prompt": ..., "output": ...} 的 JSONL。

# 3) （可选）训练隐状态预测头 —— 单独 GPU 任务
#    模型与所服务模型一致（hidden_size 强校验），如 Qwen3.5-9B：
python examples/output_length_prediction/train_length_predictor.py \
    --model Qwen/Qwen3.5-9B \
    --dataset ultrachat_train.jsonl --eval-dataset ultrachat_test.jsonl \
    --output-dir ./ckpt/length_pred \
    --num-buckets 10 --max-output-len 2048 --epochs 3
```

启动服务（单机模式，`<CFG>` 为每个实验的调度配置，见 §3/§5）：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --max-num-seqs 64 --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.9 <CFG>
```

> 复现论文负载形态的关键：`--gpu-memory-utilization` 调到 KV 空间
> "并发 20~40 个请求即接近打满"的程度（论文 A30 24GB 上的状态），
> 否则内存自适应机制不会被激活，各策略差异将很小。

---

## 3. 对比实验（复现论文 Table II / Fig.6-7 + 扩展）

统一负载：`run_benchmark.py --dataset ShareGPT.jsonl --ignore-eos`，
请求率 `--request-rate {5,10,15,20,25,30}`，每组 `--seed 0..2` 跑 3 次取均值。
所有实验逐条记录：JCT mean/p50/p99、TTFT mean/p50/p99、吞吐、preemptions。

| # | 配置 | 服务端 `<CFG>` | 客户端 `--prediction` |
|---|---|---|---|
| C1 | vLLM-FCFS（论文 baseline） | （默认） | none |
| C2 | vLLM-priority | `--scheduling-policy priority` | none |
| C3 | MSJF+oracle（调度收益上限，≈论文 PiLLM-Oracle） | `--scheduling-policy msjf --length-predictor-config '{"backend":"oracle"}'` | oracle |
| C4 | MSJF+client（外部预测器服务） | `backend:"client"` | none（预测由外部服务注入 extra_args）或 oracle |
| C5 | MSJF+mean（零成本运行时兜底） | `backend:"mean"` | none |
| C6 | MSJF+mlp（论文完整 PiLLM） | `backend:"mlp","checkpoint":"./ckpt/length_pred"`（需 `--length-predictor-config`） | none（mlp 在 prefill 后出预测） |
| C7 | MSJF+noisy（预测误差敏感性，见消融 A5） | 同 C3 | `noisy --noise-sigma {0.1,0.25,0.5}` |

预期结论（对齐论文）：C3/C6 相对 C1 的 Avg JCT 降幅随请求率升高而扩大
（论文 19.6%→54.7%），TTFT 降幅更显著（44.9%→89.4%）；ShareGPT（长输入）
上 MSJF 的内存感知收益明显高于 Alpaca（短输入，退化为普通 SJF）。

**PD 分离实验（推荐）**：按
`tests/v1/kv_connector/nixl_integration/` 的 1P1D 拓扑
（P 实例 + D 实例 + `toy_proxy_server.py` 路由），P 侧启动加
`--scheduling-policy msjf --length-predictor-config '{"backend":"mlp",...}'`，
D 侧启动加 `--scheduling-policy msjf`（+ admission 参数）。
P 节点预测自动随 kv_transfer_params 到达 D 节点并参与排队，
这是完整 PiLLM 的目标形态。负载用 `run_benchmark.py` 打到 proxy 端口。

---

## 4. 消融实验

### A1. 准入控制策略（核心消融，验证 §1 改进项）
| 变体 | 配置 |
|---|---|
| 仅排序，无准入控制 | `--scheduling-policy msjf --msjf-reservation-factor 0` |
| 预约制（本文方案） | `--msjf-reservation-factor {0.5, 0.8, 1.0}`（默认 0.8） |
| 论文式全序列准入 | `--msjf-full-fit-mode`（配合 C3 的排序） |

观察：preemptions、KV 峰值利用率、JCT。预期：仅排序在高压下抢占数
接近 FCFS；预约制以少量利用率换取抢占大幅下降；全序列准入抢占最少
但吞吐/utilization 损失最大，且对预测精度最敏感。

### A2. backfill
`--msjf-max-backfill-skips 0`（关） vs 默认 4 vs 16（开大）。
观察排队深度高时的 JCT p99 与 gate deferrals 计数。

### A3. 低估修正
`--msjf-overrun-factor 1.0`（关闭，等价不修正） vs 1.25（默认） vs 1.5。
用 `--prediction noisy`（σ=0.5）制造低估场景，观察
`vllm:msjf_underestimated_requests_total` 与抢占数。

### A4. 排序头（ranker）的贡献
训练两个 checkpoint：完整头（分类+排序）vs 仅分类头（`--rank-weight 0`）。
`backend:"mlp"` 下对比单请求 JCT 方差与 p99（排序头影响同桶内顺序）。
论文 Table IV（Kendall's Tau 0.71→0.73 / 0.76→0.82）为参考。

### A5. 预测精度敏感性（噪声注入）
C3 配置下 `--prediction noisy --noise-sigma {0, 0.1, 0.25, 0.5, 1.0}`。
画出 JCT/抢占 随 σ 的退化曲线，对比"仅排序"与"排序+预约制"两条线：
预约制应显著压低退化斜率（这是它相对论文式硬准入的核心优势）。

### A6. 桶数
重训 `--num-buckets {5,10,20}`，其余同 C6。观察 bucket_acc 与最终 JCT。

### A7. 高水位与老化（稳健性，论文未涉及）
- `--msjf-high-watermark {0, 0.9, 0.95}`：与 A1 组合，看最坏情况保护。
- `--msjf-aging-factor {0, 5, 20}`：长请求 JCT p99 与饥饿（最大等待时间）。

---

## 5. 每个实验的具体命令模板

```bash
# <NAME>=实验名  <CFG>=服务端配置（§3表）  <CLIENT>=客户端预测参数（§3表）
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 \
    --max-num-seqs 64 --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.9 <CFG> &
sleep 120   # 等待就绪
python benchmarks/output_length_scheduling/run_benchmark.py \
    --model Qwen/Qwen2.5-7B-Instruct --dataset ShareGPT.jsonl \
    --num-prompts 1024 --request-rate 10 --seed 0 \
    <CLIENT> --output results/<NAME>_r10_seed0.json
kill %1
```

汇总脚本（生成对比表）：

```bash
python - <<'EOF'
import json, glob, collections
rows = collections.defaultdict(dict)
for f in glob.glob("results/*.json"):
    r = json.load(open(f))
    key = r["config"]["prediction"] + "@" + str(r["request_rate"])
    rows[key][f] = (r["jct_mean_s"], r["ttft_mean_s"], r["preemptions"])
for k, v in sorted(rows.items()):
    print(k, {p: m for p, m in v.items()})
EOF
```

---

## 6. 功能测试（提交前在本仓库跑）

```bash
# MSJF 调度 + 预测器后端 + 准入门/backfill/低估修正（CPU）
.venv/bin/python -m pytest tests/v1/core/test_msjf_scheduler.py -v

# 预测头增量池化数值一致性（CPU）
.venv/bin/python -m pytest tests/v1/worker/test_length_predictor_head.py -v

# 既有调度回归（确认 fcfs/priority 行为零变化）
.venv/bin/python -m pytest tests/v1/core/test_scheduler.py -v

# PD 消费端准入生命周期回归（CPU）
.venv/bin/python -m pytest tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py -v
```

## 7. 注意事项

- `--ignore-eos` 回放让各策略面对**完全相同**的负载（论文同款方法）；
  `--no-ignore-eos` 则是 EOS 自然终止（更真实，但跨策略对比噪声更大）。
- `mean` 后端冷启动阶段没有历史长度，估计回退到 `max_tokens`；
  建议先低请求率预热 1~2 分钟再计时。
- `mlp` 后端的 checkpoint 的 `hidden_size` 必须与所服务模型一致
  （加载时强校验）；换模型必须重训。
- 指标 `vllm:length_prediction_*` 只统计带预测的完成请求；
  oracle 实验的 MAE 应恒为 0，可作为链路正确性的 sanity check。
