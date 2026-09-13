# MSJF：基于输出长度预测的 vLLM 内存感知调度

本仓库是基于 [vLLM](https://github.com/vllm-project/vllm)（基线 commit `22258a26b`）的研究性修改，
实现 **MSJF（memory-aware shortest job first）调度**：复用 LLM 自身 prefill 隐状态预测输出长度，
按预测 KV 足迹排序排队请求，并以"预约制准入"控制内存，从而消除抢占、降低 JCT/TTFT。

全部改动为纯 Python、由 `--scheduling-policy msjf` 门控，**默认（fcfs）行为零变化**。
上游 vLLM 的完整介绍见 [vllm-project/vllm](https://github.com/vllm-project/vllm)。

- 实现细节：[benchmarks/output_length_scheduling/IMPLEMENTATION_REPORT.md](benchmarks/output_length_scheduling/IMPLEMENTATION_REPORT.md)
- 实验数据：[benchmarks/output_length_scheduling/EXPERIMENT_REPORT_FULL.md](benchmarks/output_length_scheduling/EXPERIMENT_REPORT_FULL.md)

## 核心功能

1. **输出长度预测器**（`LengthPredictor`，五种后端）：
   - `mlp`：复用 prefill 隐状态 → 加权池化 → MLP 桶分类头。预测是 prefill 的副产品，
     零额外模型/前向；增量加权池化跨 chunk 累积，兼容 chunked prefill，抢占重启自动清理。
   - `oracle` / `client`：由请求侧注入真值/外部预测（经 `kv_transfer_params` 或 `extra_args`）。
   - `mean`：已完成请求输出长度的 EWMA 兜底，零成本。
   - `none`：退化为按 `max_tokens` 估计。
2. **MSJF 调度**：waiting 请求按估计 KV 足迹 `prompt + 预测输出`（`--msjf-cost-mode footprint`，
   默认）升序调度；`output` 模式退化为纯 SJF，用于消融。
3. **预约制准入**：running 请求按"预测完整序列还需要的块数 × λ（`--msjf-reservation-factor`，
   默认 0.8）"记预约账，新准入要求 `空闲块 − Σ预约 ≥ 门块数`。预约随请求接近完成自动缩水，
   对预测误差鲁棒（实验：σ=0.5 噪声下仍 0 抢占）。
4. **低估修正与可观测性**：实际输出超过预测时按 `--msjf-overrun-factor`（1.25）上调并重排；
   新增 5 个 Prometheus 指标（预约块数、准入推迟、低估请求数、预测 MAE、桶准确率）。

## 快速开始

```bash
# MSJF + oracle 预测（收益上限参考）
vllm serve <model> --scheduling-policy msjf \
    --length-predictor-config '{"backend": "oracle"}'

# MSJF + 训练好的 MLP 预测头（可部署形态）
vllm serve <model> --scheduling-policy msjf \
    --length-predictor-config '{"backend": "mlp", "checkpoint": "/path/to/ckpt"}'

# 零成本形态：无需任何预测器
vllm serve <model> --scheduling-policy msjf \
    --length-predictor-config '{"backend": "mean"}'
```

### 参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--scheduling-policy msjf` | `fcfs` | 启用 MSJF |
| `--length-predictor-config` | — | JSON：`backend`(none/mlp/client/oracle/mean)、`checkpoint`(mlp 必填)、`num_buckets`(10)、`max_output_len`(2048)、`mlp_hidden_size`(4096) |
| `--msjf-cost-mode` | `footprint` | 排序键：`prompt+预测输出` / `output`（退化纯 SJF） |
| `--msjf-reservation-factor` | 0.8 | 预约系数 λ；0=仅排序，1=最保守 |
| `--msjf-full-fit-mode` | 关 | 论文式硬准入：全预测序列放得下才准入 |
| `--msjf-max-backfill-skips` | 4 | 准入闸阻塞时每步 best-fit 跳过上限 |
| `--msjf-overrun-factor` | 1.25 | 低估上调系数 |
| `--msjf-high-watermark` | 关（0.0） | KV 使用率背压线，超限暂停准入 |
| `--msjf-aging-factor` | 关（0.0） | 等待时间成本折扣，防长任务饥饿 |

预测注入通道（client/oracle 用）：OpenAI 请求体 `"kv_transfer_params": {"output_len_prediction": N}`，
或 `SamplingParams.extra_args["output_len_prediction"]`。PD 分离下 P 节点 `mlp` 预测自动随
`kv_transfer_params` 传给 D 节点，排队前即可用（零传输层改动）。

## 代码改动清单

以基线 `22258a26b` 为准（`git diff 22258a26b --stat --ignore-cr-at-eol`）：**13 个修改文件（+919/−7 行）+ 8 个新增文件**。

### 修改的文件

| 文件 | 改动 |
|---|---|
| `vllm/config/scheduler.py` | `SchedulerPolicy` 增加 `msjf`；新增 7 个 `msjf_*` 配置字段 |
| `vllm/config/vllm.py`、`vllm/config/__init__.py` | `VllmConfig.length_predictor_config` 字段与导出 |
| `vllm/engine/arg_utils.py` | 全部新参数的 CLI 接线；修复 `create_engine_config` 漏传 `length_predictor_config` 的接线 bug |
| `vllm/v1/request.py` | 请求级预测字段 + `_init_length_prediction()`（从 `extra_args`/`kv_transfer_params` 注入） |
| `vllm/v1/core/sched/request_queue.py` | `MSJFRequestQueue`：按 `msjf_cost` 的小顶堆，支持重排与惰性失效 |
| `vllm/v1/core/sched/scheduler.py` | 核心集成：MSJF 准入闸、预约记账、老化、低估修正、预测摄入（约 +317 行） |
| `vllm/v1/outputs.py` | `ModelRunnerOutput.predicted_output_lens` 透传 |
| `vllm/v1/metrics/stats.py`、`vllm/v1/metrics/loggers.py` | 5 个新指标（日志 + Prometheus） |
| `vllm/v1/worker/gpu_model_runner.py` | 预测头集成（旧 runner 路径） |
| `vllm/v1/worker/gpu/model_runner.py` | 同上逻辑向重构后 runner 的移植（服务实际路径）；增量跨 chunk 加权池化 |
| `tests/v1/core/utils.py` | 测试基建支持预测注入 |

### 新增的文件

| 文件 | 内容 |
|---|---|
| `vllm/config/length_predictor.py` | `LengthPredictorConfig` |
| `vllm/v1/core/sched/length_predictor.py` | 调度侧估计回退链（显式预测 → EWMA → max_tokens）+ 精度统计 |
| `vllm/v1/worker/length_predictor_head.py` | 预测头网络（加权池化 + 桶分类）与 safetensors 加载 |
| `tests/v1/core/test_msjf_scheduler.py` | 调度层单测 13 例 |
| `tests/v1/worker/test_length_predictor_head.py` | 预测头 CPU 单测 7 例 |
| `examples/output_length_prediction/train_length_predictor.py` | 预测头训练（冻结主干） |
| `benchmarks/output_length_scheduling/` | 压测客户端、数据集转换、实验报告与结果 JSON |

## 实验结果

**环境**：1× RTX 5090 (32GB) / Python 3.12 / torch 2.13.0+cu130 / Qwen2.5-7B-Instruct (BF16)。
**负载**：ultrachat（短输入）与 LongAlign-10k（prompt 中位 11k tok）；Poisson 到达，
`--ignore-eos` 且 `max_tokens=真实输出长度` 使各策略面对**逐 token 完全相同的负载**；
`--num-gpu-blocks-override` 锁定 KV（≈23 并发占满，KV 峰值 94–100%）构造有效压力。

### 第一组：E1–E6（ultrachat，512 prompts，请求率 r10 / r20）

| 实验 | 配置 | JCT 均值 (s) | JCT p99 (s) | TTFT 均值 (s) | 吞吐 (tok/s) | 抢占 |
|---|---|---|---|---|---|---|
| E1 FCFS 基线 | 默认 | 36.6 / 36.7 | 71.0 / 71.0 | 31.9 / 32.0 | 2349 / 2333 | **681 / 764** |
| E2 MSJF+oracle | `backend:oracle` | 29.0 / 28.1 | 87.6 / 87.8 | 25.1 / 24.2 | 1804 / 1788 | **0 / 0** |
| E3 MSJF+mean | `backend:mean` | 28.6 / 28.6 | 87.1 / 78.5 | 24.7 / 24.7 | 1796 / 2090 | 0 / 2 |
| E4 论文式硬准入 | E2+`--msjf-full-fit-mode` | 28.3 / 27.6 | 93.0 / 91.3 | 24.5 / 23.7 | 1770 / 1737 | 0 / 0 |
| E5 预测噪声 σ=0.5 | 同 E2 + noisy 客户端 | 28.6 / 27.9 | 83.3 / 84.5 | 24.7 / 24.0 | 1902 / 1950 | 0 / 0 |
| **E6 MSJF+mlp** | `backend:mlp`+ckpt | 32.8 / 29.1 | 89.2 / 85.0 | 28.9 / 25.2 | 1855 / 1943 | **0 / 0** |

**判读**：

- **核心收益**（E2 vs E1）：JCT 均值 **−21%/−24%**，TTFT **−21%/−25%**，抢占 **681/764 → 0**。
- **端到端可行**（E6）：训练出的预测头仅 **65.4% 桶准确率 / MAE 106 tok**（10 桶，冻结 7B 主干，
  3 epochs 约 12 分钟）即可完全消除抢占，JCT 较 fcfs 改善 10%（r10）～21%（r20），r20 下接近 oracle 上限。
- **对预测误差鲁棒**：σ=0.5 噪声（E5）与 oracle 几乎无差；零成本 mean 后端（E3）同样近 oracle。
- **预约制优于论文式硬准入**（E4）：硬准入吞吐最低、p99 最高——在预约制已 0 抢占的前提下只剩代价。
- **代价**：饱和压力下以约 20% 吞吐换取上述收益；p99 恶化 ~23% 是 SJF 语义推迟长作业的固有属性。

### 第二组：SJF vs MSJF 解耦（LongAlign 长输入，64 条 × r0.25/r0.5，全 oracle）

| 形态 | JCT 均值 r025 / r05 (s) | 中组（2k–8k tok）JCT | 最长 25%（中位 19.6k）JCT |
|---|---|---|---|
| FCFS | 54.9 / 50.2 & 57.6 / 53.0 | 61.8 / 65.3 | 57.6 / 60.5 |
| SJF（仅输出长度排序） | 43.9 / 38.4 & 47.7 / 42.0 | 40.4 / 43.9 | 56.2 / 60.6 |
| **MSJF 排序**（足迹排序，无准入） | **39.0 / 34.1 & 41.5 / 36.5** | **7.5 / 7.3** | 78.1 / 81.0 |
| MSJF 完整（排序+预约准入） | 39.0 / 34.0 & 41.9 / 36.8 | 7.8 / 7.5 | 77.9 / 81.1 |

- **排序键的净增量**：长输入下 MSJF 比 SJF 的 JCT 再降 **11–13%**；中组请求 JCT **40.4s → 7.5s（5.4×）**
  ——SJF 对输入长度全盲，中等请求被迫陪 3 万 token 大请求排队；MSJF 按足迹先做小的。
- **准入的净增量**：取决于输出形态。输出长、KV 驻留久时（ultrachat 重放）排序类策略抢占
  681～1114 次 → 预约制归零；输出短时（LongAlign）KV 释放快，准入几乎不触发。
- **代价**：最长 25% 请求 JCT +39%，短作业优先语义的固有取舍（可用 `--msjf-aging-factor` 缓解）。

> 以上均为单 seed 单次运行，趋势结论可信，具体数值有 ±5% 量级噪声。逐实验数据见
> [results/](benchmarks/output_length_scheduling/results/) 下的结果 JSON。

## 复现实验

```bash
# 1. 数据准备（alpaca/ultrachat/sharegpt/longalign/longwriter → {prompt, output} JSONL）
python benchmarks/output_length_scheduling/prepare_dataset.py \
    --format ultrachat --input ultrachat_200k_test.jsonl --output uc.jsonl --max-samples 2000

# 2. 训练预测头（冻结主干，仅训头部；E6 产物约 12 分钟）
python examples/output_length_prediction/train_length_predictor.py \
    --model <服务模型路径> --dataset <{prompt,output} jsonl> \
    --eval-dataset <held-out jsonl> --output-dir <ckpt目录> \
    --num-buckets 10 --max-output-len 2048 --epochs 3 --rank-weight 0

# 3. 起服务 + 泊松压测（示例：E2 oracle）
vllm serve <model> --scheduling-policy msjf \
    --length-predictor-config '{"backend": "oracle"}' \
    --max-num-seqs 64 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.9 &
python benchmarks/output_length_scheduling/run_benchmark.py \
    --model <model> --dataset uc.jsonl --num-prompts 512 --request-rate 10 --seed 0 \
    --prediction oracle --output results/E2_msjf_oracle_r10_seed0.json
```

压力校准提示：GPU 过快时策略差异压不出来，用 `--num-gpu-blocks-override` 锁 KV 容量
（本轮 800 块 ≈ 23 并发占满）并以 fcfs 出现抢占验证进入有效压力区。逐实验（E1–E6 与
SJF 对比矩阵）的完整命令见 [EXPERIMENTS.md](benchmarks/output_length_scheduling/EXPERIMENTS.md)。

单元测试：

```bash
python -m pytest tests/v1/core/test_msjf_scheduler.py \
                  tests/v1/worker/test_length_predictor_head.py \
                  tests/v1/core/test_scheduler.py -v
```

## 已知限制与遗留工作

- 预测头仅集成 V1 GPU model runner（V2 路径未接，届时 `mlp` 静默降级为 mean/max_tokens 兜底）。
- 单机模式下 `mlp` 预测晚于排队（排序靠 EWMA 兜底）；PD 分离完整形态（1P1D + nixl 实测）为推荐
  目标，尚未验证。
- MSJF 不与 `priority` 组合；speculative decoding 未做专项组合测试。
- 训练/评估均在 ultrachat 分布，跨分布（LongAlign）未重训预测头。
- 优化路线图（期望值解码、分位数桶、自适应 λ、prefix-caching 感知足迹等）见
  [EXPERIMENT_REPORT_FULL.md §8](benchmarks/output_length_scheduling/EXPERIMENT_REPORT_FULL.md)。

## 文档索引

| 文档 | 内容 |
|---|---|
| [IMPLEMENTATION_REPORT.md](benchmarks/output_length_scheduling/IMPLEMENTATION_REPORT.md) | 实现报告：三组件设计、单机/PD 数据流图、修改清单明细、与论文偏离对照 |
| [EXPERIMENTS.md](benchmarks/output_length_scheduling/EXPERIMENTS.md) | 实验手册：E1–E6 快速起步、完整对比/消融矩阵、逐条命令 |
| [EXPERIMENT_REPORT_FULL.md](benchmarks/output_length_scheduling/EXPERIMENT_REPORT_FULL.md) | 完整实验报告：两组实验全量数据、判读、调参指南、路线图 |
| [VALIDATION_2026-09-12.md](benchmarks/output_length_scheduling/VALIDATION_2026-09-12.md) | 端到端验证报告 v2：预测头训练、修复清单（含接线 bug 与 runner 移植）、E1–E6 复核 |
| [results/](benchmarks/output_length_scheduling/results/) | 全部结果 JSON（含修复前归档 `archive_pre_wiring_fix/`） |

## License

沿用上游 vLLM 的 [Apache License 2.0](LICENSE)。
