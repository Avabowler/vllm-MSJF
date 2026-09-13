# PiLLM / MSJF 实现报告：输出长度预测 + 内存自适应调度

> 修改日期：2026-09-12 · 基于 vLLM v1（commit `22258a26b`）· 全部改动策略门控，默认（fcfs）行为零变化
>
> 配套文档：**[EXPERIMENTS.md](./EXPERIMENTS.md)**（实验设计与逐步执行手册，含快速起步 E1–E5）
> 快速使用：`--scheduling-policy msjf --length-predictor-config '{"backend":"oracle"}'`

---

## 1. 项目概述

实现 PiLLM（HiPC 2025, *An LLM Inference System with Integrated Output Length
Prediction and Memory Adaptive Scheduling*）的三个核心组件，并针对 vLLM v1 的
连续批处理架构做了两处超出论文的设计改进：

1. **输出长度预测器**：复用 LLM 自身 prefill 隐状态 → 加权池化 FC →
   MLP 分类的输出长度桶（默认 10 桶）+ ListMLE 排序头。
   另提供 oracle / client / mean 三种零训练后端。
2. **MSJF 调度**（memory-aware shortest job first）：waiting 请求按
   估计 KV 占用（prompt + 预测输出）升序准入；准入受"预约制聚合内存
   控制"约束；预测被低估时动态上调并重排。
3. **P/D 分离集成**：Prefill 节点在 prefill 完成时计算预测，随
   `kv_transfer_params` 免费通道传递给 Decode 节点，使 **D 节点在排队
   前就拥有预测**——这是本实现相对论文单机形态的关键工程优势。

### 与论文的两处主要偏离（及理由）

| 偏离 | 论文做法 | 本实现 | 理由 |
|---|---|---|---|
| 准入控制 | 预测的完整序列放得下才准入（硬边界） | **预约制记账**：running 请求按 `blocks(prompt+预测) − 已持有` 记预约账（×系数 λ），新准入只要求 `空闲块 ≥ 首chunk + Σ预约` | 预测偏大时硬边界会长期压低并发；预约随请求接近完成自动缩水，准入自动放宽，对预测误差鲁棒。论文式准入保留为 `--msjf-full-fit-mode` 供消融 |
| 排队排序的预测来源 | prefill 时预测（隐状态） | 单机模式下 prefill 后的预测只能服务内存准入/抢占/修正；**排队级 SJF 依赖 prompt-only 预测**（oracle/client/mean），PD 分离下隐状态预测恰好覆盖排队时点 | vLLM 连续批处理中请求一旦准入即运行到底，排队决策发生在 prefill 之前 |

---

## 2. 架构与数据流

### 2.1 单机模式

```
                        ┌────────────────────────────────────────────┐
 add_request ──────────►│ Request._init_length_prediction()          │
 (extra_args/kv_params) │   predicted_output_len / bucket / score    │
                        └──────────────┬─────────────────────────────┘
                                       ▼
                        ┌────────────────────────────────────────────┐
                        │ Scheduler._init_msjf_cost()                │
                        │  estimate = 预测 ──缺省──► EWMA均值         │
                        │  effective_output_len = min(est, max_tokens)│
                        │  msjf_cost = prompt + effective            │
                        └──────────────┬─────────────────────────────┘
                                       ▼
   ┌────────────────────── schedule() 每步 ──────────────────────────┐
   │ ① 老化刷新(msjf_aging_factor) + 计算 Σ预约块(reservation_factor) │
   │ ② running 循环（与 fcfs 相同；受害者选择不变）                    │
   │ ③ waiting 循环：                                                │
   │    高水位门(msjf_high_watermark) → 排空背压                      │
   │    队头按 msjf_cost 取出 → MSJF 准入门：                         │
   │       空闲块 − Σ预约 ≥ 门块数(首chunk 或 全预测序列*)            │
   │       失败 → best-fit 跳过(max_backfill_skips) 或 break          │
   │    allocate_slots 失败 → 同样可跳过                             │
   └──────────────────────────┬──────────────────────────────────────┘
                              ▼
   update_from_output(): 低估修正(num_output×overrun_factor，重排堆)
                         完成请求真实长度 → EWMA + 精度指标
   (* msjf_full_fit_mode=true 时门块数=全预测序列，即论文式)
```

### 2.2 P/D 分离模式（推荐部署）

```
┌─ Prefill 节点 ────────────────────────────────┐
│ prefill 最后一个 chunk 完成                   │
│  → gpu_model_runner._predict_output_lens()   │
│    隐状态切片(复用 query_start_loc)           │
│    加权池化(跨chunk增量累积) → MLP头          │
│  → ModelRunnerOutput.predicted_output_lens   │
│  → Scheduler.update_from_output 注入 Request │
│  → _connector_finished() 合并进 kv_transfer_params │
│    {"output_len_prediction": N,              │
│     "output_len_bucket": B, "output_len_rank_score": S} │
└────────────────────┬─────────────────────────┘
                     ▼  代理原样转发该 dict（现有 PD 拓扑零改动）
┌─ Decode 节点 ────────────────────────────────┐
│ Request.__init__ 自动读取预测（排队前！）      │
│ → add_request 计算 msjf_cost → MSJF 排序/准入 │
└──────────────────────────────────────────────┘
```

`kv_transfer_params` 是 vLLM 现成的全链路自由字典通道
（P 侧 `request_finished` 返回 → `EngineCoreOutput` → 代理转发 →
D 侧 `SamplingParams.extra_args`），本实现未改动任何传输代码。

---

## 3. 修改清单

### 3.1 新增文件（8 个）

| 文件 | 内容 |
|---|---|
| `vllm/config/length_predictor.py` | `LengthPredictorConfig`：backend（none/mlp/client/oracle/mean）、checkpoint、num_buckets=10、max_output_len=2048、mlp_hidden_size=4096 |
| `vllm/v1/core/sched/length_predictor.py` | 调度侧 `LengthPredictor`：estimate 回退链（显式预测 → EWMA → None/max_tokens）、`record()` 维护 EWMA 与预测 MAE/桶准确率 |
| `vllm/v1/worker/length_predictor_head.py` | `WeightedPoolingHead`（per-token FC 打分 → softmax 加权池化 → Linear+ReLU trunk → 分类头 + 排序头）；`IncrementalWeightedPooling`（跨 prefill chunk 数值稳定增量累积，O(1) 内存/请求）；safetensors checkpoint 加载（hidden_size 强校验） |
| `tests/v1/core/test_msjf_scheduler.py` | 12 个 CPU 单测 |
| `tests/v1/worker/test_length_predictor_head.py` | 7 个 CPU 单测 |
| `benchmarks/output_length_scheduling/run_benchmark.py` | Poisson 负载回放客户端：经 `/v1/tokenize` 获取真实长度、`--ignore-eos` 精确回放、`kv_transfer_params` 注入预测、抓取 `/metrics` 抢占计数，输出 JCT/TTFT/吞吐 JSON 报告 |
| `benchmarks/output_length_scheduling/prepare_dataset.py` | 数据集格式转换（alpaca/ultrachat/sharegpt/longalign/longwriter → 统一 JSONL） |
| `benchmarks/output_length_scheduling/EXPERIMENTS.md` | 实验设计与执行手册（快速起步 E1–E5、完整对比/消融矩阵、命令模板） |

### 3.2 修改文件（12 个，+802 行 / −6 行）

| 文件 | 改动 |
|---|---|
| `vllm/config/scheduler.py` | `SchedulerPolicy` 增加 `"msjf"`；6 个新参数：`msjf_reservation_factor=0.8`、`msjf_full_fit_mode=False`、`msjf_high_watermark=0.0`、`msjf_max_backfill_skips=4`、`msjf_overrun_factor=1.25`、`msjf_aging_factor=0.0` |
| `vllm/config/length_predictor.py` → `vllm/config/__init__.py`、`vllm/config/vllm.py` | 新配置类导出；`VllmConfig.length_predictor_config` 字段 |
| `vllm/engine/arg_utils.py` | EngineArgs 字段 + CLI：`--scheduling-policy msjf`、`--length-predictor-config`（JSON 自动解析）、`--msjf-*` 6 个参数；`create_engine_config` 传递 |
| `vllm/v1/request.py` | 新字段 `predicted_output_len/predicted_bucket/predicted_rank_score/effective_output_len/output_len_underestimated/msjf_cost`；`_init_length_prediction()` 在构造时从 `extra_args` / `kv_transfer_params` 注入（D 节点排队前即可用） |
| `vllm/v1/core/sched/request_queue.py` | `SchedulingPolicy.MSJF` 枚举；`MSJFRequestQueue`：堆键 `(msjf_cost, arrival_time, seq)`，`update_request()` 支持重排，惰性失效 + 单活条目守卫（`_member_costs`）保证不重弹 |
| `vllm/v1/core/sched/scheduler.py` | 核心集成（详见 §3.3） |
| `vllm/v1/outputs.py` | `ModelRunnerOutput.predicted_output_lens: dict[str, tuple[bucket, len, score]] \| None` |
| `vllm/v1/worker/gpu_model_runner.py` | 预测头在 `load_model()` 后实例化（PP 仅生效于末 rank，TP 各 rank 冗余计算）；`_bookkeeping_sync` 增加 `_predict_output_lens()`：仿 prompt-logprobs 路径用 `query_start_loc` 切片隐状态，prefill 请求按 chunk 增量池化，最后 chunk 跑 MLP 头；抢占重启/中途退出自动清理池化状态 |
| `vllm/v1/metrics/stats.py` | `SchedulerStats` 增加 `msjf_reserved_blocks`、`num_msjf_gate_deferrals`、`num_msjf_underestimated`、`length_prediction_mae`、`length_prediction_bucket_accuracy` |
| `vllm/v1/metrics/loggers.py` | 日志行追加 MSJF 字段；Prometheus 新增 `vllm:msjf_reserved_blocks`、`vllm:msjf_gate_deferrals_total`、`vllm:msjf_underestimated_requests_total`、`vllm:length_prediction_mae_tokens`、`vllm:length_prediction_bucket_accuracy` |
| `tests/v1/core/utils.py` | `create_scheduler` 支持 `length_predictor_config`；`create_requests` 支持 `output_len_predictions`/`max_tokens_list` |

### 3.3 `scheduler.py` 集成点明细

| 位置 | 改动 |
|---|---|
| `__init__` | 解析 msjf 策略、创建 `LengthPredictor`、类级默认值（兼容 `object.__new__` 构造的测试） |
| `schedule()` 开头 | `_apply_msjf_aging()`（成本按等待时间折扣，防饥饿）+ `_msjf_running_reserved_blocks()`（每步一次，λ×Σ running 预测剩余块，`apply_admission_cap=False` 保证不被 ISL 准入上限裁剪） |
| waiting 循环顶 | `msjf_high_watermark` 背压：KV 使用率超阈值暂停一切准入 |
| waiting 循环门 | `_msjf_admission_allowed()`：`空闲块 − Σ预约 ≥ 门块数 + watermark`；门块数 = 首 chunk（默认）或 全预测序列（`full_fit_mode`）。失败→best-fit 跳过（stash 复用现有 `step_skipped_waiting` 机制，请求不丢失）或 break |
| `allocate_slots` 失败路径 | 同样支持 backfill 跳过（msjf 门控） |
| `update_from_output` | ① 注入 `predicted_output_lens`（先于请求循环，同步完成的请求可携带预测走 PD 通道）② 低估修正：超预测时 `effective = min(num_output×overrun_factor, max_tokens)`，计数一次、重排堆 ③ 完成请求喂 `record()` |
| `_connector_finished` | 预测键合并进 `request_finished` 返回的 `kv_transfer_params`（PD 生产端通路） |
| `_select_waiting_queue_for_scheduling` | MSJF 按两队列队头 `msjf_cost` 择优 |
| `make_stats` | 填充 §3.2 的 5 个新统计字段 |

---

## 4. 配置参考

```bash
vllm serve <model> \
  --scheduling-policy msjf \
  --length-predictor-config '{"backend": "oracle"}' \
  # ── 以下均为可选（括号内为默认值）──────────────────────────
  --msjf-reservation-factor 0.8   # 预约系数 λ；0=仅排序；1=最保守
  --msjf-full-fit-mode            # 论文式：全预测序列放得下才准入
  --msjf-high-watermark 0.0       # KV 使用率高背压线；0=关
  --msjf-max-backfill-skips 4     # 每步 best-fit 跳过上限；0=关
  --msjf-overrun-factor 1.25      # 低估上调系数；1=不修正
  --msjf-aging-factor 0.0         # 每秒等待的成本折扣（防饥饿）；0=关
```

`--length-predictor-config` 完整字段：`backend`、`checkpoint`（mlp 必填，
训练脚本产出）、`num_buckets=10`、`max_output_len=2048`、`mlp_hidden_size=4096`。

预测注入通道（三种等价方式）：
1. OpenAI API 请求体 `"kv_transfer_params": {"output_len_prediction": N}`；
2. `SamplingParams.extra_args["output_len_prediction"]`（离线 LLM API）；
3. PD：P 节点 `mlp` 后端自动经 `kv_transfer_params` 传递（无需客户端参与）。

新增监控指标（日志 + Prometheus）：
`vllm:msjf_reserved_blocks`、`vllm:msjf_gate_deferrals_total`、
`vllm:msjf_underestimated_requests_total`、`vllm:length_prediction_mae_tokens`、
`vllm:length_prediction_bucket_accuracy`。

---

## 5. 测试与验证结果（WSL2 Ubuntu-24.04 实跑）

| 项目 | 结果 |
|---|---|
| `tests/v1/core/test_msjf_scheduler.py`（12 用例：队列排序/惰性重排/防重弹/MSJF 准入顺序/fcfs 回归/预约门 defer/backfill/full-fit 对比/高水位/低估修正/PD 通路/EWMA） | ✅ 12/12 |
| `tests/v1/worker/test_length_predictor_head.py`（7 用例：增量池化==一次性池化、重缩放稳定性、桶范围、确定性、分块数参数化） | ✅ 7/7 |
| 既有 `tests/v1/core/test_scheduler.py` 回归 | 162/164 ✅；2 个失败均与本次无关：1 个 PP=2 需双 GPU（环境限制），1 个因本次新增的类属性假设（已修复并重验 ✅） |
| `tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py` + `tests/v1/core/test_priority_preemption_bug.py`（PD 消费端生命周期 / priority 抢占回归） | ✅ 10/10 |
| CLI 冒烟（EngineArgs 解析全部新参数） | ✅ |
| ruff check + format（仓库锁定 v0.14.0） | ✅ 全部通过 |
| mypy 1.20.2（仓库钩子脚本） | 新增代码零错误；触及文件中的报错均为 HEAD 既有（逐条核对过） |

### 已知限制

- 预测头仅集成于 **V1 GPU model runner**（`VLLM_USE_V2_MODEL_RUNNER=1` 的 V2 路径未接，届时 `mlp` 后端静默降级为 mean/max_tokens 兜底）。
- `mlp` 头在单机模式下不参与排队排序（预测晚于排队），PD 模式下无此限制（见 §2.2）。
- MSJF 不与 `priority` 组合；与 async scheduling / chunked prefill / prefix caching 兼容（复用既有机制），speculative decoding 未做专项组合测试。
- 排序带来 SJF 固有的长任务饥饿风险，长尾敏感场景建议开启 `--msjf-aging-factor`。

---

## 6. 实验执行指南（摘要）

**环境**：1× 24GB NVIDIA 卡（Ampere+，原生 Linux）即可跑全部象征性实验；
模型 `Qwen/Qwen3.5-9B`（仓库原生支持，`VLLM_USE_MODELSCOPE=1` 国内直下）；
双 PCIe 卡可选跑 1P1D，无需 NVLink。

**最小实验集（1–2 小时）**：

| # | 内容 | 服务端 | 客户端 |
|---|---|---|---|
| E1 | fcfs 基线 | （默认） | `--prediction none` |
| E2 | MSJF 收益上限 | `--scheduling-policy msjf --length-predictor-config '{"backend":"oracle"}'` | `--prediction oracle` |
| E3 | 可部署形态 | E2 且 `backend:"mean"` | `--prediction none` |
| E4 | 论文式准入消融 | E2 + `--msjf-full-fit-mode` | `--prediction oracle` |
| E5 | 误差敏感性 | E2 | `--prediction noisy --noise-sigma 0.5` |

完整矩阵（对比 C1–C7、消融 A1–A7、指标定义、数据集转换、逐条命令、
结果汇总脚本）见 **[EXPERIMENTS.md](./EXPERIMENTS.md)**。

---

## 7. 附录：本地验证环境备忘（Windows + WSL2）

- 测试环境：WSL2 Ubuntu-24.04，venv 位于仓库内 `.venv`，uv 缓存/HF 缓存/
  pre-commit 缓存均在 E 盘；辅助脚本在 `E:\develop\vllm-tmp\`。
- 网络：HuggingFace 不可达，走 `HF_ENDPOINT=https://hf-mirror.com`；
  pip 走清华镜像（`UV_DEFAULT_INDEX`）；torch 不用 `--torch-backend=auto`。
- editable 安装必须带 `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=<ver>`
  （drvfs 上 setuptools-scm 跑 `git status` 会超时）。
- `arctic-inference`（仅测试依赖、懒加载）为 sdist 包且系统无 g++，安装测试
  依赖时需过滤。
- WSL 内的 git（无 autocrlf）会把全仓库显示为已修改（CRLF 幻象），查 diff
  一律用 Windows 侧 git。
