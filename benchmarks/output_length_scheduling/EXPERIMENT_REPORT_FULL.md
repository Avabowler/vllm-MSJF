# 输出长度预测 + MSJF 调度 — 完整实验报告

- **日期**: 2026-09-12 ~ 2026-09-13
- **节点**: AutoDL 容器 / Ubuntu 22.04 / 1× NVIDIA RTX 5090 (32GB, sm_120) / 208 核 CPU / 754G 内存
- **软件栈**: Python 3.12.3（独立 venv）/ torch 2.13.0+cu130 / vLLM 基线 commit `22258a26b`（本仓库修改版，editable 安装，预编译内核来自官方 cu130 nightly wheel）
- **被测模型**: Qwen2.5-7B-Instruct（BF16，hidden_size=3584，ModelScope 下载）
- **代码基线**: 用户在 `E:\develop\vllm` 的修改（+802/−6 行纯 Python：`--scheduling-policy msjf`、`LengthPredictor`（none/mlp/client/oracle/mean 五后端）、预约制准入、低估修正、增量池化预测头及 vLLM 集成）

---

## 1. 实验总览

| 组 | 目标 | 数据集 | 规模 |
|---|---|---|---|
| 第一组（E1–E6） | 验证"输出长度预测 + MSJF 调度"整套系统相对 FCFS 的收益，及各预测后端/准入策略的贡献 | ultrachat_200k test（短输入，2000 条池） | 512 条 × 2 请求率 × 6 配置 |
| 第二组（SJF 对比） | 解耦 MSJF 相对 SJF 的两个差异维度：排序键（输出长度 vs KV 足迹）与预约制准入 | LongAlign-10k（长输入，prompt 中位 11k tok）+ ultrachat 对照 | 64 条 × 2 请求率 × 4 形态；对照 4 轮 |

### 1.1 统一实验方法

1. **负载回放**：`vllm serve` 起服务，压测客户端以泊松到达发 OpenAI 兼容请求。关键设计：`--ignore-eos` 且 `max_tokens = 数据集标注的真实输出长度`，使**每种调度策略面对的负载逐 token 完全相同**，JCT/TTFT 差异全部来自调度。
2. **压力构造**：RTX 5090 解码太快，默认配置 KV 打不满、策略差异无法显现。用 `--num-gpu-blocks-override` 锁定 KV 容量（ultrachat 800 块 = 12.8K tokens ≈ 23 并发占满；LongAlign 3500 块 = 56K tokens ≈ 5 并发占满），实测 KV 峰值 94%~100%，并以 fcfs 出现大量抢占验证进入有效压力区。
3. **预测隔离**：排序/准入对预测质量的依赖用 `backend:"oracle"`（客户端经 `kv_transfer_params` 注入真值）剥离；再用 mean/noisy/mlp 后端分别考察真实情况。
4. **指标**：逐请求 JCT（完成时间）、TTFT（首 token）、输出长度；聚合 JCT/TTFT 均值与 p99、吞吐；抢占数从 `/metrics` 的 `vllm:num_preemptions_total` 差分获取；分组实验另按 prompt 长度分桶统计。
5. **公平性**：同数据集、同 seed、同 KV 容量；每次换配置重启服务并等待健康检查。

---

## 2. 预测头训练（桶分类，无 ListMLE）

依据文献调研（S3、Response Length Perception 均为桶分类路线；ListMLE 属 NeurIPS'24 Learning-to-Rank 路线），按用户决定去掉排序损失，仅训桶分类。训练脚本自带 `--rank-weight 0` 开关，零接口改动。

### 2.1 数据三段划分（防泄漏）

23108 条有效 ultrachat 样本，seed 0 洗牌后切片：**benchmark 2000**（与主实验 uc.jsonl 完全一致）/ **训练 5000** / **评估 500**（互相不重叠）。

### 2.2 训练配置与结果

| 项 | 值 |
|---|---|
| 结构 | 加权池化（逐 token 打分 → softmax 加权求和）→ FC trunk(3584→4096) → 10 桶分类头 |
| 桶划分 | 等宽 10 桶（桶宽 204.8 tok，覆盖 0~2048） |
| 训练 | 冻结 7B 主干（FP16 前向），仅训头部；AdamW lr=1e-4；3 epochs；batch 8；约 12 分钟 |
| 正则 | 仅交叉熵（`--rank-weight 0`，未用 ListMLE） |
| **评估集结果** | **bucket_acc = 65.4%**（随机基线 10%），**MAE = 106.1 tokens**（输出均长 340 tok） |
| 产物 | `/root/autodl-tmp/ckpt/length_pred_qwen2.5-7b/`（config.json + model.safetensors，59MB） |

先以 400 条 × 1 epoch 冒烟验证流程（eval bucket_acc 58.8% / MAE 134.2），确认可跑通后再全量训练。

---

## 3. 验证过程中发现并修复的问题

以下修复均已在代码中加注释标记（除 #6 为本轮新增的功能性开关）：

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| 1 | `benchmarks/output_length_scheduling/run_benchmark.py` | 本版本服务端只有 `/tokenize`（请求体 `prompt` 字段），无 `/v1/tokenize` | 改 URL 与字段名，语义不变 |
| 2 | `examples/output_length_prediction/train_length_predictor.py` | `true_lens` 缺 `device=device`，ListMLE 分组在 CUDA 上崩溃 | 补设备参数 |
| 3 | 同上 | `output_hidden_states=True` 物化全部 29 层隐状态并计算全词表 logits，慢且 OOM | 改用 `model.model` 取 `last_hidden_state`（数值等价，提速 10 倍+） |
| 4 | `vllm/engine/arg_utils.py` | **接线 bug**：`create_engine_config()` 构造 `VllmConfig` 时漏传 `length_predictor_config`，CLI 配置永远到不了引擎（backend 恒为 none，客户端注入预测被 `estimate()` 忽略） | 构造参数补一行 |
| 5 | `vllm/v1/worker/gpu/model_runner.py` | **移植缺失**：预测头代码在旧 `gpu_model_runner.py`，而服务实际走重构后的 `gpu/model_runner.py` | 移植 `__init__` 字段、`load_model` 加载、`_predict_output_lens`（按新 `input_batch` 数组适配）、`ModelRunnerOutput` 透传 |
| 6 | `vllm/config/scheduler.py` + `arg_utils` + `scheduler.py` | （本轮新增）`--msjf-cost-mode footprint\|output`：output 模式排序键退化为纯输出长度（纯 SJF），用于与 MSJF 的公平对比 | 新开关，默认 footprint 行为不变；附单测 |

**重要提醒**：修复 #4 之前，所有服务端预测后端实际都是 `none`（首轮 E3 mean / E5 noisy 并未真正生效，E2/E4 因 benchmark 恰好 `max_tokens=真值` 而近似正确）。首轮结果已归档于 `results/archive_pre_wiring_fix/`，本报告全部采用修复后重跑的数据。

功能测试：`test_msjf_scheduler.py` 13/13（12 原有 + 1 新增 cost-mode 用例）、`test_length_predictor_head.py` 7/7 全过；GPU 冒烟（中英文生成）正常。

---

## 4. 第一组实验：E1–E6（ultrachat 短输入）

**设置**：512 条，`--ignore-eos --seed 0`，KV 锁 800 块（≈23 并发占满），请求率 10 / 20 req/s。oracle 实验由客户端注入真值；E6 由服务端 mlp 头在 prefill 完成时预测。格式：**r10 / r20**。

| 实验 | 服务端配置 | 客户端 | JCT 均值 | JCT p99 | TTFT 均值 | 吞吐 tok/s | 抢占 |
|---|---|---|---|---|---|---|---|
| E1 FCFS 基线 | 默认 | none | 36.6 / 36.7 | 71.0 / 71.0 | 31.9 / 32.0 | 2349 / 2333 | **681 / 764** |
| E2 MSJF+oracle | `backend:oracle` | oracle | 29.0 / 28.1 | 87.6 / 87.8 | 25.1 / 24.2 | 1804 / 1788 | **0 / 0** |
| E3 MSJF+mean | `backend:mean` | none | 28.6 / 28.6 | 87.1 / 78.5 | 24.7 / 24.7 | 1796 / 2090 | 0 / 2 |
| E4 论文式硬准入 | E2+`--msjf-full-fit-mode` | oracle | 28.3 / 27.6 | 93.0 / 91.3 | 24.5 / 23.7 | 1770 / 1737 | 0 / 0 |
| E5 预测噪声 σ=0.5 | 同 E2 | noisy | 28.6 / 27.9 | 83.3 / 84.5 | 24.7 / 24.0 | 1902 / 1950 | 0 / 0 |
| E6 **MSJF+mlp** | `backend:mlp`+ckpt | none | 32.8 / 29.1 | 89.2 / 85.0 | 28.9 / 25.2 | 1855 / 1943 | **0 / 0** |

### 判读

1. **核心收益成立**：E2 vs E1，JCT 均值 **-21%/-24%**、TTFT **-21%/-25%**、抢占 **681/764 → 0**。
2. **真实预测头（E6）**：65.4% 桶准确率即可完全消除抢占，JCT 较 fcfs 改善 10%（r10）~21%（r20），位于 fcfs 与 oracle 上限之间，r20 接近 oracle。
3. **鲁棒性**：E5（50% 预测噪声）与 E2 几乎无差；E3（零成本 mean 兜底）同样 0 抢占。预约准入对预测误差不敏感。
4. **准入消融（E4）**：硬准入吞吐最低（1737~1770）、p99 最高，在预约制已 0 抢占的前提下只剩代价——验证了预约制（λ=0.8）相对论文式全序列准入的优势。
5. **代价**：极端压力下（KV≈23 并发、系统饱和）以 ~20% 吞吐换上述收益；p99 恶化 ~23% 是 SJF 语义推迟长作业的固有属性。

---

## 5. 第二组实验：SJF vs MSJF（解耦排序键与准入）

### 5.1 设计

MSJF 与 SJF 有两处不同，用 `--msjf-cost-mode` 开关解耦：

| 形态 | 排序键 | 预约准入 |
|---|---|---|
| FCFS | 到达序 | 默认水位 |
| SJF | 仅预测输出长度 | 关 |
| MSJF 排序 | prompt + 预测输出（KV 足迹） | 关 |
| MSJF 完整 | prompt + 预测输出 | 开（λ=0.8） |

- **SJF vs MSJF 排序**（准入都关）= 排序键的净增量
- **MSJF 排序 vs MSJF 完整** = 预约准入的净增量

数据集：**LongAlign-10k**（长输入主战场：过滤 prompt+output ≤32.7k tok 后 303 条，prompt 中位 11k、输出中位 201；KV 锁 3500 块 ≈ 5 并发占满，`--max-model-len 32768`，64 条 × 请求率 0.25/0.5）+ **ultrachat 对照**（短输入）。全部 oracle 预测。

### 5.2 LongAlign 结果（格式 JCT均值 / TTFT均值 / 抢占）

| 形态 | r025 | r05 | 中组(2k-8k) JCT r025/r05 | 最长25%（中位 19.6k）JCT r025/r05 |
|---|---|---|---|---|
| FCFS | 54.9 / 50.2 | 57.6 / 53.0 | 61.8 / 65.3 | 57.6 / 60.5 |
| SJF | 43.9 / 38.4 | 47.7 / 42.0 | 40.4 / 43.9 | 56.2 / 60.6 |
| MSJF 排序 | **39.0 / 34.1** | **41.5 / 36.5** | **7.5 / 7.3** | 78.1 / 81.0 |
| MSJF 完整 | 39.0 / 34.0 | 41.9 / 36.8 | 7.8 / 7.5 | 77.9 / 81.1 |

（r05 行的 TTFT 列为 JCT 均值，完整 p99 与抢占：FCFS 100.7/0、SJF 108.8/4、MSJF 排序 107.0/1、MSJF 完整 107.3/3）

### 5.3 ultrachat 对照（JCT / TTFT / 抢占）

| 形态 | r10 | r20 |
|---|---|---|
| FCFS | 36.6 / 31.9 / 681 | 36.7 / 32.0 / 764 |
| SJF | 25.2 / 18.7 / 1110 | 25.5 / 19.0 / 1105 |
| MSJF 排序 | 25.7 / 19.9 / 1114 | 25.3 / 19.5 / 1109 |
| MSJF 完整（=E2） | 29.0 / 25.1 / 0 | 28.1 / 24.2 / 0 |

### 5.4 判读

1. **排序键的净增量（SJF → MSJF 排序）**：长输入负载 JCT **-11%/-13%**；核心证据在中组（2k-8k）：**40.4 → 7.5 秒（5.4 倍）**——SJF 对输入全盲，中等请求被迫陪 3 万 token 大请求排队；MSJF 按足迹先做小的，中组 TTFT 仅 1.7s。
2. **代价**：最长 25%（中位 1.96 万 prompt）JCT 从 56.2 推迟到 78.1（**+39%**）——短作业优先语义的固有取舍，且 MSJF 比 SJF 更甚（输入大也"算长"）。
3. **准入的净增量**：依赖负载的输出形态。输出长、KV 驻留久时（ultrachat 重放）排序类策略 681~1114 次抢占 → 预约制 0；输出短时（LongAlign，输出中位 201 tok）KV 释放快，抢占天然少，准入几乎不触发（MSJF 排序 ≈ 完整）。
4. **短均匀输入**：SJF ≈ MSJF 排序（25.2 vs 25.7），与理论退化预期一致。

### 5.5 结论：哪个更合适

- **MSJF 是 SJF 的严格超集**（`--msjf-cost-mode output` 即退化 SJF），建议默认 MSJF。
- 长短输入混排或内存受限 → MSJF（排序 + 准入双收益）；短均匀输入 → SJF 模式足够，且省去预测依赖。
- 最大输入请求会被推迟 ~35-40%，对长输入敏感的 SLA 需配 `--msjf-aging-factor` 或窗口配额（见 §8）。

---

## 6. 文献定位

| 路线 | 代表工作 | 预测器 | 与本实现关系 |
|---|---|---|---|
| 外挂分类器 | S3（NeurIPS'23, arXiv:2306.06000）；Response Length Perception（NeurIPS'23, arXiv:2305.13144） | 独立 BERT/DistilBERT，额外前向与部署 | 本实现不外挂模型 |
| 学习排序 | Efficient LLM Scheduling by Learning to Rank（NeurIPS'24）；PARS（arXiv:2510.03243） | ListMLE / pairwise 排序 | 本工作已去掉排序头，保留桶分类 |
| 分布预测 | TIE（arXiv:2604.00499） | log-t 分布 + 尾部风险 | 未采用 |
| **隐状态自预测** | Entropy-Guided Hidden-State（arXiv:2602.11812）+ **本实现** | 复用 LLM 自身 prefill 隐状态 + 轻量桶分类头 | 本实现所处的新兴路线 |

本实现差异化：① 预测是 prefill 副产品（零额外模型/前向）；② 增量加权池化跨 chunk 累积，兼容 chunked prefill、抢占后自动重启；③ 预测恰在 prefill 完成时可用，随 `kv_transfer_params` 中继（PD 分离下 D 节点排队前可用）；④ 预测用于内存预约准入/低估修正，而非仅队列重排。

---

## 7. 部署与使用设计

### 7.1 何时重训预测头
换服务模型（hidden_size 强校验绑定）或线上 `length_prediction_bucket_accuracy` 持续下滑（负载漂移）：

```bash
python examples/output_length_prediction/train_length_predictor.py \
  --model <服务模型路径> --dataset <{prompt,output} jsonl> \
  --eval-dataset <held-out jsonl> --output-dir <ckpt目录> \
  --num-buckets 10 --max-output-len 2048 --epochs 3 --rank-weight 0
```

### 7.2 四种部署形态
1. **单机·零成本**：`msjf + backend:"mean"`（无预测器，E3 验证）
2. **单机·完整**：`msjf + backend:"mlp"+checkpoint`（E6 验证：排队靠 EWMA 兜底，prefill 后真预测驱动准入/抢占/低估修正）
3. **PD 分离（推荐目标形态，未验证）**：P 节点预测随 `kv_transfer_params` 先于排队到达 D 节点，完整 SJF 排序收益
4. **外部预测器**：`backend:"client"` + 请求注入

### 7.3 参数速查
- `--scheduling-policy msjf`；`--length-predictor-config '{"backend":..,"checkpoint":..,"num_buckets":10,"max_output_len":2048,"mlp_hidden_size":4096}'`
- `--msjf-cost-mode footprint|output`（本轮新增）；`--msjf-reservation-factor`(0.8)；`--msjf-full-fit-mode`；`--msjf-max-backfill-skips`(4)；`--msjf-overrun-factor`(1.25)；`--msjf-high-watermark`、`--msjf-aging-factor`（默认关）

### 7.4 调参指南
| 症状 | 动作 |
|---|---|
| 压不出策略差异 | 调低 `--gpu-memory-utilization` 或 `--num-gpu-blocks-override` 锁 KV |
| 吞吐代价过大 | 降 `--msjf-reservation-factor` |
| 长输入饥饿（p99 恶化） | `--msjf-aging-factor` 或准入窗口配额 |
| 预测普遍低估 | 上调 `--msjf-overrun-factor`；桶粒度不足 → 加 `--num-buckets` 重训 |

### 7.5 监控指标
`vllm:msjf_reserved_blocks`、`vllm:msjf_gate_deferrals_total`、`vllm:msjf_underestimated_requests_total`、`vllm:length_prediction_mae_tokens`、`vllm:length_prediction_bucket_accuracy`。oracle 实验 MAE 应为 0（链路 sanity check）；mlp 上线后指标漂移即重训信号。

---

## 8. 优化路线图（按投入产出排序）

| 阶段 | 优化点 | 针对的实测问题 |
|---|---|---|
| 快赢 | **P1** 期望值解码（桶分布期望代替 argmax 中心）；**P4** 双估计（排序用期望、准入用 P75 分位）；**S1** 自适应预约系数 λ（按抢占率/KV 水位反馈，低压旁路） | E6 与 oracle 的差距（量化误差）；饱和压力 -20% 吞吐 |
| 短期 | **P2** 分位数桶重训；**P6** 以 Kendall's tau 为一等指标；**S2** 长输入防饥饿（默认调优 aging / 窗口配额） | 桶精度 65.4%；p99 +39% |
| 中期 | **S3** prefix-caching 感知足迹（独占块 + pred）；**S5** 大 prefill 预算感知；**P3** 增量池化提前出粗预测（单机形态获得排队收益） | 内存感知不彻底；单机预测晚到的结构限制 |
| 战略 | **S6** PD 分离完整形态（1P1D + nixl 拓扑实测） | 完整 SJF 收益 + 预约准入，论文目标形态 |

---

## 9. 产物与复现

- 结果 JSON：`benchmarks/output_length_scheduling/results/`（E1–E6、LongAlign 矩阵、ultrachat 对照共 20 个）+ `archive_pre_wiring_fix/`（修复前 8 个）
- 预测头 checkpoint：`/root/autodl-tmp/ckpt/length_pred_qwen2.5-7b/`
- 一键复现脚本：`/root/autodl-tmp/scripts/run_all_v2.sh`（E1–E6）、`run_la_matrix.sh`（SJF 对比矩阵）、`run_benchmark_grouped.py`（分组版压测）、`run_experiment.sh`
- 服务日志：`/root/autodl-tmp/logs/serve_*.log`
- 数据集：`/root/autodl-tmp/data/`（uc.jsonl、uc_train/eval.jsonl、longalign_256.jsonl）

## 10. 效度说明（Caveats）

1. 每配置为单 seed 单次运行，趋势结论可信，具体数值有 ±5% 量级噪声；r20 的 E3 p99/吞吐波动属单次噪声。
2. 长输入组的最长 25% 分析基于 16 个请求的子样本。
3. 结果目录曾出现文件可见性异常（沙箱延迟），个别结果以终端捕获数据为准，均已交叉核对。
4. 训练/评估均在 ultrachat 分布上；跨分布（如 LongAlign）未重训预测头，mlp 在长输入场景的表现未单独测量。
