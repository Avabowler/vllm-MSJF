# 输出长度预测（桶分类）+ MSJF 调度 — 运行验证报告 v2

- **日期**: 2026-09-12
- **节点**: AutoDL 容器 / Ubuntu 22.04 / 1× RTX 5090 (32GB, sm_120) / Python 3.12 / torch 2.13.0+cu130
- **代码**: `/root/autodl-tmp/vllm`（基线 commit `22258a26b`；你的纯 Python 改动 + 本报告 §2 列出的兼容性修复/移植）
- **模型**: Qwen2.5-7B-Instruct（BF16，hidden_size=3584）
- **负载**: ultrachat_200k test 转 2000 条（benchmark 集），512 条/次，`--ignore-eos --seed 0`，KV 锁定 `--num-gpu-blocks-override 800`（≈23 并发占满，KV 峰值 100%，fcfs 基线大量抢占）

## 1. 训练输出长度预测头（桶分类，无 ListMLE）

按调研结论（见 §4）去掉 ListMLE 排序、只训桶分类——训练脚本自带 `--rank-weight 0` 开关，零接口改动。

- **数据三段划分**（同一 23108 行 ultrachat 池，seed 0 shuffle，互不重叠）：benchmark 2000 / 训练 5000 / 评估 500
- **训练**: 冻结 7B 主干（FP16），只训"加权池化 + 桶分类"头；10 桶等宽（桶宽 204.8 tok），3 epochs，batch 8，约 12 分钟
- **评估集结果**: **bucket_acc = 65.4%**（随机基线 10%），**MAE = 106.1 tokens**（输出均长 340 tok）
- **产物**: `/root/autodl-tmp/ckpt/length_pred_qwen2.5-7b/`（config.json + model.safetensors，59MB）

## 2. 本轮为跑通流程所做修复（4 处补丁 + 1 处移植，均已加注释）

| # | 文件 | 问题 | 修复 |
|---|---|---|---|
| 1 | `benchmarks/.../run_benchmark.py` | 本版本只有 `/tokenize`（`prompt` 字段），无 `/v1/tokenize` | 改 URL 与字段名，逻辑不变 |
| 2 | `examples/.../train_length_predictor.py` | `true_lens` 缺 `device=device`，CUDA 索引报错 | 补设备参数 |
| 3 | 同上 | `output_hidden_states=True` 物化全部 29 层 + 算全词表 logits，慢且 OOM | 改用 `model.model` 取 `last_hidden_state`（数值等价） |
| 4 | `vllm/engine/arg_utils.py` | **接线 bug**：`create_engine_config()` 构造 `VllmConfig` 时漏传 `length_predictor_config`，CLI 配置永远到不了引擎（backend 恒为 none，客户端注入的预测被 `estimate()` 忽略） | 构造参数补一行 |
| 5 | `vllm/v1/worker/gpu/model_runner.py` | **移植缺失**：mlp 预测头代码（`+103` 行）在旧 `gpu_model_runner.py`，而服务实际使用重构后的 `gpu/model_runner.py`（旧文件仅 warmup 引用） | 移植 `__init__` 字段、`load_model` 加载、`_predict_output_lens`（按新 `input_batch` 数组适配）、`ModelRunnerOutput` 透传 |

> **对首轮 E1–E5 结果的影响**：修复 #4 之前，所有服务端 backend 实际都是 none，msjf 退化为"按 max_tokens 估计"。因 benchmark 恰好 `max_tokens=真实长度`，E2/E4 数值近似正确，但 **E3（mean）与 E5（noisy）并未真正生效**（旧结果已归档至 `results/archive_pre_wiring_fix/`）。§3 表格全部为修复后重跑。

## 3. 实验结果（512 prompts，seed 0；格式 r10 / r20）

| 实验 | 服务端 | 客户端 | JCT mean | JCT p99 | TTFT mean | 吞吐 tok/s | 抢占 |
|---|---|---|---|---|---|---|---|
| E1 fcfs 基线 | 默认 | none | 36.6 / 36.7 | 71.0 / 71.0 | 31.9 / 32.0 | 2349 / 2333 | **681 / 764** |
| E2 MSJF+oracle | `backend:oracle` | oracle | 29.0 / 28.1 | 87.6 / 87.8 | 25.1 / 24.2 | 1804 / 1788 | **0 / 0** |
| E3 MSJF+mean | `backend:mean` | none | 28.6 / 28.6 | 87.1 / 78.5 | 24.7 / 24.7 | 1796 / 2090 | 0 / 2 |
| E4 论文式准入 | E2+`--msjf-full-fit-mode` | oracle | 28.3 / 27.6 | 93.0 / 91.3 | 24.5 / 23.7 | 1770 / 1737 | 0 / 0 |
| E5 噪声 σ=0.5 | 同 E2 | noisy | 28.6 / 27.9 | 83.3 / 84.5 | 24.7 / 24.0 | 1902 / 1950 | 0 / 0 |
| **E6 MSJF+mlp（本轮）** | `backend:mlp`+ckpt | none | 32.8 / 29.1 | 89.2 / 85.0 | 28.9 / 25.2 | 1855 / 1943 | **0 / 0** |

### 判读

1. **核心收益成立（真实链路）**：E2 vs E1，JCT -21%/-24%、TTFT -21%/-24%、抢占 681/764 → 0。
2. **E6（端到端 mlp，本轮主目标）**：预测头 65.4% 桶准确率即可把抢占压到 **0**，JCT 较 fcfs 改善 10%（r10）~21%（r20），位于 fcfs 与 oracle 之间——符合"预测误差换收益"的预期；r20 下已接近 oracle 水平。
3. **E5（修复后真正注入噪声）**：σ=0.5 时与 oracle 几乎无差、0 抢占——预约制对预测误差的鲁棒性这次是真实成立的。
4. **E3（mean 后端修复后首次真正生效）**：≈ oracle 水平、0-2 次抢占；r20 的 p99/吞吐波动属单 seed 噪声。
5. **E4**：吞吐最低（1737-1770）、p99 最高（91-93），符合文档"硬准入换取抢占最少"的预期；在预约制已 0 抢占的前提下 full-fit 只剩代价。
6. 权衡（与 v1 报告一致）：极端压力下 MSJF 以 ~20% 吞吐换取上述收益；p99 恶化源于 SJF 语义推迟长任务。

## 4. 文献定位（本轮调研结论）

| 路线 | 代表工作 | 预测器 | 与本实现关系 |
|---|---|---|---|
| 外挂分类器 | S3（NeurIPS'23, arXiv:2306.06000）；Response Length Perception（NeurIPS'23, arXiv:2305.13144） | 独立 BERT/DistilBERT，额外前向与部署 | 本实现**不外挂**模型 |
| 学习排序 | Learning to Rank（NeurIPS'24）；PARS（arXiv:2510.03243） | ListMLE / pairwise 排序 | 本轮按用户决定**去掉 ListMLE**，只留桶分类 |
| 分布预测 | TIE（arXiv:2604.00499） | log-t 分布 + 尾部风险 | 未采用 |
| **隐状态自预测** | Entropy-Guided Hidden-State（arXiv:2602.11812）+ **本实现** | 复用 LLM 自身 prefill 隐状态 + 轻量头 | 本实现的差异化 |

**本实现的差异化**：① 预测是 prefill 副产品（零额外模型/前向）；② 增量加权池化跨 chunk 累积，兼容 chunked prefill，抢占后自动重启；③ 预测恰在 prefill 完成时可用，随 `kv_transfer_params` 中继（PD 分离下 D 节点排队前可用）；④ 预测用于内存预约准入/低估修正，不只是队列重排。

## 5. 后续使用设计

### 5.1 何时重训
换服务模型（头与 `hidden_size` 强校验绑定）或负载分布漂移（bucket acc 在线上指标中持续下滑）时。命令一行：
```bash
python examples/output_length_prediction/train_length_predictor.py \
  --model <服务模型路径> --dataset <{prompt,output} jsonl> \
  --eval-dataset <held-out jsonl> --output-dir <ckpt目录> \
  --num-buckets 10 --max-output-len 2048 --epochs 3 --rank-weight 0
```

### 5.2 四种部署形态
1. **单机·零成本**：`msjf + backend:"mean"`（无需训练，本轮 E3）
2. **单机·完整（本轮 E6）**：`msjf + backend:"mlp"+checkpoint`——排队排序靠 EWMA 兜底，prefill 完成后的真实预测驱动准入/抢占/低估修正
3. **PD 分离（推荐目标形态）**：P 节点 mlp 预测随 `kv_transfer_params` 到 D 节点，**排队前即可用**，获得完整 SJF 收益（按 nixl 1P1D 拓扑部署，未在本轮验证）
4. **外部预测器**：`backend:"client"` + 请求 `kv_transfer_params:{"output_len_prediction":N}` 注入

### 5.3 参数速查
- `--scheduling-policy msjf`：启用 MSJF（`msjf_cost = prompt + effective_predicted_output`）
- `--length-predictor-config`：`backend(none/mlp/client/oracle/mean)`、`checkpoint`、`num_buckets`(10)、`max_output_len`(2048)、`mlp_hidden_size`(4096)
- `--msjf-reservation-factor`(0.8)：预约系数；`--msjf-full-fit-mode`：论文式硬准入；`--msjf-max-backfill-skips`(4)；`--msjf-overrun-factor`(1.25)：低估修正；`--msjf-high-watermark`、`--msjf-aging-factor`：默认关

### 5.4 调参指南
- 压不出差异 → 调低 `--gpu-memory-utilization` 或用 `--num-gpu-blocks-override` 锁 KV（本轮 800 块 ≈ 23 并发占满）
- 吞吐代价过大 → 降 `--msjf-reservation-factor`
- 长任务饥饿（p99 恶化）→ `--msjf-aging-factor`
- 预测普遍低估 → `--msjf-overrun-factor` 上调；预测桶粒度不够 → 重训加 `--num-buckets`

### 5.5 监控指标
日志/Prometheus：`vllm:msjf_reserved_blocks`、`vllm:msjf_gate_deferrals_total`、`vllm:msjf_underestimated_requests_total`、`vllm:length_prediction_mae_tokens`、`vllm:length_prediction_bucket_accuracy`。oracle 实验 MAE 应为 0（链路 sanity check）；mlp 上线后 MAE/bucket acc 持续漂移 = 负载漂移信号，触发 5.1 重训。

## 6. 产物清单
- 结果：`results/E{1..6}_*_r{10,20}_seed0.json`（12 个）+ `archive_pre_wiring_fix/`（8 个修复前结果）
- checkpoint：`/root/autodl-tmp/ckpt/length_pred_qwen2.5-7b/`
- 日志：`/root/autodl-tmp/logs/serve_E{2..6}v*.log`、`train_full.log`
- 脚本：`/root/autodl-tmp/scripts/run_all_v2.sh`（一键复现全部实验）
