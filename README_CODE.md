# 联邦 RGB-X 跟踪（ViPT 提示参数）代码包

打包日期：2026-10-07。内容：**全部代码，不含数据**。

## 包里有什么

| 路径 | 说明 |
|---|---|
| `ViPT_bucketgroups_e47644b/` | S-A 所在分支 `bucketgroups` 在提交 **e47644b** 的干净快照（去掉了演示动图）。这是实验里 S-A、fna、分桶使用的代码 |
| `ViPT_all_branches.bundle` | 主仓库全部分支的 git 历史（`git clone ViPT_all_branches.bundle ViPT` 即可还原）。含 `main`（b5bfa29，FedAvg 等基线）、`bucketgroups`（72ba847 → e47644b）、`ditto`、`fedper`、`e2_thermal_downsample`、`equiv_check`、`legacy-steps50`、`transfer_matrix` |
| `scripts/` | 评估与分析脚本（评估守护、判定脚本、配对区间、汇总页生成等）。脚本里的路径是我们机器上的 `D:\cvresearch\...`，换机器需要改 |
| `results/` | 最终汇总页 `final_summary_for_klint.html`、逐方法对比总表、判定表、深度配对区间 |
| `requirements_freeze.txt` | 本机评估环境（conda 环境 `sdstrack`）的 `pip freeze` |

## 没有放进来的（太大或不是代码）
- **数据集**：LasHeR、DepthTrack、VisEvent（共约 375 GB）。
- **预训练权重**：`pretrained/OSTrack_ep0300.pth.tar`（来自 OSTrack/ViPT 官方发布）。
- **检查点与联邦状态**：约 90 GB，在 `ViPT/output/` 和 `cloud_backup/`，有需要可以单独传。
- **第三方评估工具包**：LasHeR 官方 MATLAB 工具包、VisEvent_SOT_Benchmark。我们用 `scripts/lasher_eval.py`（对 MATLAB 工具包的逐行 Python 移植）评热红外，事件用 `scripts/visevent_eval.py`，深度用 vot-toolkit 0.5.3。

## 核心代码在哪
- 联邦服务器与聚合：`ViPT_bucketgroups_e47644b/lib/federated/`
  - `server.py`：训练入口（`--variant`、`--bucket-groups`、`--selective-participation`、`--seed`……）
  - `aggregator.py`：聚合规则。**S-A 的改动在 `Aggregator.aggregate`**：对非 `aux_specific` 的分桶组，把 `aux_type == NO_AUX`（纯 RGB）的客户端排除在候选之外，并在 `param_group_stats.jsonl` 里记录跳过原因
  - `param_groups.py`：四组可训练参数（`aux_specific` 664,416 / `rgb_shared` 73,836 / `fusion` 82,944 / `norm` 18,432，合计 839,628）
  - `client.py`、`client_data.py`、`split_builder.py`、`warmup.py`
  - 单元测试：`test_server.py`（46 项）、`test_aggregator.py`（29 项），无 GPU、无数据即可运行
- 客户端划分：`ViPT_bucketgroups_e47644b/configs/fed/client_splits/lasher_depthtrack_visevent_seed42_v2.json`（12 个客户端，每个数据集 4 个，各含 1 个纯 RGB 客户端）

## 方法与对应命令（实验里用的）
环境变量：`CVR_ROOT`（仓库所在根目录）、`CVR_DATA`（数据根目录）。所有训练：120 轮、每轮 50 步、12 个客户端全部参与。

```bash
COMMON="--rounds 120 --local-steps 50 --workers 0 --clients-per-round 1.0 --save-every 20 --export-at-end"
# FedAvg（主仓库 main，b5bfa29）
python lib/federated/server.py --variant fedavg --seed 42 $COMMON --out-dir OUT/fedavg_s42
# 分桶（只分 aux_specific，默认）
python lib/federated/server.py --variant modality --seed 42 $COMMON --out-dir OUT/bucket_s42
# fna：三组分桶，纯 RGB 参与
python lib/federated/server.py --variant modality --bucket-groups aux_specific,fusion,norm --seed 42 $COMMON --out-dir OUT/fna_s42
# S-A：三组分桶，纯 RGB 不参与 fusion/norm
python lib/federated/server.py --variant modality --bucket-groups aux_specific,fusion,norm --selective-participation --seed 42 $COMMON --out-dir OUT/sa_s42
# 独立联邦（每个模态单独，4 个客户端；以热红外为例）
python lib/federated/server.py --variant modality --clients rgbt_000,rgbt_001,rgbt_002,rgbt_003 --seed 42 $COMMON --out-dir OUT/indep_thermal_s42
```
三个 seed：42、0、123。

## 评估
- 全部在同一台机器（本机 RTX 4080 SUPER）上评，第 120 轮，不挑检查点；热红外用 `scripts/lasher_eval.py`（AUC 为主指标）。
- 热红外：`RGBT_workspace/test_rgbt_mgpus.py --dataset_name LasHeR`；事件：`RGBE_workspace/test_rgbe_mgpus.py --dataset_name VisEvent`；深度：`Depthtrack_workspace/run_offline_depthtrack.py` + `vot analysis`。
- 判定与汇总：`scripts/judge_SA.py`、`scripts/depth_pair_SA.py`、`scripts/final_summary.py`、`scripts/full_table_local.py`。

## 结果（原划分，三个 seed 的均值，第 120 轮）

| 方法 | 热红外 AUC | 事件 SR | 深度 F |
|---|---:|---:|---:|
| FedAvg | 46.55 | 52.93 | 53.85 |
| 分桶（aux_specific） | 47.47 | 56.46 | 56.02 |
| 独立联邦 | 48.25 | 56.45 | 55.28 |
| fna | 43.70 | 54.21 | 55.31 |
| **S-A** | **48.90** | **56.38** | **55.96** |
| 冻结判定线 | ≥ 48.0 | ≥ 56.07 | ≥ 55.0 |

S-A 三条线按三 seed 均值全部满足。S-A − fna（纯 RGB 是否参与 fusion/norm 分桶的单独效果）：热红外 +5.21，事件 +2.16，深度 +0.65（不可辨）。逐 seed 数字与区间见 `results/final_summary_for_klint.html`。

## 重要限制
- **划分只有一份**；每个方法 3 个 seed；骨干只有 ViPT；客户端是单机模拟的 12 个。
- 逐序列区间只覆盖测试序列抽样，不覆盖训练 seed 与划分的波动。
- 深度判定线 55.0 是我方提议，尚未得到确认。
- “通信量少”来自 ViPT 只训练提示参数，不是分桶或 S-A 带来的。

## 完整性指纹（供复现核对）
- `lib/` 与 `configs/` 全部被跟踪文件（去行尾后逐个取 sha256 再合并）：`6ccd0d9ce9d8d779`
- 划分文件 `lasher_depthtrack_visevent_seed42_v2.json` 的 sha256 前 16 位：`b57921b0c6289800`
- S-A 提交：`e47644b`（单元测试开关关闭时与旧行为逐位一致）

## 评估实际用到的文件（`eval_as_used/` 与 `eval_modifications.diff`）
评估是在主仓库 `ViPT` 的工作区里跑的，工作区相对提交有两处小改动和几个未纳入 git 的文件，这里一并给出，保证评估可复现：
- `RGBE_workspace/test_rgbe_mgpus.py`：把作者机器上的数据路径改成可通过 `--seq_home` 指定（不改推理）。
- `lib/test/tracker/data_utils.py`：增加环境变量 `CVR_ZERO_AUX=1` 时把辅助通道置零，用于“测试时辅助通道置零”的缺失模态评估（默认关闭，不影响正常评估）。
- `Depthtrack_workspace/run_offline_depthtrack.py`（及 smoke 版）：深度的离线评估入口（输出交给 vot-toolkit 分析）。
- `experiments/vipt/*.yaml`：每个评估用的 yaml（评估时从 `batch1_modality.yaml` 复制并改名）。
- `configs/fed/client_splits/*.json`：全部划分文件（含 E2 实验用的热红外降采样版本）。
上面两处改动的 diff 在 `eval_modifications.diff`。
