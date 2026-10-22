# RAPD 项目进展报告

**课题**：面向 RGB-only 部署的可恢复性感知特权模态蒸馏（Recoverability-Aware Privileged Distillation, RAPD）
**基线**：CMNeXt / DELIVER（Apache-2.0）
**仓库**：https://github.com/L218Y/CVPR2027_KD
**目标会议**：CVPR 2027（截稿约 2026-11-17）
**硬件**：单卡 RTX 4090 24 GB（原计划 4 卡，已按单卡重排）
**更新时间**：2026-10-22

---

## 1. 问题设定

训练阶段可以使用 RGB + 深度/热成像/事件/LiDAR 等多模态数据，但部署阶段只有 RGB 相机。已有的多模态分割方法（CMX、CMNeXt、DELIVER）在部署时必须保留全部传感器；已有的多模态蒸馏方法（如 CMX 的 KD 变体）把教师输出**均匀地**压给学生，隐含假设"教师看到的证据学生都能看到"。

这个假设在真实场景中不成立：夜间热成像、深度边界、事件流的高时间分辨率都是 RGB 无法恢复的特权证据。强行拟合这些区域会让学生在推理时产生**高置信度的幻觉**（把热成像才能分辨的行人预测成路面），这是"精度看起来还行、安全性却更差"的根源。

**核心问题**：如何在不引入推理开销的前提下，把教师的知识按"学生能否从 RGB 恢复"逐像素地区分开——可恢复的部分照常蒸馏，不可恢复的部分降级为保守先验而不是硬拟合？

---

## 2. 方法

### 2.1 可恢复性探针

学生最后阶段特征 $f_s$、教师融合特征 $f_t$ 经轻量探针 $g$（1×1 conv → BN → ReLU → 1×1 conv，隐层 128）映射后取余弦一致度：

$$
r=\frac{\cos\big(g(f_s),\,f_t\big)+1}{2}\in[0,1]
$$

$r$ 逐像素表示"教师的融合证据在多大程度上可由 RGB 特征线性解释"。$r$ 在蒸馏中被 **detach**，学生无法通过压低自身特征来提高监督权重。探针由独立 AdamW（lr $10^{-3}$）优化，损失为归一化特征上的 MSE，且学生特征在探针分支中被 detach，避免探针与学生相互共适应。

### 2.2 保守目标与加权蒸馏

教师后验 $p_t$ 与类别先验 $\pi$（训练集上滑动 EMA）混合，得到"学生可达"的目标 $p_u$：

$$
p_u=(1-\epsilon)\,p_t+\epsilon\,\pi,\qquad \epsilon=\epsilon_{\max}(1-r)
$$

$$
\mathcal{L}_{kd}=r\cdot \mathrm{KL}(p_t\Vert p_s)+(1-r)\cdot \mathrm{KL}(p_u\Vert p_s)
$$

以 $\tau^2$ 缩放，忽略 `ignore_index=255` 像素。直观含义：$r\to1$（RGB 足以解释）时退化为标准 KD；$r\to0$（特权证据）时学生只需匹配类别先验，既不放弃监督信号，也不被要求复现无法观测的证据。

可选的逐像素特征项 $\lambda_f\big(1-\cos(\mathrm{proj}(f_s),f_t)\big)$ 同样按 $r$ 加权。

### 2.3 训练调度与部署头

- **预热**：前 `WARMUP_EPOCHS=5` 轮取 $r\equiv1$（即普通 KD），让探针在"有意义的学生特征"上先收敛，避免早期随机特征给出错误权重。
- **部署头**（可选）：`RecoverabilityHead` 从 RGB 特征直接回归 $r$，使部署模型能自行标记"我在猜"的像素，用于风险告警。默认关闭，作为可选的附加能力。

---

## 3. 创新点

1. **逐像素可恢复性度量**：把"学生能否恢复教师证据"变成一个可学习的、无需人工标注的逐像素回归问题，用线性探针的余弦一致度作为代理，避免了需要特权模态才能推理的循环依赖。
2. **保守目标回退（conservative back-off）**：不可恢复区域不丢弃、不硬掩码，而是平滑回退到类别先验——保留了教师的类别排序信息，同时移除了"特权置信度"。这与 label smoothing、hard mask、confidence penalty 都不同。
3. **抗博弈的训练设计**：$r$ 与特征 detach + 独立优化器 + 预热调度，防止学生"降低自身可解释性以逃避监督"的退化解。
4. **部署期风险信号**：`RecoverabilityHead` 让最终 RGB 模型在推理时输出可恢复性图，把蒸馏阶段的内部量转化为可用的不确定性估计。
5. **评测协议**：除 mIoU/mAcc/aAcc 外，引入 PGR（特权差距恢复率）与幻觉率，直接衡量"是否安全地利用了特权信息"，而非只看均值涨点。

$$
\mathrm{PGR}=\frac{\mathrm{mIoU}_{\text{student}}-\mathrm{mIoU}_{\text{RGB}}}{\mathrm{mIoU}_{\text{teacher}}-\mathrm{mIoU}_{\text{RGB}}}
$$

---

## 4. 代码实现

| 文件 | 作用 |
|---|---|
| `semseg/kd/losses.py` | `RecoverabilityKD`（4 种模式）、`UniformKD`（Hinton 基线） |
| `semseg/kd/probe.py` | `FeatureProbe`、`RecoverabilityHead` |
| `tools/train_kd.py` | 统一训练器：teacher / 蒸馏共用同一数据管线、调度与评测 |
| `tools/analyze.py` | 逐类 IoU、ECE、幻觉率、$r$ 图可视化 |
| `scripts/prepare_mfnet.py` | MFNet → 训练格式（官方 784/392 划分） |
| `scripts/queue.sh`, `queue_kd.sh` | 串行实验队列 |
| `scripts/run_and_sync.sh` | 训练 + 结果收集 + 自动 push，崩溃也上报 |
| `scripts/collect_results.py`, `sync_results.sh`, `summarize.py` | 结果聚合、PGR 表、GitHub 同步 |
| `tests/test_kd.py` | 5 个单元测试：$r=1$ 等价性、掩码、梯度 |

对基线的唯一改动是让 `mmcv` 变为可选（`mspa.py`、`hem.py` 提供本地 `build_norm_layer` / `ConvModule` 回退），使其可在 PyTorch 2.11 + CUDA 12.8 上运行；其余保持上游一致，保证对比公平。

---

## 5. 数据集与协议

| 数据集 | 划分 | 类别 | 模态 | 分辨率 | 状态 |
|---|---|---|---|---|---|
| MFNet | 784 / 392 | 9 | RGB + T | 480×640 | 已就绪 |
| NYUv2 | 795 / 654 | 40 | RGB + D | 480×640 | 待下载 |
| DeLiVER | 3983 / 2005 / 1897 | 25 | RGB + D + E + L | 1024×1024 | 待下载（12.2 GB）|

MFNet 使用官方 `train.txt` / `val.txt`，并剔除 `_flip` / `_seed` 增强副本，保证与文献 784/392 协议一致。

---

## 6. 实验与结果

### 6.1 已完成

| 编号 | 实验 | 设置 | 耗时 | 结果 |
|---|---|---|---|---|
| T1 | 教师 | CMNeXt-B2，RGB+T，500 ep | 3 h 44 m | **mIoU 60.07 @ ep225**（末轮 59.19）|
| S0 | RGB 基线 | CMNeXt-B2，RGB only，500 ep | 1 h 51 m | **mIoU 58.95 @ ep400** |

教师验证曲线：

| epoch | 75 | 100 | 125 | 150 | 175 | 200 | 225 | 250 | 300 | 350 | 400 | 450 | 500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mIoU | 54.16 | 57.32 | 57.96 | 58.10 | 58.15 | 60.03 | **60.07** | 59.34 | 59.95 | 60.05 | 58.44 | 59.05 | 59.19 |

### 6.2 进行中 / 待跑

| 编号 | 实验 | 状态 | 预计 |
|---|---|---|---|
| S1 | uniform KD（Hinton 基线）| 待重跑（代码 bug 已修）| 4.5 h |
| S2 | RAPD 完整方法 | **运行中（ep 23/500）** | 4.5 h |
| S3 | conservative_only 消融 | 排队 | 4.5 h |
| S4 | RAPD + $\epsilon_{\max}=0.2$ | 排队 | 4.5 h |
| S5 | 多 seed（S0/S2 各 +2 seed）| 未开始 | 18 h |
| S6 | day/night 分组评测 | 未开始（脚本未写）| 0.5 h |
| T2/S7 | NYUv2 教师 + 学生 | 未开始 | 14 h |
| T3/S8 | DeLiVER 教师 + 学生 | 未开始 | 24 h |

---

## 7. 关键分析

1. **教师达到文献水平，管线可信。** CMNeXt-B2 RGB-T 在 MFNet 的公开成绩约 59–60，我们 60.07，说明划分、预处理、评测均正确。
2. **教师末段震荡 ±1.0 mIoU。** ep225 后在 58.4–60.1 波动，源自 784 张训练图 + poly 学习率衰减到 0 + 无 EMA。这意味着**任何小于 1 个点的差异都不可信**。
3. **MFNet 的特权 gap 仅 1.12 mIoU**，而 DeLiVER 为 9.10（57.20 → 66.30）、NYUv2 约 5.4。MFNet 类别少、`unlabeled` 一类占 ~97 IoU 抬高均值，且热成像增益集中在夜间。**在 gap=1.1 的数据上 PGR 的分母近乎噪声**，即使方法有效也难以测出。
4. **因此论文主线应放在 DeLiVER**（gap 大、含 4 种天气 + 5 种传感器失效场景，正是"不可恢复特权证据"最极端的情形），MFNet 作为"低 gap 场景也不掉点"的补充证据，并用 day/night 分组证明增益集中在夜间。

---

## 8. 已修复的工程问题

| 问题 | 现象 | 修复 |
|---|---|---|
| 教师权重键名不匹配 | 加载报 `proj.module.weight` vs `proj.weight` | checkpoint 与模型键双向归一化，现 896/896 全部加载 |
| probe 损失未进 GradScaler | `No inf checks were recorded for this optimizer` | 折入主 loss 并对学生特征 detach，单次 scaled backward |
| `UniformKD` 缺 `r_mean` | KeyError 中断训练 | 返回 `r_mean=1`，日志分支统一 |
| 队列同步被拒 | `! [rejected] ... (fetch first)` | push 前自动 rebase + 3 次重试 |
| 日志文件进入版本库 | 每次 pull 冲突 | `logs/` 加入 `.gitignore` 并取消跟踪 |

---

## 9. 当前进展与时间线

- **已完成**：方法实现、单元测试、数据管线、教师训练、RGB 基线、队列与自动同步基础设施。
- **进行中**：S2（RAPD）训练，约 26 s/epoch，预计 4–4.5 h 完成；随后自动执行 S3、S4。
- **待补**：S1 重跑、多 seed、day/night 评测脚本。
- **整体**：核心 MFNet 实验约 24 h 内可全部出结果；若追加 DeLiVER 主线，需再加 1–2 天。

---

## 10. 风险

1. **统计效力不足**：单 seed、gap 小，1 个点以内的提升无法支撑 CVPR 级别的结论。必须补 3 seed 并报告均值±标准差。
2. **数据集选择**：若坚持只用 MFNet，审稿人很可能质疑"特权信息量不足"，因此 DeLiVER 几乎必需。
3. **算力**：单卡 4090 使 DeLiVER 多 seed 全矩阵不可行，需在"数据集广度"与"seed 数量"之间取舍。
