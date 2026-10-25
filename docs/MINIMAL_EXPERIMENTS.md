# 最小对照实验（预注册，2026-10-25）

这份文件定义"下一步该跑什么"，以及每条结论最多能被解读成什么。它取代此前方案里的
"预测器 + 路由器"主线：那套设计还没有证据支撑（见 `RAPD_调整方案_复核与更正_2026-10-25.md`）。

**纪律：先证明简单机制有价值，再谈新机制。**

---

## 1. 为什么要先跑这个，而不是继续改方法

现有证据只支持一句话：**在这套实现下，RAPD 与 CE / 均匀 KD 的差在噪声内**。它没有回答：

- 增益（如果有）来自**教师**，还是只来自**类别重加权**？
- 均匀 KD 的基线到底带了什么？（历史上它同时带 `LAMBDA_FEAT: 0.1`）
- 跨臂比较是否被**非预期配置差异**污染？（历史上 `TRAIN.DETERMINISTIC` 在各臂不一致）

这三条不回答，任何"改进方案"都只是在换解释。

---

## 2. 五臂矩阵

| arm | 配置 | CE | KD | 改变了什么 | 能回答 | 不能回答 |
|---|---|---|---|---|---|---|
| `ce` | `configs/min/ce.yaml` | 普通 | 无 | 基线 | 纯 RGB 基线 | — |
| `cbce` | `configs/min/cbce.yaml` | **类别加权** | 无 | 只有 CE 的加权 | 重加权本身值多少 | 与教师无关 |
| `uniform_kd` | `configs/min/uniform_kd.yaml` | 普通 | 均匀 KD（`LAMBDA_FEAT: 0.0`） | 加入软标签 | 普通蒸馏有没有机会 | 多模态的作用 |
| `cbce_kd` | `configs/min/cbce_kd.yaml` | **类别加权** | 类别加权门控 | 在**同样的 CE 处理**上加 KD | `cbce_kd − cbce` = KD 的增量 | 模态归因 |
| `correctness_kd` | `configs/min/correctness_kd.yaml` | 普通 | 教师正确性门控（仅前台） | 拒绝错误的教师目标 | "拒绝坏目标"值多少 | 模态归因 |

关键设计：`cbce` 与 `cbce_kd` 的 **CE 处理完全相同**，所以二者之差只在 KD 上。这是
"特权蒸馏是否值得"这个问题的最小可判形式。`uniform_kd` 与 `cbce_kd` 相比则同时变了
CE 加权与 KD 门控，只能作为整体效果参考，不能当作门控的消融。

**本阶段不包含**：DeLiVER、RGB 教师对照、特征级监督、任何学习式路由。它们排在候选臂
出现稳定增益之后（阶段三/四）。

---

## 3. 不变量（由脚本强制，不靠人记）

`scripts/check_config_parity.py` 逐键比较五个配置，只允许以下差异存在：

| 允许不同 | 理由 |
|---|---|
| `DATASET.MODALS` | KD 臂要读教师的热成像流；学生始终只吃 RGB |
| `KD.ENABLE / MODE / GATE / FOREGROUND_ONLY` | 被操纵的因素 |
| `LOSS.CLASS_BALANCE.ENABLE` | 被操纵的因素 |
| `EVAL.MODEL_PATH`、`SAVE_DIR` | 记账用；`eval_protocol.py` 用 `--ckpt` 显式指定 |

其余每一个键（含 `TRAIN.DETERMINISTIC`、`TRAIN.BATCH_SIZE`、`OPTIMIZER.*`、
`SCHEDULER.*`、`MODEL.PRETRAINED`、`LOSS.NAME`、`EVAL.IMAGE_SIZE`、`EVAL.MSF.*`）
必须逐字节一致；**缺省键算作差异，不当作默认值**。

### 3.1 用同一脚本检查历史配置时发现的真实不一致

对旧的 P1 五臂运行同一检查，退出码 1，其中两项会直接改变训练数值：

| 键 | 旧 `rgb_mfnet` | 旧 `kd_mfnet_uniform` | 旧 `kd_mfnet_classbalance` |
|---|---|---|---|
| `TRAIN.DETERMINISTIC` | 缺省（False） | 缺省（False） | `True` |
| `KD.LAMBDA_FEAT` | 缺省 | `0.1` | `0.0` |

含义（只陈述事实，不推断因果）：

1. 旧 P1 的 CE 基线与三个门控臂**不在同一确定性设置下**训练，而确定性会改变 cuDNN 选核，
   进而改变数值轨迹；
2. 旧 `uniform` 臂不是"纯 KD"，它同时带 0.1 的特征模仿项，因此"uniform vs 门控臂"的差
   里混着特征项的存在/消失。

这两点足以说明旧的跨臂差值不能直接归因，但**不能**说明它们就是 1.5 mIoU 跨批差异的原因。

另外 `EVAL.MSF.*` 也不一致（有的臂开多尺度/翻转）。主表用 `eval_protocol.py` 时不受影响，
但只要有人用训练器自带评估，就会得到不同口径。新矩阵把它统一为 `FLIP: false, SCALES: [1.0]`。

---

## 4. 验收：smoke 必须全绿

```bash
python3 scripts/check_config_parity.py configs/min/*.yaml
python3 scripts/smoke_pipeline.py --workers 2
```

`smoke_pipeline.py` 对每个臂跑 1 epoch × 2 iter（含一次完整验证），并断言：

| 检查 | 期望 |
|---|---|
| 进程退出码 | 0 |
| `metrics.json` | 恰好新写出 1 个 |
| `final_stats.loss` / `kd` | 有限且非负 |
| KD 臂 `final_stats.kd` | > 0（证明 KD 项真的接上了） |
| `ce` / `cbce` 的 `kd` | 恰好 0（证明没有偷偷蒸馏） |
| `uniform_kd` 的 `r` | = 1（证明均匀 KD 没有走探针分支） |
| `task_aware` 臂的 `sel` | 0 < sel ≤ 1（证明门控真的选到了像素） |
| `last.pth` | 存在 |

它**不**测精度，也不产生任何可写进论文的数字。1 epoch / 784 张图的排名没有意义，
把它当结果就是上一批问题的来源。

### 4.1 这个 gate 抓到的第一个真问题

第一次运行时 5 个臂里有 2 个失败：

```
[PASS] cbce      loss=13.3359 kd=0.0000 r=0.000 sel=0.000
[FAIL] ce        - expected exactly 1 new metrics.json, got 0
[FAIL] correctness_kd - expected exactly 1 new metrics.json, got 0
```

原因不是模型，而是命名：`ce` 与 `cbce` 都没有 KD，`stage` 都算成 `rgb`，于是两者的
`save_dir` 完全相同，后跑的臂直接覆盖前一个。`correctness_kd` 与 `cbce_kd` 同理
（`stage` 都是 `task_aware`）。修法是给每个臂独立的 `SAVE_DIR`（`output/min/<arm>`），
并让 smoke 用 `--tag smoke_<arm>` 保证 `results/` 不互相覆盖。

这正是"先跑 gate"的意义：同样的碰撞如果发生在阶段二，会安静地毁掉一半数据，
而人只会在两天后发现某个目录里只有一个 metrics.json。

---

### 4.2 第二轮事故：CE 基线被训成双流模型（2026-10-25）

第一次完整跑 wave1 时，`ce` 与 `cbce` 的训练日志显示 epoch 120 mIoU 56.19 / 56.77，
而固定轮次协议评测同一 checkpoint 只得 10.01 / 11.09。原因是两个 bug 叠加：

1. `configs/min/ce.yaml` 保留了 `KD.TEACHER_MODALS: [img, thermal]`，而
   `train_kd.py` 在 KD 关闭时用 `student_modals = teacher_modals`，于是"纯 RGB 基线"
   实际是 **896 张量**的双流模型（单流 348）。
2. `load_state()` 只报 missing、不报 unexpected，并以 `strict=False` 装载。896 张量装进
   348 张量模型时丢掉 548 个张量，却打印 `loaded 348/348 (missing 0)`，评测跑在一个
   近乎未训练的网络上。

修法：KD 关闭时 `student_modals` 取 `DATASET.MODALS`；`load_state` 统计 unexpected 并在
strict 下报错；smoke 增加 `student_modals == ['img']` 断言；新增 `tests/test_load_state.py`。

教训（写进流程，不写进论文）：**"训练日志好看"不是管线正确的证据**。模态/架构一致性
必须由断言检查，因为它的失败方式是静默的——数字照常产出，只是毫无意义。

## 5. 两阶段预算

### 阶段二：机制筛选（便宜，只用来排序与排错）

- 5 臂 × seed 3407 × **120 epoch**（约 45 min/臂，2 卡并行 ≈ 2.5 h 墙钟）
- 目的：确认管线、看是否有臂明显崩坏（如 `sel` 长期接近 0）、给出下一阶段候选
- 报告方式：`eval_protocol.py` 的固定轮次指标，**不声明显著性**

### 阶段三：确认（贵，只跑有资格的两个臂）

- `ce` + `cbce` + 筛选出的最佳 KD 臂 × **3 seed（3407/3408/3409）× 500 epoch**
- 约 27 GPU·h，2 卡 ≈ 14 h
- 报告：每个 seed 的差值、均值与离散度、图像重采样区间、关键次指标（夜间 person IoU）

`cbce` 必须一路带着走：它是"这到底是蒸馏还是重加权"的唯一廉价判据。

---

## 6. 预注册的停止规则

先写下 delta（最小有意义效应），再看结果。建议以 foreground mIoU 为主指标：

| 观察 | 结论 | 动作 |
|---|---|---|
| `cbce_kd − cbce` 的区间整体高于 delta，且夜间 person 不退化 | KD 在同等 CE 处理下有增量 | 继续做模态归因（RGB 教师、热成像错配） |
| 区间整体落在 ±delta 内且功效足够 | 该配置下 KD 没有可测增量 | 停止这条路线，转"负结果 + 诊断" |
| 区间跨 0 且较宽 | 证据不足 | 按预算决定补 seed 或停止，**不**宣称不可能 |
| `cbce − ce` 已超过 delta，`cbce_kd − cbce` ≈ 0 | 收益来自重加权 | 如实按"重加权"报告，不包装成特权蒸馏 |
| 任一臂 `sel` 长期 ≈ 0 | 门控实现有问题 | 先修实现，不看指标 |

---

## 7. 命令清单

```bash
# 0) 环境与工件
bash scripts/restore_artifacts.sh          # 下载、解包、校验（bash 3.2 兼容）
python3 scripts/check_config_parity.py configs/min/*.yaml
python3 scripts/smoke_pipeline.py --workers 2

# 1) 阶段二筛选（2 卡，5 臂，120 epoch）
#    每个臂必须带自己的 --tag：ce 与 cbce 的 stage 都是 rgb，cbce_kd 与 correctness_kd
#    都是 task_aware，不加后缀会互相覆盖 results/ 里的同名目录。
CUDA_VISIBLE_DEVICES=0 python3 tools/train_kd.py --cfg configs/min/ce.yaml             --seed 3407 --epochs 120 --tag screen_ce &
CUDA_VISIBLE_DEVICES=1 python3 tools/train_kd.py --cfg configs/min/cbce.yaml           --seed 3407 --epochs 120 --tag screen_cbce &
wait
CUDA_VISIBLE_DEVICES=0 python3 tools/train_kd.py --cfg configs/min/uniform_kd.yaml     --seed 3407 --epochs 120 --tag screen_uniform_kd &
CUDA_VISIBLE_DEVICES=1 python3 tools/train_kd.py --cfg configs/min/cbce_kd.yaml        --seed 3407 --epochs 120 --tag screen_cbce_kd &
wait
CUDA_VISIBLE_DEVICES=0 python3 tools/train_kd.py --cfg configs/min/correctness_kd.yaml --seed 3407 --epochs 120 --tag screen_correctness_kd

# 2) 统一口径评测（固定轮次，不做选择）
python3 scripts/eval_protocol.py --cfg configs/min/ce.yaml \
  --ckpt output/min/ce/MFNet_CMNeXt-B2_i_rgb_s3407_screen_ce/last.pth \
  --teacher-ckpt output/MFNet_CMNeXt-B2_it_teacher_s3407/best.pth \
  --out results/protocol/MIN_ce_3407_e120.json
```

---

## 8. 本阶段明确不做

- 不实现"教师对、学生错"预测器，不实现监督通道路由；
- 不用训练期热特征旁路去论证 RGB 可达/不可达；
- 不把 1 seed 的筛选数字写进论文；
- 不在没有 `cbce` 的情况下报告任何"KD 提升"。
