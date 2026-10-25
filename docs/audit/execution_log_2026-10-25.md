# 下一步执行记录（2026-10-25）

对应上一条回答里承诺的三步：检查已有产物 → 修实验管线 → 准备最小对照。本文件记录**实际做了什么、发现了什么、验证到什么程度**。所有结论都来自本轮在 lab 上真实执行过的命令；未完成的会明确写出。

---

## 1. 产物检查（哪些还在、哪些确实不在）

| 项 | 状态 | 依据 |
|---|---|---|
| 代码 | ✅ `main @ 1948b90`，4 个提交 | `git log` |
| 数据（2353 张 + 784/392 划分） | ✅ release 可下载，sha256 与 `results/ARTIFACTS.sha256` 一致 | 本轮重新下载并校验 |
| batch A 的 14 个 checkpoint | ✅ release 里有；抽验 4 个 sha256 全对 | 见审核报告 §4.2 |
| 教师 `it_teacher_s3407/best.pth` | ✅ 可直接用于训练/蒸馏 | 本轮 smoke 实际加载成功（`strict=True`） |
| P1 四个臂的 checkpoint | ❌ 不可得（git、release、当前 lab 容器均无） | 审核报告 §1.3 |
| P1 的逐图 JSON（`results/protocol/P1_*.json`） | ❌ 不可得 | 同上 |
| `results/splits/{train,test}.txt` | ✅ 已入库，972 行 = 784 + 392 | 本轮核对 |

结论：**不需要重下数据，也不需要重训教师**；缺的只有 P1 那 4 个臂本身。

---

## 2. 代码改动（7 个文件，全部是可靠性，不含新方法）

| 文件 | 改动 | 为什么 |
|---|---|---|
| `data`（仓库根） | `git rm --cached`，改为不跟踪 | 它是一条绝对路径符号链接；在文档指定的还原路径 clone 会得到自指链接，`mkdir -p data` 直接失败 |
| `.gitignore` | 追加 `/data` | 防止再被加回来 |
| `scripts/restore_artifacts.sh` | 重写：替换自指链接、**去掉 `mapfile`**（macOS 自带 bash 3.2 没有）、自动解包数据集、补拷冻结划分 | 原脚本在 macOS 上直接语法级失败；原文档还要再手工 `tar` 一步 |
| `MANIFEST.md` | Restore 段改为一条命令，并注明 `data/` 不再跟踪 | 文档与实际不一致 |
| `configs/min/{ce,cbce,uniform_kd,cbce_kd,correctness_kd}.yaml` | 新增 5 个臂，每臂独立 `SAVE_DIR` | 让"只有被操纵的因素不同"可被机器检查 |
| `scripts/check_config_parity.py` | 新增：逐键比对多个配置，只允许白名单差异，**缺省键算差异**，退出码非 0 即失败 | 防止再次出现"跨臂比较其实换了别的东西" |
| `scripts/smoke_pipeline.py` | 新增：每臂 1 epoch × 2 iter 端到端跑通并断言 KD/门控/存档 | 长训练之前必须过的闸门 |
| `docs/MINIMAL_EXPERIMENTS.md` | 新增：五臂矩阵、不变量、验收、两阶段预算、预注册停止规则 | 把"下一步跑什么"写成可执行、可复核的文件 |

训练器本身**没有改**：它已经具备缺失教师显式报错（`tools/train_kd.py:153-155`）、`--max-iters/--skip-eval`、以及 `metrics.json` 里的配置 sha、代码 commit、划分 sha、教师 sha、类别权重。

---

## 3. 两个真实发现（不是推测）

### 3.1 旧 P1 五臂的配置并不对等（`check_config_parity.py` 退出码 1）

对 `rgb_mfnet / rgb_mfnet_cbce / kd_mfnet_uniform / kd_mfnet_classbalance / kd_mfnet_correctness_fg` 运行检查，报 19 处非预期差异，其中两项会直接改变训练数值：

| 键 | 旧 CE 基线 | 旧 uniform | 旧 classbalance |
|---|---|---|---|
| `TRAIN.DETERMINISTIC` | 缺省（False） | 缺省（False） | `True` |
| `KD.LAMBDA_FEAT` | 缺省 | `0.1` | `0.0` |

含义：旧 P1 的 CE 基线与门控臂**不在同一确定性设置**下训练；旧 `uniform` 臂其实是"KD + 0.1 特征模仿"，不是纯 KD。所以 "uniform 60.01 vs CE 58.29" 这类差值里混着实现差异。**这不能证明它就是那 1.5 mIoU 跨批差异的原因**，但足以说明旧的跨臂数字不能直接归因。

同一脚本对新矩阵返回 0，只有 8 处公认差异（`SAVE_DIR`、`EVAL.MODEL_PATH`、`DATASET.MODALS`、`KD.*`、`CLASS_BALANCE.ENABLE`）。

### 3.2 命名冲突会静默毁数据（smoke 第一次运行抓到的）

第一次 smoke，5 个臂有 2 个失败：

```
[PASS] cbce
[FAIL] ce              - expected exactly 1 new metrics.json, got 0
[FAIL] correctness_kd  - expected exactly 1 new metrics.json, got 0
```

原因：`ce` 与 `cbce` 都没有 KD，`stage` 都算成 `rgb`，实验名完全相同 → 后者覆盖前者。`cbce_kd` 与 `correctness_kd` 同理（都是 `task_aware`）。修法是每臂独立 `SAVE_DIR` + smoke 用 `--tag smoke_<arm>`。

**这就是先跑闸门的价值**：同样的碰撞如果发生在阶段二，会安静地毁掉一半数据。

---

## 4. 验证结果（本轮实际输出）

### 4.1 配置对等检查

```
$ python3 scripts/check_config_parity.py configs/min/*.yaml
parity ok: 5 configs, 8 intended difference(s)          # exit 0

$ python3 scripts/check_config_parity.py <旧 P1 五臂>
PARITY FAILED: 19 unintended difference(s)               # exit 1
```

### 4.2 端到端 smoke（1 epoch × 2 iter，含一次完整 392 张验证）

```
[PASS] ce              loss=13.8136 kd= 0.0000 r=0.000 sel=0.000
[PASS] cbce            loss=13.3359 kd= 0.0000 r=0.000 sel=0.000
[PASS] uniform_kd      loss=47.5654 kd=32.1691 r=1.000 sel=0.000
[PASS] cbce_kd         loss=40.9408 kd=27.7586 r=1.000 sel=0.554
[PASS] correctness_kd  loss=37.7113 kd=22.3151 r=1.000 sel=0.942
SMOKE PASSED for all 5 arms
```

读法（这些是管线证据，不是效果）：

- 两个无 KD 的臂 `kd` **恰好为 0**，证明没有偷偷蒸馏；
- 三个 KD 臂 `kd > 0`，证明损失真的接上了；
- `sel` 分别是 0.554（类别加权）与 0.942（教师正确性、仅前台）。0.942 意味着**前台 94.2% 的像素上教师 argmax 等于 GT**——与审核报告"教师远强于学生"一致；
- 训练器打印的类别权重 `{unlabeled 0.3478, car 1.5667, person 2.8372, bike 5.0, …}` 与我此前**独立**从 784 张标签算出的 `{0.348, 1.567, 2.837, 5.0…}` 到小数点后 3 位一致，交叉验证了审核报告 §4.9；
- 每次运行都写出 `metrics.json`（含 `n_eval=392`）与 `last.pth`。

### 4.3 既有单元测试（未改动，全部通过）

```
tests/test_kd.py          → 全部通过
tests/test_task_aware.py  → 全部通过
tests/test_pairing.py     → 全部通过
```

### 4.4 还原脚本（全新 clone 上的端到端验证）

旧脚本在全新 clone 上跑，复现了文档里写过的失败：

```
data -> /data/lab/CVPR2027_KD/data      # 自指链接
mkdir: cannot create directory 'data': File exists
```

改成新脚本后，第一轮又暴露了第二个真实缺陷：`results/ARTIFACTS.sha256` 里数据集那一行是
`../MFNet_dataset.tar.gz`（当初在 `data/` 目录里算的），从仓库根目录跑 `sha256sum -c`
必然 `No such file or directory`，于是 `set -e` 让脚本在解包前就退出。

修法是在校验前做一次路径归一（`../X` → `data/X`），不动已发布的校验清单。修正后在全新 clone 上：

```
have data/MFNet_dataset.tar.gz            # 已下载的资产全部命中，不重复下载
verifying checksums                       # 0 处 FAILED
unpacking prepared dataset
artifacts restored: 6 runs, 2353 rgb images

$ wc -l data/MFNet/train.txt data/MFNet/test.txt
 784 data/MFNet/train.txt
 392 data/MFNet/test.txt
```

即 **clone → `bash scripts/restore_artifacts.sh` → 784/392 划分就绪** 这一条链现在是通的，
而且 `data` 已经是普通目录，不再是自指链接。

---

## 5. 明确没做的事

- **没有启动任何长训练**（阶段二的 5 臂 120 epoch 还没跑）；
- **没有推送仓库**，改动只在本机工作副本里，未 commit（要提交请明确说）；
- 没有实现"教师对、学生错"预测器、监督通道路由、热特征旁路——按复核报告，它们排在候选臂出现稳定增益之后。

---

## 6. 下一步（按顺序，可直接执行）

```bash
# 1) 闸门（每次改配置后都要重跑，约 5 分钟）
python3 scripts/check_config_parity.py configs/min/*.yaml
python3 scripts/smoke_pipeline.py --workers 2

# 2) 阶段二筛选：5 臂 × 120 epoch × 1 seed（2 卡，约 2.5 小时）
#    详见 docs/MINIMAL_EXPERIMENTS.md §7，注意每臂带自己的 --tag
```

阶段二的判据已经预注册（`docs/MINIMAL_EXPERIMENTS.md` §6），核心是 `cbce_kd − cbce`：
**只有这一项有稳定增量，"特权蒸馏"才第一次有证据；如果 `cbce − ce` 已经吃掉全部收益，
那就如实按"重加权"报告，不再叫 RAPD。**

---

## 7. 开跑状态（2026-10-25 18:50 +0800 起）

| 项 | 状态 |
|---|---|
| 预训练权重 | ✅ 从 `InSAI-Lab/CMNeXt` 取到 `mit_b2.pth`，实测 `missing_keys=[]`、`unexpected=['head.weight','head.bias']`，`patch_embed1` 与文件张量差 0.0 |
| 训练 | 5 臂 × 120 epoch × seed 3407，两卡并行（wave1: ce+cbce；wave2: uniform_kd+cbce_kd；wave3: correctness_kd） |
| GPU | 0 与 1 均 ~95% 利用率、17.5 GB 显存 |
| 速度 | ≈1 min/epoch（96 核，非数据加载瓶颈），单臂约 2 h，全部约 6 h |
| 已完成并推送 | commit `80e4dab`：管线修复 + 5 臂配置 + 预注册文档（已用 PAT 推到 `main`） |
| 自动收尾 | `scripts/finish_screen.py` 在后台等待 5 臂，完成后逐臂 `eval_protocol`，写 `results/RESULTS_MIN_SCREEN.md`，再提交并推送 |

未决风险：容器可能在跑完前被回收（此前发生过一次）。缓解措施是代码已推送、结果一产出就自动提交；
若容器在中途消失，最多损失一次训练，不会丢失方法或配置。


---

## 8. 事故记录：wave1 的 CE/CB 臂其实不是 RGB 模型（2026-10-25 19:40 +0800）

**现象**：wave1（ce、cbce）训练日志显示 epoch 120 mIoU 56.19 / 56.77，但用固定轮次协议评测同一 checkpoint 得到 **10.01 / 11.09**。差 46 个点，不是噪声。

**根因（两个 bug 叠加）**：

1. `configs/min/ce.yaml` 与 `cbce.yaml` 保留了 `KD.TEACHER_MODALS: [img, thermal]`。`tools/train_kd.py` 里
   `student_modals = ["img"] if use_kd else list(teacher_modals)`——KD 关闭时它去读了**教师的模态列表**，
   于是"纯 RGB 基线"实际训练成了 **RGB+热成像 双流模型**。证据：该 checkpoint 有 **896** 个张量，
   单流模型只有 **348** 个。
2. `load_state()` 只检查 `missing`，不检查 checkpoint 里**多出来的**张量，并用 `strict=False` 装载。
   于是把 896 张量的双流权重装进 348 张量的单流模型时，它丢掉了 548 个张量，却打印
   `loaded 348/348 tensors (missing 0)`——**评测无声地跑在一个几乎随机初始化的模型上**。

**修复**：

| 文件 | 改动 |
|---|---|
| `tools/train_kd.py` | KD 关闭时 `student_modals` 改为取 `DATASET.MODALS`（不再借教师的模态表） |
| `tools/train_kd.py` | `load_state()` 计算 `unexpected`，`strict` 时报错并说明"通常意味着模态不一致" |
| `scripts/smoke_pipeline.py` | 新增断言：`metrics.student_modals == ["img"]`，否则该臂判 FAIL |
| `scripts/run_arms.sh` | 新增通用两卡跑臂脚本（按名字跑，自动两两并行） |

**验证**：修复后重跑 ce 的 1 轮检查，`student_modals: ['img']`，单流模型张量数 348（双流为 896）。

**处置**：已删除无效的 wave1 输出与两个 `MIN_*_e120.json`；wave2（uniform_kd、cbce_kd，学生本来就是单流）继续；
wave2 结束后自动重跑 ce、cbce 与 correctness_kd，再由 finisher 统一评测并推送。

**必须撤回的结论**：此前汇报的 epoch 75 / 100 / 120 的 ce 54.24 / 56.26 / 56.19 与 cbce 53.16 / 55.95 / 56.77，
都是**双流模型**的数字，不能当作 RGB 基线，也不能用来比较 CB 与 CE。它们只说明"训练过程是健康的"，
不说明任何方法结论。

---

## 9. 第一批有效结果（120 epoch / seed 3407 / 固定轮次）

修复模态 bug 后重跑，`ce` 与 `cbce` 于 12:51 UTC 完成，评测结果（单流模型已验证）：

| arm | official | foreground | 夜间 person | guardrail |
|---|---|---|---|---|
| ce | 55.32 | 50.09 | 45.43 | 29.27 |
| cbce | 56.30 | 51.28 | 42.90 | 33.85 |
| cbce_kd | 56.73 | 51.71 | 45.38 | 25.73 |

两个预注册对照：

- `cbce − ce` = **+1.19 fg**：类别重加权本身的收益。
- `cbce_kd − cbce` = **+0.43 fg，95% CI [−0.56, +1.58]**：教师在同等 CE 处理上的增量，**跨 0**。

结论（按预注册规则）：**在当前预算下，教师没有可测的整体贡献；存在的增益来自重加权。**
但增量是类内而非均匀：夜间 person +2.48、guardrail −8.12——教师强项类被拉回，代价由另一稀有类承担。

限制：单 seed、120 epoch（非 500 轮正式预算）。属"证据不足"而非"已证明为零"。
已提交推送：`48856b9`（`results/RESULTS_MIN_SCREEN.md` 标注为 3/5 臂的部分结果）。

---

## 10. 五臂屏筛完成（120 epoch / seed 3407 / 固定轮次）

`correctness_kd` 于 13:32 UTC 完成，finisher 自动评测全部五臂、写表、提交并推送（`8ffa81f`、`6066344`）。

| arm | official | foreground | 夜间 person | bump | car_stop | person | guardrail | Δfg vs ce |
|---|---|---|---|---|---|---|---|---|
| ce | 55.32 | 50.09 | 45.43 | 41.92 | 33.00 | 55.73 | 29.27 | — |
| cbce | 56.30 | 51.28 | 42.90 | 55.48 | 40.45 | 50.98 | 33.85 | +1.20 |
| uniform_kd | 55.55 | 50.34 | 45.66 | 43.31 | 34.24 | 56.18 | 29.67 | +0.25 |
| cbce_kd | 56.73 | 51.71 | 45.38 | 55.42 | 39.03 | 54.54 | 25.73 | +1.62 |
| correctness_kd | 57.32 | 52.41 | 41.08 | 57.29 | 39.61 | 51.29 | 31.40 | +2.32 |

**两个预注册对照**：

- `cbce − ce` = **+1.19 fg**（重加权本身）
- `cbce_kd − cbce` = **+0.43 fg 观测值 / +0.50 bootstrap 均值，95% CI [−0.56, +1.58]** → 跨 0

**读法（这是本次屏筛最有价值的部分）**：

1. 所有正向增益都集中在 `bump`（41.92→55.5~57.3）与 `car_stop`（33.0→39.6~40.5），代价由 `car / person / bike` 承担——**这是重加权，不是蒸馏**。
2. `uniform_kd` 与 `ce` 几乎重合（+0.25），再次确认**纯 logit KD 在这个设定下无效**。
3. 教师唯一稳定的贡献是把它强的类拉回来：`cbce`→`cbce_kd` 时 person 50.98→54.54（+3.6），但 guardrail 33.85→25.73（−8.1），净效应 ≈0。
4. `correctness_kd` 是 overall 最好的臂（57.32），但它靠的是 bump 57.29 + car_stop 39.61，而 person 只有 51.29（低于 ce 的 55.73）——**它不是"夜间行人"方案**。

**结论**：在 120 epoch、单 seed、单次评测下，**没有证据表明特权教师提供了超出类别重加权的整体增量**；区间 [−0.56, +1.58] 宽于任何可辩护的效应量，且种子间波动（此前实测 1.5–2.0）大于表中除重加权外的所有差异。若要把它变成可投稿结论，需要 `ce / cbce / cbce_kd` × 3 seed × 500 epoch。
