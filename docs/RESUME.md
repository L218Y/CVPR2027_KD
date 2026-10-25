# 恢复指南（新 Lab Compute 上继续这项工作）

最后更新：2026-10-25 16:00 UTC。这一份是新机器上从零恢复所需的全部步骤，不依赖对话历史。

## 0. 当前状态一句话

方法线已判定无效并停止：RAPD（r 加权回退）、logit 蒸馏（均匀 / 正确性 / 类别加权）、
训练期特权特征注入，三条路线全为负。**尚无可投稿的正结果**；唯一站得住的是诊断性证据
（见 `docs/audit/audit_report_2026-10-25.md`、`results/RESULTS_MIN_SCREEN.md`）。
三 seed × 500 epoch 的确认实验在 epoch 45/500 时因关闭 Lab 而中断，**没有写出 checkpoint，必须重跑**。

## 1. 恢复代码

```bash
git clone https://github.com/L218Y/CVPR2027_KD.git && cd CVPR2027_KD
bash scripts/restore_artifacts.sh     # 下载 release、解包数据、校验 sha256（bash 3.2 兼容）
```

`restore_artifacts.sh` 会：替换历史上那个自指的 `data` 符号链接、下载 17 个 release 资产、
解包 `data/MFNet/`（784 train / 392 val）、拷贝 `results/splits/`、逐个校验 sha256。
**不要**再手工 `tar -C data`，也不要重新提交 `data`。

## 2. 恢复运行环境

```bash
pip install timm pyyaml opencv-python-headless fvcore tabulate tensorboard einops segmentation-models-pytorch
mkdir -p checkpoints/pretrained/segformer
curl -L -o checkpoints/pretrained/segformer/mit_b2.pth \
  "https://huggingface.co/InSAI-Lab/CMNeXt/resolve/main/pretrained/segformers/mit_b2.pth"
```

`mit_b2.pth` 必须存在，否则所有臂都会以随机初始化训练（smoke 不报错，但结果无意义）。
正确加载的标志：`missing_keys=[]`，只有 `head.weight/head.bias` 未使用，
且 `patch_embed1.proj.weight` 与文件张量逐元素差 0.0。

教师权重来自 release：`output/MFNet_CMNeXt-B2_it_teacher_s3407/best.pth`（由 restore 脚本放好）。

## 3. 每次改配置后先过闸门

```bash
python3 scripts/check_config_parity.py configs/min/*.yaml   # 只允许白名单差异，缺省键算差异
python3 scripts/smoke_pipeline.py --workers 2               # 每臂 1 epoch × 2 iter，断言 KD/门控/存档
```

smoke 会断言 `student_modals == ['img']`。**这条不能删**：一次事故里 CE 配置残留
`KD.TEACHER_MODALS`，把"纯 RGB 基线"训成了双流模型（896 张量），而 `load_state` 当时
只查 missing、不查 unexpected，把 896 装进 348 还打印 "loaded 348/348"，
评测跑在近乎未训练的网络上却照常出数。现在 `load_state` 会直接抛错，
`tests/test_load_state.py` 覆盖了这条路径。

## 4. 重跑被中断的确认实验

```bash
bash scripts/run_confirm.sh            # ce / cbce / cbce_kd × 3407-3409 × 500 epoch
python3 scripts/finish_confirm.py      # 等九次结束，逐臂评测、按 ±0.7/+1.0 规则判读、提交推送
```

实测速度（2×4090、单流、含评测）：CE 臂 ≈16 s/epoch，带教师前向 ≈20 s/epoch。
九次约 22 GPU·h ≈ 11 小时墙钟。纯训练步 12.2 s/epoch（不含评测）——三个口径不要混用。

判据（预注册，不要事后改）：三个 seed 的 `cbce_kd − cbce` 均值与离散度都在 ±0.7 内
→ "无可测增量"；均值 > +1.0 → 才值得继续追教师机制。

## 5. 目前有哪些结果在仓库里

| 位置 | 内容 |
|---|---|
| `results/RESULTS_MIN_SCREEN.md` | 五臂 120 轮表 + 逐类 IoU + 两个预注册对照 |
| `results/protocol/MIN_*.json` | 五臂逐图评测（含 decisive 区间） |
| `results/protocol/MIN_priv_inject*_e120.json` | 两轮注入原型的评测 |
| `results/*screen*/metrics.json` | 每个臂的逐轮验证曲线、配置 sha、教师 sha、类别权重 |
| `docs/MINIMAL_EXPERIMENTS.md` | 五臂矩阵、不变量、预算、两次事故记录 |
| `docs/NEXT_STEPS.md` | 两条路的成本与止损线 |
| `docs/audit/` | 独立审核报告、调整方案及其复核更正、执行记录 |

关键数字（均已实测，勿从旧表复制）：500 轮同批同协议 CE 59.82 / RAPD-v2 59.70 / uniform 59.63；
120 轮教师增量 +0.43（CI [−0.56,+1.58]）、类别重加权 +1.19；注入重试 gate=0.177、夜间 person 44.04（CE 45.43）。

## 6. 丢失与需要重建的东西

| 项 | 状态 |
|---|---|
| 三 seed 确认的 9 次训练 | 中断于 epoch 45/500，**无 checkpoint**，需重跑 |
| 120 轮五臂与两轮注入的 checkpoint | 只在本机 `output/`，未上传 release（单文件 ~103 MB，需要时用 `gh release upload` 补） |
| 逐图评测 JSON | ✅ 已入库（这是证据本体，可脱离 checkpoint 复核） |
| P1（10-25 早）四个臂的 checkpoint | ❌ 早已丢失，无法恢复（见审核报告 §1.3） |
