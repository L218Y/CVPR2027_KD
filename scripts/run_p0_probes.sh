#!/usr/bin/env bash
# P0 probes: modality-dropout teacher (and its RGB-twice control), then the
# naive RGB->thermal hallucination baseline. Designed for 2x RTX 4090.
#
#   bash scripts/run_p0_probes.sh
#
# Stage 1 (parallel, one run per GPU):
#   GPU0  md_teacher  120 ep, thermal randomly replaced by RGB (p=0.5)
#   GPU1  rgb2_teacher 120 ep, two RGB streams (capacity control)
# Stage 2 (hallucination baseline):
#   GPU0  train the RGB->thermal U-Net, materialise data/MFNet_hallu
#   GPU0/1 two 120-ep students on the generated thermal stream
# Finally evaluate everything with scripts/eval_protocol.py and publish.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/tools"

EPOCHS="${EPOCHS:-120}"
GEN_EPOCHS="${GEN_EPOCHS:-40}"
SEED_A="${SEED_A:-3407}"
SEED_B="${SEED_B:-3408}"
mkdir -p logs_p0

echo "[$(date -u +%H:%M:%S)] stage 1: md_teacher on GPU0, rgb2_teacher on GPU1"
CUDA_VISIBLE_DEVICES=0 python3 tools/train_kd.py --cfg configs/min/md_teacher.yaml \
  --seed "$SEED_A" --epochs "$EPOCHS" --tag p0_s${SEED_A} > logs_p0/md_teacher.log 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 python3 tools/train_kd.py --cfg configs/min/rgb2_teacher.yaml \
  --seed "$SEED_A" --epochs "$EPOCHS" --tag p0_s${SEED_A} > logs_p0/rgb2_teacher.log 2>&1 &
B=$!
wait "$A"; RA=$?
wait "$B"; RB=$?
echo "[$(date -u +%H:%M:%S)] stage 1 done rc=$RA/$RB"

echo "[$(date -u +%H:%M:%S)] stage 2: RGB->thermal generator"
CUDA_VISIBLE_DEVICES=0 python3 tools/train_thermal_gen.py --epochs "$GEN_EPOCHS" \
  > logs_p0/thermal_gen.log 2>&1 || { echo "generator failed"; exit 1; }
tail -1 logs_p0/thermal_gen.log

echo "[$(date -u +%H:%M:%S)] stage 2: materialise data/MFNet_hallu"
CUDA_VISIBLE_DEVICES=0 python3 tools/make_hallu_dataset.py > logs_p0/hallu_dataset.log 2>&1 || {
  echo "hallu dataset failed"; exit 1; }
tail -1 logs_p0/hallu_dataset.log

echo "[$(date -u +%H:%M:%S)] stage 2: two hallucination students"
CUDA_VISIBLE_DEVICES=0 python3 tools/train_kd.py --cfg configs/min/hallu.yaml \
  --seed "$SEED_A" --epochs "$EPOCHS" --tag p0_s${SEED_A} > logs_p0/hallu_s${SEED_A}.log 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 python3 tools/train_kd.py --cfg configs/min/hallu.yaml \
  --seed "$SEED_B" --epochs "$EPOCHS" --tag p0_s${SEED_B} > logs_p0/hallu_s${SEED_B}.log 2>&1 &
B=$!
wait "$A"; RA=$?
wait "$B"; RB=$?
echo "[$(date -u +%H:%M:%S)] stage 2 done rc=$RA/$RB"

echo "[$(date -u +%H:%M:%S)] protocol evaluation (same 392 images, no TTA)"
# The modality-dropout teacher is deployed with RGB in both branches. A symlink
# tree makes the standard MFNet loader read RGB for the "thermal" slot, so the
# evaluation uses exactly the input the model saw during training's dropout.
mkdir -p data/MFNet_rgb2
ln -sfn ../MFNet/rgb    data/MFNet_rgb2/rgb
ln -sfn ../MFNet/rgb    data/MFNet_rgb2/ther
ln -sfn ../MFNet/labels data/MFNet_rgb2/labels
cp -f data/MFNet/train.txt data/MFNet_rgb2/train.txt
cp -f data/MFNet/test.txt  data/MFNet_rgb2/test.txt
eval_one() {
  local arm="$1" seed="$2" d out cfg
  d="$(ls -d output/min/${arm}/*p0_s${seed} 2>/dev/null | head -1)"
  [ -n "$d" ] && [ -f "$d/last.pth" ] || { echo "skip $arm seed $seed (no last.pth)"; return 0; }
  out="results/protocol/P0_${arm}_s${seed}.json"
  cfg="configs/min/${arm}.yaml"
  [ "$arm" = "md_teacher" ] && cfg="configs/min/md_teacher_eval.yaml"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_protocol.py \
    --cfg "$cfg" \
    --ckpt "$d/last.pth" \
    --teacher-ckpt output/MFNet_CMNeXt-B2_it_teacher_s3407/best.pth \
    --out "$out" > "logs_p0/eval_${arm}_s${seed}.log" 2>&1 || true
}
eval_one md_teacher "$SEED_A"
eval_one rgb2_teacher "$SEED_A"
eval_one hallu "$SEED_A"
eval_one hallu "$SEED_B"

python3 - <<'PY' || true
import glob, json
print("arm                          official   fg   night_person")
for f in sorted(glob.glob("results/protocol/P0_*.json")):
    d = json.load(open(f)); o = d["overall"]
    n = d["conditions"]["night"]["person"]["iou"]
    print("%-28s %8.2f %6.2f %10.2f" % (f.split("P0_")[-1].replace(".json", ""), o["miou_official"], o["miou_foreground"], n))
print("reference ce@120: 55.32 50.09 45.43")
PY

echo "[$(date -u +%H:%M:%S)] publishing all P0 runs"
bash scripts/autopublish.sh --status 0 --note "P0 probes" || true
echo "[$(date -u +%H:%M:%S)] all P0 probes done"
