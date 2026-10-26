#!/usr/bin/env bash
# Hallucination baseline: RGB -> thermal translation, then a two-stream model on
# the fake pair. This is what the field normally does for "train multimodal,
# deploy RGB only", so a negative result is only meaningful next to it.
#
# Runs after the confirmation job so the two do not fight for the GPUs, then uses
# both cards: two seeds of the same baseline in parallel.
#
#   bash scripts/run_hallu_baseline.sh
set -uo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-120}"
GEN_EPOCHS="${GEN_EPOCHS:-40}"
mkdir -p logs_screen output/gen

echo "[$(date -u +%H:%M:%S)] training the RGB->thermal generator"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:tools python3 tools/train_thermal_gen.py \
  --epochs "$GEN_EPOCHS" > logs_screen/gen_thermal.log 2>&1 || { echo "generator failed"; exit 1; }
tail -1 logs_screen/gen_thermal.log

echo "[$(date -u +%H:%M:%S)] materialising the hallucinated dataset"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:tools python3 tools/make_hallu_dataset.py \
  > logs_screen/hallu_dataset.log 2>&1 || { echo "dataset build failed"; exit 1; }
tail -1 logs_screen/hallu_dataset.log

echo "[$(date -u +%H:%M:%S)] training two seeds on hallucinated thermal"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:tools python3 tools/train_kd.py \
  --cfg configs/min/hallu.yaml --seed 3407 --epochs "$EPOCHS" --tag screen_hallu \
  > logs_screen/screen_hallu_s3407.log 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:tools python3 tools/train_kd.py \
  --cfg configs/min/hallu.yaml --seed 3408 --epochs "$EPOCHS" --tag screen_hallu_s3408 \
  > logs_screen/screen_hallu_s3408.log 2>&1 &
B=$!
wait "$A"; ra=$?
wait "$B"; rb=$?
echo "[$(date -u +%H:%M:%S)] hallu training rc=$ra/$rb"

for seed in 3407 3408; do
  d=$(ls -d output/min/hallu/*screen_hallu_s${seed} 2>/dev/null | head -1)
  [ -z "$d" ] && continue
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:tools python3 scripts/eval_protocol.py \
    --cfg configs/min/hallu.yaml --ckpt "$d/last.pth" \
    --teacher-ckpt output/MFNet_CMNeXt-B2_it_teacher_s3407/best.pth \
    --out "results/protocol/MIN_hallu_${seed}_e${EPOCHS}.json" > "logs_screen/hallu_eval_${seed}.log" 2>&1
done

python3 -c "
import json, glob
print('arm            official    fg   night_person')
for f in sorted(glob.glob('results/protocol/MIN_hallu_*_e${EPOCHS}.json')):
    d = json.load(open(f)); o = d['overall']
    print('%-14s %7.2f %6.2f %10.2f' % (f.split('/')[-1].replace('MIN_','').replace('_e${EPOCHS}.json',''),
          o['miou_official'], o['miou_foreground'], d['conditions']['night']['person']['iou']))
print('ce@120: 55.32 50.09 45.43')
"
