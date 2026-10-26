#!/usr/bin/env bash
# 48-hour sensor-budget gate: can a low-resolution thermal sensor recover the
# full-resolution teacher's advantage?
#
#   GPU0  lr8  (60x80 thermal, upsampled to 480x640)
#   GPU1  lr4  (120x160 thermal, upsampled to 480x640)
#
# Both are 120-ep RGB+thermal students trained exactly like the full-resolution
# teacher, so the only variable is thermal resolution.  The gate passes if the
# TCE (against the 61.28 full-resolution teacher and 55.32 CE student) is high
# enough to justify building the resolution-aware distillation method.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/tools"

EPOCHS="${EPOCHS:-120}"
SEED="${SEED:-3407}"
mkdir -p logs_p0 results/protocol

echo "[$(date -u +%H:%M:%S)] lr8 on GPU0, lr4 on GPU1"
CUDA_VISIBLE_DEVICES=0 python3 tools/train_kd.py --cfg configs/min/lr8_teacher.yaml \
  --seed "$SEED" --epochs "$EPOCHS" --tag p0lr_s${SEED} > logs_p0/lr8_train.log 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 python3 tools/train_kd.py --cfg configs/min/lr4_teacher.yaml \
  --seed "$SEED" --epochs "$EPOCHS" --tag p0lr_s${SEED} > logs_p0/lr4_train.log 2>&1 &
B=$!
wait "$A"; RA=$?
wait "$B"; RB=$?
echo "[$(date -u +%H:%M:%S)] training done rc=$RA/$RB"

for arm in lr8 lr4; do
  d="$(ls -d output/min/${arm}_teacher/*p0lr_s${SEED} 2>/dev/null | head -1)"
  [ -n "$d" ] && [ -f "$d/last.pth" ] || { echo "skip $arm (no last.pth)"; continue; }
  echo "[$(date -u +%H:%M:%S)] eval $arm"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_protocol.py \
    --cfg "configs/min/${arm}_teacher.yaml" \
    --ckpt "$d/last.pth" \
    --teacher-ckpt output/MFNet_CMNeXt-B2_it_teacher_s3407/best.pth \
    --out "results/protocol/GATE_${arm}_s${SEED}.json" \
    > "logs_p0/eval_${arm}.log" 2>&1 || true
done

python3 - <<'PY'
import glob, json
print("arm        official    fg   night_person   TCE(official)  TCE(night)")
for f in sorted(glob.glob("results/protocol/GATE_*.json")):
    d = json.load(open(f)); o = d["overall"]; n = d["conditions"]["night"]["person"]["iou"]
    tce_o = (o["miou_official"] - 55.3218) / (61.28 - 55.3218)
    tce_n = (n - 45.427) / (65.74 - 45.427)
    print("%-10s %8.2f %6.2f %12.2f %14.3f %11.3f" % (
        f.split("GATE_")[-1].replace(".json", ""), o["miou_official"],
        o["miou_foreground"], n, tce_o, tce_n))
print("reference: CE 55.32/50.09/45.43; full-T 61.28/--/65.74")
PY

echo "[$(date -u +%H:%M:%S)] publishing gate results"
bash scripts/autopublish.sh --status 0 --note "LR gate" || true
echo "[$(date -u +%H:%M:%S)] gate done"
