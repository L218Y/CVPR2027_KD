"""Backfill last10_miou / eval_history into metrics.json from the run logs."""
import json, re
from pathlib import Path

SOURCES = {
    "MFNet_CMNeXt-B2_it_teacher_s3407": "../teacher_run.log",
}


def history(path):
    text = Path(path).read_text(errors="ignore").replace("\r", "\n")
    return [(int(a), float(b)) for a, b in re.findall(r"epoch (\d+) mIoU ([\d.]+)", text)]


for metrics_path in Path("results").glob("*/metrics.json"):
    name = metrics_path.parent.name
    log = SOURCES.get(name)
    if log is None:
        cand = Path("logs") / f"{name}.log"
        log = cand if cand.is_file() else Path("output") / name / "train.log"
    if not Path(log).is_file():
        print("no log for", name)
        continue
    hist = history(log)
    if not hist:
        continue
    m = json.loads(metrics_path.read_text())
    m["eval_history"] = hist
    m["last10_miou"] = round(sum(x for _, x in hist[-10:]) / len(hist[-10:]), 3)
    metrics_path.write_text(json.dumps(m, indent=2))
    print(f"{name}: last10 {m['last10_miou']:.2f}")
