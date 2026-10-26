#!/usr/bin/env python3
"""Build the paper-facing tables and figures from protocol JSONs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("results/protocol")
FIG = Path("paper/figures")
FIG.mkdir(parents=True, exist_ok=True)
OUT = Path("results/CHEAP_HEAT.md")

CLASSES = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]


def load(p):
    return json.loads((R / p).read_text())


def person_gate(files):
    rows = []
    for f in files:
        d = load(f)["variants"]["person_or"]
        rows.append((d["overall"]["miou_official"], d["overall"]["miou_foreground"],
                     d["conditions"]["night"]["per_class_iou"]["person"]))
    a = np.array(rows)
    return a.mean(0), a.std(0), rows


ce = load("CE120_last_s3407.json")
ce_o = ce["overall"]["miou_official"]
ce_fg = ce["overall"]["miou_foreground"]
ce_np = ce["conditions"]["night"]["person"]["iou"]

lr8_m, lr8_s, _ = person_gate(["UNETGATE_variants_lr8_ce120.json",
                               "UNETGATE_variants_lr8_ce120_s3408.json",
                               "UNETGATE_variants_lr8_ce120_s3409.json"])
lr4_m, lr4_s, lr4_rows = person_gate(["UNETGATE_variants_lr4_ce120.json",
                                      "UNETGATE_variants_lr4_ce120_s3408.json",
                                      "UNETGATE_variants_lr4_ce120_s3409.json"])
lr2_m, lr2_s, _ = person_gate(["UNETGATE_variants_lr2_ce120.json"])

deg = load("UNETGATE_degraded_lr4_ce120.json")["variants"]
clean = load("UNETGATE_variants_lr4_ce120.json")["variants"]
noh_c = load("UNETGATE_nohard_clean_lr4_ce120.json")["variants"]
noh_d = load("UNETGATE_nohard_degraded_lr4_ce120.json")["variants"]

# ---------------------------------------------------------------- figure 1 --
fig, ax = plt.subplots(figsize=(5.2, 3.4))
xs = np.array([1 / 64, 1 / 16, 1 / 4])
ax.errorbar(xs, [lr8_m[2], lr4_m[2], lr2_m[2]],
            yerr=[lr8_s[2], lr4_s[2], lr2_s[2]], marker="o", capsize=3,
            color="#1f77b4", label="gate (night person)")
ax.axhline(ce_np, ls="--", color="#555555", lw=1, label=f"RGB-only ({ce_np:.1f})")
ax.axhline(65.74, ls=":", color="#d62728", lw=1, label="full-res teacher (65.7)")
ax.set_xscale("log", base=2)
ax.set_xticks(xs)
ax.set_xticklabels(["1/64\n60x80", "1/16\n120x160", "1/4\n240x320"])
ax.set_xlabel("thermal pixels relative to 480x640")
ax.set_ylabel("night pedestrian IoU")
ax.set_ylim(40, 70)
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "pareto_night_person.pdf")
fig.savefig(FIG / "pareto_night_person.png", dpi=220)
plt.close(fig)

# ---------------------------------------------------------------- figure 2 --
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=False)
names = ["RGB only", "thermal only", "full gate", "person gate"]
keys = ["A", "B", "full", "person_or"]
colors = ["#7f7f7f", "#ff7f0e", "#2ca02c", "#1f77b4"]
for ax, cond, title in ((axs[0], "clean", "clean"), (axs[1], "degraded", "degraded")):
    d = clean if cond == "clean" else deg
    vals = [d[k]["overall"]["miou_official"] for k in keys]
    nps = [d[k]["conditions"]["night"]["per_class_iou"]["person"] for k in keys]
    x = np.arange(len(keys))
    ax.bar(x - 0.2, vals, 0.4, color=colors, label="official mIoU")
    ax.bar(x + 0.2, nps, 0.4, color=colors, alpha=0.45, hatch="//", label="night person")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set_title(title)
    ax.set_ylim(25, 66)
    ax.grid(axis="y", alpha=0.25)
axs[0].set_ylabel("IoU")
axs[0].legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "clean_degraded.pdf")
fig.savefig(FIG / "clean_degraded.png", dpi=220)
plt.close(fig)

# ---------------------------------------------------------------- figure 3 --
ce_pc = ce["overall"]["per_class_iou"]
g_pc = clean["person_or"]["overall"]["per_class_iou"]
deltas = [(g_pc[c] - ce_pc[c]) for c in CLASSES[1:]]
fig, ax = plt.subplots(figsize=(5.6, 3.0))
x = np.arange(len(CLASSES[1:]))
col = ["#2ca02c" if v >= 0 else "#d62728" for v in deltas]
ax.bar(x, deltas, color=col)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(CLASSES[1:], rotation=25, ha="right", fontsize=8)
ax.set_ylabel("IoU change vs RGB-only")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "per_class_delta.pdf")
fig.savefig(FIG / "per_class_delta.png", dpi=220)
plt.close(fig)

# ------------------------------------------------------------------ report --
def fmt(m, s):
    return f"{m:.2f} ± {s:.2f}"

lines = [
    "# Cheap Heat: final protocol numbers",
    "",
    "Gate protocol: trained on a 627/157 split of `train.txt` with hard-condition "
    "augmentation, checkpoint selected on the 157 held-out hard split, official "
    "392-image test evaluated exactly once. The deployment rule only lets the "
    "thermal branch override pixels where either model predicts `person`.",
    "",
    "## Sensor-budget Pareto",
    "",
    "| thermal | pixels | official | fg | night person |",
    "|---|---:|---:|---:|---:|",
    f"| none (CE120) | 0 | {ce_o:.2f} | {ce_fg:.2f} | {ce_np:.2f} |",
    f"| LR8 60x80 | 1/64 | {fmt(lr8_m[0], lr8_s[0])} | {fmt(lr8_m[1], lr8_s[1])} | {fmt(lr8_m[2], lr8_s[2])} |",
    f"| LR4 120x160 | 1/16 | {fmt(lr4_m[0], lr4_s[0])} | {fmt(lr4_m[1], lr4_s[1])} | {fmt(lr4_m[2], lr4_s[2])} |",
    f"| LR2 240x320 | 1/4 | {fmt(lr2_m[0], lr2_s[0])} | {fmt(lr2_m[1], lr2_s[1])} | {fmt(lr2_m[2], lr2_s[2])} |",
    "| full-res teacher | 1 | 61.28 | — | 65.74 |",
    "",
    "LR8/LR4 are 3 gate-training seeds with the same frozen experts; LR2 is one seed.",
    "",
    "## Clean vs degraded",
    "",
    "| method | clean official | clean night person | degraded official | degraded night person |",
    "|---|---:|---:|---:|---:|",
]
for name, k in zip(names, keys):
    lines.append(f"| {name} | {clean[k]['overall']['miou_official']:.2f} | "
                 f"{clean[k]['conditions']['night']['per_class_iou']['person']:.2f} | "
                 f"{deg[k]['overall']['miou_official']:.2f} | "
                 f"{deg[k]['conditions']['night']['per_class_iou']['person']:.2f} |")
lines += [
    "",
    "## Hard-augmentation ablation (LR4)",
    "",
    "| gate training | clean official | clean night person | degraded official | degraded night person |",
    "|---|---:|---:|---:|---:|",
    f"| hard aug | {clean['person_or']['overall']['miou_official']:.2f} | "
    f"{clean['person_or']['conditions']['night']['per_class_iou']['person']:.2f} | "
    f"{deg['full']['overall']['miou_official']:.2f} | "
    f"{deg['full']['conditions']['night']['per_class_iou']['person']:.2f} |",
    f"| no hard aug | {noh_c['person_or']['overall']['miou_official']:.2f} | "
    f"{noh_c['person_or']['conditions']['night']['per_class_iou']['person']:.2f} | "
    f"{noh_d['full']['overall']['miou_official']:.2f} | "
    f"{noh_d['full']['conditions']['night']['per_class_iou']['person']:.2f} |",
    "",
    "## Per-class change (LR4 person gate vs RGB-only)",
    "",
    "| class | RGB-only | gate | delta |",
    "|---|---:|---:|---:|",
]
for c in CLASSES[1:]:
    lines.append(f"| {c} | {ce_pc[c]:.2f} | {g_pc[c]:.2f} | {g_pc[c]-ce_pc[c]:+.2f} |")
lines += [
    "",
    "## Caveats",
    "",
    "- Low-resolution thermal is simulated by bicubic down/up sampling, not a real sensor.",
    "- The gate uses two frozen B2 models; compute is not yet matched to a single model.",
    "- MFNet only; no second dataset or real low-resolution capture yet.",
    "- The gate is selected on a held-out split of `train.txt`; the 392-image test is touched once.",
]
OUT.write_text("\n".join(lines) + "\n")
print("wrote", OUT)
print("wrote", FIG)
