#!/usr/bin/env python3
"""Calibrate the "22% teacher-only" statistic against trivial explanations.

The claim in the paper is that the teacher holds evidence the student cannot use.
The obvious objection is that the 22% is just long-tail classes being hard, or a
base-rate artefact. This script answers it from the already-saved per-image
confusion counts, without touching a GPU.

For each class it reports

  student-wrong      = teacher_only + both_wrong
  rescue rate        = teacher_only / student-wrong   (student errs -> teacher right how often)
  asymmetry          = teacher_only / student_only    (McNemar b/c, same class, same pixels)
  p                  = two-sided McNemar on (b, c); kills "the class is just hard"
  teacher recall     = P(teacher correct | GT = c)
  student recall     = P(student correct | GT = c)

Earlier drafts printed a "ceiling IoU" that was algebraically identical to teacher
recall (its denominator reduces to the class's GT pixel count), so it did not vary
with the student and was not a ceiling. Recall is the honest quantity: IoU cannot
be recovered from this table because false positives outside the class are absent.

The independence baseline was the wrong null and is deliberately gone: teacher and
student see the same RGB, so they agree far more than independence predicts and
every class would show a negative "excess". What matters is the *within-class
asymmetry* between the two kinds of discordant pixels.

Usage:
    python3 tools/analyze_pairing.py results/protocol/MIN_ce_3407_e120.json [...]
"""

import argparse
import json
import math
from pathlib import Path

CLASSES = ["car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]


def analyse(path: Path) -> None:
    data = json.loads(path.read_text())
    print(f"\n=== {path.name} ===")
    for cond in ("day", "night", "all"):
        block = (data.get("pairing_by_condition") or {}).get(cond)
        if not block:
            continue
        by_class = block["all"]["by_class"]
        print(f"\n[{cond}]  class        student-wrong   teacher-only   student-only   rescue%   asym   p        t_recall  s_recall")
        for cls in CLASSES:
            c = by_class.get(cls)
            if not c or not c.get("valid"):
                continue
            valid = c["valid"]
            both = c["both_right"]
            t_only = c["teacher_only"]
            s_only = c["student_only"]
            wrong = c["both_wrong"]

            student_wrong = t_only + wrong
            rescue = 100 * t_only / student_wrong if student_wrong else float("nan")
            asym = t_only / s_only if s_only else float("inf")
            # McNemar without scipy: normal approximation on the discordant pairs.
            b, c = t_only, s_only
            z = (b - c) / math.sqrt(b + c) if (b + c) else 0.0
            pval = math.erfc(abs(z) / math.sqrt(2))

            t_recall = 100 * (both + t_only) / valid
            s_recall = 100 * (both + s_only) / valid
            print(f"       {cls:<12s} {student_wrong:>12d}   {t_only:>12d}   {s_only:>12d}   "
                  f"{rescue:6.1f}   {asym:5.1f}  {pval:8.1e}   {t_recall:7.1f}  {s_recall:8.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    args = ap.parse_args()
    for p in args.jsons:
        path = Path(p)
        if not path.is_file():
            print(f"missing: {path}")
            continue
        analyse(path)
    print("\nrescue%  : when the student is wrong on this class, how often is the teacher right.")
    print("asym     : teacher-only / student-only (McNemar b/c). 1.0 = no advantage.")
    print("p        : two-sided McNemar; small p means the asymmetry is not chance.")
    print("t_recall : teacher recall on this class; s_recall the student's. The gap is what a")
    print("           student could gain only if it could see what the teacher sees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
