#!/usr/bin/env python3
"""Upload finished runs to per-run GitHub releases.

Called by ``scripts/autopublish.sh``. For every run directory named on stdin it

  * reads ``output/<run>/metrics.json`` for the metadata,
  * creates (or reuses) a release tagged ``run-<run>-<started>``,
  * uploads best/last/probe checkpoints, metrics.json, train.log and small
    figures as release assets, skipping assets that already exist,
  * verifies every uploaded asset's size against the local file,
  * writes ``results/releases/<tag>.json`` and ``results/RELEASES.md``.

The token is read from ``--token-file`` and is only placed in a temporary curl
config (mode 600), never in argv or in the repository.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.github.com"
UPLOAD = "https://uploads.github.com"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def api(method: str, url: str, token: str, payload: dict | None = None):
    data = None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
            return r.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            parsed = json.loads(body) if body else None
        except Exception:
            parsed = body.decode(errors="replace")[:500]
        return e.code, parsed


def sanitize(name: str, limit: int = 80) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-").lower()
    return s[:limit].strip("-")


def release_tag(run: str, metrics: dict) -> str:
    started = (metrics.get("provenance") or {}).get("started_utc") or ""
    stamp = re.sub(r"[^0-9T]", "", started)[:13] or time.strftime("%Y%m%dT%H%M", time.gmtime())
    return f"run-{sanitize(run, 60)}-{stamp}"


def asset_files(run_dir: Path) -> list[Path]:
    files: list[Path] = []
    for name in ("best.pth", "last.pth", "probe.pth", "metrics.json", "train.log"):
        p = run_dir / name
        if p.is_file():
            files.append(p)
    for pattern in ("*.png", "*.pdf", "*.csv", "*.json"):
        for p in sorted(run_dir.glob(pattern)):
            if p.name in {"metrics.json"}:
                continue
            if p.stat().st_size <= 20 * 1024 * 1024:
                files.append(p)
    return files


def upload_asset(repo: str, release_id: int, token: str, path: Path, asset_name: str) -> tuple[bool, str]:
    url = f"{UPLOAD}/repos/{repo}/releases/{release_id}/assets?name={urllib.parse.quote(asset_name)}"
    cfg = tempfile.NamedTemporaryFile("w", prefix="cvpr_kd_curl_", delete=False)
    try:
        cfg.write(f'header = "Authorization: Bearer {token}"\n')
        cfg.write('header = "Accept: application/vnd.github+json"\n')
        cfg.write('header = "X-GitHub-Api-Version: 2022-11-28"\n')
        cfg.close()
        os.chmod(cfg.name, 0o600)
        cmd = [
            "curl", "-sS", "-K", cfg.name, "-X", "POST",
            "-H", "Content-Type: application/octet-stream",
            "--retry", "3", "--retry-delay", "10", "--max-time", "5400",
            "--data-binary", f"@{path}", url,
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            return False, f"curl rc={p.returncode}: {p.stderr[-300:]}"
        try:
            d = json.loads(p.stdout)
        except Exception:
            return False, f"bad response: {p.stdout[:300]}"
        if d.get("size") == path.stat().st_size:
            return True, f"{d.get('size')} bytes"
        return False, f"size mismatch remote={d.get('size')} local={path.stat().st_size}"
    finally:
        try:
            os.unlink(cfg.name)
        except OSError:
            pass


def publish_run(repo: str, token: str, run: str, results_dir: Path) -> dict:
    run_dir = Path("output") / run
    run_id = sanitize(run.replace(os.sep, "__").replace("/", "__"), 110)
    metrics_path = run_dir / "metrics.json"
    metrics = {}
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            metrics = {}
    tag = release_tag(run_id, metrics)
    out = {
        "run": run, "tag": tag,
        "dataset": metrics.get("dataset"),
        "kd_mode": metrics.get("kd_mode"),
        "seed": metrics.get("seed"),
        "best_miou": metrics.get("best_miou"),
        "last10_miou": metrics.get("last10_miou"),
        "assets": [], "errors": [],
    }

    status, rel = api("GET", f"{API}/repos/{repo}/releases/tags/{tag}", token)
    if status == 200 and rel:
        log(f"{run}: release {tag} exists (id={rel['id']})")
    else:
        body = (
            f"Experiment `{run}`\n\n"
            f"- dataset: {metrics.get('dataset')}\n"
            f"- mode: {metrics.get('kd_mode')}\n"
            f"- seed: {metrics.get('seed')}\n"
            f"- best mIoU: {metrics.get('best_miou')}\n"
            f"- last-10 mIoU: {metrics.get('last10_miou')}\n"
            f"- config: `{(metrics.get('provenance') or {}).get('config')}`\n"
            f"- code commit: `{(metrics.get('provenance') or {}).get('code_commit')}`\n"
            f"- started: {(metrics.get('provenance') or {}).get('started_utc')}\n\n"
            "Uploaded automatically by `scripts/autopublish.sh` when the run ended. "
            "Checkpoints are the only irreplaceable artefacts; metrics and logs are "
            "also in the repository under `results/`."
        )
        status, rel = api("POST", f"{API}/repos/{repo}/releases", token, {
            "tag_name": tag, "name": f"Run {run}", "body": body,
            "draft": False, "prerelease": False,
        })
        if status not in (200, 201) or not rel:
            out["errors"].append(f"release create failed: {status} {rel}")
            log(f"{run}: release create failed: {status} {rel}")
            return out
        log(f"{run}: created release {tag} (id={rel['id']})")

    rel_id = rel["id"]
    status, existing = api("GET", f"{API}/repos/{repo}/releases/{rel_id}/assets?per_page=100", token)
    present = {a["name"]: a.get("size") for a in (existing or [])} if status == 200 else {}

    for path in asset_files(run_dir):
        name = f"{run_id}_{path.name}"
        if name in present:
            if present[name] == path.stat().st_size:
                out["assets"].append({"name": name, "size": present[name], "status": "present"})
                log(f"{run}: skip present {name}")
                continue
            log(f"{run}: {name} exists with a different size; uploading as -v2")
            name = f"{run_id}_{path.stem}-v2{path.suffix}"
        ok, msg = upload_asset(repo, rel_id, token, path, name)
        out["assets"].append({"name": name, "size": path.stat().st_size, "status": "ok" if ok else "failed", "detail": msg})
        log(f"{run}: {'uploaded' if ok else 'FAILED'} {name} ({msg})")
        if not ok:
            out["errors"].append(f"{name}: {msg}")

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{tag}.json").write_text(json.dumps(out, indent=1) + "\n")

    # Refresh RELEASES.md from every manifest.
    rows = []
    for p in sorted(results_dir.glob("run-*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        n_ok = sum(1 for a in d.get("assets", []) if a.get("status") in ("ok", "present"))
        rows.append((d.get("tag", p.stem), d.get("run", ""), d.get("dataset", ""),
                     d.get("seed", ""), d.get("best_miou", ""), n_ok,
                     len(d.get("errors", []))))
    lines = ["# Release index", "",
             "| tag | run | dataset | seed | best mIoU | assets | errors |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| `%s` | %s | %s | %s | %s | %s | %s |" % r)
    (results_dir.parent / "RELEASES.md").write_text("\n".join(lines) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", nargs="*", help="run directory names; default: read stdin")
    args = ap.parse_args()

    token = Path(args.token_file).read_text().strip()
    if not token:
        log("empty token")
        return 2
    runs = args.runs or [l.strip() for l in sys.stdin if l.strip()]
    if not runs:
        log("no runs")
        return 0
    results_dir = Path(args.results) / "releases"
    failed = 0
    for run in runs:
        if not (Path("output") / run).is_dir():
            log(f"{run}: missing output directory")
            failed += 1
            continue
        try:
            out = publish_run(args.repo, token, run, results_dir)
            failed += 1 if out.get("errors") else 0
        except Exception as e:  # keep going: one bad run must not block the rest
            log(f"{run}: exception {e!r}")
            failed += 1
    log(f"done: {len(runs) - failed}/{len(runs)} runs clean")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
