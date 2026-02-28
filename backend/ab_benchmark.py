#!/usr/bin/env python
"""
Run an A/B backtest comparison between:
  A) a baseline git ref (commit/branch/tag), and
  B) the current working tree.

The script creates a temporary worktree for A, runs the same backtest payload
extraction in both environments, and prints/saves metric deltas.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PAYLOAD_SNIPPET = r"""
import json
import os
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from hydra import initialize, compose
from data.dataset import SalesDataset
from model.forecaster import Forecaster
from backtest import BacktestRunner
from utils.helper import resolve_target_keys

seed = int(os.environ.get("AB_SEED", "42"))
np.random.seed(seed)
random.seed(seed)

overrides = json.loads(os.environ.get("AB_OVERRIDES_JSON", "[]"))

with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name="config", overrides=overrides)

prepare_config = resolve_target_keys(cfg)
dataset_cfg = getattr(cfg, "data", {})
dataset_extra_cfg = getattr(cfg, "dataset_config", {})

ds = SalesDataset(
    csv_file=dataset_cfg.file_path,
    separator=dataset_cfg.separator,
    date_column=dataset_cfg.date_column,
    target_column=dataset_cfg.target_column,
    nr_horizon=dataset_cfg.nr_horizon,
    prepare_config=prepare_config,
    column_selection=cfg.column_selection,
    product_var=dataset_cfg.product_column,
    product_name=dataset_cfg.product_name,
    granularity=dataset_cfg.granularity,
    config=dataset_extra_cfg,
)
items = list(ds.get_all_products())

runner = BacktestRunner(
    dataset_cls=SalesDataset,
    forecaster_cls=Forecaster,
    cfg=cfg,
    tracking_uri=cfg.mlflow.tracking_uri,
    experiment_name=cfg.mlflow.experiment_name,
    output_dir=os.environ.get("AB_OUTPUT_DIR", "outputs/backtest_ab"),
    n_cuts=cfg.backtest.n_cuts,
)
payload = runner.run_all(items)
print("__AB_PAYLOAD_START__")
print(json.dumps(payload))
print("__AB_PAYLOAD_END__")
"""


def run_cmd(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=check,
    )


def extract_payload(backend_dir: Path, overrides: list[str], output_dir: str) -> dict[str, Any]:
    env = dict(os.environ)
    env["AB_SEED"] = "42"
    env["AB_OVERRIDES_JSON"] = json.dumps(overrides)
    env["AB_OUTPUT_DIR"] = output_dir
    env["PYTHONHASHSEED"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", PAYLOAD_SNIPPET],
            cwd=str(backend_dir),
            text=True,
            capture_output=True,
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "A/B payload subprocess failed.\n"
            f"cmd={exc.cmd}\n"
            f"returncode={exc.returncode}\n"
            f"STDOUT tail:\n{(exc.stdout or '')[-2000:]}\n"
            f"STDERR tail:\n{(exc.stderr or '')[-2000:]}"
        ) from exc
    out = proc.stdout
    s = "__AB_PAYLOAD_START__"
    e = "__AB_PAYLOAD_END__"
    if s not in out or e not in out:
        raise RuntimeError(
            "Could not extract payload markers from subprocess output.\n"
            f"STDOUT tail:\n{out[-2000:]}\nSTDERR tail:\n{proc.stderr[-2000:]}"
        )
    payload_json = out.split(s, 1)[1].split(e, 1)[0].strip()
    return json.loads(payload_json)


def sanitize_token(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")
    return cleaned or "unknown"


def git_short_hash(repo_dir: Path, ref: str = "HEAD") -> str:
    try:
        out = run_cmd(["git", "rev-parse", "--short", ref], cwd=repo_dir).stdout.strip()
        return sanitize_token(out)
    except Exception:
        return "unknown"


def is_git_dirty(repo_dir: Path) -> bool:
    try:
        out = run_cmd(["git", "status", "--porcelain"], cwd=repo_dir).stdout.strip()
        return bool(out)
    except Exception:
        return False


def finite_mean(values: list[float]) -> float:
    valid = [v for v in values if v is not None and math.isfinite(v)]
    if not valid:
        return float("nan")
    return float(sum(valid) / len(valid))


def finite_std(values: list[float]) -> float:
    valid = [v for v in values if v is not None and math.isfinite(v)]
    if len(valid) < 2:
        return 0.0
    m = finite_mean(valid)
    var = sum((v - m) ** 2 for v in valid) / (len(valid) - 1)
    return float(math.sqrt(var))


def per_item_summary(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item_row in payload["items"]:
        item = item_row["item"]
        out[item] = item_row["summary"]
    return out


def nonzero_item_wape(payload: dict[str, Any]) -> float:
    summaries = per_item_summary(payload).values()
    vals = [
        s.get("WAPE_mean")
        for s in summaries
        if s.get("MAPE_mean") is not None and math.isfinite(s.get("MAPE_mean"))
    ]
    return finite_mean(vals)


def model_selection_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    ms = payload.get("model_selection_summary", {}) or {}
    gd = ms.get("global_decision", {}) or {}
    gk = ms.get("global_kpi_4w_8w", {}) or {}
    return {
        "status": str(gd.get("status", "UNKNOWN")).upper(),
        "pass_rate": (gd.get("gate", {}) or {}).get("pass_rate"),
        "CumWAPE_8w": gk.get("CumWAPE_8w"),
        "CumWAPE_4w": gk.get("CumWAPE_4w"),
        "Bias_8w_pct": gk.get("Bias_8w_pct"),
        "Coverage10_90_8w": gk.get("Coverage10_90_8w"),
    }


def run_arm(
    backend_dir: Path, overrides: list[str], arm_root_dir: Path, repeats: int
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for i in range(1, repeats + 1):
        out_dir = arm_root_dir / f"run_{i:02d}" / "backtest_outputs"
        payloads.append(
            extract_payload(
                backend_dir=backend_dir,
                overrides=overrides,
                output_dir=str(out_dir),
            )
        )
    return payloads


def aggregate_arm(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    global_wape_vals = [p["global_summary"].get("WAPE_mean") for p in payloads]
    nonzero_wape_vals = [nonzero_item_wape(p) for p in payloads]
    model_sel_rows = [model_selection_snapshot(p) for p in payloads]

    # Per-item WAPE across repeats (exclude undefined/all-zero-demand via MAPE check).
    item_series: dict[str, list[float]] = {}
    for p in payloads:
        s_map = per_item_summary(p)
        for item, s in s_map.items():
            w = s.get("WAPE_mean")
            m = s.get("MAPE_mean")
            if (
                w is None
                or m is None
                or not math.isfinite(w)
                or not math.isfinite(m)
            ):
                continue
            item_series.setdefault(item, []).append(float(w))

    per_item_mean = {k: finite_mean(v) for k, v in item_series.items()}
    per_item_std = {k: finite_std(v) for k, v in item_series.items()}

    gate_pass_values = [1.0 if r.get("status") == "PASS" else 0.0 for r in model_sel_rows]
    cumwape8_vals = [r.get("CumWAPE_8w") for r in model_sel_rows]
    cumwape4_vals = [r.get("CumWAPE_4w") for r in model_sel_rows]
    bias8_vals = [r.get("Bias_8w_pct") for r in model_sel_rows]
    cov8_vals = [r.get("Coverage10_90_8w") for r in model_sel_rows]
    pass_rate_vals = [r.get("pass_rate") for r in model_sel_rows]

    return {
        "n_runs": len(payloads),
        "global_wape_values": global_wape_vals,
        "global_wape_mean": finite_mean(global_wape_vals),
        "global_wape_std": finite_std(global_wape_vals),
        "nonzero_item_wape_values": nonzero_wape_vals,
        "nonzero_item_wape_mean": finite_mean(nonzero_wape_vals),
        "nonzero_item_wape_std": finite_std(nonzero_wape_vals),
        "per_item_wape_mean": per_item_mean,
        "per_item_wape_std": per_item_std,
        "last_global_summary": payloads[-1]["global_summary"] if payloads else {},
        "model_selection_statuses": [r.get("status") for r in model_sel_rows],
        "gate_pass_rate": finite_mean(gate_pass_values),
        "pass_rate_mean": finite_mean(pass_rate_vals),
        "CumWAPE_8w_mean": finite_mean(cumwape8_vals),
        "CumWAPE_4w_mean": finite_mean(cumwape4_vals),
        "Bias_8w_pct_mean": finite_mean(bias8_vals),
        "Coverage10_90_8w_mean": finite_mean(cov8_vals),
        "last_model_selection_snapshot": model_sel_rows[-1] if model_sel_rows else {},
    }


def rank_gate_first(arm: dict[str, Any]) -> tuple:
    gate_ok = arm.get("gate_pass_rate")
    gate_ok = gate_ok if (gate_ok is not None and math.isfinite(gate_ok)) else 0.0
    cum8 = arm.get("CumWAPE_8w_mean")
    cum4 = arm.get("CumWAPE_4w_mean")
    cum8 = cum8 if (cum8 is not None and math.isfinite(cum8)) else float("inf")
    cum4 = cum4 if (cum4 is not None and math.isfinite(cum4)) else float("inf")
    return (0 if gate_ok >= 0.5 else 1, cum8, cum4)


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B compare baseline ref vs working tree.")
    parser.add_argument("--baseline-ref", default="HEAD", help="Git ref for baseline (default: HEAD)")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated runs per arm to average out variance (default: 1)",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Extra Hydra overrides. Example: --overrides model=light_gbm backtest.n_cuts=2",
    )
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parent
    repo_root = backend_dir.parent
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_worktree = repo_root.parent / f"_ab_tmp_{ts}"
    baseline_ref_label = sanitize_token(args.baseline_ref)
    baseline_hash = git_short_hash(repo_root, args.baseline_ref)
    candidate_hash = git_short_hash(repo_root, "HEAD")
    candidate_dirty = is_git_dirty(repo_root)
    candidate_label = f"{candidate_hash}-dirty" if candidate_dirty else candidate_hash

    # Keep names readable but short to avoid Windows path-length issues.
    experiment_name = f"ab_{ts}_b-{baseline_hash}_c-{candidate_label}"

    # New clear structure:
    # outputs/ab_experiments/<experiment_name>/<arm>/run_xx/backtest_outputs/...
    # results/ab_reports/<experiment_name>/ab_report_summary.json
    outputs_experiment_root = backend_dir / "outputs" / "ab_experiments" / experiment_name
    baseline_arm_dir = outputs_experiment_root / f"baseline_{baseline_hash}"
    candidate_arm_dir = outputs_experiment_root / f"candidate_{candidate_label}"
    reports_dir = backend_dir / "results" / "ab_reports" / experiment_name
    reports_dir.mkdir(parents=True, exist_ok=True)

    base_overrides = [
        "model=light_gbm",
        "backtest.enabled=true",
        "backtest.n_cuts=2",
        "model.params.random_state=42",
        "model.params.n_jobs=1",
        "+model.params.bagging_seed=42",
        "+model.params.feature_fraction_seed=42",
        "+model.params.data_random_seed=42",
        "+model.params.verbosity=-1",
        "+model.params.force_col_wise=true",
    ]
    overrides = base_overrides + args.overrides

    print(f"[A/B] baseline_ref={args.baseline_ref}")
    print(f"[A/B] baseline_hash={baseline_hash}")
    print(f"[A/B] candidate_hash={candidate_hash}  dirty={candidate_dirty}")
    print(f"[A/B] repeats={args.repeats}")
    print(f"[A/B] overrides={overrides}")
    print(f"[A/B] experiment={experiment_name}")

    run_cmd(["git", "worktree", "add", "--detach", str(tmp_worktree), args.baseline_ref], cwd=repo_root)
    try:
        baseline_payloads = run_arm(
            backend_dir=tmp_worktree / "backend",
            overrides=overrides,
            arm_root_dir=baseline_arm_dir,
            repeats=args.repeats,
        )
        candidate_payloads = run_arm(
            backend_dir=backend_dir,
            overrides=overrides,
            arm_root_dir=candidate_arm_dir,
            repeats=args.repeats,
        )
    finally:
        run_cmd(["git", "worktree", "remove", "--force", str(tmp_worktree)], cwd=repo_root, check=False)

    base_agg = aggregate_arm(baseline_payloads)
    cand_agg = aggregate_arm(candidate_payloads)

    base_item_wape = base_agg["per_item_wape_mean"]
    cand_item_wape = cand_agg["per_item_wape_mean"]
    all_items = sorted(set(base_item_wape) | set(cand_item_wape))

    base_nonzero_wape = base_agg["nonzero_item_wape_mean"]
    cand_nonzero_wape = cand_agg["nonzero_item_wape_mean"]
    per_item_delta: dict[str, float | None] = {}
    for item in all_items:
        b = base_item_wape.get(item)
        c = cand_item_wape.get(item)
        if b is None or c is None:
            per_item_delta[item] = None
        else:
            per_item_delta[item] = float(c - b)

    kpi_48_delta = {
        "CumWAPE_8w_delta": (
            cand_agg["CumWAPE_8w_mean"] - base_agg["CumWAPE_8w_mean"]
            if math.isfinite(cand_agg["CumWAPE_8w_mean"]) and math.isfinite(base_agg["CumWAPE_8w_mean"])
            else None
        ),
        "CumWAPE_4w_delta": (
            cand_agg["CumWAPE_4w_mean"] - base_agg["CumWAPE_4w_mean"]
            if math.isfinite(cand_agg["CumWAPE_4w_mean"]) and math.isfinite(base_agg["CumWAPE_4w_mean"])
            else None
        ),
        "Bias_8w_pct_delta": (
            cand_agg["Bias_8w_pct_mean"] - base_agg["Bias_8w_pct_mean"]
            if math.isfinite(cand_agg["Bias_8w_pct_mean"]) and math.isfinite(base_agg["Bias_8w_pct_mean"])
            else None
        ),
        "Coverage10_90_8w_delta": (
            cand_agg["Coverage10_90_8w_mean"] - base_agg["Coverage10_90_8w_mean"]
            if math.isfinite(cand_agg["Coverage10_90_8w_mean"]) and math.isfinite(base_agg["Coverage10_90_8w_mean"])
            else None
        ),
    }

    base_rank = rank_gate_first(base_agg)
    cand_rank = rank_gate_first(cand_agg)
    if cand_rank < base_rank:
        winner = "candidate"
    elif base_rank < cand_rank:
        winner = "baseline"
    else:
        winner = "tie"

    gate_result = {
        "baseline": {
            "status_last": base_agg["last_model_selection_snapshot"].get("status"),
            "gate_pass_rate": base_agg["gate_pass_rate"],
            "pass_rate_mean": base_agg["pass_rate_mean"],
            "CumWAPE_8w": base_agg["CumWAPE_8w_mean"],
            "CumWAPE_4w": base_agg["CumWAPE_4w_mean"],
            "Bias_8w_pct": base_agg["Bias_8w_pct_mean"],
            "Coverage10_90_8w": base_agg["Coverage10_90_8w_mean"],
        },
        "candidate": {
            "status_last": cand_agg["last_model_selection_snapshot"].get("status"),
            "gate_pass_rate": cand_agg["gate_pass_rate"],
            "pass_rate_mean": cand_agg["pass_rate_mean"],
            "CumWAPE_8w": cand_agg["CumWAPE_8w_mean"],
            "CumWAPE_4w": cand_agg["CumWAPE_4w_mean"],
            "Bias_8w_pct": cand_agg["Bias_8w_pct_mean"],
            "Coverage10_90_8w": cand_agg["Coverage10_90_8w_mean"],
        },
        "winner_by_gate_then_cumwape8w": winner,
    }

    report = {
        "timestamp": dt.datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "baseline_ref": args.baseline_ref,
        "baseline_hash": baseline_hash,
        "candidate": "working_tree",
        "candidate_hash": candidate_hash,
        "candidate_dirty": candidate_dirty,
        "repeats": args.repeats,
        "overrides": overrides,
        "artifact_paths": {
            "outputs_experiment_root": str(outputs_experiment_root),
            "baseline_outputs_root": str(baseline_arm_dir),
            "candidate_outputs_root": str(candidate_arm_dir),
            "report_dir": str(reports_dir),
        },
        "baseline_aggregate": base_agg,
        "candidate_aggregate": cand_agg,
        "gate_result": gate_result,
        "delta_kpi_4w_8w": kpi_48_delta,
        "delta_nonzero_item_wape_mean": (
            cand_nonzero_wape - base_nonzero_wape
            if math.isfinite(base_nonzero_wape) and math.isfinite(cand_nonzero_wape)
            else None
        ),
        "baseline_per_item_wape": base_item_wape,
        "candidate_per_item_wape": cand_item_wape,
        "delta_per_item_wape": per_item_delta,
    }

    report_path = reports_dir / "ab_report_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Keep a short README in the same folder so each experiment is self-descriptive.
    readme_path = reports_dir / "README.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "A/B experiment\n"
            f"experiment_name: {experiment_name}\n"
            f"baseline: ref={args.baseline_ref}, hash={baseline_hash}\n"
            f"candidate: working_tree, hash={candidate_hash}, dirty={candidate_dirty}\n"
            f"repeats: {args.repeats}\n"
            f"winner_by_gate_then_cumwape8w: {winner}\n"
            f"baseline_outputs_root: {baseline_arm_dir}\n"
            f"candidate_outputs_root: {candidate_arm_dir}\n"
            f"summary_report: {report_path}\n"
        )

    def _fmt(v: Any) -> str:
        return f"{v:.3f}" if (v is not None and isinstance(v, (int, float)) and math.isfinite(v)) else "n/a"

    print("\n=== A/B Summary ===")
    print(
        f"Baseline global WAPE_mean:   {base_agg['global_wape_mean']:.4f}"
        f"  (+/-{base_agg['global_wape_std']:.4f})"
    )
    print(
        f"Candidate global WAPE_mean:  {cand_agg['global_wape_mean']:.4f}"
        f"  (+/-{cand_agg['global_wape_std']:.4f})"
    )
    print(
        f"Baseline non-zero item WAPE: {base_nonzero_wape:.4f}"
        f"  (+/-{base_agg['nonzero_item_wape_std']:.4f})"
    )
    print(
        f"Candidate non-zero item WAPE:{cand_nonzero_wape:.4f}"
        f"  (+/-{cand_agg['nonzero_item_wape_std']:.4f})"
    )
    if report["delta_nonzero_item_wape_mean"] is not None:
        print(f"Delta non-zero item WAPE:    {report['delta_nonzero_item_wape_mean']:+.4f}")
    print("\nGate result (4/8w focus):")
    print(
        f"  Baseline  status={gate_result['baseline']['status_last']} "
        f"gate_pass_rate={_fmt(gate_result['baseline']['gate_pass_rate'])} "
        f"CumWAPE_8w={_fmt(gate_result['baseline']['CumWAPE_8w'])} "
        f"CumWAPE_4w={_fmt(gate_result['baseline']['CumWAPE_4w'])}"
    )
    print(
        f"  Candidate status={gate_result['candidate']['status_last']} "
        f"gate_pass_rate={_fmt(gate_result['candidate']['gate_pass_rate'])} "
        f"CumWAPE_8w={_fmt(gate_result['candidate']['CumWAPE_8w'])} "
        f"CumWAPE_4w={_fmt(gate_result['candidate']['CumWAPE_4w'])}"
    )
    print(
        "  Delta: "
        f"CumWAPE_8w={_fmt(kpi_48_delta['CumWAPE_8w_delta'])}  "
        f"CumWAPE_4w={_fmt(kpi_48_delta['CumWAPE_4w_delta'])}  "
        f"Bias_8w_pct={_fmt(kpi_48_delta['Bias_8w_pct_delta'])}  "
        f"Coverage10_90_8w={_fmt(kpi_48_delta['Coverage10_90_8w_delta'])}"
    )
    print(f"  Winner (gate->CumWAPE_8w): {winner}")
    print("\nPer-item WAPE delta (candidate - baseline):")
    for item in all_items:
        d = per_item_delta[item]
        if d is None:
            print(f"  {item}: n/a")
        else:
            print(f"  {item}: {d:+.4f}")
    print(f"\nReport saved: {report_path}")
    print(f"Experiment README: {readme_path}")
    print(f"Baseline plots/data: {baseline_arm_dir}")
    print(f"Candidate plots/data: {candidate_arm_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
