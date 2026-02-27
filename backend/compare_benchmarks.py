#!/usr/bin/env python
"""
compare_benchmarks.py -- Print benchmark history and delta between last two runs.

Usage:
    python compare_benchmarks.py              # show all runs + latest delta
    python compare_benchmarks.py --last N     # compare last N runs
"""
import argparse
import json
from pathlib import Path


METRICS = ["WAPE_mean", "WAPE_std", "MAPE_mean", "SMAPE_mean", "MASE_mean"]
LOG_PATH = Path(__file__).parent / "results" / "benchmark_log.jsonl"


def load_runs() -> list[dict]:
    if not LOG_PATH.exists():
        print(f"No benchmark log found at {LOG_PATH}. Run main.py first.")
        return []
    with open(LOG_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt(v) -> str:
    try:
        return f"{float(v):7.2f}"
    except Exception:
        return str(v)


def print_history(runs: list[dict]) -> None:
    header = f"{'#':>3}  {'timestamp':19}  {'git':7}  {'model':15}  " + \
             "  ".join(f"{m:>12}" for m in METRICS)
    print("\n=== Benchmark History ===")
    print(header)
    print("-" * len(header))
    for i, r in enumerate(runs, 1):
        row = (
            f"{i:>3}  {r['timestamp'][:19]}  {r.get('git_hash','?'):7}  "
            f"{r.get('model','?'):15}  "
        )
        row += "  ".join(fmt(r.get(m)) for m in METRICS)
        print(row)


def print_delta(a: dict, b: dict) -> None:
    print(f"\n=== Delta: run #{a['_idx']} -> run #{b['_idx']} "
          f"({a.get('git_hash','?')} -> {b.get('git_hash','?')}) ===")
    print(f"  model: {a.get('model')} -> {b.get('model')}")

    for m in METRICS:
        va, vb = a.get(m), b.get(m)
        if va is None or vb is None:
            continue
        delta = float(vb) - float(va)
        # For error metrics: negative delta = improvement
        improved = delta < 0 if "std" not in m else delta < 0
        arrow = "DOWN" if delta < 0 else "UP"
        tag = "BETTER" if improved else "WORSE"
        sign = "+" if delta >= 0 else ""
        print(f"  {m:15}: {float(va):7.2f} -> {float(vb):7.2f}  "
              f"({sign}{delta:.2f})  [{arrow}] {tag}")

    # Per-item breakdown
    print("\n  Per-item WAPE_mean:")
    items_a = a.get("per_item", {})
    items_b = b.get("per_item", {})
    all_items = sorted(set(items_a) | set(items_b))
    for it in all_items:
        wa = items_a.get(it, {}).get("WAPE_mean")
        wb = items_b.get(it, {}).get("WAPE_mean")
        if wa is not None and wb is not None:
            delta = float(wb) - float(wa)
            sign = "+" if delta >= 0 else ""
            tag = "BETTER" if delta < 0 else "WORSE"
            print(f"    {it:20}: {float(wa):7.2f} -> {float(wb):7.2f}  "
                  f"({sign}{delta:.2f})  {tag}")
        elif wb is not None:
            print(f"    {it:20}: new={float(wb):.2f}")
        elif wa is not None:
            print(f"    {it:20}: removed (was {float(wa):.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=2,
                        help="Compare last N runs (default: 2)")
    args = parser.parse_args()

    runs = load_runs()
    if not runs:
        return

    for i, r in enumerate(runs, 1):
        r["_idx"] = i

    print_history(runs)

    n = min(args.last, len(runs))
    if n >= 2:
        subset = runs[-n:]
        for i in range(len(subset) - 1):
            print_delta(subset[i], subset[i + 1])
    elif len(runs) == 1:
        print("\nOnly one run recorded. Run main.py again after making changes to see a delta.")


if __name__ == "__main__":
    main()
