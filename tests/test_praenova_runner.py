from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_runner_creates_expected_artifacts(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    input_fp = repo_root / "praenova_validation_2y.xlsx"
    output_dir = tmp_path / "eda_out"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.prototype.praenova.run_eda",
            "--input",
            str(input_fp),
            "--output",
            str(output_dir),
            "--base-year",
            "2023",
        ],
        check=True,
        cwd=str(repo_root),
        text=True,
    )

    expected = [
        output_dir / "series_summary.csv",
        output_dir / "quality_report.json",
        output_dir / "dashboard_trend.png",
        output_dir / "dashboard_seasonality.png",
        output_dir / "dashboard_promo_impact.png",
        output_dir / "dashboard_price_vs_sales.png",
    ]

    for fp in expected:
        assert fp.exists()
        assert fp.stat().st_size > 0

    summary = pd.read_csv(output_dir / "series_summary.csv")
    assert len(summary) == 15
