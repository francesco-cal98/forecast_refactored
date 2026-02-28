from pathlib import Path

from backend.prototype.praenova.io import load_area_panel
from backend.prototype.praenova.quality import run_quality_checks


def _dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "praenova_validation_2y.xlsx"


def test_quality_report_has_expected_structure():
    df = load_area_panel(str(_dataset_path()))
    report = run_quality_checks(df)

    assert report["duplicate_key_count"] == 0
    assert report["series_count"] == 15
    assert len(report["continuity_by_series"]) == 15
    assert "value_range_flags" in report
    assert "promo_consistency_flags" in report


def test_quality_flags_invalid_week_numbers():
    df = load_area_panel(str(_dataset_path()))
    df_bad = df.copy()
    df_bad.loc[df_bad.index[0], "week_num"] = 60
    report = run_quality_checks(df_bad)

    assert report["value_range_flags"]["invalid_week_num_count"] == 1
