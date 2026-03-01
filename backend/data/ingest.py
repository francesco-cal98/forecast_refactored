"""
backend/data/ingest.py
======================
Universal ingestion layer.  Reads ANY sales file and normalises it into the
canonical long-panel DataFrame that SalesDataset expects:

    series_id | date | sales_units [| any extra numeric feature cols]

Supported input structures
--------------------------
1. Wide cross-tab  (rows = series, columns = time periods)
   - Monthly:  column headers are month names (Jan, Feb, … Dec)
   - Weekly:   column headers are week labels (W1, W2, … W52)
   Both variants: columns may span multiple years (auto-detected by wrap-around).

2. Long / panel format  (rows = observations)
   Column roles (date, series, target) are inferred from column names and dtypes.
   Extra numeric columns are preserved as features.

3. Multi-sheet Excel with >3 sheets -> raise; caller should use the dedicated loader
   (e.g. praenova's load_area_panel).

Entry point
-----------
    from data.ingest import load_canonical
    df = load_canonical("data.xlsx", config)

`config` is the dataset_config dict from Hydra (or any plain dict). Keys used:
    base_year        int   first year for wide cross-tab year assignment (default 2023)
    date_column      str   canonical name for the date column (default "date")
    target_column    str   canonical name for the target column (default "sales_units")
    product_column   str   canonical name for the series-id column (default "series_id")
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

_MONTH_MAP: dict[str, int] = {
    # English
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
    # Italian
    "gen": 1, "gennaio": 1,
    "feb": 2, "febbraio": 2,
    "mar": 3, "marzo": 3,
    "apr": 4, "aprile": 4,
    "mag": 5, "maggio": 5,
    "giu": 6, "giugno": 6,
    "lug": 7, "luglio": 7,
    "ago": 8, "agosto": 8,
    "set": 9, "settembre": 9,
    "ott": 10, "ottobre": 10,
    "nov": 11, "novembre": 11,
    "dic": 12, "dicembre": 12,
}

_WEEK_RE = re.compile(r"^[ws]\s*0*(\d{1,2})$", re.IGNORECASE)

# Regexes for column-role detection in long-panel files
_DATE_NAME_RE = re.compile(
    r"^(date|data|period|week|month|year|time|dt|timestamp|settimana|periodo)$",
    re.IGNORECASE,
)
_SERIES_NAME_RE = re.compile(
    r"(product|serie|series|item|sku|ean|article|articolo|codice|cod|brand|"
    r"category|categor|area|region|canale|channel)",
    re.IGNORECASE,
)
_TARGET_NAME_RE = re.compile(
    r"(sales|units|qty|quantity|volume|demand|revenue|fatturato|"
    r"vendite|quantit[aà]|sell.?out|sell.?in)",
    re.IGNORECASE,
)


def _month_num(val: Any) -> int:
    """Return 1-12 if val is a month name (any language), else 0."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    return _MONTH_MAP.get(str(val).strip().lower(), 0)


def _week_num(val: Any) -> int:
    """Return 1-53 if val looks like W1 / S1 / w01, else 0."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    m = _WEEK_RE.match(str(val).strip())
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _find_time_header_row(raw: pd.DataFrame) -> tuple[int, str]:
    """
    Scan every row of `raw` (read without headers) looking for a row whose
    non-null values are predominantly month names or week labels.

    Returns (row_index, "monthly" | "weekly") or (-1, "unknown").
    """
    for idx in raw.index:
        row = raw.loc[idx]
        non_null = [v for v in row if pd.notna(v) and str(v).strip()]
        if len(non_null) < 6:
            continue
        month_hits = sum(1 for v in non_null if _month_num(v) > 0)
        week_hits = sum(1 for v in non_null if _week_num(v) > 0)
        if month_hits >= 6:
            return idx, "monthly"
        if week_hits >= 6:
            return idx, "weekly"
    return -1, "unknown"


def _looks_like_long_panel(path: Path) -> bool:
    """
    Returns True when the file, read with header=0, has named columns
    (< 40 % 'Unnamed') and at least one column that looks like a date.
    """
    try:
        df = (
            pd.read_excel(path, nrows=20)
            if path.suffix in (".xlsx", ".xls")
            else pd.read_csv(path, nrows=20)
        )
    except Exception:
        return False

    unnamed_frac = sum(
        1 for c in df.columns if "unnamed" in str(c).lower()
    ) / max(len(df.columns), 1)
    if unnamed_frac > 0.4:
        return False

    for col in df.columns:
        if _DATE_NAME_RE.search(str(col)):
            return True
        try:
            pd.to_datetime(df[col].dropna().head(5), errors="raise")
            return True
        except Exception:
            pass
    return False


def detect_format(raw: pd.DataFrame, path: Path) -> tuple[str, str]:
    """
    Returns (format_type, granularity).

    format_type : "wide_crosstab" | "long_panel" | "multisheet"
    granularity : "monthly" | "weekly" | "unknown"
    """
    # Multi-sheet Excel -> caller must use a dedicated loader
    if path.suffix in (".xlsx", ".xls"):
        try:
            sheet_names = pd.ExcelFile(path).sheet_names
            if len(sheet_names) > 3:
                return "multisheet", "unknown"
        except Exception:
            pass

    # Wide cross-tab detection
    header_idx, gran = _find_time_header_row(raw)
    if header_idx >= 0:
        return "wide_crosstab", gran

    # Long panel fallback
    if _looks_like_long_panel(path):
        return "long_panel", "unknown"

    return "long_panel", "unknown"  # best-effort fallback


# ---------------------------------------------------------------------------
# WideCrosstabIngestor
# ---------------------------------------------------------------------------

class WideCrosstabIngestor:
    """
    Converts a pivot-style file (rows = series, columns = time periods) into a
    canonical long-panel DataFrame.

    Handles:
    - Irregular header rows above the time-header row (section labels, empty rows)
    - Month or week column headers
    - Multi-year data (year wrap-around detection)
    - Duplicate series names -> appended _1, _2, … suffix
    - Trailing / leading whitespace in labels
    """

    def ingest(self, raw: pd.DataFrame, granularity: str, config: dict) -> pd.DataFrame:
        base_year: int = int(config.get("base_year", 2023))
        date_col: str = config.get("date_column", "date")
        series_col: str = config.get("product_column", "series_id")
        target_col: str = config.get("target_column", "sales_units")

        # ── Step 1: find time header row ──────────────────────────────────────
        header_idx, detected_gran = _find_time_header_row(raw)
        if header_idx < 0:
            raise ValueError(
                "WideCrosstabIngestor: cannot find a row with ≥6 time-period labels "
                "(month names or week identifiers). Check that the file matches the "
                "expected wide cross-tab format."
            )
        granularity = granularity or detected_gran

        time_header = raw.loc[header_idx]

        # ── Step 2: build column-index -> date mapping ─────────────────────────
        time_col_map: dict[Any, pd.Timestamp] = {}

        if granularity == "monthly":
            year = base_year
            prev_month = 0
            for col_idx, val in time_header.items():
                m = _month_num(val)
                if m == 0:
                    continue
                if m < prev_month:       # month wrapped around -> next year
                    year += 1
                time_col_map[col_idx] = pd.Timestamp(year=year, month=m, day=1)
                prev_month = m

        elif granularity == "weekly":
            year = base_year
            prev_week = 0
            for col_idx, val in time_header.items():
                w = _week_num(val)
                if w == 0:
                    continue
                if w < prev_week:        # week wrapped around -> next year
                    year += 1
                # ISO week -> Monday of that week
                ts = pd.Timestamp.fromisocalendar(year, w, 1)
                time_col_map[col_idx] = ts
                prev_week = w

        else:
            raise ValueError(f"Unsupported granularity: {granularity!r}")

        if not time_col_map:
            raise ValueError("WideCrosstabIngestor: no valid time-period columns found.")

        # ── Step 3: identify data rows (strictly after the header row) ────────
        data_rows_idx = [i for i in raw.index if i > header_idx]
        if not data_rows_idx:
            raise ValueError("WideCrosstabIngestor: no data rows found after the time header.")

        # ── Step 4: identify the label column ────────────────────────────────
        # The label column is the leftmost non-null string column in data rows
        # that is NOT in time_col_map.
        time_col_keys = set(time_col_map.keys())
        label_col_idx = None
        for col_idx in raw.columns:
            if col_idx in time_col_keys:
                continue
            col_data = raw.loc[data_rows_idx, col_idx].dropna()
            if len(col_data) == 0:
                continue
            if col_data.apply(lambda v: isinstance(v, str)).mean() >= 0.5:
                label_col_idx = col_idx
                break  # take the first (leftmost) candidate

        if label_col_idx is None:
            raise ValueError(
                "WideCrosstabIngestor: cannot find a column with series labels. "
                "Expected a text column to the left of the time-period columns."
            )

        # ── Step 5: detect and skip section-header rows ───────────────────────
        # A section-header row has exactly ONE non-null value and it falls outside
        # the time columns (i.e. it is a label-column value with no numeric data).
        def _is_section_header(row_idx: int) -> bool:
            row = raw.loc[row_idx]
            non_null = [(c, v) for c, v in row.items() if pd.notna(v) and str(v).strip()]
            if len(non_null) != 1:
                return False
            sole_col, _ = non_null[0]
            return sole_col not in time_col_keys

        # ── Step 6: count series-name occurrences for deduplication ───────────
        name_counts: dict[str, int] = {}
        for row_idx in data_rows_idx:
            if _is_section_header(row_idx):
                continue
            raw_label = raw.at[row_idx, label_col_idx]
            if pd.isna(raw_label):
                continue
            label = str(raw_label).strip()
            if label:
                name_counts[label] = name_counts.get(label, 0) + 1

        # ── Step 7: build canonical records ────────────────────────────────────
        records: list[dict] = []
        name_seen: dict[str, int] = {}

        for row_idx in data_rows_idx:
            if _is_section_header(row_idx):
                continue
            row = raw.loc[row_idx]

            raw_label = row[label_col_idx]
            if pd.isna(raw_label):
                continue
            label = str(raw_label).strip()
            if not label:
                continue

            name_seen[label] = name_seen.get(label, 0) + 1
            if name_counts[label] > 1:
                series_id = f"{label}_{name_seen[label]}"
            else:
                series_id = label

            for col_idx, timestamp in time_col_map.items():
                val = row[col_idx]
                try:
                    numeric_val = float(val)
                except (TypeError, ValueError):
                    continue  # skip non-numeric cells silently
                records.append(
                    {
                        series_col: series_id,
                        date_col: timestamp,
                        target_col: numeric_val,
                    }
                )

        if not records:
            raise ValueError("WideCrosstabIngestor: no valid data records extracted.")

        df = pd.DataFrame(records)
        df = df.sort_values([series_col, date_col]).reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# LongPanelIngestor
# ---------------------------------------------------------------------------

class LongPanelIngestor:
    """
    Reads a file that already has rows = observations (long / panel format).
    Auto-detects the date, series-id, and target columns by name heuristics
    and dtype inspection, then renames them to the canonical schema.
    Extra numeric columns are preserved unchanged.
    """

    def ingest(self, path: Path, config: dict) -> pd.DataFrame:
        date_col: str = config.get("date_column", "date")
        series_col: str = config.get("product_column", "series_id")
        target_col: str = config.get("target_column", "sales_units")

        # Read the file with proper header
        if path.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            sep = config.get("separator", ",")
            df = pd.read_csv(path, sep=sep)

        orig_cols = list(df.columns)

        # ── Identify date column ──────────────────────────────────────────────
        date_src = self._find_date_col(df)

        # ── Identify series / product column ─────────────────────────────────
        series_src = self._find_series_col(df, exclude=[date_src])

        # ── Identify target column ───────────────────────────────────────────
        target_src = self._find_target_col(df, exclude=[c for c in [date_src, series_src] if c])

        # ── Rename to canonical names ─────────────────────────────────────────
        rename_map: dict[str, str] = {}
        if date_src and date_src != date_col:
            rename_map[date_src] = date_col
        if series_src and series_src != series_col:
            rename_map[series_src] = series_col
        if target_src and target_src != target_col:
            rename_map[target_src] = target_col

        df = df.rename(columns=rename_map)

        # Warn if any role was not found
        for role, col in [("date", date_col), ("series", series_col), ("target", target_col)]:
            if col not in df.columns:
                warnings.warn(
                    f"LongPanelIngestor: could not auto-detect the {role} column. "
                    f"Set 'date_column', 'product_column', or 'target_column' in the config "
                    f"to match your actual column name.",
                    stacklevel=2,
                )

        # ── Parse date ────────────────────────────────────────────────────────
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        # Sort chronologically per series
        sort_cols = [c for c in [series_col, date_col] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)

        _report_inference(orig_cols, date_src, series_src, target_src, rename_map)
        return df

    # -- helpers --

    @staticmethod
    def _find_date_col(df: pd.DataFrame) -> str | None:
        # 1) Name match
        for col in df.columns:
            if _DATE_NAME_RE.search(str(col)):
                return col
        # 2) Datetime-parseable values
        for col in df.columns:
            try:
                pd.to_datetime(df[col].dropna().head(10), errors="raise")
                return col
            except Exception:
                pass
        return None

    @staticmethod
    def _find_series_col(df: pd.DataFrame, exclude: list[str | None]) -> str | None:
        excl = {c for c in exclude if c}
        # 1) Name match
        for col in df.columns:
            if col in excl:
                continue
            if _SERIES_NAME_RE.search(str(col)):
                return col
        # 2) Object/string dtype with low cardinality
        for col in df.columns:
            if col in excl:
                continue
            if df[col].dtype == object:
                cardinality = df[col].nunique()
                if 1 < cardinality < max(10, len(df) * 0.05):
                    return col
        return None

    @staticmethod
    def _find_target_col(df: pd.DataFrame, exclude: list[str | None]) -> str | None:
        excl = {c for c in exclude if c}
        # 1) Name match
        for col in df.columns:
            if col in excl:
                continue
            if _TARGET_NAME_RE.search(str(col)):
                return col
        # 2) Highest-std numeric column
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c not in excl
        ]
        if numeric_cols:
            stds = df[numeric_cols].std()
            return str(stds.idxmax())
        return None


def _report_inference(
    orig_cols: list[str],
    date_src: str | None,
    series_src: str | None,
    target_src: str | None,
    rename_map: dict[str, str],
) -> None:
    parts = []
    for src, role in [(date_src, "date"), (series_src, "series_id"), (target_src, "sales_units")]:
        if src:
            dest = rename_map.get(src, src)
            parts.append(f"  {role:<12} ← '{src}'" + (f" -> '{dest}'" if src != dest else ""))
        else:
            parts.append(f"  {role:<12} ← NOT FOUND (check config)")
    print("[ingest] LongPanelIngestor column mapping:")
    print("\n".join(parts))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_canonical(path: str | Path, config: dict) -> pd.DataFrame:
    """
    Universal file reader.  Always returns a canonical long-panel DataFrame.

    Parameters
    ----------
    path   : path to the data file (.xlsx, .xls, .csv)
    config : dataset_config dict from Hydra (or any plain dict).
             Recognised keys: base_year, date_column, product_column,
             target_column, separator.

    Returns
    -------
    pd.DataFrame with at minimum: date_column, product_column, target_column
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Read raw (no header) so we can inspect the structure freely
    if path.suffix in (".xlsx", ".xls"):
        raw = pd.read_excel(path, header=None, sheet_name=0)
    else:
        sep = config.get("separator", ",")
        raw = pd.read_csv(path, header=None, sep=sep)

    fmt, granularity = detect_format(raw, path)
    print(f"[ingest] Detected format: {fmt!r}, granularity: {granularity!r}")

    if fmt == "multisheet":
        raise ValueError(
            f"File {path.name} has >3 sheets. Use a dedicated multi-sheet loader "
            "(e.g. loader: praenova) instead of loader: auto."
        )

    if fmt == "wide_crosstab":
        # granularity from config overrides detection if explicitly set
        cfg_gran = config.get("granularity", granularity)
        df = WideCrosstabIngestor().ingest(raw, granularity=cfg_gran, config=config)
        print(
            f"[ingest] Wide cross-tab -> {df[config.get('product_column','series_id')].nunique()} series, "
            f"{len(df)} rows"
        )
        return df

    # long_panel (default)
    df = LongPanelIngestor().ingest(path, config)
    print(f"[ingest] Long panel -> {len(df)} rows, columns: {list(df.columns)}")
    return df
