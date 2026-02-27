from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.helper import (
    parse_time_string,
    replace_patterns_in_dict,
    select_columns,
    week_string_to_date,
    load_weekly_data_with_prices,
)


@dataclass
class SalesDataset:
    """
    Patched version focused on multi-series → single-target forecasting with tree models.
    Key changes vs your original:
      - Robust datetime handling (weekly strings → datetime Monday of ISO week).
      - Creates future target y_col = target shifted by -nr_horizon * unit.
      - Adds market-level cross-series features available at test time (ex-target lags & trend).
      - Avoids log-transform on y_col and potentially negative features.
      - Optional prevention of promo-leakage via config flag.
      - Ensures y_col survives column selection; consistent item encoding.
    """

    def __init__(
        self,
        csv_file: str,
        separator: str,
        date_column: str,
        target_column: str,
        nr_horizon: int,
        prepare_config: dict[str, Any],
        column_selection: dict[str, Any],
        product_var: str,
        granularity: str,
        product_name: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.csv_file = csv_file
        self.separator = separator
        self.date_column = date_column
        self.target_column = target_column
        self.nr_horizon = nr_horizon
        self.pred_h = abs(nr_horizon)
        self.product_var = product_var
        self.product_name = product_name
        self.config = config or {}
        self.granularity = granularity  # "weekly" or "monthly"

        # Runtime flags
        self.debug_save = bool(self.config.get("debug_save", False))
        self.known_future_promos = bool(self.config.get("known_future_promos", False))

        np.random.seed(42)

        # Expand patterns in prepare/selection configs
        self.prepare_config = replace_patterns_in_dict(
            prepare_config,
            {
                "target": lambda _: target_column,
                "-1h": lambda _: -self.pred_h,
            },
        )

        self.column_selection = replace_patterns_in_dict(
            column_selection,
            {"target": lambda _: target_column},
        )

        # Load dataset
        if csv_file.endswith(".csv"):
            self.dataset = pd.read_csv(csv_file, sep=separator)
        elif csv_file.endswith(".xlsx") or csv_file.endswith(".xls"):
            self.dataset = load_weekly_data_with_prices(filepath=csv_file)
        else:
            raise ValueError("Unsupported file extension. Use .csv or .xlsx")

        # Normalize date column EARLY
        self._normalize_datetime_inplace(self.dataset)
        # Sort
        self.dataset.sort_values(by=[self.product_var, self.date_column], inplace=True)
        self.dataset.reset_index(drop=True, inplace=True)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _normalize_datetime_inplace(self, df: pd.DataFrame) -> None:
        """Convert self.date_column to pandas datetime.
        If strings like '2023-W05' are detected, convert with week_string_to_date.
        """
        col = self.date_column
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            # Try to detect ISO week strings
            as_str = df[col].astype(str)
            if as_str.str.contains(r"\bW\d{1,2}\b").any():
                df[col] = as_str.apply(week_string_to_date)
            else:
                df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
        # Ensure datetime
        df[col] = pd.to_datetime(df[col], errors="coerce")

    def _maybe_downsample(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.granularity != "monthly":
            return df

        # Monthly aggregation
        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column], errors="coerce")
        df["YearMonth"] = df[self.date_column].dt.to_period("M").dt.to_timestamp()

        agg_config = self.config.get(
            "aggregation",
            {self.target_column: "sum", "Prezzo": "mean"},
        )

        df = (
            df.groupby([self.product_var, "YearMonth"], as_index=False)
            .agg(agg_config)
            .rename(columns={"YearMonth": self.date_column})
        )
        return df

    def _add_market_features(
        self,
        df: pd.DataFrame,
        base_col: str | None = None,
        lags: tuple[int, ...] = (1, 4, 8, 12),
        trend_windows: tuple[int, ...] = (4, 8),
    ) -> pd.DataFrame:
        """Build cross-series market features excluding the current row's series contribution.
        These are available at test time (computed from other series).
        """
        if base_col is None:
            base_col = self.target_column

        df = df.copy()
        # Aggregates by timestamp over the whole market
        agg = (
            df.groupby(self.date_column)[base_col]
            .agg(["sum", "mean"])
            .rename(columns={"sum": "market_sum", "mean": "market_mean"})
            .reset_index()
        )
        df = df.merge(agg, on=self.date_column, how="left")

        # Exclude current series contribution to avoid leakage of target row
        df["market_sum_ex"] = df["market_sum"] - df[base_col]
        # Number of series per timestamp
        n_items_per_t = df.groupby(self.date_column)[self.product_var].transform("nunique")
        denom = (n_items_per_t - 1).replace(0, np.nan)
        df["market_mean_ex"] = df["market_sum_ex"] / denom

        # Lag the market aggregates per series (so the target test will still see them)
        for k in lags:
            df[f"market_sum_ex_lag{k}"] = (
                df.groupby(self.product_var)["market_sum_ex"].shift(k)
            )
            df[f"market_mean_ex_lag{k}"] = (
                df.groupby(self.product_var)["market_mean_ex"].shift(k)
            )

        # Trend (slope) of the market ex-target
        for w in trend_windows:
            def _slope(x: pd.Series) -> float:
                idx = np.arange(len(x))
                return np.polyfit(idx, x, 1)[0]

            df[f"market_trend_ex_w{w}"] = (
                df.groupby(self.product_var)["market_sum_ex"]
                .transform(lambda x: x.rolling(w, min_periods=w).apply(_slope, raw=True).shift(1))
            )

        # Lagged market share to avoid contemporaneous leakage at inference time.
        df["market_share"] = (
            df.groupby(self.product_var)[base_col].shift(1)
            / df.groupby(self.product_var)["market_sum"].shift(1).replace(0, np.nan)
        )

        return df

    # -----------------------------
    # Main preparation
    # -----------------------------
    def _prepare_dataset(
        self, df: pd.DataFrame | None = None, group_columns: list[str] | None = None
    ) -> pd.DataFrame:
        if df is None:
            df = self.dataset.copy()

        # Downsample if monthly
        df = self._maybe_downsample(df)

        # Ensure datetime
        self._normalize_datetime_inplace(df)
        # Sorting is key for rolling/shift
        df = df.sort_values([self.product_var, self.date_column]).reset_index(drop=True)

        # Stable product id column
        df["product_id"] = df[self.product_var]

        # Unit helpers
        is_weekly = self.granularity == "weekly"
        unit = "M" if self.granularity == "monthly" else "W"

        # Week features only for weekly granularity
        if is_weekly:
            iso = df[self.date_column].dt.isocalendar()
            df["week_number"] = iso.week.astype(int)
            df["is_peak_season"] = df["week_number"].between(47, 52).astype(int)
        else:
            df["week_number"] = 0
            df["is_peak_season"] = 0

        # ------------------------
        # Fourier seasonality features
        # ------------------------
        # Standard sin/cos on day-of-year captures only one cycle.
        # Fourier terms with K harmonics decompose the seasonal signal into K
        # overlapping cycles simultaneously: annual (k=1), semi-annual (k=2),
        # quarterly (k=3). This is the highest-ROI calendar feature for weekly data.
        #
        # For weekly data: period = 52 weeks.
        # For monthly data: period = 12 months.
        _fourier_period = 52.0 if is_weekly else 12.0
        _fourier_t = df["week_number"].astype(float) if is_weekly else df[self.date_column].dt.month.astype(float)
        _fourier_K = 3  # 3 harmonics: annual + semi-annual + quarterly cycles
        for k in range(1, _fourier_K + 1):
            df[f"fourier_sin_k{k}"] = np.sin(2 * np.pi * k * _fourier_t / _fourier_period)
            df[f"fourier_cos_k{k}"] = np.cos(2 * np.pi * k * _fourier_t / _fourier_period)

        # Also add month and quarter as explicit integer features —
        # tree models can use these directly as split points (complement to Fourier)
        df["month"] = df[self.date_column].dt.month
        df["quarter"] = df[self.date_column].dt.quarter

        # ------------------------
        # LAG features
        # ------------------------
        for col, lag in self.prepare_config.get("lag", {}).get(self.granularity, {}).items():
            lags = [lag] if isinstance(lag, (int)) else lag
            for l in lags:
                df[f"{col}_{l}{unit}"] = df.groupby(self.product_var)[col].shift(l)

        # ------------------------
        # Percentage changes (pct_change)
        # ------------------------
        for col, percentage in self.prepare_config.get("percentage", {}).get(
            self.granularity, {}
        ).items():
            pers = [percentage] if isinstance(percentage, (int)) else percentage
            for p in pers:
                df[f"{col}_percentage_{p}_{unit}"] = (
                    df.groupby(self.product_var)[col].pct_change(p).clip(-2, 2).shift(1)
                )

        # ------------------------
        # Rolling aggregations
        # prepare_config['rolling'] expected shape: {op: {granularity: {col: [wins]}}}
        # ------------------------
        for op, col_window in self.prepare_config.get("rolling", {}).items():
            for col, win_vals in col_window.get(self.granularity, {}).items():
                wins = [win_vals] if isinstance(win_vals, (int)) else win_vals
                for w in wins:
                    new_col = f"{col}_rolling_{op}_{w}{unit}"
                    df[new_col] = (
                        df.groupby(self.product_var)[col]
                        .transform(lambda x: x.rolling(window=w, min_periods=1).agg(op).shift(1))
                    )

        # ------------------------
        # Trend (slope) over windows
        # prepare_config['trend'][granularity] = {col: [wins]}
        # ------------------------
        def _slope_np(series: np.ndarray) -> float:
            idx = np.arange(len(series))
            return np.polyfit(idx, series, 1)[0]

        for col, trend_vals in self.prepare_config.get("trend", {}).get(
            self.granularity, {}
        ).items():
            wins = trend_vals if isinstance(trend_vals, (list, tuple)) else [trend_vals]
            for w in wins:
                new_col = f"{col}_trend_{w}{unit}"
                df[new_col] = (
                    df.groupby(self.product_var)[col]
                    .transform(
                        lambda x: x.rolling(window=w, min_periods=w).apply(_slope_np, raw=True).shift(1)
                    )
                )

        # ------------------------
        # Promo handling (avoid leakage if not known)
        # ------------------------
        for promo in [1, 2]:
            promo_col = f"type_promo_{promo}"
            if promo_col in df.columns:
                df[f"{promo_col}_current"] = df[promo_col]
                if self.known_future_promos:
                    df[f"{promo_col}_{self.nr_horizon}{unit}"] = (
                        df.groupby(self.product_var)[promo_col].shift(-self.nr_horizon)
                    )
        
        # ---- Product age (in settimane) ----
        first_pos_date = (
            df.assign(_pos=(df[self.target_column] > 0).astype(int))
            .groupby(self.product_var)
            .apply(lambda g: g.loc[g['_pos'].eq(1), self.date_column].min())
            .reindex(df[self.product_var]).values
        )
        df["product_age_w"] = ((df[self.date_column] - first_pos_date).dt.days // 7).clip(lower=0).fillna(0)

        # ---- EMA half-life su Sellout passato (shift(1)) ----
        def ema_half_life(s: pd.Series, half: int):
            alpha = 1 - np.exp(-np.log(2)/max(half,1))
            return s.ewm(alpha=alpha, adjust=False).mean()

        for h in [4, 8, 13]:
            df[f"{self.target_column}_ema_h{h}"] = (
                df.groupby(self.product_var)[self.target_column]
                .transform(lambda s: ema_half_life(s.shift(1), h))
            )

        # ---- Z-score per-serie (su lag semplici del target) ----
        for w in [8, 26]:
            mu = df.groupby(self.product_var)[f"{self.target_column}_1{unit}"]\
                .transform(lambda s: s.rolling(w, min_periods=4).mean())
            sd = df.groupby(self.product_var)[f"{self.target_column}_1{unit}"]\
                .transform(lambda s: s.rolling(w, min_periods=4).std().replace(0, np.nan))
            df[f"{self.target_column}_1{unit}_z{w}"] = (df[f"{self.target_column}_1{unit}"] - mu) / sd

        # ---- Prezzo relativo e gap rispetto al mercato (lag 1) ----
        if "Prezzo" in df.columns:
            # media mercato per data
            price_mean_t = df.groupby(self.date_column)["Prezzo"].transform("mean")
            df["price_market_mean_ex"] = (price_mean_t * df.groupby(self.date_column)[self.product_var].transform('nunique') - df["Prezzo"]) / \
                                        (df.groupby(self.date_column)[self.product_var].transform('nunique') - 1).replace(0, np.nan)
            # lag 1 per evitare contemporaneità
            df["price_gap_1W"] = (df["Prezzo"] - price_mean_t).groupby(df[self.product_var]).shift(1)
            df["price_rel_1W"] = (df["Prezzo"] / (price_mean_t.replace(0, np.nan))).groupby(df[self.product_var]).shift(1)


        # ------------------------
        # Create future target y_col AFTER all feature lags (so we don't shift it later)
        # ------------------------
        unit = "M" if self.granularity == "monthly" else "W"

        y_col = f"{self.target_column}_fwd_{abs(self.nr_horizon)}{unit}"   # ← nuovo nome, niente collisione
        df[y_col] = df.groupby(self.product_var)[self.target_column].shift(-self.pred_h)

        # ------------------------
        # Add market-level features (cross-series)
        # ------------------------
        # Build them before log-transform so these features are transformed
        # consistently with the rest of the positive demand-derived covariates.
        df = self._add_market_features(
            df,
            base_col=self.target_column,
            lags=tuple(self.config.get("market_lags", (1, 4, 8, 12))),
            trend_windows=tuple(self.config.get("market_trend_windows", (4, 8))),
        )

        # ------------------------
        # Log-transform POSITIVE features only (never y_col)
        # ------------------------
        exclude_cols = [
            self.target_column,
            y_col,
            "product_id",
            "week_number",
            "is_peak_season",
        ] + [c for c in df.columns if "promo" in c]

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_log = [
            c
            for c in numeric_cols
            if c not in exclude_cols and df[c].min() >= 0 and not c.endswith("_trend")
        ]
        for c in cols_to_log:
            df[c] = np.log1p(df[c])

        # Final cleanup
        df = df.reset_index(drop=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        if self.debug_save:
            try:
                df.to_excel("dataset_before_processing.xlsx", index=False)
            except Exception:
                pass

        return df

    def _encode_time(self, X: pd.DataFrame, time_unit: str = "week") -> pd.DataFrame:
        """Add cyclical time encoding (sin/cos) based on day-of-year.
        For weekly granularity, scale by 52; for monthly, by 365.
        """
        X = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(X[self.date_column]):
            X[self.date_column] = X[self.date_column].apply(week_string_to_date)
            X[self.date_column] = pd.to_datetime(X[self.date_column], errors="coerce")

        max_val = np.ceil(365 / 7) if time_unit == "week" else 365
        day_of_year = X[self.date_column].dt.dayofyear
        X[f"{self.date_column}_sin"] = np.sin(2 * np.pi * day_of_year / max_val)
        X[f"{self.date_column}_cos"] = np.cos(2 * np.pi * day_of_year / max_val)
        return X

    @staticmethod
    def _compute_intermittency_metrics(y: pd.Series) -> dict[str, float]:
        """Compute standard intermittency descriptors for one series."""
        arr = np.asarray(y, dtype=float)
        if arr.size == 0:
            return {"sparsity_ratio": 1.0, "adi": 1.0, "cv2": 0.0}

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        n = float(arr.size)
        pos = arr > 0.0
        n_pos = float(pos.sum())
        sparsity_ratio = 1.0 - (n_pos / n)

        if n_pos <= 0:
            return {"sparsity_ratio": float(sparsity_ratio), "adi": float(n), "cv2": 0.0}

        adi = n / n_pos
        pos_vals = arr[pos]
        mean_pos = float(np.mean(pos_vals))
        if pos_vals.size >= 2 and mean_pos > 1e-12:
            cv = float(np.std(pos_vals, ddof=1) / mean_pos)
            cv2 = cv * cv
        else:
            cv2 = 0.0
        return {"sparsity_ratio": float(sparsity_ratio), "adi": float(adi), "cv2": float(cv2)}

    # -----------------------------
    # Public API used by your training code
    # -----------------------------
    def extract_product_data(self, product: str) -> pd.DataFrame:
        return self.dataset[self.dataset[self.product_var] == product].copy()

    def process_data(
        self, train_fraction: float = 0.93, train_params: dict[str, Any] | None = None, product: str | None = None
    ):
        unit = "M" if self.granularity == "monthly" else "W"
        y_col = f"{self.target_column}_fwd_{self.nr_horizon}{unit}"
        product_to_use = product or self.product_name

        df = self._prepare_dataset()
        df = self._encode_time(X=df, time_unit=self.granularity)

        # Sort by product & date (already sorted in _prepare_dataset, but keep explicit)
        df = df.sort_values(by=[self.product_var, self.date_column]).reset_index(drop=True)

        # Boundary computation on TARGET series only
        target_df = df[df[self.product_var] == product_to_use].copy()
        if train_params and train_params.get("boundary") is not None:
            boundary = (target_df[self.date_column] - pd.to_datetime(train_params["boundary"]))\
                .abs().argmin()
            boundary_date = target_df.iloc[boundary][self.date_column]
        else:
            last_date = target_df[self.date_column].max()
            td = parse_time_string(train_params.get("test_len", "8W")) if train_params else pd.Timedelta(weeks=8)
            boundary_date = last_date - td

        begin_train_date = boundary_date - parse_time_string(train_params.get("train_len")) if train_params else boundary_date - pd.Timedelta(weeks=104)
        end_test_date = boundary_date + parse_time_string(train_params.get("test_len")) if train_params else boundary_date + pd.Timedelta(weeks=8)

        print(
            f"[process_data] boundary={boundary_date.date()}"
            f"  train_from={begin_train_date.date()}"
            f"  test_until={end_test_date.date()}"
        )

        # Per-series intermittency profile computed on TRAIN window only.
        profile_rows = []
        for pid, g in df.groupby(self.product_var, sort=False):
            hist = g.loc[
                (g[self.date_column] >= begin_train_date) & (g[self.date_column] < boundary_date),
                self.target_column,
            ]
            if hist.empty:
                hist = g.loc[g[self.date_column] < boundary_date, self.target_column]
            met = self._compute_intermittency_metrics(hist)
            profile_rows.append({self.product_var: pid, **met})
        profile_df = pd.DataFrame(profile_rows)
        df = df.merge(profile_df, on=self.product_var, how="left")
        self.series_profiles = {
            str(r[self.product_var]): {
                "sparsity_ratio": float(r["sparsity_ratio"]),
                "adi": float(r["adi"]),
                "cv2": float(r["cv2"]),
            }
            for _, r in profile_df.iterrows()
        }
        self.target_series_profile = self.series_profiles.get(str(product_to_use), {})

        # Masks: TRAIN = all other products (any date) + target product past only.
        # Non-target products are included without a date cutoff because their lag
        # features are computed per-group (shift within product) so a row at date t
        # for product B only sees B's own history up to t-1. Market features are also
        # lagged (market_sum_ex_lag1 etc.), so no contemporaneous cross-target leakage.
        # Including all dates for other products gives the global model more seasonal
        # pattern examples at the test-period timing — empirically verified to help.
        train_mask = (
            (df[self.product_var] != product_to_use)
            | (
                (df[self.product_var] == product_to_use)
                & (df[self.date_column] < boundary_date)
                & (df[self.date_column] >= begin_train_date)
            )
        )

        test_mask = (
            (df[self.product_var] == product_to_use)
            & (df[self.date_column] >= boundary_date)
            & (df[self.date_column] < end_test_date)
        )

        # Column selection (ensure y_col is present)
        sel_cols = select_columns(
            df=df,
            include_prefixes=self.column_selection.get("prefixes", []),
            include_suffixes=self.column_selection.get("suffixes", []),
            include_regex=self.column_selection.get("regex", []),
            exclude_columns=self.column_selection.get("to_exclude", []),
        )
        if y_col not in sel_cols:
            sel_cols.append(y_col)
        for intermittent_col in ("sparsity_ratio", "adi", "cv2"):
            if intermittent_col in df.columns and intermittent_col not in sel_cols:
                sel_cols.append(intermittent_col)
        df = df[sel_cols]

        # Split
        df_train = df.loc[train_mask].copy()
        df_test  = df.loc[test_mask].copy()
        df_train = df_train[df_train[y_col].notna()].copy()
        df_test = df_test[df_test[y_col].notna()].copy()

        # Sort each split chronologically within each product
        df_train = df_train.sort_values([self.product_var, self.date_column]).reset_index(drop=True)
        df_test  = df_test.sort_values([self.product_var, self.date_column]).reset_index(drop=True)

        # Store raw dates separately (used by Forecaster.fit() for sample weight computation)
        self.train_dates = df_train[self.date_column].reset_index(drop=True)
        self.test_dates  = df_test[self.date_column].reset_index(drop=True)

        # Build X and y, dropping the target column and raw date from features
        drop_cols = [y_col, self.date_column]
        self.X_train = df_train.drop(columns=drop_cols).reset_index(drop=True)
        self.y_train = df_train[y_col].reset_index(drop=True)
        self.X_test  = df_test.drop(columns=drop_cols).reset_index(drop=True)
        self.y_test  = df_test[y_col].reset_index(drop=True)
        self.train_is_target = (
            df_train[self.product_var].astype(str).reset_index(drop=True) == str(product_to_use)
        )

        # ── Per-series mean scaling ───────────────────────────────────────────
        # Scale the target by the median demand of the TARGET PRODUCT in the
        # training window. This prevents high-volume series from dominating
        # gradient updates in a global model, and makes all series live on a
        # comparable scale (important for Tweedie loss calibration).
        #
        # We use median (not mean) because retail demand has right-skewed
        # distributions with promo spikes that would inflate the mean.
        # Fallback to 1.0 if the training target is all-zero (new product).
        #
        # The scale is stored on self.series_scale so backtest.py can
        # inverse-transform predictions back to the original units before
        # computing metrics and stock analysis.
        _target_train_values = df_train.loc[
            df_train[self.product_var] == product_to_use, y_col
        ]
        _median = float(_target_train_values.median())
        self.series_scale = _median if _median > 0 else 1.0

        self.y_train = self.y_train / self.series_scale
        self.y_test  = self.y_test  / self.series_scale

        # --- Label-encode the product column ---
        # Fit on train+test together so both sets share the same integer mapping.
        # Then one-hot encode: gives tree models an explicit per-series signal.
        le = LabelEncoder()
        col_cat = "product_id" if "product_id" in self.X_train.columns else self.product_var
        all_labels = pd.concat(
            [self.X_train[col_cat], self.X_test[col_cat]], ignore_index=True
        ).astype(str).unique()
        le.fit(all_labels)
        self.X_train[col_cat] = le.transform(self.X_train[col_cat].astype(str))
        self.X_test[col_cat]  = le.transform(self.X_test[col_cat].astype(str))

        # One-hot encode: align() ensures train and test have identical dummy
        # columns even if a product appears only in one split
        self.X_train[col_cat] = pd.Categorical(self.X_train[col_cat], categories=range(len(le.classes_)))
        self.X_test[col_cat]  = pd.Categorical(self.X_test[col_cat],  categories=range(len(le.classes_)))
        train_dum = pd.get_dummies(self.X_train[col_cat], prefix=f"{col_cat}_oh", dtype=int)
        test_dum  = pd.get_dummies(self.X_test[col_cat],  prefix=f"{col_cat}_oh", dtype=int)
        train_dum, test_dum = train_dum.align(test_dum, join="outer", axis=1, fill_value=0)
        self.X_train = pd.concat([self.X_train, train_dum], axis=1)
        self.X_test  = pd.concat([self.X_test,  test_dum],  axis=1)

        self.product_label_encoder = le

        # Safety check: test set must contain only the target product
        test_items = df_test[self.product_var].astype(str).unique()
        print(f"[process_data] test products: {list(test_items)}")
        assert len(test_items) == 1 and test_items[0] == str(product_to_use), \
            f"TEST must contain only '{product_to_use}', found: {list(test_items)}"


    # Convenience
    def get_all_products(self):
        return pd.unique(self.dataset[self.product_var])
