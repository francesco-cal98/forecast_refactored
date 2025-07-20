from dataclasses import dataclass
import json
import re
from typing import Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import SplineTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder



from utils.helper import (
    parse_time_string,
    replace_patterns_in_dict,
    select_columns,
    week_string_to_date,
    load_weekly_data_with_prices,
    normalize_selected_columns
)

@dataclass
class SalesDataset:
    def __init__(self, 
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
                 config: dict[str, Any] | None = None):

        self.csv_file = csv_file
        self.separator = separator
        self.date_column = date_column
        self.target_column = target_column
        self.nr_horizon = nr_horizon
        self.product_var = product_var
        self.product_name = product_name
        self.config = config or {}
        self.granularity = granularity

        np.random.seed(42)

        self.prepare_config = replace_patterns_in_dict(
            prepare_config, 
            {"target": lambda x: target_column,
             "-1h": lambda x: int(re.search(r'-?\d+', x).group()) * nr_horizon}
        )

        self.column_selection = replace_patterns_in_dict(
            column_selection,
            {"target": lambda x: target_column}
        )

        if csv_file.endswith('.csv'):
            self.dataset = pd.read_csv(csv_file, sep=separator)
        elif csv_file.endswith('.xlsx'):
            self.dataset = load_weekly_data_with_prices(filepath=csv_file)

        if self.dataset[self.date_column].astype(str).str.contains("W").all():
            self.dataset[target_column] = (
                self.dataset[target_column].astype(str).str.replace(",", "").astype(float)
            )
        else:
            self.dataset[date_column] = pd.to_datetime(self.dataset[date_column], format='%Y-%m-%d', errors='coerce')
            self.dataset.sort_values(by=date_column, inplace=True)
            self.dataset.reset_index(drop=True, inplace=True)

    def _maybe_downsample(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.granularity != "monthly":
            return df

        print("📉 Downsampling mensile attivato: aggregazione per mese.")

        df[self.date_column] = pd.to_datetime(df[self.date_column], errors='coerce')
        df["YearMonth"] = df[self.date_column].dt.to_period("M").dt.to_timestamp()

        agg_config = self.config.get("aggregation", {
            self.target_column: "sum",
            "Prezzo": "mean"
        })

        df = df.groupby([self.product_var, "YearMonth"], as_index=False).agg(agg_config)

        df[self.date_column] = df["YearMonth"]
        df.drop(columns=["YearMonth"], inplace=True)
        return df

    def _prepare_dataset(self, df: pd.DataFrame | None = None, group_columns: list[str] = ['product_group']) -> pd.DataFrame:
        if df is None:
            df = self.dataset.copy()

        df = self._maybe_downsample(df)
        df["product_id"] = df[self.product_var]

        unit = "M" if self.granularity == "monthly" else "W"

        df["week_number"] = df[self.date_column].str.extract(r"W(\d{1,2})")[0].astype(int)
        df["is_peak_season"] = df["week_number"].between(47, 52).astype(int)

        for col, lag in self.prepare_config.get('lag', {}).get(self.granularity, {}).items():
            if isinstance(lag, int): lag = [lag]
            for l in lag:
                df[f'{col}_{l}{unit}'] = df.groupby(self.product_var)[col].shift(l)

        for col, percentage in self.prepare_config.get('percentage', {}).get(self.granularity, {}).items():
            if isinstance(percentage, int): percentage = [percentage]
            for p in percentage:
                df[f'{col}_percentage_{p}_{unit}'] = df.groupby(self.product_var)[col].pct_change(p).clip(-2, 2).shift(1)

            for op, col_window in self.prepare_config.get('rolling', {}).items():
                for col, win_vals in col_window.get(self.granularity, {}).items():
                    if isinstance(win_vals, int): win_vals = [win_vals]
                    for w in win_vals:
                        new_col = f"{col}_rolling_{op}_{w}{unit}"
                        df[new_col] = (
                            df
                            .groupby(self.product_var)[col]
                            .transform(lambda x: x.rolling(window=w, min_periods=1).agg(op).shift(1))
                        )

            # Trend
            for col, trend_vals in self.prepare_config.get('trend', {}).get(self.granularity, {}).items():
                if isinstance(trend_vals, int): trend_vals = [trend_vals]
                for w in trend_vals:
                    new_col = f"{col}_trend_{w}{unit}"
                    df[new_col] = (
                        df
                        .groupby(self.product_var)[col]
                        .transform(lambda x: x
                                .rolling(window=w, min_periods=w)
                                .apply(lambda s: np.polyfit(np.arange(len(s)), s, 1)[0], raw=True)
                                .shift(1))
                    )

        for promo in [1, 2]:
            promo_col = f'type_promo_{promo}'
            if promo_col in df.columns:
                df[f'type_promo_{promo}_current'] = df[promo_col]
                df[f'type_promo_{promo}_{self.nr_horizon}{unit}'] = df.groupby(self.product_var)[promo_col].shift(-self.nr_horizon)

        # ——————————————————————————————————
        # 3) Log‐transform delle feature numeriche positive
        exclude_cols = [
            self.target_column,
            "product_id",
            "week_number",
            "is_peak_season"
        ] + [c for c in df.columns if "promo" in c]

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_log = [
            c for c in numeric_cols
            if c not in exclude_cols
               and df[c].min() >= 0
        ]
        for c in cols_to_log:
            df[c] = np.log1p(df[c])

        # 4) Rimuovi i NaN prodotti da tutte le trasformazioni
        df.dropna(inplace=True)
        df = df.reset_index(drop=True)

        return df


    def _encode_time(self, X: pd.DataFrame, time_unit='week'):
        if not pd.api.types.is_datetime64_any_dtype(X[self.date_column]):
            X[self.date_column] = X[self.date_column].apply(week_string_to_date)
            X[self.date_column] = pd.to_datetime(X[self.date_column], errors='coerce')

        if time_unit == 'week':
            max_val = np.ceil(365 / 7)
        else:
            max_val = 365

        day_of_year = X[self.date_column].dt.dayofyear

        X[f"{self.date_column}_sin"] = np.sin(2 * np.pi * day_of_year / max_val)
        X[f"{self.date_column}_cos"] = np.cos(2 * np.pi * day_of_year / max_val)

        return X

    def extract_product_data(self, product: str) -> pd.DataFrame:
        return self.dataset[self.dataset[self.product_var] == product].copy()

    def process_data(self, train_fraction=0.93, train_params: dict[str, Any] = None, product: str | None = None):
        y_col = f'{self.target_column}_{self.nr_horizon}W'
        product_to_use = product or self.product_name
        df = self._prepare_dataset()  # Passa tutti i prodotti
        df = self._encode_time(X=df, time_unit=self.granularity)
        
        # Ordina i prodotti mettendo il target product_to_use per ultimo
        #products = df[self.product_var].unique().tolist()

        # Rimuovi e metti in fondo
        #products = [p for p in products if p != product_to_use] + [product_to_use]

        # Crea una categoria ordinata
        #df[self.product_var] = pd.Categorical(df[self.product_var], categories=products, ordered=True)

        # Ora ordina per prodotto e data
        #df = df.sort_values(by=[self.product_var, self.date_column]).reset_index(drop=True)
        
        if train_params['boundary'] is not None:
            boundary = (df[self.date_column] - pd.to_datetime(train_params['boundary'])).abs().argmin()
            boundary_date = df.iloc[boundary][self.date_column]
        else:
            last_date = df[self.date_column].max()
            boundary_date = last_date - parse_time_string(train_params['test_len'])
            boundary = (df[self.date_column] - boundary_date).abs().argmin()

        begin_train_date = boundary_date - parse_time_string(train_params['train_len'])
        end_test_date = boundary_date + parse_time_string(train_params['test_len'])

        print("Boundary date:", boundary_date)
        print("End test date:", end_test_date)
        print("Unique dates in df:", sorted(df[self.date_column].unique())[:5], "...", sorted(df[self.date_column].unique())[-5:])

        sel_cols = select_columns(
            df=df,
            include_prefixes=self.column_selection['prefixes'],
            include_suffixes=self.column_selection['suffixes'],
            include_regex=self.column_selection['regex'],
            exclude_columns=self.column_selection['to_exclude']
        )

        train_mask = (
            ((df[self.product_var] == product_to_use) & (df[self.date_column] < boundary_date)) | 
            (df[self.product_var] != product_to_use)
        )

        test_mask = (df[self.product_var] == product_to_use) & (df[self.date_column] >= boundary_date) & (df[self.date_column] < end_test_date)

        self.X_train = df.loc[train_mask].copy()
        self.X_test = df.loc[test_mask].copy()
        self.y_train = self.X_train[y_col]
        self.y_test = self.X_test[y_col]

        print("Train dates")
        print(self.X_train[self.date_column].head(3))
        print(self.y_train.head(3))

        print("\nTest dates")
        print(self.X_test[self.date_column].head(3))
        print(self.y_test.head(3))

        df = df[sel_cols]

        self.X_train = df.loc[train_mask].copy().reset_index(drop=True)
        self.X_test = df.loc[test_mask].copy().reset_index(drop=True)

        self.y_train = self.X_train[y_col].reset_index(drop=True)
        self.X_train.drop(y_col, axis=1, inplace=True)
        self.y_test = self.X_test[y_col].reset_index(drop=True)
        self.X_test.drop(y_col, axis=1, inplace=True)
        

        # Assumendo che il dataset sia già processato:
        le = LabelEncoder()

        # Codifica il campo product_id
        self.X_train["product_id"] = le.fit_transform(self.X_train["product_id"])
        self.X_test["product_id"] = le.transform(self.X_test["product_id"])


        print(f"🗕️ Train: {begin_train_date.date()} → {boundary_date.date()} → {len(self.X_train)} rows")
        print(f"🗕️ Test : {boundary_date.date()} → {end_test_date.date()} → {len(self.X_test)} rows")

    def get_all_products(self):
        return pd.unique(self.dataset[self.product_var])

