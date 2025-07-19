from dataclasses import dataclass
import json
import re
from typing import Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import SplineTransformer
from sklearn.preprocessing import StandardScaler


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

          # Load JSON configs
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


        unit = "M" if self.granularity == "monthly" else "W"

        for col, lag in self.prepare_config.get('lag', {}).get(self.granularity, {}).items():
            if isinstance(lag, int): lag = [lag]
            for l in lag:
                df[f'{col}_{l}{unit}'] = df.groupby(group_columns)[col].shift(l)

        for col, percentage in self.prepare_config.get('percentage', {}).get(self.granularity, {}).items():
            if isinstance(percentage, int): percentage = [percentage]
            for p in percentage:
                df[f'{col}_percentage_{p}_{unit}'] = df.groupby(group_columns)[col].pct_change(p).clip(-2, 2).shift(1)

        for op, col_window in self.prepare_config.get('rolling', {}).items():
            for col, w in col_window.get(self.granularity, {}).items():
                if isinstance(w, int): w = [w]
                for wi in w:
                    for g in df[group_columns].drop_duplicates().values:
                        mask = (df[group_columns] == g).all(axis=1)
                        df.loc[mask, f'{col}_rolling_{op}_{wi}{unit}'] = df.loc[mask, col].rolling(window=wi, min_periods=1).agg(op).shift(1)

        for col, windows in self.prepare_config.get('trend', {}).get(self.granularity, {}).items():
            if isinstance(windows, int): windows = [windows]
            for wi in windows:
                for g in df[group_columns].drop_duplicates().values:
                    mask = (df[group_columns] == g).all(axis=1)
                    df.loc[mask, f"{col}_trend_{wi}{unit}"] = df.loc[mask, col].rolling(window=wi, min_periods=wi).apply(
                        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
                    ).shift(1)

        for promo in [1, 2]:
            promo_col = f'type_promo_{promo}'
            if promo_col in df.columns:
                df[f'type_promo_{promo}_current'] = df[promo_col]
                df[f'type_promo_{promo}_{self.nr_horizon}{unit}'] = df.groupby(group_columns)[promo_col].shift(-self.nr_horizon)

        df.dropna(inplace=True)
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
        df = self._prepare_dataset() if product_to_use is None else self._prepare_dataset(df=self.extract_product_data(product_to_use), group_columns=[self.product_var])
        df = self._encode_time(X=df, time_unit=self.granularity)

        if train_params['boundary'] is not None:
            boundary = (df[self.date_column] - pd.to_datetime(train_params['boundary'])).abs().argmin()
            boundary_date = df.iloc[boundary][self.date_column]
        else:
            last_date = df[self.date_column].max()
            boundary_date = last_date - parse_time_string(train_params['test_len'])
            boundary = (df[self.date_column] - boundary_date).abs().argmin()

        begin_train_date = boundary_date - parse_time_string(train_params['train_len'])
        end_test_date = boundary_date + parse_time_string(train_params['test_len'])

        sel_cols = select_columns(
            df=df,
            include_prefixes=self.column_selection['prefixes'],
            include_suffixes=self.column_selection['suffixes'],
            include_regex=self.column_selection['regex'],
            exclude_columns=self.column_selection['to_exclude']
        )

        train_mask = (df[self.date_column] >= begin_train_date) & (df[self.date_column] < boundary_date)
        test_mask = (df[self.date_column] >= boundary_date) & (df[self.date_column] < end_test_date)


        # TODO ERASE
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

        self.X_train = df.loc[train_mask].copy()
        self.X_test = df.loc[test_mask].copy()
        



      


        self.y_train = self.X_train[y_col]
        self.X_train.drop(y_col, axis=1, inplace=True)
        self.y_test = self.X_test[y_col]
        self.X_test.drop(y_col, axis=1, inplace=True)
        


        """
        self.X_train, self.X_test, self.scaler = normalize_selected_columns(
            X_train=self.X_train,
            X_test=self.X_test,
            include_keywords=["Prezzo", "pct_change", "trend"],
            scaler_type="robust"
        )
        """
        

        print(f"🗕️ Train: {begin_train_date.date()} → {boundary_date.date()} → {len(self.X_train)} rows")
        print(f"🗕️ Test : {boundary_date.date()} → {end_test_date.date()} → {len(self.X_test)} rows")

    def get_all_products(self):
        return pd.unique(self.dataset[self.product_var])
