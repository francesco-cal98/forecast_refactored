from dataclasses import dataclass
import json
import re
from typing import Any, cast
import pandas as pd
import numpy as np

from utils.helper import (
    parse_time_string,
    replace_patterns_in_dict,
    select_columns,
    week_string_to_date,
    load_weekly_data_with_prices
)

@dataclass
class SalesDataset:
    def __init__(self, 
                csv_file: str, 
                separator: str, 
                date_column: str,
                target_column: str,
                nr_horizon: int,
                prepare_config: str,
                column_selection: str,
                product_var: str,
                product_name: str | None = None):
        
        self.csv_file = csv_file
        self.separator = separator
        self.date_column = date_column
        self.target_column = target_column
        self.nr_horizon = nr_horizon
        self.product_var = product_var
        self.product_name = product_name

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
            
        # Auto-reshape se formato settimanale (W1 2024, W2 2024, ecc.)
        if self.dataset[self.date_column].astype(str).str.contains("W").all():
            # Pulisce valori numerici tipo "1,240"
            self.dataset[target_column] = (
                self.dataset[target_column].astype(str).str.replace(",", "").astype(float)
            )
        else:
            self.dataset[date_column] = pd.to_datetime(self.dataset[date_column], format='%Y-%m-%d', errors='coerce')
            self.dataset.sort_values(by=date_column, inplace=True)
            self.dataset.reset_index(drop=True, inplace=True)

    def _prepare_dataset(self, df: pd.DataFrame | None = None, group_columns: list[str] = ['product_group']) -> pd.DataFrame:
        if df is None:
            df = self.dataset.copy()
        
        # LAG features
        for col, lag in self.prepare_config['lag'].items():
            if isinstance(lag, int): lag = [lag]
            for l in lag:
                df[f'{col}_{l}W'] = df.groupby(group_columns)[col].shift(l)
        
        
        # % variation feature
        for col, percentage in self.prepare_config['percentage'].items():
            if isinstance(percentage, int): percentage = [percentage]
            for p in percentage:
                df[f'{col}_percentage_{p}_W'] = df.groupby(group_columns)[col].pct_change(p).clip(-2,2).shift(1)


        # Rolling features
        for op, col_window in self.prepare_config['rolling'].items():
            for col, w in col_window.items():
                if isinstance(w, int): w = [w]
                for wi in w:
                    for g in df[group_columns].drop_duplicates().values:
                        mask = (df[group_columns] == g).all(axis=1)
                        df.loc[mask, f'{col}_rolling_{op}_{wi}W'] = df.loc[mask, col].rolling(window=wi, min_periods=1).agg(op).shift(1)

        # Trend features
        for col, windows in self.prepare_config.get("trend", {}).items():
            if isinstance(windows, int): windows = [windows]
            for wi in windows:
                for g in df[group_columns].drop_duplicates().values:
                    mask = (df[group_columns] == g).all(axis=1)
                    df.loc[mask, f"{col}_trend_{wi}W"] = df.loc[mask, col].rolling(window=wi, min_periods=wi).apply(
                        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
                    ).shift(1)

        for promo in [1, 2]:
            promo_col = f'type_promo_{promo}'
            if promo_col in df.columns:
                df[f'type_promo_{promo}_current'] = df[f'type_promo_{promo}']
                df[f'type_promo_{promo}_{self.nr_horizon}W'] = df.groupby(group_columns)[f'type_promo_{promo}'].shift(-self.nr_horizon)
        
        # eliminate every possible NaN 
        
        df.dropna(inplace=True)
        return df

    def _encode_time(self, X: pd.DataFrame, time_unit='week'):
        # Se la colonna non è in formato datetime, proviamo a convertirla
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
        df = self._encode_time(X=df, time_unit='week')

        if train_params['boundary'] is not None:
            boundary = (df[self.date_column] - pd.to_datetime(train_params['boundary'])).abs().argmin()
            boundary_date = df.iloc[boundary] [self.date_column]
        else:
            last_date = df[self.date_column].max()
            boundary_date = last_date - parse_time_string(train_params['test_len'])
            boundary = (df[self.date_column] - boundary_date).abs().argmin()

        begin_train_date = boundary_date - parse_time_string(train_params['train_len'])
        begin_train = (df[self.date_column] - begin_train_date).abs().argmin()
        end_test_date = boundary_date + parse_time_string(train_params['test_len'])
        end_test = (df[self.date_column] - end_test_date).abs().argmin()

        sel_cols = select_columns(
            df=df,
            include_prefixes=self.column_selection['prefixes'],
            include_suffixes=self.column_selection['suffixes'],
            include_regex=self.column_selection['regex'],
            exclude_columns=self.column_selection['to_exclude']
        )

        train_mask = (df[self.date_column] >= begin_train_date) & (df[self.date_column] < boundary_date)
        test_mask = (df[self.date_column] >= boundary_date) & (df[self.date_column] < end_test_date)

        df = df[sel_cols]

        self.X_train = df.loc[train_mask].copy()
        self.X_test = df.loc[test_mask].copy()

        self.y_train = self.X_train[y_col]
        self.X_train.drop(y_col, axis=1, inplace=True)
        self.y_test = self.X_test[y_col]
        self.X_test.drop(y_col, axis=1, inplace=True)

        print(f"📅 Train: {begin_train_date.date()} → {boundary_date.date()} → {len(self.X_train)} rows")
        print(f"📅 Test : {boundary_date.date()} → {end_test_date.date()} → {len(self.X_test)} rows")


    def get_all_products(self):
        return pd.unique(self.dataset[self.product_var])
 