from typing import Callable, List
import pandas as pd
from omegaconf import OmegaConf
from omegaconf import DictConfig
from copy import deepcopy

import datetime
import re

def replace_patterns_in_dict(d, pattern_transforms) -> dict:
    """
    Recursively replace patterns in dictionary structure with their transformed values.
    
    Args:
        d: Input structure (can be dict, list, or single value)
        pattern_transforms: Dictionary mapping patterns to their transform functions
        
    Returns:
        Transformed structure maintaining original hierarchy
    """
    # Case 1: Dictionary handling
    if isinstance(d, dict):
        return {
            # For each key-value pair:
            # 1. Transform key if it matches any pattern using next() and generator expression
            # 2. If no match found (next() reaches end), keep original key (k)
            next((transform(k) for p, transform in pattern_transforms.items() if k == p), k): 
            # Recursively process the value
            replace_patterns_in_dict(v, pattern_transforms)
            for k, v in d.items()
        }
    
    # Case 2: List handling - recursively process each item
    elif isinstance(d, list):
        return [replace_patterns_in_dict(item, pattern_transforms) for item in d]
    
    # Case 3: Single value handling
    # Transform if matches pattern, otherwise return unchanged
    return next(
        (transform(d) for p, transform in pattern_transforms.items() if d == p), 
        d
    )
    


def resolve_target_keys(cfg):
    prepare_cfg = deepcopy(cfg.prepare)
    target_col = cfg.data.target_column

    # LAG (chiave diretta)
    if "lag" in prepare_cfg:
        new_lag = {}
        for col, lags in prepare_cfg["lag"].items():
            new_lag[target_col if col in ("${data.target_column}", "${target_column}") else col] = lags
        prepare_cfg["lag"] = new_lag

    # ROLLING (chiave annidata)
    if "rolling" in prepare_cfg:
        new_rolling = {}
        for func, cols_dict in prepare_cfg["rolling"].items():
            new_cols_dict = {}
            for col, windows in cols_dict.items():
                new_cols_dict[target_col if col in ("${data.target_column}", "${target_column}") else col] = windows
            new_rolling[func] = new_cols_dict
        prepare_cfg["rolling"] = new_rolling

    # TREND (chiave diretta, simile a LAG)
    if "trend" in prepare_cfg:
        new_trend = {}
        for col, windows in prepare_cfg["trend"].items():
            new_trend[target_col if col in ("${data.target_column}", "${target_column}") else col] = windows
        prepare_cfg["trend"] = new_trend

    return prepare_cfg

def resolve_config_refs(obj, cfg):
    """Ricorsivamente sostituisce tutte le stringhe tipo '${...}' con il valore effettivo da cfg."""
    if isinstance(obj, DictConfig):
        return {resolve_config_refs(k, cfg): resolve_config_refs(v, cfg) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_config_refs(i, cfg) for i in obj]
    elif isinstance(obj, str) and "${" in obj:
        return OmegaConf.to_container(OmegaConf.create({"key": obj}), resolve=True)["key"]
    return obj


def week_string_to_date(week_str):
    """Converte 'W6 2024' in datetime corrispondente al lunedì di quella settimana"""
    match = re.match(r"W(\d{1,2})\s+(\d{4})", str(week_str))
    if match:
        week, year = int(match.group(1)), int(match.group(2))
        return datetime.date.fromisocalendar(year, week, 1)
    else:
        return pd.NaT



def select_columns(
                df: pd.DataFrame,
                include_prefixes: List[str]| None = None,
                include_suffixes: List[str]| None = None,
                include_regex:    List[str]| None = None,
                exclude_columns:  List[str]| None = None,
                custom_filter:    Callable | None = None) -> List[str]:
    """
    Flexible column selector based on multiple criteria
    """
    selected_cols = set()
    
    # Add columns matching prefixes
    if include_prefixes:
        for prefix in include_prefixes:
            selected_cols.update(df.filter(like=prefix).columns)
    
    # Add columns matching suffixes
    if include_suffixes:
        selected_cols.update([col for col in df.columns 
                            if any(col.endswith(suffix) for suffix in include_suffixes)])
    
    # Add columns matching regex patterns
    if include_regex:
        for pattern in include_regex:
            selected_cols.update(df.filter(regex=pattern).columns)
    
    # Apply custom filter function
    if custom_filter:
        selected_cols.update(custom_filter(df.columns))
    
    # Remove excluded columns
    if exclude_columns:
        selected_cols = selected_cols - set(exclude_columns)
    
    return list(selected_cols)


def parse_time_string(time_str: str) -> pd.DateOffset:
    """Convert string time duration to pandas DateOffset
    Accepts formats like '1y', '3m', '2w', '5d'
    """
    amount = int(time_str[:-1])
    unit = time_str[-1]
    
    offset_map = {
        'y': pd.DateOffset(years=amount),
        'm': pd.DateOffset(months=amount),
        'w': pd.DateOffset(weeks=amount),
        'd': pd.DateOffset(days=amount)
    }
    
    return offset_map.get(unit, pd.DateOffset())

import pandas as pd
import re


def load_weekly_data_with_prices(filepath, product_var='Item', date_column='Settimana', target_column='Sellout'):
    import pandas as pd

    # === 1. Carica il file Excel ===
    xls = pd.ExcelFile(filepath)

    # === 2. Leggi sellout settimanale ===
    df_sellout = pd.read_excel(xls, sheet_name=0, engine='openpyxl', header=2)
    week_cols = [col for col in df_sellout.columns if isinstance(col, str) and col.startswith("W")]
    id_col = df_sellout.columns[0]
    
    df_sellout = df_sellout.melt(id_vars=id_col, var_name=date_column, value_name=target_column)
    df_sellout = df_sellout.rename(columns={id_col: product_var})
    df_sellout[target_column] = pd.to_numeric(df_sellout[target_column].astype(str).str.replace(",", ""), errors='coerce')
    
    df_sellout[['Settimana_num', 'Anno']] = df_sellout[date_column].str.extract(r'W(\d{1,2}) (\d{4})').astype(int)
    df_sellout['Mese'] = ((df_sellout['Settimana_num'] - 1) // 4) + 1

    # === 3. Carica prezzi con MultiIndex (Anno, MESE)
# Leggi il foglio senza header, così puoi gestire tu l'intestazione
# Carica senza header per gestire a mano
    df_raw = pd.read_excel(filepath, sheet_name=1, engine='openpyxl', header=None)

    # Estrai intestazioni da riga 0 e 1 (colonne da 1 in poi)
    header = list(zip(df_raw.iloc[0, 1:], df_raw.iloc[1, 1:]))  # Skip col 0 che ha 'Item'
    header = [('Item', '')] + header  # Prima colonna è 'Item'

    # Prendi i dati (da riga 2 in poi)
    df_data = df_raw.iloc[2:].copy()
    df_data.columns = pd.MultiIndex.from_tuples(header)
    df_data = df_data.rename(columns={('Item', ''): 'Item'})

    # Trasforma in formato long (Item, Anno, Mese, Prezzo)
    df_price = df_data.melt(id_vars='Item', var_name=['Anno', 'Mese'], value_name='Prezzo')

    # Pulizia
    df_price = df_price[df_price['Prezzo'].notna()]
    df_price = df_price[~df_price['Prezzo'].astype(str).str.upper().str.contains("EOL")]
    df_price['Prezzo'] = df_price['Prezzo'].astype(str).str.replace('$', '', regex=False).astype(float)
    df_price['Anno'] = df_price['Anno'].astype(int)
    df_price['Mese'] = df_price['Mese'].astype(int)

    rows = []

    for _, row in df_price.iterrows():
        for w in range(1, 5):  # 4 settimane per mese
            settimana_num = (row['Mese'] - 1) * 4 + w
            rows.append({
                'Item': row['Item'],
                'Anno': row['Anno'],
                'Settimana_num': settimana_num,
                'Prezzo': row['Prezzo']
            })

    df_price_weekly = pd.DataFrame(rows)
    
    # merge del dataset
    df_final = pd.merge(
    df_sellout,
    df_price_weekly,
    how='left',
    on=['Item', 'Anno', 'Settimana_num']
)


    
    # 4. Merge prezzi settimanali con dati settimanali sellout
    df_final = pd.merge(df_sellout, df_price_weekly, how='left', on=[product_var, 'Anno', 'Settimana_num'])

    # 5. Rimuovi colonne Anno e Mese (non servono più)
    df_final = df_final[[product_var, date_column, target_column, 'Prezzo']]

    return df_final


