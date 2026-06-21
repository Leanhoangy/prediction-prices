import json
from pathlib import Path

import pandas as pd

from config import COLUMN_ALIASES


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_file(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def normalize_text(value):
    return str(value).strip().lower()


def guess_column(system_col, user_columns):
    """Tự động gợi ý cột upload tương ứng với cột chuẩn của hệ thống."""
    aliases = COLUMN_ALIASES.get(system_col, [])
    normalized_user_cols = {normalize_text(col): col for col in user_columns}

    for alias in aliases:
        if normalize_text(alias) in normalized_user_cols:
            return normalized_user_cols[normalize_text(alias)]

    for user_col in user_columns:
        user_norm = normalize_text(user_col)
        for alias in aliases:
            alias_norm = normalize_text(alias)
            if alias_norm in user_norm or user_norm in alias_norm:
                return user_col

    return ""


def apply_mapping(df, mapping):
    """Đổi tên cột file upload về tên cột chuẩn của hệ thống."""
    rename_dict = {
        user_col: system_col
        for system_col, user_col in mapping.items()
        if user_col and user_col in df.columns
    }
    return df.rename(columns=rename_dict)
