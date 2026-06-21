import numpy as np
import pandas as pd

from config import BDS_CONFIG


def clean_price_value(x):
    """Chuyển giá về đơn vị triệu đồng. Không chuyển được thì trả về NaN."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).lower().strip().replace(",", ".").replace(" ", "")
    try:
        if "tỷ" in s or "ty" in s:
            return float(s.replace("tỷ", "").replace("ty", "")) * 1000
        if "triệu" in s or "trieu" in s:
            return float(s.replace("triệu", "").replace("trieu", ""))
        return float(s)
    except Exception:
        return np.nan


def to_numeric_series(series):
    """Chuyển cột số về float. Giá trị sai kiểu thành NaN."""
    return (
        series.astype(str)
        .str.lower()
        .str.replace("m2", "", regex=False)
        .str.replace("m²", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)")[0]
        .astype(float)
    )


def remove_iqr_outliers(df, cols):
    """Loại dòng nằm ngoài khoảng [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    removed_report = {}
    before_all = len(df)

    for col in cols:
        before = len(df)
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            removed_report[col] = 0
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        removed_report[col] = before - len(df)

    return df, removed_report, before_all - len(df)


def validate_and_clean_dataset(df, bds_type):
    """
    Làm sạch dữ liệu theo quy tắc: thiếu/sai/không hợp lệ/outlier thì loại cả dòng.
    Trả về (errors, warnings, df_clean, report).
    """
    config = BDS_CONFIG[bds_type]
    required_cols = config["required"]
    errors, warnings = [], []

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        errors.append("Thiếu cột bắt buộc: " + ", ".join(missing_cols))
        return errors, warnings, df, {}

    report = {
        "original_rows": int(len(df)),
        "missing_removed": 0,
        "invalid_removed": 0,
        "duplicate_removed": 0,
        "outlier_removed": 0,
        "outlier_by_column": {},
        "final_rows": 0,
    }

    df = df[required_cols].copy()

    before = len(df)
    df = df.dropna()
    report["missing_removed"] = before - len(df)

    df["Gia"] = df["Gia"].apply(clean_price_value)
    for col in config["numeric"]:
        df[col] = to_numeric_series(df[col])

    _TINH_NORMALIZE = {
        "tp hồ chí minh": "Hồ Chí Minh",
        "tp. hồ chí minh": "Hồ Chí Minh",
        "tp.hcm": "Hồ Chí Minh",
        "tphcm": "Hồ Chí Minh",
        "hcm": "Hồ Chí Minh",
        "tp hcm": "Hồ Chí Minh",
        "thành phố hồ chí minh": "Hồ Chí Minh",
        "hồ chí minh": "Hồ Chí Minh",
        "hà nội": "Hà Nội",
        "ha noi": "Hà Nội",
    }

    for col in config["categorical"]:
        df[col] = df[col].astype(str).str.strip()
        if col == "TinhThanh":
            df[col] = df[col].str.lower().map(
                lambda x: _TINH_NORMALIZE.get(x, x.title())
            )
        df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    before = len(df)
    df = df.dropna()
    df = df[df["Gia"] > 0]
    for col in config["numeric"]:
        df = df[df[col] > 0]
    report["invalid_removed"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates()
    report["duplicate_removed"] = before - len(df)

    iqr_cols = ["Gia"] + config["numeric"]
    df, outlier_by_col, outlier_total = remove_iqr_outliers(df, iqr_cols)
    report["outlier_removed"] = int(outlier_total)
    report["outlier_by_column"] = {k: int(v) for k, v in outlier_by_col.items()}
    report["final_rows"] = int(len(df))

    if report["missing_removed"] > 0:
        warnings.append(f"Đã loại {report['missing_removed']} dòng có giá trị thiếu.")
    if report["invalid_removed"] > 0:
        warnings.append(f"Đã loại {report['invalid_removed']} dòng sai kiểu, giá <= 0 hoặc số liệu <= 0.")
    if report["duplicate_removed"] > 0:
        warnings.append(f"Đã loại {report['duplicate_removed']} dòng trùng.")
    if report["outlier_removed"] > 0:
        warnings.append(f"Đã loại {report['outlier_removed']} dòng outlier bằng phương pháp IQR.")

    if len(df) < 30:
        errors.append("Dữ liệu sau làm sạch còn quá ít dòng để train. Cần ít nhất khoảng 30 dòng.")

    return errors, warnings, df, report
