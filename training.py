import shutil
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from config import BDS_CONFIG, MODEL_ARCHIVE_DIR, REPORT_DIR
from file_utils import read_file, save_json
from visualization import create_actual_vs_pred_chart, create_feature_importance_chart


def build_model_pipeline(bds_type, algorithm, verbose=0):
    config = BDS_CONFIG[bds_type]
    preprocessor = ColumnTransformer([
        ("num", "passthrough", config["numeric"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config["categorical"]),
    ])
    if algorithm == "linear":
        regressor = LinearRegression()
    elif algorithm == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, verbose=verbose
        )
    else:
        regressor = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=verbose, tree_method="hist"
        )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def calc_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "MAE": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(np.sqrt(mse)), 4),
        "R2": round(float(r2_score(y_test, y_pred)), 4),
    }, y_pred


def cross_validate_pipeline(bds_type, algorithm, X, y, k=5):
    """Chạy k-fold cross-validation. Trả về mean ± std của R² và MAE."""
    model = build_model_pipeline(bds_type, algorithm)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    r2 = cross_val_score(model, X, y, cv=kf, scoring="r2")
    mae = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
    return {
        "k": k,
        "R2_mean": round(float(r2.mean()), 4),
        "R2_std": round(float(r2.std()), 4),
        "MAE_mean": round(float(mae.mean()), 4),
        "MAE_std": round(float(mae.std()), 4),
    }


def train_one_algorithm(bds_type, algorithm, X_train, X_test, y_train, y_test):
    model = build_model_pipeline(bds_type, algorithm)
    model.fit(X_train, y_train)
    metrics, y_pred = calc_metrics(model, X_test, y_test)
    charts = {
        "actual_vs_predicted": create_actual_vs_pred_chart(
            y_test, y_pred, f"{algorithm} - Actual vs Predicted"
        ),
        "feature_importance": create_feature_importance_chart(model, bds_type, algorithm),
    }
    return model, metrics, charts


def compare_algorithms_and_select_best(df, bds_type):
    """Train Linear và Random Forest, chọn model tốt nhất theo R2/MAE."""
    config = BDS_CONFIG[bds_type]
    X = df[config["numeric"] + config["categorical"]]
    y = df["Gia"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    linear_model, linear_metrics, linear_charts = train_one_algorithm(
        bds_type, "linear", X_train, X_test, y_train, y_test
    )
    rf_model, rf_metrics, rf_charts = train_one_algorithm(
        bds_type, "random_forest", X_train, X_test, y_train, y_test
    )
    xgb_model, xgb_metrics, xgb_charts = train_one_algorithm(
        bds_type, "xgboost", X_train, X_test, y_train, y_test
    )

    candidates = [
        ("linear", linear_model, linear_metrics),
        ("random_forest", rf_model, rf_metrics),
        ("xgboost", xgb_model, xgb_metrics),
    ]
    best_algorithm, best_model, best_metrics = max(
        candidates,
        key=lambda x: (x[2]["R2"], -x[2]["MAE"])
    )

    return {
        "linear": {"model": linear_model, "metrics": linear_metrics, "charts": linear_charts},
        "random_forest": {"model": rf_model, "metrics": rf_metrics, "charts": rf_charts},
        "xgboost": {"model": xgb_model, "metrics": xgb_metrics, "charts": xgb_charts},
        "best_algorithm": best_algorithm,
        "best_model": best_model,
        "best_metrics": best_metrics,
        "X_test": X_test,
        "y_test": y_test,
    }


def evaluate_existing_model(path, X_test, y_test):
    if not Path(path).exists():
        return None
    metrics, _ = calc_metrics(joblib.load(path), X_test, y_test)
    return metrics


def is_new_model_better(old_metrics, new_metrics):
    if old_metrics is None:
        return True
    return new_metrics["R2"] >= old_metrics["R2"] and new_metrics["MAE"] <= old_metrics["MAE"]


def archive_if_exists(path, bds_type, suffix):
    path = Path(path)
    if not path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = MODEL_ARCHIVE_DIR / f"{bds_type}_{suffix}_{timestamp}.pkl"
    shutil.copy2(path, archive_path)
    return archive_path


def append_history(record):
    history_path = REPORT_DIR / "train_history.json"
    history = []
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            import json
            history = json.load(f)
    history.append(record)
    save_json(history_path, history)


def merge_dataset_keep_original_columns(old_path, new_raw_df):
    if old_path.exists():
        combined = pd.concat([read_file(old_path), new_raw_df], ignore_index=True).drop_duplicates()
    else:
        combined = new_raw_df.drop_duplicates()
    combined.to_excel(old_path, index=False)


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


def get_dropdown_options():
    """Lấy danh sách dropdown Quận/Hướng/Pháp lý riêng theo từng loại BĐS."""
    dropdown_data = {}
    for bds_type, config in BDS_CONFIG.items():
        dropdown_data[bds_type] = {}
        try:
            df = read_file(config["dataset"]).rename(columns=config.get("rename", {}))
            for col in config["categorical"]:
                if col in df.columns:
                    values = df[col].dropna().astype(str).str.strip()
                    if col == "TinhThanh":
                        values = values.str.lower().map(
                            lambda x: _TINH_NORMALIZE.get(x, x.title())
                        )
                    dropdown_data[bds_type][col] = sorted(
                        v for v in values.unique().tolist() if v and v != "nan"
                    )
                else:
                    dropdown_data[bds_type][col] = []
        except Exception:
            for col in config["categorical"]:
                dropdown_data[bds_type][col] = []
    return dropdown_data
