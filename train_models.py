"""
train_models.py
===============
Script độc lập để train model và lưu vào models/production/.

Cách chạy:
    python train_models.py

Kết quả sinh ra:
- models/production/linear_<type>.pkl
- models/production/rf_<type>.pkl
- models/production/best_<type>.pkl
- reports/training_results.xlsx

Quy tắc làm sạch dữ liệu:
- Thiếu/sai/không hợp lệ/outlier theo IQR thì loại bỏ cả dòng.
- Không tự điền missing value.
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from config import BDS_CONFIG, CHART_DIR, MODEL_PRODUCTION_DIR, REPORT_DIR
from preprocessing import validate_and_clean_dataset
from training import build_model_pipeline, calc_metrics, cross_validate_pipeline
from visualization import create_actual_vs_pred_chart, create_feature_importance_chart


def train_one_dataset(bds_type, config):
    print(f"\n========== TRAIN {config['label'].upper()} ==========")

    if not config["dataset"].exists():
        raise FileNotFoundError(f"Không tìm thấy file: {config['dataset']}")

    df_raw = pd.read_excel(config["dataset"])
    df_renamed = df_raw.rename(columns=config["rename"])

    errors, warnings, df, clean_report = validate_and_clean_dataset(df_renamed, bds_type)
    if errors:
        raise ValueError("\n".join(errors))
    for w in warnings:
        print(f"[WARN] {w}")

    X = df[config["numeric"] + config["categorical"]]
    y = df["Gia"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train set: {len(X_train)} dòng | Test set: {len(X_test)} dòng")

    print("  [1/3] Đang train Linear Regression...")
    linear_model = build_model_pipeline(bds_type, "linear")
    linear_model.fit(X_train, y_train)
    linear_metrics, linear_pred = calc_metrics(linear_model, X_test, y_test)
    print(f"        → R²={linear_metrics['R2']}  MAE={linear_metrics['MAE']}")

    print("  [2/3] Đang train Random Forest (200 cây)...")
    rf_model = build_model_pipeline(bds_type, "random_forest", verbose=1)
    rf_model.fit(X_train, y_train)
    rf_metrics, rf_pred = calc_metrics(rf_model, X_test, y_test)
    print(f"        → R²={rf_metrics['R2']}  MAE={rf_metrics['MAE']}")

    print("  [3/3] Đang train XGBoost (300 cây)...")
    xgb_model = build_model_pipeline(bds_type, "xgboost")
    xgb_model.fit(X_train, y_train)
    xgb_metrics, xgb_pred = calc_metrics(xgb_model, X_test, y_test)
    print(f"        → R²={xgb_metrics['R2']}  MAE={xgb_metrics['MAE']}")

    create_actual_vs_pred_chart(y_test, linear_pred, f"linear_{bds_type} - Actual vs Predicted")
    create_actual_vs_pred_chart(y_test, rf_pred, f"rf_{bds_type} - Actual vs Predicted")
    create_actual_vs_pred_chart(y_test, xgb_pred, f"xgboost_{bds_type} - Actual vs Predicted")
    create_feature_importance_chart(linear_model, bds_type, "linear")
    create_feature_importance_chart(rf_model, bds_type, "random_forest")
    create_feature_importance_chart(xgb_model, bds_type, "xgboost")

    print("  [CV] Đang chạy 5-fold cross-validation...")
    linear_cv = cross_validate_pipeline(bds_type, "linear", X, y, k=5)
    rf_cv = cross_validate_pipeline(bds_type, "random_forest", X, y, k=5)
    xgb_cv = cross_validate_pipeline(bds_type, "xgboost", X, y, k=5)
    print(f"       Linear  R²={linear_cv['R2_mean']} ± {linear_cv['R2_std']}  MAE={linear_cv['MAE_mean']} ± {linear_cv['MAE_std']}")
    print(f"       RF      R²={rf_cv['R2_mean']} ± {rf_cv['R2_std']}  MAE={rf_cv['MAE_mean']} ± {rf_cv['MAE_std']}")
    print(f"       XGB     R²={xgb_cv['R2_mean']} ± {xgb_cv['R2_std']}  MAE={xgb_cv['MAE_mean']} ± {xgb_cv['MAE_std']}")

    candidates = [
        ("linear", linear_model, linear_metrics),
        ("random_forest", rf_model, rf_metrics),
        ("xgboost", xgb_model, xgb_metrics),
    ]
    best_name, best_model, _ = max(candidates, key=lambda x: (x[2]["R2"], -x[2]["MAE"]))

    joblib.dump(linear_model, MODEL_PRODUCTION_DIR / f"linear_{bds_type}.pkl")
    joblib.dump(rf_model, MODEL_PRODUCTION_DIR / f"rf_{bds_type}.pkl")
    joblib.dump(xgb_model, MODEL_PRODUCTION_DIR / f"xgb_{bds_type}.pkl")
    joblib.dump(best_model, MODEL_PRODUCTION_DIR / f"best_{bds_type}.pkl")

    print(f"  Best: {best_name}")

    return {
        "LoaiBDS": config["label"],
        "RowsBefore": clean_report["original_rows"],
        "RowsAfter": clean_report["final_rows"],
        "MissingRemoved": clean_report["missing_removed"],
        "InvalidRemoved": clean_report["invalid_removed"],
        "DuplicatesRemoved": clean_report["duplicate_removed"],
        "OutliersRemovedIQR": clean_report["outlier_removed"],
        "Linear_MAE": linear_metrics["MAE"],
        "Linear_MSE": linear_metrics["MSE"],
        "Linear_RMSE": linear_metrics["RMSE"],
        "Linear_R2": linear_metrics["R2"],
        "Linear_CV_R2": f"{linear_cv['R2_mean']} ± {linear_cv['R2_std']}",
        "Linear_CV_MAE": f"{linear_cv['MAE_mean']} ± {linear_cv['MAE_std']}",
        "RF_MAE": rf_metrics["MAE"],
        "RF_MSE": rf_metrics["MSE"],
        "RF_RMSE": rf_metrics["RMSE"],
        "RF_R2": rf_metrics["R2"],
        "RF_CV_R2": f"{rf_cv['R2_mean']} ± {rf_cv['R2_std']}",
        "RF_CV_MAE": f"{rf_cv['MAE_mean']} ± {rf_cv['MAE_std']}",
        "XGB_MAE": xgb_metrics["MAE"],
        "XGB_MSE": xgb_metrics["MSE"],
        "XGB_RMSE": xgb_metrics["RMSE"],
        "XGB_R2": xgb_metrics["R2"],
        "XGB_CV_R2": f"{xgb_cv['R2_mean']} ± {xgb_cv['R2_std']}",
        "XGB_CV_MAE": f"{xgb_cv['MAE_mean']} ± {xgb_cv['MAE_std']}",
        "BestModel": best_name,
    }


def cleanup_old_charts():
    """Xóa chart cũ, giữ lại pipeline.png và model_comparison_r2.png."""
    keep = {"pipeline.png", "pipeline_slide.png", "model_comparison_r2.png"}
    for f in CHART_DIR.glob("*.png"):
        if f.name not in keep:
            f.unlink()


def main():
    cleanup_old_charts()
    results = [train_one_dataset(bds_type, config) for bds_type, config in BDS_CONFIG.items()]

    result_df = pd.DataFrame(results)
    report_path = REPORT_DIR / "training_results.xlsx"
    result_df.to_excel(report_path, index=False)

    print("\n========== HOÀN TẤT ==========")
    print(result_df)
    print(f"\nĐã lưu báo cáo: {report_path}")
    print(f"Đã lưu model tại: {MODEL_PRODUCTION_DIR}")


if __name__ == "__main__":
    main()
