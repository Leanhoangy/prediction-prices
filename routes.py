import json
import uuid
from datetime import datetime
from pathlib import Path

_eda_cache = {}

import joblib
import pandas as pd
from flask import Blueprint, flash, redirect, render_template, request, url_for

from config import BDS_CONFIG, COLUMN_LABELS, MODEL_STAGING_DIR, REPORT_DIR, UPLOAD_DIR
from file_utils import apply_mapping, guess_column, read_file, save_json
from preprocessing import validate_and_clean_dataset
from training import (
    append_history,
    archive_if_exists,
    compare_algorithms_and_select_best,
    evaluate_existing_model,
    get_dropdown_options,
    is_new_model_better,
    merge_dataset_keep_original_columns,
)
from visualization import analyze_dataframe, create_correlation_heatmap, create_eda_charts

main = Blueprint("main", __name__)


@main.route("/")
def dashboard():
    cards = []
    for bds_type, config in BDS_CONFIG.items():
        rows = 0
        if config["dataset"].exists():
            try:
                rows = len(read_file(config["dataset"]))
            except Exception:
                rows = 0
        best_name = "Chưa train"
        if config["best_model"].exists():
            try:
                import joblib as _jl
                m = _jl.load(config["best_model"])
                reg = m.named_steps["regressor"]
                cls = type(reg).__name__
                if "Linear" in cls:
                    best_name = "Linear Regression"
                elif "XGB" in cls or "xgb" in cls.lower():
                    best_name = "XGBoost"
                else:
                    best_name = "Random Forest"
            except Exception:
                best_name = "Có"
        cards.append({
            "type": bds_type,
            "label": config["label"],
            "rows": rows,
            "dataset_exists": config["dataset"].exists(),
            "linear_exists": config["linear_model"].exists(),
            "rf_exists": config["rf_model"].exists(),
            "xgb_exists": config.get("xgb_model", Path("")).exists(),
            "best_exists": config["best_model"].exists(),
            "best_name": best_name,
        })
    return render_template("dashboard.html", cards=cards)


@main.route("/predict", methods=["GET", "POST"])
def predict():
    result, error = None, None
    selected_type = request.form.get("bds_type", "chungcu")
    dropdown_data = get_dropdown_options()

    if request.method == "POST":
        try:
            bds_type = request.form["bds_type"]
            selected_type = bds_type
            config = BDS_CONFIG[bds_type]

            if not config["best_model"].exists():
                raise ValueError("Chưa có best model. Hãy chạy file train_models.py trước.")

            model = joblib.load(config["best_model"])
            input_row = {}

            for col in config["numeric"]:
                input_row[col] = float(request.form[f"{bds_type}_{col}"])
            for col in config["categorical"]:
                input_row[col] = request.form[f"{bds_type}_{col}"]

            price = float(model.predict(pd.DataFrame([input_row]))[0])
            result = {
                "price": round(price, 2),
                "price_billion": round(price / 1000, 2),
                "bds_label": config["label"],
            }
        except Exception as e:
            error = str(e)

    return render_template(
        "predict.html",
        configs=BDS_CONFIG,
        labels=COLUMN_LABELS,
        selected_type=selected_type,
        result=result,
        error=error,
        dropdown_data=dropdown_data,
    )


@main.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        mode = request.form.get("mode")
        bds_type = request.form.get("bds_type")

        if not file or file.filename == "":
            flash("Vui lòng chọn file.", "error")
            return redirect(url_for("main.upload"))

        ext = Path(file.filename).suffix.lower()
        if ext not in [".xlsx", ".csv"]:
            flash("Chỉ hỗ trợ file .xlsx hoặc .csv", "error")
            return redirect(url_for("main.upload"))

        session_id = str(uuid.uuid4())
        saved_path = UPLOAD_DIR / f"{session_id}{ext}"
        file.save(saved_path)

        try:
            df = read_file(saved_path)
        except Exception as e:
            flash(f"Không đọc được file: {e}", "error")
            return redirect(url_for("main.upload"))

        config = BDS_CONFIG[bds_type]
        suggestions = {
            system_col: guess_column(system_col, df.columns)
            for system_col in config["required"]
        }
        meta = {
            "session_id": session_id,
            "file_path": str(saved_path),
            "original_name": file.filename,
            "mode": mode,
            "bds_type": bds_type,
            "columns": list(df.columns),
            "suggestions": suggestions,
        }
        save_json(UPLOAD_DIR / f"{session_id}.json", meta)
        return redirect(url_for("main.map_columns", session_id=session_id))

    return render_template("upload.html", configs=BDS_CONFIG)


@main.route("/map/<session_id>", methods=["GET", "POST"])
def map_columns(session_id):
    meta_path = UPLOAD_DIR / f"{session_id}.json"

    if not meta_path.exists():
        flash("Phiên upload không tồn tại.", "error")
        return redirect(url_for("main.upload"))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    bds_type = meta["bds_type"]
    config = BDS_CONFIG[bds_type]

    if request.method == "GET":
        return render_template("map_columns.html", meta=meta, config=config, labels=COLUMN_LABELS)

    mapping = {system_col: request.form.get(system_col, "") for system_col in config["required"]}
    df_raw = read_file(meta["file_path"])
    df_mapped = apply_mapping(df_raw, mapping)

    errors, warnings, df_clean, cleaning_report = validate_and_clean_dataset(df_mapped, bds_type)

    if errors:
        return render_template(
            "result.html",
            title="Kiểm tra dữ liệu thất bại",
            errors=errors,
            warnings=warnings,
            cleaning_report=cleaning_report,
        )

    analysis = analyze_dataframe(df_clean)
    corr_chart = create_correlation_heatmap(df_clean, bds_type)

    if meta["mode"] == "analyze":
        return render_template(
            "result.html",
            title="Kết quả phân tích file",
            errors=[],
            warnings=warnings,
            analysis=analysis,
            cleaning_report=cleaning_report,
            corr_chart=corr_chart,
        )

    train_result = compare_algorithms_and_select_best(df_clean, bds_type)
    old_best_metrics = evaluate_existing_model(
        config["best_model"], train_result["X_test"], train_result["y_test"]
    )

    better = is_new_model_better(old_best_metrics, train_result["best_metrics"])
    decision = {
        "better": better,
        "message": (
            "Model mới tốt hơn hoặc chưa có model cũ."
            if better
            else "Model mới chưa tốt hơn model production cũ."
        ),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    updated_production = False

    if meta["mode"] == "train_private":
        joblib.dump(train_result["linear"]["model"], MODEL_STAGING_DIR / f"private_linear_{bds_type}_{timestamp}.pkl")
        joblib.dump(train_result["random_forest"]["model"], MODEL_STAGING_DIR / f"private_rf_{bds_type}_{timestamp}.pkl")
        joblib.dump(train_result["best_model"], MODEL_STAGING_DIR / f"private_best_{bds_type}_{timestamp}.pkl")
    elif better:
        archive_if_exists(config["linear_model"], bds_type, "linear")
        archive_if_exists(config["rf_model"], bds_type, "rf")
        archive_if_exists(config["best_model"], bds_type, "best")

        joblib.dump(train_result["linear"]["model"], config["linear_model"])
        joblib.dump(train_result["random_forest"]["model"], config["rf_model"])
        joblib.dump(train_result["best_model"], config["best_model"])

        merge_dataset_keep_original_columns(config["dataset"], df_raw)
        updated_production = True

    append_history({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": meta["mode"],
        "bds_type": bds_type,
        "bds_label": config["label"],
        "file": meta["original_name"],
        "rows_after_clean": analysis["rows"],
        "linear_metrics": train_result["linear"]["metrics"],
        "rf_metrics": train_result["random_forest"]["metrics"],
        "best_algorithm": train_result["best_algorithm"],
        "best_metrics": train_result["best_metrics"],
        "old_best_metrics": old_best_metrics,
        "updated_production": updated_production,
    })

    return render_template(
        "result.html",
        title="Kết quả train và so sánh mô hình",
        errors=[],
        warnings=warnings,
        analysis=analysis,
        cleaning_report=cleaning_report,
        corr_chart=corr_chart,
        old_best_metrics=old_best_metrics,
        train_result=train_result,
        decision=decision,
        updated_production=updated_production,
    )


@main.route("/eda")
@main.route("/eda/<bds_type>")
def eda(bds_type="chungcu"):
    if bds_type not in BDS_CONFIG:
        bds_type = "chungcu"

    config = BDS_CONFIG[bds_type]

    if not config["dataset"].exists():
        flash("Chưa có dataset cho loại BĐS này.", "error")
        return redirect(url_for("main.dashboard"))

    dataset_mtime = config["dataset"].stat().st_mtime
    cache_key = (bds_type, dataset_mtime)

    if cache_key not in _eda_cache:
        df_raw = read_file(config["dataset"])
        df_renamed = df_raw.rename(columns=config["rename"])
        _, _, df, _ = validate_and_clean_dataset(df_renamed, bds_type)
        _eda_cache[cache_key] = {
            "stats": {
                "total": len(df),
                "price_mean": round(float(df["Gia"].mean()), 2),
                "price_median": round(float(df["Gia"].median()), 2),
                "price_min": round(float(df["Gia"].min()), 2),
                "price_max": round(float(df["Gia"].max()), 2),
                "area_mean": round(float(df["DienTich"].mean()), 2),
                "area_median": round(float(df["DienTich"].median()), 2),
            },
            "charts": create_eda_charts(df, bds_type),
        }

    stats = _eda_cache[cache_key]["stats"]
    charts = _eda_cache[cache_key]["charts"]

    return render_template(
        "eda.html",
        configs=BDS_CONFIG,
        selected=bds_type,
        stats=stats,
        charts=charts,
    )


@main.route("/history")
def history():
    history_path = REPORT_DIR / "train_history.json"
    records = []
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    return render_template("history.html", records=list(reversed(records)))
