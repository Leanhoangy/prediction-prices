import uuid
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import BDS_CONFIG, CHART_DIR, COLUMN_LABELS


def analyze_dataframe(df):
    """Tạo thống kê mô tả cho DataFrame đã làm sạch."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "missing": df.isna().sum().to_dict(),
        "numeric_columns": numeric_cols,
        "stats": df[numeric_cols].describe().round(2).to_dict() if numeric_cols else {},
    }


def _safe_name(text):
    """Chuẩn hóa tên file: bỏ ký tự lạ, đổi khoảng trắng thành dấu _."""
    text = str(text).lower().strip()
    text = text.replace(" - actual vs predicted", "")
    text = text.replace("actual vs predicted", "")
    text = text.replace(" ", "_")
    text = text.replace("/", "_")
    text = text.replace("\\", "_")
    text = text.replace(":", "_")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("–", "_")
    text = text.replace("-", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _chart_name(prefix, unique=False):
    """
    unique=False: ghi đè chart mới nhất, tránh sinh quá nhiều file rác.
    unique=True: tạo file random theo thời gian nếu cần lưu lịch sử.
    """
    if unique:
        return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
    return f"{prefix}.png"


def create_actual_vs_pred_chart(y_test, y_pred, title, unique=False):
    """
    Tạo biểu đồ Actual vs Predicted.

    Ví dụ title:
    - linear_nha - Actual vs Predicted
    - rf_nha - Actual vs Predicted
    - xgboost_nha - Actual vs Predicted

    File xuất ra:
    - linear_nha_actual_vs_predicted.png
    - rf_nha_actual_vs_predicted.png
    - xgboost_nha_actual_vs_predicted.png
    """

    base_name = _safe_name(title)
    filename = _chart_name(f"{base_name}_actual_vs_predicted", unique=unique)
    path = CHART_DIR / filename

    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)

    min_val = min(float(np.min(y_test)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_test)), float(np.max(y_pred)))

    padding = (max_val - min_val) * 0.05
    min_axis = max(0, min_val - padding)
    max_axis = max_val + padding

    plt.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--")

    plt.xlim(min_axis, max_axis)
    plt.ylim(min_axis, max_axis)

    plt.xlabel("Giá thực tế (triệu đồng)")
    plt.ylabel("Giá dự đoán (triệu đồng)")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

    return f"charts/{filename}"


def create_correlation_heatmap(df, bds_type):
    config = BDS_CONFIG[bds_type]
    cols = ["Gia"] + config["numeric"]
    corr = df[cols].corr()

    filename = _chart_name("corr")
    path = CHART_DIR / filename

    plt.figure(figsize=(7, 5))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(cols)), [COLUMN_LABELS.get(c, c) for c in cols], rotation=45, ha="right")
    plt.yticks(range(len(cols)), [COLUMN_LABELS.get(c, c) for c in cols])
    plt.title("Correlation Heatmap")

    for i in range(len(cols)):
        for j in range(len(cols)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

    return f"charts/{filename}"


def create_eda_charts(df, bds_type):
    """Tạo bộ 5 biểu đồ EDA. Trả về dict {key: đường dẫn chart}."""
    label = BDS_CONFIG[bds_type]["label"]
    charts = {}

    # 1. Histogram phân phối giá
    fn = _chart_name(f"{bds_type}_eda_price_hist")
    plt.figure(figsize=(8, 4))
    plt.hist(df["Gia"], bins=30, color="#2563eb", alpha=0.85, edgecolor="white")
    plt.xlabel("Giá (triệu đồng)")
    plt.ylabel("Số lượng")
    plt.title(f"Phân phối giá – {label}")
    plt.tight_layout()
    plt.savefig(CHART_DIR / fn, dpi=140)
    plt.close()
    charts["price_hist"] = f"charts/{fn}"

    # 2. Histogram phân phối diện tích
    fn = _chart_name(f"{bds_type}_eda_area_hist")
    plt.figure(figsize=(8, 4))
    plt.hist(df["DienTich"], bins=30, color="#10b981", alpha=0.85, edgecolor="white")
    plt.xlabel("Diện tích (m²)")
    plt.ylabel("Số lượng")
    plt.title(f"Phân phối diện tích – {label}")
    plt.tight_layout()
    plt.savefig(CHART_DIR / fn, dpi=140)
    plt.close()
    charts["area_hist"] = f"charts/{fn}"

    # 3. Scatter giá vs diện tích (jitter để tránh cột dọc khi data là số nguyên)
    fn = _chart_name(f"{bds_type}_eda_scatter")
    jitter = np.random.default_rng(42).uniform(-0.5, 0.5, size=len(df))
    plt.figure(figsize=(8, 5))
    plt.scatter(df["DienTich"] + jitter, df["Gia"], alpha=0.35, color="#7c3aed", s=14)
    plt.xlabel("Diện tích (m²)")
    plt.ylabel("Giá (triệu đồng)")
    plt.title(f"Tương quan Giá – Diện tích ({label})")
    plt.tight_layout()
    plt.savefig(CHART_DIR / fn, dpi=140)
    plt.close()
    charts["scatter"] = f"charts/{fn}"

    # 4. Top 10 quận/huyện theo giá trung bình
    # Chỉ lấy các quận/huyện có đủ số mẫu để tránh kết luận sai
    min_samples = 5

    quan_stats = (
        df.groupby("Quan")
        .agg(
            GiaTrungBinh=("Gia", "mean"),
            SoMau=("Gia", "count")
        )
        .reset_index()
    )

    quan_stats = quan_stats[quan_stats["SoMau"] >= min_samples]
    top_quan = quan_stats.sort_values("GiaTrungBinh", ascending=False).head(10)

    fn = _chart_name(f"{bds_type}_eda_quan_bar")
    plt.figure(figsize=(9, 5))

    labels = [
        f"{row['Quan']} (n={int(row['SoMau'])})"
        for _, row in top_quan.iterrows()
    ]

    plt.barh(range(len(top_quan)), top_quan["GiaTrungBinh"], color="#2563eb")
    plt.yticks(range(len(top_quan)), labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Giá trung bình (triệu đồng)")
    plt.title(f"Top 10 quận/huyện theo giá trung bình (n ≥ {min_samples})")
    plt.tight_layout()
    plt.savefig(CHART_DIR / fn, dpi=140)
    plt.close()

    charts["quan_bar"] = f"charts/{fn}"

    # 5. Boxplot giá theo hướng
    # Chỉ lấy các hướng có đủ số mẫu
    min_samples = 5

    huong_counts = df["Huong"].value_counts()
    valid_huong = huong_counts[huong_counts >= min_samples].head(6)

    top_huong = valid_huong.index.tolist()
    groups = [df[df["Huong"] == h]["Gia"].values for h in top_huong]

    labels = [
        f"{h}\n(n={int(valid_huong[h])})"
        for h in top_huong
    ]

    fn = _chart_name(f"{bds_type}_eda_huong_box")
    plt.figure(figsize=(9, 5))
    plt.boxplot(
        groups,
        labels=labels,
        patch_artist=True,
        boxprops=dict(facecolor="#dbeafe"),
        medianprops=dict(color="#1d4ed8", linewidth=2)
    )
    plt.ylabel("Giá (triệu đồng)")
    plt.title(f"Phân phối giá theo hướng nhà (n ≥ {min_samples})")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(CHART_DIR / fn, dpi=140)
    plt.close()

    charts["huong_box"] = f"charts/{fn}"
    return charts


def _get_feature_names(model, bds_type):
    config = BDS_CONFIG[bds_type]
    cat_encoder = model.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(config["categorical"]).tolist()
    return config["numeric"] + cat_names


def _format_feature_label(name):
    """Đổi tên feature sau OneHotEncoder cho dễ đọc trên biểu đồ."""
    name = str(name)

    replacements = {
        "Quan_": "",
        "TinhThanh_": "",
        "Huong_": "Hướng: ",
        "PhapLy_": "Pháp lý: ",
        "Tang_": "Tầng: ",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    name = name.replace("DienTich", "Diện tích")
    name = name.replace("SoPhong", "Số phòng")
    name = name.replace("SoToilet", "Số toilet")
    name = name.replace("SoTang", "Số tầng")
    name = name.replace("ChieuNgang", "Chiều ngang")
    name = name.replace("ChieuDai", "Chiều dài")
    name = name.replace("TangTruong", "Tăng trưởng")

    return name


def create_feature_importance_chart(model, bds_type, algorithm):
    feature_names = _get_feature_names(model, bds_type)
    reg = model.named_steps["regressor"]
    bds_label = BDS_CONFIG[bds_type]["label"]

    if algorithm == "linear":
        values = np.abs(reg.coef_)
        algorithm_label = "Linear Regression"
        title = f"{bds_label} – Linear Regression Coefficients"
    elif algorithm == "random_forest":
        values = reg.feature_importances_
        algorithm_label = "Random Forest"
        title = f"{bds_label} – Random Forest Feature Importance"
    else:
        values = reg.feature_importances_
        algorithm_label = "XGBoost"
        title = f"{bds_label} – XGBoost Feature Importance"

    top = sorted(zip(feature_names, values), key=lambda x: x[1], reverse=True)[:10]
    if not top:
        return None

    labels, vals = zip(*top)
    labels = [_format_feature_label(label) for label in labels]

    filename = _chart_name(f"{bds_type}_{algorithm}_feature_importance")
    path = CHART_DIR / filename

    plt.figure(figsize=(8, 5))
    plt.barh(range(len(labels)), vals)
    plt.yticks(range(len(labels)), labels, fontsize=9)
    plt.gca().invert_yaxis()
    plt.xlabel("Mức độ quan trọng")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

    return f"charts/{filename}"