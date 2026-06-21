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


def _chart_name(prefix):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"


def create_actual_vs_pred_chart(y_test, y_pred, title):
    filename = _chart_name("actual_pred")
    path = CHART_DIR / filename

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    min_val = min(float(np.min(y_test)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_test)), float(np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Giá thực tế")
    plt.ylabel("Giá dự đoán")
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
    fn = _chart_name("eda_price_hist")
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
    fn = _chart_name("eda_area_hist")
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
    fn = _chart_name("eda_scatter")
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
    avg_quan = df.groupby("Quan")["Gia"].mean().sort_values(ascending=False).head(10)
    fn = _chart_name("eda_quan_bar")
    plt.figure(figsize=(9, 5))
    plt.barh(range(len(avg_quan)), avg_quan.values, color="#2563eb")
    plt.yticks(range(len(avg_quan)), avg_quan.index)
    plt.gca().invert_yaxis()
    plt.xlabel("Giá trung bình (triệu đồng)")
    plt.title("Top 10 quận/huyện theo giá trung bình")
    plt.tight_layout()
    plt.savefig(CHART_DIR / fn, dpi=140)
    plt.close()
    charts["quan_bar"] = f"charts/{fn}"

    # 5. Boxplot giá theo hướng (top 6 phổ biến nhất)
    top_huong = df["Huong"].value_counts().head(6).index.tolist()
    groups = [df[df["Huong"] == h]["Gia"].values for h in top_huong]
    fn = _chart_name("eda_huong_box")
    plt.figure(figsize=(9, 5))
    plt.boxplot(groups, labels=top_huong, patch_artist=True,
                boxprops=dict(facecolor="#dbeafe"), medianprops=dict(color="#1d4ed8", linewidth=2))
    plt.ylabel("Giá (triệu đồng)")
    plt.title("Phân phối giá theo hướng")
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


def create_feature_importance_chart(model, bds_type, algorithm):
    feature_names = _get_feature_names(model, bds_type)
    reg = model.named_steps["regressor"]

    if algorithm == "linear":
        values = np.abs(reg.coef_)
        title = "Linear Regression Coefficients"
    elif algorithm == "random_forest":
        values = reg.feature_importances_
        title = "Random Forest Feature Importance"
    else:
        values = reg.feature_importances_
        title = "XGBoost Feature Importance"

    top = sorted(zip(feature_names, values), key=lambda x: x[1], reverse=True)[:10]
    if not top:
        return None

    labels, vals = zip(*top)
    filename = _chart_name("feature_importance")
    path = CHART_DIR / filename

    plt.figure(figsize=(8, 5))
    plt.barh(range(len(labels)), vals)
    plt.yticks(range(len(labels)), labels)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

    return f"charts/{filename}"
