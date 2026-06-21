from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"
CHART_DIR = BASE_DIR / "static" / "charts"
MODEL_PRODUCTION_DIR = BASE_DIR / "models" / "production"
MODEL_STAGING_DIR = BASE_DIR / "models" / "staging"
MODEL_ARCHIVE_DIR = BASE_DIR / "models" / "archive"

for _folder in [DATASET_DIR, UPLOAD_DIR, REPORT_DIR, CHART_DIR,
                MODEL_PRODUCTION_DIR, MODEL_STAGING_DIR, MODEL_ARCHIVE_DIR]:
    _folder.mkdir(parents=True, exist_ok=True)

BDS_CONFIG = {
    "chungcu": {
        "label": "Chung cư",
        "dataset": DATASET_DIR / "chungcu.xlsx",
        "best_model": MODEL_PRODUCTION_DIR / "best_chungcu.pkl",
        "linear_model": MODEL_PRODUCTION_DIR / "linear_chungcu.pkl",
        "rf_model": MODEL_PRODUCTION_DIR / "rf_chungcu.pkl",
        "xgb_model": MODEL_PRODUCTION_DIR / "xgb_chungcu.pkl",
        "required": ["Gia", "DienTich", "SoPhong", "SoToilet", "Tang", "Quan", "Huong", "PhapLy"],
        "numeric": ["DienTich", "SoPhong", "SoToilet", "Tang"],
        "categorical": ["Quan", "Huong", "PhapLy"],
        "rename": {
            "GIÁ - TRIỆU ĐỒNG": "Gia",
            "DIỆN TÍCH - M2": "DienTich",
            "SỐ PHÒNG": "SoPhong",
            "SỐ TOILETS": "SoToilet",
            "QUẬN HUYỆN": "Quan",
            "HƯỚNG": "Huong",
            "GIẤY TỜ PHÁP LÝ": "PhapLy",
            "TẦNG": "Tang",
        },
    },
    "nha": {
        "label": "Nhà",
        "dataset": DATASET_DIR / "nha.xlsx",
        "best_model": MODEL_PRODUCTION_DIR / "best_nha.pkl",
        "linear_model": MODEL_PRODUCTION_DIR / "linear_nha.pkl",
        "rf_model": MODEL_PRODUCTION_DIR / "rf_nha.pkl",
        "xgb_model": MODEL_PRODUCTION_DIR / "xgb_nha.pkl",
        "required": ["Gia", "DienTich", "SoTang", "SoPhong", "Quan", "Huong", "PhapLy", "TinhThanh"],
        "numeric": ["DienTich", "SoTang", "SoPhong"],
        "categorical": ["Quan", "Huong", "PhapLy", "TinhThanh"],
        "rename": {
            "Giá (triệu đồng)": "Gia",
            "Diện tích (m2)": "DienTich",
            "Số tầng": "SoTang",
            "Số phòng": "SoPhong",
            "Quận/Huyện": "Quan",
            "Hướng": "Huong",
            "Giấy tờ pháp lý": "PhapLy",
            "Tỉnh thành": "TinhThanh",
        },
    },
    "dat": {
        "label": "Đất",
        "dataset": DATASET_DIR / "dat.xlsx",
        "best_model": MODEL_PRODUCTION_DIR / "best_dat.pkl",
        "linear_model": MODEL_PRODUCTION_DIR / "linear_dat.pkl",
        "rf_model": MODEL_PRODUCTION_DIR / "rf_dat.pkl",
        "xgb_model": MODEL_PRODUCTION_DIR / "xgb_dat.pkl",
        "required": ["Gia", "DienTich", "ChieuNgang", "ChieuDai", "Quan", "Huong", "PhapLy", "TinhThanh", "TangTruong"],
        "numeric": ["DienTich", "ChieuNgang", "ChieuDai", "TangTruong"],
        "categorical": ["Quan", "Huong", "PhapLy", "TinhThanh"],
        "rename": {
            "Giá( triệu)": "Gia",
            "Diện tích": "DienTich",
            "Chiều ngang": "ChieuNgang",
            "Chiều dài": "ChieuDai",
            "Quận": "Quan",
            "Hướng": "Huong",
            "Pháp lý": "PhapLy",
            "Tỉnh": "TinhThanh",
            "Tăng trưởng": "TangTruong",
        },
    },
}

COLUMN_LABELS = {
    "Gia": "Giá (triệu đồng)",
    "DienTich": "Diện tích (m²)",
    "SoPhong": "Số phòng",
    "SoToilet": "Số toilet",
    "SoTang": "Số tầng",
    "ChieuNgang": "Chiều ngang (m)",
    "ChieuDai": "Chiều dài (m)",
    "Quan": "Quận/Huyện",
    "Huong": "Hướng",
    "PhapLy": "Pháp lý",
    "TinhThanh": "Tỉnh/Thành phố",
    "TangTruong": "Tăng trưởng (%)",
    "Tang": "Số tầng",
}

COLUMN_ALIASES = {
    "Gia": ["gia", "giá", "price", "gia ban", "giá bán", "giá( triệu)", "giá (triệu đồng)", "giá - triệu đồng"],
    "DienTich": ["dien tich", "diện tích", "area", "dt", "diện tích - m2", "diện tích (m2)"],
    "SoPhong": ["so phong", "số phòng", "bedroom", "bedrooms"],
    "SoToilet": ["so toilet", "số toilets", "số toilet", "wc", "toilet"],
    "SoTang": ["so tang", "số tầng", "floor", "floors"],
    "ChieuNgang": ["chieu ngang", "chiều ngang", "ngang", "width"],
    "ChieuDai": ["chieu dai", "chiều dài", "dai", "length"],
    "Quan": ["quan", "quận", "quận huyện", "quận/huyện", "district"],
    "Huong": ["huong", "hướng", "direction"],
    "PhapLy": ["phap ly", "pháp lý", "giấy tờ pháp lý", "legal"],
}
