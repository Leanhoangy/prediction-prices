╔══════════════════════════════════════════════════════════════════╗
║          DỰ ÁN DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN – TÀI LIỆU KỸ THUẬT   ║
╚══════════════════════════════════════════════════════════════════╝

=======================================================================
1. TỔNG QUAN PROJECT
=======================================================================

Mục tiêu  : Dự đoán giá bất động sản (chung cư, nhà riêng, đất nền)
            dựa trên các đặc trưng như diện tích, quận, số phòng...
Công nghệ : Python, Flask, scikit-learn, XGBoost, pandas, matplotlib
Dữ liệu   : Crawl từ mogi.vn (~1000 tin/loại BĐS)

Luồng chạy chính:
  crawler.py  →  datasets/*.xlsx  →  train_models.py  →  models/*.pkl
                                                              ↓
                                                app.py (Flask web)

=======================================================================
2. CẤU TRÚC THƯ MỤC
=======================================================================

project/
├── app.py                  # Khởi động web Flask
├── config.py               # Cấu hình trung tâm (đường dẫn, features)
├── crawler.py              # Thu thập dữ liệu từ mogi.vn
├── preprocessing.py        # Làm sạch & kiểm tra dữ liệu
├── training.py             # Xây dựng và đánh giá model
├── train_models.py         # Script chạy độc lập để train
├── visualization.py        # Vẽ biểu đồ
├── routes.py               # Các trang web Flask
├── file_utils.py           # Tiện ích đọc file, mapping cột
│
├── datasets/               # Dữ liệu Excel
│   ├── chungcu.xlsx
│   ├── nha.xlsx
│   └── dat.xlsx
│
├── models/
│   ├── production/         # Model đang dùng (best, linear, rf, xgb)
│   ├── staging/            # Model mới upload, chờ kiểm duyệt
│   └── archive/            # Model cũ được backup tự động
│
├── static/charts/          # Biểu đồ PNG (EDA, actual vs pred, ...)
├── templates/              # Giao diện HTML (Jinja2)
├── reports/                # Kết quả training (Excel)
└── uploads/                # File upload tạm

=======================================================================
3. CHI TIẾT TỪNG FILE VÀ HÀM
=======================================================================

───────────────────────────────────────────────────────────────────────
FILE: config.py
Vai trò: Cấu hình trung tâm, tất cả file khác import từ đây
───────────────────────────────────────────────────────────────────────

  Biến/Hằng số:
  ┌─────────────────────┬──────────────────────────────────────────────┐
  │ BASE_DIR            │ Thư mục gốc của project                      │
  │ DATASET_DIR         │ datasets/                                    │
  │ CHART_DIR           │ static/charts/                               │
  │ MODEL_PRODUCTION_DIR│ models/production/                           │
  │ MODEL_STAGING_DIR   │ models/staging/                              │
  │ MODEL_ARCHIVE_DIR   │ models/archive/                              │
  │ REPORT_DIR          │ reports/                                     │
  └─────────────────────┴──────────────────────────────────────────────┘

  BDS_CONFIG (dict):
    Cấu hình riêng cho mỗi loại BĐS: chungcu, nha, dat.
    Mỗi loại gồm:
    - label      : Tên hiển thị ("Chung cư", "Nhà", "Đất")
    - dataset    : Đường dẫn file Excel
    - required   : Danh sách cột bắt buộc phải có
    - numeric    : Cột số dùng làm features (DienTich, SoPhong...)
    - categorical: Cột chữ dùng làm features (Quan, Huong, PhapLy...)
    - rename     : Map tên cột Excel gốc → tên chuẩn nội bộ
    - *_model    : Đường dẫn file .pkl của từng model

  COLUMN_LABELS (dict):
    Map tên cột nội bộ → tên hiển thị tiếng Việt.
    Dùng để hiện thị trên giao diện web và biểu đồ.

  COLUMN_ALIASES (dict):
    Map tên cột chuẩn → danh sách tên khác có thể có trong file upload.
    Dùng trong chức năng auto-detect cột khi upload Excel.

───────────────────────────────────────────────────────────────────────
FILE: preprocessing.py
Vai trò: Làm sạch dữ liệu trước khi đưa vào train model
───────────────────────────────────────────────────────────────────────

  clean_price_value(x)
    Input : 1 giá trị giá (có thể là "3,2 tỷ", "1500 triệu", 1500...)
    Output: float (triệu đồng) hoặc NaN nếu không parse được
    Ví dụ : "3,2 tỷ" → 3200.0, "500 triệu" → 500.0

  to_numeric_series(series)
    Input : pandas Series kiểu chuỗi (vd: "50 m²", "2,5")
    Output: pandas Series kiểu float
    Dùng cho: cột DienTich, SoPhong, SoTang, SoToilet...
    Xử lý  : xóa "m2", "m²", đổi dấu phẩy → chấm, extract số

  remove_iqr_outliers(df, cols)
    Input : DataFrame df và danh sách cột cần lọc
    Output: (df_sạch, dict_từng_cột_đã_xóa_bao_nhiêu, tổng_xóa)
    Cách làm:
      Q1 = percentile 25%
      Q3 = percentile 75%
      IQR = Q3 - Q1
      Giữ dòng có: Q1 - 1.5*IQR ≤ giá trị ≤ Q3 + 1.5*IQR

  validate_and_clean_dataset(df, bds_type)   ← HÀM CHÍNH
    Input : DataFrame đã rename cột, loại BĐS ("chungcu"/"nha"/"dat")
    Output: (errors, warnings, df_sạch, report)
    Thực hiện tuần tự:
      Bước 1: Kiểm tra đủ cột bắt buộc
      Bước 2: Xóa dòng có NaN (dropna)
      Bước 3: Chuyển cột số về float (clean_price_value, to_numeric_series)
      Bước 4: Chuẩn hóa tên tỉnh/thành (Hồ Chí Minh, Hà Nội...)
      Bước 5: Xóa dòng không hợp lệ (giá ≤ 0, diện tích ≤ 0, chuỗi rỗng)
      Bước 6: Xóa dòng trùng (drop_duplicates)
      Bước 7: Loại outlier bằng IQR
    report trả về: {original_rows, missing_removed, invalid_removed,
                    duplicate_removed, outlier_removed, final_rows}

───────────────────────────────────────────────────────────────────────
FILE: training.py
Vai trò: Xây dựng, train và đánh giá model Machine Learning
───────────────────────────────────────────────────────────────────────

  build_model_pipeline(bds_type, algorithm, verbose=0)
    Input : loại BĐS, tên model ("linear"/"random_forest"/"xgboost")
    Output: sklearn Pipeline gồm 2 bước:
      - preprocessor: ColumnTransformer
          + numeric  → passthrough (giữ nguyên)
          + categorical → OneHotEncoder (mã hóa cột chữ thành 0/1)
      - regressor: LinearRegression / RandomForestRegressor / XGBRegressor
    Tham số model:
      Random Forest: n_estimators=200, max_depth=12
      XGBoost      : n_estimators=300, max_depth=6, learning_rate=0.1

  calc_metrics(model, X_test, y_test)
    Input : model đã train, tập test
    Output: (dict metrics, y_pred)
    Metrics tính: MAE, MSE, RMSE, R²

  cross_validate_pipeline(bds_type, algorithm, X, y, k=5)
    Input : loại BĐS, tên model, toàn bộ X và y, số fold
    Output: {R2_mean, R2_std, MAE_mean, MAE_std}
    Dùng KFold shuffle để đánh giá ổn định hơn test set 1 lần

  compare_algorithms_and_select_best(df, bds_type)   ← Dùng trong web
    Input : DataFrame đã làm sạch, loại BĐS
    Output: dict chứa tất cả model, metrics, charts và model tốt nhất
    So sánh: chọn model có R² cao nhất, nếu bằng thì chọn MAE thấp hơn

  train_one_algorithm(bds_type, algorithm, X_train, X_test, y_train, y_test)
    Input : loại BĐS, tên model, dữ liệu train/test
    Output: (model, metrics, charts)

  evaluate_existing_model(path, X_test, y_test)
    Đánh giá model đã lưu trên tập test mới → dùng để so sánh với model mới

  is_new_model_better(old_metrics, new_metrics)
    So sánh: mới tốt hơn khi R² >= cũ VÀ MAE <= cũ

  archive_if_exists(path, bds_type, suffix)
    Copy model hiện tại vào models/archive/ trước khi bị ghi đè

  get_dropdown_options()
    Đọc dataset của từng loại BĐS, trả về giá trị unique cho:
    Quận/Huyện, Hướng, Pháp lý, Tỉnh thành
    Dùng để render dropdown trên trang Dự đoán

  append_history(record)
    Ghi thêm kết quả train vào reports/train_history.json

  merge_dataset_keep_original_columns(old_path, new_raw_df)
    Gộp dataset mới upload vào dataset hiện có, xóa trùng, lưu lại

───────────────────────────────────────────────────────────────────────
FILE: train_models.py
Vai trò: Script chạy độc lập từ terminal để train toàn bộ
         python train_models.py
───────────────────────────────────────────────────────────────────────

  cleanup_old_charts()
    Xóa tất cả PNG trong static/charts/ trước khi train
    Giữ lại: pipeline.png, pipeline_slide.png, model_comparison_r2.png

  train_one_dataset(bds_type, config)   ← HÀM CHÍNH
    Thực hiện đầy đủ cho 1 loại BĐS:
      1. Đọc Excel → rename cột
      2. validate_and_clean_dataset() → làm sạch
      3. Tách train/test (80%/20%, random_state=42)
      4. Train Linear Regression → tính metrics
      5. Train Random Forest (200 cây) → tính metrics
      6. Train XGBoost (300 cây) → tính metrics
      7. Vẽ Actual vs Predicted chart cho cả 3 model
      8. Vẽ Feature Importance chart cho cả 3 model
      9. Chạy 5-fold cross-validation cho cả 3 model
     10. Chọn best model (R² cao nhất, MAE thấp nhất)
     11. Lưu .pkl: linear_*.pkl, rf_*.pkl, xgb_*.pkl, best_*.pkl
    Output: dict kết quả để ghi vào Excel báo cáo

  main()
    Gọi train_one_dataset() cho chungcu, nha, dat
    Lưu kết quả ra reports/training_results.xlsx

───────────────────────────────────────────────────────────────────────
FILE: visualization.py
Vai trò: Tạo và lưu tất cả biểu đồ PNG vào static/charts/
───────────────────────────────────────────────────────────────────────

  _chart_name(prefix)
    Tạo tên file unique dạng: prefix_YYYYMMDD_HHMMSS_xxxxxx.png
    Tránh trùng tên khi train nhiều lần

  analyze_dataframe(df)
    Tính thống kê mô tả: số dòng, số cột, missing values, mean/std/min/max
    Dùng trên trang EDA của web

  create_actual_vs_pred_chart(y_test, y_pred, title)
    Vẽ scatter plot: trục X = giá thực tế, trục Y = giá dự đoán
    Đường chéo 45° = dự đoán hoàn hảo → điểm càng gần đường = càng tốt

  create_feature_importance_chart(model, bds_type, algorithm)
    Vẽ horizontal bar chart: top 10 features quan trọng nhất
    Linear: dùng |coef_| (hệ số tuyệt đối)
    RF/XGBoost: dùng feature_importances_ (tỷ lệ đóng góp)

  create_correlation_heatmap(df, bds_type)
    Vẽ ma trận tương quan giữa các cột số
    Giá trị từ -1 (nghịch chiều) đến +1 (cùng chiều)

  create_eda_charts(df, bds_type)   ← Tạo 5 biểu đồ cùng lúc
    1. price_hist  : Histogram phân phối giá
    2. area_hist   : Histogram phân phối diện tích
    3. scatter     : Scatter giá vs diện tích (có jitter chống chồng điểm)
    4. quan_bar    : Bar chart top 10 quận theo giá trung bình
    5. huong_box   : Boxplot giá theo hướng (6 hướng phổ biến nhất)

  _get_feature_names(model, bds_type)
    Lấy tên đầy đủ các feature sau OneHotEncoder
    VD: ["DienTich", "SoPhong", "Quan_Quận 1", "Quan_Quận 2", ...]

───────────────────────────────────────────────────────────────────────
FILE: file_utils.py
Vai trò: Tiện ích xử lý file và mapping cột
───────────────────────────────────────────────────────────────────────

  read_file(path)
    Đọc file .xlsx hoặc .csv → trả về DataFrame
    Tự phát hiện định dạng qua đuôi file

  save_json(path, data)
    Lưu dict/list Python ra file JSON (UTF-8, có indent)

  guess_column(system_col, user_columns)
    Input : tên cột chuẩn hệ thống, danh sách cột từ file upload
    Output: tên cột trong file upload khớp nhất
    Cách so sánh: duyệt COLUMN_ALIASES, so khớp gần đúng (chứa nhau)
    Ví dụ: "DienTich" sẽ khớp với "diện tích", "Diện Tích (m2)", "dt"

  apply_mapping(df, mapping)
    Input : DataFrame và dict {tên_chuẩn: tên_trong_file}
    Output: DataFrame đã đổi tên cột về chuẩn nội bộ

───────────────────────────────────────────────────────────────────────
FILE: crawler.py
Vai trò: Thu thập dữ liệu BĐS từ mogi.vn (không dùng Selenium)
───────────────────────────────────────────────────────────────────────

  Chạy: python crawler.py --type chungcu --pages 70

  parse_price(text)
    Chuyển "2 tỷ 50 triệu" → 2050.0 (triệu đồng)
    Xử lý các dạng: "5 tỷ", "850 triệu", "2 tỷ 300 triệu"

  parse_number(text)
    Trích số đầu tiên từ chuỗi: "82 m2" → 82.0, "2 PN" → 2.0

  parse_location(addr_text)
    Input : "Quận 7, TPHCM"
    Output: (quan="Quận 7", tinh="Hồ Chí Minh")

  fetch_page(url)
    GET request đến mogi.vn, trả về BeautifulSoup object
    Dùng requests thường (không Selenium) vì mogi.vn không có Cloudflare

  parse_cards(soup, bds_type)
    Parse tất cả listing cards trong 1 trang listing
    Selectors dùng: .re__card-config-price, .re__card-config-area,
                    .re__card-config-bedroom, .prop-attr li, .re__card-location
    Trả về list các dict (1 dict = 1 tin BĐS)

  fill_missing_from_existing(df, bds_type)
    Sau khi crawl, các cột Hướng/Pháp lý/Tầng không có trên listing card
    → Điền ngẫu nhiên từ phân phối của dataset hiện có
    Giữ nguyên: Giá, Diện tích, Số phòng, Quận (là dữ liệu thật)

  main()
    Chạy vòng lặp qua từng trang (page 1: URL gốc, page N: URL?cp=N)
    Auto-save mỗi 100 tin, fill missing, lưu Excel cuối

───────────────────────────────────────────────────────────────────────
FILE: routes.py
Vai trò: Định nghĩa các URL và xử lý request của web Flask
───────────────────────────────────────────────────────────────────────

  /  (dashboard)
    Đọc trạng thái từng loại BĐS: số dòng dataset, model nào đã train
    Detect tên best model từ class của regressor trong pkl
    Render dashboard.html

  /predict  (GET + POST)
    GET : Hiển thị form nhập thông tin, dropdown từ get_dropdown_options()
    POST: Đọc input → load best_*.pkl → model.predict() → hiển thị kết quả

  /eda/<bds_type>
    Đọc dataset → validate_and_clean_dataset() → analyze_dataframe()
    Tạo EDA charts (cache theo mtime file để không vẽ lại nếu data không đổi)
    Render eda.html với stats và biểu đồ

  /upload  (GET + POST)
    GET : Form chọn loại BĐS và upload file
    POST: Đọc file → guess_column() → hiển thị mapping cột để user xác nhận

  /map  (POST)
    Nhận mapping cột đã xác nhận → apply_mapping() → lưu vào staging

  /train  (POST)
    Lấy dataset đã staging → validate → compare_algorithms_and_select_best()
    Nếu model mới tốt hơn: archive model cũ → lưu model mới vào production
    Lưu lịch sử vào train_history.json

  _eda_cache (dict module-level)
    Cache kết quả EDA theo key (bds_type, dataset_mtime)
    Tránh vẽ lại biểu đồ mỗi lần load trang nếu data chưa thay đổi

───────────────────────────────────────────────────────────────────────
FILE: app.py
Vai trò: Entry point, khởi động Flask server
───────────────────────────────────────────────────────────────────────

  Đăng ký Blueprint từ routes.py
  Chạy: python app.py → http://localhost:5000

=======================================================================
4. HƯỚNG DẪN CHẠY
=======================================================================

  Bước 1 – Cài thư viện:
    pip install -r requirements.txt

  Bước 2 – (Tuỳ chọn) Crawl data mới:
    python crawler.py --type chungcu --pages 70
    python crawler.py --type nha     --pages 70
    python crawler.py --type dat     --pages 70

  Bước 3 – Train model:
    python train_models.py
    → Sinh ra models/production/*.pkl và reports/training_results.xlsx

  Bước 4 – Chạy web:
    python app.py
    → Mở http://localhost:5000

=======================================================================
5. KẾT QUẢ MODEL (data thực tế từ mogi.vn)
=======================================================================

  Loại BĐS │ Best Model    │ R² test │ R² CV (5-fold)
  ──────────┼───────────────┼─────────┼──────────────────
  Chung cư  │ Random Forest │  0.38   │ 0.50 ± 0.09
  Nhà riêng │ XGBoost       │  0.70   │ 0.72 ± 0.01
  Đất nền   │ Random Forest │  0.60   │ 0.61 ± 0.06

  Ghi chú: R² = 0.70 trên data thực tế là tốt (industry benchmark ~0.65)
           R² chung cư thấp hơn do ít data sau khi lọc outlier

=======================================================================
6. GIẢI THÍCH CÁC METRICS
=======================================================================

  R² (R-squared)
    Đo % biến động giá mà model giải thích được.
    R²=1.0 → dự đoán hoàn hảo | R²=0 → dự đoán bằng mean | R²<0 → tệ hơn mean

  MAE (Mean Absolute Error)
    Sai số tuyệt đối trung bình (đơn vị: triệu đồng)
    MAE=1386 triệu → trung bình mỗi dự đoán sai ~1.4 tỷ

  RMSE (Root Mean Squared Error)
    Nhạy cảm với outlier hơn MAE (phạt nặng sai số lớn)

  Cross-Validation 5-fold
    Chia data thành 5 phần → train 4, test 1, xoay vòng 5 lần
    Kết quả ổn định hơn test 1 lần duy nhất

  IQR Outlier Removal
    Xóa dòng ngoài khoảng [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    Không bị ảnh hưởng bởi giá trị cực đoan (robust hơn mean ± 2σ)

=======================================================================
7. THƯ VIỆN SỬ DỤNG
=======================================================================

  scikit-learn  : Pipeline, LinearRegression, RandomForestRegressor,
                  OneHotEncoder, train_test_split, cross_val_score
  xgboost       : XGBRegressor (gradient boosting)
  pandas        : Đọc/ghi Excel/CSV, xử lý bảng dữ liệu
  numpy         : Tính toán số học, IQR
  flask         : Web framework
  matplotlib    : Vẽ biểu đồ PNG
  requests      : HTTP request để crawl mogi.vn
  beautifulsoup4: Parse HTML từ web
  joblib        : Lưu/load file model .pkl
