"""
crawler.py
==========
Crawl dữ liệu BĐS từ mogi.vn bằng requests (không cần Selenium).

Cách chạy:
    python crawler.py --type chungcu --pages 20
    python crawler.py --type nha     --pages 20
    python crawler.py --type dat     --pages 20

Kết quả lưu vào:
    datasets/crawled_chungcu.xlsx
    datasets/crawled_nha.xlsx
    datasets/crawled_dat.xlsx
"""

import argparse
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BDS_URLS = {
    "chungcu": "https://mogi.vn/mua-can-ho-chung-cu",
    "nha":     "https://mogi.vn/mua-nha",
    "dat":     "https://mogi.vn/mua-dat",
}

OUTPUT_DIR = Path("datasets")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
}

# Chuẩn hoá tên tỉnh/thành
_TINH_MAP = {
    "tphcm": "Hồ Chí Minh",
    "tp.hcm": "Hồ Chí Minh",
    "tp hcm": "Hồ Chí Minh",
    "hcm": "Hồ Chí Minh",
    "hồ chí minh": "Hồ Chí Minh",
    "hà nội": "Hà Nội",
    "ha noi": "Hà Nội",
    "đà nẵng": "Đà Nẵng",
    "da nang": "Đà Nẵng",
    "bình dương": "Bình Dương",
    "binh duong": "Bình Dương",
    "đồng nai": "Đồng Nai",
    "long an": "Long An",
    "khánh hòa": "Khánh Hòa",
    "nha trang": "Khánh Hòa",
    "cần thơ": "Cần Thơ",
    "hải phòng": "Hải Phòng",
}


def parse_price(text):
    """'2 tỷ 50 triệu' → 2050.0  |  '5 tỷ' → 5000.0  |  '850 triệu' → 850.0"""
    text = (text or "").lower().replace("\xa0", " ").strip()
    ty    = re.search(r"([\d\.,]+)\s*tỷ", text)
    trieu = re.search(r"([\d\.,]+)\s*triệu", text)
    result = 0.0
    if ty:
        result += float(ty.group(1).replace(",", ".").replace(".", "")) * 1000
        # "5 tỷ" có thể là "5,000" → cần normalize
        raw = ty.group(1).replace(",", "")
        result = float(raw) * 1000
    if trieu:
        result += float(trieu.group(1).replace(",", ""))
    return round(result, 1) if result > 0 else None


def parse_number(text):
    """'82 m2' → 82.0  |  '2 PN' → 2.0"""
    m = re.search(r"([\d]+(?:[,\.]\d+)?)", (text or ""))
    return float(m.group(1).replace(",", ".")) if m else None


def parse_location(addr_text):
    """
    'Quận 7, TPHCM'              → (quan='Quận 7',   tinh='Hồ Chí Minh')
    'Quận Gò Vấp, TPHCM'         → (quan='Gò Vấp',   tinh='Hồ Chí Minh')
    'Hà Nội'                     → (quan='',          tinh='Hà Nội')
    """
    addr_text = (addr_text or "").strip()
    parts = [p.strip() for p in addr_text.split(",")]

    tinh = ""
    if len(parts) >= 2:
        tinh_raw = parts[-1].lower().strip()
        tinh = _TINH_MAP.get(tinh_raw, parts[-1].strip())
    elif parts:
        raw = parts[0].lower().strip()
        tinh = _TINH_MAP.get(raw, parts[0].strip())

    quan = ""
    if len(parts) >= 2:
        quan_raw = parts[0].strip()
        # Bỏ prefix "Quận " nếu có số (VD: "Quận 7" → "Quận 7", "Quận Gò Vấp" → "Gò Vấp")
        m = re.match(r"Quận\s+(\d+)", quan_raw)
        if m:
            quan = "Quận " + m.group(1)
        else:
            quan = re.sub(r"^(Quận|Huyện|Thị xã|Thành phố)\s+", "", quan_raw)
        # Bỏ phần "(TP. Thủ Đức)" v.v.
        quan = re.sub(r"\s*\(.*?\)", "", quan).strip()

    return quan, tinh


def fetch_page(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"  [WARN] Lỗi tải {url}: {e}")
        return None


def parse_cards(soup, bds_type):
    rows = []
    for card in soup.select(".prop-info"):
        price_el  = card.select_one(".price")
        attr_li   = card.select(".prop-attr li")
        addr_el   = card.select_one(".prop-addr")

        gia       = parse_price(price_el.get_text(strip=True) if price_el else "")
        dien_tich = parse_number(attr_li[0].get_text(strip=True) if attr_li else "")
        so_phong  = parse_number(attr_li[1].get_text(strip=True) if len(attr_li) > 1 else "")
        so_toilet = parse_number(attr_li[2].get_text(strip=True) if len(attr_li) > 2 else "")
        quan, tinh = parse_location(addr_el.get_text(strip=True) if addr_el else "")

        if not gia or not dien_tich:
            continue

        if bds_type == "chungcu":
            rows.append({
                "GIÁ - TRIỆU ĐỒNG": gia,
                "DIỆN TÍCH - M2":    dien_tich,
                "SỐ PHÒNG":          so_phong,
                "SỐ TOILETS":        so_toilet,
                "QUẬN HUYỆN":        quan,
                "HƯỚNG":             "",
                "GIẤY TỜ PHÁP LÝ":  "",
                "TẦNG":              None,
            })
        elif bds_type == "nha":
            rows.append({
                "Giá (triệu đồng)": gia,
                "Diện tích (m2)":   dien_tich,
                "Số tầng":          None,
                "Số phòng":         so_phong,
                "Quận/Huyện":       quan,
                "Hướng":            "",
                "Giấy tờ pháp lý":  "",
                "Tỉnh thành":       tinh,
            })
        else:  # dat
            rows.append({
                "Giá( triệu)":  gia,
                "Diện tích":    dien_tich,
                "Chiều ngang":  None,
                "Chiều dài":    None,
                "Quận":         quan,
                "Hướng":        "",
                "Pháp lý":      "",
                "Tỉnh":         tinh,
            })
    return rows


def fill_missing_from_existing(df, bds_type):
    """
    Điền các cột thiếu (None/NaN hoặc hoàn toàn vắng mặt) bằng giá trị
    ngẫu nhiên lấy từ phân phối của dataset gốc. Giữ nguyên Gia/DienTich/Quan
    là giá trị thực crawled.
    """
    existing_path = OUTPUT_DIR / f"{bds_type}.xlsx"
    if not existing_path.exists():
        print("  [WARN] Không tìm thấy dataset gốc, bỏ qua fill.")
        return df

    from config import BDS_CONFIG
    config = BDS_CONFIG[bds_type]
    existing = pd.read_excel(existing_path).rename(columns=config.get("rename", {}))
    # df vẫn dùng raw column names → rename trước
    df = df.rename(columns=config.get("rename", {}))

    rng = random.Random(42)

    for col in config["numeric"] + config["categorical"]:
        if col == "Gia":
            continue
        if col not in df.columns:
            df[col] = None  # thêm cột bị thiếu hoàn toàn
        null_mask = df[col].isna()
        if not null_mask.any():
            continue
        if col not in existing.columns:
            continue
        pool = existing[col].dropna().tolist()
        if not pool:
            continue
        df[col] = df[col].astype(object)
        df.loc[null_mask, col] = [rng.choice(pool) for _ in range(null_mask.sum())]
        print(f"  fill '{col}': {null_mask.sum()} giá trị")

    # Đổi lại về raw column names để lưu file
    reverse_rename = {v: k for k, v in config.get("rename", {}).items()}
    return df.rename(columns=reverse_rename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",  default="chungcu", choices=["chungcu", "nha", "dat"])
    parser.add_argument("--pages", type=int, default=10)
    parser.add_argument("--no-fill", action="store_true",
                        help="Không điền giá trị thiếu từ dataset gốc")
    args = parser.parse_args()

    base_url = BDS_URLS[args.type]
    bds_type = args.type
    out_path = OUTPUT_DIR / f"crawled_{bds_type}.xlsx"

    print(f"\n{'='*55}")
    print(f"Crawl: {bds_type.upper()}  |  {args.pages} trang  |  mogi.vn")
    print(f"{'='*55}")

    records = []

    for page in range(1, args.pages + 1):
        url = base_url if page == 1 else f"{base_url}?cp={page}"
        print(f"\n[Trang {page}/{args.pages}] {url}")

        soup = fetch_page(url)
        if not soup:
            continue

        rows = parse_cards(soup, bds_type)
        print(f"  Parse được {len(rows)} tin hợp lệ")

        if not rows and page > 2:
            print("  Hết dữ liệu — dừng lại.")
            break

        records.extend(rows)
        for r in rows[:3]:
            vals = list(r.values())
            print(f"    Giá={vals[0]}tr  DT={vals[1]}m²  Quận={vals[4] if len(vals)>4 else ''}")

        if len(records) % 100 == 0 and records:
            pd.DataFrame(records).to_excel(out_path, index=False)
            print(f"  [Auto-save] {len(records)} tin → {out_path}")

        time.sleep(random.uniform(0.8, 1.5))

    if records:
        df = pd.DataFrame(records)
        if not args.no_fill:
            print("\n[Fill] Điền giá trị thiếu từ dataset gốc...")
            df = fill_missing_from_existing(df, bds_type)
        df.to_excel(out_path, index=False)
        print(f"\nHoàn tất! Đã lưu {len(records)} tin → {out_path}")
        print(df.head(5).to_string())
    else:
        print("\nKhông lấy được dữ liệu nào.")


if __name__ == "__main__":
    main()
