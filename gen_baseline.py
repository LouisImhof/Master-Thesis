# gen_baseline.py - Jiang, Kelly & Xiu (2021) image generator
# Downloads data, then creates 60×64 grayscale OHLC images (I20R20)
# 3-way thesis split: Train <2019 | Val 2019-2021 | OOS >=2021
import yfinance as yf
import pandas as pd
import numpy as np
import os
import shutil
import time
from PIL import Image

# ── TICKERS ───────────────────────────────────────────────────────────────────
TICKERS = [
    'ASML.AS', 'MC.PA',   'TTE.PA',  'SAP.DE',   'SIE.DE',
    'SAN.PA',  'OR.PA',   'SU.PA',   'AI.PA',    'ALV.DE',
    'AIR.PA',  'RMS.PA',  'IBE.MC',  'DTE.DE',   'DG.PA',
    'BNP.PA',  'MBG.DE',  'SAN.MC',  'SAF.PA',   'EL.PA',
    'CS.PA',   'BAYN.DE', 'IFX.DE',  'PRX.AS',   'ABI.BR',
    'ENEL.MI', 'MUV2.DE', 'INGA.AS', 'ADYEN.AS', 'DHL.DE',
    'BBVA.MC', 'BAS.DE',  'RI.PA',   'ISP.MI',   'ITX.MC',
    'KER.PA',  'UCG.MI',  'STLAM.MI','CRH.L',    'NDA-FI.HE',
    'BMW.DE',  'BN.PA',   'FLTR.L',  'DB1.DE',   'ENI.MI',
    'AD.AS',   'ADS.DE',  'VOW3.DE', 'NOKIA.HE', 'VNA.DE',
]

# ── CONFIG ────────────────────────────────────────────────────────────────────
DAYS      = 20
HORIZON   = 20
IMG_W     = 60    # 3px per day × 20 days
IMG_H     = 64

TRAIN_END = pd.Timestamp('2018-12-31')
VAL_END   = pd.Timestamp('2020-12-31')
START     = '2000-01-01'
OUT_DIR   = r'C:\Users\limga\Master Thesis\MasterThesis\thesis_data_baseline\images'
CSV_PATH  = r'C:\Users\limga\Master Thesis\MasterThesis\thesis_data_baseline\baseline_tickers.csv'

# ── DOWNLOAD ──────────────────────────────────────────────────────────────────
os.makedirs(r'C:\Users\limga\Master Thesis\MasterThesis\thesis_data_baseline', exist_ok=True)

if os.path.exists(CSV_PATH):
    print(f"Loading cached data from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
else:
    print(f"Downloading {len(TICKERS)} tickers from {START} ...")
    rows = []
    for i, ticker in enumerate(TICKERS):
        try:
            stock = yf.download(ticker, start=START, progress=False, auto_adjust=True)
            if isinstance(stock.columns, pd.MultiIndex):
                stock.columns = stock.columns.get_level_values(0)
            if not stock.empty:
                stock = stock[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                stock['Ticker'] = ticker
                rows.append(stock)
                print(f"  [{i+1:2d}/{len(TICKERS)}] {ticker}: {len(stock):,} rows")
            else:
                print(f"  [{i+1:2d}/{len(TICKERS)}] {ticker}: EMPTY")
        except Exception as e:
            print(f"  [{i+1:2d}/{len(TICKERS)}] {ticker}: SKIP ({e})")
        time.sleep(0.1)

    df = pd.concat(rows)
    df.index.name = 'Date'
    df.to_csv(CSV_PATH)
    print(f"Saved {CSV_PATH}  Shape: {df.shape}")

df.index.name = 'Date'
df.index = pd.to_datetime(df.index)
print(f"Loaded: {df.shape}, {df['Ticker'].nunique()} tickers")


# ── IMAGE GENERATION ──────────────────────────────────────────────────────────
def generate_image(ohlcv, days, img_w, img_h):
    price_h = int(img_h * 0.8)
    vol_h   = img_h - price_h

    cl = ohlcv['Close'].values
    hi = ohlcv['High'].values
    lo = ohlcv['Low'].values
    op = ohlcv['Open'].values
    vo = ohlcv['Volume'].values

    base = cl[0] if cl[0] != 0 else 1e-8
    cl_r = cl / base - 1
    hi_r = hi / base - 1
    lo_r = lo / base - 1
    op_r = op / base - 1
    ma_r = pd.Series(cl_r).rolling(days, min_periods=1).mean().values

    all_p = np.concatenate([hi_r, lo_r])
    p_min, p_max = all_p.min(), all_p.max()
    p_rng = p_max - p_min if p_max != p_min else 1e-8
    vol_max = vo.max() if vo.max() > 0 else 1

    def py(r):
        return int(np.clip((r - p_min) / p_rng * (price_h - 1), 0, price_h - 1))

    canvas = np.zeros((img_h, img_w), dtype=np.uint8)

    for d in range(days):
        x    = d * 3 + 1
        y_hi = py(hi_r[d]); y_lo = py(lo_r[d])
        y_op = py(op_r[d]); y_cl = py(cl_r[d])
        y_ma = py(ma_r[d])

        for y in range(y_lo, y_hi + 1):
            canvas[price_h - 1 - y, x] = 255
        if x - 1 >= 0:
            canvas[price_h - 1 - y_op, x - 1] = 255
        if x + 1 < img_w:
            canvas[price_h - 1 - y_cl, x + 1] = 255
        canvas[price_h - 1 - y_ma, x] = 200

        v_h = int((vo[d] / vol_max) * (vol_h - 1))
        for y in range(v_h):
            row = img_h - 1 - y
            if row >= price_h:
                canvas[row, x] = 180

    return Image.fromarray(canvas, mode='L')


# ── GENERATE SPLITS ───────────────────────────────────────────────────────────
for split in ['train', 'val', 'test']:
    folder = f"{OUT_DIR}/{split}"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

train_n = val_n = test_n = 0

for ticker, group in df.groupby('Ticker'):
    group = group.sort_index().dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    if len(group) < DAYS + HORIZON:
        continue

    for i in range(DAYS, len(group) - HORIZON):
        window = group.iloc[i - DAYS : i]
        date   = group.index[i]
        label  = int(group['Close'].iloc[i + HORIZON] > group['Close'].iloc[i])

        if date <= TRAIN_END:
            split = 'train'
        elif date <= VAL_END:
            split = 'val'
        else:
            split = 'test'

        img = generate_image(window, DAYS, IMG_W, IMG_H)
        ticker_clean = ticker.replace('-', '_').replace('.', '_')
        fname = f"{OUT_DIR}/{split}/{ticker_clean}_{date.strftime('%Y%m%d')}_{label}.png"
        img.save(fname)

        if split == 'train':   train_n += 1
        elif split == 'val':   val_n   += 1
        else:                  test_n  += 1

    print(f"  {ticker}: done", end='\r')

print(f"\nDone:")
print(f"  Train (<= 2018) : {train_n:,}")
print(f"  Val  (2019-20)  : {val_n:,}")
print(f"  OOS  (>= 2021)  : {test_n:,}")
print(f"  Total           : {train_n+val_n+test_n:,}")
print(f"Saved to {OUT_DIR}/")
