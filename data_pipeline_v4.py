"""
Claude_Data_v4.py — Dataset Generator v4
==========================================
Improvements over v3:

  1. PREDICT_TERM: 5d → 20d   (consistent with Jiang et al. baseline)
  2. Per-window z-score        (fixes MinMaxScaler data leakage in v3)
  3. 25 features: +ADX, +Stoch_K/D, +CCI  (vs 21 in v3)
  4. Multi-scale 3-channel image: R=5d · G=20d · B=60d  (Jiang-style OHLC)
  5. Relative image paths in CSV (portable, models resolve via base_dir)

Output:
  thesis_data_claudev4/images/<ticker>_<YYYYMMDD>.png
  thesis_data_claudev4/dataset_mapping.csv

Features (25):
  Open, High, Low, Close, Volume
  SMA5, SMA20
  BB_Upper, BB_Lower, BB_Position
  RSI, Williams_R
  MACD, MACD_Signal, MACD_Hist
  OBV
  Donchian_High, Donchian_Low
  ATR, ROC_10, Volume_MA_Ratio
  ADX, Stoch_K, Stoch_D, CCI          ← NEW
"""

import ssl
import time
import os
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
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

START_DATE   = '2000-01-01'   # earlier start: 60-day channel needs warm-up
END_DATE     = '2026-05-01'
WINDOW_SIZE  = 20             # numerical sequence length (days)
PREDICT_TERM = 20             # prediction horizon (days) — matches baseline
MIN_LOOKBACK = 60             # minimum history needed for 60-day image channel
STRIDE       = 1
IMAGE_SIZE   = 224

BASE_DIR   = r'C:\Users\limga\Master Thesis\MasterThesis'
OUTPUT_DIR = os.path.join(BASE_DIR, 'thesis_data_claudev4')
IMAGE_DIR  = os.path.join(OUTPUT_DIR, 'images')
CSV_PATH   = os.path.join(OUTPUT_DIR, 'dataset_mapping.csv')

N_WORKERS = min(6, mp.cpu_count() - 2)

os.makedirs(IMAGE_DIR, exist_ok=True)

NUM_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',        # 0-4
    'SMA5', 'SMA20',                                  # 5-6
    'BB_Upper', 'BB_Lower', 'BB_Position',            # 7-9
    'RSI', 'Williams_R',                              # 10-11
    'MACD', 'MACD_Signal', 'MACD_Hist',               # 12-14
    'OBV',                                            # 15
    'Donchian_High', 'Donchian_Low',                  # 16-17
    'ATR', 'ROC_10', 'Volume_MA_Ratio',               # 18-20
    'ADX', 'Stoch_K', 'Stoch_D', 'CCI',              # 21-24  ← NEW
]
assert len(NUM_COLS) == 25


# ─────────────────────────────────────────────────────────────────────────────
# 2.  INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']

    df['SMA5']  = c.rolling(5).mean()
    df['SMA20'] = c.rolling(20).mean()

    bb_mid            = c.rolling(20).mean()
    bb_std            = c.rolling(20).std()
    df['BB_Upper']    = bb_mid + 2 * bb_std
    df['BB_Lower']    = bb_mid - 2 * bb_std
    df['BB_Position'] = (c - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-9)

    delta     = c.diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - 100 / (1 + gain / (loss + 1e-9))

    highest_h        = h.rolling(14).max()
    lowest_l         = l.rolling(14).min()
    df['Williams_R'] = -100 * (highest_h - c) / (highest_h - lowest_l + 1e-9)

    ema12             = c.ewm(span=12, adjust=False).mean()
    ema26             = c.ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

    df['OBV'] = (np.sign(c.diff()) * v).fillna(0).cumsum()

    df['Donchian_High'] = h.rolling(10).max()
    df['Donchian_Low']  = l.rolling(10).min()

    hl        = h - l
    hpc       = (h - c.shift()).abs()
    lpc       = (l - c.shift()).abs()
    tr        = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    df['ROC_10']          = c.pct_change(10) * 100
    df['Volume_MA_Ratio'] = v / (v.rolling(20).mean() + 1e-9)

    # ── ADX (14) ─────────────────────────────────────────────────────────────
    plus_dm  = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm >= minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm,  0.0)
    atr14    = tr.rolling(14).mean()
    plus_di  = 100 * plus_dm.rolling(14).mean()  / (atr14 + 1e-9)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-9)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df['ADX'] = dx.rolling(14).mean()

    # ── Stochastic %K / %D (14 / 3) ──────────────────────────────────────────
    low14         = l.rolling(14).min()
    high14        = h.rolling(14).max()
    df['Stoch_K'] = 100 * (c - low14) / (high14 - low14 + 1e-9)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # ── CCI (Commodity Channel Index, 20) ────────────────────────────────────
    tp         = (h + l + c) / 3
    tp_ma      = tp.rolling(20).mean()
    tp_md      = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df['CCI']  = (tp - tp_ma) / (0.015 * tp_md + 1e-9)

    return df.bfill().ffill()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  IMAGE RENDERING  (Jiang-style, numpy-only, no matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
def _ohlc_channel(ohlcv: pd.DataFrame, img_h: int = 64) -> Image.Image:
    """
    Renders a single grayscale OHLC chart at 3px/day × img_h.
    Returns PIL 'L' image at native resolution (caller resizes to 224×224).
    """
    n       = len(ohlcv)
    img_w   = n * 3
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
    ma_r = pd.Series(cl_r).rolling(min(n, 20), min_periods=1).mean().values

    all_p = np.concatenate([hi_r, lo_r])
    p_min, p_max = all_p.min(), all_p.max()
    p_rng = p_max - p_min if p_max != p_min else 1e-8
    vol_max = vo.max() if vo.max() > 0 else 1

    def py(r):
        return int(np.clip((r - p_min) / p_rng * (price_h - 1), 0, price_h - 1))

    canvas = np.zeros((img_h, img_w), dtype=np.uint8)

    for d in range(n):
        x = d * 3 + 1
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
            if price_h <= row < img_h:
                canvas[row, x] = 180

    return Image.fromarray(canvas, mode='L')


def render_multiscale(df: pd.DataFrame, end_idx: int,
                      img_size: int = IMAGE_SIZE) -> Image.Image:
    """
    3-channel RGB image encoding three time scales:
      R = 5-day  chart  (short-term momentum)
      G = 20-day chart  (medium-term trend)
      B = 60-day chart  (long-term context)
    All channels resized to img_size × img_size.
    """
    channels = []
    for scale in [5, 20, 60]:
        start  = max(0, end_idx - scale)
        ch     = _ohlc_channel(df.iloc[start:end_idx], img_h=64)
        ch_rs  = ch.resize((img_size, img_size), Image.LANCZOS)
        channels.append(ch_rs)
    return Image.merge('RGB', channels)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PER-TICKER PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def process_ticker(ticker: str) -> list:
    records = []

    df = pd.DataFrame()
    for _ in range(3):
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE,
                             auto_adjust=True, progress=False)
            if not df.empty:
                break
            time.sleep(2)
        except Exception:
            time.sleep(2)

    if df.empty:
        return records

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    if len(df) < MIN_LOOKBACK + PREDICT_TERM + 30:
        return records

    df = calculate_indicators(df)
    df = df.dropna(subset=NUM_COLS)

    if len(df) < MIN_LOOKBACK + PREDICT_TERM:
        return records

    ticker_clean = ticker.replace('.', '_').replace('-', '_')

    for end_idx in range(MIN_LOOKBACK, len(df) - PREDICT_TERM, STRIDE):
        label    = int(df['Close'].iloc[end_idx + PREDICT_TERM] > df['Close'].iloc[end_idx])
        date_str = df.index[end_idx].strftime('%Y%m%d')

        img_name = f"{ticker_clean}_{date_str}.png"
        img_abs  = os.path.join(IMAGE_DIR, img_name)
        img_rel  = os.path.join('thesis_data_claudev4', 'images', img_name)

        if not os.path.exists(img_abs):
            try:
                img = render_multiscale(df, end_idx, IMAGE_SIZE)
                img.save(img_abs)
            except Exception:
                continue

        # per-window z-score — no look-ahead leakage
        window_arr = df[NUM_COLS].iloc[end_idx - WINDOW_SIZE:end_idx].values.astype(np.float32)
        mu  = window_arr.mean(axis=0)
        std = window_arr.std(axis=0) + 1e-8
        seq = np.where(np.isfinite((window_arr - mu) / std),
                       (window_arr - mu) / std, 0.0)
        seq_str = ','.join(f'{v:.6f}' for v in seq.flatten())

        records.append({
            'ticker':        ticker,
            'date':          date_str,
            'image_path':    img_rel,
            'numerical_seq': seq_str,
            'label':         label,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset():
    print(f"\n--- Claude_Data_v4 | {len(TICKERS)} Ticker ---")
    print(f"Features  : {len(NUM_COLS)}  (21 original + ADX + Stoch_K/D + CCI)")
    print(f"Window    : {WINDOW_SIZE}d  |  Horizon: {PREDICT_TERM}d  |  Lookback: {MIN_LOOKBACK}d")
    print(f"Image     : {IMAGE_SIZE}×{IMAGE_SIZE}px RGB  (R=5d · G=20d · B=60d)")
    print(f"Norm      : per-window z-score  (no data leakage)")
    print(f"Workers   : {N_WORKERS}")
    print(f"Output    : {OUTPUT_DIR}\n")

    all_records = []

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_ticker, t): t for t in TICKERS}
        for future in tqdm(as_completed(futures), total=len(TICKERS), desc="Ticker"):
            ticker = futures[future]
            try:
                records = future.result()
                all_records.extend(records)
                print(f"  ✓ {ticker}: {len(records):,} samples")
            except Exception as e:
                print(f"  ✗ {ticker}: {e}")

    if not all_records:
        print("No data generated.")
        return

    out_df = pd.DataFrame(all_records)
    out_df['date'] = pd.to_datetime(out_df['date'])
    out_df = out_df.sort_values('date').reset_index(drop=True)
    out_df['date'] = out_df['date'].dt.strftime('%Y%m%d')
    out_df.to_csv(CSV_PATH, index=False)

    up   = (out_df['label'] == 1).sum()
    down = (out_df['label'] == 0).sum()
    print(f"\n✅  Done: {len(out_df):,} samples")
    print(f"   Up={up:,} ({up/len(out_df)*100:.1f}%)  Down={down:,} ({down/len(out_df)*100:.1f}%)")
    print(f"   CSV: {CSV_PATH}")


if __name__ == '__main__':
    mp.freeze_support()
    build_dataset()
