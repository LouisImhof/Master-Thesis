"""
generate_v3_fixed_csv.py
Regenerates the v3 dataset CSV with two fixes applied:
  1. PREDICT_TERM 5 → 20  (matches v4 and baseline)
  2. Per-window z-score    (replaces MinMaxScaler fitted on full history)

Images are NOT re-rendered — they already exist in thesis_data_claudev3/images/.
Only the CSV (labels + numerical sequences) is regenerated.

Output: thesis_data_claudev3_fixed/dataset_mapping.csv
"""

import ssl
import time
import os
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
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

START_DATE   = '2000-01-01'
END_DATE     = '2026-05-01'
WINDOW_SIZE  = 20
PREDICT_TERM = 20        # FIX: was 5 in v3
STRIDE       = 1

BASE_DIR      = r'C:\Users\limga\Master Thesis\MasterThesis'
IMAGE_SRC_DIR = os.path.join(BASE_DIR, 'thesis_data_claudev3', 'images')  # existing
OUTPUT_DIR    = os.path.join(BASE_DIR, 'thesis_data_claudev3_fixed')
CSV_PATH      = os.path.join(OUTPUT_DIR, 'dataset_mapping.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

N_WORKERS = min(6, mp.cpu_count() - 2)

NUM_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA5', 'SMA20',
    'BB_Upper', 'BB_Lower', 'BB_Position',
    'RSI', 'Williams_R',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'OBV',
    'Donchian_High', 'Donchian_Low',
    'ATR', 'ROC_10', 'Volume_MA_Ratio',
]
assert len(NUM_COLS) == 21


# ── INDICATORS ────────────────────────────────────────────────────────────────
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']

    df['SMA5']  = c.rolling(5).mean()
    df['SMA20'] = c.rolling(20).mean()

    bb_mid         = c.rolling(20).mean()
    bb_std         = c.rolling(20).std()
    df['BB_Upper'] = bb_mid + 2 * bb_std
    df['BB_Lower'] = bb_mid - 2 * bb_std
    bb_range       = df['BB_Upper'] - df['BB_Lower'] + 1e-9
    df['BB_Position'] = (c - df['BB_Lower']) / bb_range

    delta     = c.diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - 100 / (1 + gain / (loss + 1e-9))

    highest_h       = h.rolling(14).max()
    lowest_l        = l.rolling(14).min()
    df['Williams_R'] = -100 * (highest_h - c) / (highest_h - lowest_l + 1e-9)

    ema12             = c.ewm(span=12, adjust=False).mean()
    ema26             = c.ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

    df['OBV'] = (np.sign(c.diff()) * v).fillna(0).cumsum()

    df['Donchian_High'] = h.rolling(10).max()
    df['Donchian_Low']  = l.rolling(10).min()

    hl  = h - l
    hpc = (h - c.shift()).abs()
    lpc = (l - c.shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    df['ROC_10'] = c.pct_change(10) * 100

    vol_ma20              = v.rolling(20).mean()
    df['Volume_MA_Ratio'] = v / (vol_ma20 + 1e-9)

    return df.bfill().ffill()


def generate_labels(df: pd.DataFrame, lookforward: int = 20) -> pd.DataFrame:
    df['Future_Close'] = df['Close'].shift(-lookforward)
    df['Label']        = (df['Future_Close'] > df['Close']).astype(int)
    return df.dropna(subset=['Future_Close', 'Label'])


# ── TICKER PROCESSOR ─────────────────────────────────────────────────────────
def process_ticker(ticker: str) -> list:
    records = []

    df = pd.DataFrame()
    for attempt in range(3):
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

    min_rows = WINDOW_SIZE + PREDICT_TERM + 26 + 10
    if len(df) < min_rows:
        return records

    df = calculate_indicators(df)
    df = generate_labels(df, lookforward=PREDICT_TERM)
    df = df.dropna(subset=NUM_COLS)

    if len(df) < WINDOW_SIZE + PREDICT_TERM:
        return records

    raw_arr = df[NUM_COLS].values  # no global scaler — z-score applied per window

    for i in range(0, len(df) - WINDOW_SIZE, STRIDE):
        end_idx = i + WINDOW_SIZE
        if end_idx > len(df):
            break

        label    = int(df['Label'].iloc[end_idx - 1])
        date_str = df.index[end_idx - 1].strftime('%Y%m%d')

        img_name = f"{ticker.replace('.', '_')}_{date_str}.png"
        img_path = os.path.join(IMAGE_SRC_DIR, img_name)

        if not os.path.exists(img_path):
            continue  # skip windows where image wasn't rendered in v3

        # Per-window z-score (fixes MinMaxScaler data leakage from v3)
        window_raw = raw_arr[i:end_idx].copy()  # (20, 21)
        mean = window_raw.mean(axis=0)
        std  = window_raw.std(axis=0)
        std[std < 1e-8] = 1e-8
        window_z = (window_raw - mean) / std

        seq_str = ','.join(
            f'{v:.6f}' for v in
            np.where(np.isfinite(window_z.flatten()), window_z.flatten(), 0.0)
        )

        records.append({
            'ticker':        ticker,
            'date':          date_str,
            'image_path':    img_path,
            'numerical_seq': seq_str,
            'label':         label,
        })

    return records


# ── MAIN ──────────────────────────────────────────────────────────────────────
def build_dataset():
    print(f"\n--- generate_v3_fixed_csv: {len(TICKERS)} tickers ---")
    print(f"  PREDICT_TERM : {PREDICT_TERM}d  (was 5 in v3)")
    print(f"  Normalisation: per-window z-score  (was MinMaxScaler)")
    print(f"  Images from  : {IMAGE_SRC_DIR}")
    print(f"  CSV out      : {CSV_PATH}")
    print(f"  Workers      : {N_WORKERS}\n")

    all_records = []

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_ticker, t): t for t in TICKERS}
        for future in tqdm(as_completed(futures), total=len(TICKERS),
                           desc='Tickers'):
            ticker = futures[future]
            try:
                records = future.result()
                all_records.extend(records)
                print(f"  {ticker}: {len(records):,} windows")
            except Exception as e:
                print(f"  {ticker}: FAILED — {e}")

    if not all_records:
        print("ERROR: no records generated.")
        return

    df_out = pd.DataFrame(all_records)
    df_out.to_csv(CSV_PATH, index=False)

    n  = len(df_out)
    up = df_out['label'].sum()
    print(f"\nSaved {n:,} samples  UP={up:,} ({up/n*100:.1f}%)  DOWN={n-up:,} ({(n-up)/n*100:.1f}%)")
    print(f"CSV → {CSV_PATH}")


if __name__ == '__main__':
    build_dataset()
