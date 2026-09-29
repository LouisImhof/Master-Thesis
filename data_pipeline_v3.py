"""
Phase 1 — Dataset Generator  (Windows + CUDA optimiert)
=========================================================
Hardware: AMD Ryzen 7 3700X · 16 GB RAM · RTX 2070 8 GB

Änderungen gegenüber der Mac-Version:
  - Ichimoku Cloud ENTFERNT  (shift(26) = Data Leakage im 20-Tage-Fenster)
  - Parabolic SAR  ENTFERNT  (redundant zu MA, instabil auf kurzen Fenstern)
  - NEU: ATR, Williams %R, ROC-10, BB_Position, Volume_MA_Ratio

Finale 21 Features (identische Dimension, bessere Qualität):
  Open, High, Low, Close, Volume
  SMA5, SMA20
  BB_Upper, BB_Lower, BB_Position
  RSI, Williams_%R
  MACD, MACD_Signal, MACD_Hist
  OBV
  Donchian_High, Donchian_Low
  ATR, ROC_10, Volume_MA_Ratio

Image-Layout (5 Panels, dark theme, 224×224, kein Axes/Labels):
  ┌──────────────────────────────────────┐  58%
  │  Candlestick + BB-Shading            │
  │  SMA5 (cyan) · SMA20 (orange)        │
  │  Donchian Channels (purple)          │
  │  Williams %R als Heatmap-Overlay     │
  ├──────────────────────────────────────┤  13%
  │  Volume × Volume_MA_Ratio (Intensität)│
  ├──────────────────────────────────────┤  10%
  │  RSI  (30/70 Linien)                 │
  ├──────────────────────────────────────┤  10%
  │  MACD Hist (g/r) + MACD + Signal     │
  ├──────────────────────────────────────┤   9%
  │  ATR (Volatilität)                   │
  └──────────────────────────────────────┘

Output:
  thesis_data_v3/images/<TICKER>_<YYYYMMDD>.png
  thesis_data_v3/dataset_mapping.csv
"""

import ssl
import time
import io
import os
import warnings
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# 1.  KONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
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

START_DATE   = '2005-01-01'   # Walk-Forward: Training ab 2005
END_DATE     = '2026-05-01'
WINDOW_SIZE  = 20
PREDICT_TERM = 5
STRIDE       = 1       # FIX: 1 statt 5 → ~5× mehr Samples, Überlappung ist OK
IMAGE_SIZE   = 224

# Windows-Pfade mit Backslash funktionieren, aber forward-slash ist sicherer
OUTPUT_DIR = './thesis_data_claudev3'
IMAGE_DIR  = os.path.join(OUTPUT_DIR, 'images')
CSV_PATH   = os.path.join(OUTPUT_DIR, 'dataset_mapping.csv')

# Auf Windows: CPU-Kerne für parallele Bildgenerierung
# Ryzen 7 3700X hat 8 Kerne / 16 Threads → 6 lassen genug Luft
N_WORKERS = min(6, mp.cpu_count() - 2)

os.makedirs(IMAGE_DIR, exist_ok=True)

# ── Farben ────────────────────────────────────────────────────────────────────
BG      = 'black'
UP_C    = '#26a641'
DOWN_C  = '#e05252'
SMA5_C  = 'cyan'
SMA20_C = 'orange'
BB_C    = '#5b9bd5'
DON_C   = 'purple'
VOL_U   = '#26a641'
VOL_D   = '#e05252'
RSI_C   = 'white'
MACD_C  = 'yellow'
SIG_C   = 'magenta'
HIST_U  = '#4caf50'
HIST_D  = '#f44336'
ATR_C   = '#ff9800'

# ── Feature-Spalten (MUSS 21 sein, identisch mit train_hybrid_v3.py) ──────────
NUM_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',        # 0-4
    'SMA5', 'SMA20',                                  # 5-6
    'BB_Upper', 'BB_Lower', 'BB_Position',            # 7-9
    'RSI', 'Williams_R',                              # 10-11
    'MACD', 'MACD_Signal', 'MACD_Hist',               # 12-14
    'OBV',                                            # 15
    'Donchian_High', 'Donchian_Low',                  # 16-17
    'ATR', 'ROC_10', 'Volume_MA_Ratio',               # 18-20
]
assert len(NUM_COLS) == 21


# ──────────────────────────────────────────────────────────────────────────────
# 2.  INDIKATOREN
# ──────────────────────────────────────────────────────────────────────────────
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']

    # ── Trend ──────────────────────────────────────────────────────────
    df['SMA5']  = c.rolling(5).mean()
    df['SMA20'] = c.rolling(20).mean()

    # ── Bollinger Bands ────────────────────────────────────────────────
    bb_mid         = c.rolling(20).mean()
    bb_std         = c.rolling(20).std()
    df['BB_Upper'] = bb_mid + 2 * bb_std
    df['BB_Lower'] = bb_mid - 2 * bb_std
    bb_range       = df['BB_Upper'] - df['BB_Lower'] + 1e-9
    # Position 0–1: wo im Band befindet sich der Kurs?
    df['BB_Position'] = (c - df['BB_Lower']) / bb_range

    # ── RSI (14) ───────────────────────────────────────────────────────
    delta       = c.diff()
    gain        = delta.clip(lower=0).rolling(14).mean()
    loss        = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI']   = 100 - 100 / (1 + gain / (loss + 1e-9))

    # ── Williams %R (14) — Komplementär zu RSI ─────────────────────────
    highest_h       = h.rolling(14).max()
    lowest_l        = l.rolling(14).min()
    df['Williams_R'] = -100 * (highest_h - c) / (highest_h - lowest_l + 1e-9)

    # ── MACD (12/26/9) ─────────────────────────────────────────────────
    ema12              = c.ewm(span=12, adjust=False).mean()
    ema26              = c.ewm(span=26, adjust=False).mean()
    df['MACD']         = ema12 - ema26
    df['MACD_Signal']  = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']    = df['MACD'] - df['MACD_Signal']

    # ── OBV ────────────────────────────────────────────────────────────
    df['OBV'] = (np.sign(c.diff()) * v).fillna(0).cumsum()

    # ── Donchian Channels (10) ─────────────────────────────────────────
    df['Donchian_High'] = h.rolling(10).max()
    df['Donchian_Low']  = l.rolling(10).min()

    # ── ATR (14) — Average True Range ──────────────────────────────────
    hl  = h - l
    hpc = (h - c.shift()).abs()
    lpc = (l - c.shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    # ── ROC-10 — Rate of Change ─────────────────────────────────────────
    df['ROC_10'] = c.pct_change(10) * 100

    # ── Volume MA Ratio ────────────────────────────────────────────────
    vol_ma20              = v.rolling(20).mean()
    df['Volume_MA_Ratio'] = v / (vol_ma20 + 1e-9)

    return df.bfill().ffill()


def generate_labels(df: pd.DataFrame, lookforward: int = 5) -> pd.DataFrame:
    df['Future_Close'] = df['Close'].shift(-lookforward)
    df['Label']        = (df['Future_Close'] > df['Close']).astype(int)
    return df.dropna(subset=['Future_Close', 'Label'])


# ──────────────────────────────────────────────────────────────────────────────
# 3.  BILD-RENDERER
# ──────────────────────────────────────────────────────────────────────────────
def render_chart(window: pd.DataFrame) -> Image.Image:
    """224×224 RGB dunkles Komposit-Chart für ein WINDOW_SIZE-Fenster."""
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100, facecolor=BG)
    gs  = gridspec.GridSpec(
        5, 1,
        height_ratios=[0.58, 0.13, 0.10, 0.10, 0.09],
        hspace=0, figure=fig
    )
    ax_p = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1], sharex=ax_p)
    ax_r = fig.add_subplot(gs[2], sharex=ax_p)
    ax_m = fig.add_subplot(gs[3], sharex=ax_p)
    ax_a = fig.add_subplot(gs[4], sharex=ax_p)

    x = np.arange(len(window))

    # ── Preis-Panel ───────────────────────────────────────────────────────
    ax_p.set_facecolor(BG)

    # Bollinger Band Shading
    ax_p.fill_between(x, window['BB_Lower'], window['BB_Upper'],
                      color=BB_C, alpha=0.12)
    ax_p.plot(x, window['BB_Upper'], color=BB_C, lw=0.4, alpha=0.6)
    ax_p.plot(x, window['BB_Lower'], color=BB_C, lw=0.4, alpha=0.6)

    # Donchian Channels
    ax_p.plot(x, window['Donchian_High'], color=DON_C, lw=0.6, alpha=0.8)
    ax_p.plot(x, window['Donchian_Low'],  color=DON_C, lw=0.6, alpha=0.8)

    # Williams %R als subtiles Heatmap-Overlay auf Preis-Background
    # (normalisiert auf 0–1, bläulich wenn oversold, rötlich wenn overbought)
    wr_norm = (window['Williams_R'] + 100) / 100   # 0 = oversold, 1 = overbought
    price_min = window['Low'].min()
    price_max = window['High'].max()
    for i, wr in enumerate(wr_norm):
        alpha = 0.06
        col   = '#e05252' if wr > 0.8 else ('#26a641' if wr < 0.2 else None)
        if col:
            ax_p.axvspan(i - 0.5, i + 0.5,
                         ymin=0, ymax=1, color=col, alpha=alpha, zorder=0)

    # Kerzen
    w_body = 0.6
    for i, (_, row) in enumerate(window.iterrows()):
        o, h_, lo, c = row['Open'], row['High'], row['Low'], row['Close']
        col = UP_C if c >= o else DOWN_C
        ax_p.plot([i, i], [lo, h_], color=col, lw=0.7, zorder=1)
        body_lo = min(o, c)
        body_h  = max(abs(c - o), (h_ - lo) * 0.008 + 1e-9)
        ax_p.add_patch(Rectangle((i - w_body/2, body_lo), w_body, body_h,
                                  color=col, zorder=2))

    # Moving Averages
    ax_p.plot(x, window['SMA5'],  color=SMA5_C,  lw=0.8, alpha=0.9)
    ax_p.plot(x, window['SMA20'], color=SMA20_C, lw=0.8, alpha=0.9)

    # ── Volumen-Panel (Intensität = Volume_MA_Ratio) ───────────────────────
    ax_v.set_facecolor(BG)
    for i in range(len(window)):
        col   = UP_C if window['Close'].iloc[i] >= window['Open'].iloc[i] else DOWN_C
        ratio = min(window['Volume_MA_Ratio'].iloc[i], 3.0) / 3.0  # cap bei 3×
        ax_v.bar(i, window['Volume'].iloc[i],
                 color=col, width=0.8, alpha=0.4 + 0.6 * ratio)

    # ── RSI-Panel ─────────────────────────────────────────────────────────
    ax_r.set_facecolor(BG)
    ax_r.plot(x, window['RSI'], color=RSI_C, lw=0.7)
    ax_r.axhline(70, color='white', lw=0.3, ls='--', alpha=0.4)
    ax_r.axhline(30, color='white', lw=0.3, ls='--', alpha=0.4)
    ax_r.set_ylim(0, 100)

    # ── MACD-Panel ────────────────────────────────────────────────────────
    ax_m.set_facecolor(BG)
    hcols = [HIST_U if v >= 0 else HIST_D for v in window['MACD_Hist']]
    ax_m.bar(x, window['MACD_Hist'], color=hcols, width=0.8, alpha=0.75)
    ax_m.plot(x, window['MACD'],        color=MACD_C, lw=0.7)
    ax_m.plot(x, window['MACD_Signal'], color=SIG_C,  lw=0.7)
    ax_m.axhline(0, color='white', lw=0.3, alpha=0.3)

    # ── ATR-Panel ─────────────────────────────────────────────────────────
    ax_a.set_facecolor(BG)
    ax_a.plot(x, window['ATR'], color=ATR_C, lw=0.7)
    ax_a.fill_between(x, 0, window['ATR'], color=ATR_C, alpha=0.15)

    # ── Alle Achsen entfernen ─────────────────────────────────────────────
    for ax in [ax_p, ax_v, ax_r, ax_m, ax_a]:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.margins(x=0)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor=BG,
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB').resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    buf.close()
    return img


# ──────────────────────────────────────────────────────────────────────────────
# 4.  TICKER VERARBEITUNG  (für Multiprocessing serialisierbar)
# ──────────────────────────────────────────────────────────────────────────────
def process_ticker(ticker: str) -> list:
    """
    Verarbeitet einen Ticker komplett: Download → Indikatoren → Fenster.
    Gibt Liste von Metadata-Dicts zurück.
    Läuft in separatem Prozess (Ryzen Multicore-Nutzung).
    """
    records = []

    # Download mit 3 Versuchen
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

    # Scaler auf voller Ticker-History fitten
    scaler     = MinMaxScaler()
    scaled_arr = scaler.fit_transform(df[NUM_COLS].values)

    for i in range(0, len(df) - WINDOW_SIZE, STRIDE):
        end_idx = i + WINDOW_SIZE
        if end_idx > len(df):
            break

        label    = int(df['Label'].iloc[end_idx - 1])
        date_str = df.index[end_idx - 1].strftime('%Y%m%d')

        img_name = f"{ticker.replace('.', '_')}_{date_str}.png"
        img_path = os.path.join(IMAGE_DIR, img_name)

        if not os.path.exists(img_path):
            try:
                img = render_chart(df.iloc[i:end_idx].copy())
                img.save(img_path)
            except Exception as e:
                continue

        seq_str = ','.join(f'{v:.6f}' for v in
                           np.where(np.isfinite(scaled_arr[i:end_idx].flatten()),
                                    scaled_arr[i:end_idx].flatten(), 0.0))

        records.append({
            'ticker':        ticker,
            'date':          date_str,
            'image_path':    img_path,
            'numerical_seq': seq_str,
            'label':         label,
        })

    return records


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ──────────────────────────────────────────────────────────────────────────────
def build_dataset():
    print(f"\n--- Phase 1 (v3 Windows): {len(TICKERS)} Ticker ---")
    print(f"Parallele Worker: {N_WORKERS}  (Ryzen 7 3700X)")

    all_records = []

    # Parallele Verarbeitung mit ProcessPoolExecutor
    # Auf Windows muss das unter if __name__ == '__main__' laufen (spawn)
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_ticker, t): t for t in TICKERS}
        for future in tqdm(as_completed(futures), total=len(TICKERS),
                           desc="Ticker verarbeitet"):
            ticker  = futures[future]
            try:
                records = future.result()
                all_records.extend(records)
                print(f"  ✓ {ticker}: {len(records)} Fenster")
            except Exception as e:
                print(f"  ✗ {ticker}: {e}")

    if not all_records:
        print("❌  Keine Daten generiert.")
        return

    out_df = pd.DataFrame(all_records)
    out_df['date'] = pd.to_datetime(out_df['date'])
    out_df = out_df.sort_values('date').reset_index(drop=True)
    out_df['date'] = out_df['date'].dt.strftime('%Y%m%d')
    out_df.to_csv(CSV_PATH, index=False)

    up   = (out_df['label'] == 1).sum()
    down = (out_df['label'] == 0).sum()
    print(f"\n✅  Fertig — {len(out_df):,} Samples gespeichert")
    print(f"   Labels:  Up={up:,} ({up/len(out_df)*100:.1f}%)  "
          f"Down={down:,} ({down/len(out_df)*100:.1f}%)")
    print(f"   Gespeichert: {CSV_PATH}")


if __name__ == '__main__':
    # Windows benötigt diesen Guard für ProcessPoolExecutor (spawn statt fork)
    mp.freeze_support()
    build_dataset()
