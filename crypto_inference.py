"""
crypto_inference.py — H4: Transfer Learning to Cryptocurrency Markets
=======================================================================
Tests whether Euro Stoxx 50 trained models retain predictive power
when applied to Bitcoin and Ethereum (no retraining — pure zero-shot transfer).

Pipeline:
  1. Download BTC-USD / ETH-USD OHLCV (from 2018 for indicator warm-up)
  2. Compute same 25 technical indicators as equity training data
  3. Render same image formats:
       - 60×64 grayscale OHLC  (for Baseline JiangCNN)
       - 224×224 RGB multi-scale R=5d/G=20d/B=60d  (for EfficientNet + ConvNeXt)
  4. Load equity normalization stats from training CSV (no leakage)
  5. Load trained model weights — NO retraining
  6. Run OOS inference on 2021-01-01 to 2026-05-01
  7. Save prediction CSVs + print AUC per model

Outputs: thesis_data_crypto/
  baseline_crypto_predictions.csv
  efficientnet_crypto_predictions.csv
  convnext_crypto_predictions.csv
  ensemble_crypto_predictions.csv
"""

import os
import ssl
import math
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from sklearn.metrics import roc_auc_score

try:
    import timm
except ImportError:
    timm = None

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = r'C:\Users\limga\Master Thesis\MasterThesis'

CRYPTO_TICKERS  = ['BTC-USD', 'ETH-USD']
DOWNLOAD_START  = '2018-01-01'   # 60-day warm-up needed for indicators
OOS_START       = '2021-01-01'
OOS_END         = '2026-05-01'

WINDOW_SIZE     = 20    # sequence length (days)
PREDICT_TERM    = 20    # prediction horizon (days) — matches equity training
MIN_LOOKBACK    = 60    # minimum history for 60-day image channel
IMAGE_SIZE      = 224

TRAIN_END = '2018-12-31'   # equity train split end — for normalization stats

EQUITY_CSV_PATH     = os.path.join(BASE_DIR, 'thesis_data_claudev4', 'dataset_mapping.csv')
BASELINE_MODEL_DIR  = os.path.join(BASE_DIR, 'thesis_data_baseline', 'models')
EFFNET_MODEL_PATH   = os.path.join(BASE_DIR, 'thesis_data_EfficientNetB2_Transformer', 'best_model.pth')
CONVNEXT_MODEL_PATH = os.path.join(BASE_DIR, 'thesis_data_ConvNeXtV2_iTransformer', 'best_model.pth')

OUTPUT_DIR       = os.path.join(BASE_DIR, 'thesis_data_crypto')
CRYPTO_RGB_DIR   = os.path.join(OUTPUT_DIR, 'images_rgb')
CRYPTO_GRAY_DIR  = os.path.join(OUTPUT_DIR, 'images_grayscale')

os.makedirs(CRYPTO_RGB_DIR,  exist_ok=True)
os.makedirs(CRYPTO_GRAY_DIR, exist_ok=True)

NUM_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA5', 'SMA20',
    'BB_Upper', 'BB_Lower', 'BB_Position',
    'RSI', 'Williams_R',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'OBV',
    'Donchian_High', 'Donchian_Low',
    'ATR', 'ROC_10', 'Volume_MA_Ratio',
    'ADX', 'Stoch_K', 'Stoch_D', 'CCI',
]
assert len(NUM_COLS) == 25


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TECHNICAL INDICATORS  (identical to Claude_Data_v4.py)
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

    hl  = h - l
    hpc = (h - c.shift()).abs()
    lpc = (l - c.shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    df['ROC_10']          = c.pct_change(10) * 100
    df['Volume_MA_Ratio'] = v / (v.rolling(20).mean() + 1e-9)

    plus_dm  = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm >= minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm,  0.0)
    atr14    = tr.rolling(14).mean()
    plus_di  = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-9)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-9)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df['ADX'] = dx.rolling(14).mean()

    low14        = l.rolling(14).min()
    high14       = h.rolling(14).max()
    df['Stoch_K'] = 100 * (c - low14) / (high14 - low14 + 1e-9)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    tp        = (h + l + c) / 3
    tp_ma     = tp.rolling(20).mean()
    tp_md     = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df['CCI'] = (tp - tp_ma) / (0.015 * tp_md + 1e-9)

    return df.bfill().ffill()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  IMAGE RENDERING
# ─────────────────────────────────────────────────────────────────────────────
def _ohlc_channel(ohlcv: pd.DataFrame, img_h: int = 64) -> Image.Image:
    n       = len(ohlcv)
    img_w   = n * 3
    price_h = int(img_h * 0.8)
    vol_h   = img_h - price_h

    cl = ohlcv['Close'].values;  hi = ohlcv['High'].values
    lo = ohlcv['Low'].values;    op = ohlcv['Open'].values
    vo = ohlcv['Volume'].values

    base  = cl[0] if cl[0] != 0 else 1e-8
    cl_r  = cl / base - 1;  hi_r = hi / base - 1
    lo_r  = lo / base - 1;  op_r = op / base - 1
    ma_r  = pd.Series(cl_r).rolling(min(n, 20), min_periods=1).mean().values

    all_p             = np.concatenate([hi_r, lo_r])
    p_min, p_max      = all_p.min(), all_p.max()
    p_rng             = p_max - p_min if p_max != p_min else 1e-8
    vol_max           = vo.max() if vo.max() > 0 else 1

    def py(r): return int(np.clip((r - p_min) / p_rng * (price_h - 1), 0, price_h - 1))

    canvas = np.zeros((img_h, img_w), dtype=np.uint8)
    for d in range(n):
        x    = d * 3 + 1
        y_hi = py(hi_r[d]);  y_lo = py(lo_r[d])
        y_op = py(op_r[d]);  y_cl = py(cl_r[d])
        y_ma = py(ma_r[d])
        for y in range(y_lo, y_hi + 1):
            canvas[price_h - 1 - y, x] = 255
        if x - 1 >= 0:    canvas[price_h - 1 - y_op, x - 1] = 255
        if x + 1 < img_w: canvas[price_h - 1 - y_cl, x + 1] = 255
        canvas[price_h - 1 - y_ma, x] = 200
        v_h = int((vo[d] / vol_max) * (vol_h - 1))
        for y in range(v_h):
            row = img_h - 1 - y
            if price_h <= row < img_h:
                canvas[row, x] = 180

    return Image.fromarray(canvas, mode='L')


def render_multiscale_rgb(df: pd.DataFrame, end_idx: int) -> Image.Image:
    """R=5d, G=20d, B=60d — identical to Claude_Data_v4.py."""
    channels = []
    for scale in [5, 20, 60]:
        start = max(0, end_idx - scale)
        ch    = _ohlc_channel(df.iloc[start:end_idx], img_h=64)
        ch_rs = ch.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        channels.append(ch_rs)
    return Image.merge('RGB', channels)


def render_grayscale_20d(ohlcv: pd.DataFrame) -> Image.Image:
    """60×64 grayscale OHLC — identical to gen_baseline.py generate_image()."""
    return _ohlc_channel(ohlcv, img_h=64)   # n=20 → img_w=60 automatically


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DATA DOWNLOAD & SAMPLE GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def build_crypto_samples():
    all_records = []

    for ticker in CRYPTO_TICKERS:
        print(f"\nDownloading {ticker} ...")
        df = pd.DataFrame()
        for attempt in range(3):
            try:
                df = yf.download(ticker, start=DOWNLOAD_START, end=OOS_END,
                                 auto_adjust=True, progress=False)
                if not df.empty:
                    break
                time.sleep(2)
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}")
                time.sleep(2)

        if df.empty:
            print(f"  SKIP {ticker}: no data")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        if len(df) < MIN_LOOKBACK + PREDICT_TERM + 30:
            print(f"  SKIP {ticker}: only {len(df)} rows")
            continue

        df = calculate_indicators(df)
        df = df.dropna(subset=NUM_COLS)
        print(f"  {ticker}: {len(df):,} rows after indicators")

        ticker_clean = ticker.replace('-', '_').replace('.', '_')
        count = 0

        for end_idx in range(MIN_LOOKBACK, len(df) - PREDICT_TERM):
            date     = df.index[end_idx]
            date_str = date.strftime('%Y%m%d')

            if date < pd.Timestamp(OOS_START):
                continue

            label = int(df['Close'].iloc[end_idx + PREDICT_TERM] > df['Close'].iloc[end_idx])

            # RGB multi-scale image (for advanced models)
            img_rgb_path = os.path.join(CRYPTO_RGB_DIR, f"{ticker_clean}_{date_str}.png")
            if not os.path.exists(img_rgb_path):
                try:
                    img = render_multiscale_rgb(df, end_idx)
                    img.save(img_rgb_path)
                except Exception:
                    continue

            # Grayscale OHLC image (for baseline model)
            img_gray_path = os.path.join(CRYPTO_GRAY_DIR, f"{ticker_clean}_{date_str}.png")
            if not os.path.exists(img_gray_path):
                try:
                    window_20 = df.iloc[end_idx - 20:end_idx]
                    if len(window_20) < 20:
                        continue
                    img_gray = render_grayscale_20d(window_20)
                    img_gray.save(img_gray_path)
                except Exception:
                    continue

            # Per-window z-score sequence (same as Claude_Data_v4.py)
            window_arr = df[NUM_COLS].iloc[end_idx - WINDOW_SIZE:end_idx].values.astype(np.float32)
            mu  = window_arr.mean(axis=0)
            std = window_arr.std(axis=0) + 1e-8
            seq = np.where(np.isfinite((window_arr - mu) / std),
                           (window_arr - mu) / std, 0.0)

            all_records.append({
                'ticker':        ticker,
                'date':          date_str,
                'label':         label,
                'img_rgb_path':  img_rgb_path,
                'img_gray_path': img_gray_path,
                'seq':           seq.astype(np.float32),
            })
            count += 1

        print(f"  {ticker}: {count:,} OOS samples (2021–2026)")

    return all_records


# ─────────────────────────────────────────────────────────────────────────────
# 5.  EQUITY NORMALIZATION STATS
# ─────────────────────────────────────────────────────────────────────────────
def load_equity_norm_stats():
    """
    Recomputes the train-split normalization applied during equity model training.
    Applied on top of per-window z-score — must match Claude_Model_*.py exactly.
    """
    if not os.path.exists(EQUITY_CSV_PATH):
        print(f"  Equity CSV not found: {EQUITY_CSV_PATH}")
        print("  Skipping global normalization (per-window z-score only).")
        return None, None

    print("Loading equity train-split for normalization stats ...")
    df = pd.read_csv(EQUITY_CSV_PATH)

    dates      = pd.to_datetime(df['date'].astype(str))
    train_mask = dates <= pd.Timestamp(TRAIN_END)
    train_seqs_str = df.loc[train_mask, 'numerical_seq']

    print(f"  Parsing {len(train_seqs_str):,} train sequences ...")
    seq_array = np.stack([
        np.fromstring(s, sep=',', dtype=np.float32).reshape(WINDOW_SIZE, 25)
        for s in tqdm(train_seqs_str, ncols=80)
    ])  # (N_train, 20, 25)

    seq_mean = seq_array.mean(axis=(0, 1))     # (25,)
    seq_std  = seq_array.std(axis=(0, 1)) + 1e-8
    print(f"  Done. Mean range: [{seq_mean.min():.3f}, {seq_mean.max():.3f}]")
    return seq_mean, seq_std


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MODEL ARCHITECTURES  (must match training exactly)
# ─────────────────────────────────────────────────────────────────────────────

# ── Baseline: JiangCNN ────────────────────────────────────────────────────────
class JiangCNN(nn.Module):
    def __init__(self, in_ch=1, img_h=64, img_w=60):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64,  (5, 3), stride=(3, 1), dilation=(2, 1), padding=(7, 1)),
            nn.BatchNorm2d(64),  nn.LeakyReLU(0.01), nn.MaxPool2d((2, 1)),
            nn.Conv2d(64,  128, (5, 3), padding=(2, 1)),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.01), nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, (5, 3), padding=(2, 1)),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.01), nn.MaxPool2d((2, 1)),
        )
        with torch.no_grad():
            fc_in = self.features(torch.zeros(1, in_ch, img_h, img_w)).flatten(1).shape[1]
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(fc_in, 2))

    def forward(self, x):
        return self.classifier(self.features(x))


# ── EfficientNet-B2 + Transformer (HybridModelV4) ────────────────────────────
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid())
        self.sp = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = x * self.ch(x).view(x.size(0), -1, 1, 1)
        return x * self.sp(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1))


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=100, drop=0.1):
        super().__init__()
        self.drop = nn.Dropout(drop)
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class SequenceTransformer(nn.Module):
    def __init__(self, n_feat=25, d=128, heads=8, layers=3, drop=0.1):
        super().__init__()
        self.proj    = nn.Linear(n_feat, d)
        self.pos     = PositionalEncoding(d, drop=drop)
        enc          = nn.TransformerEncoderLayer(d, heads, d * 4, drop, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)

    def forward(self, x):
        return self.encoder(self.pos(self.proj(x)))[:, -1]


class CrossModalFusion(nn.Module):
    def __init__(self, img_d=128, seq_d=128, out_d=128):
        super().__init__()
        self.q   = nn.Linear(img_d, seq_d)
        self.k   = nn.Linear(seq_d, seq_d)
        self.v   = nn.Linear(seq_d, seq_d)
        self.out = nn.Sequential(
            nn.Linear(img_d + seq_d, out_d),
            nn.BatchNorm1d(out_d), nn.GELU(), nn.Dropout(0.1))

    def forward(self, xi, xs):
        a = F.scaled_dot_product_attention(
            self.q(xi).unsqueeze(1), self.k(xs).unsqueeze(1), self.v(xs).unsqueeze(1)).squeeze(1)
        return self.out(torch.cat([xi, a], 1))


class HybridModelV4(nn.Module):
    def __init__(self):
        super().__init__()
        base      = tv_models.efficientnet_b2(weights=tv_models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.cnn  = base.features
        self.cbam = CBAM(1408)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cnn_head = nn.Sequential(
            nn.Linear(1408, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1))
        self.seq    = SequenceTransformer()
        self.fusion = CrossModalFusion()
        self.clf    = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 2))

    def forward(self, img, seq):
        x = self.pool(self.cbam(self.cnn(img))).flatten(1)
        x = self.cnn_head(x)
        s = self.seq(seq)
        return self.clf(self.fusion(x, s))


# ── ConvNeXt V2-Tiny + iTransformer (HybridModelV5) ──────────────────────────
class iTransformer(nn.Module):
    def __init__(self, n_feat=25, seq_len=20, d=128, heads=4, layers=2, drop=0.1):
        super().__init__()
        self.proj    = nn.Linear(seq_len, d)
        enc          = nn.TransformerEncoderLayer(d, heads, d * 4, drop, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.norm    = nn.LayerNorm(d)

    def forward(self, x):
        x = x.transpose(1, 2)    # (B, T, F) → (B, F, T)
        x = self.proj(x)          # (B, F, T) → (B, F, d)
        x = self.encoder(x)
        return self.norm(x).mean(dim=1)


class HybridModelV5(nn.Module):
    def __init__(self):
        super().__init__()
        if timm is None:
            raise ImportError("pip install timm")
        try:
            self.backbone = timm.create_model(
                'convnextv2_tiny.fcmae_ft_in22k_in1k',
                pretrained=False, num_classes=0, global_pool='avg')
        except Exception:
            self.backbone = timm.create_model(
                'convnextv2_tiny', pretrained=False,
                num_classes=0, global_pool='avg')
        self.cnn_head = nn.Sequential(
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1))
        self.seq    = iTransformer()
        self.fusion = CrossModalFusion()
        self.clf    = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 2))

    def forward(self, img, seq):
        x = self.cnn_head(self.backbone(img))
        s = self.seq(seq)
        return self.clf(self.fusion(x, s))


# ─────────────────────────────────────────────────────────────────────────────
# 7.  STATE DICT LOADER  (handles torch.compile prefix)
# ─────────────────────────────────────────────────────────────────────────────
def load_weights(model, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if any(k.startswith('_orig_mod.') for k in state.keys()):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 8.  DATASETS
# ─────────────────────────────────────────────────────────────────────────────
class CryptoRGBDataset(Dataset):
    """224×224 RGB multi-scale images + sequence — for EfficientNet & ConvNeXt."""
    def __init__(self, records, seq_mean, seq_std):
        self.records   = records
        self.seq_mean  = seq_mean
        self.seq_std   = seq_std
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        img   = Image.open(rec['img_rgb_path']).convert('RGB')
        img_t = self.transform(img)

        seq = rec['seq'].copy()   # already per-window z-scored
        if self.seq_mean is not None:
            seq = (seq - self.seq_mean) / self.seq_std
        seq = np.where(np.isfinite(seq), seq, 0.0).astype(np.float32)

        return img_t, torch.from_numpy(seq), torch.tensor(rec['label'], dtype=torch.long)


class CryptoGrayDataset(Dataset):
    """60×64 grayscale OHLC images — for JiangCNN baseline."""
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = np.array(Image.open(rec['img_gray_path']), dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img).unsqueeze(0)   # (1, 64, 60)
        return img_t, torch.tensor(rec['label'], dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def infer_advanced(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, seqs, labels in tqdm(loader, desc='  Inference', ncols=80):
            imgs = imgs.to(device)
            seqs = seqs.to(device)
            with autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(imgs, seqs)
            probs = F.softmax(out.float(), dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def infer_baseline(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='  Inference', ncols=80):
            imgs = imgs.to(device)
            out  = model(imgs)
            probs = F.softmax(out.float(), dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def save_predictions(tickers, dates, labels, probs, filename):
    df = pd.DataFrame({
        'ticker':          tickers,
        'date':            dates,
        'label':           labels,
        'Predicted_Class': (probs > 0.5).astype(int),
        'Prob_Up':         probs,
    })
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 10.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("H4: Crypto Transfer Inference")
    print(f"Tickers  : {CRYPTO_TICKERS}")
    print(f"OOS      : {OOS_START} – {OOS_END}")
    print(f"Horizon  : {PREDICT_TERM}d  |  Window: {WINDOW_SIZE}d")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── Step 1: Build crypto samples ─────────────────────────────────────────
    records = build_crypto_samples()
    if not records:
        print("No samples generated. Check download or date range.")
        return

    tickers_list = [r['ticker'] for r in records]
    dates_list   = [r['date']   for r in records]
    labels_list  = [r['label']  for r in records]
    print(f"\nTotal OOS samples: {len(records):,}")
    print(f"  BTC-USD: {sum(t == 'BTC-USD' for t in tickers_list):,}")
    print(f"  ETH-USD: {sum(t == 'ETH-USD' for t in tickers_list):,}")
    up   = sum(labels_list)
    down = len(labels_list) - up
    print(f"  Up={up:,} ({up/len(labels_list)*100:.1f}%)  Down={down:,} ({down/len(labels_list)*100:.1f}%)")

    # ── Step 2: Equity normalization stats ────────────────────────────────────
    print()
    seq_mean, seq_std = load_equity_norm_stats()

    # ── Step 3: DataLoaders ───────────────────────────────────────────────────
    rgb_ds   = CryptoRGBDataset(records, seq_mean, seq_std)
    gray_ds  = CryptoGrayDataset(records)

    rgb_loader  = DataLoader(rgb_ds,  batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    gray_loader = DataLoader(gray_ds, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    results = {}   # model_name → probs array

    # ── Step 4a: EfficientNet-B2 ─────────────────────────────────────────────
    if os.path.exists(EFFNET_MODEL_PATH):
        print(f"\n[1/3] EfficientNet-B2 + Transformer")
        print(f"  Loading: {EFFNET_MODEL_PATH}")
        model  = HybridModelV4().to(device)
        model  = load_weights(model, EFFNET_MODEL_PATH, device)
        probs, lbls = infer_advanced(model, rgb_loader, device)
        auc = roc_auc_score(lbls, probs)
        print(f"  Crypto OOS AUC: {auc:.4f}  (n={len(probs):,})")
        results['efficientnet'] = probs
        path = save_predictions(tickers_list, dates_list, lbls, probs,
                                'efficientnet_crypto_predictions.csv')
        print(f"  Saved: {path}")
        del model;  torch.cuda.empty_cache()
    else:
        print(f"\n[1/3] EfficientNet: not found — {EFFNET_MODEL_PATH}")

    # ── Step 4b: ConvNeXt V2-Tiny ─────────────────────────────────────────────
    if os.path.exists(CONVNEXT_MODEL_PATH):
        print(f"\n[2/3] ConvNeXt V2-Tiny + iTransformer")
        print(f"  Loading: {CONVNEXT_MODEL_PATH}")
        model  = HybridModelV5().to(device)
        model  = load_weights(model, CONVNEXT_MODEL_PATH, device)
        probs, lbls = infer_advanced(model, rgb_loader, device)
        auc = roc_auc_score(lbls, probs)
        print(f"  Crypto OOS AUC: {auc:.4f}  (n={len(probs):,})")
        results['convnext'] = probs
        path = save_predictions(tickers_list, dates_list, lbls, probs,
                                'convnext_crypto_predictions.csv')
        print(f"  Saved: {path}")
        del model;  torch.cuda.empty_cache()
    else:
        print(f"\n[2/3] ConvNeXt: not found — {CONVNEXT_MODEL_PATH}")

    # ── Step 4c: Baseline JiangCNN (ensemble of N runs) ──────────────────────
    if os.path.exists(BASELINE_MODEL_DIR):
        run_files = sorted(f for f in os.listdir(BASELINE_MODEL_DIR) if f.endswith('.pth'))
        if run_files:
            print(f"\n[3/3] Baseline JiangCNN ({len(run_files)} runs)")
            all_run_probs = []
            for rf in run_files:
                print(f"  Run: {rf}")
                model = JiangCNN().to(device)
                model = load_weights(model, os.path.join(BASELINE_MODEL_DIR, rf), device)
                probs_run, lbls = infer_baseline(model, gray_loader, device)
                all_run_probs.append(probs_run)
                del model

            probs = np.mean(all_run_probs, axis=0)
            auc   = roc_auc_score(lbls, probs)
            print(f"  Crypto OOS AUC (avg {len(run_files)} runs): {auc:.4f}")
            results['baseline'] = probs
            path = save_predictions(tickers_list, dates_list, lbls, probs,
                                    'baseline_crypto_predictions.csv')
            print(f"  Saved: {path}")
        else:
            print(f"\n[3/3] Baseline: no .pth files in {BASELINE_MODEL_DIR}")
    else:
        print(f"\n[3/3] Baseline: model dir not found — run cnn_paper.py first")

    # ── Step 5: Ensemble ──────────────────────────────────────────────────────
    if len(results) >= 2:
        ens_probs = np.stack(list(results.values()), axis=0).mean(axis=0)
        ens_auc   = roc_auc_score(labels_list, ens_probs)
        print(f"\nEnsemble ({'+'.join(results.keys())}) — Crypto OOS AUC: {ens_auc:.4f}")
        path = save_predictions(tickers_list, dates_list, labels_list, ens_probs,
                                'ensemble_crypto_predictions.csv')
        print(f"  Saved: {path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Output: {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.csv'):
            n = len(pd.read_csv(os.path.join(OUTPUT_DIR, f)))
            print(f"  {f}: {n:,} rows")
    print("=" * 70)


if __name__ == '__main__':
    main()
