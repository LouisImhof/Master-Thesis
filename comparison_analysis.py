# comparison_analysis.py
# Image Encoding Comparison: v3 (candlestick) vs v4 (multi-scale RGB)
# Models: EfficientNet-B2+Transformer  |  ConvNeXt V2+iTransformer
# Reads 4 prediction files from thesis_data_comparison/ — does NOT touch
# performance_analysis.py or any existing result folders.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import os, glob, warnings
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                             f1_score, brier_score_loss, confusion_matrix)
from sklearn.calibration import calibration_curve
warnings.filterwarnings('ignore')

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE = r'C:\Users\limga\Master Thesis\MasterThesis'
COMP = os.path.join(BASE, 'thesis_data_comparison')
OUT  = os.path.join(BASE, 'comparison_results')
os.makedirs(OUT, exist_ok=True)

BASELINE_IMG_DIR = os.path.join(BASE, 'thesis_data_baseline', 'images', 'test')
BASELINE_NPY_DIR = os.path.join(BASE, 'thesis_data_baseline', 'models')

PRED_FILES = {
    'EfficientNet-v3': os.path.join(COMP, 'efficientnet_v3', 'predictions.csv'),
    'EfficientNet-v4': os.path.join(COMP, 'efficientnet_v4', 'predictions.csv'),
    'ConvNeXt-v3':     os.path.join(COMP, 'convnext_v3',     'predictions.csv'),
    'ConvNeXt-v4':     os.path.join(COMP, 'convnext_v4',     'predictions.csv'),
}

# Visual identity: gray = Baseline; solid = v3, dashed = v4;
#                  blue = EfficientNet, orange = ConvNeXt
STYLES = {
    'Baseline CNN':    {'color': '#616161', 'ls': '-',  'marker': 'D'},
    'EfficientNet-v3': {'color': '#1565C0', 'ls': '-',  'marker': 'o'},
    'EfficientNet-v4': {'color': '#1565C0', 'ls': '--', 'marker': 's'},
    'ConvNeXt-v3':     {'color': '#E65100', 'ls': '-',  'marker': 'o'},
    'ConvNeXt-v4':     {'color': '#E65100', 'ls': '--', 'marker': 's'},
}

REGIMES = {
    'Post-COVID Bull\n(2021)': ('2021-01-01', '2021-12-31'),
    'Bear/Rate Hike\n(2022)':  ('2022-01-01', '2022-12-31'),
    'Recovery\n(2023-24)':     ('2023-01-01', '2024-12-31'),
    'Recent\n(2025+)':         ('2025-01-01', '2099-12-31'),
}


# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_baseline():
    npy_files = glob.glob(os.path.join(BASELINE_NPY_DIR, 'baseline_ensemble_auc*.npy'))
    if not npy_files:
        raise FileNotFoundError(f"No .npy found in {BASELINE_NPY_DIR}")
    probs = np.load(sorted(npy_files)[-1])
    files = sorted(f for f in os.listdir(BASELINE_IMG_DIR) if f.endswith('.png'))
    records = []
    for fname in files:
        stem  = fname.replace('.png', '')
        parts = stem.rsplit('_', 2)   # ticker | YYYYMMDD | label
        records.append({
            'ticker': parts[0],
            'date':   pd.Timestamp(parts[1]),
            'label':  int(parts[2]),
        })
    df = pd.DataFrame(records)
    df['Prob_Up']         = probs
    df['Predicted_Class'] = (probs > 0.5).astype(int)
    print(f"  {'Baseline CNN':<20}: {len(df):,} samples  "
          f"UP={df['label'].sum():,} ({df['label'].mean()*100:.1f}%)  "
          f"({df['date'].dt.year.min()}–{df['date'].dt.year.max()})")
    return df[['ticker', 'date', 'label', 'Predicted_Class', 'Prob_Up']]


def load_predictions():
    dfs = {}
    # Baseline CNN (Jiang et al. 2021) — loaded first so it appears first in tables/plots
    try:
        dfs['Baseline CNN'] = load_baseline()
    except Exception as e:
        print(f"  Baseline CNN: SKIP — {e}")

    for name, path in PRED_FILES.items():
        if not os.path.exists(path):
            print(f"  {name}: NOT FOUND — {path}")
            continue
        df = pd.read_csv(path)
        # date column is YYYYMMDD integer/string — parse explicitly
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
        # extract ticker from image filename
        df['ticker'] = df['image_path'].apply(
            lambda p: os.path.basename(p).replace('.png', '').rsplit('_', 2)[0]
        )
        print(f"  {name:<20}: {len(df):,} samples  "
              f"UP={df['label'].sum():,} ({df['label'].mean()*100:.1f}%)  "
              f"({df['date'].dt.year.min()}–{df['date'].dt.year.max()})")
        dfs[name] = df[['ticker', 'date', 'label', 'Predicted_Class', 'Prob_Up']]
    return dfs


# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs > threshold).astype(int)
    return {
        'AUC':   roc_auc_score(labels, probs),
        'Acc':   accuracy_score(labels, preds),
        'F1':    f1_score(labels, preds, average='macro', zero_division=0),
        'Brier': brier_score_loss(labels, probs),
        'IC':    spearmanr(probs, labels).statistic,
    }

def bootstrap_ci(labels, probs, n=1000, ci=0.95):
    rng = np.random.default_rng(42)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        try:
            scores.append(roc_auc_score(labels[idx], probs[idx]))
        except Exception:
            pass
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return lo, hi


# ── DELONG TEST ───────────────────────────────────────────────────────────────

def delong_test(labels, prob_a, prob_b):
    """DeLong et al. (1988) — compare two AUCs on identical samples."""
    def midrank(x):
        sx = np.sort(x)
        return (np.searchsorted(sx, x, 'right') + np.searchsorted(sx, x, 'left') + 1) / 2.0

    def fast_delong(preds, m):
        n  = preds.shape[1] - m
        tx = np.array([midrank(preds[r, :m]) for r in range(2)])
        ty = np.array([midrank(preds[r, m:]) for r in range(2)])
        tz = np.array([midrank(preds[r, :])  for r in range(2)])
        aucs = (tz[:, :m].sum(1) - tx.sum(1)) / (m * n)
        v01  = (tz[:, :m] - tx) / n
        v10  = 1 - (tz[:, m:] - ty) / m
        cov  = np.cov(v01) / m + np.cov(v10) / n
        return aucs, cov

    order = (-labels).argsort()
    m = int(labels.sum())
    preds = np.array([prob_a[order], prob_b[order]])
    aucs, cov = fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    se   = np.sqrt(cov[0,0] + cov[1,1] - 2*cov[0,1])
    z    = diff / (se + 1e-12)
    p    = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)

def delong_aligned(dfs, name_a, name_b):
    """DeLong on inner-joined (ticker, date) pairs."""
    merged = pd.merge(
        dfs[name_a][['ticker','date','label','Prob_Up']].rename(columns={'Prob_Up':'a'}),
        dfs[name_b][['ticker','date','Prob_Up']].rename(columns={'Prob_Up':'b'}),
        on=['ticker','date'], how='inner'
    )
    if len(merged) < 50:
        return None
    return delong_test(merged['label'].values, merged['a'].values, merged['b'].values), len(merged)


# ── PORTFOLIO ─────────────────────────────────────────────────────────────────

def get_positions(prob_up, strategy='longshort', threshold=0.5, top_q=0.2):
    p = np.array(prob_up)
    if strategy == 'longonly':
        return np.where(p > threshold, 1.0, 0.0)
    if strategy == 'longshort':
        return np.where(p > threshold, 1.0, -1.0)
    if strategy == 'quantile':
        hi = np.quantile(p, 1 - top_q)
        lo = np.quantile(p, top_q)
        return np.where(p >= hi, 1.0, np.where(p <= lo, -1.0, 0.0))
    if strategy == 'probweight':
        return 2 * p - 1
    raise ValueError(strategy)

def portfolio_returns(df, strategy, threshold=0.5, top_q=0.2, scale=0.01):
    ds   = df.sort_values(['date', 'ticker']).copy()
    port = {}
    for date, grp in ds.groupby('date'):
        pos    = get_positions(grp['Prob_Up'].values, strategy, threshold, top_q)
        actual = np.where(grp['label'].values == 1, 1.0, -1.0)
        active = pos != 0
        if not active.any():
            continue
        port[date] = (pos[active] * actual[active]).mean() * scale
    return pd.Series(port).sort_index()

def portfolio_stats(returns):
    r = np.array(returns)
    periods_per_year = 252 / 20
    mean_r = r.mean()
    std_r  = r.std() + 1e-10
    sharpe  = mean_r / std_r * np.sqrt(periods_per_year)
    neg     = r[r < 0]
    sortino = mean_r / (neg.std() + 1e-10) * np.sqrt(periods_per_year) if len(neg) else np.nan
    cum     = np.cumprod(1 + r)
    dd      = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)
    max_dd  = dd.min()
    calmar  = (mean_r * periods_per_year) / (abs(max_dd) + 1e-10)
    return {
        'Sharpe':   sharpe,
        'Sortino':  sortino,
        'Max DD':   max_dd,
        'Calmar':   calmar,
        'Hit Rate': (r > 0).mean(),
        'Cum Ret':  float(cum[-1]) - 1.0,
        'N':        len(r),
    }


# ── PLOT HELPERS ──────────────────────────────────────────────────────────────

def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}")


# ── PLOT 1: ROC — three panels (Baseline | EfficientNet | ConvNeXt) ──────────

def plot_roc_panels(dfs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 0: Baseline reference only
    ax0 = axes[0]
    if 'Baseline CNN' in dfs:
        df  = dfs['Baseline CNN']
        fpr, tpr, _ = roc_curve(df['label'], df['Prob_Up'])
        auc = roc_auc_score(df['label'], df['Prob_Up'])
        st  = STYLES['Baseline CNN']
        ax0.plot(fpr, tpr, lw=2, color=st['color'], ls=st['ls'],
                 label=f"Baseline CNN  (AUC={auc:.4f})")
    ax0.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4, label='Random')
    ax0.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
            title='Baseline CNN\n(Jiang et al. 2021)', xlim=[0,1], ylim=[0,1])
    ax0.legend(loc='lower right', fontsize=9)
    ax0.grid(alpha=0.3)

    # Panels 1–2: EfficientNet and ConvNeXt (with baseline as grey reference)
    for ax, model_key, title in [
        (axes[1], 'EfficientNet', 'EfficientNet-B2 + Transformer'),
        (axes[2], 'ConvNeXt',     'ConvNeXt V2 + iTransformer'),
    ]:
        # grey baseline reference line
        if 'Baseline CNN' in dfs:
            df  = dfs['Baseline CNN']
            fpr, tpr, _ = roc_curve(df['label'], df['Prob_Up'])
            auc = roc_auc_score(df['label'], df['Prob_Up'])
            ax.plot(fpr, tpr, lw=1.5, color='#9E9E9E', ls=':', alpha=0.7,
                    label=f"Baseline CNN  (AUC={auc:.4f})")
        for name, df in dfs.items():
            if not name.startswith(model_key):
                continue
            fpr, tpr, _ = roc_curve(df['label'], df['Prob_Up'])
            auc = roc_auc_score(df['label'], df['Prob_Up'])
            st  = STYLES[name]
            ver = 'v3 Candlestick' if name.endswith('v3') else 'v4 Multi-scale RGB'
            ax.plot(fpr, tpr, lw=2, color=st['color'], ls=st['ls'],
                    label=f"{ver}  (AUC={auc:.4f})")
        ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4, label='Random')
        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
               title=title, xlim=[0,1], ylim=[0,1])
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('ROC Curves — OOS 2021–2026\nBaseline vs Image Encoding: v3 Candlestick vs v4 Multi-scale RGB',
                 fontsize=12)
    plt.tight_layout()
    _save(fig, '01_roc_panels.png')


# ── PLOT 2: Calibration — 2×2 grid ───────────────────────────────────────────

def plot_calibration(dfs):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (name, df) in zip(axes.flat, dfs.items()):
        fp, mp = calibration_curve(df['label'], df['Prob_Up'], n_bins=10)
        st = STYLES[name]
        ax.plot(mp, fp, color=st['color'], lw=2, marker=st['marker'], ms=5, label=name)
        ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Perfect')
        ax.set(xlabel='Mean P(Up)', ylabel='Fraction Up', title=name,
               xlim=[0,1], ylim=[0,1])
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle('Calibration Curves — OOS 2021–2026', fontsize=12)
    plt.tight_layout()
    _save(fig, '02_calibration.png')


# ── PLOT 3: AUC by Year — grouped bars ───────────────────────────────────────

def plot_yearly_auc(dfs):
    years = sorted({y for df in dfs.values() for y in df['date'].dt.year.unique()})
    x = np.arange(len(years))
    w = 0.8 / len(dfs)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (name, df) in enumerate(dfs.items()):
        aucs = []
        for yr in years:
            sub = df[df['date'].dt.year == yr]
            aucs.append(roc_auc_score(sub['label'], sub['Prob_Up'])
                        if len(sub) > 20 and sub['label'].nunique() == 2 else np.nan)
        st = STYLES[name]
        ax.bar(x + i*w - 0.4 + w/2, aucs, w,
               color=st['color'], alpha=0.8 if name.endswith('v4') else 0.55,
               hatch='' if name.endswith('v3') else '//', edgecolor='white',
               label=name)
    ax.axhline(0.5, color='k', lw=1, ls='--', alpha=0.6)
    ax.set(xticks=x, xticklabels=years, ylabel='AUC',
           title='AUC per Year — OOS 2021–2026')
    ax.legend(ncol=2, fontsize=8); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save(fig, '03_yearly_auc.png')


# ── PLOT 4: AUC by Market Regime ─────────────────────────────────────────────

def plot_regime_auc(dfs):
    reg_names = list(REGIMES.keys())
    x = np.arange(len(reg_names))
    w = 0.8 / len(dfs)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, df) in enumerate(dfs.items()):
        aucs = []
        for start, end in REGIMES.values():
            sub = df[(df['date'] >= start) & (df['date'] <= end)]
            aucs.append(roc_auc_score(sub['label'], sub['Prob_Up'])
                        if len(sub) > 20 and sub['label'].nunique() == 2 else np.nan)
        st = STYLES[name]
        ax.bar(x + i*w - 0.4 + w/2, aucs, w,
               color=st['color'], alpha=0.8 if name.endswith('v4') else 0.55,
               hatch='' if name.endswith('v3') else '//', edgecolor='white',
               label=name)
    ax.axhline(0.5, color='k', lw=1, ls='--', alpha=0.6)
    ax.set(xticks=x, xticklabels=reg_names, ylabel='AUC',
           title='AUC by Market Regime — OOS 2021–2026')
    ax.legend(ncol=2, fontsize=8); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save(fig, '04_regime_auc.png')


# ── PLOT 5: Cumulative Returns (longshort + quantile) ─────────────────────────

def plot_cumulative_returns(dfs):
    strategies = ['longshort', 'quantile', 'probweight', 'longonly']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, strat in zip(axes.flat, strategies):
        for name, df in dfs.items():
            rets = portfolio_returns(df, strat)
            if len(rets) == 0:
                continue
            cum = (1 + rets).cumprod()
            st  = STYLES[name]
            ax.plot(cum.index, cum.values, lw=1.5,
                    color=st['color'], ls=st['ls'], label=name)
        ax.axhline(1.0, color='k', lw=1, ls='--', alpha=0.5)
        ax.set(title=f'Strategy: {strat}', xlabel='Date', ylabel='Cum. Return')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.suptitle('Cumulative Returns — Image Encoding Comparison\n'
                 'OOS 2021–2026  |  Cross-sectional  |  1% max per period', fontsize=12)
    plt.tight_layout()
    _save(fig, '05_cumulative_returns.png')


# ── PLOT 6: Prediction Distribution ──────────────────────────────────────────

def plot_pred_distribution(dfs):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (name, df) in zip(axes.flat, dfs.items()):
        ax.hist(df[df['label']==1]['Prob_Up'], bins=40, alpha=0.55,
                color='#388E3C', label='Actual Up', density=True)
        ax.hist(df[df['label']==0]['Prob_Up'], bins=40, alpha=0.55,
                color='#C62828', label='Actual Down', density=True)
        ax.axvline(0.5, color='k', lw=1, ls='--')
        ax.set(title=name, xlabel='P(Up)', ylabel='Density')
        ax.legend(fontsize=8)
    plt.suptitle('Prediction Distribution — OOS 2021–2026', fontsize=12)
    plt.tight_layout()
    _save(fig, '06_pred_distribution.png')


# ── PLOT 7: Confusion Matrices ────────────────────────────────────────────────

def plot_confusion_matrices(dfs):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (name, df) in zip(axes.flat, dfs.items()):
        cm = confusion_matrix(df['label'], df['Predicted_Class'])
        im = ax.imshow(cm, cmap='Blues')
        ax.set(xticks=[0,1], yticks=[0,1],
               xticklabels=['Down','Up'], yticklabels=['Down','Up'],
               xlabel='Predicted', ylabel='Actual', title=name)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                        fontsize=12, color='black' if cm[i,j] < cm.max()/2 else 'white')
    plt.suptitle('Confusion Matrices — OOS 2021–2026', fontsize=12)
    plt.tight_layout()
    _save(fig, '07_confusion_matrices.png')


# ── PLOT 8: Rolling AUC ───────────────────────────────────────────────────────

def plot_rolling_auc(dfs, window=1000):
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, df in dfs.items():
        ds = df.sort_values('date').reset_index(drop=True)
        aucs, dates = [], []
        for i in range(window, len(ds)):
            sub = ds.iloc[i-window:i]
            if sub['label'].nunique() == 2:
                aucs.append(roc_auc_score(sub['label'], sub['Prob_Up']))
                dates.append(sub['date'].iloc[-1])
        st = STYLES[name]
        ax.plot(dates, aucs, lw=1.5, color=st['color'], ls=st['ls'], label=name)
    ax.axhline(0.5, color='k', lw=1, ls='--', alpha=0.5)
    ax.set(xlabel='Date', ylabel=f'Rolling AUC (w={window})',
           title=f'Rolling AUC — OOS 2021–2026  (window = {window} samples)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, '08_rolling_auc.png')


# ── PLOT 9: v3 vs v4 AUC delta summary ───────────────────────────────────────

def plot_v4_delta(dfs, all_metrics):
    # Only uses the four comparison models — baseline has no v3/v4 split
    years = sorted({y for n, df in dfs.items() if n != 'Baseline CNN'
                    for y in df['date'].dt.year.unique()})
    eff_delta, cnx_delta = [], []
    for yr in years:
        def yr_auc(name):
            if name not in dfs:
                return np.nan
            sub = dfs[name][dfs[name]['date'].dt.year == yr]
            return roc_auc_score(sub['label'], sub['Prob_Up']) if len(sub) > 20 and sub['label'].nunique() == 2 else np.nan
        eff_delta.append(yr_auc('EfficientNet-v4') - yr_auc('EfficientNet-v3'))
        cnx_delta.append(yr_auc('ConvNeXt-v4') - yr_auc('ConvNeXt-v3'))

    x = np.arange(len(years))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, eff_delta, w, color='#1565C0', alpha=0.8, label='EfficientNet  Δ(v4−v3)')
    ax.bar(x + w/2, cnx_delta, w, color='#E65100', alpha=0.8, label='ConvNeXt  Δ(v4−v3)')
    ax.axhline(0, color='k', lw=1)
    ax.set(xticks=x, xticklabels=years, ylabel='ΔAUC  (v4 − v3)',
           title='Yearly AUC Gain of Multi-scale RGB (v4) over Candlestick (v3)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save(fig, '09_v4_delta_yearly.png')


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Image Encoding Comparison Analysis")
    print("  v3 Candlestick  vs  v4 Multi-scale RGB")
    print("  EfficientNet-B2+Transformer  |  ConvNeXt V2+iTransformer")
    print("=" * 72)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\nLoading predictions...")
    dfs = load_predictions()
    if not dfs:
        print("ERROR: no prediction files found. Run run_image_comparison.py first.")
        return

    # ── Overall Metrics Table ─────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  {'Model':<22} {'AUC':>7} {'95% CI':>15} {'Acc':>7} {'F1':>7} {'Brier':>7} {'IC':>7}")
    print(f"{'='*72}")
    all_metrics = {}
    for name, df in dfs.items():
        L, P = df['label'].values, df['Prob_Up'].values
        m    = compute_metrics(L, P)
        lo, hi = bootstrap_ci(L, P)
        all_metrics[name] = {**m, 'CI_lo': lo, 'CI_hi': hi}
        print(f"  {name:<22} {m['AUC']:>7.4f} [{lo:.4f}-{hi:.4f}]"
              f" {m['Acc']:>7.4f} {m['F1']:>7.4f} {m['Brier']:>7.4f} {m['IC']:>7.4f}")

    # ── Summary: v3 vs v4 per model ───────────────────────────────────────────
    sign = lambda d: '+' if d > 0 else ''
    print(f"\n  Image Encoding Effect (v4 − v3):")
    print(f"  {'Metric':<30} {'EfficientNet':>14} {'ConvNeXt':>14}")
    for metric in ['AUC', 'Acc', 'F1', 'Brier', 'IC']:
        eff_d = all_metrics.get('EfficientNet-v4', {}).get(metric, np.nan) - \
                all_metrics.get('EfficientNet-v3', {}).get(metric, np.nan)
        cnx_d = all_metrics.get('ConvNeXt-v4',     {}).get(metric, np.nan) - \
                all_metrics.get('ConvNeXt-v3',      {}).get(metric, np.nan)
        print(f"  {metric:<30} {sign(eff_d)}{eff_d:>13.4f} {sign(cnx_d)}{cnx_d:>13.4f}")

    # Best model vs Baseline
    if 'Baseline CNN' in all_metrics:
        print(f"\n  Best Model vs Baseline (v4 − Baseline CNN):")
        print(f"  {'Metric':<30} {'EfficientNet-v4':>16} {'ConvNeXt-v4':>14}")
        for metric in ['AUC', 'Acc', 'F1', 'Brier', 'IC']:
            b = all_metrics['Baseline CNN'].get(metric, np.nan)
            ed = all_metrics.get('EfficientNet-v4', {}).get(metric, np.nan) - b
            cd = all_metrics.get('ConvNeXt-v4',     {}).get(metric, np.nan) - b
            print(f"  {metric:<30} {sign(ed)}{ed:>15.4f} {sign(cd)}{cd:>13.4f}")

    # ── DeLong Tests ──────────────────────────────────────────────────────────
    COMPARISONS = [
        ('EfficientNet-v3', 'EfficientNet-v4', 'EfficientNet: v3 vs v4  (image encoding)'),
        ('ConvNeXt-v3',     'ConvNeXt-v4',     'ConvNeXt:     v3 vs v4  (image encoding)'),
        ('EfficientNet-v3', 'ConvNeXt-v3',     'v3 image:  EfficientNet vs ConvNeXt  (architecture)'),
        ('EfficientNet-v4', 'ConvNeXt-v4',     'v4 image:  EfficientNet vs ConvNeXt  (architecture)'),
        ('Baseline CNN',    'EfficientNet-v4', 'Baseline CNN vs EfficientNet-v4  (best model)'),
        ('Baseline CNN',    'ConvNeXt-v4',     'Baseline CNN vs ConvNeXt-v4      (best model)'),
    ]

    print(f"\n  DeLong AUC Tests:")
    print(f"  {'Comparison':<45} {'AUC-A':>7} {'AUC-B':>7} {'z':>7} {'p':>8} {'sig':>4} {'n':>8}")
    delong_rows = []
    for na, nb, label in COMPARISONS:
        if na not in dfs or nb not in dfs:
            continue
        result = delong_aligned(dfs, na, nb)
        if result is None:
            continue
        (auc_a, auc_b, z, p), n_aligned = result
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {label:<45} {auc_a:>7.4f} {auc_b:>7.4f} {z:>7.3f} {p:>8.4f} {sig:>4} {n_aligned:>8,}")
        delong_rows.append({'Comparison': label, 'AUC_A': auc_a, 'AUC_B': auc_b,
                            'z': z, 'p': p, 'sig': sig, 'n_aligned': n_aligned})

    # ── AUC by Year ───────────────────────────────────────────────────────────
    years = sorted({y for df in dfs.values() for y in df['date'].dt.year.unique()})
    print(f"\n  AUC by Year:")
    print(f"  {'Model':<22}" + "".join(f" {y:>7}" for y in years))
    yearly_rows = []
    for name, df in dfs.items():
        row = {'Model': name}
        line = f"  {name:<22}"
        for yr in years:
            sub = df[df['date'].dt.year == yr]
            if len(sub) > 20 and sub['label'].nunique() == 2:
                auc = roc_auc_score(sub['label'], sub['Prob_Up'])
                line += f" {auc:>7.4f}"
                row[str(yr)] = auc
            else:
                line += f" {'N/A':>7}"
                row[str(yr)] = np.nan
        print(line)
        yearly_rows.append(row)

    # ── AUC by Market Regime ──────────────────────────────────────────────────
    print(f"\n  AUC by Market Regime:")
    reg_labels = [k.replace('\n', ' ') for k in REGIMES]
    print(f"  {'Model':<22}" + "".join(f" {r[:16]:>17}" for r in reg_labels))
    for name, df in dfs.items():
        line = f"  {name:<22}"
        for start, end in REGIMES.values():
            sub = df[(df['date'] >= start) & (df['date'] <= end)]
            if len(sub) > 20 and sub['label'].nunique() == 2:
                line += f" {roc_auc_score(sub['label'], sub['Prob_Up']):>17.4f}"
            else:
                line += f" {'N/A':>17}"
        print(line)

    # ── Trading Strategies ────────────────────────────────────────────────────
    strategies = ['longonly', 'longshort', 'quantile', 'probweight']
    print(f"\n  Trading Strategy Performance (cross-sectional, 1% max per period):")
    port_rows = []
    for strat in strategies:
        print(f"\n  [{strat.upper()}]")
        print(f"  {'Model':<22} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'HitRate':>8} {'CumRet':>8} {'N':>8}")
        for name, df in dfs.items():
            rets = portfolio_returns(df, strat)
            pm   = portfolio_stats(rets.values) if len(rets) else {}
            if not pm:
                continue
            print(f"  {name:<22} {pm['Sharpe']:>8.3f} {pm['Sortino']:>8.3f}"
                  f" {pm['Max DD']:>8.3f} {pm['Hit Rate']:>8.3f} {pm['Cum Ret']:>8.3f} {pm['N']:>8,}")
            port_rows.append({'Strategy': strat, 'Model': name, **pm})

    # ── Ticker-level AUC ──────────────────────────────────────────────────────
    for name, df in dfs.items():
        t_aucs = {}
        for ticker, sub in df.groupby('ticker'):
            if len(sub) > 20 and sub['label'].nunique() == 2:
                t_aucs[ticker] = roc_auc_score(sub['label'], sub['Prob_Up'])
        if not t_aucs:
            continue
        sorted_t = sorted(t_aucs.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top/Bottom 5 Tickers — {name}:")
        print(f"  Best : " + "  ".join(f"{t}={v:.4f}" for t, v in sorted_t[:5]))
        print(f"  Worst: " + "  ".join(f"{t}={v:.4f}" for t, v in sorted_t[-5:]))

    # ── Generate Plots ────────────────────────────────────────────────────────
    print(f"\nGenerating plots → {OUT}")
    plot_roc_panels(dfs)
    plot_calibration(dfs)
    plot_yearly_auc(dfs)
    plot_regime_auc(dfs)
    plot_cumulative_returns(dfs)
    plot_pred_distribution(dfs)
    plot_confusion_matrices(dfs)
    plot_rolling_auc(dfs)
    plot_v4_delta(dfs, all_metrics)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    pd.DataFrame([
        {'Model': n, **{k: v for k, v in m.items()}}
        for n, m in all_metrics.items()
    ]).to_csv(os.path.join(OUT, 'metrics_summary.csv'), index=False)
    print(f"  Saved: metrics_summary.csv")

    if delong_rows:
        pd.DataFrame(delong_rows).to_csv(os.path.join(OUT, 'delong_tests.csv'), index=False)
        print(f"  Saved: delong_tests.csv")

    if yearly_rows:
        pd.DataFrame(yearly_rows).to_csv(os.path.join(OUT, 'yearly_auc.csv'), index=False)
        print(f"  Saved: yearly_auc.csv")

    if port_rows:
        pd.DataFrame(port_rows).to_csv(os.path.join(OUT, 'portfolio_stats.csv'), index=False)
        print(f"  Saved: portfolio_stats.csv")

    # ── Thesis-ready Summary Table ────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  THESIS SUMMARY TABLE")
    print(f"  {'Model':<22} {'Image':>12} {'AUC':>7} {'95% CI':>15} {'F1':>7} {'Brier':>7}")
    print(f"{'='*72}")
    for name, m in all_metrics.items():
        if name == 'Baseline CNN':
            img = 'Baseline'
        elif name.endswith('v3'):
            img = 'Candlestick'
        else:
            img = 'Multi-scale'
        print(f"  {name:<22} {img:>12} {m['AUC']:>7.4f} [{m['CI_lo']:.4f}-{m['CI_hi']:.4f}]"
              f" {m['F1']:>7.4f} {m['Brier']:>7.4f}")

    print(f"\n  Key Findings:")
    if 'EfficientNet-v3' in all_metrics and 'EfficientNet-v4' in all_metrics:
        eff_gain = all_metrics['EfficientNet-v4']['AUC'] - all_metrics['EfficientNet-v3']['AUC']
        print(f"  EfficientNet v4 vs v3 gain : {eff_gain:+.4f} AUC")
    if 'ConvNeXt-v3' in all_metrics and 'ConvNeXt-v4' in all_metrics:
        cnx_gain = all_metrics['ConvNeXt-v4']['AUC'] - all_metrics['ConvNeXt-v3']['AUC']
        print(f"  ConvNeXt     v4 vs v3 gain : {cnx_gain:+.4f} AUC")
    if 'Baseline CNN' in all_metrics:
        b_auc = all_metrics['Baseline CNN']['AUC']
        for best in ['EfficientNet-v4', 'ConvNeXt-v4']:
            if best in all_metrics:
                gain = all_metrics[best]['AUC'] - b_auc
                print(f"  {best} vs Baseline   : {gain:+.4f} AUC")

    print(f"\n{'='*72}")
    print(f"  Done.  All outputs → {OUT}")
    print(f"{'='*72}")


if __name__ == '__main__':
    main()
