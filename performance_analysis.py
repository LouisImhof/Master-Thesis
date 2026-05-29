# performance_analysis.py
# Master Thesis — Complete Performance Analysis
# Baseline CNN (Jiang et al. 2021) vs EfficientNet-B2+Transformer vs ConvNeXtV2+iTransformer
# Outputs: metrics table, DeLong tests, trading strategies, 8 plots, summary CSV

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
BASELINE_IMG_DIR = r'C:\Users\limga\Master Thesis\MasterThesis\thesis_data_baseline\images\test'
BASELINE_NPY_DIR = r'C:\Users\limga\Master Thesis\MasterThesis\thesis_data_baseline\models'
EFFICIENT_CSV    = r'C:\Users\limga\Master Thesis\MasterThesis\thesis_data_EfficientNetB2_Transformer\out_of_sample_predictions.csv'
CONVNEXT_CSV     = r'C:\Users\limga\Master Thesis\MasterThesis\thesis_data_ConvNeXtV2_iTransformer\out_of_sample_predictions.csv'
OUT_DIR          = r'C:\Users\limga\Master Thesis\MasterThesis\analysis_results'
CRYPTO_DIR       = r'C:\Users\limga\Master Thesis\MasterThesis\thesis_data_crypto'
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    'Baseline CNN':   '#2196F3',
    'EfficientNet-B2':'#4CAF50',
    'ConvNeXt V2':    '#FF5722',
    'Ensemble':       '#9C27B0',
}

# Market regimes for OOS period
REGIMES = {
    'Post-COVID Bull (2021)':  ('2021-01-01', '2021-12-31'),
    'Bear / Rate Hike (2022)': ('2022-01-01', '2022-12-31'),
    'Recovery (2023-2024)':    ('2023-01-01', '2024-12-31'),
    'Recent (2025+)':          ('2025-01-01', '2099-12-31'),
}

# ── LOAD PREDICTIONS ──────────────────────────────────────────────────────────

def load_baseline():
    npy_files = glob.glob(os.path.join(BASELINE_NPY_DIR, 'baseline_ensemble_auc*.npy'))
    if not npy_files:
        raise FileNotFoundError(f"No .npy found in {BASELINE_NPY_DIR}")
    probs = np.load(sorted(npy_files)[-1])
    files = sorted(f for f in os.listdir(BASELINE_IMG_DIR) if f.endswith('.png'))
    records = []
    for fname in files:
        stem  = fname.replace('.png', '')
        parts = stem.rsplit('_', 2)          # ticker_clean | YYYYMMDD | label
        ticker = parts[0]
        date   = pd.Timestamp(parts[1])
        label  = int(parts[2])
        records.append({'ticker': ticker, 'date': date, 'label': label})
    df = pd.DataFrame(records)
    df['Prob_Up']         = probs
    df['Predicted_Class'] = (probs > 0.5).astype(int)
    print(f"  Baseline CNN   : {len(df):,} samples  "
          f"({df['date'].dt.year.min()}–{df['date'].dt.year.max()})")
    return df

def load_model_csv(csv_path, name):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    # Extract ticker from image_path (last path component, strip date+ext suffix)
    def _ticker(p):
        stem = os.path.basename(p).replace('.png', '')
        parts = stem.rsplit('_', 2)
        return parts[0] if len(parts) == 3 else stem
    df['ticker'] = df['image_path'].apply(_ticker)
    print(f"  {name:<18}: {len(df):,} samples  "
          f"({df['date'].dt.year.min()}–{df['date'].dt.year.max()})")
    return df[['ticker', 'date', 'label', 'Predicted_Class', 'Prob_Up']]


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
    """DeLong et al. (1988) — compare two AUCs on the same samples."""
    def midrank(x):
        sx = np.sort(x)
        return (np.searchsorted(sx, x, 'right') + np.searchsorted(sx, x, 'left') + 1) / 2.0

    def fast_delong(preds, m):
        n  = preds.shape[1] - m
        tx = np.array([midrank(preds[r, :m])   for r in range(2)])
        ty = np.array([midrank(preds[r, m:])   for r in range(2)])
        tz = np.array([midrank(preds[r, :])    for r in range(2)])
        aucs = (tz[:, :m].sum(1) - tx.sum(1)) / (m * n)
        v01  = (tz[:, :m] - tx) / n
        v10  = 1 - (tz[:, m:] - ty) / m
        cov  = np.cov(v01) / m + np.cov(v10) / n
        return aucs, cov

    order  = (-labels).argsort()
    m      = int(labels.sum())
    preds  = np.array([prob_a[order], prob_b[order]])
    aucs, cov = fast_delong(preds, m)
    diff   = aucs[0] - aucs[1]
    se     = np.sqrt(cov[0,0] + cov[1,1] - 2*cov[0,1])
    z      = diff / (se + 1e-12)
    p      = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


# ── TRADING STRATEGIES ────────────────────────────────────────────────────────

def get_positions(prob_up, strategy='longshort', threshold=0.5, top_q=0.2):
    """Return position array: +1 long, -1 short, 0 neutral."""
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
        return 2 * p - 1          # continuous position in [-1, +1]
    if strategy == 'momentum':
        # combine model signal with recent momentum (consecutive up labels)
        return np.where(p > threshold, 1.0, -1.0) * np.clip(p * 2, 0.5, 1.5)
    raise ValueError(strategy)

def portfolio_stats(returns):
    """Expects returns already in decimal form (e.g. 0.005 = 0.5% per period)."""
    r = np.array(returns)
    periods_per_year = 252 / 20
    mean_r   = r.mean()
    std_r    = r.std() + 1e-10
    sharpe   = mean_r / std_r * np.sqrt(periods_per_year)
    neg      = r[r < 0]
    sortino  = mean_r / (neg.std() + 1e-10) * np.sqrt(periods_per_year) if len(neg) > 0 else np.nan
    cum      = np.cumprod(1 + r)
    drawdown = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)
    max_dd   = drawdown.min()
    calmar   = (mean_r * periods_per_year) / (abs(max_dd) + 1e-10)
    return {
        'Sharpe':   sharpe,
        'Sortino':  sortino,
        'Max DD':   max_dd,
        'Calmar':   calmar,
        'Hit Rate': (r > 0).mean(),
        'Cum Ret':  float(cum[-1]) - 1.0 if len(cum) > 0 else 0.0,
        'N':        len(r),
    }


def compute_portfolio_returns(df, strategy, threshold=0.5, top_q=0.2, scale=0.01):
    """
    Correct cross-sectional portfolio construction.

    For each date:
      1. Rank all stocks by P(Up) and assign positions via strategy
      2. Compute actual return direction: +1 if stock went up, -1 if down
      3. Portfolio return = mean(position_i * direction_i) across active positions
      4. Scale to ±scale (default 1%) max return per period

    Compounding is then done on one return per date, not one per stock-date pair.
    """
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


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}")


def plot_roc(dfs):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, df in dfs.items():
        fpr, tpr, _ = roc_curve(df['label'], df['Prob_Up'])
        auc = roc_auc_score(df['label'], df['Prob_Up'])
        ax.plot(fpr, tpr, lw=2, color=COLORS.get(name, 'gray'),
                label=f"{name}  (AUC={auc:.4f})")
    ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5)
    ax.set(xlabel='FPR', ylabel='TPR', title='ROC Curves — OOS 2021-2026')
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, 'roc_curves.png')


def plot_calibration(dfs):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, df in dfs.items():
        fp, mp = calibration_curve(df['label'], df['Prob_Up'], n_bins=10)
        ax.plot(mp, fp, 's-', lw=2, color=COLORS.get(name, 'gray'), label=name)
    ax.plot([0,1],[0,1], 'k--', lw=1, label='Perfect')
    ax.set(xlabel='Mean Predicted P(Up)', ylabel='Fraction Up',
           title='Calibration — OOS 2021-2026')
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, 'calibration.png')


def plot_yearly_auc(dfs):
    years = sorted({y for df in dfs.values() for y in df['date'].dt.year.unique()})
    x = np.arange(len(years)); w = 0.8 / len(dfs)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, df) in enumerate(dfs.items()):
        aucs = []
        for yr in years:
            sub = df[df['date'].dt.year == yr]
            aucs.append(roc_auc_score(sub['label'], sub['Prob_Up'])
                        if len(sub) > 20 and sub['label'].nunique() == 2 else np.nan)
        ax.bar(x + i*w - 0.4 + w/2, aucs, w,
               color=COLORS.get(name, 'gray'), label=name, alpha=0.85)
    ax.axhline(0.5, color='k', lw=1, linestyle='--')
    ax.set(xticks=x, xticklabels=years, ylabel='AUC',
           title='AUC per Year — OOS 2021-2026')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    _save(fig, 'yearly_auc.png')


def plot_rolling_auc(dfs, window=500):
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, df in dfs.items():
        ds = df.sort_values('date').reset_index(drop=True)
        aucs, dates = [], []
        for i in range(window, len(ds)):
            sub = ds.iloc[i-window:i]
            if sub['label'].nunique() == 2:
                aucs.append(roc_auc_score(sub['label'], sub['Prob_Up']))
                dates.append(sub['date'].iloc[-1])
        ax.plot(dates, aucs, lw=2, color=COLORS.get(name, 'gray'), label=name)
    ax.axhline(0.5, color='k', lw=1, linestyle='--')
    ax.set(xlabel='Date', ylabel=f'Rolling AUC (w={window})',
           title=f'Rolling AUC — OOS 2021-2026')
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, 'rolling_auc.png')


def plot_cumulative_returns(dfs):
    strategies = ['longshort', 'quantile', 'probweight', 'longonly']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, strat in zip(axes.flat, strategies):
        for name, df in dfs.items():
            rets = compute_portfolio_returns(df, strat)
            if len(rets) == 0:
                continue
            cum = (1 + rets).cumprod()
            ax.plot(cum.index, cum.values, lw=1.5,
                    color=COLORS.get(name, 'gray'), label=name)
        ax.axhline(1.0, color='k', lw=1, linestyle='--')
        ax.set(title=f'Strategy: {strat}', xlabel='Date', ylabel='Cum. Return')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle('Cumulative Returns — OOS 2021-2026\n(cross-sectional, 1% max per period)',
                 fontsize=13)
    plt.tight_layout()
    _save(fig, 'cumulative_returns.png')


def plot_confusion_matrices(dfs):
    fig, axes = plt.subplots(1, len(dfs), figsize=(5*len(dfs), 4))
    if len(dfs) == 1:
        axes = [axes]
    for ax, (name, df) in zip(axes, dfs.items()):
        cm = confusion_matrix(df['label'], df['Predicted_Class'])
        ax.imshow(cm, cmap='Blues')
        ax.set(xticks=[0,1], yticks=[0,1],
               xticklabels=['Down','Up'], yticklabels=['Down','Up'],
               xlabel='Predicted', ylabel='Actual', title=name)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center', fontsize=12)
    plt.suptitle('Confusion Matrices — OOS 2021-2026')
    plt.tight_layout()
    _save(fig, 'confusion_matrices.png')


def plot_pred_distribution(dfs):
    fig, axes = plt.subplots(1, len(dfs), figsize=(5*len(dfs), 4))
    if len(dfs) == 1:
        axes = [axes]
    for ax, (name, df) in zip(axes, dfs.items()):
        ax.hist(df[df['label']==1]['Prob_Up'], bins=40, alpha=0.55,
                color='green', label='Actual Up', density=True)
        ax.hist(df[df['label']==0]['Prob_Up'], bins=40, alpha=0.55,
                color='red',   label='Actual Down', density=True)
        ax.axvline(0.5, color='k', lw=1, linestyle='--')
        ax.set(title=name, xlabel='P(Up)', ylabel='Density')
        ax.legend(fontsize=8)
    plt.suptitle('Prediction Distribution — OOS 2021-2026')
    plt.tight_layout()
    _save(fig, 'pred_distribution.png')


def plot_regime_auc(dfs):
    reg_names = list(REGIMES.keys())
    x = np.arange(len(reg_names)); w = 0.8 / len(dfs)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, df) in enumerate(dfs.items()):
        aucs = []
        for start, end in REGIMES.values():
            sub = df[(df['date'] >= start) & (df['date'] <= end)]
            aucs.append(roc_auc_score(sub['label'], sub['Prob_Up'])
                        if len(sub) > 20 and sub['label'].nunique() == 2 else np.nan)
        ax.bar(x + i*w - 0.4 + w/2, aucs, w,
               color=COLORS.get(name, 'gray'), label=name, alpha=0.85)
    ax.axhline(0.5, color='k', lw=1, linestyle='--')
    ax.set(xticks=x, xticklabels=reg_names, ylabel='AUC',
           title='AUC by Market Regime — OOS 2021-2026')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=15)
    _save(fig, 'regime_auc.png')


def plot_threshold_sensitivity(dfs):
    thresholds = np.linspace(0.3, 0.7, 41)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for name, df in dfs.items():
        accs, sharpes = [], []
        for thr in thresholds:
            preds = (df['Prob_Up'].values > thr).astype(int)
            accs.append(accuracy_score(df['label'].values, preds))
            rets = compute_portfolio_returns(df, 'longshort', threshold=thr)
            sharpes.append(portfolio_stats(rets.values)['Sharpe'] if len(rets) > 0 else np.nan)
        ax1.plot(thresholds, accs,    lw=2, color=COLORS.get(name, 'gray'), label=name)
        ax2.plot(thresholds, sharpes, lw=2, color=COLORS.get(name, 'gray'), label=name)
    ax1.axvline(0.5, color='k', lw=1, linestyle='--')
    ax2.axhline(0.0, color='k', lw=1, linestyle='--')
    ax1.set(xlabel='Threshold', ylabel='Accuracy', title='Accuracy vs Threshold')
    ax2.set(xlabel='Threshold', ylabel='Sharpe',   title='Sharpe vs Threshold (Long/Short)')
    ax1.legend(); ax2.legend()
    ax1.grid(alpha=0.3); ax2.grid(alpha=0.3)
    plt.suptitle('Threshold Sensitivity — OOS 2021-2026')
    plt.tight_layout()
    _save(fig, 'threshold_sensitivity.png')


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Master Thesis — Performance Analysis")
    print("  Baseline CNN  |  EfficientNet-B2  |  ConvNeXt V2")
    print("=" * 70)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\nLoading predictions...")
    dfs = {}
    for loader, name in [
        (lambda: load_baseline(),                          'Baseline CNN'),
        (lambda: load_model_csv(EFFICIENT_CSV, name),     'EfficientNet-B2'),
        (lambda: load_model_csv(CONVNEXT_CSV, name),      'ConvNeXt V2'),
    ]:
        try:
            dfs[name] = loader()
        except Exception as e:
            print(f"  {name}: SKIP — {e}")

    if not dfs:
        print("No predictions found. Run all models first.")
        return

    # Ensemble: inner join on date+ticker, average probs
    if len(dfs) >= 2:
        merged = None
        for name, df in dfs.items():
            dk = df[['ticker', 'date', 'label', 'Prob_Up']].rename(columns={'Prob_Up': f'p_{name}'})
            if merged is None:
                merged = dk
            else:
                merged = pd.merge(merged, dk[['ticker', 'date', f'p_{name}']],
                                  on=['ticker', 'date'], how='inner')
        if merged is not None and len(merged) > 100:
            pcols = [c for c in merged.columns if c.startswith('p_')]
            merged['Prob_Up']         = merged[pcols].mean(axis=1)
            merged['Predicted_Class'] = (merged['Prob_Up'] > 0.5).astype(int)
            dfs['Ensemble']           = merged[['ticker','date','label','Predicted_Class','Prob_Up']]
            print(f"  {'Ensemble':<18}: {len(merged):,} aligned samples")

    # ── Metrics table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {'Model':<20} {'AUC':>7} {'95% CI':>15} {'Acc':>7} {'F1':>7} {'Brier':>7} {'IC':>7}")
    print(f"{'='*70}")
    all_metrics = {}
    for name, df in dfs.items():
        L = df['label'].values; P = df['Prob_Up'].values
        m = compute_metrics(L, P)
        lo, hi = bootstrap_ci(L, P)
        all_metrics[name] = m
        print(f"  {name:<20} {m['AUC']:>7.4f} [{lo:.4f}-{hi:.4f}]"
              f" {m['Acc']:>7.4f} {m['F1']:>7.4f} {m['Brier']:>7.4f} {m['IC']:>7.4f}")

    # ── DeLong pairwise ───────────────────────────────────────────────────────
    base_names = [n for n in dfs if n != 'Ensemble']
    if len(base_names) >= 2:
        print(f"\n  DeLong Pairwise AUC Tests:")
        print(f"  {'Comparison':<40} {'AUC-A':>7} {'AUC-B':>7} {'z':>7} {'p':>8} {'sig':>4}")
        for i in range(len(base_names)):
            for j in range(i+1, len(base_names)):
                na, nb = base_names[i], base_names[j]
                merged = pd.merge(
                    dfs[na][['ticker','date','label','Prob_Up']].rename(columns={'Prob_Up':'a'}),
                    dfs[nb][['ticker','date','Prob_Up']].rename(columns={'Prob_Up':'b'}),
                    on=['ticker','date'], how='inner'
                )
                if len(merged) < 50:
                    continue
                auc_a, auc_b, z, p = delong_test(merged['label'].values,
                                                  merged['a'].values,
                                                  merged['b'].values)
                sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
                label = f"{na} vs {nb}"
                print(f"  {label:<40} {auc_a:>7.4f} {auc_b:>7.4f} {z:>7.3f} {p:>8.4f} {sig:>4}")

    # ── Regime Analysis ───────────────────────────────────────────────────────
    print(f"\n  AUC by Market Regime:")
    header = f"  {'Model':<20}" + "".join(f" {r[:12]:>13}" for r in REGIMES)
    print(header)
    for name, df in dfs.items():
        row = f"  {name:<20}"
        for start, end in REGIMES.values():
            sub = df[(df['date'] >= start) & (df['date'] <= end)]
            if len(sub) > 20 and sub['label'].nunique() == 2:
                row += f" {roc_auc_score(sub['label'], sub['Prob_Up']):>13.4f}"
            else:
                row += f" {'N/A':>13}"
        print(row)

    # ── Yearly AUC ────────────────────────────────────────────────────────────
    years = sorted({y for df in dfs.values() for y in df['date'].dt.year.unique()})
    print(f"\n  AUC by Year:")
    header = f"  {'Model':<20}" + "".join(f" {y:>7}" for y in years)
    print(header)
    for name, df in dfs.items():
        row = f"  {name:<20}"
        for yr in years:
            sub = df[df['date'].dt.year == yr]
            if len(sub) > 20 and sub['label'].nunique() == 2:
                row += f" {roc_auc_score(sub['label'], sub['Prob_Up']):>7.4f}"
            else:
                row += f" {'N/A':>7}"
        print(row)

    # ── Trading Strategies ────────────────────────────────────────────────────
    strategies = ['longonly', 'longshort', 'quantile', 'probweight']
    print(f"\n  Trading Strategy Performance (cross-sectional, 1% max per period):")
    for strat in strategies:
        print(f"\n  [{strat.upper()}]")
        print(f"  {'Model':<20} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'HitRate':>8} {'CumRet':>8} {'N':>8}")
        for name, df in dfs.items():
            rets = compute_portfolio_returns(df, strat)
            pm   = portfolio_stats(rets.values)
            print(f"  {name:<20} {pm['Sharpe']:>8.3f} {pm['Sortino']:>8.3f}"
                  f" {pm['Max DD']:>8.3f} {pm['Hit Rate']:>8.3f} {pm['Cum Ret']:>8.3f} {pm['N']:>8,}")

    # ── Ticker-level (models with ticker info) ────────────────────────────────
    for name, df in dfs.items():
        if df['ticker'].nunique() <= 1:
            continue
        print(f"\n  Top/Bottom 5 Tickers by AUC — {name}:")
        t_aucs = {}
        for ticker, sub in df.groupby('ticker'):
            if len(sub) > 20 and sub['label'].nunique() == 2:
                t_aucs[ticker] = roc_auc_score(sub['label'], sub['Prob_Up'])
        if t_aucs:
            sorted_t = sorted(t_aucs.items(), key=lambda x: x[1], reverse=True)
            print(f"  Best : " + "  ".join(f"{t}={v:.4f}" for t,v in sorted_t[:5]))
            print(f"  Worst: " + "  ".join(f"{t}={v:.4f}" for t,v in sorted_t[-5:]))

    # ── Generate plots ────────────────────────────────────────────────────────
    print(f"\nGenerating plots → {OUT_DIR}")
    plot_roc(dfs)
    plot_calibration(dfs)
    plot_yearly_auc(dfs)
    plot_rolling_auc(dfs)
    plot_cumulative_returns(dfs)
    plot_confusion_matrices(dfs)
    plot_pred_distribution(dfs)
    plot_regime_auc(dfs)
    plot_threshold_sensitivity(dfs)

    # ── Save CSV summary ──────────────────────────────────────────────────────
    rows = [{'Model': n, **m} for n, m in all_metrics.items()]
    out_csv = os.path.join(OUT_DIR, 'metrics_summary.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n  Metrics CSV → {out_csv}")
    print(f"{'='*70}")
    print("  Done.")
    print(f"{'='*70}")


def analyze_crypto():
    print("\n" + "=" * 70)
    print("  H4: Crypto Transfer Learning Analysis")
    print("  Euro Stoxx 50 trained models → BTC-USD + ETH-USD (zero-shot)")
    print("=" * 70)

    crypto_files = {
        'Baseline CNN':    'baseline_crypto_predictions.csv',
        'EfficientNet-B2': 'efficientnet_crypto_predictions.csv',
        'ConvNeXt V2':     'convnext_crypto_predictions.csv',
        'Ensemble':        'ensemble_crypto_predictions.csv',
    }

    dfs = {}
    for name, fname in crypto_files.items():
        path = os.path.join(CRYPTO_DIR, fname)
        if not os.path.exists(path):
            print(f"  {name}: NOT FOUND — {path}")
            continue
        df = pd.read_csv(path, parse_dates=['date'])
        dfs[name] = df
        print(f"  {name:<18}: {len(df):,} samples  "
              f"Up={df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")

    if not dfs:
        print("  No crypto predictions found.")
        return

    # ── Overall metrics + bootstrap CI ──────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {'Model':<20} {'AUC':>7} {'95% CI':>15} {'Acc':>7} {'F1':>7} {'Brier':>7}")
    print(f"{'='*70}")
    crypto_metrics = {}
    for name, df in dfs.items():
        L = df['label'].values; P = df['Prob_Up'].values
        m = compute_metrics(L, P)
        lo, hi = bootstrap_ci(L, P)
        crypto_metrics[name] = m
        sig = '***' if lo > 0.5 else ('*' if (lo + hi) / 2 > 0.5 else 'ns')
        print(f"  {name:<20} {m['AUC']:>7.4f} [{lo:.4f}-{hi:.4f}]"
              f" {m['Acc']:>7.4f} {m['F1']:>7.4f} {m['Brier']:>7.4f}  {sig}")

    # ── Per-coin breakdown ───────────────────────────────────────────────────
    print(f"\n  AUC by Coin:")
    print(f"  {'Model':<20} {'BTC-USD':>10} {'ETH-USD':>10}")
    for name, df in dfs.items():
        row = f"  {name:<20}"
        for coin in ['BTC-USD', 'ETH-USD']:
            sub = df[df['ticker'] == coin]
            if len(sub) > 20 and sub['label'].nunique() == 2:
                row += f" {roc_auc_score(sub['label'], sub['Prob_Up']):>10.4f}"
            else:
                row += f" {'N/A':>10}"
        print(row)

    # ── DeLong pairwise on crypto ────────────────────────────────────────────
    base_names = [n for n in dfs if n != 'Ensemble']
    if len(base_names) >= 2:
        print(f"\n  DeLong Pairwise Tests (Crypto):")
        print(f"  {'Comparison':<40} {'AUC-A':>7} {'AUC-B':>7} {'z':>7} {'p':>8} {'sig':>4}")
        for i in range(len(base_names)):
            for j in range(i + 1, len(base_names)):
                na, nb = base_names[i], base_names[j]
                merged = pd.merge(
                    dfs[na][['ticker', 'date', 'label', 'Prob_Up']].rename(columns={'Prob_Up': 'a'}),
                    dfs[nb][['ticker', 'date', 'Prob_Up']].rename(columns={'Prob_Up': 'b'}),
                    on=['ticker', 'date'], how='inner'
                )
                if len(merged) < 50:
                    continue
                auc_a, auc_b, z, p = delong_test(merged['label'].values,
                                                   merged['a'].values,
                                                   merged['b'].values)
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                label = f"{na} vs {nb}"
                print(f"  {label:<40} {auc_a:>7.4f} {auc_b:>7.4f} {z:>7.3f} {p:>8.4f} {sig:>4}")

    # ── Yearly AUC on crypto ─────────────────────────────────────────────────
    years = sorted({y for df in dfs.values() for y in df['date'].dt.year.unique()})
    print(f"\n  Crypto AUC by Year:")
    print(f"  {'Model':<20}" + "".join(f" {y:>7}" for y in years))
    for name, df in dfs.items():
        row = f"  {name:<20}"
        for yr in years:
            sub = df[df['date'].dt.year == yr]
            if len(sub) > 20 and sub['label'].nunique() == 2:
                row += f" {roc_auc_score(sub['label'], sub['Prob_Up']):>7.4f}"
            else:
                row += f" {'N/A':>7}"
        print(row)

    # ── ROC curve plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, df in dfs.items():
        fpr, tpr, _ = roc_curve(df['label'], df['Prob_Up'])
        auc = roc_auc_score(df['label'], df['Prob_Up'])
        ax.plot(fpr, tpr, lw=2, color=COLORS.get(name, 'gray'),
                label=f"{name}  (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set(xlabel='FPR', ylabel='TPR',
           title='ROC Curves — Crypto Transfer (H4)\nBTC-USD + ETH-USD, 2021-2026')
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, 'crypto_roc_curves.png')

    # ── Cumulative returns on crypto ─────────────────────────────────────────
    strategies = ['longshort', 'quantile']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, strat in zip(axes, strategies):
        for name, df in dfs.items():
            if name == 'Ensemble':
                continue
            rets = compute_portfolio_returns(df, strat)
            if len(rets) == 0:
                continue
            cum = (1 + rets).cumprod()
            ax.plot(cum.index, cum.values, lw=1.5,
                    color=COLORS.get(name, 'gray'), label=name)
        ax.axhline(1.0, color='k', lw=1, linestyle='--')
        ax.set(title=f'Crypto Strategy: {strat}', xlabel='Date', ylabel='Cum. Return')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle('Crypto Cumulative Returns — H4 Transfer (1% max per period)', fontsize=12)
    plt.tight_layout()
    _save(fig, 'crypto_cumulative_returns.png')

    # ── Save crypto metrics CSV ──────────────────────────────────────────────
    rows = [{'Model': n, **m} for n, m in crypto_metrics.items()]
    out_csv = os.path.join(OUT_DIR, 'crypto_metrics_summary.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n  Crypto Metrics CSV → {out_csv}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
    analyze_crypto()
