# corrected_backtest.py
# Replaces the ±1% proxy-return portfolio with a proper backtest on actual
# 20-day forward returns.
#
# Primary: non-overlapping rebalancing every 20 trading days, top/bottom
#          quintile long-short, dollar-neutral, equal-weighted.
# Robustness: overlapping Jegadeesh-Titman cohorts, Newey-West (21 lags).
# Costs: 10 bps round-trip per leg (5 bps one-way entry + exit).
# Benchmark: equal-weighted buy-and-hold Euro Stoxx 50.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, warnings
from scipy import stats
from pathlib import Path
warnings.filterwarnings('ignore')

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE    = r'C:\Users\limga\Master Thesis\MasterThesis'
OUT     = os.path.join(BASE, 'backtest_results')
os.makedirs(OUT, exist_ok=True)

PRICE_CSV   = os.path.join(BASE, 'thesis_data_baseline', 'baseline_tickers.csv')
MAPPING_CSV = os.path.join(BASE, 'thesis_data_claudev4', 'dataset_mapping.csv')
BASELINE_NPY_DIR = os.path.join(BASE, 'thesis_data_baseline', 'models')
BASELINE_IMG_DIR = os.path.join(BASE, 'thesis_data_baseline', 'images', 'test')

PRED_FILES = {
    'Baseline CNN':    None,   # loaded from .npy + image filenames
    'EfficientNet-B2': os.path.join(BASE, 'thesis_data_EfficientNetB2_Transformer',
                                    'out_of_sample_predictions.csv'),
    'ConvNeXt V2':     os.path.join(BASE, 'thesis_data_ConvNeXtV2_iTransformer',
                                    'out_of_sample_predictions.csv'),
}

HORIZON    = 20       # holding period in trading days
TOP_Q      = 0.20     # top / bottom quintile fraction
COST_RT    = 0.0010   # 10 bps round-trip per leg (5 bps in + 5 bps out)
WINSOR_PCT = 1        # winsorise returns at this percentile (each tail)


# ── STEP 1: LOAD PRICES AND COMPUTE ACTUAL FORWARD RETURNS ───────────────────

print("Loading price data ...")
prices_raw = pd.read_csv(PRICE_CSV, parse_dates=['Date'])
prices_raw = prices_raw.rename(columns={'Date': 'date'})

# Pivot to wide-format close prices: index = date, columns = ticker (yfinance)
close = (prices_raw
         .pivot_table(index='date', columns='Ticker', values='Close')
         .sort_index())

print(f"  Price data: {close.shape[0]} dates x {close.shape[1]} tickers "
      f"({close.index.min().date()} – {close.index.max().date()})")

# 20-day forward return: r_{i,t} = Close_{t+H} / Close_t - 1
# Shift backward by H so fwd_ret.loc[t, ticker] = actual 20-day return FROM t
fwd_ret = close.shift(-HORIZON) / close - 1
fwd_ret_oos = fwd_ret.loc['2021-01-01':'2026-01-01']  # OOS window

# Winsorise at 1st/99th percentile of the full OOS return distribution
ret_vals = fwd_ret_oos.values.ravel()
ret_vals = ret_vals[~np.isnan(ret_vals)]
lo_w, hi_w = np.percentile(ret_vals, WINSOR_PCT), np.percentile(ret_vals, 100 - WINSOR_PCT)
fwd_ret_oos = fwd_ret_oos.clip(lower=lo_w, upper=hi_w)

print(f"  Forward returns: OOS rows={len(fwd_ret_oos)}, "
      f"winsorise [{lo_w:.3f}, {hi_w:.3f}]")

# Melt to long form: (date, ticker_yf, fwd_ret_20d)
fwd_long = (fwd_ret_oos
            .reset_index()
            .melt(id_vars='date', var_name='ticker_yf', value_name='fwd_ret_20d')
            .dropna(subset=['fwd_ret_20d']))
print(f"  Forward return observations: {len(fwd_long):,}")


# ── STEP 2: BUILD TICKER MAP (underscore image name → yfinance dot ticker) ───

# dataset_mapping.csv has proper yfinance tickers for the hybrid models
mapping = pd.read_csv(MAPPING_CSV, usecols=['ticker', 'image_path'])
mapping['img_base'] = mapping['image_path'].apply(os.path.basename)

# Also build a simple underscore→dot dict for the baseline images
def underscore_to_yf(s):
    """ADYEN_AS → ADYEN.AS, NDA-FI_HE → NDA-FI.HE"""
    parts = s.rsplit('_', 1)
    return '.'.join(parts) if len(parts) == 2 else s


# ── STEP 3: LOAD PREDICTIONS ─────────────────────────────────────────────────

def load_baseline():
    npy_files = glob.glob(os.path.join(BASELINE_NPY_DIR, 'baseline_ensemble_auc*.npy'))
    if not npy_files:
        raise FileNotFoundError(f"No .npy in {BASELINE_NPY_DIR}")
    probs = np.load(sorted(npy_files)[-1])
    files = sorted(f for f in os.listdir(BASELINE_IMG_DIR) if f.endswith('.png'))
    records = []
    for i, fname in enumerate(files):
        stem  = fname.replace('.png', '')
        parts = stem.rsplit('_', 2)          # TICKER_EXCHANGE | YYYYMMDD | label
        ticker_raw = parts[0]                # e.g. ADYEN_AS
        ticker_yf  = underscore_to_yf(ticker_raw)
        records.append({
            'ticker_yf': ticker_yf,
            'date':      pd.Timestamp(parts[1]),
            'label':     int(parts[2]),
            'Prob_Up':   float(probs[i]),
        })
    df = pd.DataFrame(records)
    df = df[df['date'] >= '2021-01-01'].copy()
    print(f"  Baseline CNN   : {len(df):,} OOS predictions")
    return df

def load_hybrid(csv_path, name):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    df['img_base'] = df['image_path'].apply(os.path.basename)
    df = df.merge(mapping[['img_base', 'ticker']].rename(columns={'ticker': 'ticker_yf'}),
                  on='img_base', how='left')
    n_miss = df['ticker_yf'].isna().sum()
    if n_miss:
        print(f"  WARNING: {n_miss} unmatched image paths in {name}")
    df = df[df['date'] >= '2021-01-01'].dropna(subset=['ticker_yf']).copy()
    print(f"  {name:<20}: {len(df):,} OOS predictions")
    return df[['ticker_yf', 'date', 'label', 'Prob_Up']]

print("\nLoading predictions ...")
preds = {}
preds['Baseline CNN']    = load_baseline()
preds['EfficientNet-B2'] = load_hybrid(PRED_FILES['EfficientNet-B2'], 'EfficientNet-B2')
preds['ConvNeXt V2']     = load_hybrid(PRED_FILES['ConvNeXt V2'],     'ConvNeXt V2')


# ── STEP 4: MERGE PREDICTIONS WITH ACTUAL FORWARD RETURNS ────────────────────

print("\nMerging with actual forward returns ...")
for name in preds:
    before = len(preds[name])
    preds[name] = preds[name].merge(fwd_long, on=['ticker_yf', 'date'], how='inner')
    after = len(preds[name])
    print(f"  {name:<20}: {before:,} -> {after:,} matched "
          f"({100 * after / before:.1f}%)")


# ── STEP 5: PORTFOLIO STATISTICS ─────────────────────────────────────────────

def newey_west_se(r, n_lags):
    n = len(r)
    r_dm = r - r.mean()
    var = np.dot(r_dm, r_dm) / n
    for lag in range(1, n_lags + 1):
        cov = np.dot(r_dm[lag:], r_dm[:-lag]) / n
        weight = 1 - lag / (n_lags + 1)
        var += 2 * weight * cov
    return np.sqrt(max(var, 0) / n)

def portfolio_stats(r_series, periods_per_year, nw_lags=1):
    r = r_series.dropna().values
    if len(r) < 5:
        return {}
    n         = len(r)
    mean_r    = r.mean()
    std_r     = r.std(ddof=1)
    nw_se     = newey_west_se(r, nw_lags)
    t_stat    = mean_r / nw_se if nw_se > 0 else 0.0
    p_val     = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    sharpe    = mean_r / std_r * np.sqrt(periods_per_year) if std_r > 0 else 0.0
    neg       = r[r < 0]
    sortino   = (mean_r / (neg.std(ddof=1) + 1e-10) * np.sqrt(periods_per_year)
                 if len(neg) > 1 else np.nan)
    cum       = np.cumprod(1 + r)
    dd        = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)
    return {
        'Ann. Return (%)': mean_r * periods_per_year * 100,
        'Sharpe':          sharpe,
        'Sortino':         sortino,
        'Max DD':          dd.min(),
        'Hit Rate':        (r > 0).mean(),
        'NW t-stat':       t_stat,
        'p-value':         p_val,
        'N periods':       n,
    }


# ── STEP 6: NON-OVERLAPPING BACKTEST ─────────────────────────────────────────

def run_nonoverlapping(df_pred, top_q=TOP_Q, cost_rt=COST_RT):
    all_dates = sorted(df_pred['date'].unique())
    rebal_dates = all_dates[::HORIZON]   # every 20th trading day
    records = []
    for d in rebal_dates:
        day = df_pred[df_pred['date'] == d].dropna(subset=['fwd_ret_20d'])
        if len(day) < 10:
            continue
        q_hi = day['Prob_Up'].quantile(1 - top_q)
        q_lo = day['Prob_Up'].quantile(top_q)
        longs  = day[day['Prob_Up'] >= q_hi]['fwd_ret_20d']
        shorts = day[day['Prob_Up'] <= q_lo]['fwd_ret_20d']
        if len(longs) == 0 or len(shorts) == 0:
            continue
        gross = longs.mean() - shorts.mean()
        net   = gross - cost_rt          # 10 bps deducted from the spread
        records.append({'date': d, 'gross': gross, 'net': net,
                        'n_long': len(longs), 'n_short': len(shorts)})
    return pd.DataFrame(records).set_index('date')


# ── STEP 7: OVERLAPPING (JEGADEESH-TITMAN) BACKTEST ─────────────────────────

def run_overlapping(df_pred, top_q=TOP_Q, cost_rt=COST_RT):
    """
    Trade daily. On each day t, form a cohort of top/bottom quintile stocks.
    Each cohort is held for HORIZON days. The daily portfolio return is the
    equal-weighted average of all active cohorts (up to HORIZON cohorts per day).
    """
    all_dates = sorted(df_pred['date'].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    cohort_returns = {}  # {formation_date: pd.Series of daily cohort returns}

    for d in all_dates:
        day = df_pred[df_pred['date'] == d].dropna(subset=['fwd_ret_20d'])
        if len(day) < 10:
            continue
        q_hi = day['Prob_Up'].quantile(1 - top_q)
        q_lo = day['Prob_Up'].quantile(top_q)
        longs  = day[day['Prob_Up'] >= q_hi]['fwd_ret_20d']
        shorts = day[day['Prob_Up'] <= q_lo]['fwd_ret_20d']
        if len(longs) == 0 or len(shorts) == 0:
            continue
        # The cohort earns its full H-period return spread on the exit date
        cohort_returns[d] = longs.mean() - shorts.mean()

    if not cohort_returns:
        return pd.DataFrame()

    # Build daily return series: on each day, average over the H active cohorts
    daily = {}
    for exit_date in all_dates:
        exit_idx = date_to_idx[exit_date]
        # Cohorts formed on the previous H days contribute their realized return today
        cohort_values = []
        for form_date, gross_ret in cohort_returns.items():
            form_idx = date_to_idx[form_date]
            if 0 < exit_idx - form_idx <= HORIZON:
                cohort_values.append(gross_ret / HORIZON)
        if cohort_values:
            daily[exit_date] = np.mean(cohort_values)

    daily_s = pd.Series(daily).sort_index()
    # Net: subtract daily cost amortised over the holding period
    daily_net = daily_s - cost_rt / HORIZON
    return pd.DataFrame({'gross': daily_s, 'net': daily_net})


# ── STEP 8: BENCHMARK — EW BUY-AND-HOLD EURO STOXX 50 ───────────────────────

def compute_benchmark(all_dates):
    records = []
    for d in all_dates:
        day_ret = fwd_long[fwd_long['date'] == d]['fwd_ret_20d']
        if len(day_ret) > 0:
            records.append({'date': d, 'ret': day_ret.mean()})
    return pd.DataFrame(records).set_index('date')['ret']


# ── STEP 9: RUN EVERYTHING ───────────────────────────────────────────────────

print("\n" + "="*72)
print("  NON-OVERLAPPING BACKTEST  (primary, every 20 trading days)")
print("  Quintile L/S | Equal-Weight | Dollar-Neutral | 10 bps round-trip")
print("="*72)

periods_per_year_NO = 252 / HORIZON  # ≈ 12.6

no_rets   = {}
row_store = []

for name, df_p in preds.items():
    r = run_nonoverlapping(df_p)
    no_rets[name] = r
    for col in ['gross', 'net']:
        st = portfolio_stats(r[col], periods_per_year_NO, nw_lags=1)
        tag = f"{name} [{col}]"
        print(f"\n  {tag}")
        print(f"    Ann. Return : {st['Ann. Return (%)']:+.2f}%")
        print(f"    Sharpe      : {st['Sharpe']:.3f}")
        print(f"    Sortino     : {st['Sortino']:.3f}")
        print(f"    Max DD      : {st['Max DD']:.3f}")
        print(f"    Hit Rate    : {st['Hit Rate']:.3f}")
        print(f"    NW t-stat   : {st['NW t-stat']:.2f}  (p = {st['p-value']:.3f})")
        print(f"    N periods   : {st['N periods']}")
        row_store.append({'Model': name, 'Type': col, **st})

# Benchmark
all_dates_common = sorted(set.union(*[set(no_rets[n].index) for n in no_rets]))
bm = compute_benchmark(all_dates_common)
bm_st = portfolio_stats(bm, periods_per_year_NO, nw_lags=1)
print(f"\n  Benchmark [EW Buy-and-Hold]")
print(f"    Ann. Return : {bm_st['Ann. Return (%)']:+.2f}%")
print(f"    Sharpe      : {bm_st['Sharpe']:.3f}")
print(f"    Max DD      : {bm_st['Max DD']:.3f}")

print("\n" + "="*72)
print("  OVERLAPPING BACKTEST  (robustness, Newey-West 21 lags)")
print("="*72)

periods_per_year_OV = 252   # daily observations

ov_rets = {}
for name, df_p in preds.items():
    r = run_overlapping(df_p)
    if len(r) == 0:
        continue
    ov_rets[name] = r
    for col in ['gross', 'net']:
        st = portfolio_stats(r[col], periods_per_year_OV, nw_lags=21)
        print(f"\n  {name} [{col}]  (overlapping)")
        print(f"    Ann. Return : {st['Ann. Return (%)']:+.2f}%")
        print(f"    Sharpe      : {st['Sharpe']:.3f}")
        print(f"    NW t-stat   : {st['NW t-stat']:.2f}  (p = {st['p-value']:.3f})")
        print(f"    N obs       : {st['N periods']}")
        row_store.append({'Model': name + ' [OV]', 'Type': col, **st})


# ── STEP 10: PLOTS ───────────────────────────────────────────────────────────

COLORS = {
    'Baseline CNN':    '#616161',
    'EfficientNet-B2': '#1565C0',
    'ConvNeXt V2':     '#E65100',
}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Panel A: Cumulative net returns (non-overlapping)
ax = axes[0]
for name, r in no_rets.items():
    cum = (1 + r['net']).cumprod()
    ax.plot(cum.index, cum.values, lw=2, color=COLORS[name], label=f"{name} (net)")
# Benchmark on same rebalance dates
bm_cum = (1 + bm).cumprod()
ax.plot(bm_cum.index, bm_cum.values, lw=1.5, ls='--', color='black',
        label='EW Benchmark')
ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.6)
ax.set(title='Cumulative Returns — Non-Overlapping Quintile L/S\n'
             f'Net of {int(COST_RT*10000)} bps Round-Trip | Rebalanced Every {HORIZON} Days',
       xlabel='Date', ylabel='Cumulative Return')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel B: Annual net return bars
ax2 = axes[1]
years = sorted({d.year for r in no_rets.values() for d in r.index})
x = np.arange(len(years))
w = 0.25
for i, (name, r) in enumerate(no_rets.items()):
    ann_rets = []
    for yr in years:
        yr_r = r[r.index.year == yr]['net']
        if len(yr_r) > 0:
            ann_rets.append(yr_r.mean() * periods_per_year_NO * 100)
        else:
            ann_rets.append(np.nan)
    ax2.bar(x + (i - 1) * w, ann_rets, w,
            color=COLORS[name], alpha=0.85, label=name)
ax2.axhline(0, color='k', lw=1)
ax2.set(xticks=x, xticklabels=years,
        ylabel='Annualised Return (%)',
        title='Annual Performance — Non-Overlapping Quintile L/S (Net)')
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Corrected Backtest: Actual 20-Day Forward Returns | OOS 2021–2026',
             fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT, '01_corrected_backtest.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: 01_corrected_backtest.png -> {OUT}")


# ── STEP 11: SAVE STATS CSV ──────────────────────────────────────────────────

pd.DataFrame(row_store).to_csv(os.path.join(OUT, 'corrected_portfolio_stats.csv'), index=False)
print(f"Saved: corrected_portfolio_stats.csv")

print(f"\n{'='*72}")
print(f"  Done.  Outputs -> {OUT}")
print(f"{'='*72}")
