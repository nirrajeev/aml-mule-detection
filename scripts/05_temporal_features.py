"""
05_temporal_features.py — Temporal Feature Engineering (pure pandas, no cuDF)
Detects:
  1. Dormancy gaps     — longest inactivity period per account
  2. Post-mobile spike — transaction surge after mobile number update
  3. Suspicious window — peak 3-month activity burst (for temporal IoU)

Design:
  - Reads transactions_full.parquet in batches of 20 row groups (~20M rows each)
  - Aggregates per batch → combines on CPU, never loads full 397M rows at once
  - Tracks cross-batch last-seen timestamp in a dict to get correct dormancy gaps
  - No cuDF, no lambdas — pure pandas + pyarrow

Run: python scripts/05_temporal_features.py
Output: cache/features/temporal_features.parquet
"""

import os
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
from scipy.stats import mannwhitneyu
import gc
import warnings
warnings.filterwarnings('ignore')

CACHE    = Path("/home/niranjan/AML/cache")
DATA     = Path("/home/niranjan/AML/data")
FEATS    = CACHE / "features"
FEATS.mkdir(exist_ok=True)

BATCH_RG = 20   # row groups per batch

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def read_rg_batch(pf, rg_start, rg_end, columns=None):
    tables = []
    for i in range(rg_start, rg_end):
        try:
            tables.append(pf.read_row_group(i, columns=columns))
        except Exception:
            break
    return pa.concat_tables(tables) if tables else None

# ─────────────────────────────────────────────────────────────────────────────
log("Loading account static data...")
# ─────────────────────────────────────────────────────────────────────────────

accounts = pd.read_parquet(
    DATA / "accounts.parquet",
    columns=['account_id', 'last_mobile_update_date']
)
accounts['last_mobile_update_date'] = pd.to_datetime(
    accounts['last_mobile_update_date'], errors='coerce'
)
accounts['mobile_update_epoch'] = (
    accounts['last_mobile_update_date']
    .astype('int64') // 1_000_000_000
).where(accounts['last_mobile_update_date'].notna(), other=-1)

mobile_epoch_lookup = accounts.set_index('account_id')['mobile_update_epoch'].to_dict()

log(f"  Total accounts: {len(accounts):,}")
log(f"  With mobile update date: {(accounts['mobile_update_epoch'] != -1).sum():,}")

txn_path     = CACHE / "transactions_full.parquet"
pf           = pq.ParquetFile(txn_path)
total_rg     = pq.read_metadata(txn_path).num_row_groups
batch_ranges = [(i, min(i + BATCH_RG, total_rg)) for i in range(0, total_rg, BATCH_RG)]
log(f"  Row groups: {total_rg}  |  Batches: {len(batch_ranges)}")

# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — Dormancy gaps + transaction time bounds
#
# Cross-batch problem: an account's transactions span multiple batches.
# We track last_seen_epoch per account in a dict that survives across batches.
# Gap = time between last_seen (end of previous batch) and first tx (this batch).
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== PASS 1: Dormancy gaps ===")

last_seen_epoch = {}   # account_id → epoch of last seen transaction (int)
max_gap_days    = {}   # account_id → largest gap seen so far (float)
pass1_aggs      = []   # list of per-batch agg DataFrames

for b_idx, (rg_start, rg_end) in enumerate(batch_ranges):
    log(f"  P1 {b_idx+1}/{len(batch_ranges)}  rg {rg_start}:{rg_end}")

    tbl = read_rg_batch(pf, rg_start, rg_end,
                        columns=['account_id', 'transaction_timestamp'])
    if tbl is None:
        continue

    df = tbl.to_pandas()
    del tbl

    df['ts'] = pd.to_datetime(df['transaction_timestamp'],
                               infer_datetime_format=True, errors='coerce')
    df = df.dropna(subset=['ts'])
    df['ts_epoch'] = df['ts'].astype('int64') // 1_000_000_000

    # Per-account min/max epoch in this batch
    batch_agg = df.groupby('account_id').agg(
        batch_min_epoch = ('ts_epoch', 'min'),
        batch_max_epoch = ('ts_epoch', 'max'),
        batch_txn_count = ('ts_epoch', 'count'),
    ).reset_index()

    # Update cross-batch gap tracking
    for row in batch_agg.itertuples(index=False):
        acct  = row.account_id
        b_min = row.batch_min_epoch
        b_max = row.batch_max_epoch

        if acct in last_seen_epoch:
            gap_days = (b_min - last_seen_epoch[acct]) / 86400
            if gap_days > max_gap_days.get(acct, 0):
                max_gap_days[acct] = gap_days

        if acct not in last_seen_epoch or b_max > last_seen_epoch[acct]:
            last_seen_epoch[acct] = b_max

    pass1_aggs.append(batch_agg)

    del df, batch_agg
    gc.collect()

log("  Combining pass 1...")
all_agg = pd.concat(pass1_aggs, ignore_index=True)
del pass1_aggs
gc.collect()

# Global first/last ts and total count per account
txn_bounds = all_agg.groupby('account_id').agg(
    first_ts_epoch  = ('batch_min_epoch', 'min'),
    last_ts_epoch   = ('batch_max_epoch', 'max'),
    total_txn_count = ('batch_txn_count', 'sum'),
).reset_index()

del all_agg
gc.collect()

dormancy_df = pd.DataFrame([
    {'account_id': acct, 'dormancy_gap_days': gap}
    for acct, gap in max_gap_days.items()
])
log(f"  Accounts with at least 1 gap: {len(dormancy_df):,}")
log(f"  Transaction bounds: {txn_bounds.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — Post-mobile-update spike features
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== PASS 2: Post-mobile-update spike ===")

pass2_aggs = []

for b_idx, (rg_start, rg_end) in enumerate(batch_ranges):
    log(f"  P2 {b_idx+1}/{len(batch_ranges)}")

    tbl = read_rg_batch(pf, rg_start, rg_end,
                        columns=['account_id', 'transaction_timestamp', 'amount'])
    if tbl is None:
        continue

    df = tbl.to_pandas()
    del tbl

    df['ts']       = pd.to_datetime(df['transaction_timestamp'],
                                     infer_datetime_format=True, errors='coerce')
    df             = df.dropna(subset=['ts'])
    df['ts_epoch'] = df['ts'].astype('int64') // 1_000_000_000
    df['abs_amt']  = df['amount'].abs()

    # Map mobile update epoch via lookup (vectorised using map)
    df['mobile_epoch'] = df['account_id'].map(mobile_epoch_lookup).fillna(-1).astype('int64')

    # Pre/post flags — only meaningful when mobile date exists (epoch > 0)
    has_mobile     = df['mobile_epoch'] > 0
    df['is_post']  = (has_mobile & (df['ts_epoch'] >= df['mobile_epoch'])).astype(int)
    df['is_pre']   = (has_mobile & (df['ts_epoch'] <  df['mobile_epoch'])).astype(int)
    df['post_amt'] = df['abs_amt'].where(df['is_post'] == 1, other=0.0)
    df['pre_amt']  = df['abs_amt'].where(df['is_pre']  == 1, other=0.0)

    agg = df.groupby('account_id').agg(
        pre_mobile_txn_count  = ('is_pre',   'sum'),
        post_mobile_txn_count = ('is_post',  'sum'),
        pre_mobile_amt        = ('pre_amt',  'sum'),
        post_mobile_amt       = ('post_amt', 'sum'),
    ).reset_index()

    pass2_aggs.append(agg)
    del df, agg
    gc.collect()

log("  Combining pass 2...")
p2 = pd.concat(pass2_aggs, ignore_index=True)
del pass2_aggs
gc.collect()

mobile_feats = p2.groupby('account_id').agg(
    pre_mobile_txn_count  = ('pre_mobile_txn_count',  'sum'),
    post_mobile_txn_count = ('post_mobile_txn_count', 'sum'),
    pre_mobile_amt        = ('pre_mobile_amt',         'sum'),
    post_mobile_amt       = ('post_mobile_amt',        'sum'),
).reset_index()

mobile_feats['post_mobile_txn_ratio'] = (
    mobile_feats['post_mobile_txn_count'] /
    (mobile_feats['pre_mobile_txn_count'] + 1)
)
mobile_feats['post_mobile_amt_ratio'] = (
    mobile_feats['post_mobile_amt'] /
    (mobile_feats['pre_mobile_amt'] + 1)
)

del p2
gc.collect()
log(f"  Mobile features: {mobile_feats.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# PASS 3 — Monthly transaction density → peak 3-month suspicious window
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== PASS 3: Monthly density for suspicious window ===")

pass3_aggs = []

for b_idx, (rg_start, rg_end) in enumerate(batch_ranges):
    log(f"  P3 {b_idx+1}/{len(batch_ranges)}")

    tbl = read_rg_batch(pf, rg_start, rg_end,
                        columns=['account_id', 'transaction_timestamp', 'amount'])
    if tbl is None:
        continue

    df = tbl.to_pandas()
    del tbl

    df['ts']     = pd.to_datetime(df['transaction_timestamp'],
                                   infer_datetime_format=True, errors='coerce')
    df           = df.dropna(subset=['ts'])
    df['abs_amt']= df['amount'].abs()
    df['ym']     = df['ts'].dt.year * 100 + df['ts'].dt.month  # e.g. 202307

    monthly = df.groupby(['account_id', 'ym']).agg(
        month_txn_count = ('abs_amt', 'count'),
        month_total_amt = ('abs_amt', 'sum'),
    ).reset_index()

    pass3_aggs.append(monthly)
    del df, monthly
    gc.collect()

log("  Combining pass 3...")
monthly_all = pd.concat(pass3_aggs, ignore_index=True)
del pass3_aggs
gc.collect()

monthly_final = monthly_all.groupby(['account_id', 'ym']).agg(
    month_txn_count = ('month_txn_count', 'sum'),
    month_total_amt = ('month_total_amt', 'sum'),
).reset_index()

del monthly_all
gc.collect()
log(f"  Monthly records: {len(monthly_final):,}")

# ─────────────────────────────────────────────────────────────────────────────
# Find peak 3-month window per account (CPU — 160k accounts × ~60 months each)
# ─────────────────────────────────────────────────────────────────────────────
log("  Finding peak activity windows...")

WINDOW_MONTHS = 3
windows = []

for acct, grp in monthly_final.groupby('account_id'):
    grp    = grp.sort_values('ym').reset_index(drop=True)
    n      = len(grp)
    counts = grp['month_txn_count'].values
    amts   = grp['month_total_amt'].values
    yms    = grp['ym'].values

    # Find contiguous window of WINDOW_MONTHS with the most transactions
    w          = min(WINDOW_MONTHS, n)
    best_i     = int(np.argmax(
        [counts[i:i+w].sum() for i in range(n - w + 1)]
    ))
    end_i      = min(best_i + w - 1, n - 1)

    start_ym   = int(yms[best_i])
    end_ym     = int(yms[end_i])
    start_yr, start_mo = start_ym // 100, start_ym % 100
    end_yr,   end_mo   = end_ym   // 100, end_ym   % 100

    win_start  = pd.Timestamp(year=start_yr, month=start_mo, day=1)
    win_end    = pd.Timestamp(year=end_yr,   month=end_mo,   day=1) + pd.offsets.MonthEnd(1)

    windows.append({
        'account_id':        acct,
        'peak_window_start': win_start,
        'peak_window_end':   win_end,
        'peak_txn_count':    int(counts[best_i:end_i+1].sum()),
        'peak_amt':          float(amts[best_i:end_i+1].sum()),
    })

windows_df = pd.DataFrame(windows)
log(f"  Windows computed: {len(windows_df):,}")

# ══════════════════════════════════════════════════════════════════════════════
# MERGE ALL TEMPORAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== Merging all temporal features ===")

feats = txn_bounds.copy()
feats = feats.merge(dormancy_df,  on='account_id', how='left')
feats = feats.merge(mobile_feats, on='account_id', how='left')
feats = feats.merge(windows_df,   on='account_id', how='left')

# Join mobile update date for suspicious window override
feats = feats.merge(
    accounts[['account_id', 'last_mobile_update_date']],
    on='account_id', how='left'
)

# Fill nulls
feats['dormancy_gap_days']       = feats['dormancy_gap_days'].fillna(0)
feats['pre_mobile_txn_count']    = feats['pre_mobile_txn_count'].fillna(0)
feats['post_mobile_txn_count']   = feats['post_mobile_txn_count'].fillna(0)
feats['post_mobile_txn_ratio']   = feats['post_mobile_txn_ratio'].fillna(1.0)
feats['post_mobile_amt_ratio']   = feats['post_mobile_amt_ratio'].fillna(1.0)

# ── Derived features ──────────────────────────────────────────────────────
feats['activity_span_days']      = (
    (feats['last_ts_epoch'] - feats['first_ts_epoch']) / 86400
).clip(lower=0)

feats['dormancy_fraction']       = (
    feats['dormancy_gap_days'] / (feats['activity_span_days'] + 1)
).clip(upper=1.0)

feats['is_dormant_reactivated']  = (feats['dormancy_gap_days'] > 90).astype(int)

feats['mobile_spike']            = (
    (feats['post_mobile_txn_ratio'] > 2.0) &
    (feats['post_mobile_txn_count'] > 10)
).astype(int)

# ── Suspicious window ─────────────────────────────────────────────────────
# Default: peak transaction density window
feats['suspicious_start_ts'] = feats['peak_window_start']
feats['suspicious_end_ts']   = feats['peak_window_end']

# Override with mobile update date when a spike is detected
mob_mask = feats['mobile_spike'] == 1
feats.loc[mob_mask, 'suspicious_start_ts'] = feats.loc[mob_mask, 'last_mobile_update_date']

# String format for submission
feats['suspicious_start'] = feats['suspicious_start_ts'].dt.strftime('%Y-%m-%dT%H:%M:%S')
feats['suspicious_end']   = feats['suspicious_end_ts'].dt.strftime('%Y-%m-%dT%H:%M:%S')

log(f"  Final shape: {feats.shape}")

# Null summary
nulls = feats.isnull().mean() * 100
nulls = nulls[nulls > 0].sort_values(ascending=False)
if len(nulls):
    print("\n  Null rates (non-zero only):")
    print(nulls.to_string())

# ── Save ──────────────────────────────────────────────────────────────────
out = FEATS / "temporal_features.parquet"
feats.to_parquet(out, index=False)
log(f"\n  Saved → {out}  ({out.stat().st_size/1e6:.1f} MB)")

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== Validation: mule vs legit ===")

labels = pd.read_parquet(DATA / "train_labels.parquet")
check  = feats.merge(labels[['account_id', 'is_mule']], on='account_id', how='inner')

key_cols = [
    'dormancy_gap_days', 'dormancy_fraction', 'is_dormant_reactivated',
    'post_mobile_txn_ratio', 'post_mobile_amt_ratio', 'mobile_spike',
    'peak_txn_count', 'peak_amt', 'activity_span_days',
]
key_cols = [c for c in key_cols if c in check.columns]

print(f"\n  {'Feature':<35s} {'Mule':>10} {'Legit':>10} {'p-value':>10}")
print("  " + "-"*70)
for col in key_cols:
    m = check[check.is_mule == 1][col].dropna()
    l = check[check.is_mule == 0][col].dropna()
    if len(m) < 5:
        continue
    try:
        _, pval = mannwhitneyu(m, l, alternative='two-sided')
    except Exception:
        pval = 1.0
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"  {col:<35s} {m.mean():>10.3f} {l.mean():>10.3f} {pval:>10.2e} {sig}")

n_dormant_mules = int(check[(check.is_mule==1) & (check.is_dormant_reactivated==1)]['account_id'].count())
n_mobile_mules  = int(check[(check.is_mule==1) & (check.mobile_spike==1)]['account_id'].count())
log(f"\n  Dormant-reactivated mules : {n_dormant_mules} / {int(check.is_mule.sum())}")
log(f"  Mobile-spike mules        : {n_mobile_mules} / {int(check.is_mule.sum())}")
log(f"\n  Suspicious window coverage:")
log(f"    Accounts with start ts  : {feats['suspicious_start'].notna().sum():,}")
log(f"    Accounts with end ts    : {feats['suspicious_end'].notna().sum():,}")

log("\nDone. Next: python scripts/06_model_v2.py")