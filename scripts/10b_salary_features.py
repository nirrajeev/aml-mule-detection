"""
10b_salary_features.py — Salary Cycle Anomaly Detection

Detects accounts with regular monthly credit inflows (salary pattern)
that then show abnormal outflow behavior — a key mule pattern.

Features:
  - has_salary_pattern     : bool — regular monthly credits detected
  - salary_regularity      : coefficient of variation of monthly credit intervals
  - salary_month_count     : number of months with a large credit
  - post_salary_drain_ratio: fraction of salary credit drained within 3 days
  - salary_to_debit_ratio  : ratio of salary-like credits to total debits

Run: python scripts/10b_salary_features.py
Output: cache/features/salary_features.parquet
"""

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
BATCH_RG = 20

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
# What counts as a "salary-like" credit:
#   - txn_type == 'C' (credit into account)
#   - amount > 10,000 (meaningful inflow, not petty cash)
#   - amount < 500,000 (not a large business transfer)
#   - MCC is salary/payroll related OR counterparty is employer-like
#     (we approximate this by looking for regular monthly patterns)
SALARY_MIN = 10_000
SALARY_MAX = 500_000
# ─────────────────────────────────────────────────────────────────────────────

txn_path     = CACHE / "transactions_full.parquet"
pf           = pq.ParquetFile(txn_path)
total_rg     = pq.read_metadata(txn_path).num_row_groups
batch_ranges = [(i, min(i + BATCH_RG, total_rg)) for i in range(0, total_rg, BATCH_RG)]
log(f"Row groups: {total_rg}  |  Batches: {len(batch_ranges)}")

# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — Collect monthly large credits per account
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== PASS 1: Monthly large credits ===")

pass1_chunks = []

for b_idx, (rg_start, rg_end) in enumerate(batch_ranges):
    log(f"  Batch {b_idx+1}/{len(batch_ranges)}")

    tbl = read_rg_batch(pf, rg_start, rg_end,
                        columns=['account_id', 'transaction_timestamp',
                                 'txn_type', 'amount'])
    if tbl is None:
        continue

    df = tbl.to_pandas()
    del tbl

    df['ts'] = pd.to_datetime(df['transaction_timestamp'],
                               infer_datetime_format=True, errors='coerce')
    df = df.dropna(subset=['ts'])
    df['abs_amt'] = df['amount'].abs()

    # Keep only large credits in salary range
    sal = df[
        (df['txn_type'] == 'C') &
        (df['abs_amt'] >= SALARY_MIN) &
        (df['abs_amt'] <= SALARY_MAX)
    ].copy()

    if len(sal) == 0:
        del df, sal
        gc.collect()
        continue

    sal['ym']      = sal['ts'].dt.year * 100 + sal['ts'].dt.month
    sal['day']     = sal['ts'].dt.day
    sal['ts_epoch']= sal['ts'].astype('int64') // 1_000_000_000

    # Per account per month: largest single credit (salary is usually one payment)
    monthly = sal.groupby(['account_id', 'ym']).agg(
        max_credit_amt   = ('abs_amt',   'max'),
        max_credit_day   = ('day',       lambda x: x.iloc[x.values.argmax()]),
        max_credit_epoch = ('ts_epoch',  lambda x: x.iloc[x.values.argmax()]),
        n_large_credits  = ('abs_amt',   'count'),
    ).reset_index()

    pass1_chunks.append(monthly)
    del df, sal, monthly
    gc.collect()

log("  Combining...")
monthly_all = pd.concat(pass1_chunks, ignore_index=True)
del pass1_chunks
gc.collect()

monthly_final = monthly_all.groupby(['account_id', 'ym']).agg(
    max_credit_amt   = ('max_credit_amt',   'max'),
    max_credit_day   = ('max_credit_day',   'first'),
    max_credit_epoch = ('max_credit_epoch', 'first'),
    n_large_credits  = ('n_large_credits',  'sum'),
).reset_index()

del monthly_all
gc.collect()
log(f"  Monthly large credit records: {len(monthly_final):,}")

# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — Compute salary regularity per account
# A genuine salary arrives on roughly the same day each month.
# Regularity = low CV of credit day-of-month across months.
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== PASS 2: Salary regularity ===")

salary_rows = []

for acct, grp in monthly_final.groupby('account_id'):
    grp = grp.sort_values('ym').reset_index(drop=True)
    n_months = len(grp)

    if n_months < 2:
        # Can't assess regularity from a single month
        salary_rows.append({
            'account_id':         acct,
            'salary_month_count': n_months,
            'salary_regularity':  1.0,    # max disorder (unknown)
            'has_salary_pattern': 0,
            'salary_day_mean':    float(grp['max_credit_day'].mean()),
        })
        continue

    days = grp['max_credit_day'].values.astype(float)
    mean_day = days.mean()
    std_day  = days.std()
    # Coefficient of variation — low = regular (salary), high = irregular
    cv = std_day / (mean_day + 1e-6)

    # Salary pattern: at least 3 months, CV < 0.3 (arrives within ~3 days of same date)
    has_salary = int(n_months >= 3 and cv < 0.3)

    salary_rows.append({
        'account_id':         acct,
        'salary_month_count': n_months,
        'salary_regularity':  float(cv),
        'has_salary_pattern': has_salary,
        'salary_day_mean':    float(mean_day),
    })

salary_df = pd.DataFrame(salary_rows)
log(f"  Accounts with salary analysis: {len(salary_df):,}")
log(f"  With salary pattern (CV<0.3, >=3 months): {salary_df['has_salary_pattern'].sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
# PASS 3 — Post-salary drain: how quickly is the salary spent after arrival?
# Mule accounts drain salary credits within 1-3 days (rapid pass-through).
# Legit accounts spend gradually over the month.
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== PASS 3: Post-salary drain ratio ===")

# Get salary credit epochs per account (the exact timestamp of each salary)
salary_accts = set(salary_df[salary_df['has_salary_pattern'] == 1]['account_id'])
salary_epochs = (
    monthly_final[monthly_final['account_id'].isin(salary_accts)]
    [['account_id', 'max_credit_epoch', 'max_credit_amt']]
    .copy()
)

# For each salary credit, count debits within 3 days (259200 seconds)
DRAIN_WINDOW_SEC = 3 * 86400

pass3_chunks = []

for b_idx, (rg_start, rg_end) in enumerate(batch_ranges):
    log(f"  P3 Batch {b_idx+1}/{len(batch_ranges)}")

    tbl = read_rg_batch(pf, rg_start, rg_end,
                        columns=['account_id', 'transaction_timestamp',
                                 'txn_type', 'amount'])
    if tbl is None:
        continue

    df = tbl.to_pandas()
    del tbl

    # Only salary accounts, only debits
    df = df[
        (df['account_id'].isin(salary_accts)) &
        (df['txn_type'] == 'D')
    ].copy()

    if len(df) == 0:
        del df; gc.collect(); continue

    df['ts'] = pd.to_datetime(df['transaction_timestamp'],
                               infer_datetime_format=True, errors='coerce')
    df = df.dropna(subset=['ts'])
    df['ts_epoch'] = df['ts'].astype('int64') // 1_000_000_000
    df['abs_amt']  = df['amount'].abs()

    # For each debit, check if it falls within DRAIN_WINDOW_SEC of any salary credit
    debits = df[['account_id', 'ts_epoch', 'abs_amt']].copy()
    debits = debits.merge(salary_epochs, on='account_id', how='inner')

    # Is this debit within drain window after salary?
    debits['time_after_salary'] = debits['ts_epoch'] - debits['max_credit_epoch']
    drain = debits[
        (debits['time_after_salary'] >= 0) &
        (debits['time_after_salary'] <= DRAIN_WINDOW_SEC)
    ].copy()

    if len(drain) == 0:
        del df, debits, drain; gc.collect(); continue

    drain_agg = drain.groupby('account_id').agg(
        post_salary_debit_amt   = ('abs_amt', 'sum'),
        post_salary_debit_count = ('abs_amt', 'count'),
    ).reset_index()

    pass3_chunks.append(drain_agg)
    del df, debits, drain, drain_agg
    gc.collect()

log("  Combining pass 3...")
if pass3_chunks:
    drain_all = pd.concat(pass3_chunks, ignore_index=True)
    drain_final = drain_all.groupby('account_id').agg(
        post_salary_debit_amt   = ('post_salary_debit_amt',   'sum'),
        post_salary_debit_count = ('post_salary_debit_count', 'sum'),
    ).reset_index()
    del pass3_chunks, drain_all
    gc.collect()
else:
    drain_final = pd.DataFrame(columns=['account_id', 'post_salary_debit_amt',
                                         'post_salary_debit_count'])

# Total salary received per account
total_salary = (
    monthly_final.groupby('account_id')['max_credit_amt']
    .sum().reset_index().rename(columns={'max_credit_amt': 'total_salary_amt'})
)

# Merge and compute drain ratio
drain_final = drain_final.merge(total_salary, on='account_id', how='left')
drain_final['post_salary_drain_ratio'] = (
    drain_final['post_salary_debit_amt'] /
    (drain_final['total_salary_amt'] + 1)
).clip(upper=2.0)

log(f"  Accounts with post-salary drain data: {len(drain_final):,}")

# ══════════════════════════════════════════════════════════════════════════════
# MERGE all salary features
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== Merging salary features ===")

txn_feats = pd.read_parquet(FEATS / "txn_features.parquet", columns=['account_id'])
all_accts = txn_feats[['account_id']].copy()

feats = all_accts.merge(salary_df, on='account_id', how='left')
feats = feats.merge(drain_final[['account_id', 'post_salary_debit_amt',
                                  'post_salary_debit_count',
                                  'post_salary_drain_ratio']], on='account_id', how='left')

# Fill nulls — accounts with no large credits get defaults
feats['salary_month_count']       = feats['salary_month_count'].fillna(0)
feats['salary_regularity']        = feats['salary_regularity'].fillna(1.0)
feats['has_salary_pattern']       = feats['has_salary_pattern'].fillna(0).astype(int)
feats['salary_day_mean']          = feats['salary_day_mean'].fillna(0)
feats['post_salary_debit_amt']    = feats['post_salary_debit_amt'].fillna(0)
feats['post_salary_debit_count']  = feats['post_salary_debit_count'].fillna(0)
feats['post_salary_drain_ratio']  = feats['post_salary_drain_ratio'].fillna(0)

# Combined mule signal: has salary pattern AND drains it quickly
feats['salary_mule_signal'] = (
    (feats['has_salary_pattern'] == 1) &
    (feats['post_salary_drain_ratio'] > 0.5)
).astype(int)

log(f"  Final shape: {feats.shape}")
log(f"  has_salary_pattern: {feats['has_salary_pattern'].sum():,} accounts")
log(f"  salary_mule_signal: {feats['salary_mule_signal'].sum():,} accounts")

# ── Save ──────────────────────────────────────────────────────────────────
out = FEATS / "salary_features.parquet"
feats.to_parquet(out, index=False)
log(f"  Saved → {out}  ({out.stat().st_size/1e6:.1f} MB)")

# ══════════════════════════════════════════════════════════════════════════════
log("\n=== Validation: mule vs legit ===")
# ══════════════════════════════════════════════════════════════════════════════

labels = pd.read_parquet(DATA / "train_labels.parquet")
check  = feats.merge(labels[['account_id', 'is_mule']], on='account_id', how='inner')

key_cols = [
    'salary_month_count', 'salary_regularity', 'has_salary_pattern',
    'post_salary_drain_ratio', 'post_salary_debit_count', 'salary_mule_signal',
]

print(f"\n  {'Feature':<30s} {'Mule':>10} {'Legit':>10} {'p-value':>10}")
print("  " + "-"*65)
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
    print(f"  {col:<30s} {m.mean():>10.4f} {l.mean():>10.4f} {pval:>10.2e} {sig}")

# Check salary_mule_signal on previously missed mules
oof_v3 = pd.read_parquet(FEATS / "oof_lgbm_v3.parquet")
missed = oof_v3[(oof_v3.is_mule == 1) & (oof_v3.oof_lgbm_v3 < 0.335)]
missed_sal = missed.merge(feats, on='account_id', how='left')

log(f"\n  Salary signal on previously missed mules ({len(missed):,}):")
log(f"    has_salary_pattern   : {int(missed_sal['has_salary_pattern'].sum()):,}")
log(f"    salary_mule_signal   : {int(missed_sal['salary_mule_signal'].sum()):,}")
log(f"    mean drain_ratio     : {missed_sal['post_salary_drain_ratio'].mean():.4f}")

log("\nDone. Next: python scripts/11_model_v4.py")
