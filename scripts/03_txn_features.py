"""
03_txn_features.py — Full Transaction Feature Engineering
Processes all 397M rows on GPU using cuDF.
No lambda functions in groupby — cuDF only supports named aggregations.

Run: python scripts/03_txn_features.py
Output: cache/features/txn_features.parquet
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import cudf
import cupy as cp
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from datetime import datetime
from scipy.stats import mannwhitneyu
import gc
import warnings
warnings.filterwarnings('ignore')

CACHE = Path("/home/niranjan/AML/cache")
DATA  = Path("/home/niranjan/AML/data")
FEATS = CACHE / "features"
FEATS.mkdir(exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def read_rg_batch(pf, rg_start, rg_end):
    tables = []
    for i in range(rg_start, rg_end):
        try:
            tables.append(pf.read_row_group(i))
        except Exception:
            break
    return pa.concat_tables(tables) if tables else None

txn_path = CACHE / "transactions_full.parquet"
add_path = CACHE / "transactions_additional_full.parquet"

pf_txn = pq.ParquetFile(txn_path)
pf_add = pq.ParquetFile(add_path)

total_rg     = pq.read_metadata(txn_path).num_row_groups
BATCH_RG     = 20
batch_ranges = [(i, min(i + BATCH_RG, total_rg)) for i in range(0, total_rg, BATCH_RG)]
log(f"Total row groups: {total_rg}  |  Batches: {len(batch_ranges)}  |  BATCH_RG={BATCH_RG}")

# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — Core aggregations (all named, no lambdas)
# ══════════════════════════════════════════════════════════════════════════════
log("=== PASS 1: Core aggregations ===")
pass1_aggs = []

for b_idx, (rg_start, rg_end) in enumerate(batch_ranges):
    log(f"  P1 batch {b_idx+1}/{len(batch_ranges)}  rg {rg_start}:{rg_end}")

    tbl_txn = read_rg_batch(pf_txn, rg_start, rg_end)
    tbl_add = read_rg_batch(pf_add, rg_start, rg_end)
    if tbl_txn is None:
        continue

    txn = cudf.DataFrame.from_arrow(tbl_txn)
    add = cudf.DataFrame.from_arrow(tbl_add)
    del tbl_txn, tbl_add

    # ── Parse & cast ───────────────────────────────────────────────────────
    ts_clean = (
        txn['transaction_timestamp']
        .str.replace(" ", "T")
        .str.slice(0,19)
    )

    txn['ts'] = cudf.to_datetime(ts_clean)
    txn = txn.dropna(subset=["ts"])
    txn['amount']  = txn['amount'].astype('float64')
    txn['abs_amt'] = txn['amount'].abs()
    txn['ts_epoch']= txn['ts'].astype('int64') // 1_000_000_000

    # Join balance + sub_type from additional
    txn = txn.merge(
        add[['transaction_id', 'balance_after_transaction', 'transaction_sub_type']],
        on='transaction_id', how='left'
    )
    del add
    gc.collect()

    # ── Pre-compute all binary/value columns before groupby ───────────────
    is_credit = txn['txn_type'] == 'C'
    is_debit  = txn['txn_type'] == 'D'

    txn['credit_amt']      = txn['amount'].where(is_credit, other=0.0)
    txn['debit_amt']       = txn['abs_amt'].where(is_debit, other=0.0)
    txn['is_credit']       = is_credit.astype('float32')
    txn['is_debit']        = is_debit.astype('float32')

    txn['near_50k']        = ((txn['abs_amt'] >= 45000) & (txn['abs_amt'] < 50000)).astype('float32')
    txn['round_1k']        = ((txn['abs_amt'] % 1000  == 0) & (txn['abs_amt'] > 0)).astype('float32')
    txn['round_5k']        = ((txn['abs_amt'] % 5000  == 0) & (txn['abs_amt'] > 0)).astype('float32')
    txn['round_10k']       = ((txn['abs_amt'] % 10000 == 0) & (txn['abs_amt'] > 0)).astype('float32')

    hour = txn['ts'].dt.hour
    txn['is_night']        = ((hour >= 22) | (hour <= 6)).astype('float32')
    txn['is_upi']          = txn['channel'].isin(['UPC', 'UPD']).astype('float32')
    txn['is_cash']         = (txn['transaction_sub_type'].str.lower().str.strip() == 'cash').astype('float32')
    txn['is_month_edge']   = ((txn['ts'].dt.day >= 28) | (txn['ts'].dt.day <= 3)).astype('float32')
    txn['month']           = txn['ts'].dt.month.astype('float32')

    txn['large_credit']    = ((is_credit) & (txn['abs_amt'] >= 50000)).astype('float32')
    txn['large_debit']     = ((is_debit)  & (txn['abs_amt'] >= 50000)).astype('float32')

    # ── Single groupby — all named aggs, no lambdas ────────────────────────
    agg = txn.groupby('account_id').agg(
        txn_count          = ('transaction_id',            'count'),
        total_abs_amt      = ('abs_amt',                   'sum'),
        mean_abs_amt       = ('abs_amt',                   'mean'),
        std_abs_amt        = ('abs_amt',                   'std'),
        max_abs_amt        = ('abs_amt',                   'max'),
        median_abs_amt     = ('abs_amt',                   'median'),
        total_credit_amt   = ('credit_amt',                'sum'),
        total_debit_amt    = ('debit_amt',                 'sum'),
        credit_count       = ('is_credit',                 'sum'),
        debit_count        = ('is_debit',                  'sum'),
        near_50k_count     = ('near_50k',                  'sum'),
        round_1k_count     = ('round_1k',                  'sum'),
        round_5k_count     = ('round_5k',                  'sum'),
        round_10k_count    = ('round_10k',                 'sum'),
        night_count        = ('is_night',                  'sum'),
        upi_count          = ('is_upi',                    'sum'),
        cash_count         = ('is_cash',                   'sum'),
        month_edge_count   = ('is_month_edge',             'sum'),
        large_credit_count = ('large_credit',              'sum'),
        large_debit_count  = ('large_debit',               'sum'),
        n_unique_cp        = ('counterparty_id',           'nunique'),
        n_unique_mcc       = ('mcc_code',                  'nunique'),
        n_unique_months    = ('month',                     'nunique'),
        first_ts           = ('ts_epoch',                  'min'),
        last_ts            = ('ts_epoch',                  'max'),
        bal_min            = ('balance_after_transaction', 'min'),
        bal_max            = ('balance_after_transaction', 'max'),
        bal_mean           = ('balance_after_transaction', 'mean'),
        bal_std            = ('balance_after_transaction', 'std'),
        bal_last           = ('balance_after_transaction', 'last'),
    ).reset_index()

    pass1_aggs.append(agg.to_pandas())
    del txn, agg
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

log("  Combining pass 1...")
p1 = pd.concat(pass1_aggs, ignore_index=True)
del pass1_aggs; gc.collect()

p1_final = p1.groupby('account_id').agg(
    txn_count          = ('txn_count',          'sum'),
    total_abs_amt      = ('total_abs_amt',       'sum'),
    mean_abs_amt       = ('mean_abs_amt',        'mean'),
    std_abs_amt        = ('std_abs_amt',         'mean'),
    max_abs_amt        = ('max_abs_amt',         'max'),
    median_abs_amt     = ('median_abs_amt',      'mean'),
    total_credit_amt   = ('total_credit_amt',    'sum'),
    total_debit_amt    = ('total_debit_amt',     'sum'),
    credit_count       = ('credit_count',        'sum'),
    debit_count        = ('debit_count',         'sum'),
    near_50k_count     = ('near_50k_count',      'sum'),
    round_1k_count     = ('round_1k_count',      'sum'),
    round_5k_count     = ('round_5k_count',      'sum'),
    round_10k_count    = ('round_10k_count',     'sum'),
    night_count        = ('night_count',         'sum'),
    upi_count          = ('upi_count',           'sum'),
    cash_count         = ('cash_count',          'sum'),
    month_edge_count   = ('month_edge_count',    'sum'),
    large_credit_count = ('large_credit_count',  'sum'),
    large_debit_count  = ('large_debit_count',   'sum'),
    n_unique_cp        = ('n_unique_cp',         'sum'),
    n_unique_mcc       = ('n_unique_mcc',        'sum'),
    n_unique_months    = ('n_unique_months',     'max'),
    first_ts           = ('first_ts',            'min'),
    last_ts            = ('last_ts',             'max'),
    bal_min            = ('bal_min',             'min'),
    bal_max            = ('bal_max',             'max'),
    bal_mean           = ('bal_mean',            'mean'),
    bal_std            = ('bal_std',             'mean'),
    bal_last           = ('bal_last',            'last'),
).reset_index()

del p1; gc.collect()
log(f"  Pass 1 done. Shape: {p1_final.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — Fan-in/fan-out, active days, MCC std
# ══════════════════════════════════════════════════════════════════════════════
log("=== PASS 2: Fan-in/fan-out + active days + MCC std ===")
pass2_aggs = []

for b_idx, (rg_start, rg_end) in enumerate(batch_ranges):
    log(f"  P2 batch {b_idx+1}/{len(batch_ranges)}")

    tbl_txn = read_rg_batch(pf_txn, rg_start, rg_end)
    if tbl_txn is None:
        continue

    txn = cudf.DataFrame.from_arrow(tbl_txn)
    del tbl_txn

    ts_clean = (
        txn['transaction_timestamp']
        .str.replace(" ", "T")
        .str.slice(0,19)
    )

    txn['ts'] = cudf.to_datetime(ts_clean)
    txn = txn.dropna(subset=["ts"])
    txn['abs_amt'] = txn['amount'].abs()
    txn['date_str']= txn['ts'].dt.strftime('%Y-%m-%d')

    cred = txn[txn['txn_type'] == 'C']
    deb  = txn[txn['txn_type'] == 'D']

    cred_agg = cred.groupby('account_id').agg(
        n_credit_cp = ('counterparty_id', 'nunique'),
        max_credit  = ('abs_amt',         'max'),
    ).reset_index()

    deb_agg = deb.groupby('account_id').agg(
        n_debit_cp  = ('counterparty_id', 'nunique'),
        max_debit   = ('abs_amt',         'max'),
    ).reset_index()

    days_agg = txn.groupby('account_id').agg(
        n_active_days = ('date_str', 'nunique'),
    ).reset_index()

    mcc_grp = txn.groupby(['account_id', 'mcc_code'])['abs_amt'].std().reset_index()
    mcc_grp.columns = ['account_id', 'mcc_code', 'mcc_std']
    mcc_agg = mcc_grp.groupby('account_id').agg(
        mean_mcc_std = ('mcc_std', 'mean'),
        max_mcc_std  = ('mcc_std', 'max'),
    ).reset_index()

    batch_p2 = (
        days_agg
        .merge(cred_agg, on='account_id', how='left')
        .merge(deb_agg,  on='account_id', how='left')
        .merge(mcc_agg,  on='account_id', how='left')
    )
    pass2_aggs.append(batch_p2.to_pandas())

    del txn, cred, deb, cred_agg, deb_agg, days_agg, mcc_grp, mcc_agg, batch_p2
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

log("  Combining pass 2...")
p2 = pd.concat(pass2_aggs, ignore_index=True)
del pass2_aggs; gc.collect()

p2_final = p2.groupby('account_id').agg(
    n_active_days = ('n_active_days', 'sum'),
    n_credit_cp   = ('n_credit_cp',   'sum'),
    n_debit_cp    = ('n_debit_cp',    'sum'),
    max_credit    = ('max_credit',    'max'),
    max_debit     = ('max_debit',     'max'),
    mean_mcc_std  = ('mean_mcc_std',  'mean'),
    max_mcc_std   = ('max_mcc_std',   'max'),
).reset_index()

del p2; gc.collect()
log(f"  Pass 2 done. Shape: {p2_final.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# MERGE + DERIVED FEATURES
# ══════════════════════════════════════════════════════════════════════════════
log("=== Merging & deriving final features ===")

feats = p1_final.merge(p2_final, on='account_id', how='left')
del p1_final, p2_final; gc.collect()

feats['credit_ratio']       = feats['credit_count']        / (feats['txn_count'] + 1)
feats['near_50k_ratio']     = feats['near_50k_count']      / (feats['txn_count'] + 1)
feats['round_ratio']        = feats['round_1k_count']      / (feats['txn_count'] + 1)
feats['night_ratio']        = feats['night_count']         / (feats['txn_count'] + 1)
feats['upi_ratio']          = feats['upi_count']           / (feats['txn_count'] + 1)
feats['cash_ratio']         = feats['cash_count']          / (feats['txn_count'] + 1)
feats['month_edge_ratio']   = feats['month_edge_count']    / (feats['txn_count'] + 1)
feats['large_credit_ratio'] = feats['large_credit_count']  / (feats['credit_count'] + 1)
feats['large_debit_ratio']  = feats['large_debit_count']   / (feats['debit_count'] + 1)
feats['fan_in_ratio']       = feats['n_credit_cp']         / (feats['n_debit_cp'] + 1)
feats['cp_credit_ratio']    = feats['n_credit_cp']         / (feats['n_unique_cp'] + 1)
feats['activity_span_days'] = (feats['last_ts'] - feats['first_ts']) / 86400
feats['txn_per_active_day'] = feats['txn_count']           / (feats['n_active_days'] + 1)
feats['amt_per_active_day'] = feats['total_abs_amt']       / (feats['n_active_days'] + 1)
feats['active_day_ratio']   = feats['n_active_days']       / (feats['activity_span_days'] + 1)
feats['passthrough_proxy']  = feats['total_credit_amt']    / (feats['bal_max'].clip(lower=1) + 1)
feats['credit_debit_diff']  = feats['total_credit_amt']    - feats['total_debit_amt']
feats['bal_range']          = feats['bal_max']             - feats['bal_min']
feats['bal_volatility']     = feats['bal_std']             / (feats['total_abs_amt'].clip(lower=1) + 1)
feats['bal_drawdown']       = (feats['bal_max'] - feats['bal_last']) / (feats['bal_max'].abs() + 1)
feats['max_txn_conc']       = feats['max_abs_amt']         / (feats['total_abs_amt'] + 1)
feats['max_credit_conc']    = feats['max_credit']          / (feats['total_credit_amt'] + 1)

log(f"  Final shape: {feats.shape}  |  Columns: {len(feats.columns)}")

out = FEATS / "txn_features.parquet"
feats.to_parquet(out, index=False)
log(f"  Saved → {out}  ({out.stat().st_size/1e6:.1f} MB)")

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
log("=== Validation ===")
labels = pd.read_parquet(DATA / "train_labels.parquet")
check  = feats.merge(labels[['account_id','is_mule']], on='account_id', how='inner')

print(f"\n  Feature table accounts : {len(feats):,}")
print(f"  Labeled matches        : {len(check):,}")
print(f"  Mule accounts matched  : {check['is_mule'].sum():,}")

key_feats = [
    'txn_count','total_abs_amt','near_50k_ratio','round_ratio',
    'activity_span_days','txn_per_active_day','passthrough_proxy',
    'n_unique_cp','fan_in_ratio','large_credit_ratio',
    'bal_volatility','n_unique_mcc','mean_mcc_std',
    'night_ratio','cash_ratio','amt_per_active_day',
]
key_feats = [f for f in key_feats if f in check.columns]

print(f"\n  {'Feature':<30s} {'Mule':>10} {'Legit':>10} {'p-val':>10}")
print("  " + "-"*65)
for col in key_feats:
    m = check[check.is_mule==1][col].dropna()
    l = check[check.is_mule==0][col].dropna()
    if len(m) < 5: continue
    try:    _, pval = mannwhitneyu(m, l, alternative='two-sided')
    except: pval = 1.0
    sig = "***" if pval<0.001 else "**" if pval<0.01 else "*" if pval<0.05 else ""
    print(f"  {col:<30s} {m.mean():>10.3f} {l.mean():>10.3f} {pval:>10.2e} {sig}")

log("Done. Next: python scripts/04_model_baseline.py")