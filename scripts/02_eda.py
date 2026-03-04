"""
02_eda.py — Exhaustive EDA for AML Mule Detection
Covers:
  A. Static file EDA (accounts, customers, demographics, branch, product)
  B. Label analysis (mule rate, alert reasons, red herring signals)
  C. Transaction EDA via row-group sampling (no OOM)
  D. Cross-feature mule vs legit comparisons
  E. Saves master_static.parquet for feature engineering

Run: python 02_eda.py
Outputs: cache/features/master_static.parquet
         notebooks/eda_plots/  (all plots)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import warnings
warnings.filterwarnings('ignore')

import cudf
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats
import gc

# ── PATHS ─────────────────────────────────────────────────────────────────────
DATA   = Path("/home/niranjan/AML/data")
CACHE  = Path("/home/niranjan/AML/cache")
PLOTS  = Path("/home/niranjan/AML/notebooks/eda_plots")
FEATS  = CACHE / "features"

PLOTS.mkdir(parents=True, exist_ok=True)
FEATS.mkdir(parents=True, exist_ok=True)

REF_DATE = pd.Timestamp("2025-06-30")  # end of data window

def section(title):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")

def savefig(name):
    plt.tight_layout()
    plt.savefig(PLOTS / f"{name}.png", dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → saved: eda_plots/{name}.png")

# ══════════════════════════════════════════════════════════════════════════════
section("A. LOAD ALL STATIC FILES")
# ══════════════════════════════════════════════════════════════════════════════

customers  = pd.read_parquet(DATA / "customers.parquet")
accounts   = pd.read_parquet(DATA / "accounts.parquet")
labels     = pd.read_parquet(DATA / "train_labels.parquet")
test       = pd.read_parquet(DATA / "test_accounts.parquet")
linkage    = pd.read_parquet(DATA / "customer_account_linkage.parquet")
demog      = pd.read_parquet(DATA / "demographics.parquet")
acct_add   = pd.read_parquet(DATA / "accounts-additional.parquet")
product    = pd.read_parquet(DATA / "product_details.parquet")
branch     = pd.read_parquet(DATA / "branch.parquet")

for name, df in [("customers", customers), ("accounts", accounts),
                 ("labels", labels), ("test", test), ("linkage", linkage),
                 ("demographics", demog), ("acct_add", acct_add),
                 ("product", product), ("branch", branch)]:
    print(f"  {name:20s} {df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
section("B. LABEL ANALYSIS")
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n  Total training accounts : {len(labels):,}")
print(f"  Total test accounts     : {len(test):,}")
print(f"  Mule accounts           : {labels['is_mule'].sum():,}")
print(f"  Legitimate accounts     : {(labels['is_mule']==0).sum():,}")
print(f"  Mule rate               : {labels['is_mule'].mean():.4%}")

print(f"\n  Alert reasons (mules only):")
ar = labels[labels.is_mule==1]['alert_reason'].value_counts(dropna=False)
print(ar.to_string())

print(f"\n  Top 20 flagging branches:")
fb = labels[labels.is_mule==1]['flagged_by_branch'].value_counts().head(20)
print(fb.to_string())

# Mule flag date distribution
labels['mule_flag_date'] = pd.to_datetime(labels['mule_flag_date'], errors='coerce')
flag_by_month = labels[labels.is_mule==1].groupby(
    labels['mule_flag_date'].dt.to_period('M')
).size()

fig, ax = plt.subplots(figsize=(14, 4))
flag_by_month.plot(kind='bar', ax=ax, color='crimson', alpha=0.8)
ax.set_title('Mule Flag Dates — Monthly Distribution')
ax.set_xlabel('Month')
ax.set_ylabel('Accounts Flagged')
plt.xticks(rotation=45, ha='right')
savefig('B1_mule_flag_dates')

# ══════════════════════════════════════════════════════════════════════════════
section("C. BUILD MASTER JOINED TABLE")
# ══════════════════════════════════════════════════════════════════════════════

# Parse dates in accounts
date_cols_acct = ['account_opening_date', 'last_mobile_update_date',
                  'last_kyc_date', 'freeze_date', 'unfreeze_date']
for c in date_cols_acct:
    accounts[c] = pd.to_datetime(accounts[c], errors='coerce')

date_cols_cust = ['date_of_birth', 'relationship_start_date']
for c in date_cols_cust:
    customers[c] = pd.to_datetime(customers[c], errors='coerce')

demog['address_last_update_date']   = pd.to_datetime(demog['address_last_update_date'], errors='coerce')
demog['passbook_last_update_date']  = pd.to_datetime(demog['passbook_last_update_date'], errors='coerce')

# Join everything
master = (
    labels
    .merge(linkage,   on='account_id',  how='left')
    .merge(accounts,  on='account_id',  how='left')
    .merge(customers, on='customer_id', how='left')
    .merge(demog,     on='customer_id', how='left')
    .merge(acct_add,  on='account_id',  how='left')
    .merge(product,   on='customer_id', how='left')
    .merge(branch,    on='branch_code', how='left')
)

print(f"\n  Master table shape: {master.shape}")
print(f"  Null counts (top 20):")
nulls = (master.isnull().sum() / len(master) * 100).sort_values(ascending=False)
print(nulls[nulls > 0].head(20).to_string())

# ── Derived date features ──────────────────────────────────────────────────
master['account_age_days']          = (REF_DATE - master['account_opening_date']).dt.days
master['customer_age_years']        = (REF_DATE - master['date_of_birth']).dt.days / 365.25
master['relationship_age_days']     = (REF_DATE - master['relationship_start_date']).dt.days
master['days_since_mobile_update']  = (REF_DATE - master['last_mobile_update_date']).dt.days
master['days_since_kyc']            = (REF_DATE - master['last_kyc_date']).dt.days
master['days_since_address_update'] = (REF_DATE - master['address_last_update_date']).dt.days
master['was_frozen']                = master['freeze_date'].notna().astype(int)
master['pin_mismatch']              = (master['customer_pin'] != master['permanent_pin']).astype(int)

# ══════════════════════════════════════════════════════════════════════════════
section("D. MULE vs LEGIT — NUMERIC FEATURES")
# ══════════════════════════════════════════════════════════════════════════════

numeric_feats = [
    'account_age_days', 'customer_age_years', 'relationship_age_days',
    'days_since_mobile_update', 'days_since_kyc', 'days_since_address_update',
    'avg_balance', 'monthly_avg_balance', 'quarterly_avg_balance', 'daily_avg_balance',
    'loan_sum', 'loan_count', 'cc_sum', 'cc_count',
    'od_sum', 'od_count', 'ka_sum', 'ka_count', 'sa_sum', 'sa_count',
    'num_chequebooks', 'branch_employee_count', 'branch_turnover', 'branch_asset_size',
]
numeric_feats = [f for f in numeric_feats if f in master.columns]

print(f"\n  {'Feature':<35s} {'Mule mean':>12} {'Legit mean':>12} {'Ratio':>8} {'p-value':>10}")
print("  " + "-"*80)

stat_results = []
for col in numeric_feats:
    mule_vals  = master[master.is_mule==1][col].dropna()
    legit_vals = master[master.is_mule==0][col].dropna()
    if len(mule_vals) < 10 or len(legit_vals) < 10:
        continue
    m_mean = mule_vals.mean()
    l_mean = legit_vals.mean()
    ratio  = m_mean / (l_mean + 1e-9)
    try:
        _, pval = stats.mannwhitneyu(mule_vals, legit_vals, alternative='two-sided')
    except:
        pval = 1.0
    stat_results.append({'feature': col, 'mule_mean': m_mean, 'legit_mean': l_mean,
                          'ratio': ratio, 'pval': pval})
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"  {col:<35s} {m_mean:>12.1f} {l_mean:>12.1f} {ratio:>8.2f} {pval:>10.2e} {sig}")

stat_df = pd.DataFrame(stat_results).sort_values('pval')

# Plot top discriminative numeric features
top_feats = stat_df[stat_df.pval < 0.05]['feature'].head(12).tolist()
if top_feats:
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    for i, feat in enumerate(top_feats):
        ax = axes[i]
        mule_v  = master[master.is_mule==1][feat].dropna().clip(
            master[feat].quantile(0.01), master[feat].quantile(0.99))
        legit_v = master[master.is_mule==0][feat].dropna().clip(
            master[feat].quantile(0.01), master[feat].quantile(0.99))
        ax.hist(legit_v, bins=40, alpha=0.6, color='steelblue', label='Legit', density=True)
        ax.hist(mule_v,  bins=40, alpha=0.6, color='crimson',   label='Mule',  density=True)
        ax.set_title(feat, fontsize=9)
        ax.legend(fontsize=7)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Mule vs Legit — Numeric Feature Distributions', fontsize=13)
    savefig('D1_numeric_distributions')

# ══════════════════════════════════════════════════════════════════════════════
section("E. MULE vs LEGIT — CATEGORICAL FEATURES")
# ══════════════════════════════════════════════════════════════════════════════

cat_feats = [
    'account_status', 'product_family', 'scheme_code', 'rural_branch',
    'kyc_compliant', 'nomination_flag', 'cheque_allowed', 'cheque_availed',
    'pan_available', 'aadhaar_available', 'passport_available',
    'mobile_banking_flag', 'internet_banking_flag', 'atm_card_flag',
    'demat_flag', 'credit_card_flag', 'fastag_flag',
    'gender', 'joint_account_flag', 'nri_flag', 'branch_type',
    'was_frozen', 'pin_mismatch',
]
cat_feats = [f for f in cat_feats if f in master.columns]

print(f"\n  {'Feature':<30s} {'Values & mule rates'}")
print("  " + "-"*70)

cat_results = []
for col in cat_feats:
    ct = pd.crosstab(master[col], master['is_mule'])
    ct.columns = ['legit', 'mule']
    ct['mule_rate'] = ct['mule'] / (ct['mule'] + ct['legit'])
    ct['total']     = ct['mule'] + ct['legit']
    max_diff = ct['mule_rate'].max() - ct['mule_rate'].min()
    cat_results.append({'feature': col, 'max_rate_diff': max_diff,
                         'values': ct['mule_rate'].to_dict()})
    print(f"  {col:<30s} max_diff={max_diff:.3f}  {ct['mule_rate'].to_dict()}")

cat_df = pd.DataFrame(cat_results).sort_values('max_rate_diff', ascending=False)

# Plot top categorical features
top_cats = cat_df.head(9)['feature'].tolist()
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for i, feat in enumerate(top_cats):
    ax = axes[i]
    ct = pd.crosstab(master[feat], master['is_mule'], normalize='index')
    if 1 in ct.columns:
        ct[1].plot(kind='bar', ax=ax, color='crimson', alpha=0.8)
    ax.set_title(f'{feat}\n(mule rate by value)', fontsize=9)
    ax.set_ylabel('Mule rate')
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelrotation=30)
    ax.axhline(master['is_mule'].mean(), color='navy', linestyle='--',
               linewidth=1, label='overall rate')
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Mule Rate by Categorical Feature Value', fontsize=13)
savefig('E1_categorical_mule_rates')

# ══════════════════════════════════════════════════════════════════════════════
section("F. BRANCH-LEVEL COLLUSION ANALYSIS")
# ══════════════════════════════════════════════════════════════════════════════

branch_stats = (
    master.groupby('branch_code')
    .agg(
        n_accounts   = ('account_id', 'count'),
        n_mules      = ('is_mule', 'sum'),
        mule_rate    = ('is_mule', 'mean'),
        avg_balance  = ('avg_balance', 'mean'),
    )
    .reset_index()
    .query('n_accounts >= 10')
    .sort_values('mule_rate', ascending=False)
)

print(f"\n  Branches with >=10 accounts: {len(branch_stats)}")
print(f"  Branches with mule_rate > 50%: {(branch_stats.mule_rate > 0.5).sum()}")
print(f"  Branches with mule_rate > 80%: {(branch_stats.mule_rate > 0.8).sum()}")
print(f"\n  Top 15 most suspicious branches:")
print(branch_stats.head(15).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(branch_stats['mule_rate'], bins=50, color='crimson', alpha=0.8, edgecolor='white')
axes[0].axvline(master['is_mule'].mean(), color='navy', linestyle='--', label='overall rate')
axes[0].set_title('Distribution of Branch Mule Rates')
axes[0].set_xlabel('Mule Rate')
axes[0].legend()

axes[1].scatter(branch_stats['n_accounts'], branch_stats['mule_rate'],
                alpha=0.4, s=20, color='steelblue')
axes[1].set_title('Branch Size vs Mule Rate')
axes[1].set_xlabel('Number of Accounts')
axes[1].set_ylabel('Mule Rate')
savefig('F1_branch_analysis')

# Flag high-collusion branches for later use
high_collusion_branches = branch_stats[branch_stats.mule_rate > 0.5]['branch_code'].tolist()
print(f"\n  High-collusion branches (>50% mule rate): {len(high_collusion_branches)}")
master['high_collusion_branch'] = master['branch_code'].isin(high_collusion_branches).astype(int)

# ══════════════════════════════════════════════════════════════════════════════
section("G. SCHEME CODE ANALYSIS (PMJDY etc.)")
# ══════════════════════════════════════════════════════════════════════════════

# PMJDY = Jan Dhan accounts, commonly exploited for mule activity in India
scheme_mule = pd.crosstab(master['scheme_code'], master['is_mule'], normalize='index')
print(f"\n  Mule rate by scheme code:")
print(scheme_mule.to_string())

fig, ax = plt.subplots(figsize=(10, 4))
if 1 in scheme_mule.columns:
    scheme_mule[1].sort_values(ascending=False).plot(kind='bar', ax=ax,
                                                      color='crimson', alpha=0.8)
ax.axhline(master['is_mule'].mean(), color='navy', linestyle='--', label='overall rate')
ax.set_title('Mule Rate by Scheme Code')
ax.set_ylabel('Mule Rate')
ax.legend()
savefig('G1_scheme_code_mule_rates')

# ══════════════════════════════════════════════════════════════════════════════
section("H. TEMPORAL SIGNALS FROM STATIC DATA")
# ══════════════════════════════════════════════════════════════════════════════

# Post-mobile-change spike pattern: mobile updated recently → higher mule rate?
master['mobile_update_bucket'] = pd.cut(
    master['days_since_mobile_update'],
    bins=[0, 30, 90, 180, 365, 730, 99999],
    labels=['<30d', '30-90d', '90-180d', '180-365d', '1-2yr', '>2yr']
)
mob_mule = master.groupby('mobile_update_bucket', observed=True)['is_mule'].agg(['mean','count'])
print(f"\n  Mule rate by days since mobile update:")
print(mob_mule.to_string())

# New account high value: account age vs mule rate
master['account_age_bucket'] = pd.cut(
    master['account_age_days'],
    bins=[0, 90, 180, 365, 730, 1825, 99999],
    labels=['<3m', '3-6m', '6-12m', '1-2yr', '2-5yr', '>5yr']
)
age_mule = master.groupby('account_age_bucket', observed=True)['is_mule'].agg(['mean','count'])
print(f"\n  Mule rate by account age:")
print(age_mule.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
mob_mule['mean'].plot(kind='bar', ax=axes[0], color='coral', alpha=0.9)
axes[0].axhline(master['is_mule'].mean(), color='navy', linestyle='--')
axes[0].set_title('Mule Rate vs Days Since Mobile Update\n(Post-mobile-change spike pattern)')
axes[0].set_ylabel('Mule Rate')

age_mule['mean'].plot(kind='bar', ax=axes[1], color='steelblue', alpha=0.9)
axes[1].axhline(master['is_mule'].mean(), color='navy', linestyle='--')
axes[1].set_title('Mule Rate vs Account Age\n(New account high value pattern)')
axes[1].set_ylabel('Mule Rate')
savefig('H1_temporal_static_signals')

# KYC recency signal
master['kyc_age_bucket'] = pd.cut(
    master['days_since_kyc'],
    bins=[0, 180, 365, 730, 1825, 99999],
    labels=['<6m', '6-12m', '1-2yr', '2-5yr', '>5yr']
)
kyc_mule = master.groupby('kyc_age_bucket', observed=True)['is_mule'].agg(['mean','count'])
print(f"\n  Mule rate by KYC recency:")
print(kyc_mule.to_string())

# ══════════════════════════════════════════════════════════════════════════════
section("I. TRANSACTION EDA — ROW GROUP SAMPLING")
# ══════════════════════════════════════════════════════════════════════════════

print("\n  Sampling 15 row groups from transactions_full (~800K rows)...")

txn_path = CACHE / "transactions_full.parquet"
pf = pq.ParquetFile(txn_path)

sampled_batches = []
for i, batch in enumerate(pf.iter_batches()):
    sampled_batches.append(batch)
    if i >= 14:  # 15 row groups
        break

txn_sample = pd.DataFrame(
    pa.Table.from_batches(sampled_batches).to_pandas()
)
txn_sample['transaction_timestamp'] = pd.to_datetime(txn_sample['transaction_timestamp'])
print(f"  Sample shape: {txn_sample.shape}")

# Tag with mule label
mule_accts = set(labels[labels.is_mule==1]['account_id'])
legit_accts = set(labels[labels.is_mule==0]['account_id'])
txn_sample['is_mule'] = txn_sample['account_id'].map(
    lambda x: 1 if x in mule_accts else (0 if x in legit_accts else -1)
)
txn_labeled = txn_sample[txn_sample.is_mule >= 0]
print(f"  Labeled transactions in sample: {len(txn_labeled):,}")

# Amount distributions
print(f"\n  Amount statistics (labeled transactions):")
print(txn_labeled.groupby('is_mule')['amount'].describe().to_string())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Amount distribution (log scale)
for label, color in [(0,'steelblue'), (1,'crimson')]:
    vals = txn_labeled[txn_labeled.is_mule==label]['amount'].clip(0, 200000)
    axes[0,0].hist(vals, bins=60, alpha=0.5, color=color,
                   label='Legit' if label==0 else 'Mule', density=True)
axes[0,0].set_title('Transaction Amount Distribution')
axes[0,0].set_xlabel('Amount (INR, clipped at 200k)')
axes[0,0].legend()

# Structuring check: amounts near 50k threshold
for label, color in [(0,'steelblue'), (1,'crimson')]:
    vals = txn_labeled[txn_labeled.is_mule==label]['amount']
    near_thresh = vals[(vals >= 40000) & (vals <= 55000)]
    axes[0,1].hist(near_thresh, bins=50, alpha=0.5, color=color,
                   label='Legit' if label==0 else 'Mule', density=True)
axes[0,1].axvline(50000, color='black', linestyle='--', label='50k threshold')
axes[0,1].set_title('Structuring Check: Amounts 40k–55k')
axes[0,1].legend()

# Channel mix
channel_ct = pd.crosstab(txn_labeled['channel'], txn_labeled['is_mule'], normalize='columns')
channel_ct.plot(kind='bar', ax=axes[1,0], color=['steelblue','crimson'], alpha=0.8)
axes[1,0].set_title('Channel Mix: Mule vs Legit (normalized)')
axes[1,0].set_xlabel('Channel')
axes[1,0].tick_params(axis='x', labelrotation=45)

# Hour of day pattern
txn_labeled['hour'] = txn_labeled['transaction_timestamp'].dt.hour
hour_mule = txn_labeled.groupby(['hour','is_mule']).size().unstack(fill_value=0)
hour_mule_norm = hour_mule.div(hour_mule.sum(axis=0), axis=1)
hour_mule_norm.plot(ax=axes[1,1], color=['steelblue','crimson'], alpha=0.8)
axes[1,1].set_title('Transaction Hour Distribution: Mule vs Legit')
axes[1,1].set_xlabel('Hour of Day')
axes[1,1].legend(['Legit', 'Mule'])

fig.suptitle('Transaction-Level EDA', fontsize=13)
savefig('I1_transaction_eda')

# Round amount analysis
print(f"\n  Round amount analysis:")
round_thresholds = [1000, 2000, 5000, 10000, 25000, 50000, 100000]
for thresh in round_thresholds:
    exact = (txn_labeled['amount'] == thresh)
    mule_rate_exact = txn_labeled[exact]['is_mule'].mean()
    pct_of_all = exact.mean() * 100
    print(f"  Exactly {thresh:>8,}: {pct_of_all:.3f}% of txns, mule_rate={mule_rate_exact:.3f}")

# Txn type (C/D) by mule
print(f"\n  C/D ratio by mule label:")
print(pd.crosstab(txn_labeled['txn_type'], txn_labeled['is_mule'], normalize='columns').to_string())

# ══════════════════════════════════════════════════════════════════════════════
section("J. RED HERRING INVESTIGATION")
# ══════════════════════════════════════════════════════════════════════════════

# Look for mule accounts that have NO obvious static signals
# These are either red herrings OR accounts where txn features will carry all signal

# Score each mule account on static features
mules = master[master.is_mule == 1].copy()

# How many mules are in high-collusion branches?
print(f"\n  Mules in high-collusion branches : {mules['high_collusion_branch'].sum()} / {len(mules)}")

# How many mules have recent mobile updates?
print(f"  Mules with mobile update <30 days: {(mules['days_since_mobile_update'] < 30).sum()}")
print(f"  Mules with mobile update <90 days: {(mules['days_since_mobile_update'] < 90).sum()}")

# How many mules have very young accounts?
print(f"  Mules with account age <180 days  : {(mules['account_age_days'] < 180).sum()}")

# How many mules have non-KYC compliant?
if 'kyc_compliant' in mules.columns:
    print(f"  Mules with kyc_compliant != Y     : {(mules['kyc_compliant'] != 'Y').sum()}")

# Mules with NO static signal at all
no_signal_mules = mules[
    (mules['high_collusion_branch'] == 0) &
    (mules['days_since_mobile_update'] >= 90) &
    (mules['account_age_days'] >= 180)
]
print(f"\n  Mules with NO obvious static signal: {len(no_signal_mules)} / {len(mules)}")
print(f"  → These are 'hard' mules — signal must come from transaction behavior")
print(f"  → OR they may include red herring labels")

# Alert reason vs account characteristics — looking for inconsistencies
print(f"\n  Alert reason breakdown:")
for reason, grp in master[master.is_mule==1].groupby('alert_reason'):
    print(f"\n  [{reason}] n={len(grp)}")
    print(f"    avg account_age_days     : {grp['account_age_days'].mean():.0f}")
    print(f"    avg days_since_mob_update: {grp['days_since_mobile_update'].mean():.0f}")
    print(f"    avg_balance mean         : {grp['avg_balance'].mean():.0f}")
    print(f"    high_collusion_branch %  : {grp['high_collusion_branch'].mean():.2%}")

# ══════════════════════════════════════════════════════════════════════════════
section("K. SAVE MASTER STATIC FEATURES")
# ══════════════════════════════════════════════════════════════════════════════

# Columns to keep for feature engineering
keep_cols = [
    'account_id', 'customer_id', 'is_mule',
    # account
    'account_age_days', 'days_since_mobile_update', 'days_since_kyc',
    'days_since_address_update', 'avg_balance', 'monthly_avg_balance',
    'quarterly_avg_balance', 'daily_avg_balance',
    'account_status', 'product_family', 'scheme_code', 'rural_branch',
    'kyc_compliant', 'nomination_flag', 'cheque_allowed', 'cheque_availed',
    'num_chequebooks', 'was_frozen', 'pin_mismatch',
    'branch_code', 'branch_type', 'high_collusion_branch',
    'branch_employee_count', 'branch_turnover', 'branch_asset_size',
    # customer
    'customer_age_years', 'relationship_age_days',
    'pan_available', 'aadhaar_available', 'passport_available',
    'mobile_banking_flag', 'internet_banking_flag', 'atm_card_flag',
    'demat_flag', 'credit_card_flag', 'fastag_flag',
    # demographics
    'gender', 'joint_account_flag', 'nri_flag',
    # product
    'loan_sum', 'loan_count', 'cc_sum', 'cc_count',
    'od_sum', 'od_count', 'ka_sum', 'ka_count', 'sa_sum', 'sa_count',
]
keep_cols = [c for c in keep_cols if c in master.columns]

master_out = master[keep_cols].copy()
master_out.to_parquet(FEATS / "master_static.parquet", index=False)
print(f"\n  Saved: cache/features/master_static.parquet")
print(f"  Shape: {master_out.shape}")
print(f"  Size:  {(FEATS / 'master_static.parquet').stat().st_size / 1e6:.1f} MB")

# Also save test set version (no is_mule column)
test_master = (
    test
    .merge(linkage,   on='account_id',  how='left')
    .merge(accounts,  on='account_id',  how='left')
    .merge(customers, on='customer_id', how='left')
    .merge(demog,     on='customer_id', how='left')
    .merge(acct_add,  on='account_id',  how='left')
    .merge(product,   on='customer_id', how='left')
    .merge(branch,    on='branch_code', how='left')
)
for c in date_cols_acct:
    test_master[c] = pd.to_datetime(test_master[c], errors='coerce')
for c in date_cols_cust:
    test_master[c] = pd.to_datetime(test_master[c], errors='coerce')
test_master['address_last_update_date'] = pd.to_datetime(test_master['address_last_update_date'], errors='coerce')

test_master['account_age_days']          = (REF_DATE - test_master['account_opening_date']).dt.days
test_master['customer_age_years']        = (REF_DATE - test_master['date_of_birth']).dt.days / 365.25
test_master['relationship_age_days']     = (REF_DATE - test_master['relationship_start_date']).dt.days
test_master['days_since_mobile_update']  = (REF_DATE - test_master['last_mobile_update_date']).dt.days
test_master['days_since_kyc']            = (REF_DATE - test_master['last_kyc_date']).dt.days
test_master['days_since_address_update'] = (REF_DATE - test_master['address_last_update_date']).dt.days
test_master['was_frozen']                = test_master['freeze_date'].notna().astype(int)
test_master['pin_mismatch']              = (test_master['customer_pin'] != test_master['permanent_pin']).astype(int)
test_master['high_collusion_branch']     = test_master['branch_code'].isin(high_collusion_branches).astype(int)

test_keep = [c for c in keep_cols if c != 'is_mule' and c in test_master.columns]
test_master[test_keep].to_parquet(FEATS / "test_static.parquet", index=False)
print(f"  Saved: cache/features/test_static.parquet")
print(f"  Shape: {test_master[test_keep].shape}")

# ══════════════════════════════════════════════════════════════════════════════
section("SUMMARY")
# ══════════════════════════════════════════════════════════════════════════════

print(f"""
  KEY FINDINGS FROM EDA:
  ─────────────────────────────────────────────────────────────
  Label stats  : {labels['is_mule'].sum():,} mules / {len(labels):,} total
                 ({labels['is_mule'].mean():.2%} mule rate)

  Top signals from static data (check plots for confirmation):
  1. branch_code        — high-collusion branches are strong signal
  2. scheme_code        — PMJDY / Jan Dhan accounts show elevated rates
  3. days_since_mobile  — recent mobile updates → higher mule rate
  4. account_age_days   — new accounts more suspicious
  5. was_frozen         — frozen accounts likely flagged
  6. kyc_compliant      — KYC non-compliance elevated in mules
  7. pin_mismatch       — address mismatch = possible identity issue

  Red herring estimate : {len(no_signal_mules):,} mules with NO static signal
                         ({len(no_signal_mules)/len(mules):.1%} of mules — need txn features)

  Files saved:
    cache/features/master_static.parquet  (train)
    cache/features/test_static.parquet    (test)
    notebooks/eda_plots/*.png             ({len(list(PLOTS.glob('*.png')))} plots)

  Next step: python scripts/03_txn_features.py
""")