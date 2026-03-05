"""
07_geo_features.py — Geographic Feature Engineering (pure pandas, no cuDF)

Reads lat/lon from transactions_additional_full.parquet.
~26% of 397M rows have geo data (~103M rows with valid coordinates).

Features per account:
  - n_geo_txns          : number of transactions with geo data
  - n_unique_locations  : distinct (lat_r, lon_r) pairs (rounded to 2dp ~1km grid)
  - geo_spread_km       : max great-circle distance between any two transaction locations
  - location_entropy    : Shannon entropy of location distribution (high = scattered)
  - home_lat/home_lon   : modal (most frequent) transaction location
  - max_distance_from_home_km : max distance from modal location
  - geo_txn_ratio       : fraction of transactions that have geo data

Run: python scripts/07_geo_features.py
Output: cache/features/geo_features.parquet
"""

import os
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import cdist
import gc
import warnings
warnings.filterwarnings('ignore')

CACHE    = Path("/home/niranjan/AML/cache")
DATA     = Path("/home/niranjan/AML/data")
FEATS    = CACHE / "features"
FEATS.mkdir(exist_ok=True)

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

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in km."""
    R    = 6371.0
    phi1 = np.radians(lat1);  phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a    = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a.clip(0, 1)))

def shannon_entropy(counts):
    """Shannon entropy of a count array."""
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))

# ─────────────────────────────────────────────────────────────────────────────
add_path     = CACHE / "transactions_additional_full.parquet"
pf           = pq.ParquetFile(add_path)
total_rg     = pq.read_metadata(add_path).num_row_groups
batch_ranges = [(i, min(i + BATCH_RG, total_rg)) for i in range(0, total_rg, BATCH_RG)]
log(f"transactions_additional row groups: {total_rg}  |  Batches: {len(batch_ranges)}")

# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — Collect all non-null geo rows per account
#
# We round lat/lon to 2 decimal places (~1.1 km grid) before aggregating.
# This keeps unique location counts meaningful without floating point noise.
# We only read transaction_id, latitude, longitude from the additional file,
# then join account_id from the main transactions file per batch.
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== PASS 1: Collecting geo data ===")

txn_path   = CACHE / "transactions_full.parquet"
pf_txn     = pq.ParquetFile(txn_path)

pass1_aggs = []

for b_idx, (rg_start, rg_end) in enumerate(batch_ranges):
    log(f"  Batch {b_idx+1}/{len(batch_ranges)}  rg {rg_start}:{rg_end}")

    tbl_add = read_rg_batch(pf,     rg_start, rg_end,
                            columns=['transaction_id', 'latitude', 'longitude'])
    tbl_txn = read_rg_batch(pf_txn, rg_start, rg_end,
                            columns=['transaction_id', 'account_id'])
    if tbl_add is None or tbl_txn is None:
        continue

    add = tbl_add.to_pandas()
    txn = tbl_txn.to_pandas()
    del tbl_add, tbl_txn

    # Keep only rows with valid geo data
    add = add.dropna(subset=['latitude', 'longitude'])
    if len(add) == 0:
        del add, txn
        gc.collect()
        continue

    # Join account_id
    df = add.merge(txn, on='transaction_id', how='inner')
    del add, txn
    gc.collect()

    # Round to 2dp grid (~1.1km)
    df['lat_r'] = df['latitude'].round(2)
    df['lon_r'] = df['longitude'].round(2)

    # Per account: count geo txns and collect location counts
    # We aggregate (lat_r, lon_r, count) per account — not raw rows
    loc_counts = (
        df.groupby(['account_id', 'lat_r', 'lon_r'])
        .size()
        .reset_index(name='loc_count')
    )
    pass1_aggs.append(loc_counts)

    del df, loc_counts
    gc.collect()

log("  Combining pass 1...")
loc_all = pd.concat(pass1_aggs, ignore_index=True)
del pass1_aggs
gc.collect()

# Re-aggregate across batches (same account+location may appear in multiple batches)
loc_final = (
    loc_all
    .groupby(['account_id', 'lat_r', 'lon_r'])['loc_count']
    .sum()
    .reset_index()
)
del loc_all
gc.collect()

log(f"  Total (account, location) pairs: {len(loc_final):,}")
log(f"  Accounts with geo data: {loc_final['account_id'].nunique():,}")

# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — Compute per-account geo features from location counts
#
# For geo_spread_km and max_distance_from_home we need pairwise distances.
# To avoid O(n²) explosion, we cap at 50 unique locations per account
# (accounts with >50 locations get distances computed on a sample).
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== PASS 2: Computing per-account geo features ===")

MAX_LOCS  = 50   # cap for pairwise distance computation
geo_rows  = []

for acct, grp in loc_final.groupby('account_id'):
    grp    = grp.reset_index(drop=True)
    n_locs = len(grp)
    total  = int(grp['loc_count'].sum())

    # Basic counts
    n_unique = n_locs

    # Shannon entropy of location distribution
    entropy = shannon_entropy(grp['loc_count'].values)

    # Modal (home) location — most visited
    home_idx = grp['loc_count'].idxmax()
    home_lat = float(grp.loc[home_idx, 'lat_r'])
    home_lon = float(grp.loc[home_idx, 'lon_r'])

    # Distance from home for each location
    dist_from_home = haversine_km(
        grp['lat_r'].values, grp['lon_r'].values,
        home_lat,            home_lon
    )
    max_from_home = float(dist_from_home.max())
    mean_from_home= float(dist_from_home.mean())

    # Geo spread: max pairwise distance (capped at MAX_LOCS locations)
    if n_locs <= 1:
        geo_spread = 0.0
    else:
        sample = grp if n_locs <= MAX_LOCS else grp.sample(MAX_LOCS, random_state=42)
        coords  = np.radians(sample[['lat_r', 'lon_r']].values)
        # Use haversine pairwise
        lats = sample['lat_r'].values
        lons = sample['lon_r'].values
        # Vectorised pairwise — only upper triangle
        max_d = 0.0
        for i in range(len(lats)):
            if i + 1 >= len(lats):
                break
            d = haversine_km(lats[i], lons[i], lats[i+1:], lons[i+1:])
            max_d = max(max_d, float(d.max()))
        geo_spread = max_d

    geo_rows.append({
        'account_id':            acct,
        'n_geo_txns':            total,
        'n_unique_locations':    n_unique,
        'location_entropy':      entropy,
        'home_lat':              home_lat,
        'home_lon':              home_lon,
        'max_dist_from_home_km': max_from_home,
        'mean_dist_from_home_km':mean_from_home,
        'geo_spread_km':         geo_spread,
    })

geo_df = pd.DataFrame(geo_rows)
log(f"  Geo features computed: {len(geo_df):,} accounts")
del geo_rows
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# PASS 3 — Get total txn count per account to compute geo_txn_ratio
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== PASS 3: geo_txn_ratio ===")

# We already have total_txn_count in txn_features.parquet — just load it
txn_feats = pd.read_parquet(
    FEATS / "txn_features.parquet",
    columns=['account_id', 'txn_count']
)
geo_df = geo_df.merge(txn_feats, on='account_id', how='left')
geo_df['geo_txn_ratio'] = geo_df['n_geo_txns'] / (geo_df['txn_count'] + 1)
geo_df = geo_df.drop(columns=['txn_count'])

# ── Fill zeros for accounts with no geo data ──────────────────────────────
all_accounts = txn_feats['account_id']
geo_full = all_accounts.to_frame().merge(geo_df, on='account_id', how='left')

fill_zero = [
    'n_geo_txns', 'n_unique_locations', 'location_entropy',
    'max_dist_from_home_km', 'mean_dist_from_home_km',
    'geo_spread_km', 'geo_txn_ratio',
]
for col in fill_zero:
    geo_full[col] = geo_full[col].fillna(0.0)

# home_lat/lon stay NaN for accounts with no geo — that's correct
log(f"  Final geo features shape: {geo_full.shape}")

# ── Null summary ──────────────────────────────────────────────────────────
nulls = geo_full.isnull().mean() * 100
nulls = nulls[nulls > 0].sort_values(ascending=False)
if len(nulls):
    print("\n  Null rates:")
    print(nulls.to_string())

# ── Save ──────────────────────────────────────────────────────────────────
out = FEATS / "geo_features.parquet"
geo_full.to_parquet(out, index=False)
log(f"\n  Saved → {out}  ({out.stat().st_size/1e6:.1f} MB)")

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION — mule vs legit separation
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== Validation: mule vs legit ===")

labels = pd.read_parquet(DATA / "train_labels.parquet")
check  = geo_full.merge(labels[['account_id', 'is_mule']], on='account_id', how='inner')

key_cols = [
    'n_geo_txns', 'n_unique_locations', 'location_entropy',
    'geo_spread_km', 'max_dist_from_home_km', 'mean_dist_from_home_km',
    'geo_txn_ratio',
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
    print(f"  {col:<30s} {m.mean():>10.3f} {l.mean():>10.3f} {pval:>10.2e} {sig}")

# Extra: geographic anomaly mules — accounts with geo_spread > 500km
geo_anomaly_mules  = check[(check.is_mule==1) & (check.geo_spread_km > 500)]
geo_anomaly_legit  = check[(check.is_mule==0) & (check.geo_spread_km > 500)]
log(f"\n  Accounts with geo_spread > 500km:")
log(f"    Mules : {len(geo_anomaly_mules)} / {int(check.is_mule.sum())}")
log(f"    Legit : {len(geo_anomaly_legit)} / {int((check.is_mule==0).sum())}")

log("\nDone. Next: update 06_model_v2.py to include geo features, then retrain.")