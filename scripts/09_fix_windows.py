"""
09_fix_windows.py — Fix Temporal IoU Without Retraining

Problem: v2/v3 used peak 3-month windows → IoU dropped from 0.427 to 0.224
Fix: keep v3 model predictions (best AUC/F1) but use smarter window logic

Window strategy (priority order):
  1. Dormant reactivation accounts → burst_start to last_ts
  2. Mobile spike accounts         → mobile_update_date to last_ts
  3. Default                       → middle 60% of activity span
                                     (v1 used 50%, we try 60% for more coverage)

Run: python scripts/09_fix_windows.py
Output: outputs/submissions/submission_v3_fixed_windows.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA    = Path("/home/niranjan/AML/data")
CACHE   = Path("/home/niranjan/AML/cache")
FEATS   = CACHE / "features"
SUBS    = Path("/home/niranjan/AML/outputs/submissions")
SUBS.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Load v3 predictions (keep model scores as-is) ─────────────────────────
log("Loading v3 submission...")
sub_v3 = pd.read_csv(SUBS / "submission_v3.csv")
log(f"  Rows: {len(sub_v3):,}  Predicted mules: {(sub_v3.is_mule > 0).sum():,}")

# ── Load temporal features for window signals ──────────────────────────────
log("Loading temporal features...")
temporal = pd.read_parquet(FEATS / "temporal_features.parquet")

# ── Load txn bounds (first/last ts per account) ────────────────────────────
log("Loading transaction bounds...")
txn_feats = pd.read_parquet(FEATS / "txn_features.parquet",
                             columns=['account_id', 'first_ts', 'last_ts'])
txn_feats['first_dt'] = pd.to_datetime(txn_feats['first_ts'], unit='s', errors='coerce')
txn_feats['last_dt']  = pd.to_datetime(txn_feats['last_ts'],  unit='s', errors='coerce')

# ── Load accounts for mobile update date ──────────────────────────────────
log("Loading account mobile dates...")
accounts = pd.read_parquet(DATA / "accounts.parquet",
                            columns=['account_id', 'last_mobile_update_date'])
accounts['last_mobile_update_date'] = pd.to_datetime(
    accounts['last_mobile_update_date'], errors='coerce'
)

# ── Merge everything onto the submission ─────────────────────────────────
log("Merging signals...")
sub = sub_v3.copy()
sub = sub.merge(txn_feats[['account_id','first_dt','last_dt']], on='account_id', how='left')
sub = sub.merge(
    temporal[['account_id','is_dormant_reactivated','mobile_spike',
              'dormancy_gap_days','peak_window_start','peak_window_end']],
    on='account_id', how='left'
)
sub = sub.merge(accounts[['account_id','last_mobile_update_date']],
                on='account_id', how='left')

# ── Determine the threshold used in v3 ───────────────────────────────────
# v3 best_t_cal = 0.335 — use same threshold to identify predicted mules
THRESHOLD = 0.335
predicted_mule_mask = sub['is_mule'] >= THRESHOLD
log(f"  Predicted mules at t={THRESHOLD}: {predicted_mule_mask.sum():,}")

# ══════════════════════════════════════════════════════════════════════════
# WINDOW LOGIC
# ══════════════════════════════════════════════════════════════════════════
log("Computing windows...")

def compute_window(row):
    """
    Returns (start_str, end_str) for a predicted mule account.
    Priority:
      1. Dormant reactivation → burst onset to last transaction
      2. Mobile spike         → mobile update date to last transaction  
      3. Default              → middle 60% of activity span
    """
    first = row['first_dt']
    last  = row['last_dt']

    if pd.isna(first) or pd.isna(last):
        return '', ''

    span_days = max((last - first).days, 1)

    # ── Case 1: Dormant reactivation ──────────────────────────────────
    if row['is_dormant_reactivated'] == 1 and row['dormancy_gap_days'] > 90:
        # Window starts at peak_window_start (which is the burst onset in
        # temporal features) and ends at last transaction
        burst_start = row['peak_window_start']
        if pd.notna(burst_start):
            return (
                pd.Timestamp(burst_start).strftime('%Y-%m-%dT%H:%M:%S'),
                last.strftime('%Y-%m-%dT%H:%M:%S'),
            )

    # ── Case 2: Mobile spike ──────────────────────────────────────────
    if row['mobile_spike'] == 1 and pd.notna(row['last_mobile_update_date']):
        mob_date = pd.Timestamp(row['last_mobile_update_date'])
        if first <= mob_date <= last:
            return (
                mob_date.strftime('%Y-%m-%dT%H:%M:%S'),
                last.strftime('%Y-%m-%dT%H:%M:%S'),
            )

    # ── Case 3: Default — middle 60% of activity span ─────────────────
    # v1 used 25%-75% (middle 50%). We use 20%-80% (middle 60%) for
    # slightly more coverage while keeping the window centered.
    start = first + pd.Timedelta(days=int(span_days * 0.20))
    end   = first + pd.Timedelta(days=int(span_days * 0.80))
    return (
        start.strftime('%Y-%m-%dT%H:%M:%S'),
        end.strftime('%Y-%m-%dT%H:%M:%S'),
    )

# Apply only to predicted mules; leave others as empty string
starts = []
ends   = []

for _, row in sub.iterrows():
    if not predicted_mule_mask[row.name]:
        starts.append('')
        ends.append('')
        continue
    s, e = compute_window(row)
    starts.append(s)
    ends.append(e)

sub['suspicious_start'] = starts
sub['suspicious_end']   = ends

# ── Build final submission ─────────────────────────────────────────────
out_sub = sub[['account_id', 'is_mule', 'suspicious_start', 'suspicious_end']].copy()

with_windows = (out_sub['suspicious_start'] != '').sum()
log(f"  Predicted mules    : {predicted_mule_mask.sum():,}")
log(f"  With windows       : {with_windows:,}")

# ── Window length stats (sanity check) ───────────────────────────────
mule_rows = out_sub[out_sub['suspicious_start'] != ''].copy()
mule_rows['start_dt'] = pd.to_datetime(mule_rows['suspicious_start'], errors='coerce')
mule_rows['end_dt']   = pd.to_datetime(mule_rows['suspicious_end'],   errors='coerce')
mule_rows['window_days'] = (mule_rows['end_dt'] - mule_rows['start_dt']).dt.days

log(f"\n  Window length stats (days):")
log(f"    Median : {mule_rows['window_days'].median():.0f}")
log(f"    Mean   : {mule_rows['window_days'].mean():.0f}")
log(f"    p10    : {mule_rows['window_days'].quantile(0.10):.0f}")
log(f"    p90    : {mule_rows['window_days'].quantile(0.90):.0f}")

# Case breakdown
dormant_windows = sub[predicted_mule_mask & (sub['is_dormant_reactivated']==1)].shape[0]
mobile_windows  = sub[predicted_mule_mask & (sub['mobile_spike']==1)].shape[0]
default_windows = with_windows - dormant_windows - mobile_windows
log(f"\n  Window type breakdown:")
log(f"    Dormant reactivation : {dormant_windows:,}")
log(f"    Mobile spike         : {mobile_windows:,}")
log(f"    Default (middle 60%) : {default_windows:,}")

# ── Save ──────────────────────────────────────────────────────────────
out_path = SUBS / "submission_v3_fixed_windows.csv"
out_sub.to_csv(out_path, index=False)
log(f"\n  Saved → {out_path}")

# ── Also save a v1-style version (middle 50%) for comparison ─────────
log("\nGenerating v1-style windows (middle 50%) for comparison...")

def compute_window_v1style(row):
    first = row['first_dt']
    last  = row['last_dt']
    if pd.isna(first) or pd.isna(last):
        return '', ''
    span_days = max((last - first).days, 1)
    start = first + pd.Timedelta(days=int(span_days * 0.25))
    end   = first + pd.Timedelta(days=int(span_days * 0.75))
    return (
        start.strftime('%Y-%m-%dT%H:%M:%S'),
        end.strftime('%Y-%m-%dT%H:%M:%S'),
    )

starts_v1, ends_v1 = [], []
for _, row in sub.iterrows():
    if not predicted_mule_mask[row.name]:
        starts_v1.append(''); ends_v1.append('')
        continue
    s, e = compute_window_v1style(row)
    starts_v1.append(s); ends_v1.append(e)

out_v1 = sub[['account_id','is_mule']].copy()
out_v1['suspicious_start'] = starts_v1
out_v1['suspicious_end']   = ends_v1
out_v1_path = SUBS / "submission_v3_windows_50pct.csv"
out_v1.to_csv(out_v1_path, index=False)
log(f"  Saved → {out_v1_path}")

log(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  Window Fix Summary                                     │
  │  Model scores unchanged (v3 predictions)                │
  ├─────────────────────────────────────────────────────────┤
  │  submission_v3_fixed_windows.csv  ← submit this first  │
  │    Windows: middle 60% + dormant/mobile overrides       │
  │                                                         │
  │  submission_v3_windows_50pct.csv  ← try if above fails  │
  │    Windows: middle 50% (pure v1 style)                  │
  └─────────────────────────────────────────────────────────┘
  Expected: AUC/F1 same as v3, IoU closer to v1's 0.427
""")
