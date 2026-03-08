"""
reproduce_best_submission.py — Reproduces Current Best Submission

Leaderboard score: AUC=0.989983 | F1=0.946524 | IoU=0.511942 (896/960)
Submission file  : submission_v4_t0p335.csv

Assumes all feature parquets are already computed:
  cache/features/master_static.parquet       (from 03_txn_features.py)
  cache/features/test_static.parquet
  cache/features/txn_features.parquet
  cache/features/temporal_features.parquet   (from 05_temporal_features.py)
  cache/features/geo_features.parquet        (from 07_geo_features.py)
  cache/features/graph_features.parquet      (from 10_graph_features.py)
  cache/features/salary_features.parquet     (from 10b_salary_features.py)

Run: python scripts/reproduce_best_submission.py
Output: outputs/submissions/submission_best_reproduced.csv
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

DATA    = Path("/home/niranjan/AML/data")
CACHE   = Path("/home/niranjan/AML/cache")
FEATS   = CACHE / "features"
SUBS    = Path("/home/niranjan/AML/outputs/submissions")
OUTPUTS = Path("/home/niranjan/AML/outputs")
SUBS.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.335   # threshold that gave IoU=0.511

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
log("Loading feature tables...")
# ══════════════════════════════════════════════════════════════════════════════

static_train = pd.read_parquet(FEATS / "master_static.parquet")
static_test  = pd.read_parquet(FEATS / "test_static.parquet")
txn          = pd.read_parquet(FEATS / "txn_features.parquet")
labels       = pd.read_parquet(DATA  / "train_labels.parquet")
test_ids     = pd.read_parquet(DATA  / "test_accounts.parquet")
oof_v1       = pd.read_parquet(FEATS / "oof_lgbm.parquet")

temporal_all = pd.read_parquet(FEATS / "temporal_features.parquet")
TEMPORAL_COLS = ['account_id','dormancy_gap_days','dormancy_fraction',
                 'is_dormant_reactivated','pre_mobile_txn_count',
                 'post_mobile_txn_count','post_mobile_txn_ratio',
                 'post_mobile_amt_ratio','mobile_spike','peak_txn_count','peak_amt']

geo_all  = pd.read_parquet(FEATS / "geo_features.parquet")
GEO_COLS = ['account_id','n_geo_txns','n_unique_locations',
            'location_entropy','geo_txn_ratio']

graph_all  = pd.read_parquet(FEATS / "graph_features.parquet")
GRAPH_COLS = ['account_id','is_high_mule_comm','community_mule_density',
              'avg_neighbor_pagerank','in_degree','out_degree']

salary_all  = pd.read_parquet(FEATS / "salary_features.parquet")
SALARY_COLS = ['account_id','salary_month_count','salary_regularity',
               'has_salary_pattern']

temporal_timestamps = temporal_all[['account_id','suspicious_start',
                                     'suspicious_end']].copy()
txn = txn.drop(columns=['activity_span_days'], errors='ignore')

def merge_all(base):
    return (base
            .merge(txn,                         on='account_id', how='left')
            .merge(temporal_all[TEMPORAL_COLS], on='account_id', how='left')
            .merge(geo_all[GEO_COLS],           on='account_id', how='left')
            .merge(graph_all[GRAPH_COLS],       on='account_id', how='left')
            .merge(salary_all[SALARY_COLS],     on='account_id', how='left'))

train = merge_all(static_train)
test  = merge_all(static_test)
log(f"  train: {train.shape}  test: {test.shape}")

# ══════════════════════════════════════════════════════════════════════════════
log("Red herring label cleaning (weight=0.3 for low-confidence mules)...")
# ══════════════════════════════════════════════════════════════════════════════

oof_check = oof_v1.drop(columns=['is_mule'], errors='ignore').merge(
    labels[['account_id','alert_reason','is_mule']], on='account_id', how='left'
)
rh_mask = (
    (oof_check['oof_lgbm'] < 0.05) |
    (oof_check['alert_reason'].isna()) |
    (oof_check['alert_reason'] == 'Round Amount Pattern')
)
rh_accounts = set(oof_check[oof_check['is_mule'] == 1][rh_mask]['account_id'])
log(f"  Downweighted: {len(rh_accounts):,} / {int(labels.is_mule.sum()):,} mules")

sample_weights = train[['account_id']].copy()
sample_weights['sample_weight'] = 1.0
sample_weights.loc[
    sample_weights['account_id'].isin(rh_accounts), 'sample_weight'
] = 0.3

# ══════════════════════════════════════════════════════════════════════════════
log("Preprocessing features...")
# ══════════════════════════════════════════════════════════════════════════════

TARGET = 'is_mule'
DROP   = ['account_id','customer_id','is_mule','branch_code',
          'account_status','was_frozen']

CAT_COLS = [c for c in [
    'product_family','scheme_code','rural_branch','kyc_compliant',
    'nomination_flag','cheque_allowed','cheque_availed','pan_available',
    'aadhaar_available','passport_available','mobile_banking_flag',
    'internet_banking_flag','atm_card_flag','demat_flag','credit_card_flag',
    'fastag_flag','gender','joint_account_flag','nri_flag','branch_type',
] if c in train.columns]

for col in CAT_COLS:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).fillna('MISSING').astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].fillna('MISSING').astype(str))
    test[col]  = le.transform(test[col].fillna('MISSING').astype(str))

feature_cols = [c for c in train.columns
                if c not in DROP and train[c].dtype != 'object']

X       = train[feature_cols].fillna(-999)
y       = train[TARGET].values
weights = sample_weights['sample_weight'].values
X_test  = test[feature_cols].fillna(-999)

neg, pos  = (y == 0).sum(), (y == 1).sum()
scale_pos = neg / pos
log(f"  Features: {len(feature_cols)}  |  scale_pos_weight: {scale_pos:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
log("Training LightGBM — 5-fold stratified CV (800 trees, CPU)...")
# ══════════════════════════════════════════════════════════════════════════════

PARAMS = dict(
    objective        = 'binary',
    metric           = 'auc',
    boosting_type    = 'gbdt',
    num_leaves       = 127,
    learning_rate    = 0.03,
    n_estimators     = 800,
    scale_pos_weight = scale_pos,
    subsample        = 0.8,
    subsample_freq   = 1,
    colsample_bytree = 0.8,
    min_child_samples= 20,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = 42,
    n_jobs           = -1,
    # GPU: set device='gpu', gpu_device_id=3 if a free GPU is available
    verbose          = -1,
)

cv          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof         = np.zeros(len(X))
test_preds  = np.zeros(len(X_test))
fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
    log(f"  Fold {fold+1}/5")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    w_tr        = weights[tr_idx]

    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.log_evaluation(200)])

    val_pred     = model.predict_proba(X_val)[:, 1]
    oof[val_idx] = val_pred
    auc = roc_auc_score(y_val, val_pred)
    ap  = average_precision_score(y_val, val_pred)
    ths = np.linspace(0.01, 0.99, 100)
    f1s = [f1_score(y_val, (val_pred>t).astype(int), zero_division=0) for t in ths]
    log(f"    AUC={auc:.4f}  AP={ap:.4f}  F1={max(f1s):.4f}")
    fold_scores.append({'fold': fold+1, 'auc': auc, 'ap': ap})
    test_preds += model.predict_proba(X_test)[:, 1] / 5

scores_df = pd.DataFrame(fold_scores)
oof_auc   = roc_auc_score(y, oof)
oof_ap    = average_precision_score(y, oof)
log(f"\n  OOF AUC={oof_auc:.4f}  AP={oof_ap:.4f}")
log(f"  Per-fold:\n{scores_df.to_string(index=False)}")

# ── Calibrate ─────────────────────────────────────────────────────────────
iso           = IsotonicRegression(out_of_bounds='clip')
iso.fit(oof, y)
oof_cal       = iso.predict(oof)
test_preds_cal= iso.predict(test_preds)
oof_auc_cal   = roc_auc_score(y, oof_cal)
log(f"  Calibrated AUC={oof_auc_cal:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
log("\nGenerating submission with fixed windows...")
# ══════════════════════════════════════════════════════════════════════════════

sub = test_ids[['account_id']].copy()
sub['is_mule'] = test_preds_cal

# Load txn bounds for window computation
txn_bounds = pd.read_parquet(FEATS / "txn_features.parquet",
                              columns=['account_id','first_ts','last_ts'])
txn_bounds['first_dt'] = pd.to_datetime(txn_bounds['first_ts'], unit='s', errors='coerce')
txn_bounds['last_dt']  = pd.to_datetime(txn_bounds['last_ts'],  unit='s', errors='coerce')

accts = pd.read_parquet(DATA / "accounts.parquet",
                         columns=['account_id','last_mobile_update_date'])
accts['last_mobile_update_date'] = pd.to_datetime(
    accts['last_mobile_update_date'], errors='coerce'
)

sub = sub.merge(txn_bounds[['account_id','first_dt','last_dt']], on='account_id', how='left')
sub = sub.merge(
    temporal_all[['account_id','is_dormant_reactivated','mobile_spike',
                  'dormancy_gap_days','peak_window_start']],
    on='account_id', how='left'
)
sub = sub.merge(accts, on='account_id', how='left')

predicted_mask = sub['is_mule'] >= THRESHOLD
log(f"  Predicted mules at t={THRESHOLD}: {predicted_mask.sum():,}")

starts, ends = [], []
for _, row in sub.iterrows():
    if not predicted_mask[row.name]:
        starts.append(''); ends.append(''); continue

    first, last = row['first_dt'], row['last_dt']
    if pd.isna(first) or pd.isna(last):
        starts.append(''); ends.append(''); continue

    span = max((last - first).days, 1)

    # Priority 1: dormant reactivation → burst onset to last transaction
    if row['is_dormant_reactivated'] == 1 and pd.notna(row['peak_window_start']):
        starts.append(pd.Timestamp(row['peak_window_start']).strftime('%Y-%m-%dT%H:%M:%S'))
        ends.append(last.strftime('%Y-%m-%dT%H:%M:%S'))
        continue

    # Priority 2: mobile spike → mobile update date to last transaction
    if row['mobile_spike'] == 1 and pd.notna(row['last_mobile_update_date']):
        mob = pd.Timestamp(row['last_mobile_update_date'])
        if first <= mob <= last:
            starts.append(mob.strftime('%Y-%m-%dT%H:%M:%S'))
            ends.append(last.strftime('%Y-%m-%dT%H:%M:%S'))
            continue

    # Default: middle 60% of activity span
    s = first + pd.Timedelta(days=int(span * 0.20))
    e = first + pd.Timedelta(days=int(span * 0.80))
    starts.append(s.strftime('%Y-%m-%dT%H:%M:%S'))
    ends.append(e.strftime('%Y-%m-%dT%H:%M:%S'))

sub['suspicious_start'] = starts
sub['suspicious_end']   = ends

out = sub[['account_id','is_mule','suspicious_start','suspicious_end']].copy()
out_path = SUBS / "submission_best_reproduced.csv"
out.to_csv(out_path, index=False)

with_windows = (out['suspicious_start'] != '').sum()

log(f"""
  ┌──────────────────────────────────────────────────────────┐
  │  Reproduced Best Submission                              │
  ├──────────────────────────────────────────────────────────┤
  │  Target score : AUC=0.9900  F1=0.9465  IoU=0.5119       │
  │  OOF AUC (raw)      : {oof_auc:.4f}                         │
  │  OOF AUC (cal)      : {oof_auc_cal:.4f}                         │
  │  Threshold          : {THRESHOLD}                          │
  │  Predicted mules    : {predicted_mask.sum():,}                         │
  │  With time windows  : {with_windows:,}                         │
  │  Features           : {len(feature_cols)}                           │
  ├──────────────────────────────────────────────────────────┤
  │  Window logic:                                           │
  │    1. Dormant reactivation → burst_start to last_ts      │
  │    2. Mobile spike         → mobile_date to last_ts      │
  │    3. Default              → middle 60% of activity span │
  └──────────────────────────────────────────────────────────┘
  Saved → {out_path}
""")
