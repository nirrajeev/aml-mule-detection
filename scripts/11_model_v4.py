"""
11_model_v4.py — LightGBM + CatBoost Ensemble (Final)

Feature set: static + txn + temporal + graph (5) + salary (3)
Models:
  - LightGBM (same config as v3, fixed 800 trees)
  - CatBoost (GPU, native categoricals, 800 iterations)
  - Ensemble: weighted average OOF-tuned blend

Run: python scripts/11_model_v4.py
Output:
  cache/features/oof_lgbm_v4.parquet
  cache/features/oof_cat_v4.parquet
  outputs/submissions/submission_v4.csv
  outputs/cv_scores_v4.csv
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_recall_curve)
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

DATA    = Path("/home/niranjan/AML/data")
CACHE   = Path("/home/niranjan/AML/cache")
FEATS   = CACHE / "features"
PLOTS   = Path("/home/niranjan/AML/notebooks/eda_plots")
SUBS    = Path("/home/niranjan/AML/outputs/submissions")
OUTPUTS = Path("/home/niranjan/AML/outputs")
for p in [SUBS, PLOTS, OUTPUTS]: p.mkdir(parents=True, exist_ok=True)

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

# Temporal features
temporal_all = pd.read_parquet(FEATS / "temporal_features.parquet")
TEMPORAL_COLS = [
    'account_id',
    'dormancy_gap_days', 'dormancy_fraction', 'is_dormant_reactivated',
    'pre_mobile_txn_count', 'post_mobile_txn_count',
    'post_mobile_txn_ratio', 'post_mobile_amt_ratio', 'mobile_spike',
    'peak_txn_count', 'peak_amt',
]
temporal_feats      = temporal_all[TEMPORAL_COLS].copy()
temporal_timestamps = temporal_all[['account_id', 'suspicious_start',
                                     'suspicious_end']].copy()

# Graph features — only the 5 with significant signal
graph_all  = pd.read_parquet(FEATS / "graph_features.parquet")
GRAPH_COLS = ['account_id', 'is_high_mule_comm', 'community_mule_density',
              'avg_neighbor_pagerank', 'in_degree', 'out_degree']
graph_feats = graph_all[GRAPH_COLS].copy()

# Salary features — only 3 significant ones (drop drain ratio and mule signal)
salary_all  = pd.read_parquet(FEATS / "salary_features.parquet")
SALARY_COLS = ['account_id', 'salary_month_count', 'salary_regularity',
               'has_salary_pattern']
salary_feats = salary_all[SALARY_COLS].copy()

# Geo features
geo_all  = pd.read_parquet(FEATS / "geo_features.parquet")
GEO_COLS = ['account_id', 'n_geo_txns', 'n_unique_locations',
            'location_entropy', 'geo_txn_ratio']
geo_feats = geo_all[GEO_COLS].copy()

# Merge all
def merge_all(base):
    return (base
            .merge(txn,           on='account_id', how='left')
            .merge(temporal_feats,on='account_id', how='left')
            .merge(geo_feats,     on='account_id', how='left')
            .merge(graph_feats,   on='account_id', how='left')
            .merge(salary_feats,  on='account_id', how='left'))

train = merge_all(static_train)
test  = merge_all(static_test)
log(f"  train: {train.shape}  test: {test.shape}")

# ══════════════════════════════════════════════════════════════════════════════
log("Red herring label cleaning...")
# ══════════════════════════════════════════════════════════════════════════════

oof_check = oof_v1.drop(columns=['is_mule'], errors='ignore').merge(
    labels[['account_id', 'alert_reason', 'is_mule']], on='account_id', how='left'
)
rh_mask = (
    (oof_check['oof_lgbm'] < 0.05) |
    (oof_check['alert_reason'].isna()) |
    (oof_check['alert_reason'] == 'Round Amount Pattern')
)
rh_accounts = set(oof_check[oof_check['is_mule'] == 1][rh_mask]['account_id'])
log(f"  Red herring downweighted (w=0.3): {len(rh_accounts):,} / {int(labels.is_mule.sum()):,}")

sample_weights = train[['account_id']].copy()
sample_weights['sample_weight'] = 1.0
sample_weights.loc[
    sample_weights['account_id'].isin(rh_accounts), 'sample_weight'
] = 0.3
log(f"  Confirmed weight=0.3 accounts: {(sample_weights['sample_weight']==0.3).sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
log("Preprocessing features...")
# ══════════════════════════════════════════════════════════════════════════════

TARGET = 'is_mule'
DROP   = [
    'account_id', 'customer_id', 'is_mule', 'branch_code',
    'account_status', 'was_frozen',
]

CAT_COLS = [c for c in [
    'product_family', 'scheme_code', 'rural_branch',
    'kyc_compliant', 'nomination_flag', 'cheque_allowed', 'cheque_availed',
    'pan_available', 'aadhaar_available', 'passport_available',
    'mobile_banking_flag', 'internet_banking_flag', 'atm_card_flag',
    'demat_flag', 'credit_card_flag', 'fastag_flag',
    'gender', 'joint_account_flag', 'nri_flag', 'branch_type',
] if c in train.columns]

# LabelEncode for LightGBM
train_lgb = train.copy()
test_lgb  = test.copy()
for col in CAT_COLS:
    le = LabelEncoder()
    combined = pd.concat([train_lgb[col], test_lgb[col]], axis=0).fillna('MISSING').astype(str)
    le.fit(combined)
    train_lgb[col] = le.transform(train_lgb[col].fillna('MISSING').astype(str))
    test_lgb[col]  = le.transform(test_lgb[col].fillna('MISSING').astype(str))

feature_cols = [c for c in train.columns
                if c not in DROP and train[c].dtype != 'object']

X_lgb   = train_lgb[feature_cols].fillna(-999)
y       = train[TARGET].values
weights = sample_weights['sample_weight'].values
X_test_lgb = test_lgb[feature_cols].fillna(-999)

# CatBoost uses raw strings for categoricals
X_cat      = train[feature_cols].copy()
X_test_cat = test[feature_cols].copy()
for col in CAT_COLS:
    if col in X_cat.columns:
        X_cat[col]      = X_cat[col].fillna('MISSING').astype(str)
        X_test_cat[col] = X_test_cat[col].fillna('MISSING').astype(str)
# Fill numeric NaN with -999 for CatBoost
num_cols = [c for c in feature_cols if c not in CAT_COLS]
X_cat[num_cols]      = X_cat[num_cols].fillna(-999)
X_test_cat[num_cols] = X_test_cat[num_cols].fillna(-999)

cat_indices = [feature_cols.index(c) for c in CAT_COLS if c in feature_cols]

neg, pos  = (y == 0).sum(), (y == 1).sum()
scale_pos = neg / pos
log(f"  Features: {len(feature_cols)}  |  scale_pos_weight: {scale_pos:.1f}")
log(f"  New features vs v3: {[c for c in GRAPH_COLS[1:]+SALARY_COLS[1:] if c in feature_cols]}")

# ══════════════════════════════════════════════════════════════════════════════
log("\n=== Training LightGBM — 5-fold ===")
# ══════════════════════════════════════════════════════════════════════════════

LGB_PARAMS = dict(
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
    verbose          = -1,
)

cv             = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb        = np.zeros(len(X_lgb))
test_preds_lgb = np.zeros(len(X_test_lgb))
lgb_models     = []
lgb_scores     = []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_lgb, y)):
    log(f"  Fold {fold+1}/5")
    X_tr, X_val = X_lgb.iloc[tr_idx], X_lgb.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    w_tr        = weights[tr_idx]

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.log_evaluation(200)])

    val_pred           = model.predict_proba(X_val)[:, 1]
    oof_lgb[val_idx]   = val_pred
    auc                = roc_auc_score(y_val, val_pred)
    ap                 = average_precision_score(y_val, val_pred)
    ths  = np.linspace(0.01, 0.99, 100)
    f1s  = [f1_score(y_val, (val_pred > t).astype(int), zero_division=0) for t in ths]
    log(f"    LGB AUC={auc:.4f}  AP={ap:.4f}  F1={max(f1s):.4f}")
    lgb_scores.append({'fold': fold+1, 'model': 'lgb', 'auc': auc, 'ap': ap})

    test_preds_lgb += model.predict_proba(X_test_lgb)[:, 1] / 5
    lgb_models.append(model)

lgb_oof_auc = roc_auc_score(y, oof_lgb)
log(f"\n  LGB OOF AUC: {lgb_oof_auc:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
log("\n=== Training CatBoost — 5-fold ===")
# ══════════════════════════════════════════════════════════════════════════════

CAT_PARAMS = dict(
    iterations        = 800,
    learning_rate     = 0.03,
    depth             = 8,
    l2_leaf_reg       = 3.0,
    random_strength   = 1.0,
    bagging_temperature = 1.0,
    scale_pos_weight  = scale_pos,
    eval_metric       = 'AUC',
    task_type         = 'CPU',
    thread_count      = -1,
    random_seed       = 42,
    verbose           = 200,
)

oof_cat        = np.zeros(len(X_cat))
test_preds_cat = np.zeros(len(X_test_cat))
cat_models     = []
cat_scores     = []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_cat, y)):
    log(f"  Fold {fold+1}/5")
    X_tr      = X_cat.iloc[tr_idx]
    X_val_f   = X_cat.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    w_tr        = weights[tr_idx]

    train_pool = Pool(X_tr,      y_tr,    cat_features=cat_indices, weight=w_tr)
    val_pool   = Pool(X_val_f,   y_val,   cat_features=cat_indices)
    test_pool  = Pool(X_test_cat,         cat_features=cat_indices)

    model = CatBoostClassifier(**CAT_PARAMS)
    model.fit(train_pool, eval_set=val_pool, use_best_model=False)

    val_pred         = model.predict_proba(val_pool)[:, 1]
    oof_cat[val_idx] = val_pred
    auc              = roc_auc_score(y_val, val_pred)
    ap               = average_precision_score(y_val, val_pred)
    ths = np.linspace(0.01, 0.99, 100)
    f1s = [f1_score(y_val, (val_pred > t).astype(int), zero_division=0) for t in ths]
    log(f"    CAT AUC={auc:.4f}  AP={ap:.4f}  F1={max(f1s):.4f}")
    cat_scores.append({'fold': fold+1, 'model': 'cat', 'auc': auc, 'ap': ap})

    test_preds_cat += model.predict_proba(test_pool)[:, 1] / 5
    cat_models.append(model)

cat_oof_auc = roc_auc_score(y, oof_cat)
log(f"\n  CAT OOF AUC: {cat_oof_auc:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
log("\n=== Ensemble: OOF-tuned blend ===")
# ══════════════════════════════════════════════════════════════════════════════

# Find optimal blend weight on OOF predictions
best_w, best_auc = 0.5, 0.0
for w in np.linspace(0.0, 1.0, 21):
    blend = w * oof_lgb + (1 - w) * oof_cat
    auc   = roc_auc_score(y, blend)
    if auc > best_auc:
        best_auc, best_w = auc, w

log(f"  Best blend: {best_w:.2f} × LGB + {1-best_w:.2f} × CAT → OOF AUC={best_auc:.4f}")
log(f"  LGB alone : {lgb_oof_auc:.4f}")
log(f"  CAT alone : {cat_oof_auc:.4f}")

oof_ensemble        = best_w * oof_lgb + (1 - best_w) * oof_cat
test_preds_ensemble = best_w * test_preds_lgb + (1 - best_w) * test_preds_cat

# ── Calibrate ensemble ────────────────────────────────────────────────────
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(oof_ensemble, y)
oof_cal         = iso.predict(oof_ensemble)
test_preds_cal  = iso.predict(test_preds_ensemble)

oof_auc_cal = roc_auc_score(y, oof_cal)
ths         = np.linspace(0.01, 0.99, 200)
f1s_cal     = [f1_score(y, (oof_cal > t).astype(int), zero_division=0) for t in ths]
best_t_cal  = float(ths[np.argmax(f1s_cal)])
best_f1_cal = float(max(f1s_cal))

log(f"  Calibrated AUC: {oof_auc_cal:.4f}  F1: {best_f1_cal:.4f}@t={best_t_cal:.3f}")

# ── Version comparison ────────────────────────────────────────────────────
for vname, fname in [('v2','cv_scores_v2.csv'), ('v3','cv_scores_v3.csv')]:
    try:
        prev     = pd.read_csv(OUTPUTS / fname)
        prev_auc = prev['auc'].mean()
        diff     = best_auc - prev_auc
        log(f"  {vname}→v4: {prev_auc:.4f}→{best_auc:.4f}  "
            f"({'▲' if diff > 0 else '▼'}{abs(diff):.4f})")
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
log("\nSHAP — LightGBM best fold...")
# ══════════════════════════════════════════════════════════════════════════════

lgb_aucs      = [s['auc'] for s in lgb_scores]
best_fold_idx = int(np.argmax(lgb_aucs))
best_lgb      = lgb_models[best_fold_idx]
log(f"  Using LGB fold {best_fold_idx+1} (AUC={lgb_aucs[best_fold_idx]:.4f})")

val_sample = X_lgb.sample(min(3000, len(X_lgb)), random_state=42)
explainer  = shap.TreeExplainer(best_lgb)
sv         = explainer.shap_values(val_sample)
if isinstance(sv, list): sv = sv[1]

feat_imp = pd.DataFrame({
    'feature':   feature_cols,
    'shap_mean': np.abs(sv).mean(axis=0),
    'lgbm_gain': best_lgb.booster_.feature_importance(importance_type='gain'),
}).sort_values('shap_mean', ascending=False).reset_index(drop=True)

log(f"\n  Top 30 features (v4):")
print(feat_imp.head(30).to_string(index=False))

# Highlight new features
new_feat_names = GRAPH_COLS[1:] + SALARY_COLS[1:] + GEO_COLS[1:]
new_in_top30   = feat_imp.head(30)[feat_imp.head(30)['feature'].isin(new_feat_names)]
log(f"\n  New features in top 30:")
for _, row in new_in_top30.iterrows():
    rank = int(feat_imp[feat_imp['feature'] == row['feature']].index[0]) + 1
    log(f"    #{rank:3d}  {row['feature']:<30s}  SHAP={row['shap_mean']:.4f}")

feat_imp.to_parquet(FEATS / "feature_importance_v4.parquet", index=False)
feat_imp.to_csv(FEATS / "feature_importance_v4.csv", index=False)

# SHAP plot
new_set = set(new_feat_names)
fig, ax = plt.subplots(figsize=(10, 14))
t30     = feat_imp.head(30)
colors  = ['crimson' if f in new_set else 'steelblue' for f in t30['feature'][::-1]]
ax.barh(t30['feature'][::-1], t30['shap_mean'][::-1], color=colors, alpha=0.85)
ax.set_xlabel('Mean |SHAP|')
ax.set_title(f'Top 30 Features — v4 Ensemble  (OOF AUC={best_auc:.4f})\nRed = new features')
plt.tight_layout()
plt.savefig(PLOTS / 'shap_v4.png', dpi=130, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
log("\nSaving OOF predictions...")
# ══════════════════════════════════════════════════════════════════════════════

oof_df = train[['account_id']].copy()
oof_df['is_mule']            = y
oof_df['oof_lgb_v4']         = oof_lgb
oof_df['oof_cat_v4']         = oof_cat
oof_df['oof_ensemble_v4']    = oof_ensemble
oof_df['oof_ensemble_v4_cal']= oof_cal
oof_df.to_parquet(FEATS / "oof_v4.parquet", index=False)

scores_df = pd.DataFrame(lgb_scores + cat_scores)
scores_df.to_csv(OUTPUTS / "cv_scores_v4.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
log("\nGenerating submission...")
# ══════════════════════════════════════════════════════════════════════════════

sub = test_ids[['account_id']].copy()
sub['is_mule'] = test_preds_cal

sub = sub.merge(temporal_timestamps, on='account_id', how='left')

# Apply fixed window logic (same as 09_fix_windows.py — middle 60% default)
txn_bounds = pd.read_parquet(FEATS / "txn_features.parquet",
                              columns=['account_id', 'first_ts', 'last_ts'])
txn_bounds['first_dt'] = pd.to_datetime(txn_bounds['first_ts'], unit='s', errors='coerce')
txn_bounds['last_dt']  = pd.to_datetime(txn_bounds['last_ts'],  unit='s', errors='coerce')
sub = sub.merge(txn_bounds[['account_id','first_dt','last_dt']], on='account_id', how='left')
sub = sub.merge(
    temporal_all[['account_id','is_dormant_reactivated','peak_window_start']],
    on='account_id', how='left'
)

THRESHOLD = best_t_cal
predicted_mask = sub['is_mule'] >= THRESHOLD

starts, ends = [], []
for _, row in sub.iterrows():
    if not predicted_mask[row.name]:
        starts.append(''); ends.append(''); continue

    first = row['first_dt']; last = row['last_dt']
    if pd.isna(first) or pd.isna(last):
        starts.append(''); ends.append(''); continue

    span = max((last - first).days, 1)

    # Dormant reactivation → burst onset to last_ts
    if row['is_dormant_reactivated'] == 1 and pd.notna(row['peak_window_start']):
        starts.append(pd.Timestamp(row['peak_window_start']).strftime('%Y-%m-%dT%H:%M:%S'))
        ends.append(last.strftime('%Y-%m-%dT%H:%M:%S'))
        continue

    # Default → middle 60%
    s = first + pd.Timedelta(days=int(span * 0.20))
    e = first + pd.Timedelta(days=int(span * 0.80))
    starts.append(s.strftime('%Y-%m-%dT%H:%M:%S'))
    ends.append(e.strftime('%Y-%m-%dT%H:%M:%S'))

sub['suspicious_start'] = starts
sub['suspicious_end']   = ends

out_sub  = sub[['account_id', 'is_mule', 'suspicious_start', 'suspicious_end']]
out_path = SUBS / "submission_v4.csv"
out_sub.to_csv(out_path, index=False)

predicted_mules = predicted_mask.sum()
with_windows    = (sub['suspicious_start'] != '').sum()

log(f"""
  ┌──────────────────────────────────────────────────────────┐
  │  v4 Ensemble Results                                     │
  ├──────────────────────────────────────────────────────────┤
  │  LGB OOF AUC          : {lgb_oof_auc:.4f}                         │
  │  CAT OOF AUC          : {cat_oof_auc:.4f}                         │
  │  Ensemble OOF AUC     : {best_auc:.4f}  (w={best_w:.2f} LGB)         │
  │  Calibrated AUC       : {oof_auc_cal:.4f}                         │
  │  F1 (calibrated)      : {best_f1_cal:.4f} @ t={best_t_cal:.3f}            │
  ├──────────────────────────────────────────────────────────┤
  │  Features             : {len(feature_cols)}                           │
  │  Predicted mules      : {predicted_mules:,}                         │
  │  With time windows    : {with_windows:,}                         │
  └──────────────────────────────────────────────────────────┘
  Saved → {out_path}
""")
