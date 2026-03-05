"""
06_model_v2.py — LightGBM v2
Improvements over baseline:
  1. Adds temporal features (dormancy, mobile spike, suspicious windows)
  2. Red herring label cleaning — downweights/removes low-confidence mule labels
  3. Better probability calibration via isotonic regression
  4. Updated submission with real suspicious_start / suspicious_end timestamps

Run: python scripts/06_model_v2.py
Output:
  cache/features/oof_lgbm_v2.parquet
  outputs/submissions/submission_v2.csv
  outputs/cv_scores_v2.csv
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score,
                             average_precision_score, precision_recall_curve)
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
import warnings
import json
warnings.filterwarnings('ignore')

DATA    = Path("/home/niranjan/AML/data")
CACHE   = Path("/home/niranjan/AML/cache")
FEATS   = CACHE / "features"
PLOTS   = Path("/home/niranjan/AML/notebooks/eda_plots")
SUBS    = Path("/home/niranjan/AML/outputs/submissions")
OUTPUTS = Path("/home/niranjan/AML/outputs")
SUBS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
log("Loading feature tables...")
# ══════════════════════════════════════════════════════════════════════════════

static_train = pd.read_parquet(FEATS / "master_static.parquet")
static_test  = pd.read_parquet(FEATS / "test_static.parquet")
txn          = pd.read_parquet(FEATS / "txn_features.parquet")
temporal     = pd.read_parquet(FEATS / "temporal_features.parquet")
oof_v1       = pd.read_parquet(FEATS / "oof_lgbm.parquet")   # from baseline
test_ids     = pd.read_parquet(DATA  / "test_accounts.parquet")
labels       = pd.read_parquet(DATA  / "train_labels.parquet")

# Temporal cols to keep as ML features (exclude timestamp cols used for submission)
TEMPORAL_FEAT_COLS = [
    'account_id',
    'dormancy_gap_days', 'dormancy_fraction', 'is_dormant_reactivated',
    'pre_mobile_txn_count', 'post_mobile_txn_count',
    'post_mobile_txn_ratio', 'post_mobile_amt_ratio', 'mobile_spike',
    'peak_txn_count', 'peak_amt',
]
temporal_feats = temporal[TEMPORAL_FEAT_COLS].copy()

# Keep timestamp cols separately for submission generation
temporal_timestamps = temporal[['account_id', 'suspicious_start', 'suspicious_end']].copy()

# Merge everything
train = (static_train
         .merge(txn,           on='account_id', how='left')
         .merge(temporal_feats,on='account_id', how='left'))

test  = (static_test
         .merge(txn,           on='account_id', how='left')
         .merge(temporal_feats,on='account_id', how='left'))

log(f"  train: {train.shape}  test: {test.shape}")

# ══════════════════════════════════════════════════════════════════════════════
log("Red herring label cleaning...")
# ══════════════════════════════════════════════════════════════════════════════

# Strategy: merge OOF predictions from v1 with alert_reason.
# Accounts labelled mule=1 but scored very low by the model are likely noise.
# We use sample_weight to downweight them rather than hard-flipping labels,
# which is safer when we're not 100% certain.

oof_check = oof_v1.drop(columns=['is_mule'], errors='ignore').merge(
    labels[['account_id', 'alert_reason', 'is_mule']],
    on='account_id', how='left'
)

# Red herring conditions (any one of these = suspicious label):
# 1. Model gave prob < 0.05 (very confident it's legit)
# 2. alert_reason is None/NaN (no reason given)
# 3. alert_reason is Round Amount — confirmed red herring (negative SHAP signal)
rh_mask = (
    (oof_check['oof_lgbm'] < 0.05) |
    (oof_check['alert_reason'].isna()) |
    (oof_check['alert_reason'] == 'Round Amount Pattern')
)
rh_accounts = set(oof_check[oof_check['is_mule'] == 1][rh_mask]['account_id'])
log(f"  Red herring candidates: {len(rh_accounts):,} / {int(labels.is_mule.sum()):,} mules")

# Alert reason breakdown for red herring candidates
rh_detail = oof_check[oof_check['account_id'].isin(rh_accounts)]
log(f"  Alert reason breakdown:")
print(rh_detail['alert_reason'].value_counts(dropna=False).to_string())

# Build sample weights: red herring mules get weight 0.1 (not 0 — keep some signal)
train_with_weights = train[['account_id']].copy()
train_with_weights['sample_weight'] = 1.0
train_with_weights.loc[
    train_with_weights['account_id'].isin(rh_accounts), 'sample_weight'
] = 0.3

log(f"  Accounts with weight=0.1: {(train_with_weights['sample_weight'] == 0.1).sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
log("Preprocessing features...")
# ══════════════════════════════════════════════════════════════════════════════

TARGET = 'is_mule'
DROP   = [
    'account_id', 'customer_id', 'is_mule', 'branch_code',
    'account_status', 'was_frozen',       # confirmed leakage
    'alert_reason', 'sample_weight',      # meta columns
]

CAT_COLS = [c for c in [
    'product_family', 'scheme_code', 'rural_branch',
    'kyc_compliant', 'nomination_flag', 'cheque_allowed', 'cheque_availed',
    'pan_available', 'aadhaar_available', 'passport_available',
    'mobile_banking_flag', 'internet_banking_flag', 'atm_card_flag',
    'demat_flag', 'credit_card_flag', 'fastag_flag',
    'gender', 'joint_account_flag', 'nri_flag', 'branch_type',
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
weights = train_with_weights['sample_weight'].values
X_test  = test[feature_cols].fillna(-999)

neg, pos    = (y == 0).sum(), (y == 1).sum()
scale_pos   = neg / pos

log(f"  Features : {len(feature_cols)}")
log(f"  Mule rate: {y.mean():.4%}  |  scale_pos_weight: {scale_pos:.1f}")

# New temporal features added vs v1
new_feats = [c for c in TEMPORAL_FEAT_COLS[1:] if c in feature_cols]
log(f"  New temporal features added: {new_feats}")

# ══════════════════════════════════════════════════════════════════════════════
log("Training LightGBM v2 — 5-fold stratified CV...")
# ══════════════════════════════════════════════════════════════════════════════

PARAMS = dict(
    objective        = 'binary',
    metric           = 'auc',
    boosting_type    = 'gbdt',
    num_leaves       = 127,
    learning_rate    = 0.03,
    n_estimators     = 3000,
    scale_pos_weight = scale_pos,
    subsample        = 0.8,
    subsample_freq   = 1,
    colsample_bytree = 0.8,
    min_child_samples= 20,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = 42,
    n_jobs           = -1,
    device           = 'gpu',
    gpu_device_id    = 0,
    verbose          = -1,
)

cv          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof         = np.zeros(len(X))
test_preds  = np.zeros(len(X_test))
models      = []
fold_scores = []
importances = pd.DataFrame({'feature': feature_cols})

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
    log(f"\n  Fold {fold+1}/5")

    X_tr, X_val   = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val   = y[tr_idx],      y[val_idx]
    w_tr          = weights[tr_idx]

    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(
        X_tr, y_tr,
        sample_weight  = w_tr,
        eval_set       = [(X_val, y_val)],
        callbacks      = [
            lgb.early_stopping(150, verbose=False),
            lgb.log_evaluation(200),
        ],
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    oof[val_idx] = val_pred

    auc = roc_auc_score(y_val, val_pred)
    ap  = average_precision_score(y_val, val_pred)

    ths  = np.linspace(0.01, 0.99, 100)
    f1s  = [f1_score(y_val, (val_pred > t).astype(int), zero_division=0) for t in ths]
    bt   = float(ths[np.argmax(f1s)])
    bf1  = float(max(f1s))

    log(f"    AUC={auc:.4f}  AP={ap:.4f}  F1={bf1:.4f}@{bt:.2f}  trees={model.best_iteration_}")
    fold_scores.append({'fold': fold+1, 'auc': auc, 'ap': ap,
                        'best_f1': bf1, 'best_t': bt,
                        'best_iter': model.best_iteration_})

    test_preds += model.predict_proba(X_test)[:, 1] / 5
    importances[f'fold_{fold+1}'] = model.feature_importances_
    models.append(model)

# ── OOF summary ───────────────────────────────────────────────────────────
scores_df   = pd.DataFrame(fold_scores)
oof_auc     = roc_auc_score(y, oof)
oof_ap      = average_precision_score(y, oof)
ths         = np.linspace(0.01, 0.99, 200)
f1s         = [f1_score(y, (oof > t).astype(int), zero_division=0) for t in ths]
best_t_oof  = float(ths[np.argmax(f1s)])
best_f1_oof = float(max(f1s))

log(f"\n  OOF AUC={oof_auc:.4f}  AP={oof_ap:.4f}  F1={best_f1_oof:.4f}@t={best_t_oof:.3f}")
log(f"\n  Per-fold:\n{scores_df.to_string(index=False)}")

scores_df.to_csv(OUTPUTS / "cv_scores_v2.csv", index=False)

# ── Comparison vs v1 ──────────────────────────────────────────────────────
try:
    v1 = pd.read_csv(OUTPUTS / "cv_scores_lgbm_v1.csv")
    log(f"\n  v1 mean AUC: {v1['auc'].mean():.4f}  →  v2 mean AUC: {scores_df['auc'].mean():.4f}  "
        f"({'▲' if scores_df['auc'].mean() > v1['auc'].mean() else '▼'} "
        f"{abs(scores_df['auc'].mean() - v1['auc'].mean()):.4f})")
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════════════════
log("\nCalibrating probabilities (isotonic regression)...")
# ══════════════════════════════════════════════════════════════════════════════

# Fit isotonic regression on OOF predictions to fix threshold instability
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(oof, y)
oof_cal       = iso.predict(oof)
test_preds_cal= iso.predict(test_preds)

oof_auc_cal = roc_auc_score(y, oof_cal)
log(f"  Calibrated OOF AUC: {oof_auc_cal:.4f}  (raw: {oof_auc:.4f})")

# Recalculate best threshold on calibrated probs
f1s_cal     = [f1_score(y, (oof_cal > t).astype(int), zero_division=0) for t in ths]
best_t_cal  = float(ths[np.argmax(f1s_cal)])
best_f1_cal = float(max(f1s_cal))
log(f"  Calibrated F1: {best_f1_cal:.4f}@t={best_t_cal:.3f}  (raw: {best_f1_oof:.4f}@t={best_t_oof:.3f})")

# ══════════════════════════════════════════════════════════════════════════════
log("\nSHAP feature importance...")
# ══════════════════════════════════════════════════════════════════════════════

val_sample  = X.sample(min(3000, len(X)), random_state=42)
explainer   = shap.TreeExplainer(models[-1])
sv          = explainer.shap_values(val_sample)
if isinstance(sv, list):
    sv = sv[1]

feat_imp = pd.DataFrame({
    'feature':    feature_cols,
    'shap_mean':  np.abs(sv).mean(axis=0),
    'lgbm_gain':  models[-1].booster_.feature_importance(importance_type='gain'),
}).sort_values('shap_mean', ascending=False).reset_index(drop=True)

log(f"\n  Top 30 features (v2):")
print(feat_imp.head(30).to_string(index=False))

feat_imp.to_parquet(FEATS / "feature_importance_v2.parquet", index=False)
feat_imp.to_csv(FEATS / "feature_importance_v2.csv", index=False)

# SHAP plot
fig, ax = plt.subplots(figsize=(10, 14))
t30 = feat_imp.head(30)
ax.barh(t30['feature'][::-1], t30['shap_mean'][::-1], color='steelblue', alpha=0.85)
ax.set_xlabel('Mean |SHAP|')
ax.set_title(f'Top 30 Features — LightGBM v2  (OOF AUC={oof_auc:.4f})')
plt.tight_layout()
plt.savefig(PLOTS / 'shap_v2.png', dpi=130, bbox_inches='tight')
plt.close()

# PR curve
prec, rec, _ = precision_recall_curve(y, oof_cal)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(rec, prec, color='crimson', lw=2)
axes[0].axhline(y.mean(), color='navy', ls='--', label='Random')
axes[0].set_title(f'PR Curve v2 (AP={oof_ap:.4f})')
axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
axes[0].legend()
axes[1].plot(ths, f1s_cal, color='steelblue', lw=2)
axes[1].axvline(best_t_cal, color='crimson', ls='--', label=f't={best_t_cal:.3f}')
axes[1].set_title(f'F1 vs Threshold — calibrated (Best={best_f1_cal:.4f})')
axes[1].set_xlabel('Threshold'); axes[1].legend()
plt.tight_layout()
plt.savefig(PLOTS / 'pr_curve_v2.png', dpi=130, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
log("\nSaving OOF predictions...")
# ══════════════════════════════════════════════════════════════════════════════

oof_df = train[['account_id']].copy()
oof_df['is_mule']      = y
oof_df['oof_lgbm_v2']  = oof
oof_df['oof_lgbm_v2_cal'] = oof_cal
oof_df.to_parquet(FEATS / "oof_lgbm_v2.parquet", index=False)

# Red herring analysis on v2
rh_v2 = oof_df[(oof_df.is_mule == 1) & (oof_df.oof_lgbm_v2 < 0.05)]
log(f"  Remaining low-confidence mules (prob<0.05): {len(rh_v2):,}")

# ══════════════════════════════════════════════════════════════════════════════
log("\nGenerating submission...")
# ══════════════════════════════════════════════════════════════════════════════

sub = test_ids[['account_id']].copy()
sub['is_mule'] = test_preds_cal   # calibrated probabilities

# Merge suspicious windows from temporal features
sub = sub.merge(temporal_timestamps, on='account_id', how='left')

# For accounts below confidence threshold, clear the window
low_conf = sub['is_mule'] < best_t_cal
sub.loc[low_conf, 'suspicious_start'] = ''
sub.loc[low_conf, 'suspicious_end']   = ''

# Fill any remaining NaNs
sub['suspicious_start'] = sub['suspicious_start'].fillna('')
sub['suspicious_end']   = sub['suspicious_end'].fillna('')

out_path = SUBS / "submission_v2.csv"
sub.to_csv(out_path, index=False)

predicted_mules = (sub['is_mule'] >= best_t_cal).sum()
with_windows    = (sub['suspicious_start'] != '').sum()

log(f"\n  Submission rows    : {len(sub):,}")
log(f"  Predicted mules    : {predicted_mules:,} / {len(sub):,} ({predicted_mules/len(sub):.2%})")
log(f"  With time windows  : {with_windows:,}")
log(f"  Score range        : {sub['is_mule'].min():.4f} – {sub['is_mule'].max():.4f}")
log(f"  Saved → {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
log(f"""
  ┌──────────────────────────────────────────────────────┐
  │  LightGBM v2 Results                                 │
  │  OOF AUC (raw)       : {oof_auc:.4f}                       │
  │  OOF AUC (calibrated): {oof_auc_cal:.4f}                       │
  │  OOF AP              : {oof_ap:.4f}                       │
  │  OOF F1 (calibrated) : {best_f1_cal:.4f} @ t={best_t_cal:.3f}          │
  │  Features            : {len(feature_cols)}                          │
  │  RH downweighted     : {len(rh_accounts)}                         │
  │  Predicted mules     : {predicted_mules:,}                       │
  │  With time windows   : {with_windows:,}                       │
  └──────────────────────────────────────────────────────┘
  Next: python scripts/07_geo_features.py
""")