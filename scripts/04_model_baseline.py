"""
04_model_baseline.py — LightGBM Baseline with SHAP + Submission
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
import warnings
warnings.filterwarnings('ignore')

CACHE = Path("/home/niranjan/AML/cache")
DATA  = Path("/home/niranjan/AML/data")
FEATS = CACHE / "features"
PLOTS = Path("/home/niranjan/AML/notebooks/eda_plots")
SUBS  = Path("/home/niranjan/AML/outputs/submissions")
SUBS.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

log("Loading and merging features...")
static   = pd.read_parquet(FEATS / "master_static.parquet")
txn      = pd.read_parquet(FEATS / "txn_features.parquet")
t_static = pd.read_parquet(FEATS / "test_static.parquet")
labels   = pd.read_parquet(DATA  / "train_labels.parquet")
test_ids = pd.read_parquet(DATA  / "test_accounts.parquet")

train = static.merge(txn, on='account_id', how='left')
test  = t_static.merge(txn, on='account_id', how='left')
log(f"  Train: {train.shape}  |  Test: {test.shape}")

CAT_COLS = [c for c in [
    'account_status','product_family','scheme_code','rural_branch',
    'kyc_compliant','nomination_flag','cheque_allowed','cheque_availed',
    'pan_available','aadhaar_available','passport_available',
    'mobile_banking_flag','internet_banking_flag','atm_card_flag',
    'demat_flag','credit_card_flag','fastag_flag',
    'gender','joint_account_flag','nri_flag','branch_type',
] if c in train.columns]

for col in CAT_COLS:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], ignore_index=True).astype(str).fillna('MISSING')
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str).fillna('MISSING'))
    test[col]  = le.transform(test[col].astype(str).fillna('MISSING'))

DROP = ['account_id','customer_id','is_mule','branch_code',
        'account_status','was_frozen',    # LEAKAGE — frozen because flagged
        'freeze_date','unfreeze_date',    # LEAKAGE — same reason
       ]
       
FEATS_COLS = [c for c in train.columns if c not in DROP]
log(f"  Feature columns: {len(FEATS_COLS)}")

X      = train[FEATS_COLS].fillna(-999)
y      = train['is_mule'].values
X_test = test[FEATS_COLS].fillna(-999)
train.to_parquet(CACHE / "features" / "master_features.parquet", index=False)

neg_count = (y==0).sum(); pos_count = (y==1).sum()
scale_pos = neg_count / pos_count
log(f"  scale_pos_weight={scale_pos:.1f}")

PARAMS = dict(
    objective='binary', metric='auc', boosting_type='gbdt',
    num_leaves=127, learning_rate=0.05, n_estimators=2000,
    scale_pos_weight=scale_pos, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, min_child_samples=20,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42,
    n_jobs=-1, device='gpu', gpu_device_id=0, verbose=-1,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X)); test_preds = np.zeros(len(X_test))
models = []; fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
    log(f"  Fold {fold+1}/5 ...")
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X.iloc[tr_idx], y[tr_idx], eval_set=[(X.iloc[val_idx], y[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
    val_pred = model.predict_proba(X.iloc[val_idx])[:, 1]
    oof[val_idx] = val_pred
    auc = roc_auc_score(y[val_idx], val_pred)
    ap  = average_precision_score(y[val_idx], val_pred)
    ths = np.linspace(0.01,0.99,100)
    f1s = [f1_score(y[val_idx],(val_pred>t).astype(int),zero_division=0) for t in ths]
    best_t = float(ths[np.argmax(f1s)]); best_f1 = float(max(f1s))
    log(f"    AUC={auc:.4f}  AP={ap:.4f}  BestF1={best_f1:.4f}@{best_t:.2f}  iters={model.best_iteration_}")
    fold_scores.append({'fold':fold+1,'auc':auc,'ap':ap,'best_f1':best_f1,'best_t':best_t})
    test_preds += model.predict_proba(X_test)[:, 1] / 5
    models.append(model)

oof_auc = roc_auc_score(y, oof)
oof_ap  = average_precision_score(y, oof)
ths = np.linspace(0.01,0.99,200)
f1s = [f1_score(y,(oof>t).astype(int),zero_division=0) for t in ths]
best_t_oof  = float(ths[np.argmax(f1s)])
best_f1_oof = float(max(f1s))

log(f"\n  OOF AUC={oof_auc:.4f}  AP={oof_ap:.4f}  BestF1={best_f1_oof:.4f}@t={best_t_oof:.3f}")
log(f"\n  Per-fold:\n{pd.DataFrame(fold_scores).to_string(index=False)}")

prec, rec, _ = precision_recall_curve(y, oof)
fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].plot(rec, prec, color='crimson', lw=2)
axes[0].axhline(y.mean(), color='navy', ls='--', label='Random')
axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
axes[0].set_title(f'PR Curve OOF (AP={oof_ap:.4f})'); axes[0].legend()
axes[1].plot(ths, f1s, color='steelblue', lw=2)
axes[1].axvline(best_t_oof, color='crimson', ls='--', label=f't={best_t_oof:.3f}')
axes[1].set_xlabel('Threshold'); axes[1].set_ylabel('F1')
axes[1].set_title(f'F1 vs Threshold (Best={best_f1_oof:.4f})'); axes[1].legend()
plt.tight_layout()
plt.savefig(PLOTS / 'model_pr_curve.png', dpi=130, bbox_inches='tight')
plt.close()

log("SHAP...")
explainer = shap.TreeExplainer(models[-1])
val_sample = X.iloc[val_idx].sample(min(2000, len(val_idx)), random_state=42)
sv = explainer.shap_values(val_sample)
if isinstance(sv, list): sv = sv[1]
feat_imp = pd.DataFrame({
    'feature':   FEATS_COLS,
    'shap_mean': np.abs(sv).mean(axis=0),
    'lgbm_gain': models[-1].booster_.feature_importance(importance_type='gain'),
}).sort_values('shap_mean', ascending=False).reset_index(drop=True)

log(f"\n  Top 30 features:\n{feat_imp.head(30).to_string(index=False)}")

fig, ax = plt.subplots(figsize=(10,14))
t30 = feat_imp.head(30)
ax.barh(t30['feature'][::-1], t30['shap_mean'][::-1], color='steelblue', alpha=0.85)
ax.set_xlabel('Mean |SHAP|'); ax.set_title('Top 30 Features — SHAP')
plt.tight_layout()
plt.savefig(PLOTS / 'shap_importance.png', dpi=130, bbox_inches='tight')
plt.close()
feat_imp.to_parquet(CACHE/"features"/"feature_importance.parquet", index=False)
feat_imp.to_csv(CACHE/"features"/"feature_importance.csv", index=False)

log("Generating submission...")
txn_times = txn[['account_id','first_ts','last_ts']].copy()
txn_times['first_dt'] = pd.to_datetime(txn_times['first_ts'], unit='s', errors='coerce')
txn_times['last_dt']  = pd.to_datetime(txn_times['last_ts'],  unit='s', errors='coerce')
txn_lkp = txn_times.set_index('account_id')
labels['mule_flag_date'] = pd.to_datetime(labels['mule_flag_date'], errors='coerce')
flag_lkp = labels.set_index('account_id')['mule_flag_date'].to_dict()

sub = test_ids[['account_id']].copy()
sub['is_mule'] = test_preds
ss, se = [], []
for row in sub.itertuples():
    if row.is_mule < best_t_oof or row.account_id not in txn_lkp.index:
        ss.append(''); se.append(''); continue
    r = txn_lkp.loc[row.account_id]
    fd = flag_lkp.get(row.account_id)
    if pd.notna(fd):
        s = fd - pd.Timedelta(days=90); e = fd
    else:
        span = max((r['last_dt']-r['first_dt']).days, 0) if pd.notna(r['first_dt']) else 0
        s = r['first_dt'] + pd.Timedelta(days=int(span*0.25))
        e = r['first_dt'] + pd.Timedelta(days=int(span*0.75))
    ss.append(s.isoformat() if pd.notna(s) else '')
    se.append(e.isoformat() if pd.notna(e) else '')

sub['suspicious_start'] = ss
sub['suspicious_end']   = se
out = SUBS / "baseline_lgbm.csv"
sub.to_csv(out, index=False)
log(f"  Saved → {out}")
log(f"  Predicted mules: {(sub.is_mule>best_t_oof).sum():,} / {len(sub):,}")

oof_df = train[['account_id']].copy()
oof_df['oof_lgbm'] = oof; oof_df['is_mule'] = y
oof_df.to_parquet(CACHE/"features"/"oof_lgbm.parquet", index=False)

log("Done.")
print(f"""
  OOF AUC           : {oof_auc:.4f}
  OOF Avg Precision : {oof_ap:.4f}
  OOF Best F1       : {best_f1_oof:.4f}  @  t={best_t_oof:.3f}
  Features used     : {len(FEATS_COLS)}
  Top 5:
{feat_imp.head(5)[['feature','shap_mean']].to_string(index=False)}

  Next: python scripts/05_temporal_features.py
""")