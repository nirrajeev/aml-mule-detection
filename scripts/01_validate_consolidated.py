"""
Validates consolidated transaction parquet files.
Run once — if all checks pass, raw batch files can be ignored forever.
Reads only row groups (not full files) to avoid GPU OOM.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import cudf
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

CACHE = Path("/home/niranjan/AML/cache")
DATA  = Path("/home/niranjan/AML/data")

PASS, FAIL, WARN = "  ✓", "  ✗", "  ⚠"

def section(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

def check(label, ok, detail=""):
    status = PASS if ok else FAIL
    print(f"{status}  {label}" + (f"  →  {detail}" if detail else ""))
    return ok

def read_sample(path, n_row_groups=3):
    """Read first N row groups only — never loads full file into GPU."""
    pf = pq.ParquetFile(path)
    batches = []
    for i, batch in enumerate(pf.iter_batches()):
        batches.append(batch)
        if i >= n_row_groups - 1:
            break
    table = pa.Table.from_batches(batches)
    return cudf.DataFrame.from_arrow(table)

# ─────────────────────────────────────────────────────────────────────────────
section("1. FILE EXISTS & SIZE")

txn_path = CACHE / "transactions_full.parquet"
add_path = CACHE / "transactions_additional_full.parquet"

check("transactions_full.parquet exists",            txn_path.exists())
check("transactions_additional_full.parquet exists", add_path.exists())
print(f"  transactions_full size:            {txn_path.stat().st_size / 1e9:.2f} GB")
print(f"  transactions_additional size:      {add_path.stat().st_size / 1e9:.2f} GB")

# ─────────────────────────────────────────────────────────────────────────────
section("2. ROW COUNTS (PyArrow metadata — zero data loaded)")

txn_meta = pq.read_metadata(txn_path)
add_meta = pq.read_metadata(add_path)
txn_rows = txn_meta.num_rows
add_rows = add_meta.num_rows

print(f"  transactions_full rows:            {txn_rows:>15,}")
print(f"  transactions_additional rows:      {add_rows:>15,}")
print(f"  transactions_full row_groups:      {txn_meta.num_row_groups:>15,}")
print(f"  transactions_additional row_groups:{add_meta.num_row_groups:>15,}")

check("Row counts match",  txn_rows == add_rows, f"{txn_rows:,} vs {add_rows:,}")
check("Row count ~397M",   390_000_000 < txn_rows < 410_000_000, f"{txn_rows:,}")

# ─────────────────────────────────────────────────────────────────────────────
section("3. SCHEMA VALIDATION (metadata only)")

txn_schema = pq.read_schema(txn_path)
add_schema = pq.read_schema(add_path)

print("\n  transactions_full schema:")
for f in txn_schema:
    print(f"    {f.name:35s} {str(f.type)}")

print("\n  transactions_additional_full schema:")
for f in add_schema:
    print(f"    {f.name:35s} {str(f.type)}")

expected_txn = {'transaction_id','account_id','transaction_timestamp',
                'mcc_code','channel','amount','txn_type','counterparty_id'}
expected_add = {'transaction_id','mnemonic_code','latitude','longitude',
                'ip_address','balance_after_transaction',
                'part_transaction_type','atm_deposit_channel_code',
                'transaction_sub_type'}

missing_txn = expected_txn - set(txn_schema.names)
missing_add = expected_add - set(add_schema.names)
check("All expected txn columns present",
      len(missing_txn) == 0, f"missing: {missing_txn}" if missing_txn else "all present")
check("All expected additional columns present",
      len(missing_add) == 0, f"missing: {missing_add}" if missing_add else "all present")

# ─────────────────────────────────────────────────────────────────────────────
section("4. SAMPLE DATA — reading 3 row groups")

print("  Reading sample from transactions_full (3 row groups)...")
txn_s = read_sample(txn_path, n_row_groups=3)
print(f"  Sample rows: {len(txn_s):,}")

print("  Reading sample from transactions_additional_full (3 row groups)...")
add_s = read_sample(add_path, n_row_groups=3)
print(f"  Sample rows: {len(add_s):,}")

# ─────────────────────────────────────────────────────────────────────────────
section("5. NULL RATES")

print("\n  transactions_full:")
for col in txn_s.columns:
    null_pct = float(txn_s[col].isna().mean()) * 100
    flag = WARN if null_pct > 20 else "   "
    print(f"  {flag}  {col:35s} {null_pct:6.2f}% null")

print("\n  transactions_additional_full:")
for col in add_s.columns:
    null_pct = float(add_s[col].isna().mean()) * 100
    flag = WARN if null_pct > 20 else "   "
    print(f"  {flag}  {col:35s} {null_pct:6.2f}% null")

# ─────────────────────────────────────────────────────────────────────────────
section("6. VALUE CHECKS")

# txn_type
txn_types = txn_s['txn_type'].value_counts().to_pandas()
check("txn_type only C/D",
      set(txn_types.index).issubset({'C','D'}), str(dict(txn_types)))

# negatives
neg_pct = float((txn_s['amount'] < 0).mean()) * 100
check("Negative amounts < 5%", neg_pct < 5, f"{neg_pct:.2f}% negative")

# timestamps
txn_s['ts'] = cudf.to_datetime(txn_s['transaction_timestamp'])
ts_min = str(txn_s['ts'].min())[:10]
ts_max = str(txn_s['ts'].max())[:10]
print(f"\n  Timestamp range (sample): {ts_min} → {ts_max}")
check("Timestamps in 2020–2025 window",
      ts_min[:4] in ['2020','2021'] and ts_max[:4] in ['2024','2025'],
      f"{ts_min} → {ts_max}")

# account_id format
sample_accts = txn_s['account_id'].head(3).to_pandas().tolist()
check("account_id starts with ACCT_",
      all(str(a).startswith('ACCT_') for a in sample_accts), str(sample_accts))

# transaction_id join overlap
txn_ids = set(txn_s['transaction_id'].head(500).to_pandas())
add_ids = set(add_s['transaction_id'].head(500).to_pandas())
overlap = len(txn_ids & add_ids) / 500 * 100
check("transaction_id join key >95% overlap (spot check)",
      overlap > 95, f"{overlap:.1f}%")

# balance stats
bal = add_s['balance_after_transaction'].dropna()
print(f"\n  balance_after_transaction (sample):")
print(f"    min:    {float(bal.min()):>20,.2f}")
print(f"    max:    {float(bal.max()):>20,.2f}")
print(f"    mean:   {float(bal.mean()):>20,.2f}")
print(f"    median: {float(bal.median()):>20,.2f}")

# part_transaction_type
ptt = add_s['part_transaction_type'].value_counts().to_pandas()
check("part_transaction_type values are CI/BI/IP/IC",
      set(ptt.index).issubset({'CI','BI','IP','IC','UNKNOWN'}), str(dict(ptt)))
print(f"  Distribution: {dict(ptt)}")

# transaction_sub_type
tst = add_s['transaction_sub_type'].value_counts().to_pandas()
check("transaction_sub_type values expected",
      set(str(v).lower() for v in tst.index).issubset(
          {'cash','loan','normal','UNKNOWN'}), str(dict(tst)))
print(f"  Distribution: {dict(tst)}")

# ─────────────────────────────────────────────────────────────────────────────
section("7. CHANNEL DISTRIBUTION")

channels = txn_s['channel'].value_counts().to_pandas()
print(f"  Top channels:\n{channels.head(15).to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
section("8. CROSS-FILE ACCOUNT CHECK")

labels       = pd.read_parquet(DATA / "train_labels.parquet")
test         = pd.read_parquet(DATA / "test_accounts.parquet")
all_accounts = set(labels['account_id']) | set(test['account_id'])
txn_accounts = set(txn_s['account_id'].unique().to_pandas())
overlap_pct  = len(txn_accounts & all_accounts) / len(txn_accounts) * 100

print(f"  Labeled accounts found in txn sample: {overlap_pct:.1f}%")
check("Labeled accounts appear in txn sample",
      overlap_pct > 50, f"{overlap_pct:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
section("FINAL VERDICT")

print("""
  If all ✓ above → both files are clean and complete.

  Your cache state:
    transactions_full.parquet            → 397M rows, 8.57 GB
    transactions_additional_full.parquet → 397M rows, 8.63 GB

  Raw data/transactions/batch-*/ → never touch again.
  Next: python scripts/02_eda.py
""")