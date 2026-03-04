"""
Fixed consolidation script.
Writes in batches of N parts → avoids GPU OOM entirely.
Uses pyarrow for the final merge so it never loads everything into RAM at once.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import cudf
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from glob import glob
from pathlib import Path
from tqdm import tqdm
import gc
import shutil

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path("/home/niranjan/AML/data")
CACHE_ROOT  = Path("/home/niranjan/AML/cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE  = 40          # parts per GPU batch — tune down to 20 if OOM again
# ─────────────────────────────────────────────────────────────────────────────

def make_nullable(schema: pa.Schema) -> pa.Schema:
    """Convert all fields to nullable — fixes mismatches across batches."""
    return pa.schema([
        pa.field(f.name, f.type, nullable=True)
        for f in schema
    ])

def consolidate_batched(glob_pattern: str, output_path: str, label: str):
    output_path = Path(output_path)
    if output_path.exists():
        print(f"[{label}] Already exists at {output_path}, skipping.")
        return

    parts = sorted(glob(glob_pattern))
    print(f"\n[{label}] Found {len(parts)} parts → writing in batches of {BATCH_SIZE}")

    # Temporary directory for batch parquet files
    tmp_dir = output_path.parent / f"_tmp_{output_path.stem}"
    tmp_dir.mkdir(exist_ok=True)

    # ── Step 1: Process parts in GPU batches, write each batch to disk ────────
    batch_files = []
    batches = [parts[i:i+BATCH_SIZE] for i in range(0, len(parts), BATCH_SIZE)]

    for b_idx, batch in enumerate(tqdm(batches, desc=f"{label} batches")):
        batch_out = tmp_dir / f"batch_{b_idx:04d}.parquet"

        if batch_out.exists():
            print(f"  batch {b_idx} already written, skipping")
            batch_files.append(str(batch_out))
            continue

        # Load batch on GPU
        chunks = [cudf.read_parquet(p) for p in batch]
        df_gpu = cudf.concat(chunks, ignore_index=True)

        # Write batch back to disk as parquet
        df_gpu.to_parquet(str(batch_out))
        batch_files.append(str(batch_out))

        # Explicitly free GPU memory before next batch
        del chunks, df_gpu
        import gc; gc.collect()

    print(f"\n[{label}] All batches written. Merging {len(batch_files)} files via PyArrow...")

    # ── Step 2: Infer unified schema from first batch (all fields nullable) ───
    first_table = pq.read_table(batch_files[0])
    unified_schema = make_nullable(first_table.schema)
    print(f"  Unified schema ({len(unified_schema)} fields, all nullable)")

    # ── Step 3: Merge batch parquet files on CPU using PyArrow streaming ──────
    writer = pq.ParquetWriter(str(output_path), unified_schema, compression='snappy')

    for bf in tqdm(batch_files, desc=f"{label} merge"):
        table = pq.read_table(bf)
        table = table.cast(unified_schema)   # ← the actual fix
        writer.write_table(table)
        del table

    writer.close()

    # Cleanup temp files
    import shutil
    shutil.rmtree(tmp_dir)

    # Final sanity check
    meta = pq.read_metadata(str(output_path))
    print(f"[{label}] Done → {output_path}")
    print(f"  Rows: {meta.num_rows:,}")
    print(f"  Size: {output_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    consolidate_batched(
        str(DATA_ROOT / "transactions/batch-*/part_*.parquet"),
        str(CACHE_ROOT / "transactions_full.parquet"),
        "TRANSACTIONS"
    )

    consolidate_batched(
        str(DATA_ROOT / "transactions_additional/batch-*/part_*.parquet"),
        str(CACHE_ROOT / "transactions_additional_full.parquet"),
        "TRANSACTIONS_ADDITIONAL"
    )

    print("\n✓ All done. Use cache/ files from now on.")