"""
10_graph_features.py — Counterparty Graph Feature Engineering

Builds a directed money-flow graph from transaction data.
Nodes = account_ids + counterparty_ids
Edges = money flows (account → counterparty for debits,
                     counterparty → account for credits)

Features per account:
  - pagerank_score        : importance in money flow network
  - in_degree             : unique sources sending money in
  - out_degree            : unique destinations money flows to
  - in_out_degree_ratio   : fan-in vs fan-out asymmetry
  - clustering_coeff      : local network density
  - community_id          : Louvain community label
  - community_mule_density: fraction of known mules in same community (train only)
  - neighbor_mule_count   : direct neighbors that are known mules
  - neighbor_mule_ratio   : fraction of neighbors that are known mules
  - avg_neighbor_pagerank : average PageRank of direct neighbors
  - is_bridge_node        : account connects otherwise separate communities

Run: python scripts/10_graph_features.py
Output: cache/features/graph_features.parquet

Note: Builds graph on CPU using networkx. Uses all counterparty edges from
      a sample of transactions (1 in 5 row groups) to keep memory manageable.
      Full graph has ~160k accounts + ~N counterparties.
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import networkx as nx
from pathlib import Path
from datetime import datetime
from scipy.stats import mannwhitneyu
import gc
import warnings
warnings.filterwarnings('ignore')

CACHE    = Path("/home/niranjan/AML/cache")
DATA     = Path("/home/niranjan/AML/data")
FEATS    = CACHE / "features"
FEATS.mkdir(exist_ok=True)

# Sample every Nth row group to keep graph buildable in RAM
# Full 397M rows → too many edges. 1-in-4 sample still gives robust graph.
SAMPLE_EVERY_N = 4
BATCH_RG       = 20

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

# ─────────────────────────────────────────────────────────────────────────────
log("Loading train labels for community mule density...")
labels = pd.read_parquet(DATA / "train_labels.parquet",
                         columns=['account_id', 'is_mule'])
mule_set   = set(labels[labels.is_mule == 1]['account_id'])
legit_set  = set(labels[labels.is_mule == 0]['account_id'])
log(f"  Known mules : {len(mule_set):,}")
log(f"  Known legit : {len(legit_set):,}")

# ─────────────────────────────────────────────────────────────────────────────
log("\nBuilding edge list from transactions (1 in 4 row groups)...")
# ─────────────────────────────────────────────────────────────────────────────

txn_path     = CACHE / "transactions_full.parquet"
pf           = pq.ParquetFile(txn_path)
total_rg     = pq.read_metadata(txn_path).num_row_groups
batch_ranges = [(i, min(i + BATCH_RG, total_rg)) for i in range(0, total_rg, BATCH_RG)]

# Sample batches: take every SAMPLE_EVERY_N-th batch
sampled_batches = batch_ranges[::SAMPLE_EVERY_N]
log(f"  Total batches: {len(batch_ranges)}  |  Sampled: {len(sampled_batches)} (1 in {SAMPLE_EVERY_N})")

edge_chunks = []

for b_idx, (rg_start, rg_end) in enumerate(sampled_batches):
    log(f"  Batch {b_idx+1}/{len(sampled_batches)}  rg {rg_start}:{rg_end}")

    tbl = read_rg_batch(pf, rg_start, rg_end,
                        columns=['account_id', 'counterparty_id', 'txn_type', 'amount'])
    if tbl is None:
        continue

    df = tbl.to_pandas()
    del tbl

    df['abs_amt'] = df['amount'].abs()

    # Credit: counterparty → account (money flows IN to account)
    cred = df[df['txn_type'] == 'C'][['counterparty_id', 'account_id', 'abs_amt']].copy()
    cred.columns = ['src', 'dst', 'weight']

    # Debit: account → counterparty (money flows OUT of account)
    deb  = df[df['txn_type'] == 'D'][['account_id', 'counterparty_id', 'abs_amt']].copy()
    deb.columns = ['src', 'dst', 'weight']

    edges = pd.concat([cred, deb], ignore_index=True)
    edge_chunks.append(edges)

    del df, cred, deb, edges
    gc.collect()

log("  Combining edge list...")
all_edges = pd.concat(edge_chunks, ignore_index=True)
del edge_chunks
gc.collect()

# Aggregate parallel edges (same src→dst, sum weights, count transactions)
log("  Aggregating parallel edges...")
edge_agg = all_edges.groupby(['src', 'dst']).agg(
    total_weight = ('weight', 'sum'),
    edge_count   = ('weight', 'count'),
).reset_index()
del all_edges
gc.collect()

log(f"  Total unique edges: {len(edge_agg):,}")
log(f"  Unique nodes: {pd.concat([edge_agg['src'], edge_agg['dst']]).nunique():,}")

# ─────────────────────────────────────────────────────────────────────────────
log("\nBuilding directed graph (networkx DiGraph)...")
# ─────────────────────────────────────────────────────────────────────────────

G = nx.DiGraph()
for row in edge_agg.itertuples(index=False):
    G.add_edge(row.src, row.dst,
               weight=row.total_weight,
               count=row.edge_count)

del edge_agg
gc.collect()

log(f"  Nodes: {G.number_of_nodes():,}")
log(f"  Edges: {G.number_of_edges():,}")

# ─────────────────────────────────────────────────────────────────────────────
log("\nComputing graph features...")
# ─────────────────────────────────────────────────────────────────────────────

# ── PageRank ──────────────────────────────────────────────────────────────
log("  PageRank (alpha=0.85, max_iter=100)...")
pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6, weight='weight')
log(f"  PageRank computed for {len(pagerank):,} nodes")

# ── In/Out degree ─────────────────────────────────────────────────────────
log("  In/Out degree...")
in_degree  = dict(G.in_degree())
out_degree = dict(G.out_degree())

# Weighted degree (sum of edge weights)
in_strength  = {n: sum(d['weight'] for _, _, d in G.in_edges(n, data=True))
                for n in G.nodes()}
out_strength = {n: sum(d['weight'] for _, _, d in G.out_edges(n, data=True))
                for n in G.nodes()}

# ── Clustering coefficient (undirected view) ──────────────────────────────
log("  Clustering coefficient (undirected)...")
G_undirected = G.to_undirected()
clustering   = nx.clustering(G_undirected)
del G_undirected
gc.collect()

# ── Community detection (Louvain on undirected graph) ─────────────────────
log("  Community detection (Louvain)...")
try:
    import community as community_louvain   # python-louvain
    G_und = G.to_undirected()
    partition = community_louvain.best_partition(G_und, weight='weight', random_state=42)
    del G_und
    gc.collect()
    log(f"  Communities found: {len(set(partition.values())):,}")
except ImportError:
    log("  python-louvain not installed, using connected components as fallback")
    G_und = G.to_undirected()
    partition = {}
    for comp_id, comp in enumerate(nx.connected_components(G_und)):
        for node in comp:
            partition[node] = comp_id
    del G_und
    gc.collect()
    log(f"  Connected components: {len(set(partition.values())):,}")

# ── Neighbor mule density ─────────────────────────────────────────────────
log("  Neighbor mule density...")
neighbor_mule_count = {}
neighbor_mule_ratio = {}

for node in G.nodes():
    # All direct neighbors (in + out)
    neighbors = set(G.predecessors(node)) | set(G.successors(node))
    n_total   = len(neighbors)
    n_mules   = len(neighbors & mule_set)
    neighbor_mule_count[node] = n_mules
    neighbor_mule_ratio[node] = n_mules / n_total if n_total > 0 else 0.0

# ── Community mule density ────────────────────────────────────────────────
log("  Community mule density...")
# For each community, compute fraction of KNOWN labelled nodes that are mules
community_mule_density = {}
community_members      = {}

for node, comm_id in partition.items():
    if comm_id not in community_members:
        community_members[comm_id] = []
    community_members[comm_id].append(node)

for comm_id, members in community_members.items():
    labelled = [m for m in members if m in mule_set or m in legit_set]
    if len(labelled) == 0:
        density = 0.0
    else:
        n_mule_members = sum(1 for m in labelled if m in mule_set)
        density = n_mule_members / len(labelled)
    community_mule_density[comm_id] = density

del community_members
gc.collect()

# ── Average neighbor PageRank ─────────────────────────────────────────────
log("  Average neighbor PageRank...")
avg_neighbor_pr = {}
for node in G.nodes():
    neighbors = list(set(G.predecessors(node)) | set(G.successors(node)))
    if len(neighbors) == 0:
        avg_neighbor_pr[node] = 0.0
    else:
        avg_neighbor_pr[node] = np.mean([pagerank.get(n, 0.0) for n in neighbors])

# ─────────────────────────────────────────────────────────────────────────────
log("\nAssembling feature table for account nodes only...")
# ─────────────────────────────────────────────────────────────────────────────

# Load all account IDs we need to cover
txn_feats    = pd.read_parquet(FEATS / "txn_features.parquet", columns=['account_id'])
all_accounts = set(txn_feats['account_id'].tolist())

rows = []
for acct in all_accounts:
    comm_id = partition.get(acct, -1)
    rows.append({
        'account_id':             acct,
        'pagerank_score':         pagerank.get(acct, 0.0),
        'in_degree':              in_degree.get(acct, 0),
        'out_degree':             out_degree.get(acct, 0),
        'in_strength':            in_strength.get(acct, 0.0),
        'out_strength':           out_strength.get(acct, 0.0),
        'in_out_degree_ratio':    (in_degree.get(acct, 0) /
                                   max(out_degree.get(acct, 1), 1)),
        'in_out_strength_ratio':  (in_strength.get(acct, 0.0) /
                                   max(out_strength.get(acct, 1.0), 1.0)),
        'clustering_coeff':       clustering.get(acct, 0.0),
        'community_id':           comm_id,
        'community_mule_density': community_mule_density.get(comm_id, 0.0),
        'neighbor_mule_count':    neighbor_mule_count.get(acct, 0),
        'neighbor_mule_ratio':    neighbor_mule_ratio.get(acct, 0.0),
        'avg_neighbor_pagerank':  avg_neighbor_pr.get(acct, 0.0),
    })

graph_df = pd.DataFrame(rows)
log(f"  Graph features shape: {graph_df.shape}")

# Derived features
graph_df['pagerank_log']       = np.log1p(graph_df['pagerank_score'] * 1e6)
graph_df['degree_total']       = graph_df['in_degree'] + graph_df['out_degree']
graph_df['strength_total']     = graph_df['in_strength'] + graph_df['out_strength']
graph_df['is_high_mule_comm']  = (graph_df['community_mule_density'] > 0.05).astype(int)

# ── Save ─────────────────────────────────────────────────────────────────
out = FEATS / "graph_features.parquet"
graph_df.to_parquet(out, index=False)
log(f"\n  Saved → {out}  ({out.stat().st_size/1e6:.1f} MB)")

# ══════════════════════════════════════════════════════════════════════════════
log("\n=== Validation: mule vs legit ===")
# ══════════════════════════════════════════════════════════════════════════════

check = graph_df.merge(labels, on='account_id', how='inner')

key_cols = [
    'pagerank_score', 'pagerank_log', 'in_degree', 'out_degree',
    'in_out_degree_ratio', 'in_strength', 'out_strength',
    'clustering_coeff', 'community_mule_density',
    'neighbor_mule_count', 'neighbor_mule_ratio',
    'avg_neighbor_pagerank', 'is_high_mule_comm',
]

print(f"\n  {'Feature':<30s} {'Mule':>12} {'Legit':>12} {'p-value':>10}")
print("  " + "-"*70)
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
    print(f"  {col:<30s} {m.mean():>12.4f} {l.mean():>12.4f} {pval:>10.2e} {sig}")

# Community analysis
high_comm = check[check['is_high_mule_comm'] == 1]
log(f"\n  Accounts in high-mule communities (>5% mule rate):")
log(f"    Total  : {len(high_comm):,}")
log(f"    Mules  : {high_comm['is_mule'].sum():,} ({high_comm['is_mule'].mean():.1%} rate)")

# Missed mules — do they have higher neighbor_mule_ratio?
oof_v3 = pd.read_parquet(FEATS / "oof_lgbm_v3.parquet")
missed = oof_v3[(oof_v3.is_mule == 1) & (oof_v3.oof_lgbm_v3 < 0.335)]
missed_graph = missed.merge(graph_df, on='account_id', how='left')
log(f"\n  Graph stats for previously missed mules ({len(missed):,} accounts):")
for col in ['neighbor_mule_count', 'neighbor_mule_ratio', 'community_mule_density',
            'pagerank_log', 'in_out_degree_ratio']:
    if col in missed_graph.columns:
        log(f"    {col:<30s} mean={missed_graph[col].mean():.4f}  "
            f"median={missed_graph[col].median():.4f}")

log("\nDone. Next: python scripts/11_model_v4.py")
