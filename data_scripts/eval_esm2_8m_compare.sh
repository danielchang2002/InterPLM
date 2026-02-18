#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=cpu_batch,cpu_batch_high_mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=logs/eval_esm2_8m_compare-%A_%a.out
#SBATCH --error=logs/eval_esm2_8m_compare-%A_%a.err
#SBATCH --array=0-15
#SBATCH --requeue

# Step 4: Compare cached SAE features to annotations (CPU, array per shard+eval_set)
# Array 0-7 = valid shards 0-3, Array 8-15 = test shards 4-7
# Run after eval_esm2_8m.sh completes.
# After this completes, run: sbatch data_scripts/eval_esm2_8m_f1.sh

set -e

INTERPLM_DIR="/large_storage/hielab/changdan/mech-interp-mining/flagship_evo2_sae/autointerp/InterPLM"
cd "$INTERPLM_DIR"

MODEL_SHORT="esm2_8m"
LAYER=4

# Map array task to (eval_set, shard)
TASK_ID=$SLURM_ARRAY_TASK_ID
if [ $TASK_ID -lt 8 ]; then
    EVAL_SET="valid"
    SHARD=$TASK_ID
else
    EVAL_SET="test"
    SHARD=$((TASK_ID - 8))
fi

ANNOT_DIR="$INTERPLM_DIR/data/annotations/uniprotkb/processed"
FEATS_DIR="$INTERPLM_DIR/data/sae_features/${MODEL_SHORT}/layer_${LAYER}"
RESULTS_DIR="$INTERPLM_DIR/results/${MODEL_SHORT}_layer_${LAYER}"

echo "=== ESM2-8M Compare — $EVAL_SET shard $SHARD (task $TASK_ID) ==="
echo "Started: $(date)"

python3 -c "
import json
import numpy as np
from scipy import sparse
from pathlib import Path
from tqdm import tqdm
from interplm.analysis.concepts.concept_constants import is_aa_level_concept

annot_dir = Path('$ANNOT_DIR')
feats_dir = Path('$FEATS_DIR')
results_dir = Path('$RESULTS_DIR')
shard = $SHARD
eval_set = '$EVAL_SET'

threshold_percents = [0, 0.15, 0.5, 0.6, 0.8]

eval_dir = annot_dir / eval_set
with open(eval_dir / 'metadata.json') as f:
    meta = json.load(f)

if shard not in meta['shard_source']:
    print(f'Shard {shard} not in {eval_set} set, skipping', flush=True)
    exit(0)

out_file = results_dir / f'{eval_set}_counts' / f'shard_{shard}_counts.npz'
if out_file.exists():
    print(f'{eval_set} shard {shard} already done, skipping', flush=True)
    exit(0)

# Load cached features
print(f'Loading cached features for shard {shard}...', flush=True)
sae_feats = sparse.load_npz(feats_dir / f'shard_{shard}_features.npz').tocsr()
n_features = sae_feats.shape[1]

print(f'Processing {eval_set} shard {shard}...', flush=True)

concept_names = (eval_dir / 'aa_concepts_columns.txt').read_text().splitlines()
is_aa_concept_list = [is_aa_level_concept(c) for c in concept_names]
aa_concept_indices = [i for i, v in enumerate(is_aa_concept_list) if v]
non_aa_concept_indices = [i for i, v in enumerate(is_aa_concept_list) if not v]

per_token_labels = sparse.load_npz(meta['path_to_shards'][str(shard)])
per_token_labels = per_token_labels[:, meta['indices_of_concepts_to_keep']]
per_token_labels = sparse.csc_matrix(per_token_labels)

n_concepts = per_token_labels.shape[1]
n_thresholds = len(threshold_percents)
tp = np.zeros((n_concepts, n_features, n_thresholds))
fp = np.zeros((n_concepts, n_features, n_thresholds))
tp_per_domain = np.zeros((n_concepts, n_features, n_thresholds))

# Precompute binary labels (tokens x concepts), sparse
binary_labels = (per_token_labels > 0).astype(np.float32)

def count_unique_nonzero_vectorized(mat_csc):
    \"\"\"Count unique non-zero values per column, fully vectorized.\"\"\"
    n_cols = mat_csc.shape[1]
    if mat_csc.nnz == 0:
        return np.zeros(n_cols, dtype=np.int64)

    col_indices = np.repeat(np.arange(n_cols, dtype=np.int64), np.diff(mat_csc.indptr))
    data = mat_csc.data

    # Remove explicit zeros
    mask = data != 0
    if not mask.any():
        return np.zeros(n_cols, dtype=np.int64)
    col_indices = col_indices[mask]
    data = data[mask]

    # Sort by (col, val) — col is already sorted from CSC, just need stable sort on val within col
    sort_idx = np.lexsort((data, col_indices))
    sorted_cols = col_indices[sort_idx]
    sorted_vals = data[sort_idx]

    # Find unique (col, val) pairs
    changes = np.empty(len(sorted_cols), dtype=bool)
    changes[0] = True
    changes[1:] = (sorted_cols[1:] != sorted_cols[:-1]) | (sorted_vals[1:] != sorted_vals[:-1])

    # Count per column
    counts = np.zeros(n_cols, dtype=np.int64)
    np.add.at(counts, sorted_cols[changes], 1)
    return counts

for threshold_idx, threshold in enumerate(threshold_percents):
    print(f'  Threshold {threshold} ({threshold_idx+1}/{n_thresholds})...', flush=True)

    # Binarize features
    binarized = sae_feats.copy()
    binarized.data = (binarized.data > threshold).astype(np.float32)
    binarized.eliminate_zeros()

    # TP for all concepts at once via sparse matmul
    tp_matrix = (binarized.T @ binary_labels).toarray()  # (features, concepts)
    tp[:, :, threshold_idx] = tp_matrix.T  # (concepts, features)

    # FP = total_active_per_feature - TP
    total_active = np.asarray(binarized.sum(axis=0)).ravel()  # (features,)
    fp[:, :, threshold_idx] = total_active[np.newaxis, :] - tp[:, :, threshold_idx]

    # tp_per_domain for non-AA concepts
    for ci in tqdm(non_aa_concept_indices, desc=f'tp_per_domain t={threshold}'):
        concept_col = per_token_labels[:, ci]
        product = binarized.multiply(concept_col).tocsc()
        tp_per_domain[ci, :, threshold_idx] = count_unique_nonzero_vectorized(product)

    print(f'    Done.', flush=True)

out_file.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out_file, tp=tp, fp=fp, tp_per_domain=tp_per_domain)
print(f'Saved {out_file}', flush=True)
"

echo "=== $EVAL_SET shard $SHARD complete ==="
echo "Finished: $(date)"
