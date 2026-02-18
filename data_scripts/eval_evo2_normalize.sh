#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=cpu_batch,cpu_batch_high_mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --output=logs/eval_evo2_normalize-%j.out
#SBATCH --error=logs/eval_evo2_normalize-%j.err
#SBATCH --requeue

# Step 2.5: Normalize cached Evo2 SAE features by per-feature global max.
# Run after eval_evo2_cache.sh completes, before eval_evo2_compare.sh.
#
# Usage: sbatch data_scripts/eval_evo2_normalize.sh <model> <strand>
#   model:  7b or 40b
#   strand: fwd or revcomp
#
# Example (submit all 4):
#   sbatch data_scripts/eval_evo2_normalize.sh 7b fwd
#   sbatch data_scripts/eval_evo2_normalize.sh 7b revcomp
#   sbatch data_scripts/eval_evo2_normalize.sh 40b fwd
#   sbatch data_scripts/eval_evo2_normalize.sh 40b revcomp

set -e

MODEL="${1:?Usage: eval_evo2_normalize.sh <model: 7b|40b> <strand: fwd|revcomp>}"
STRAND="${2:?Usage: eval_evo2_normalize.sh <model: 7b|40b> <strand: fwd|revcomp>}"

INTERPLM_DIR="/large_storage/hielab/changdan/mech-interp-mining/flagship_evo2_sae/autointerp/InterPLM"
cd "$INTERPLM_DIR"

CONFIG="evo2_${MODEL}_${STRAND}"
FEATURES_DIR="$INTERPLM_DIR/data/sae_features/${CONFIG}"
ANNOT_SHARD_DIR="$INTERPLM_DIR/data/annotations/uniprotkb/processed"

echo "=== Normalize Evo2 ${MODEL} ${STRAND} ==="
echo "Started: $(date)"
echo "Features dir: $FEATURES_DIR"

python3 data_scripts/normalize_evo2_shard_features.py \
    --features_dir "$FEATURES_DIR" \
    --annot_shard_dir "$ANNOT_SHARD_DIR" \
    --strand "$STRAND"

echo "=== Normalization complete ==="
echo "Finished: $(date)"
