#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=cpu_batch,cpu_batch_high_mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=logs/eval_evo2_cache-%A_%a.out
#SBATCH --error=logs/eval_evo2_cache-%A_%a.err
#SBATCH --array=0-7
#SBATCH --requeue

# Step 2: Cache Evo2 SAE features (codon-pooled to AA resolution) per shard.
# Usage: sbatch data_scripts/eval_evo2_cache.sh <model> <strand>
#   model:  7b or 40b
#   strand: fwd or revcomp
#
# Example:
#   sbatch data_scripts/eval_evo2_cache.sh 7b fwd
#   sbatch data_scripts/eval_evo2_cache.sh 7b revcomp
#   sbatch data_scripts/eval_evo2_cache.sh 40b fwd
#   sbatch data_scripts/eval_evo2_cache.sh 40b revcomp
#
# After all 4 complete, run: sbatch data_scripts/eval_evo2_compare.sh <model> <strand>

set -e

MODEL="${1:?Usage: eval_evo2_cache.sh <model: 7b|40b> <strand: fwd|revcomp>}"
STRAND="${2:?Usage: eval_evo2_cache.sh <model: 7b|40b> <strand: fwd|revcomp>}"

INTERPLM_DIR="/large_storage/hielab/changdan/mech-interp-mining/flagship_evo2_sae/autointerp/InterPLM"
cd "$INTERPLM_DIR"

FEATURES_BASE="/large_storage/hielab/changdan/features"

# Set model-specific paths
if [ "$MODEL" = "7b" ]; then
    N_FEATURES=65536
    SAE_DIR="evo2_7b_opengenome2_mini_layer_26_embeddings_shuffled-matryoshka_batch_top_k-64-65536"
elif [ "$MODEL" = "40b" ]; then
    N_FEATURES=131072
    SAE_DIR="evo2_40b_opengenome2_mini_layer_18_embeddings_shuffled-matryoshka_batch_top_k-64-131072"
else
    echo "ERROR: Invalid model '$MODEL' (expected 7b or 40b)"
    exit 1
fi

if [ "$STRAND" = "fwd" ]; then
    FEATURE_DIR="$FEATURES_BASE/$SAE_DIR/swissprot_cds_sample_50k"
elif [ "$STRAND" = "revcomp" ]; then
    FEATURE_DIR="$FEATURES_BASE/$SAE_DIR/swissprot_cds_sample_50k_revcomp"
else
    echo "ERROR: Invalid strand '$STRAND' (expected fwd or revcomp)"
    exit 1
fi

MAPPING_TSV="$INTERPLM_DIR/data/uniprot/swissprot_cds_sample_50k_mapping.tsv"
ANNOT_SHARD_DIR="$INTERPLM_DIR/data/annotations/uniprotkb/processed"
OUTPUT_DIR="$INTERPLM_DIR/data/sae_features/evo2_${MODEL}_${STRAND}"
SHARD=$SLURM_ARRAY_TASK_ID

echo "=== Evo2 ${MODEL} ${STRAND} — Cache shard ${SHARD} ==="
echo "Started: $(date)"
echo "Feature dir: $FEATURE_DIR"
echo "Output dir:  $OUTPUT_DIR"

python3 data_scripts/cache_evo2_shard_features.py \
    --feature_dir "$FEATURE_DIR" \
    --mapping_tsv "$MAPPING_TSV" \
    --annot_shard_dir "$ANNOT_SHARD_DIR" \
    --shard "$SHARD" \
    --output_dir "$OUTPUT_DIR" \
    --strand "$STRAND" \
    --n_features "$N_FEATURES"

echo "=== Shard ${SHARD} complete ==="
echo "Finished: $(date)"
