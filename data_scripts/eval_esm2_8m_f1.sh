#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=cpu_batch,cpu_batch_high_mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=logs/eval_esm2_8m_f1-%j.out
#SBATCH --error=logs/eval_esm2_8m_f1-%j.err
#SBATCH --requeue

# Step 6: Calculate F1 for ESM2-8M
# Run after eval_esm2_8m_compare.sh array completes.

set -e

INTERPLM_DIR="/large_storage/hielab/changdan/mech-interp-mining/flagship_evo2_sae/autointerp/InterPLM"
cd "$INTERPLM_DIR"

ANNOT_DIR="$INTERPLM_DIR/data/annotations/uniprotkb/processed"
RESULTS_DIR="$INTERPLM_DIR/results/esm2_8m_layer_4"

echo "=== ESM2-8M F1 Calculation ==="
for EVAL_SET in valid test; do
    echo "--- $EVAL_SET ---"
    python3 -m interplm.analysis.concepts.calculate_f1 \
        --eval_res_dir "$RESULTS_DIR/${EVAL_SET}_counts" \
        --eval_set_dir "$ANNOT_DIR/${EVAL_SET}/"
done

echo "=== Done! Results in $RESULTS_DIR ==="
