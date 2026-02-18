#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=cpu_batch,cpu_batch_high_mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --output=logs/eval_evo2_f1-%A_%a.out
#SBATCH --error=logs/eval_evo2_f1-%A_%a.err
#SBATCH --array=0-7
#SBATCH --requeue

# Step 4: Calculate F1 scores for all Evo2 configs.
# Array 0-3 = valid (7b_fwd, 7b_revcomp, 40b_fwd, 40b_revcomp)
# Array 4-7 = test  (7b_fwd, 7b_revcomp, 40b_fwd, 40b_revcomp)
#
# Run after all eval_evo2_compare.sh array jobs complete.
# After this completes, run: sbatch data_scripts/eval_evo2_report.sh

set -e

INTERPLM_DIR="/large_storage/hielab/changdan/mech-interp-mining/flagship_evo2_sae/autointerp/InterPLM"
cd "$INTERPLM_DIR"

ANNOT_DIR="$INTERPLM_DIR/data/annotations/uniprotkb/processed"

CONFIGS=("evo2_7b_fwd" "evo2_7b_revcomp" "evo2_40b_fwd" "evo2_40b_revcomp")

TASK_ID=$SLURM_ARRAY_TASK_ID
CONFIG_IDX=$((TASK_ID % 4))
if [ $TASK_ID -lt 4 ]; then
    EVAL_SET="valid"
else
    EVAL_SET="test"
fi

CONFIG="${CONFIGS[$CONFIG_IDX]}"
RESULTS_DIR="$INTERPLM_DIR/results/${CONFIG}"
COUNTS_DIR="$RESULTS_DIR/${EVAL_SET}_counts"

echo "=== ${CONFIG} ${EVAL_SET} F1 Calculation (task $TASK_ID) ==="

if [ ! -d "$COUNTS_DIR" ]; then
    echo "No counts dir, skipping"
    exit 0
fi

if [ -f "$COUNTS_DIR/concept_f1_scores.csv" ]; then
    echo "Already done, skipping"
    exit 0
fi

python3 -m interplm.analysis.concepts.calculate_f1 \
    --eval_res_dir "$COUNTS_DIR" \
    --eval_set_dir "$ANNOT_DIR/${EVAL_SET}/"

echo "=== Done! ==="
