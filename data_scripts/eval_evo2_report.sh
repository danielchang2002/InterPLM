#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --partition=cpu_batch,cpu_batch_high_mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=logs/eval_evo2_report-%j.out
#SBATCH --error=logs/eval_evo2_report-%j.err
#SBATCH --requeue

# Step 5: Report metrics for all Evo2 configs.
# Run after eval_evo2_f1.sh completes.

set -e

INTERPLM_DIR="/large_storage/hielab/changdan/mech-interp-mining/flagship_evo2_sae/autointerp/InterPLM"
cd "$INTERPLM_DIR"

CONFIGS=("evo2_7b_fwd" "evo2_7b_revcomp" "evo2_40b_fwd" "evo2_40b_revcomp")

for CONFIG in "${CONFIGS[@]}"; do
    RESULTS_DIR="$INTERPLM_DIR/results/${CONFIG}"
    VALID_PATH="$RESULTS_DIR/valid_counts/concept_f1_scores.csv"
    TEST_PATH="$RESULTS_DIR/test_counts/concept_f1_scores.csv"

    if [ ! -f "$VALID_PATH" ] || [ ! -f "$TEST_PATH" ]; then
        echo "Skipping ${CONFIG} — missing F1 scores (valid: $VALID_PATH, test: $TEST_PATH)"
        continue
    fi

    echo "=== ${CONFIG} Report ==="
    python3 -m interplm.analysis.concepts.report_metrics \
        --valid_path "$VALID_PATH" \
        --test_path "$TEST_PATH"
    echo ""
done

echo "=== All reports done! ==="
