#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --partition=cpu_batch,cpu_batch_high_mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=logs/eval_report_metrics-%j.out
#SBATCH --error=logs/eval_report_metrics-%j.err
#SBATCH --requeue

# Step 7: Report metrics for both models
# Run after eval_esm2_8m_f1.sh and eval_esm2_650m_f1.sh complete.

set -e

INTERPLM_DIR="/large_storage/hielab/changdan/mech-interp-mining/flagship_evo2_sae/autointerp/InterPLM"
cd "$INTERPLM_DIR"

echo "=== ESM2-8M Report ==="
python3 -m interplm.analysis.concepts.report_metrics \
    --valid_path results/esm2_8m_layer_4/valid_counts/concept_f1_scores.csv \
    --test_path results/esm2_8m_layer_4/test_counts/concept_f1_scores.csv

echo ""
echo "=== ESM2-650M Report ==="
python3 -m interplm.analysis.concepts.report_metrics \
    --valid_path results/esm2_650m_layer_24/valid_counts/concept_f1_scores.csv \
    --test_path results/esm2_650m_layer_24/test_counts/concept_f1_scores.csv
