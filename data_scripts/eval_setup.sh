#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=cpu_batch_high_mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512GB
#SBATCH --output=logs/eval_setup-%j.out
#SBATCH --error=logs/eval_setup-%j.err
#SBATCH --requeue

# Step 1 of 5: Extract annotations, download SAEs, prepare eval sets
# After this completes, run:
#   sbatch data_scripts/eval_esm2_8m.sh
#   sbatch data_scripts/eval_esm2_650m.sh

set -e
mkdir -p logs

INTERPLM_DIR="/large_storage/hielab/changdan/mech-interp-mining/flagship_evo2_sae/autointerp/InterPLM"
cd "$INTERPLM_DIR"

export INTERPLM_DATA="$INTERPLM_DIR/data"
N_SHARDS=8
ANNOTATIONS="$INTERPLM_DATA/uniprotkb/swissprot_sample_50k_annotations.tsv.gz"
ANNOT_DIR="$INTERPLM_DATA/annotations/uniprotkb/processed"

echo "=== Eval Setup ==="
echo "Node: $(hostname)"
echo "Started: $(date)"

# Extract annotations
echo ""
echo "=== Extract annotations ==="
rm -f "$ANNOT_DIR/uniprotkb_aa_concepts_columns.txt"
python3 -m interplm.analysis.concepts.extract_annotations \
    --input_uniprot_path "$ANNOTATIONS" \
    --output_dir "$ANNOT_DIR" \
    --n_shards $N_SHARDS \
    --min_required_instances 10 \
    --overwrite

# Prepare eval sets
echo ""
echo "=== Prepare eval sets ==="
# Remove old eval sets so they get regenerated with new concept list
rm -f "$ANNOT_DIR/valid/metadata.json" "$ANNOT_DIR/test/metadata.json"
python3 -m interplm.analysis.concepts.prepare_eval_set \
    --valid_shard_range 0 $((N_SHARDS/2 - 1)) \
    --test_shard_range $((N_SHARDS/2)) $((N_SHARDS - 1)) \
    --uniprot_dir "$ANNOT_DIR" \
    --min_aa_per_concept 1500 \
    --min_domains_per_concept 10

# Download SAEs from HuggingFace
echo ""
echo "=== Download SAEs ==="
for MODEL in esm2-8m esm2-650m; do
    if [ "$MODEL" = "esm2-8m" ]; then
        LAYER=4; SHORT="esm2_8m"
    else
        LAYER=24; SHORT="esm2_650m"
    fi
    SAE_DIR="$INTERPLM_DIR/models/${SHORT}/layer_${LAYER}"
    mkdir -p "$SAE_DIR"

    if [ ! -f "$SAE_DIR/ae_normalized.pt" ]; then
        echo "Downloading $MODEL layer $LAYER..."
        python3 -c "
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

weights = hf_hub_download('Elana/InterPLM-${MODEL}', 'layer_${LAYER}/ae_normalized.pt')
shutil.copy(weights, '$SAE_DIR/ae_normalized.pt')
print('Downloaded ae_normalized.pt')

src = Path('interplm/sae/migration/dummy_config_${MODEL}.yaml')
if src.exists():
    shutil.copy(src, '$SAE_DIR/config.yaml')
    print('Copied config.yaml')
"
    else
        echo "$MODEL layer $LAYER already downloaded, skipping"
    fi
done

echo ""
echo "=== Setup complete! ==="
echo "Finished: $(date)"
