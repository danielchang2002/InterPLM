#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=evo_gpu_priority,gpu_batch,gpu_batch_high_mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --output=logs/eval_esm2_650m-%A_%a.out
#SBATCH --error=logs/eval_esm2_650m-%A_%a.err
#SBATCH --array=0-7
#SBATCH --requeue
#SBATCH --signal=B:TERM@120

# Step 3 of 5: ESM2-650M layer 24 — embed + SAE encode per shard (GPU)
# Run after eval_setup.sh completes.
# After this completes, run: sbatch data_scripts/eval_esm2_650m_compare.sh

set -e
mkdir -p logs

INTERPLM_DIR="/large_storage/hielab/changdan/mech-interp-mining/flagship_evo2_sae/autointerp/InterPLM"
cd "$INTERPLM_DIR"

MODEL_NAME="facebook/esm2_t33_650M_UR50D"
MODEL_SHORT="esm2_650m"
LAYER=24
SHARD=$SLURM_ARRAY_TASK_ID

ANNOT_DIR="$INTERPLM_DIR/data/annotations/uniprotkb/processed"
FEATS_DIR="$INTERPLM_DIR/data/sae_features/${MODEL_SHORT}/layer_${LAYER}"
SAE_DIR="$INTERPLM_DIR/models/${MODEL_SHORT}/layer_${LAYER}"

echo "=== ESM2-650M Layer $LAYER — Shard $SHARD ==="
echo "Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"

FEATS_OUT="$FEATS_DIR/shard_${SHARD}_features.npz"
if [ -f "$FEATS_OUT" ]; then
    echo "Features already cached, skipping"
    exit 0
fi

python3 -c "
import torch
import numpy as np
from scipy import sparse
from pathlib import Path
from interplm.embedders import get_embedder
from interplm.sae.inference import load_sae, get_sae_feats_in_batches
import pandas as pd

shard_file = Path('$ANNOT_DIR/shard_${SHARD}/protein_data.tsv')
df = pd.read_csv(shard_file, sep='\t')
sequences = df['Sequence'].tolist()
print(f'Shard $SHARD: {len(sequences)} sequences', flush=True)

# Step 1: ESM embedding
print('Extracting ESM embeddings...', flush=True)
embedder = get_embedder('esm', model_name='$MODEL_NAME')
embeddings_dict = embedder.extract_embeddings_with_boundaries(sequences, layer=$LAYER, batch_size=8)
aa_embds = embeddings_dict['embeddings']
print(f'Embeddings shape: {aa_embds.shape}', flush=True)

# Step 2: SAE encoding in token chunks (to avoid GPU OOM)
print('Encoding with SAE...', flush=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sae = load_sae(model_dir=Path('$SAE_DIR'), model_name='ae_normalized.pt', device=device)

n_tokens = aa_embds.shape[0]
chunk_size = 4096
chunks = []
for i in range(0, n_tokens, chunk_size):
    batch = aa_embds[i:i+chunk_size].to(device)
    with torch.no_grad():
        feats = sae.encode(batch)
    chunks.append(sparse.csr_matrix(feats.cpu().numpy()))
    if (i // chunk_size) % 50 == 0:
        print(f'  Encoded {min(i+chunk_size, n_tokens):,}/{n_tokens:,} tokens', flush=True)

sae_sparse = sparse.vstack(chunks)
print(f'SAE features shape: {sae_sparse.shape}', flush=True)

# Save
out_dir = Path('$FEATS_DIR')
out_dir.mkdir(parents=True, exist_ok=True)
sparse.save_npz('$FEATS_OUT', sae_sparse)

# Also save boundaries for potential per-protein analysis
torch.save(embeddings_dict['boundaries'], out_dir / 'shard_${SHARD}_boundaries.pt')

print(f'Saved features ({sae_sparse.nnz:,} nonzero / {sae_sparse.shape[0] * sae_sparse.shape[1]:,} total)')
print(f'Sparsity: {1 - sae_sparse.nnz / (sae_sparse.shape[0] * sae_sparse.shape[1]):.4f}')
"

echo ""
echo "=== Shard $SHARD complete ==="
echo "Finished: $(date)"
