#!/bin/bash
#SBATCH --time=14-00:00:00
#SBATCH --partition=cpu_preemptible
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --array=0-199
#SBATCH --output=/large_storage/hielab/changdan/mech-interp-mining/flagship_evo2_sae/autointerp/InterPLM/data_scripts/logs/download_cds-%A_%a.out
#SBATCH --error=/large_storage/hielab/changdan/mech-interp-mining/flagship_evo2_sae/autointerp/InterPLM/data_scripts/logs/download_cds-%A_%a.err
#SBATCH --requeue
#SBATCH --signal=B:TERM@120

SCRIPT_DIR="/large_storage/hielab/changdan/mech-interp-mining/flagship_evo2_sae/autointerp/InterPLM/data_scripts"

# Set your NCBI API key (10 req/sec with key, 3/sec without)
export NCBI_API_KEY="${NCBI_API_KEY:?Set NCBI_API_KEY environment variable}"

NUM_SHARDS=200

echo "Shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} starting at $(date)"
echo "Node: $(hostname)"

python3 "${SCRIPT_DIR}/download_swissprot_cds.py" \
    --shard-id "${SLURM_ARRAY_TASK_ID}" \
    --num-shards "${NUM_SHARDS}"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "Shard ${SLURM_ARRAY_TASK_ID} completed successfully at $(date)"
elif [ $exit_code -eq 143 ]; then
    echo "Shard ${SLURM_ARRAY_TASK_ID} interrupted (preempted) at $(date), will resume on requeue"
else
    echo "Shard ${SLURM_ARRAY_TASK_ID} failed with exit code ${exit_code} at $(date)"
fi

exit $exit_code
