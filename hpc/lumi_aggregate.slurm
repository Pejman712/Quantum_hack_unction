#!/bin/bash
# Aggregate the per-circuit JSON checkpoints written by lumi_batch.slurm into the result CSVs.
# Run this once after all array shards have finished. It appends *pending* candidate rows to
# results/<difficulty>_bitstrings.csv without touching rows already marked success/failed, and
# does NOT re-solve anything (--aggregate-only).
#
# This is tiny and needs no GPU. You can run it as a job (below) or directly on a login node:
#   cd <repo-root>
#   PYTHONPATH=$PWD/src /appl/local/quantum/qiskit/run-singularity \
#       /appl/local/quantum/qiskit/qiskit_2.3.0_csc.sif \
#       python -m quantum_hack.batch --aggregate-only
#
# Usage as a batch job:  sbatch hpc/lumi_aggregate.slurm

#SBATCH --job-name=peak-aggregate
#SBATCH --account=project_465XXXXXX     # <-- set your LUMI project (or pass --account on sbatch)
#SBATCH --partition=small               # LUMI-C (CPU); no GPU needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail

export LUMI_QISKIT_SINGULARITY_CONTAINER_PATH=${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH:-/appl/local/quantum/qiskit/qiskit_2.3.0_csc.sif}
export WRAPPER_PATH=${WRAPPER_PATH:-/appl/local/quantum/qiskit/run-singularity}
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

srun "$WRAPPER_PATH" "$LUMI_QISKIT_SINGULARITY_CONTAINER_PATH" \
    python -m quantum_hack.batch --aggregate-only
