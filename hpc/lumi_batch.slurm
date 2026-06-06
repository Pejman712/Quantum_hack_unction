#!/bin/bash
# Solve the peaked-circuit challenges on LUMI's AMD GPUs using CSC's Qiskit container.
#
# Submit as a job array (recommended): one task per shard, each task solving every Nth circuit on
# one GPU and writing a per-circuit JSON checkpoint under results/_peak_cache/ (safe to write
# concurrently). Re-running is idempotent: finished circuits are skipped, and circuits that hit the
# per-circuit time budget are recorded as timeouts (not cached) so they are retried next run. After
# all tasks finish, run lumi_aggregate.slurm once to append pending rows to the result CSVs.
#
# Tunables (override on the sbatch command line with --export=ALL,NAME=VALUE,...):
#   DIFFICULTY   tier to solve (also accepted as the first positional arg); empty = all tiers
#   SHOTS        MPS sampling shots             (default 16384)
#   BOND_DIM     MPS max bond dimension         (default 256)
#   TIME_BUDGET  per-circuit seconds, 0 = none  (default 3600)
#
# Example — moderate tier (8 circuits), 16328 shots, bond 256, 1-hour per-circuit budget:
#   sbatch --array=0-7 --account=project_465XXXXXX \
#       --export=ALL,DIFFICULTY=moderate,SHOTS=16328,BOND_DIM=256,TIME_BUDGET=3600 \
#       hpc/lumi_batch.slurm
#
# Docs: https://docs.csc.fi/apps/qiskit/  and  https://docs.lumi-supercomputer.eu/

#SBATCH --job-name=peak-batch
#SBATCH --account=project_465XXXXXX     # <-- set your LUMI project (or pass --account on sbatch)
#SBATCH --partition=small-g             # partial-node GPU jobs (1 GCD); standard-g bills whole nodes
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:15:00                 # wall clock PER array task; must exceed TIME_BUDGET plus
                                        # container/import overhead. Raise it if a task does >1 circuit.
#SBATCH --output=slurm-%x-%A_%a.out

set -uo pipefail

# CSC-provided Qiskit container (qiskit-aer-gpu-rocm for GPU Aer on MI250X) and its run wrapper.
export LUMI_QISKIT_SINGULARITY_CONTAINER_PATH=${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH:-/appl/local/quantum/qiskit/qiskit_2.3.0_csc.sif}
export WRAPPER_PATH=${WRAPPER_PATH:-/appl/local/quantum/qiskit/run-singularity}

# Run from the submit directory (the repo root) so qasm_data/ and results/ resolve, and expose the
# package on PYTHONPATH inside the container. If import fails, install once with the container's pip:
#   $WRAPPER_PATH $LUMI_QISKIT_SINGULARITY_CONTAINER_PATH pip install --user --no-deps -e .
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DIFFICULTY="${1:-${DIFFICULTY:-}}"
SHOTS="${SHOTS:-16384}"
BOND_DIM="${BOND_DIM:-256}"
TIME_BUDGET="${TIME_BUDGET:-3600}"

DIFFICULTY_ARG=""
if [ -n "$DIFFICULTY" ]; then
    DIFFICULTY_ARG="--difficulty $DIFFICULTY"
fi

# Shard across the array when submitted with --array; otherwise process the whole set in one task.
SHARD_ARG=""
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]; then
    SHARD_ARG="--shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
fi

echo "Solving ${DIFFICULTY:-all-tiers} ${SHARD_ARG} shots=${SHOTS} bond=${BOND_DIM} budget=${TIME_BUDGET}s on $(hostname)"

# The tensor-network oracle (quimb) is optional and only used for the largest circuits; install it
# once with: $WRAPPER_PATH $LUMI_QISKIT_SINGULARITY_CONTAINER_PATH pip install --user quimb cotengra
srun "$WRAPPER_PATH" "$LUMI_QISKIT_SINGULARITY_CONTAINER_PATH" \
    python -m quantum_hack.batch \
        $DIFFICULTY_ARG $SHARD_ARG \
        --gpu --device GPU \
        --shots "$SHOTS" --bond-dim "$BOND_DIM" \
        --time-budget "$TIME_BUDGET"

# When every array task has finished, aggregate the JSON cache into the result CSVs (run once):
#   sbatch hpc/lumi_aggregate.slurm
