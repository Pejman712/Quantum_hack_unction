#!/bin/bash
# Run run_unswap.py on Roihu's GH200 GPUs.
#
# Single circuit:
#   sbatch --export=ALL,QASM=qasm_data/very_hard/challenge-48_42.qasm \
#       hpc/roihu_unswap.slurm
#
# Job array over an entire difficulty tier (one task per circuit):
#   sbatch --array=0-N \
#       --export=ALL,DIFFICULTY=very_hard,MAX_BOND=64,MAX_ITS=1024 \
#       hpc/roihu_unswap.slurm
#
# Needs quimb (tensor-network MPO/unswap engine), which isn't in the base python-pytorch
# module. Install it once on a login node before submitting:
#   module load python-pytorch/2.10
#   pip install --user quimb qiskit-quimb
#
# Tunables (--export=ALL,VAR=value on the sbatch line):
#   QASM         explicit path to a single .qasm (takes priority over DIFFICULTY)
#   DIFFICULTY   tier subfolder under qasm_data/  (used with --array)
#   MAX_BOND     MPO max bond dimension            (default 512)
#   MAX_ITS      max unswap iterations             (default 1024)
#   CUTOFF       SVD truncation cutoff             (default 0.002)
#   SAMPLES      MPS sampling shots                (default 10000)
#
# Docs: https://docs.csc.fi/computing/systems-roihu/  and  https://docs.csc.fi/apps/pytorch/

#SBATCH --job-name=unswap
#SBATCH --account=project_2019766
#SBATCH --partition=gpumedium            # partial-node GPU jobs (1 of 4 GH200s); gpularge needs >=4 nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                # quimb uses a few CPU threads alongside GPU
#SBATCH --time=04:00:00                  # raise for very_hard circuits
#SBATCH --gres=gpu:gh200:1               # DefMemPerGPU (~212G) is applied automatically, no --mem needed
#SBATCH --output=slurm-%x-%A_%a.out

set -uo pipefail

# ── tunables ─────────────────────────────────────────────────────────────────
QASM="${QASM:-}"
DIFFICULTY="${DIFFICULTY:-}"
MAX_BOND="${MAX_BOND:-512}"
MAX_ITS="${MAX_ITS:-1024}"
CUTOFF="${CUTOFF:-0.002}"
SAMPLES="${SAMPLES:-10000}"

# ── repo root — SLURM_SUBMIT_DIR is always where sbatch was called ───────────
# Submit from the repo root: cd /scratch/project_2019766/lala/Quantum_hack_unction && sbatch hpc/roihu_unswap.slurm
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
if [ ! -f "$REPO_ROOT/hpc/roihu_unswap.slurm" ]; then
    echo "ERROR: sbatch must be run from the repo root (the directory containing hpc/)." >&2
    echo "       Got SLURM_SUBMIT_DIR=$REPO_ROOT" >&2
    exit 1
fi
cd "$REPO_ROOT"

# ── QASM resolution ──────────────────────────────────────────────────────────
if [ -z "$QASM" ]; then
    if [ -z "$DIFFICULTY" ]; then
        echo "ERROR: set QASM=<path> or DIFFICULTY=<tier> and submit with --array" >&2
        exit 1
    fi
    mapfile -t FILES < <(ls "$REPO_ROOT/qasm_data/${DIFFICULTY}"/*.qasm | sort)
    QASM="${FILES[${SLURM_ARRAY_TASK_ID:-0}]}"
fi
# resolve relative paths against repo root
[[ "$QASM" = /* ]] || QASM="$REPO_ROOT/$QASM"
QASM_ABS="$(realpath "$QASM")"

echo "Circuit : $QASM_ABS"
echo "Params  : max_bond=$MAX_BOND cutoff=$CUTOFF max_its=$MAX_ITS samples=$SAMPLES"
echo "Host    : $(hostname)"
echo "GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# ── environment ──────────────────────────────────────────────────────────────
# CSC-provided PyTorch module (aarch64 build, cu13, container-backed) — there is no plain
# venv here: the old x86_64/V100 .venv from Puhti cannot run on Roihu's aarch64 GH200 nodes.
module load python-pytorch/2.10
echo "Python  : $(which python)  ($(python --version))"

if ! python -c "import quimb" 2>/dev/null; then
    echo "ERROR: quimb not installed into the python-pytorch/2.10 user site-packages." >&2
    echo "Install it once from a login node, then resubmit:" >&2
    echo "  module load python-pytorch/2.10 && pip install --user quimb qiskit-quimb" >&2
    exit 1
fi

# ── run ──────────────────────────────────────────────────────────────────────
SIM_DIR="$REPO_ROOT/exp/peak-kremer/peaked-circuit-simulation"
cd "$SIM_DIR"

SCRIPT="$SIM_DIR/run_unswap.py"
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT not found — did you sync the full repo?" >&2
    ls "$SIM_DIR" >&2
    exit 1
fi

python "$SCRIPT" "$QASM_ABS" \
    --max-bond "$MAX_BOND" \
    --max-its  "$MAX_ITS" \
    --cutoff   "$CUTOFF"  \
    --samples  "$SAMPLES"
