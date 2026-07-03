#!/bin/bash
# Train AutoregressiveTransformer NQS on Roihu's GH200 GPUs.
#
# First run (fresh training):
#   sbatch hpc/roihu_nqs_train.slurm
#
# Resume after a timeout (update RESUME_FROM to the latest checkpoint):
#   sbatch --export=ALL,RESUME_FROM=checkpoints/nqs/epoch_0099.pt \
#       hpc/roihu_nqs_train.slurm
#
# Override model size for a quick test:
#   sbatch --export=ALL,N_EPOCHS=50,D_MODEL=64,N_HEADS=2,N_LAYERS=2 \
#       hpc/roihu_nqs_train.slurm
#
# Tunables (--export=ALL,VAR=value on the sbatch line):
#   DIFFICULTIES   space-separated list of tiers   (default: very_easy easy moderate)
#   N_EPOCHS       total training epochs            (default: 500)
#   LR             learning rate                    (default: 1e-3)
#   D_MODEL        transformer hidden dim           (default: 256)
#   N_HEADS        attention heads                  (default: 8)
#   N_LAYERS       transformer depth                (default: 6)
#   D_FF           feed-forward inner dim           (default: 512)
#   DROPOUT        dropout probability              (default: 0.1)
#   CHECKPOINT_DIR checkpoint output directory      (default: checkpoints/nqs)
#   CHECKPOINT_EVERY save every N epochs            (default: 50)
#   RESUME_FROM    path to .pt checkpoint to resume (default: "")
#
# Docs: https://docs.csc.fi/computing/systems-roihu/  and  https://docs.csc.fi/apps/pytorch/

#SBATCH --job-name=nqs-train
#SBATCH --account=project_2019766
#SBATCH --partition=gpumedium            # partial-node GPU jobs (1 of 4 GH200s); gpularge needs >=4 nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00                  # gpumedium max is 1-12:00:00 (36h)
#SBATCH --gres=gpu:gh200:1               # DefMemPerGPU (~212G) is applied automatically, no --mem needed
#SBATCH --output=logs/nqs_train_%j.out

set -uo pipefail

# ── tunables ──────────────────────────────────────────────────────────────────
DIFFICULTIES="${DIFFICULTIES:-very_easy easy moderate}"
N_EPOCHS="${N_EPOCHS:-500}"
LR="${LR:-1e-3}"
D_MODEL="${D_MODEL:-256}"
N_HEADS="${N_HEADS:-8}"
N_LAYERS="${N_LAYERS:-6}"
D_FF="${D_FF:-512}"
DROPOUT="${DROPOUT:-0.1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/nqs}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
RESUME_FROM="${RESUME_FROM:-}"

# ── repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
if [ ! -f "$REPO_ROOT/hpc/roihu_nqs_train.slurm" ]; then
    echo "ERROR: submit from the repo root (directory containing hpc/)." >&2
    echo "       Got SLURM_SUBMIT_DIR=$REPO_ROOT" >&2
    exit 1
fi
cd "$REPO_ROOT"

mkdir -p logs "$CHECKPOINT_DIR"

echo "Job      : $SLURM_JOB_ID"
echo "Host     : $(hostname)"
echo "GPU      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "Epochs   : $N_EPOCHS   LR: $LR"
echo "Model    : d_model=$D_MODEL  n_heads=$N_HEADS  n_layers=$N_LAYERS  d_ff=$D_FF"
echo "Data     : $DIFFICULTIES"
[ -n "$RESUME_FROM" ] && echo "Resume   : $RESUME_FROM"

# ── environment ───────────────────────────────────────────────────────────────
# CSC-provided PyTorch module (aarch64 build, cu13, container-backed) — there is no plain
# venv here: the old x86_64/V100 .venv from Puhti cannot run on Roihu's aarch64 GH200 nodes,
# and this module is the CSC-maintained way to get a working aarch64+CUDA13 torch build.
module load python-pytorch/2.10
echo "Python   : $(which python)  ($(python --version))"
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

# ── run ───────────────────────────────────────────────────────────────────────
TRAIN_SCRIPT="$REPO_ROOT/exp/transformer-nqs/train.py"

RESUME_ARG=""
[ -n "$RESUME_FROM" ] && RESUME_ARG="--resume-from $RESUME_FROM"

# shellcheck disable=SC2086
python "$TRAIN_SCRIPT" \
    --difficulties $DIFFICULTIES \
    --n-epochs    "$N_EPOCHS" \
    --lr          "$LR" \
    --d-model     "$D_MODEL" \
    --n-heads     "$N_HEADS" \
    --n-layers    "$N_LAYERS" \
    --d-ff        "$D_FF" \
    --dropout     "$DROPOUT" \
    --device      cuda \
    --checkpoint-dir   "$CHECKPOINT_DIR" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --log-every   10 \
    $RESUME_ARG
