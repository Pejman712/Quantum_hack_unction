#!/bin/bash
# Full NQS pipeline on Roihu: supervised curriculum training → VMC fine-tuning on hard circuits.
#
# ── Quickstart ────────────────────────────────────────────────────────────────
#
# Run the complete pipeline (submits two chained jobs automatically):
#   bash hpc/roihu_nqs_pipeline.slurm --submit
#
# Or submit stages manually:
#
#   Stage 1 — supervised training:
#     sbatch hpc/roihu_nqs_pipeline.slurm
#
#   Stage 2 — VMC fine-tuning (after stage 1 finishes):
#     sbatch --export=ALL,STAGE=vmc,SUPERVISED_CKPT=checkpoints/nqs/final.pt \
#         hpc/roihu_nqs_pipeline.slurm
#
#   Resume a timed-out supervised run:
#     sbatch --export=ALL,STAGE=supervised,RESUME_FROM=checkpoints/nqs/epoch_0299.pt \
#         hpc/roihu_nqs_pipeline.slurm
#
#   Resume a timed-out VMC run:
#     sbatch --export=ALL,STAGE=vmc,SUPERVISED_CKPT=checkpoints/nqs_vmc/vmc_epoch_0099.pt \
#         hpc/roihu_nqs_pipeline.slurm
#
# The VMC stage needs quimb (tensor-network amplitude oracle), which isn't in the base
# python-pytorch module. Install it once on a login node before running STAGE=vmc:
#   module load python-pytorch/2.10
#   pip install --user quimb qiskit-quimb
#
# ── Tunables (override with --export=ALL,VAR=value) ──────────────────────────
#
#   STAGE              supervised | vmc         (default: supervised)
#
#   --- Supervised ---
#   SUP_DIFFICULTIES   space-separated tiers    (default: very_easy easy moderate)
#   SUP_N_EPOCHS       total epochs             (default: 500)
#   SUP_LR             learning rate            (default: 1e-3)
#   SUP_D_MODEL        hidden dim               (default: 256)
#   SUP_N_HEADS        attention heads          (default: 8)
#   SUP_N_LAYERS       transformer depth        (default: 6)
#   SUP_D_FF           feed-forward dim         (default: 512)
#   SUP_CKPT_DIR       checkpoint directory     (default: checkpoints/nqs)
#   SUP_CKPT_EVERY     save every N epochs      (default: 50)
#   RESUME_FROM        resume from .pt file     (default: "")
#
#   --- VMC ---
#   VMC_DIFFICULTIES   space-separated tiers    (default: hard)
#   VMC_N_EPOCHS       VMC epochs               (default: 200)
#   VMC_LR             VMC learning rate        (default: 1e-4)
#   VMC_N_SAMPLES       MC samples per step     (default: 64)
#   VMC_CKPT_DIR       VMC checkpoint dir       (default: checkpoints/nqs_vmc)
#   VMC_SAVE_EVERY     save every N VMC epochs  (default: 50)
#   SUPERVISED_CKPT    pre-trained .pt to load  (default: checkpoints/nqs/final.pt)
#
# Docs: https://docs.csc.fi/computing/systems-roihu/  and  https://docs.csc.fi/apps/pytorch/

#SBATCH --job-name=nqs-pipeline
#SBATCH --account=project_2019766
#SBATCH --partition=gpumedium            # partial-node GPU jobs (1 of 4 GH200s); gpularge needs >=4 nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00                  # gpumedium max is 1-12:00:00 (36h)
#SBATCH --gres=gpu:gh200:1               # DefMemPerGPU (~212G) is applied automatically, no --mem needed
#SBATCH --output=logs/nqs_%x_%j.out

# ── --submit mode: runs on the login node, not inside SLURM ──────────────────
# Must come before set -uo pipefail and before any ${SLURM_*} references.
if [[ "${1:-}" == "--submit" ]]; then
    SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/roihu_nqs_pipeline.slurm"

    echo "Submitting supervised job..."
    SUP_JOB=$(sbatch --export=ALL,STAGE=supervised --parsable "$SCRIPT")
    echo "  Supervised job ID: $SUP_JOB"

    echo "Submitting VMC job (starts after supervised succeeds)..."
    VMC_JOB=$(sbatch --dependency=afterok:"$SUP_JOB" \
                     --export=ALL,STAGE=vmc --parsable "$SCRIPT")
    echo "  VMC job ID: $VMC_JOB"

    echo ""
    echo "Monitor:"
    echo "  squeue --jobs $SUP_JOB,$VMC_JOB"
    echo "  tail -f logs/nqs_nqs-pipeline_${SUP_JOB}.out"
    echo "  tail -f logs/nqs_nqs-pipeline_${VMC_JOB}.out"
    exit 0
fi

set -uo pipefail

# ── tunables ──────────────────────────────────────────────────────────────────
STAGE="${STAGE:-supervised}"

# Supervised defaults
SUP_DIFFICULTIES="${SUP_DIFFICULTIES:-very_easy easy moderate}"
SUP_N_EPOCHS="${SUP_N_EPOCHS:-500}"
SUP_LR="${SUP_LR:-1e-3}"
SUP_D_MODEL="${SUP_D_MODEL:-256}"
SUP_N_HEADS="${SUP_N_HEADS:-8}"
SUP_N_LAYERS="${SUP_N_LAYERS:-6}"
SUP_D_FF="${SUP_D_FF:-512}"
SUP_CKPT_DIR="${SUP_CKPT_DIR:-checkpoints/nqs}"
SUP_CKPT_EVERY="${SUP_CKPT_EVERY:-50}"
RESUME_FROM="${RESUME_FROM:-}"

# VMC defaults
VMC_DIFFICULTIES="${VMC_DIFFICULTIES:-hard}"
VMC_N_EPOCHS="${VMC_N_EPOCHS:-200}"
VMC_LR="${VMC_LR:-1e-4}"
VMC_N_SAMPLES="${VMC_N_SAMPLES:-64}"
VMC_CKPT_DIR="${VMC_CKPT_DIR:-checkpoints/nqs_vmc}"
VMC_SAVE_EVERY="${VMC_SAVE_EVERY:-50}"
SUPERVISED_CKPT="${SUPERVISED_CKPT:-checkpoints/nqs/final.pt}"

# ── repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
if [ ! -f "$REPO_ROOT/hpc/roihu_nqs_pipeline.slurm" ]; then
    echo "ERROR: submit from the repo root (the directory containing hpc/)." >&2
    exit 1
fi
cd "$REPO_ROOT"

mkdir -p logs "$SUP_CKPT_DIR" "$VMC_CKPT_DIR"

echo "════════════════════════════════════════"
echo " NQS pipeline — stage: $STAGE"
echo "════════════════════════════════════════"
echo "Job      : $SLURM_JOB_ID"
echo "Host     : $(hostname)"
echo "GPU      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
date

# ── environment ───────────────────────────────────────────────────────────────
# CSC-provided PyTorch module (aarch64 build, cu13, container-backed) — there is no plain
# venv here: the old x86_64/V100 .venv from Puhti cannot run on Roihu's aarch64 GH200 nodes.
module load python-pytorch/2.10
echo "Python   : $(which python)  ($(python --version))"

# ── stage: supervised ─────────────────────────────────────────────────────────
if [ "$STAGE" = "supervised" ]; then
    echo ""
    echo "── Supervised curriculum training ──"
    echo "  Difficulties : $SUP_DIFFICULTIES"
    echo "  Epochs       : $SUP_N_EPOCHS"
    echo "  LR           : $SUP_LR"
    echo "  Model        : d_model=$SUP_D_MODEL  n_heads=$SUP_N_HEADS  n_layers=$SUP_N_LAYERS  d_ff=$SUP_D_FF"
    echo "  Checkpoint   : $SUP_CKPT_DIR  (every $SUP_CKPT_EVERY epochs)"
    [ -n "$RESUME_FROM" ] && echo "  Resume from  : $RESUME_FROM"
    echo ""

    RESUME_ARG=""
    [ -n "$RESUME_FROM" ] && RESUME_ARG="--resume-from $RESUME_FROM"

    # shellcheck disable=SC2086
    python "$REPO_ROOT/exp/transformer-nqs/train.py" \
        --difficulties $SUP_DIFFICULTIES \
        --n-epochs     "$SUP_N_EPOCHS" \
        --lr           "$SUP_LR" \
        --d-model      "$SUP_D_MODEL" \
        --n-heads      "$SUP_N_HEADS" \
        --n-layers     "$SUP_N_LAYERS" \
        --d-ff         "$SUP_D_FF" \
        --device       cuda \
        --checkpoint-dir   "$SUP_CKPT_DIR" \
        --checkpoint-every "$SUP_CKPT_EVERY" \
        --log-every    10 \
        $RESUME_ARG

    echo ""
    echo "Supervised training complete."
    echo "Final checkpoint: $SUP_CKPT_DIR/final.pt"
    date

# ── stage: vmc ────────────────────────────────────────────────────────────────
elif [ "$STAGE" = "vmc" ]; then
    echo ""
    echo "── VMC fine-tuning ──"
    echo "  Difficulties : $VMC_DIFFICULTIES"
    echo "  Epochs       : $VMC_N_EPOCHS"
    echo "  LR           : $VMC_LR"
    echo "  MC samples   : $VMC_N_SAMPLES per circuit"
    echo "  Checkpoint   : $VMC_CKPT_DIR  (every $VMC_SAVE_EVERY epochs)"
    echo "  Loading      : $SUPERVISED_CKPT"
    echo ""

    if [ ! -f "$SUPERVISED_CKPT" ]; then
        echo "ERROR: supervised checkpoint not found: $SUPERVISED_CKPT" >&2
        echo "Run the supervised stage first, or set SUPERVISED_CKPT= to an existing file." >&2
        exit 1
    fi

    if ! python -c "import quimb" 2>/dev/null; then
        echo "ERROR: quimb not installed into the python-pytorch/2.10 user site-packages." >&2
        echo "Install it once from a login node, then resubmit:" >&2
        echo "  module load python-pytorch/2.10 && pip install --user quimb qiskit-quimb" >&2
        exit 1
    fi

    # shellcheck disable=SC2086
    python "$REPO_ROOT/exp/transformer-nqs/vmc_finetune.py" \
        --checkpoint   "$SUPERVISED_CKPT" \
        --difficulties $VMC_DIFFICULTIES \
        --n-epochs     "$VMC_N_EPOCHS" \
        --n-samples    "$VMC_N_SAMPLES" \
        --lr           "$VMC_LR" \
        --device       cuda \
        --out-dir      "$VMC_CKPT_DIR" \
        --save-every   "$VMC_SAVE_EVERY" \
        --log-every    10

    echo ""
    echo "VMC fine-tuning complete."
    echo "Final checkpoint: $VMC_CKPT_DIR/vmc_final.pt"
    date

else
    echo "ERROR: unknown STAGE='$STAGE'. Use 'supervised' or 'vmc'." >&2
    exit 1
fi
