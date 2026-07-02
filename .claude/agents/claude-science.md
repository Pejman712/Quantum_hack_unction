---
name: claude-science
description: Science agent for the Quantum_hack_unction / qmill challenge. Deep understanding of the transformer-nqs codebase, the circuit simulation pipeline, and the physics of peaked quantum circuits.
---

# Project: Quantum_hack_unction — qmill Challenge

## Challenge Goal

Find the **highest-probability bitstring** x* for each quantum circuit |ψ⟩, i.e. the computational basis state x that maximises |⟨x|ψ⟩|². Circuits range from 8 to 104 qubits across five difficulty tiers. The submission is a bitstring per circuit; evaluation is by match against the graded ground truth.

---

## Repository Layout

```
Quantum_hack_unction/
├── qasm_data/              # OpenQASM 2.0 circuit files
│   ├── very_easy/          # 10 circuits, 8–64 qubits
│   ├── easy/               # 6 circuits, 16–64 qubits
│   ├── moderate/           # 3 circuits, 8–64 qubits
│   ├── hard/               # 7 circuits, 40–64 qubits
│   └── very_hard/          # 8 circuits, 48–104 qubits
├── results/                # Ground truth CSVs (columns: challenge, qubits, method, bitstring, probability, status)
│   ├── very_easy_bitstrings.csv    # 10/10 solved (statevector + MPS)
│   ├── easy_bitstrings.csv         # 16/16 solved
│   ├── moderate_bitstrings.csv     # 8/9 solved (one failed row kept)
│   └── hard_bitstrings.csv         # 2/7 solved (marginal method, low prob)
├── exp/
│   ├── transformer-nqs/    # Neural Quantum State transformer (main work)
│   └── peak-kremer/        # MPO/unswap simulation baseline
├── hpc/                    # Puhti SLURM job scripts
└── scripts/                # Utility scripts (check_slurm_results.py)
```

---

## Dataset Statistics

| Difficulty | Circuits | Qubits  | Solved | Peak prob range   | Method used         |
|------------|----------|---------|--------|-------------------|---------------------|
| very_easy  | 10       | 8–64    | 10/10  | 0.08 – 0.91       | statevector / MPS   |
| easy       | 16       | 16–64   | 16/16  | 0.00 – 0.47       | statevector / MPS   |
| moderate   | 9        | 8–64    | 8/9    | 0.001 – 0.40      | statevector / MPS   |
| hard       | 7        | 40–64   | 2/7    | 0.0007 – 0.003    | marginal            |
| very_hard  | 8        | 48–104  | 0/8    | unknown           | —                   |

`load_dataset()` loads only rows with `status == "success"` and deduplicates by challenge (last success wins), giving **36 labeled samples** total.

**Bitstring convention**: CSV and model both use Qiskit little-endian — character i is the state of qubit q[i]. quimb uses big-endian; always pass `bitstring[::-1]` to `quimb_amplitude()`.

---

## Circuit Structure

All circuits use exactly four gate types: `rx`, `rz` (single-qubit rotations), `cx` (CNOT), `swap`. Each QASM file registers n qubits and lists gates in application order. A typical 8-qubit circuit has ~46 gates (17 rx, 14 rz, 14 cx, 1 swap).

The circuits are **peaked**: one or a few bitstrings carry almost all the probability amplitude. This is the key property that makes both MPS simulation and NQS learning tractable — the model only needs to find one dominant mode.

---

## Approach 1: MPO/Unswap Simulation

**Location**: `exp/peak-kremer/peaked-circuit-simulation/`

Compresses the quantum circuit as a Matrix Product Operator (MPO) by inserting SWAP gates to reduce long-range entanglement. The unswap algorithm searches for an optimal qubit reordering that minimises bond dimension χ, then samples from the resulting MPS.

Key parameters:
- `--max-bond` (default 512): maximum MPO bond dimension χ. SVD truncation cost is O(χ³). Hard circuits often need χ > 1024.
- `--cutoff` (default 0.002): SVD singular value truncation threshold.
- Runs on V100 GPU using complex64 (FP32, ~14 TFLOPS) instead of complex128.

**Qubit ordering bug**: quimb's `simulate_qasm` returns amplitudes in big-endian order; the CSV stores bitstrings in little-endian. The reversal bug causes all circuits to fail the `check_slurm_results.py` validation — tracked but not yet fixed in `unswap.py`.

Run via: `sbatch hpc/puhti_unswap.slurm` (job array over a difficulty tier).

---

## Approach 2: Autoregressive Transformer NQS

**Location**: `exp/transformer-nqs/`  
**Python**: 3.12 via `exp/peak-kremer/peaked-circuit-simulation/.venv` (torch 2.10.0+cu128)  
**Run tests**: `exp/peak-kremer/peaked-circuit-simulation/.venv/bin/python -m pytest exp/transformer-nqs/tests/ -v`  
**150 tests, all passing.**

### Core idea

Represent the peaked distribution as an autoregressive product:

```
p_θ(x₁,...,xₙ) = ∏ᵢ p_θ(xᵢ | x₀,...,xᵢ₋₁, circuit)
```

A decoder-only transformer models this. The circuit's qubit-coupling graph is injected as an additive attention bias so the model has a structural prior before any gradient updates.

### Model: `model/transformer.py` — `AutoregressiveTransformer`

```
AutoregressiveTransformer(
    d_model=256, n_heads=8, n_layers=6, d_ff=512,
    max_qubits=128, dropout=0.1
)  →  3,190,793 parameters
```

**Token vocabulary**: `{0: bit-0, 1: bit-1, 2: BOS}`. BOS is prepended to create the shifted input `[BOS, x₀, ..., x_{n-2}]`.

**Forward pass**:
1. Embed tokens + learned positional embeddings
2. Apply causal mask (upper-triangular bool, prevents position i seeing j > i)
3. At each layer, add `conditioner.get_layer_bias(normalised_interaction, layer_idx)` to attention scores before softmax
4. Final linear head → sigmoid → per-qubit probabilities p(xᵢ=1|...)

**API (backward compatible)**:
```python
# Legacy (Sprint 1-2): shared interaction matrix
probs = model(tokens, interaction_matrix)          # (B, n)

# Typed (Sprint 3+): separate cx and swap matrices
probs = model(tokens, cx_matrix=cx_t, swap_matrix=swap_t)
```

### Circuit Conditioning: `model/conditioning.py` — `CircuitConditioner`

Each transformer layer gets its own independently learned scale of the circuit interaction bias:

```python
conditioner.layer_scales    # nn.Parameter, shape (n_layers,), init=0
conditioner.cx_weight       # nn.Parameter, scalar, init=1.0
conditioner.swap_weight     # nn.Parameter, scalar, init=1.0
```

`combine(cx_matrix, swap_matrix)` → `softplus(cx_weight)*cx + softplus(swap_weight)*swap`

`normalize_interaction(m)` → `log1p(m) / max(log1p(m))` (maps raw gate counts to [0,1])

`get_layer_bias(norm_bias, layer_idx)` → `layer_scales[i] * norm_bias`

Layer scales start at 0 so the model begins as a plain transformer and learns circuit structure through gradient descent.

### Data Pipeline: `data/`

**`circuit_encoder.py`**:
```python
encode_circuit(qasm_path)
# → (n_qubits, gate_sequence, interaction_matrix)

encode_circuit_typed(qasm_path)
# → (n_qubits, gate_sequence, cx_matrix, swap_matrix, interaction_matrix)
```
`gate_sequence`: list of `[type_id, cos(θ), sin(θ), q0, q1]`; q1=n_qubits for 1-qubit gates.
`interaction_matrix[i,j]` = number of 2-qubit gates between qubits i and j (symmetric, diagonal=0).

**`dataset.py`** — `CircuitSample` dataclass fields:
`challenge, difficulty, n_qubits, bitstring, tokens, probability, gate_sequence, interaction_matrix (ndarray), qasm_path`

```python
samples = load_dataset(difficulties=("very_easy", "easy", "moderate"))
# returns list[CircuitSample], success rows only, deduplicated
```

**`tokenizer.py`**: `bitstring_to_tokens("10101101") → [1,0,1,0,1,1,0,1]`

### Sampler: `model/sampler.py`

```python
bitstrings = sample(model, interaction_matrix, n_qubits=8, n_samples=200)
# → LongTensor (n_samples, n_qubits), values in {0,1}

bitstrings, log_probs = sample(..., return_log_probs=True)
# log_probs: FloatTensor (n_samples,) ≤ 0
```
Autoregressive left-to-right sampling, runs under `@torch.no_grad()`.

### Training: `training/`

#### Supervised BCE (`training/supervised.py`)

```python
loss = compute_loss(model, tokens, interaction_matrix)
# tokens: LongTensor (1, n), interaction: FloatTensor (n, n)
# BCE between model probs and target bits

history = train(
    model, samples, n_epochs=500,
    lr=1e-3, device="cuda", scheduler=curriculum_scheduler,
    start_epoch=0,            # for resume
    optimizer=opt,            # pass restored optimizer for resume
    checkpoint_dir="checkpoints/nqs",
    checkpoint_every=50,
    config=model_config_dict,
    existing_history=history, # for resume
)
```

Each sample is processed with batch_size=1 because circuits have different n_qubits.

#### Curriculum Scheduler (`training/curriculum.py`)

```python
scheduler = CurriculumScheduler([
    ("very_easy", ve_samples, 167),  # epochs 0–166
    ("easy",      ea_samples, 167),  # epochs 167–333 (cumulative: ve + ea)
    ("moderate",  mo_samples, 166),  # epochs 334–499 (cumulative: all three)
])
scheduler.get_samples(epoch)   # returns cumulative sample list for this epoch
scheduler.get_stage(epoch)     # returns stage name string
```

#### Checkpointing (`training/checkpointing.py`)

```python
save_checkpoint(path, model, optimizer, epoch, history, config)

epoch, history, config = load_checkpoint(path, model, optimizer)
# Restores weights + optimizer momentum buffers in-place
# Returns the last completed epoch (0-indexed)
```

Checkpoint `.pt` keys: `epoch, model_state, optimizer_state, history{epochs,losses}, config`.

#### VMC Fine-tuning (`training/vmc.py`, `training/amplitude.py`)

For hard/very_hard circuits where no ground truth exists. Uses REINFORCE:

```
∇E_{x~p_θ}[log|⟨x|ψ⟩|²] = E_x[(log|A(x)|² − baseline) · ∇log p_θ(x)]
```

```python
log_p = compute_log_prob(model, tokens, interaction)  # (B,) WITH gradient
loss  = reinforce_loss(log_p, rewards, use_baseline=True)

mean_reward, loss_val = vmc_step(
    model, interaction_matrix, qasm_path, amplitude_fn,
    n_qubits=48, n_samples=64, optimizer=opt, device="cuda"
)

history = vmc_train(
    model, hard_samples, quimb_amplitude,
    n_epochs=200, n_samples=64, lr=1e-4, device="cuda"
)
```

**Amplitude evaluators** (`training/amplitude.py`):
```python
quimb_amplitude(qasm_path, bitstring)   # → log|A(x)|² via tensor network
# Uses simplify_sequence='DC' to avoid quimb ZeroDivisionError bug
# Internally reverses bitstring for quimb big-endian convention
# Returns float("-inf") for zero-amplitude or errors

make_peaked_mock(target, peak_prob=0.9) # → amplitude_fn for tests
# Hamming-distance decay: reward drops as exp(-3*hamming/n)
```

---

## Training Scripts (CLI)

### Supervised training
```bash
python exp/transformer-nqs/train.py \
    --difficulties very_easy easy moderate \
    --n-epochs 500 --lr 1e-3 \
    --d-model 256 --n-heads 8 --n-layers 6 --d-ff 512 \
    --device cuda \
    --checkpoint-dir checkpoints/nqs --checkpoint-every 50

# Resume after timeout:
python exp/transformer-nqs/train.py \
    --resume-from checkpoints/nqs/epoch_0299.pt \
    --n-epochs 500 --checkpoint-dir checkpoints/nqs
```

### VMC fine-tuning
```bash
python exp/transformer-nqs/vmc_finetune.py \
    --checkpoint checkpoints/nqs/final.pt \
    --difficulties hard \
    --n-epochs 200 --n-samples 64 --lr 1e-4 \
    --device cuda --out-dir checkpoints/nqs_vmc

# Resume VMC:
python exp/transformer-nqs/vmc_finetune.py \
    --checkpoint checkpoints/nqs_vmc/vmc_epoch_0099.pt \
    --difficulties hard --n-epochs 200 --out-dir checkpoints/nqs_vmc
```

---

## Puhti HPC

**Account**: `project_2019510`  
**GPU**: V100 (16 GB, complex64 preferred — 14 TFLOPS vs 7 TFLOPS for complex128)  
**Venv**: `exp/peak-kremer/peaked-circuit-simulation/.venv` (torch 2.10.0+cu128, quimb, cotengra)

```bash
# Full pipeline: supervised then VMC, automatically chained
bash hpc/puhti_nqs_pipeline.slurm --submit

# Supervised only
sbatch hpc/puhti_nqs_pipeline.slurm

# VMC only (needs SUPERVISED_CKPT)
sbatch --export=ALL,STAGE=vmc,SUPERVISED_CKPT=checkpoints/nqs/final.pt \
    hpc/puhti_nqs_pipeline.slurm

# Unswap MPO simulation (job array over a tier)
sbatch --array=0-9 --export=ALL,DIFFICULTY=very_easy \
    hpc/puhti_unswap.slurm
```

Logs: `logs/nqs_nqs-pipeline_<jobid>.out`  
Checkpoints: `checkpoints/nqs/epoch_NNNN.pt`, `checkpoints/nqs/final.pt`  
VMC checkpoints: `checkpoints/nqs_vmc/vmc_epoch_NNNN.pt`, `checkpoints/nqs_vmc/vmc_history.json`

---

## Known Issues / Gotchas

1. **Endianness**: quimb big-endian vs CSV/model little-endian. Always reverse the bitstring before calling `circ.amplitude()`. `quimb_amplitude()` handles this internally.

2. **quimb ZeroDivisionError**: default `simplify_sequence='ADCRS'` crashes on some circuits. Use `simplify_sequence='DC'` (already in `quimb_amplitude()`).

3. **Unswap qubit ordering**: `simulate_qasm` in `unswap.py` returns big-endian amplitudes, causing all circuits to fail `check_slurm_results.py` validation. The reversal fix has not been applied to `unswap.py` yet.

4. **Hard CSV coverage**: only 2 of 7 hard circuits have ground truth (challenge-56_39 and challenge-64_41, both found by marginal methods with very low probabilities 0.003 and 0.0007). Very hard circuits have no ground truth at all.

5. **No batching across circuits**: `train_epoch` uses batch_size=1 because different circuits have different n_qubits and therefore different interaction matrix shapes. Circuits of the same size could be batched but this is not yet implemented.

6. **layer_scales init at 0**: the conditioner has zero effect at init, so the first training steps are equivalent to a plain transformer. This is by design — the model learns circuit conditioning gradually.

---

## Sprint Status

| Sprint | Description                          | Status   |
|--------|--------------------------------------|----------|
| 0      | Data pipeline                        | Done ✅  |
| 1      | Core transformer architecture         | Done ✅  |
| 2      | Supervised training + memorization   | Done ✅  |
| 3      | Circuit conditioning (cx/swap typed) | Done ✅  |
| 4      | Curriculum scale + checkpointing     | Done ✅  |
| 5      | VMC fine-tuning for hard circuits    | Done ✅  |
| 6      | Very hard inference (48–104 qubits)  | Pending  |
