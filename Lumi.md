# Running on LUMI

How to solve the peaked-circuit challenges on LUMI's AMD MI250X GPUs using CSC's Qiskit
container, plus the pitfalls that cost us time so you don't hit them again.

The two job scripts live in [hpc/](hpc/):

- [hpc/lumi_batch.slurm](hpc/lumi_batch.slurm) — solves a tier as a job array (one circuit per shard, GPU).
- [hpc/lumi_aggregate.slurm](hpc/lumi_aggregate.slurm) — appends solved results to the CSVs (CPU, tiny).

Examples below use project `project_465003017` and user `zuhaikha` — substitute your own.

---

## TL;DR (happy path)

```bash
# 1. From your LAPTOP (not inside ssh), copy the repo to /scratch:
scp -i <key> -r src qasm_data results hpc pyproject.toml main.py README.md \
    zuhaikha@lumi.csc.fi:/scratch/project_465003017/Quantum_hack_unction/

# 2. ssh in, cd to the repo, smoke-test on the login node (CPU, a tiny cached circuit):
PYTHONPATH=$PWD/src /appl/local/quantum/qiskit/run-singularity \
    /appl/local/quantum/qiskit/qiskit_2.3.0_csc2.sif \
    python -m quantum_hack.batch --challenge challenge-8_1

# 3. Submit a tier (array size = circuit count - 1). For the slow tiers use no per-circuit
#    cap and a long wall clock (see the table below):
sbatch --array=0-6 --account=project_465003017 \
    --export=ALL,LUMI_QISKIT_SINGULARITY_CONTAINER_PATH=/appl/local/quantum/qiskit/qiskit_2.3.0_csc2.sif,DIFFICULTY=hard,SHOTS=16384,BOND_DIM=256,TIME_BUDGET=0 \
    --time=06:00:00 hpc/lumi_batch.slurm

# 4. When everything is solved, write the CSVs:
sbatch hpc/lumi_aggregate.slurm
```

---

## Prerequisites

- A LUMI account and a project with GPU allocation (here `project_465003017`).
- Your SSH key registered with LUMI: `ssh -i <key> zuhaikha@lumi.csc.fi`.
- The CSC Qiskit container, already on LUMI at `/appl/local/quantum/qiskit/` — **no Python
  environment to build**. The container ships `qiskit-aer-gpu-rocm` (GPU Aer for MI250X).

---

## 1. Get the code onto LUMI

Run the copy **from your laptop**, into `/scratch` (or `/project`) — not your home directory,
which has tight quotas and isn't meant for job I/O.

```powershell
# local PowerShell — scp ships with Windows OpenSSH; listing dirs explicitly skips
# the 400 MB Windows-only .venv and the .git history (both useless on LUMI):
scp -i C:\Users\khanm\.ssh\id_ed25519_personal -r `
    src qasm_data results hpc pyproject.toml main.py README.md `
    zuhaikha@lumi.csc.fi:/scratch/project_465003017/Quantum_hack_unction/
```

`src/`, `qasm_data/`, `results/`, and `hpc/` are everything the jobs need. `results/` carries
the existing `_peak_cache/` and CSVs, so already-solved circuits are skipped.

> Cloning on LUMI works too (`git clone https://github.com/Pejman712/Quantum_hack_unction.git`)
> but needs GitHub auth from LUMI and won't include uncommitted local changes.

## 2. The Qiskit container

There are **two** builds in `/appl/local/quantum/qiskit/`. The version the scripts default to
(`qiskit_2.3.0_csc.sif`) is mode `-rwxr-x---`, owned by a project group you are probably **not**
in — so you get *"not readable by the current user"*. Use the world-readable one:

| File | Perms | Use it? |
|------|-------|---------|
| `qiskit_2.3.0_csc.sif`  | `-rwxr-x---` (group-only) | ❌ not readable by you |
| **`qiskit_2.3.0_csc2.sif`** | `-rwxr-xr-x` (world) | ✅ **yes** |
| `qiskit_2.3.0_hpcqc-ml-training_csc2.sif` | `-rwxr-xr-x` | ❌ different (ML-training) variant |

The most robust way to select it is to set the path **explicitly in the job**, so you don't
depend on editing the script or on a shell env var surviving:

```
--export=ALL,LUMI_QISKIT_SINGULARITY_CONTAINER_PATH=/appl/local/quantum/qiskit/qiskit_2.3.0_csc2.sif,...
```

The launcher wrapper `/appl/local/quantum/qiskit/run-singularity` is world-executable; use it as
`run-singularity <container.sif> python -m quantum_hack.batch ...`.

## 3. Smoke-test before submitting at scale

**Login nodes (`uanXX`) have no GPU**, so test imports on the login node with CPU and a small,
already-cached circuit:

```bash
cd /scratch/project_465003017/Quantum_hack_unction
PYTHONPATH=$PWD/src /appl/local/quantum/qiskit/run-singularity \
    /appl/local/quantum/qiskit/qiskit_2.3.0_csc2.sif \
    python -m quantum_hack.batch --challenge challenge-8_1
```

A clean `[cached]`/`[solved]` line means the container opens and the package imports. To verify
the **GPU** path, grab a short interactive GPU shell (don't use the login node):

```bash
srun --account=project_465003017 --partition=small-g --gpus=1 --cpus-per-task=7 \
     --time=00:15:00 --pty bash
# inside the allocation, pick an uncached circuit:
PYTHONPATH=$PWD/src /appl/local/quantum/qiskit/run-singularity \
    /appl/local/quantum/qiskit/qiskit_2.3.0_csc2.sif \
    python -m quantum_hack.batch --challenge <name> --gpu --device GPU
```

## 4. Submit the solve jobs

One `sbatch` per tier. **Array size = (number of circuits − 1)**, so each shard solves one
circuit on its own GCD (`small-g` is partial-node, so the scheduler packs several shards onto a
node, each with a distinct GCD). Set `--difficulty` via the `DIFFICULTY` export.

| Tier | Circuits | `--array` | Notes |
|------|----------|-----------|-------|
| very_easy | 10 | `0-9`  | fast |
| easy | 17 | `0-16` | mostly fast; one slow straggler |
| moderate | 8 | `0-7`  | the 48–64q circuits need > 1 h |
| hard | 7 | `0-6`  | slow — use the long-budget recipe |
| very_hard | 8 | `0-7`  | slowest — use the long-budget recipe |

**Tunables** (via `--export=ALL,...`): `DIFFICULTY`, `SHOTS` (16384), `BOND_DIM` (256),
`TIME_BUDGET` (per-circuit seconds; **`0` = no cap**).

For the big tiers, the default 1-hour per-circuit budget and 75-minute wall clock are too short
(see pitfall #5). Use **`TIME_BUDGET=0` and a generous `--time`** so each one-circuit shard can
use the whole allocation:

```bash
# hard / very_hard / moderate-stragglers: no per-circuit cap, 6 h wall clock
sbatch --array=0-6 --account=project_465003017 \
    --export=ALL,LUMI_QISKIT_SINGULARITY_CONTAINER_PATH=/appl/local/quantum/qiskit/qiskit_2.3.0_csc2.sif,DIFFICULTY=hard,SHOTS=16384,BOND_DIM=256,TIME_BUDGET=0 \
    --time=06:00:00 hpc/lumi_batch.slurm
```

The smaller tiers are fine with the script defaults (1-hour budget).

## 5. Monitor

```bash
squeue --me                                  # PD = queued (normal), R = running, gone = finished
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList%14   # post-mortem
# live coverage per tier:
for d in very_easy easy moderate hard very_hard; do
  echo "$d: $(ls results/_peak_cache/$d 2>/dev/null | grep -c json)/$(ls qasm_data/$d | grep -c qasm)"
done
```

Tail the **right** log by job ID (a bare glob grabs stale failed jobs — see pitfall #8). The
first line of each log echoes `budget=<N>s`, which is how you confirm you're looking at the run
you think you are.

## 6. Aggregate, check, retrieve

```bash
# write/update results/<tier>_bitstrings.csv from the cache (CPU, no GPU, safe to re-run):
PYTHONPATH=$PWD/src /appl/local/quantum/qiskit/run-singularity \
    /appl/local/quantum/qiskit/qiskit_2.3.0_csc2.sif \
    python -m quantum_hack.batch --aggregate-only

# spot-check confidence / local-max for each cached answer:
PYTHONPATH=$PWD/src /appl/local/quantum/qiskit/run-singularity \
    /appl/local/quantum/qiskit/qiskit_2.3.0_csc2.sif \
    python -m quantum_hack.batch --compare
```

Pull the results back **from your laptop**:

```powershell
scp -i C:\Users\khanm\.ssh\id_ed25519_personal -r `
    zuhaikha@lumi.csc.fi:/scratch/project_465003017/Quantum_hack_unction/results `
    c:\dev\Quantum_hack_unction\results_lumi
```

Find and re-solve any gaps (idempotent — cached circuits skip):

```bash
for d in easy moderate hard very_hard; do
  for f in qasm_data/$d/*.qasm; do b=$(basename "$f" .qasm)
    [ -f "results/_peak_cache/$d/$b.json" ] || echo "MISSING: $d/$b"; done
done
```

---

## Common pitfalls

1. **Running the copy from inside the ssh session.** `rsync .../laptop/... user@lumi:...` *while
   logged into LUMI* fails with *"source and destination cannot both be remote"* — the laptop
   path doesn't exist on LUMI. Run the copy from your **laptop**. (Windows PowerShell has `scp`
   but not `rsync`.)

2. **Container "not readable".** The scripts default to `qiskit_2.3.0_csc.sif`, which is
   group-restricted. Use **`qiskit_2.3.0_csc2.sif`** and set it explicitly with
   `--export=ALL,LUMI_QISKIT_SINGULARITY_CONTAINER_PATH=...`. Don't rely on a `sed` edit or an
   exported shell variable — both are easy to lose track of.

3. **Testing GPU on a login node.** `uanXX` login nodes have no GPU; `--gpu --device GPU` there
   fails. Smoke-test imports on CPU on the login node, and test GPU inside an `srun --partition=small-g --gpus=1` allocation.

4. **Pasting `<name>` literally.** Bash reads `<name>` as a redirect (`-bash: name: No such file
   or directory`). Replace placeholders with a real value, e.g. `challenge-8_1`.

5. **Slow circuits silently "time out".** Large circuits (≈48q+) exceed the default 1-hour
   `TIME_BUDGET`. The solver records a `[timeout]` (empty bitstring, **not cached**, so it's
   retried) and the *job still exits 0*. Result: `sacct` says `COMPLETED` but the cache count
   doesn't move. Fix: `TIME_BUDGET=0` + a long `--time` (e.g. `06:00:00`). Check the **cache
   count**, not just `sacct State`.

6. **Duplicate submissions.** Submitting a tier while a previous run is still in `squeue --me`
   makes two shards solve the same circuit. It can't corrupt results (writes are idempotent) but
   it wastes GPU. Always `squeue --me` first; only resubmit a tier to mop up circuits that are
   missing *after* the previous run finished.

7. **Flaky-node exec-format error.** Occasionally a node fails `run-singularity` with
   `Exec format error` / `Exited with exit code 8` in ~3 s. It's that node, not your setup (the
   same command works on other nodes). Just resubmit (idempotent); if one node repeats, add
   `--exclude=<nidXXXXXX>` and report it to LUMI support.

8. **Tailing stale logs.** `tail slurm-peak-batch-*.out` grabs *every* job that ever started,
   including old failures. Identify the run you want by job ID and by the `budget=<N>s` line in
   its first row before trusting what you read.

9. **`COMPLETED` ≠ solved.** A job that completes cleanly may have hit a per-circuit timeout
   (#5) or skipped a cached circuit in seconds. Trust the JSON cache (`results/_peak_cache/`),
   not the SLURM state.

10. **Distinguishing failure modes** (read `sacct ... Elapsed,MaxRSS,ExitCode` + the log tail):
    - `ExitCode 127`, ~3 s, `MaxRSS` a few MB → container never opened (bad/unreadable path).
    - `ExitCode 8`, ~3 s → flaky-node exec-format error (#7).
    - `State=CANCELLED`, `ExitCode 0:15`, long `Elapsed`, growing `MaxRSS` → it was **healthy and
      progressing** and got `scancel`'d. Don't cancel a job with a long elapsed and rising memory
      thinking it's stuck — it's working.
    - `State=OUT_OF_MEMORY` / `oom-kill` in the log → too big for the task memory; lower
      `BOND_DIM` and/or raise `--mem`.
    - SLURM's own kill messages (`DUE TO TIME LIMIT`, `oom-kill`) are in the **log tail**, not in
      the solver's `[timeout]`/`[failed]` tags — grep both.

11. **`MaxRSS` is host RAM only.** GPU VRAM isn't reported there, so a low `MaxRSS` doesn't rule
    out a GPU-memory problem (and a host OOM shows as a high `MaxRSS` near `ReqMem`).

12. **The `Solving …` line is a shell echo**, printed by the script *before* the container runs
    ([hpc/lumi_batch.slurm](hpc/lumi_batch.slurm)). Seeing it does **not** mean the solve started
    — a container failure happens right after it.

13. **Forgetting `--account`.** The scripts ship with a placeholder
    `--account=project_465XXXXXX`. Pass `--account=project_465003017` on the `sbatch` line (it
    overrides the script) or your jobs won't be billed to the right project.

---

## How the batch solver behaves

- **Idempotent / resumable.** Each circuit writes `results/_peak_cache/<tier>/<name>.json`.
  Re-running skips circuits with a cached result; timeouts and crashes are *not* cached, so they
  are retried on the next submission. Per-circuit JSON files are safe to write concurrently.
- **Timeout enforcement.** With `TIME_BUDGET > 0`, each solve runs in a spawned subprocess that
  the parent hard-kills on overrun (Aer's C++ kernels ignore signals). With `TIME_BUDGET=0` the
  solve runs **inline** with no cap — the SLURM wall clock is then the only bound, which is what
  you want for one-circuit-per-shard arrays.
- **Aggregation is separate.** Solving only writes JSON; `--aggregate-only` (or
  [hpc/lumi_aggregate.slurm](hpc/lumi_aggregate.slurm)) appends pending rows to the CSVs. It
  never re-solves and needs no GPU, so it's safe to run anytime, including while jobs are still
  in flight.
- **Quality signals.** Each result carries `bitstring`, `probability`, `confidence` (0–1), and
  `method`. `--compare` flags low-confidence or non-local-max answers — candidates to re-solve
  with a higher `BOND_DIM` or the `--optimize` pipeline.
