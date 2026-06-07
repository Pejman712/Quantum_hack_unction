# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.0.1

Add majority bit string function to get answer out of low probability string.

### Added

- **`quantum_hack` package** — the core circuit toolkit:
  - `simulation` — `statevector_simulation` (exact peak bitstring) and
    `matrix_product_operators` (approximate, shot-based MPS estimate for larger circuits).
    Both accept `top_n` to return the N most likely `(bitstring, probability)` pairs
    instead of just the peak, and `verbose` to print the result(s). Both also accept
    `device="GPU"` to run on Aer's GPU (LUMI `standard-g` via the CSC container). New
    `mps_sample_counts` exposes the raw MPS sample counts (so the marginal attack can reuse one
    sampling run), and `statevector_probability` returns the exact probability of a given bitstring.
  - `peak` — an MPS-based, confidence-gated peak-bitstring solver that never uses an exact
    statevector: `profile_circuit`/`select_method` (MPS sampling → marginal vote → greedy refine →
    optional tensor-network amplitude oracle), `exact_z_marginals`/`z_marginals_from_counts`/
    `marginal_bitstring` (the marginal "Z-expectation" attack, which recovers peaks that
    argmax-over-samples misses), `greedy_refine` (steepest-ascent bit-flip hill-climb on an
    amplitude oracle), `tensor_network` (lazy `quimb`/`cotengra` exact-amplitude backend for large
    circuits), and `solve_peak` returning a `PeakResult` with a `score_confidence` composite
    (magnitude + single-bit-flip local-max + multi-method consensus). It consults
    `is_known_failure` so a rejected bitstring is never re-proposed.
  - `optimize` — `optimize_circuit`, a light greedy-with-rollback reducer (`strip` → angle-snap →
    `transpile` opt-3 → optional `pyzx`) that keeps a step only if it does not grow a weighted
    `cost` and preserves the peak bitstring; reports a `by_construction`/`peak_verified` outcome
    guarantee. Benchmarking helpers (`optimize_circuits`, `OptimizationStats`,
    `OptimizationSummary`, `format_optimization_table`) compare before/after CX depth, size, and CX
    count across a set of circuits.
  - `batch` — `run_batch`/`solve_job` solve every challenge (or one `--challenge`, a tier, or a
    SLURM `--shard`) with per-circuit JSON checkpoints (resume- and array-safe) and a per-circuit
    `--time-budget` that runs each solve in a subprocess and hard-kills it on overrun, recording a
    timeout (not cached) so it is retried later. An opt-in `--optimize` flag (off by default) runs
    the peak-preserving `optimize_circuit` pipeline (strip → snap → transpile) on each circuit
    before solving and records the applied steps in the result notes. `write_results_csvs` appends `pending` candidate
    rows to `results/<difficulty>_bitstrings.csv` without disturbing human `success`/`failed` marks;
    `compare_cache_to_results`/`failed_challenges` cross-check the cached answers against those CSVs
    (match / mismatch / known-failure / unconfirmed).
  - `strip` — `strip_rz_and_cx_from_start`, `strip_rz_from_start`, and `strip_rz_from_end`
    to drop leading/trailing rotation gates per qubit, plus `strip` to apply the leading
    RZ/CX and trailing RZ passes together.
  - `transform` — `transpile_to_basis` (rewrite to an `rx`/`rz`/`cx` basis, restoring qubit order
    after any SWAP-elision permutation), `snap_rotations_to_pi_over_8` (snap near-grid angles to
    multiples of π/8), and `optimize_with_pyzx` (optional ZX-calculus reduction via the `pyzx`
    `[zx]` extra).
  - `metrics` — `circuit_stats` and `compare_circuits` for before/after depth, CX depth, size, and
    per-gate deltas.
  - `verification` — `verify_equivalence` and `circuits_equivalent`, wrapping MQT QCEC and
    treating equivalence up to global/relative phase as equivalent.
  - `viz` — `save_circuit_drawing`, `save_cx_drawing`, `cx_only_circuit`, and
    `save_counts_histogram`.
  - `challenges` — `load_circuit`, `find_challenge` (locate a challenge's QASM file by name
    across difficulty folders), and `iter_challenges` to load and batch over the
    `qasm_data/` set by difficulty.
  - `results` — read recorded submissions from `results/*_bitstrings.csv` and judge a
    challenge: `load_results` parses every row into `SubmissionRecord`s; `challenge_status`
    aggregates all submissions for a challenge into a `ChallengeStatus` (`worked` verdict
    plus a one-line suggestion), collecting **every** failed bitstring so none are retried;
    `is_known_failure` checks whether a candidate bitstring was already submitted and
    rejected.
- **`quantum-hack` CLI** — render a circuit plus its transpiled and optimized variants to
  `circuit_drawings/` (`--out-dir`, `--no-timestamp`).
- **`quantum-hack-batch` CLI** — batch-solve from the command line with `--difficulty`,
  `--challenge`, `--shard`, `--gpu`/`--device`, `--shots`, `--bond-dim`, `--time-budget`,
  `--write-csv`, `--aggregate-only`, and `--compare`.
- **`hpc/` SLURM scripts for LUMI** — `lumi_batch.slurm` (GPU job array on `small-g` via CSC's
  Qiskit container, with overridable `DIFFICULTY`/`SHOTS`/`BOND_DIM`/`TIME_BUDGET`) and
  `lumi_aggregate.slurm` (write the result CSVs from the cache).
- **`main.py`** — a scratch single-circuit experiment script.
- **QASM challenge dataset** under `qasm_data/` — 49 circuits across five difficulty tiers.
- **Tooling** — `pyproject.toml` (uv-managed, Python 3.12+) with an optional `tn` extra
  (`quimb`/`cotengra`) for the tensor-network backend, pinned `uv.lock`, `Makefile`, `setup.ps1`,
  a `pre-push` lint+test hook, GitHub Actions CI, and a pull-request template.
- **Documentation** — `README.md`, `GUIDE.md`, and `Lumi.md` (a LUMI runbook: container
  selection, per-tier `sbatch` recipes, monitoring, and the common pitfalls hit bringing it up).

### Changed

- `snap_rotations_to_pi_over_8` snaps near-grid RZ/RX angles to multiples of π/8 (a finer grid
  than the earlier π/4). It never snaps an RZ to 0 (or to a multiple of 2*pi) by default: zeroing
  an RZ would discard a real phase, so such an RZ keeps its original angle. RX angles still snap to
  0. Pass `snap_rz_to_zero=True` to opt in to snapping near-zero RZ angles to 0 as well.
- The `optimize` pipeline gained a final, cost-guarded `pyzx` ZX-calculus step (skipped above
  `PYZX_MAX_QUBITS` qubits or when `pyzx` is absent), and now reports CX depth as its headline
  metric. On the arbitrary-angle challenge circuits pyzx consistently grows the gate count, so the
  cost guard rolls it back; transpile (opt-2 ≈ opt-3) does the real reduction.

### Fixed

- `transpile_to_basis` now restores the input qubit order after transpilation. At
  `optimization_level >= 1` the transpiler can elide SWAP gates into a layout permutation, which
  relabels the peak bitstring of a measurement-free circuit — silently producing a wrong answer
  that `optimize_circuit` trusted as `by_construction`. The permutation is detected from the
  transpile layout and undone by relabelling wires (gate counts unchanged).
- `statevector_simulation` now reads the raw probability array (numpy `argpartition`) instead of
  `Statevector.probabilities_dict()`, which materialised a string label for every one of the
  `2**n` basis states (~4 GiB at 24 qubits) and effectively limited exact simulation to tiny
  circuits.
- `matrix_product_operators` no longer fails on circuits wider than the
  `AerSimulator`'s memory-based default qubit ceiling (e.g. 64-qubit circuits hit
  a `CircuitTooWideForTarget` at the 63-qubit limit). Transpilation now targets the
  simulator's supported standard gates rather than the backend target, so the MPS
  method's true width capacity is used.
- `strip_rz_and_cx_from_start` no longer drops RZ/CX gates on a qubit that an
  earlier CX took off `|0>` (as the target of an active control) before its own
  first RX. Those gates carry observable phase, so the strip is now
  equivalence-preserving up to global phase.
- Aligned the test suite with the current API: the simulators return
  `(bitstring, probability)` tuples, and `circuit_stats` reports metrics in the
  normalized `rz`/`rx`/`cx` basis.

[Unreleased]: https://github.com/Pejman712/Quantum_hack_unction/commits/main
