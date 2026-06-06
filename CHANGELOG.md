# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Add majority bit string function to get answer out of low probability string.

### Added

- **`quantum_hack` package** — the core circuit toolkit:
  - `simulation` — `statevector_simulation` (exact peak bitstring) and
    `matrix_product_operators` (approximate, shot-based MPS estimate for larger circuits).
    Both accept `top_n` to return the N most likely `(bitstring, probability)` pairs
    instead of just the peak, and `verbose` to print the result(s).
  - `strip` — `strip_rz_and_cx_from_start`, `strip_rz_from_start`, and `strip_rz_from_end`
    to drop leading/trailing rotation gates per qubit, plus `strip` to apply the leading
    RZ/CX and trailing RZ passes together.
  - `transform` — `transpile_to_basis` (rewrite to an `rx`/`rz`/`cx` basis) and
    `snap_rotations_to_pi_over_4` (snap near-grid angles to multiples of π/4).
  - `metrics` — `circuit_stats` and `compare_circuits` for before/after depth, size, and
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
- **`main.py`** — a scratch single-circuit experiment script.
- **QASM challenge dataset** under `qasm_data/` — 49 circuits across five difficulty tiers.
- **Tooling** — `pyproject.toml` (uv-managed, Python 3.12+), pinned `uv.lock`, `Makefile`,
  `setup.ps1`, a `pre-push` lint+test hook, GitHub Actions CI, and a pull-request template.
- **Documentation** — `README.md` and `GUIDE.md`.

### Changed

- `snap_rotations_to_pi_over_4` never snaps an RZ to 0 (or to a multiple of 2*pi)
  by default: zeroing an RZ would discard a real phase, so such an RZ keeps its
  original angle. RX angles still snap to 0. Pass `snap_rz_to_zero=True` to opt in
  to snapping near-zero RZ angles to 0 as well.

### Fixed

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
