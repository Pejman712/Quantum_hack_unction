# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`quantum_hack` package** — the core circuit toolkit:
  - `simulation` — `statevector_simulation` (exact peak bitstring) and
    `matrix_product_operators` (approximate, shot-based MPS estimate for larger circuits).
  - `strip` — `strip_rz_and_cx_from_start` and `strip_rz_from_end` to drop leading/trailing
    rotation gates per qubit.
  - `transform` — `transpile_to_basis` (rewrite to an `rx`/`rz`/`cx` basis) and
    `snap_rotations_to_pi_over_4` (snap near-grid angles to multiples of π/4).
  - `metrics` — `circuit_stats` and `compare_circuits` for before/after depth, size, and
    per-gate deltas.
  - `verification` — `verify_equivalence` and `circuits_equivalent`, wrapping MQT QCEC and
    treating equivalence up to global/relative phase as equivalent.
  - `viz` — `save_circuit_drawing`, `save_cx_drawing`, `cx_only_circuit`, and
    `save_counts_histogram`.
  - `challenges` — `load_circuit` and `iter_challenges` to load and batch over the
    `qasm_data/` set by difficulty.
- **`quantum-hack` CLI** — render a circuit plus its transpiled and optimized variants to
  `circuit_drawings/` (`--out-dir`, `--no-timestamp`).
- **`main.py`** — a scratch single-circuit experiment script.
- **QASM challenge dataset** under `qasm_data/` — 49 circuits across five difficulty tiers.
- **Tooling** — `pyproject.toml` (uv-managed, Python 3.12+), pinned `uv.lock`, `Makefile`,
  `setup.ps1`, a `pre-push` lint+test hook, GitHub Actions CI, and a pull-request template.
- **Documentation** — `README.md` and `GUIDE.md`.

### Fixed

- `strip_rz_and_cx_from_start` no longer drops RZ/CX gates on a qubit that an
  earlier CX took off `|0>` (as the target of an active control) before its own
  first RX. Those gates carry observable phase, so the strip is now
  equivalence-preserving up to global phase.
- Aligned the test suite with the current API: the simulators return
  `(bitstring, probability)` tuples, and `circuit_stats` reports metrics in the
  normalized `rz`/`rx`/`cx` basis.

[Unreleased]: https://github.com/Pejman712/Quantum_hack_unction/commits/main
