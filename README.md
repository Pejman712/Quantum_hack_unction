# Quantum Hack — QASM Circuit Toolkit

[![CI](https://github.com/Pejman712/Quantum_hack_unction/actions/workflows/ci.yml/badge.svg)](https://github.com/Pejman712/Quantum_hack_unction/actions/workflows/ci.yml)

Tooling for the quantum hackathon challenge set: load OpenQASM circuits, find their
**peak measurement bitstring**, and **simplify / optimize** circuits while preserving
their function. Built on [Qiskit](https://www.ibm.com/quantum/qiskit),
[Qiskit Aer](https://github.com/Qiskit/qiskit-aer), and
[MQT QCEC](https://github.com/cda-tum/mqt-qcec).

The challenge circuits live under [`qasm_data/`](qasm_data/) — 49 circuits across five
difficulty tiers (`very_easy`, `easy`, `moderate`, `hard`, `very_hard`).

## What it does

- **Peak bitstring estimation** — find the most likely measurement outcome of a circuit,
  either exactly (dense statevector) or approximately (matrix-product-state sampling) for
  circuits too large to simulate exactly.
- **Circuit simplification** — strip leading/trailing rotation gates, snap near-grid
  rotation angles to multiples of π/8, and transpile to an `rx`/`rz`/`cx` basis.
- **Equivalence checking** — prove that an optimized circuit still implements the same
  operation as the original (up to unobservable global/relative phase) via MQT QCEC.
- **Metrics & diagrams** — before/after gate-count and depth comparisons, plus PNG/SVG
  renderings of circuits, their CX-only structure, and measurement histograms.

## Requirements

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

One command installs dependencies and enables the pre-push hook (lint + tests):

```bash
# macOS / Linux
make setup

# Windows (PowerShell)
.\setup.ps1
```

Both run `uv sync --dev` and `git config core.hooksPath .githooks`.

## Usage

### CLI

Render a circuit and its transpiled/optimized variants to `circuit_drawings/`:

```bash
uv run quantum-hack qasm_data/easy/challenge-16_12.qasm
uv run quantum-hack --help          # all options (--out-dir, --no-timestamp)
# or, via make:
make run QASM=qasm_data/easy/challenge-16_12.qasm
```

### Scratch script

[`main.py`](main.py) is a single-circuit playground (stats, peak bitstrings, before/after
drawings):

```bash
uv run python main.py
```

### As a library

```python
from quantum_hack import (
    load_circuit,
    statevector_simulation,
    matrix_product_operators,
    strip_rz_and_cx_from_start,
    snap_rotations_to_pi_over_8,
    transpile_to_basis,
    compare_circuits,
    circuits_equivalent,
    save_circuit_drawing,
)

qc = load_circuit("qasm_data/very_easy/challenge-8_1.qasm")

# Both return (bitstring, probability). Statevector is exact but limited to small
# circuits; matrix_product_operators is an approximate, shot-based estimate.
print(statevector_simulation(qc))      # -> (peak_bitstring, exact_probability)
print(matrix_product_operators(qc))    # -> (peak_bitstring, estimated_probability)

# Simplify, then confirm the result is still functionally the same circuit.
clean = strip_rz_and_cx_from_start(qc)
snapped = snap_rotations_to_pi_over_8(clean, threshold=0.02)
optimized = transpile_to_basis(snapped, optimization_level=2)

print(compare_circuits(qc, optimized))     # depth/size/per-gate deltas
print(circuits_equivalent(qc, optimized))  # True when functionally identical
save_circuit_drawing(optimized, "optimized", timestamp=False)
```

Batch over a whole difficulty tier:

```python
from quantum_hack import iter_challenges

for name, peak in iter_challenges("very_easy"):
    print(name, peak)   # e.g. "very_easy/challenge-8_1", ("01001000", 0.97)
```

See [GUIDE.md](GUIDE.md) for the full API reference and common workflows.

## Project structure

```
src/quantum_hack/
  challenges.py     load_circuit, iter_challenges (batch over qasm_data/)
  simulation.py     statevector_simulation, matrix_product_operators
  strip.py          strip_rz_and_cx_from_start, strip_rz_from_end
  transform.py      transpile_to_basis, snap_rotations_to_pi_over_8
  metrics.py        circuit_stats, compare_circuits
  verification.py   verify_equivalence, circuits_equivalent
  viz.py            save_circuit_drawing, save_cx_drawing, cx_only_circuit, save_counts_histogram
  cli.py            `quantum-hack` entry point
tests/              pytest suite mirroring each module
qasm_data/          challenge circuits, grouped by difficulty
main.py             scratch single-circuit experiment
```

## Development

```bash
make test          # uv run pytest
make lint          # ruff check (import sorting + lint)
make format        # ruff format
```

- A **pre-push hook** ([`.githooks/pre-push`](.githooks/pre-push)) runs import-sort, lint,
  and the full test suite before every push.
- **CI** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the same checks on
  push and pull requests and **fails on any failing test**.
- Pull requests must update [`CHANGELOG.md`](CHANGELOG.md) under `[Unreleased]` — CI blocks
  PRs that don't.

## License

No license has been declared yet. Add a `LICENSE` file before distributing.
