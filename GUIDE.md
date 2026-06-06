# Developer & Usage Guide

A deeper companion to the [README](README.md): the problem the toolkit solves, the API
module by module, and the workflows you'll actually run.

## Contents

- [The problem](#the-problem)
- [Key concepts](#key-concepts)
- [Environment](#environment)
- [API reference](#api-reference)
- [Workflows](#workflows)
- [CLI and scripts](#cli-and-scripts)
- [Development workflow](#development-workflow)
- [Extending the toolkit](#extending-the-toolkit)
- [Gotchas](#gotchas)

## The problem

The hackathon provides OpenQASM circuits in [`qasm_data/`](qasm_data/), grouped into five
difficulty tiers (`very_easy` → `very_hard`). Filenames follow
`challenge-<qubits>_<id>.qasm`, e.g. `challenge-8_1.qasm` is an 8-qubit circuit. Two goals:

1. **Find the peak bitstring** — the measurement outcome with the highest probability for
   each circuit.
2. **Simplify the circuit** — reduce gate count / depth while preserving the function, so
   the peak bitstring is unchanged.

This toolkit provides composable helpers for both.

## Key concepts

- **Peak bitstring** — the most probable measurement result. Returned together with its
  probability as a `tuple[str, float]`. Bitstrings use Qiskit's ordering (qubit *n-1 … 0*,
  little-endian on the right).
- **Two simulators**:
  - *Exact* (`statevector_simulation`) — dense statevector; accurate but memory scales as
    `2**num_qubits`, so it is only practical for small circuits.
  - *Approximate* (`matrix_product_operators`) — matrix-product-state (MPS) sampling; scales
    to many more qubits at the cost of being a shot-based estimate.
- **Basis gates** — circuits are normalized to `rx`, `rz`, `cx` (`transform.DEFAULT_BASIS_GATES`).
- **Equivalence up to phase** — global and relative phase are not observable, so a
  transpiled/optimized circuit is considered equivalent to its original even if phase differs.

## Environment

- Python **3.12+**, managed with [uv](https://docs.astral.sh/uv/).
- `make setup` (or `.\setup.ps1` on Windows) runs `uv sync --dev` and points
  `core.hooksPath` at [`.githooks/`](.githooks/).
- Everything runs through `uv run …` so the project virtualenv is used automatically.

## API reference

Everything below is re-exported from the package root, so `from quantum_hack import …`
works for all of it.

### `challenges` — loading and batching

- `load_circuit(path: Path | str) -> QuantumCircuit` — load one OpenQASM file, raising
  `FileNotFoundError` with a clear message if it's missing.
- `iter_challenges(difficulty=None, func=matrix_product_operators, *, data_dir="qasm_data")
  -> Iterator[tuple[str, object]]` — walk the dataset in sorted order, yielding
  `("<difficulty>/<stem>", func(circuit))`. Restrict to one tier with `difficulty="easy"`;
  pass `func=None` to get the raw `QuantumCircuit` instead of a computed value.

### `results` — tracking which submissions worked

Submitted peak-bitstring guesses are logged in `results/<difficulty>_bitstrings.csv`
(columns `challenge, qubits, method, bitstring, probability, status`, where `status` is
`success` / `failed` / `pending`).

- `load_results(results_dir="results") -> list[SubmissionRecord]` — parse every row across
  all `*_bitstrings.csv` files into `SubmissionRecord`s (difficulty is derived from the
  filename).
- `challenge_status(challenge, *, results_dir="results", records=None) -> ChallengeStatus`
  — aggregate **all** submissions for one challenge. `worked` is `True` if any succeeded,
  `False` if attempts exist but all failed, `None` if untried/only pending. A challenge may
  have several failed bitstrings; they are all collected in `.failed_bitstrings`, and
  `.summary` (also `str(status)`) is a one-line suggestion. Accepts a bare name, an
  `iter_challenges` name (`"easy/challenge-16_12"`), or a path. Pass `records=` to query
  pre-loaded records instead of reading disk.
- `is_known_failure(challenge, bitstring, *, results_dir="results", records=None) -> bool`
  — `True` if that bitstring was already submitted for the challenge and rejected.

### `simulation` — peak bitstrings

- `statevector_simulation(qc, verbose=False) -> tuple[str, float]` — exact peak bitstring
  and probability. Final measurements are removed on a copy first; the input is untouched.
- `matrix_product_operators(qc, shots=4096, bond_dim=64, verbose=False) -> tuple[str, float]`
  — MPS estimate. Adds a fresh `measure_all` on a copy, samples `shots` times, and returns
  the most frequent bitstring with its empirical probability.

### `transform` — basis and angle normalization

- `transpile_to_basis(qc, basis_gates=None, optimization_level=0, **kw) -> QuantumCircuit`
  — rewrite into the basis (default `["rx", "rz", "cx"]`). Use `optimization_level=0` to only
  rewrite, `2`/`3` to optimize harder. Extra kwargs pass through to `qiskit.transpile`.
- `snap_rotations_to_pi_over_4(qc, threshold=0.01) -> QuantumCircuit` — for every `rz`/`rx`,
  if its angle is within `threshold * (π/4)` of a multiple of π/4, snap it to that exact
  multiple (wrapped into `(-π, π]`). `threshold` is a fraction of the grid spacing, so the
  default snaps within 1% of π/4. Symbolic/unbound parameters are left alone; an RZ is
  never snapped to 0 (RX may be), since zeroing an RZ would drop a real phase.

### `strip` — dropping boundary rotations

- `strip_rz_and_cx_from_start(qc) -> QuantumCircuit` — per qubit, drop `rz`/`cx` gates that
  occur before that qubit's first `rx`. A `cx` is dropped only when it precedes the first
  `rx` on *both* of its qubits.
- `strip_rz_from_end(qc) -> QuantumCircuit` — per qubit, drop `rz` gates that occur after
  that qubit's last `rx`/`cx`.

### `metrics` — measuring the effect

- `circuit_stats(qc) -> dict` — `{"num_qubits", "depth", "size", "ops"}`, measured after a
  light transpile to `rz`/`rx`/`cx` so counts are comparable.
- `compare_circuits(before, after) -> dict` — `depth`, `size`, and per-gate `ops` each as
  `{"before", "after", "delta"}`. A negative delta means the metric shrank.

### `verification` — proving equivalence

- `verify_equivalence(qc1, qc2, **kw) -> EquivalenceCriterion` — the raw QCEC criterion.
- `circuits_equivalent(qc1, qc2, **kw) -> bool` — `True` for `equivalent`,
  `equivalent_up_to_global_phase`, or `equivalent_up_to_phase`. A non-definitive
  `probably_equivalent` returns `False`; use `verify_equivalence` when you need the detail.
  (`mqt.qcec` is imported lazily, so importing `quantum_hack` doesn't require it until used.)

### `viz` — diagrams and histograms

- `save_circuit_drawing(qc, name="circuit", *, out_dir="circuit_drawings", timestamp=True,
  image_format="png", **draw_kwargs) -> Path` — render with the matplotlib drawer. Creates
  `out_dir`; timestamped filenames sort chronologically. Use `timestamp=False` for a stable
  name.
- `cx_only_circuit(qc) -> QuantumCircuit` — a copy containing only the `cx` gates.
- `save_cx_drawing(...)` — render the CX-only structure (same options as above).
- `save_counts_histogram(counts, name="histogram", ...)` — plot a measurement histogram.

> Rendered PNGs land in `circuit_drawings/`, which is git-ignored.

## Workflows

### Peak bitstring for one circuit

```python
from quantum_hack import load_circuit, statevector_simulation, matrix_product_operators

qc = load_circuit("qasm_data/very_easy/challenge-8_1.qasm")
print(statevector_simulation(qc))     # exact, small circuits
print(matrix_product_operators(qc))   # approximate, scales further
```

### Batch a difficulty tier

```python
from quantum_hack import iter_challenges

for name, peak in iter_challenges("moderate"):
    print(name, peak)
```

### Optimize and verify

```python
from quantum_hack import (
    load_circuit, strip_rz_and_cx_from_start, snap_rotations_to_pi_over_4,
    transpile_to_basis, compare_circuits, circuits_equivalent,
)

qc = load_circuit("qasm_data/easy/challenge-16_12.qasm")
optimized = transpile_to_basis(
    snap_rotations_to_pi_over_4(strip_rz_and_cx_from_start(qc), threshold=0.02),
    optimization_level=2,
)
print(compare_circuits(qc, optimized))      # how much smaller?
assert circuits_equivalent(qc, optimized)   # still the same operation?
```

> Stripping boundary gates can change the circuit's function — always confirm with
> `circuits_equivalent` before trusting a stripped result.

## CLI and scripts

| Command | What it does |
| --- | --- |
| `uv run quantum-hack <qasm>` | Render original, transpiled (`opt 0`), and optimized (`opt 2`) diagrams |
| `uv run quantum-hack --help` | Show CLI options (`--out-dir`, `--no-timestamp`) |
| `make run QASM=<qasm>` | Same, via make |
| `uv run python main.py` | Scratch experiment on `challenge-8_1` |
| `make test` | `uv run pytest` |
| `make lint` | `ruff` import-sort + lint |
| `make format` | `ruff format` |

## Development workflow

- **Tests** mirror each module in [`tests/`](tests/); add or update them in the same change
  as the code. `tests/conftest.py` forces matplotlib's headless `Agg` backend so drawing
  works in CI.
- **Pre-push hook** ([`.githooks/pre-push`](.githooks/pre-push)) runs import-sort, lint, and
  the full suite before every push. Enable it with `make setup` / `setup.ps1`.
- **CI** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the same checks on
  push and PRs and fails on any failing test.
- **Changelog gate** — every PR must add an entry under `[Unreleased]` in
  [`CHANGELOG.md`](CHANGELOG.md); CI blocks PRs that don't.
- **Style** — ruff with `line-length = 100` and the `E,F,I,W,UP,B` rule set
  (see [`pyproject.toml`](pyproject.toml)). Annotate every parameter and return type, and
  document input/output types in docstrings.

## Extending the toolkit

To add a new transform or helper:

1. Add the function to the relevant module under `src/quantum_hack/` with full type hints and
   a docstring documenting input/output types.
2. Re-export it from [`src/quantum_hack/__init__.py`](src/quantum_hack/__init__.py) (add to
   both the import and `__all__`).
3. Add tests under `tests/` covering the behavior and edge cases.
4. Run `make lint && make test`, and add a `[Unreleased]` entry to `CHANGELOG.md`.

## Gotchas

- **Bitstring ordering** is Qiskit's: the rightmost character is qubit 0.
- **Measurements are removed on a copy** by the simulators and the drawer; your input
  circuit is never mutated.
- **MPS results are estimates** — increase `shots` (or `bond_dim`) if a peak looks unstable.
- **Angle wrapping** — snapped angles are wrapped into `(-π, π]`, so e.g. `~2π` becomes `0`.
- **Equivalence up to phase counts as equivalent**; a definitive `not equivalent` from QCEC
  means the transform changed the circuit's function.
