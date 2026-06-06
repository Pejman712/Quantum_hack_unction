"""Loading challenge circuits from the qasm_data folder and batch-running over them."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

from qiskit import QuantumCircuit

from quantum_hack.simulation import auto_simulation, matrix_product_operators

DEFAULT_DATA_DIR = Path("qasm_data")
DIFFICULTIES = ("very_easy", "easy", "moderate", "hard", "very_hard")


def load_circuit(path: Path | str) -> QuantumCircuit:
    """Load a circuit from an OpenQASM file, with a clear error if it is missing.

    Args:
        path: Path to the OpenQASM file.

    Returns:
        QuantumCircuit: The loaded circuit.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"QASM file not found: {path}")
    return QuantumCircuit.from_qasm_file(str(path))


def iter_challenges(
    difficulty: str | None = None,
    func: Callable[[QuantumCircuit], object] | None = matrix_product_operators,
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> Iterator[tuple[str, object]]:
    """Yield ``(name, func(circuit))`` for each challenge QASM file.

    ``difficulty`` restricts to one subfolder (e.g. ``"very_easy"``); ``None`` walks
    every difficulty. ``func`` is applied to each loaded circuit and defaults to
    :func:`~quantum_hack.simulation.matrix_product_operators` (so you get the peak
    bitstring per circuit); pass ``func=None`` to yield the raw
    ``(name, QuantumCircuit)`` instead. ``name`` is
    ``"<difficulty>/<file stem>"`` so it stays unique across difficulties. Files are
    visited in sorted order for reproducibility.

    Args:
        difficulty: Difficulty subfolder to restrict to, or ``None`` for all.
        func: Applied to each loaded circuit; ``None`` yields the raw circuit.
        data_dir: Root directory holding the challenge QASM files.

    Yields:
        tuple[str, object]: ``(name, func(circuit))`` per QASM file, or
        ``(name, QuantumCircuit)`` when ``func`` is ``None``.
    """
    data_dir = Path(data_dir)
    search_root = data_dir if difficulty is None else data_dir / difficulty
    for path in sorted(search_root.rglob("*.qasm")):
        qc = load_circuit(path)
        name = f"{path.parent.name}/{path.stem}"
        yield name, (qc if func is None else func(qc))


def iter_challenges_gpu(
    difficulty: str | None = None,
    *,
    config: "GpuClusterConfig | None" = None,
    shots: int = 4096,
    bond_dim: int = 128,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    max_workers: int | None = None,
    verbose: bool = False,
) -> Iterator[tuple[str, tuple[str, float]]]:
    """Yield ``(name, (bitstring, prob))`` using the best available GPU backend.

    Circuits are processed sequentially using :func:`~quantum_hack.simulation.auto_simulation`,
    which selects GPU statevector, GPU MPS, CPU statevector, or CPU MPS per circuit
    based on qubit count and detected hardware.

    For throughput across many independent circuits on a multi-GPU node, the
    optional ``max_workers`` parameter enables :mod:`concurrent.futures` thread
    parallelism (one worker per GPU up to ``config.num_gpus``).

    Args:
        difficulty: Difficulty subfolder, or ``None`` for all.
        config: GPU cluster config.  ``None`` triggers auto-detection.
        shots: Shot count forwarded to MPS paths.
        bond_dim: Bond dimension forwarded to MPS paths.
        data_dir: Root directory holding QASM files.
        max_workers: Number of parallel workers.  ``None`` uses ``config.num_gpus``
            (falls back to 1 on CPU-only nodes).
        verbose: If ``True``, print per-circuit backend selection.

    Yields:
        tuple[str, tuple[str, float]]: ``(name, (peak_bitstring, probability))``.
    """
    from quantum_hack.gpu import GpuClusterConfig

    if config is None:
        config = GpuClusterConfig.auto_detect()

    if max_workers is None:
        max_workers = max(1, config.num_gpus)

    data_dir = Path(data_dir)
    search_root = data_dir if difficulty is None else data_dir / difficulty
    paths = sorted(search_root.rglob("*.qasm"))

    def _run(path: Path) -> tuple[str, tuple[str, float]]:
        qc = load_circuit(path)
        name = f"{path.parent.name}/{path.stem}"
        result = auto_simulation(
            qc,
            config=config,
            shots=shots,
            bond_dim=bond_dim,
            verbose=verbose,
        )
        return name, result

    if max_workers <= 1:
        for path in paths:
            yield _run(path)
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run, p): p for p in paths}
        for future in as_completed(futures):
            yield future.result()
