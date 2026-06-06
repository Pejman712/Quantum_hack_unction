"""Loading challenge circuits from the qasm_data folder and batch-running over them."""

from collections.abc import Callable, Iterator
from pathlib import Path

from qiskit import QuantumCircuit

from quantum_hack.simulation import matrix_product_operators

DEFAULT_DATA_DIR = Path("qasm_data")
DIFFICULTIES = ("very_easy", "easy", "moderate", "hard", "very_hard")


def load_circuit(path: Path | str) -> QuantumCircuit:
    """Load a circuit from an OpenQASM file, with a clear error if it is missing.

    Args:
        path (Path | str): Path to the OpenQASM file.

    Returns:
        QuantumCircuit: The loaded circuit.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"QASM file not found: {path}")
    return QuantumCircuit.from_qasm_file(str(path))


def find_challenge(task: str, *, data_dir: Path | str = DEFAULT_DATA_DIR) -> Path:
    """Locate the QASM file for a task within the challenge data tree.

    Searches ``data_dir`` recursively so the difficulty subfolder (e.g.
    ``very_easy``) does not need to be known up front.

    Args:
        task (str): The challenge name without extension (e.g. ``"challenge-8_1"``).
        data_dir (Path | str): Root directory holding the challenge QASM files.

    Returns:
        Path: Absolute-or-relative path to the matching ``<task>.qasm`` file.

    Raises:
        FileNotFoundError: If no file, or more than one file, matches ``task``.
    """
    data_dir = Path(data_dir)
    matches = sorted(data_dir.rglob(f"{task}.qasm"))
    if not matches:
        raise FileNotFoundError(f"No QASM file named {task!r}.qasm under {data_dir}")
    if len(matches) > 1:
        joined = ", ".join(str(m) for m in matches)
        raise FileNotFoundError(f"Multiple QASM files match {task!r}: {joined}")
    return matches[0]


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
        difficulty (str | None): Difficulty subfolder to restrict to, or ``None`` for all.
        func (Callable[[QuantumCircuit], object] | None): Applied to each loaded circuit;
            ``None`` yields the raw circuit.
        data_dir (Path | str): Root directory holding the challenge QASM files.

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
