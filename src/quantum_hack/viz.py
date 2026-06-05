"""Helpers for rendering circuit diagrams to an output folder."""

from datetime import datetime
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

DEFAULT_OUTPUT_DIR = Path("circuit_drawings")
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"  # sortable: lexical order == chronological order


def _timestamped(name: str) -> str:
    """Append a sortable ``YYYYMMDD_HHMMSS`` timestamp to a name.

    Args:
        name (str): Base name to suffix.

    Returns:
        str: The name with a ``_<timestamp>`` suffix.
    """
    return f"{name}_{datetime.now().strftime(TIMESTAMP_FORMAT)}"


def _resolve_path(out_dir: Path | str, name: str, timestamp: bool, image_format: str) -> Path:
    """Resolve the output file path, creating the directory if needed.

    Args:
        out_dir (Path | str): Directory to write into (created if missing).
        name (str): Base filename (without extension).
        timestamp (bool): If ``True``, suffix the name with a timestamp.
        image_format (str): Image file extension/format (e.g. ``"png"``).

    Returns:
        Path: The resolved ``<out_dir>/<name>.<image_format>`` path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _timestamped(name) if timestamp else name
    return out_dir / f"{stem}.{image_format}"


def save_circuit_drawing(
    qc: QuantumCircuit,
    name: str = "circuit",
    *,
    out_dir: Path | str = DEFAULT_OUTPUT_DIR,
    timestamp: bool = True,
    image_format: str = "png",
    **draw_kwargs: object,
) -> Path:
    """Draw ``qc`` with the matplotlib drawer and save it under ``out_dir``.

    The output directory is created if needed. By default the filename is suffixed
    with a ``YYYYMMDD_HHMMSS`` timestamp so repeated runs sort chronologically; pass
    ``timestamp=False`` for a stable ``<name>.<ext>`` filename. Extra keyword
    arguments are forwarded to :meth:`QuantumCircuit.draw`. Returns the path written.

    Args:
        qc (QuantumCircuit): Circuit to draw.
        name (str): Base filename (without extension).
        out_dir (Path | str): Directory to write the image into.
        timestamp (bool): If ``True``, suffix the filename with a timestamp.
        image_format (str): Image file extension/format (e.g. ``"png"``).
        **draw_kwargs (object): Extra keyword arguments forwarded to :meth:`QuantumCircuit.draw`.

    Returns:
        Path: The path of the written image.
    """
    path = _resolve_path(out_dir, name, timestamp, image_format)
    measure_free = qc.remove_final_measurements(inplace=False)
    measure_free.draw("mpl", filename=str(path), **draw_kwargs)
    return path


def cx_only_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of ``qc`` containing only its CX gates.

    All single-qubit and non-CX operations (and any final measurements) are
    dropped, leaving just the two-qubit entangling structure on the same number
    of qubits. Useful for inspecting a circuit's connectivity in isolation.

    Args:
        qc (QuantumCircuit): Source circuit.

    Returns:
        QuantumCircuit: A new circuit containing only the CX gates of ``qc``.
    """
    cx_only = QuantumCircuit(qc.num_qubits)
    for instruction in qc.data:
        if instruction.operation.name == "cx":
            indices = [qc.find_bit(q).index for q in instruction.qubits]
            cx_only.cx(*indices)
    return cx_only


def save_cx_drawing(
    qc: QuantumCircuit,
    name: str = "circuit_cx",
    *,
    out_dir: Path | str = DEFAULT_OUTPUT_DIR,
    timestamp: bool = True,
    image_format: str = "png",
    **draw_kwargs: object,
) -> Path:
    """Draw only the CX gates of ``qc`` and save it under ``out_dir``.

    Builds the CX-only circuit via :func:`cx_only_circuit`, then renders it with
    the same folder/timestamp conventions as :func:`save_circuit_drawing`. Extra
    keyword arguments are forwarded to :meth:`QuantumCircuit.draw`. Returns the
    path written.

    Args:
        qc (QuantumCircuit): Circuit whose CX gates to draw.
        name (str): Base filename (without extension).
        out_dir (Path | str): Directory to write the image into.
        timestamp (bool): If ``True``, suffix the filename with a timestamp.
        image_format (str): Image file extension/format (e.g. ``"png"``).
        **draw_kwargs (object): Extra keyword arguments forwarded to :meth:`QuantumCircuit.draw`.

    Returns:
        Path: The path of the written image.
    """
    return save_circuit_drawing(
        cx_only_circuit(qc),
        name,
        out_dir=out_dir,
        timestamp=timestamp,
        image_format=image_format,
        **draw_kwargs,
    )


def save_counts_histogram(
    counts: dict,
    name: str = "histogram",
    *,
    out_dir: Path | str = DEFAULT_OUTPUT_DIR,
    timestamp: bool = True,
    image_format: str = "png",
    **plot_kwargs: object,
) -> Path:
    """Plot measurement ``counts`` as a histogram and save it under ``out_dir``.

    Mirrors :func:`save_circuit_drawing` (same folder/timestamp conventions). Extra
    keyword arguments are forwarded to :func:`qiskit.visualization.plot_histogram`.
    Returns the path written.

    Args:
        counts (dict): Measurement counts keyed by bitstring.
        name (str): Base filename (without extension).
        out_dir (Path | str): Directory to write the image into.
        timestamp (bool): If ``True``, suffix the filename with a timestamp.
        image_format (str): Image file extension/format (e.g. ``"png"``).
        **plot_kwargs (object): Extra keyword arguments forwarded to
            :func:`qiskit.visualization.plot_histogram`.

    Returns:
        Path: The path of the written image.
    """
    path = _resolve_path(out_dir, name, timestamp, image_format)
    fig = plot_histogram(counts, **plot_kwargs)
    fig.savefig(str(path), bbox_inches="tight")
    return path
