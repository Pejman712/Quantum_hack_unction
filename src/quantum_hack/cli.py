"""Command-line entry point: load a QASM circuit and render its diagrams."""

import argparse
from pathlib import Path

from qiskit import QuantumCircuit

from quantum_hack.transform import transpile_to_basis
from quantum_hack.viz import DEFAULT_OUTPUT_DIR, save_circuit_drawing

DEFAULT_QASM = Path("qasm_data/very_easy/challenge-8_1.qasm")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the CLI.

    Returns:
        argparse.ArgumentParser: Parser configured with the ``qasm``, ``--out-dir``,
        and ``--no-timestamp`` arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "qasm",
        nargs="?",
        type=Path,
        default=DEFAULT_QASM,
        help=f"Path to the OpenQASM file (default: {DEFAULT_QASM}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write the circuit PNGs into (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_false",
        dest="timestamp",
        help="Use stable filenames instead of timestamped ones.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    qc = QuantumCircuit.from_qasm_file(str(args.qasm))
    transpiled_qc = transpile_to_basis(qc, optimization_level=0)
    optimized_qc = transpile_to_basis(qc, optimization_level=2)

    for circuit, name in (
        (qc, "circuit"),
        (transpiled_qc, "transpiled_circuit"),
        (optimized_qc, "optimized_circuit"),
    ):
        path = save_circuit_drawing(
            circuit, name, out_dir=args.out_dir, timestamp=args.timestamp
        )
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
