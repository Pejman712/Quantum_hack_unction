"""Command-line entry point: load QASM circuit(s), print peak bitstrings, and render diagrams."""

import argparse
from pathlib import Path

from qiskit import QuantumCircuit

from quantum_hack.simulation import matrix_product_operators, statevector_simulation
from quantum_hack.transform import transpile_to_basis
from quantum_hack.viz import DEFAULT_OUTPUT_DIR, save_circuit_drawing

DEFAULT_QASM = Path("qasm_data/very_easy/challenge-8_1.qasm")
_SV_QUBIT_LIMIT = 20


def _peak_bitstring(qc: QuantumCircuit) -> tuple[str, float]:
    """Use statevector for small circuits, MPS otherwise."""
    if qc.num_qubits <= _SV_QUBIT_LIMIT:
        return statevector_simulation(qc)
    return matrix_product_operators(qc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "qasm",
        nargs="?",
        type=Path,
        default=DEFAULT_QASM,
        help="Path to a QASM file or a directory of QASM files (default: %(default)s).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write circuit PNGs into (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_false",
        dest="timestamp",
        help="Use stable filenames instead of timestamped ones.",
    )
    return parser


def _process_file(qasm_path: Path, out_dir: Path, timestamp: bool) -> None:
    qc = QuantumCircuit.from_qasm_file(str(qasm_path))
    transpiled_qc = transpile_to_basis(qc, optimization_level=0)
    optimized_qc = transpile_to_basis(qc, optimization_level=2)

    bitstring, prob = _peak_bitstring(transpiled_qc)
    method = "statevector" if qc.num_qubits <= _SV_QUBIT_LIMIT else "MPS"
    print(f"Peak bitstring ({method}): {bitstring}  ({prob:.2%})")

    for circuit, name in (
        (qc, "circuit"),
        (transpiled_qc, "transpiled_circuit"),
        (optimized_qc, "optimized_circuit"),
    ):
        path = save_circuit_drawing(circuit, name, out_dir=out_dir, timestamp=timestamp)
        print(f"Saved {path}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    target: Path = args.qasm

    if target.is_dir():
        files = sorted(target.glob("*.qasm"))
        if not files:
            print(f"No .qasm files found in {target}")
            return
        print(f"{'File':<30} {'Qubits':>6}  {'Method':<12} {'Bitstring':<70} {'Prob':>6}")
        print("-" * 130)
        for qasm_path in files:
            qc = QuantumCircuit.from_qasm_file(str(qasm_path))
            qc_t = transpile_to_basis(qc, optimization_level=0)
            bitstring, prob = _peak_bitstring(qc_t)
            method = "statevector" if qc.num_qubits <= _SV_QUBIT_LIMIT else "MPS"
            print(f"{qasm_path.stem:<30} {qc.num_qubits:>6}  {method:<12} {bitstring:<70} {prob:>5.1%}")
    else:
        print(f"Circuit: {target}")
        _process_file(target, args.out_dir, args.timestamp)


if __name__ == "__main__":
    main()
