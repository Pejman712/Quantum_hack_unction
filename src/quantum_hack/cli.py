"""Command-line entry point: load QASM circuit(s), print peak bitstrings, and render diagrams."""

from __future__ import annotations

import argparse
from pathlib import Path

from qiskit import QuantumCircuit

from quantum_hack.simulation import (
    auto_simulation,
    matrix_product_operators,
    statevector_simulation,
)
from quantum_hack.transform import transpile_to_basis
from quantum_hack.viz import DEFAULT_OUTPUT_DIR, save_circuit_drawing

DEFAULT_QASM = Path("qasm_data/very_easy/challenge-8_1.qasm")
_SV_QUBIT_LIMIT = 20


def _peak_bitstring_cpu(qc: QuantumCircuit) -> tuple[str, float]:
    """Use CPU statevector for small circuits, CPU MPS otherwise."""
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

    # GPU cluster options
    gpu_group = parser.add_argument_group("GPU cluster options")
    gpu_group.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Enable GPU-accelerated simulation (auto-detects available GPUs).",
    )
    gpu_group.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        metavar="N",
        help="Override the number of GPUs to use (default: auto-detect).",
    )
    gpu_group.add_argument(
        "--blocking-qubits",
        type=int,
        default=None,
        metavar="N",
        help=(
            "log2 chunk size for multi-GPU blocking mode "
            "(default: auto, typically 21-23 depending on GPU count)."
        ),
    )
    gpu_group.add_argument(
        "--no-custatevec",
        action="store_false",
        dest="custatevec",
        default=True,
        help="Disable cuStateVec even when a GPU is present.",
    )
    gpu_group.add_argument(
        "--gpu-info",
        action="store_true",
        default=False,
        help="Print detected GPU cluster information and exit.",
    )

    return parser


def _build_gpu_config(args: argparse.Namespace) -> "GpuClusterConfig":
    """Build a GpuClusterConfig from parsed CLI arguments."""
    from quantum_hack.gpu import GpuClusterConfig, count_gpus, _optimal_blocking_qubits

    if args.num_gpus is not None:
        n = args.num_gpus
    else:
        n = count_gpus()

    bq = args.blocking_qubits if args.blocking_qubits is not None else _optimal_blocking_qubits(n)

    return GpuClusterConfig(
        num_gpus=n,
        blocking_qubits=bq,
        custatevec_enable=args.custatevec,
    )


def _process_file(
    qasm_path: Path,
    out_dir: Path,
    timestamp: bool,
    gpu_config: "GpuClusterConfig | None",
) -> None:
    qc = QuantumCircuit.from_qasm_file(str(qasm_path))
    transpiled_qc = transpile_to_basis(qc, optimization_level=0)
    optimized_qc = transpile_to_basis(qc, optimization_level=2)

    if gpu_config is not None:
        bitstring, prob = auto_simulation(transpiled_qc, config=gpu_config)
        method = f"GPU({gpu_config})" if gpu_config.has_gpu else "CPU-auto"
    else:
        bitstring, prob = _peak_bitstring_cpu(transpiled_qc)
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

    if args.gpu_info:
        from quantum_hack.gpu import cluster_info, GpuClusterConfig
        info = cluster_info()
        cfg = GpuClusterConfig.auto_detect()
        print(f"GPUs detected  : {info['num_gpus']}")
        print(f"MPI world size : {info['mpi_world_size']}")
        print(f"Auto config    : {cfg}")
        return

    gpu_config = _build_gpu_config(args) if args.gpu else None

    target: Path = args.qasm

    if target.is_dir():
        files = sorted(target.glob("*.qasm"))
        if not files:
            print(f"No .qasm files found in {target}")
            return
        print(f"{'File':<30} {'Qubits':>6}  {'Method':<20} {'Bitstring':<70} {'Prob':>6}")
        print("-" * 138)
        for qasm_path in files:
            qc = QuantumCircuit.from_qasm_file(str(qasm_path))
            qc_t = transpile_to_basis(qc, optimization_level=0)
            if gpu_config is not None:
                bitstring, prob = auto_simulation(qc_t, config=gpu_config)
                method = "GPU-SV" if (gpu_config.has_gpu and qc.num_qubits <= _SV_QUBIT_LIMIT) else "GPU-MPS"
                if not gpu_config.has_gpu:
                    method = "CPU-SV" if qc.num_qubits <= _SV_QUBIT_LIMIT else "CPU-MPS"
            else:
                bitstring, prob = _peak_bitstring_cpu(qc_t)
                method = "statevector" if qc.num_qubits <= _SV_QUBIT_LIMIT else "MPS"
            print(f"{qasm_path.stem:<30} {qc.num_qubits:>6}  {method:<20} {bitstring:<70} {prob:>5.1%}")
    else:
        print(f"Circuit: {target}")
        _process_file(target, args.out_dir, args.timestamp, gpu_config)


if __name__ == "__main__":
    main()
