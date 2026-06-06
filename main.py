"""Scratch entry point for experimenting with a single challenge circuit.

Run with: uv run python main.py
For the full CLI (diagram rendering, arbitrary paths) use: uv run quantum-hack --help
"""

from pathlib import Path

from quantum_hack import (
    circuit_stats,
    load_circuit,
    matrix_product_operators,
    save_circuit_drawing,
    snap_rotations_to_pi_over_4,
    statevector_simulation,
    strip_rz_and_cx_from_start,
    transpile_to_basis,
)

# Anchor to this file's directory so the script runs from any working directory.
HERE = Path(__file__).resolve().parent
QASM_PATH = HERE / "qasm_data" / "very_easy" / "challenge-8_1.qasm"


def main() -> None:
    qc = load_circuit(QASM_PATH)
    # Flatten out any macros so stats match the drawing.
    qc = transpile_to_basis(qc, optimization_level=0)

    print(f"Circuit: {QASM_PATH}")
    print("Stats:", circuit_stats(qc))

    bitstring, prob = statevector_simulation(qc)
    print("Exact peak bitstring (statevector):", bitstring, f"(probability {prob:.2%})")
    bitstring, prob = matrix_product_operators(qc)
    print("Estimated peak bitstring (MPS):    ", bitstring, f"(probability {prob:.2%})")

    path = save_circuit_drawing(
        qc, "challenge-8_1_before", out_dir=HERE / "circuit_drawings", timestamp=False
    )
    print("Saved drawing:", path)

    clean_qc = strip_rz_and_cx_from_start(qc)

    print("\nStats after stripping Rz and Cx from start:", circuit_stats(clean_qc))

    bitstring, prob = statevector_simulation(clean_qc)
    print("Exact peak bitstring (statevector):", bitstring, f"(probability {prob:.2%})")
    bitstring, prob = matrix_product_operators(clean_qc)
    print("Estimated peak bitstring (MPS):    ", bitstring, f"(probability {prob:.2%})")

    path = save_circuit_drawing(
        clean_qc, "challenge-8_1_clean", out_dir=HERE / "circuit_drawings", timestamp=False
    )
    print("Saved drawing:", path)

    snapped_qc = snap_rotations_to_pi_over_4(clean_qc, threshold=0.02)
    snapped_qc = transpile_to_basis(snapped_qc, optimization_level=2)

    print("\nStats after snapping and transpilation:", circuit_stats(snapped_qc))

    bitstring, prob = statevector_simulation(snapped_qc)
    print("Exact peak bitstring (statevector):", bitstring, f"(probability {prob:.2%})")
    bitstring, prob = matrix_product_operators(snapped_qc)
    print("Estimated peak bitstring (MPS):    ", bitstring, f"(probability {prob:.2%})")

    path = save_circuit_drawing(
        snapped_qc, "challenge-8_1_after", out_dir=HERE / "circuit_drawings", timestamp=False
    )
    print("Saved drawing:", path)


if __name__ == "__main__":
    main()
