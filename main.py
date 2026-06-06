"""Scratch entry point for experimenting with a single challenge circuit.

Run with: uv run python main.py
For the full CLI (diagram rendering, arbitrary paths) use: uv run quantum-hack --help
"""

from pathlib import Path

from quantum_hack import (
    circuit_stats,
    find_challenge,
    load_circuit,
    matrix_product_operators,
    save_circuit_drawing,
    snap_rotations_to_pi_over_4,
    statevector_simulation,
    strip,
    transpile_to_basis,
)

# The challenge to run. The QASM file and drawing directory are detected from this.
TASK = "challenge-8_1"

# Anchor to this file's directory so the script runs from any working directory.
HERE = Path(__file__).resolve().parent
DRAWINGS_DIR = HERE / "circuit_drawings"


def main() -> None:
    QASM_PATH = find_challenge(TASK, data_dir=HERE / "qasm_data")
    # Mirror the difficulty subfolder (e.g. "easy") in the drawings directory.
    difficulty = QASM_PATH.parent.name
    drawings_dir = DRAWINGS_DIR / difficulty
    qc = load_circuit(QASM_PATH)
    # Flatten out any macros so stats match the drawing.
    qc = transpile_to_basis(qc, basis_gates=["rx", "rz", "cx", "swap"])

    print(f"Circuit: {QASM_PATH}")
    print("Stats:", circuit_stats(qc))

    bitstring, prob = statevector_simulation(qc)
    print("Exact peak bitstring (statevector):", bitstring, f"(probability {prob:.2%})")
    bitstring, prob = matrix_product_operators(qc)
    print("Estimated peak bitstring (MPS):    ", bitstring, f"(probability {prob:.2%})")

    path = save_circuit_drawing(
        qc, f"{TASK}_before", out_dir=drawings_dir, timestamp=False
    )
    print("Saved drawing:", path)

    clean_qc = strip(qc)

    print("\nStats after stripping Rz and Cx from start:", circuit_stats(clean_qc))

    bitstring, prob = statevector_simulation(clean_qc)
    print("Exact peak bitstring (statevector):", bitstring, f"(probability {prob:.2%})")
    bitstring, prob = matrix_product_operators(clean_qc)
    print("Estimated peak bitstring (MPS):    ", bitstring, f"(probability {prob:.2%})")

    path = save_circuit_drawing(
        clean_qc, f"{TASK}_clean", out_dir=drawings_dir, timestamp=False
    )
    print("Saved drawing:", path)

    snapped_qc = snap_rotations_to_pi_over_4(clean_qc, threshold=0.03, snap_rz_to_zero=True)
    snapped_qc = transpile_to_basis(snapped_qc, optimization_level=2)

    path = save_circuit_drawing(
        snapped_qc, f"{TASK}_snapped", out_dir=drawings_dir, timestamp=False
    )

    print("\nStats after snapping and transpilation:", circuit_stats(snapped_qc))

    bitstring, prob = statevector_simulation(snapped_qc)
    print("Exact peak bitstring (statevector):", bitstring, f"(probability {prob:.2%})")
    bitstring, prob = matrix_product_operators(snapped_qc)
    print("Estimated peak bitstring (MPS):    ", bitstring, f"(probability {prob:.2%})")

    final_qc = strip(snapped_qc)

    print("\nStats after final stripping:", circuit_stats(final_qc))

    bitstring, prob = statevector_simulation(final_qc)
    print("Exact peak bitstring (statevector):", bitstring, f"(probability {prob:.2%})")
    bitstring, prob = matrix_product_operators(final_qc)
    print("Estimated peak bitstring (MPS):    ", bitstring, f"(probability {prob:.2%})")

    path = save_circuit_drawing(
        final_qc, f"{TASK}_after", out_dir=drawings_dir, timestamp=False
    )
    print("Saved drawing:", path)


if __name__ == "__main__":
    main()
