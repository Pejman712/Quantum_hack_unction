"""Scratch entry point for experimenting with a single challenge circuit.

Run with: uv run python main.py
For the full CLI (diagram rendering, arbitrary paths) use: uv run quantum-hack --help
"""

from pathlib import Path

from quantum_hack import (
    challenge_status,
    circuit_stats,
    find_challenge,
    load_circuit,
    matrix_product_operators,
    snap_rotations_to_pi_over_4,
    transpile_to_basis,
    weighted_majority_bitstring,
    strip,
)

# The challenge to run. The QASM file and drawing directory are detected from this.
TASK = "challenge-64_26"

# Anchor to this file's directory so the script runs from any working directory.
HERE = Path(__file__).resolve().parent


def main() -> None:
    QASM_PATH = find_challenge(TASK, data_dir=HERE / "qasm_data")
    qc = load_circuit(QASM_PATH)
    # Flatten out any macros so stats match the drawing.
    qc = transpile_to_basis(qc, basis_gates=["rx", "rz", "cx", "swap"])

    print(f"Circuit: {QASM_PATH}")
    print("Stats:", circuit_stats(qc))

    status = challenge_status(TASK, results_dir=HERE / "results")
    if status.worked:
        # Challenge already finished: reuse the accepted bitstring instead of recomputing.
        best = max(status.successes, key=lambda r: r.probability or 0.0)
        prob_str = "n/a" if best.probability is None else f"{best.probability:.2%}"
        print(f"Exact peak bitstring (from results): {best.bitstring} (probability {prob_str})")

    bitstring, prob = matrix_product_operators(qc)
    print("Estimated peak bitstring (MPS):    ", bitstring, f"(probability {prob:.2%})")

    clean_qc = strip(qc)

    print("\nStats after stripping Rz and Cx from start:", circuit_stats(clean_qc))

    bitstring, prob = matrix_product_operators(clean_qc)
    print("Estimated peak bitstring (MPS):    ", bitstring, f"(probability {prob:.2%})")

    snapped_qc = snap_rotations_to_pi_over_4(clean_qc, threshold=0.03, snap_rz_to_zero=True)
    snapped_qc = transpile_to_basis(snapped_qc, optimization_level=2)

    print("\nStats after snapping and transpilation:", circuit_stats(snapped_qc))

    bitstring, prob = matrix_product_operators(snapped_qc)
    print("Estimated peak bitstring (MPS):    ", bitstring, f"(probability {prob:.2%})")

    final_qc = strip(snapped_qc)

    print("\nStats after final stripping:", circuit_stats(final_qc))

    result_list = matrix_product_operators(final_qc, top_n=10, verbose=True)
    bitstring = weighted_majority_bitstring(result_list)
    print("weighted bitstring majority:")
    print(bitstring)


if __name__ == "__main__":
    main()
