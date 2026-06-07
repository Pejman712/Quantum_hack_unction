"""
Angle attack on UU†-structured peaked circuits.

Physics
-------
C = U1 U1† P1 U2 U2† P2 U3 U3†
C|0⟩ = P2 P1 |0⟩   (each UiUi† = I, so they vanish)

P1 applies RX(a_i) and P2 applies RX(π·bit_i − a_i) per qubit i.
Combined per-qubit rotation: RX(π·bit_i).
The output is *close* to a product state but not exactly one, because the
angle sweep perturbs individual gate angles by ~0.01 rad, breaking the exact
UiUi† cancellation.  This creates small but nonzero entanglement.

Why bond_dim=1 fails
---------------------
Bond dimension 1 can only represent a perfect product state.  The perturbed
circuit output has small entanglement, so bond_dim=1 gives wrong marginals.
Bond_dim=4 is empirically sufficient for sweep_std ≤ 0.02.

Why raw RX-angle summing fails
-------------------------------
After transpile(optimization_level=1), P-layer RX gates are merged into
adjacent U-block single-qubit gates.  CX gates in the U blocks do NOT let
U/U† single-qubit angles cancel qubit-by-qubit — they only cancel at the
full-unitary level.  So there is no gate-level angle to extract.

What actually works
-------------------
1. statevector_attack  — exact, O(2^n). Feasible for n ≤ ~20.
2. marginal_attack     — per-qubit P(|1⟩) via MPS with bond_dim ≥ 4.
                         Marginals of the peaked state are ≈1 for bit_i=1
                         and ≈0 for bit_i=0, regardless of background noise.
                         Bond_dim=4 correct for sweep_std ≤ 0.02.

Usage
-----
    from obs_gen import make_peaked
    from angle_attack import statevector_attack, marginal_attack, compare_attacks

    qc, secret = make_peaked(n=8, depth=3, seed=0)
    print(statevector_attack(qc))         # exact (small n)
    print(marginal_attack(qc))            # MPS marginals (scales to larger n)
    compare_attacks(qc, secret)
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


# ═══════════════════════════════════════════════════════════════════════════
# Attack 1: exact statevector — works for n ≤ ~20
# ═══════════════════════════════════════════════════════════════════════════

def statevector_attack(
    circuit: QuantumCircuit,
    prob_threshold: float = 0.5,
) -> str:
    """
    Extract the peak bitstring from the exact statevector.

    Computes per-qubit marginal P(|1⟩) exactly.  For a state peaked at
    |x⟩ with probability p, marginal for qubit i ≈ p + (1−p)/2 ≫ 0.5
    when bit_i=1, and ≈ (1−p)/2 ≪ 0.5 when bit_i=0.

    Args:
        circuit:        Circuit to attack (measurements stripped if present).
        prob_threshold: P(|1⟩) > threshold → bit = 1.

    Returns:
        Bitstring with q0 as rightmost bit (challenge convention).

    Raises:
        MemoryError: if n > ~22 (2^n statevector too large).
    """
    n = circuit.num_qubits
    qc = circuit.remove_final_measurements(inplace=False)

    probs = Statevector(qc).probabilities_dict()

    marginals = np.zeros(n)
    for bitstr, p in probs.items():
        for pos, ch in enumerate(bitstr):
            q = n - 1 - pos            # Qiskit: leftmost char = highest qubit
            if ch == "1":
                marginals[q] += p

    bits = marginals > prob_threshold
    return "".join(str(int(b)) for b in bits[::-1])   # q[n-1]...q[0]


# ═══════════════════════════════════════════════════════════════════════════
# Attack 2: MPS marginals — scales to larger n
# ═══════════════════════════════════════════════════════════════════════════

def marginal_attack(
    circuit: QuantumCircuit,
    shots: int = 4096,
    bond_dim: int = 16,
    prob_threshold: float = 0.5,
) -> str:
    """
    Extract the peak bitstring via per-qubit marginal probabilities from MPS.

    Runs the circuit on Aer's MPS simulator.  Since C|0⟩ is highly peaked
    at the secret bitstring, per-qubit marginal P(q_i=1) is close to 1 for
    bit_i=1 and close to 0 for bit_i=0.

    Bond dimension: bond_dim=4 is empirically sufficient for sweep_std≤0.02.
    Use bond_dim=16 for a safe margin.  Higher values slow simulation but do
    not change the result once the peak is resolved.

    Args:
        circuit:        Circuit to attack.
        shots:          Number of MPS samples.
        bond_dim:       MPS max bond dimension (4 sufficient; 16 safe default).
        prob_threshold: P(|1⟩) > threshold → bit = 1.

    Returns:
        Bitstring with q0 as rightmost bit (challenge convention).
    """
    n = circuit.num_qubits
    qc = circuit.remove_final_measurements(inplace=False)
    qc.measure_all()

    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=bond_dim,
    )
    qc_t = transpile(qc, basis_gates=["rz", "rx", "cx", "swap", "measure"],
                     optimization_level=0)
    counts = sim.run(qc_t, shots=shots).result().get_counts()

    marginals = np.zeros(n)
    total = sum(counts.values())
    for bitstr, cnt in counts.items():
        for pos, ch in enumerate(bitstr):
            q = n - 1 - pos            # Qiskit: leftmost char = highest qubit
            if ch == "1":
                marginals[q] += cnt / total

    bits = marginals > prob_threshold
    return "".join(str(int(b)) for b in bits[::-1])   # q[n-1]...q[0]


# ═══════════════════════════════════════════════════════════════════════════
# Comparison helper
# ═══════════════════════════════════════════════════════════════════════════

def compare_attacks(circuit: QuantumCircuit, secret: str) -> dict[str, dict]:
    """
    Run both attacks and report accuracy vs planted secret.

    Args:
        circuit: Obfuscated peaked circuit.
        secret:  Planted bitstring (q0 rightmost, from make_peaked).

    Returns:
        Dict with keys 'statevector' and 'marginal', each containing
        {'predicted': str, 'correct': bool, 'bit_errors': int}.
    """
    attacks = {"statevector": statevector_attack, "marginal": marginal_attack}
    results = {}
    for name, fn in attacks.items():
        predicted = fn(circuit)
        errors = sum(a != b for a, b in zip(predicted, secret))
        results[name] = {
            "predicted": predicted,
            "correct": predicted == secret,
            "bit_errors": errors,
        }
        status = "✓" if predicted == secret else f"✗ ({errors} bit errors)"
        print(f"[{name:12s}]  predicted={predicted}  secret={secret}  {status}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "obs_gen", Path(__file__).parent / "obs_gen.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    make_peaked = mod.make_peaked

    print("=== angle_attack self-test ===\n")
    all_ok = True
    for n, depth, seed in [(4, 2, 0), (6, 2, 1), (8, 3, 0), (8, 3, 7), (10, 3, 42)]:
        print(f"n={n}, depth={depth}, seed={seed}")
        qc, secret = make_peaked(n=n, depth=depth, seed=seed)
        r = compare_attacks(qc, secret)
        all_ok = all_ok and all(v["correct"] for v in r.values())
        print()

    print("All correct:", all_ok)
