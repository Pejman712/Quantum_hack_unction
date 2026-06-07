"""
Generate 5 AUU†A circuits sized to match the easy qasm challenge data.

Structure: A · U · U† · A
After transpile: UU† cancels → A · A remains (2 * a_gates ops).

Target sizes (total gate lines ≈ easy challenge files):
  8q  ~116  → challenge-8_11.qasm
  16q ~290  → challenge-16_12.qasm
  24q ~550  → challenge-24_13.qasm
  32q ~874  → challenge-32_14.qasm
  64q ~1854 → challenge-64_26.qasm
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dump

# (n_qubits, a_gates, u_gates)
# total circuit gates = 2*a_gates + 2*u_gates
# transpiled remaining  = 2*a_gates
CONFIGS = [
    (8,  20,  38),   # total=116, transpiled=40
    (16, 50,  95),   # total=290, transpiled=100
    (24, 90, 185),   # total=550, transpiled=180
    (32, 140, 297),  # total=874, transpiled=280
    (64, 300, 627),  # total=1854, transpiled=600
]

RNG = np.random.default_rng(42)


def random_circuit(n_qubits: int, n_gates: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for _ in range(n_gates):
        gate_type = int(RNG.integers(0, 3))
        if gate_type == 0:
            q = int(RNG.integers(0, n_qubits))
            theta = float(RNG.uniform(-np.pi, np.pi))
            qc.rx(theta, q)
        elif gate_type == 1:
            q = int(RNG.integers(0, n_qubits))
            theta = float(RNG.uniform(-np.pi, np.pi))
            qc.rz(theta, q)
        else:
            pair = RNG.choice(n_qubits, size=2, replace=False)
            qc.cx(int(pair[0]), int(pair[1]))
    return qc


out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_qasm")
fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_figs")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

for i, (n_qubits, a_gates, u_gates) in enumerate(CONFIGS):
    a = random_circuit(n_qubits, a_gates)
    u = random_circuit(n_qubits, u_gates)

    # A · U · U† · A  (second A is not dagger)
    circuit = a.compose(u).compose(u.inverse()).compose(a)

    fname = os.path.join(out_dir, f"a_uu_dag_a_{n_qubits}q_{i + 1}.qasm")
    with open(fname, "w") as f:
        dump(circuit, f)

    with open(fname) as f:
        lines = f.readlines()

    fig_c = circuit.draw(output="mpl", fold=-1)
    fig_c.savefig(os.path.join(fig_dir, f"a_uu_dag_a_{n_qubits}q_{i + 1}.png"), dpi=72, bbox_inches="tight")
    fig_c.clf()

    transpiled = transpile(circuit, optimization_level=1, basis_gates=["rx", "rz", "cx"])
    remaining = transpiled.count_ops()
    total_remaining = sum(remaining.values())

    fig_t = transpiled.draw(output="mpl", fold=-1)
    fig_t.savefig(os.path.join(fig_dir, f"transpiled_a_uu_dag_a_{n_qubits}q_{i + 1}.png"), dpi=72, bbox_inches="tight")
    fig_t.clf()

    print(
        f"[{i + 1}/5] {n_qubits}q, A={a_gates} U={u_gates} → {len(lines)} lines"
        f" | transpiled ops: {total_remaining} (expected ~2*A={2 * a_gates}) {dict(remaining)}"
    )

print("Done.")
