"""
Generate 5 UU† circuits sized to match the easy qasm challenge data.

Target sizes (UU† line count ≈ easy challenge files):
  8q  ~120  → challenge-8_11.qasm
  16q ~294  → challenge-16_12.qasm
  24q ~553  → challenge-24_13.qasm
  32q ~878  → challenge-32_14.qasm
  64q ~1857 → challenge-64_26.qasm
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dump

# (n_qubits, n_gates_in_U)
# UU† has 2*n_gates gates; QASM header = 3 lines, so total ≈ 3 + 2*n_gates
CONFIGS = [
    (8,  58),    # UUdag ≈ 119 lines
    (16, 145),   # UUdag ≈ 293 lines
    (24, 275),   # UUdag ≈ 553 lines
    (32, 437),   # UUdag ≈ 877 lines
    (64, 927),   # UUdag ≈ 1857 lines
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

for i, (n_qubits, n_gates) in enumerate(CONFIGS):
    u = random_circuit(n_qubits, n_gates)
    uu_dag = u.compose(u.inverse())

    fname = os.path.join(out_dir, f"uu_dag_{n_qubits}q_{i + 1}.qasm")
    with open(fname, "w") as f:
        dump(uu_dag, f)

    with open(fname) as f:
        lines = f.readlines()

    # save U figure
    fig_u = u.draw(output="mpl", fold=-1)
    fig_u.savefig(os.path.join(fig_dir, f"u_{n_qubits}q_{i + 1}.png"), dpi=72, bbox_inches="tight")
    fig_u.clf()

    # save UU† figure
    fig_uud = uu_dag.draw(output="mpl", fold=-1)
    fig_uud.savefig(os.path.join(fig_dir, f"uu_dag_{n_qubits}q_{i + 1}.png"), dpi=72, bbox_inches="tight")
    fig_uud.clf()

    # sanity check: transpile at opt level 3 should cancel most UU† gates
    transpiled = transpile(uu_dag, optimization_level=2, basis_gates=["rx", "rz", "cx"])
    remaining = transpiled.count_ops()
    total_remaining = sum(remaining.values())
    print(
        f"[{i + 1}/5] {n_qubits}q, U={n_gates} gates → {len(lines)} lines"
        f" | transpiled ops: {total_remaining} {dict(remaining)}"
    )

    fig_t = transpiled.draw(output="mpl", fold=-1)
    fig_t.savefig(os.path.join(fig_dir, f"transpiled_{n_qubits}q_{i + 1}.png"), dpi=72, bbox_inches="tight")
    fig_t.clf()
    

print("Done.")
