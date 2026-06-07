"""
Generate 5 UU†U†U circuits sized to match the easy qasm challenge data.

Structure: U · U† · U† · U
After transpile: first UU† cancels, then U†U cancels → identity (0 ops).

Total gate lines = 4 * u_gates, so u_gates ≈ (target_lines - 3) / 4.

Target sizes (total lines ≈ easy challenge files):
  8q  ~119  → challenge-8_11.qasm  (120)
  16q ~293  → challenge-16_12.qasm (294)
  24q ~551  → challenge-24_13.qasm (553)
  32q ~879  → challenge-32_14.qasm (878)
  64q ~1855 → challenge-64_26.qasm (1857)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dump

# (n_qubits, u_gates)  — total lines ≈ 3 + 4*u_gates
CONFIGS = [
    (8,  29),   # 3 + 4*29  = 119
    (16, 73),   # 3 + 4*73  = 295
    (24, 137),  # 3 + 4*137 = 551
    (32, 219),  # 3 + 4*219 = 879
    (64, 463),  # 3 + 4*463 = 1855
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

for i, (n_qubits, u_gates) in enumerate(CONFIGS):
    u = random_circuit(n_qubits, u_gates)
    u_dag = u.inverse()

    # U · U† · U† · U
    circuit = u.compose(u_dag).compose(u_dag).compose(u)

    fname = os.path.join(out_dir, f"uu_dag_udag_u_{n_qubits}q_{i + 1}.qasm")
    with open(fname, "w") as f:
        dump(circuit, f)

    with open(fname) as f:
        lines = f.readlines()

    fig_c = circuit.draw(output="mpl", fold=-1)
    fig_c.savefig(os.path.join(fig_dir, f"uu_dag_udag_u_{n_qubits}q_{i + 1}.png"), dpi=72, bbox_inches="tight")
    fig_c.clf()

    transpiled = transpile(circuit, optimization_level=2, basis_gates=["rx", "rz", "cx"])
    remaining = transpiled.count_ops()
    total_remaining = sum(remaining.values())

    fig_t = transpiled.draw(output="mpl", fold=-1)
    fig_t.savefig(os.path.join(fig_dir, f"transpiled_uu_dag_udag_u_{n_qubits}q_{i + 1}.png"), dpi=72, bbox_inches="tight")
    fig_t.clf()

    print(
        f"[{i + 1}/5] {n_qubits}q, U={u_gates} gates → {len(lines)} lines"
        f" | transpiled ops: {total_remaining} (expected 0) {dict(remaining)}"
    )

print("Done.")
