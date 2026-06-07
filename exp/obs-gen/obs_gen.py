#!/usr/bin/env python3
"""
Transparent QMill-style peaked-circuit generator / obfuscator -- a KNOWN-ANSWER
TEST HARNESS, not a copy of QMill's proprietary engine.

Builds the documented skeleton  C = U1 U1d P1 U2 U2d P2 U3 U3d  (three local
UU-dagger identity mirrors; P1,P2 are RX layers merging to the RX(pi/0) column that
encodes secret x), then applies the documented obfuscation: identity-block padding,
the swap-trick (answer-preserving permutation conjugation of each UUd block),
angle-sweep (small drift of 1q angles, off the Clifford lattice), and recompile to
rz/rx/cx to blur boundaries.

Use it to generate circuits WITH the planted answer, then check your unswapping /
Path-B attack recovers it -> direct evidence the method generalizes (Summary points).

  python peaked_gen.py --n 32 --depth 6 --out mytest.qasm --seed 7
  # writes mytest.qasm and mytest.answer.txt ; verifies exactly if --n <= 14

Difficulty knobs:
  --depth      deeper U blocks  -> more entanglement / harder
  --swap-k     more swaps       -> more permutation obfuscation (stresses unswapping)
  --sweep-std  larger angle drift -> lower peak probability (harder levels);
               keep small (<=0.02) or the peak can sink into the background
"""
import argparse
import numpy as np
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.quantum_info import random_unitary, Statevector

def _random_block(n, depth, rng):
    qc = QuantumCircuit(n)
    for d in range(depth):
        for i in range(d % 2, n - 1, 2):
            qc.unitary(random_unitary(4, seed=int(rng.integers(1 << 31))), [i, i + 1])
    return qc

def _encoder_layers(secret, rng):
    n = len(secret)
    P1, P2 = QuantumCircuit(n), QuantumCircuit(n)
    for i, b in enumerate(secret):
        target = np.pi if b else 0.0
        a = rng.uniform(0, 2 * np.pi)
        P1.rx(a, i); P2.rx(target - a, i)
    return P1, P2

def make_peaked(n=8, depth=3, secret=None, swap_k=None, sweep_std=0.01, seed=0):
    rng = np.random.default_rng(seed)
    if secret is None:
        secret = rng.integers(0, 2, size=n).tolist()
    swap_k = swap_k if swap_k is not None else max(2, n // 2)
    P1, P2 = _encoder_layers(secret, rng)

    qc = QuantumCircuit(n)
    for P in (None, P1, None, P2, None):           # U1U1d P1 U2U2d P2 U3U3d
        if P is None:                              # obfuscated identity block
            U = _random_block(n, depth, rng)
            sw = [(int(j), int(j) + 1) for j in rng.integers(0, n - 1, size=swap_k)]
            for a, b in sw: qc.swap(a, b)          # swap-trick: pi (UUd) pi_dag = I
            qc.compose(U, inplace=True); qc.compose(U.inverse(), inplace=True)
            for a, b in reversed(sw): qc.swap(a, b)
        else:
            qc.compose(P, inplace=True)

    qc = transpile(qc, basis_gates=["rz", "rx", "cx"], optimization_level=1, seed_transpiler=seed)
    swept = QuantumCircuit(n)
    for inst in qc.data:                           # angle sweep
        op = inst.operation
        if op.name in ("rz", "rx") and rng.random() < 0.5:
            op = op.copy(); op.params[0] = float(op.params[0] + rng.normal(0, sweep_std))
        swept.append(op, inst.qubits, inst.clbits)
    qc = transpile(swept, basis_gates=["rz", "rx", "cx"], optimization_level=1, seed_transpiler=seed + 1)

    secret_str = "".join(str(b) for b in secret[::-1])   # q0 = rightmost (challenge convention)
    return qc, secret_str

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--swap-k", type=int, default=None)
    ap.add_argument("--sweep-std", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="peaked_test.qasm")
    a = ap.parse_args()

    qc, secret = make_peaked(n=a.n, depth=a.depth, swap_k=a.swap_k,
                             sweep_std=a.sweep_std, seed=a.seed)
    with open(a.out, "w") as f:
        f.write(qasm2.dumps(qc))
    ans = a.out.rsplit(".", 1)[0] + ".answer.txt"
    with open(ans, "w") as f:
        f.write(secret + "\n")

    print(f"wrote {a.out}  ({a.n} qubits, {len(qc.data)} gates, depth {qc.depth()})")
    print(f"planted answer (q0 rightmost) -> {ans}: {secret}")
    if a.n <= 14:
        probs = Statevector(qc).probabilities_dict()
        peak = max(probs, key=probs.get)
        print(f"statevector check: peak={peak}  prob={probs[peak]:.3f}  MATCH={peak == secret}")
    else:
        print("(n>14: statevector verify skipped; answer is planted by construction)")

if __name__ == "__main__":
    main()