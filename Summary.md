# Circuit Optimization Investigation — Summary

How well the available transforms reduce the challenge circuits, which method to trust, and a
correctness bug found and fixed along the way. Metrics are measured in the `rz`/`rx`/`cx` basis;
**CX depth** (two-qubit critical-path length) is the headline metric, since two-qubit gates
dominate runtime and error.

## TL;DR

- **Transpile is the only thing that reduces these circuits**, and **opt-2 ≈ opt-3** (opt-3 buys
  essentially nothing extra here).
- **Pure pyzx (ZX-calculus) hurts** — it *grows* CX count 40–80% and CX depth/total gates 2–3× on
  every easy circuit, so the optimize pipeline's cost guard always rolls it back.
- **Snapping angles to the grid first gives a small extra gain** and noticeably helps pyzx, but not
  enough to make pyzx a net win.
- **Fixed a silent wrong-answer bug**: `transpile` at optimization level ≥1 elides SWAP gates into a
  qubit *permutation*, which relabels the peak bitstring; the optimizer trusted it as
  `by_construction`. Qubit order is now restored after transpilation.
- **`snap` retargeted from π/4 → π/8** (finer grid; see note at the end).

## The optimize pipeline

`optimize_circuit` greedily applies `strip → snap → transpile(opt-3) → pyzx`, keeping each step only
if it does not increase a weighted cost (`3·CX + depth`) and does not change the peak bitstring. On
all 17 `easy` circuits (peak verified ≤24q; wider circuits rely on outcome-preserving transforms):

| metric | reduction | guarantees |
|--------|-----------|------------|
| CX depth | **−5.4%** | by_construction = 14 |
| total gates | **−5.5%** | peak_verified = 3 |
| CX count | **−5.9%** | unverified = **0** (after the permutation fix) |

All 17 circuits shrank. `pyzx` appears in **zero** accepted step lists — it never helps.

## Method comparison on the raw `easy` circuits

Aggregate over the 12 circuits ≤50q (where pyzx also runs); `+%` = shrank, `−%` = grew:

| method | depth | CX depth | CX (2q) | total gates |
|--------|-------|----------|---------|-------------|
| raw  | 1955 | 1030 | 3931 | 11365 |
| **opt2** | 1912 (+2%) | 984 (+4%) | 3780 (+4%) | 11225 (+1%) |
| **opt3** | 1912 (+2%) | 984 (+4%) | 3778 (+4%) | 11221 (+1%) |
| **pyzx** | 6527 (−234%) | 2064 (−100%) | 6334 (−61%) | 32591 (−187%) |

opt2 and opt3 produce **identical** CX counts on 16 of 17 circuits (they differ only on `problem15`,
by 2 CX). pyzx inflates every metric.

## Same comparison with `snap` applied first

(`snap`-first results below were measured with the **π/4** grid, the version in place at the time.)

| method | depth | CX depth | CX (2q) | total gates |
|--------|-------|----------|---------|-------------|
| raw  | 1955 | 1030 | 3931 | 11365 |
| **opt2** | 1911 (+2%) | 978 (+5%) | 3740 (+5%) | 11186 (+2%) |
| **opt3** | 1911 (+2%) | 978 (+5%) | 3738 (+5%) | 11182 (+2%) |
| **pyzx** | 5774 (−195%) | 1818 (−77%) | 5627 (−43%) | 29130 (−156%) |

Snapping first nudges opt2/opt3 from +4%→+5% CX reduction, and markedly reduces pyzx's blow-up
(CX −61%→−43%; e.g. `challenge-16_12` pyzx 220→131) — consistent with the fact that grid angles move
circuits toward the Clifford+T structure ZX-calculus is built for. But even snapped, pyzx stays
larger than the raw circuit, so it is never the winner.

**Lossy check:** `snap` (default 0.01 threshold) **preserved** the exact peak bitstring on every
verifiable circuit (8q, 16q, 24q); 32–64q are too large for an exact statevector check.

## Why pyzx hurts here

`full_reduce` + `extract_circuit` preserve the unitary (the peak is safe), but ZX extraction is
tuned for Clifford+T circuits. The challenge circuits are dominated by *arbitrary continuous
rotations*, which extraction re-expands into many gates — so the output is bigger, not smaller. It
is fast (≤10s even at 48q) and provably correct, just not useful on this circuit family.

## The qubit-permutation bug (found and fixed)

At `optimization_level ≥ 1`, Qiskit's transpiler elides SWAP gates into a layout permutation (a real
CX saving) instead of leaving them in the gate list. For a measurement-free circuit read via
statevector, this **relabels the peak bitstring** — e.g. `…001000` → `…100000` — even though the
probability distribution is unchanged. Because the `transpile` step is non-lossy, the optimizer
applied no peak check and returned the *wrong* bitstring labeled `by_construction`; the batch solver
would have submitted it. (This showed up as `unverified` on 2 of 10 `very_easy` circuits.)

**Fix:** `transpile_to_basis` now reads the transpile layout and relabels the output wires back to
the input order — keeping the SWAP-elision savings (gate counts unchanged), only correcting which
wire is which. After the fix, all 17 `easy` circuits verify cleanly (0 `unverified`). Guarded by
unit tests at both the `transpile_to_basis` and `optimize_circuit` levels, plus a routing/ancilla
case.

## Practical guidance

- Use **transpile opt-2** to reduce these circuits — it matches opt-3 and is cheaper.
- Leave **pyzx** in the pipeline (it's cost-guarded and harmless) but don't expect it to help this
  circuit family; it would help Clifford+T-structured circuits.
- The dominant runtime cost of `--optimize` is the **exact-statevector peak check** (only ≤~24q is
  affordable; a 28q statevector is ~4 GiB), not the transforms themselves.

## Note on the π/4 → π/8 change

`snap_rotations_to_pi_over_8` now snaps near-grid angles to multiples of **π/8** (a finer grid than
π/4; the default 1% threshold therefore tolerates half the absolute angle deviation). This is less
lossy per snap but, since π/8 rotations are *not* Clifford+T, it is further from pyzx's sweet spot
than π/4 was — the `snap`-first numbers above were taken with the π/4 grid and have not been
re-benchmarked at π/8.
