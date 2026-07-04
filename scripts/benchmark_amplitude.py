"""
Go/no-go check for the VMC amplitude oracle: how long does a single quimb
tensor-network contraction take on each circuit, and does that cost scale into
the very_hard tier (48-104 qubits)?

VMC (exp/transformer-nqs/training/vmc.py) calls quimb_amplitude(qasm_path, bitstring)
once per sampled bitstring per training step. If contraction on the largest circuits
is too slow (or blows up memory), the amplitude oracle -- not the model architecture --
is what makes NQS training infeasible there, regardless of which sampler drives it.

Reports build time (QASM -> quimb Circuit, paid once and cacheable across a batch)
separately from marginal per-amplitude contraction time (paid once per sample), since
the two have very different implications for how VMC should be structured.

Usage:
    python scripts/benchmark_amplitude.py --difficulty very_hard --n-bitstrings 10
    python scripts/benchmark_amplitude.py --qasm qasm_data/very_hard/challenge-104_49.qasm
    python scripts/benchmark_amplitude.py --difficulty hard --n-bitstrings 10   # baseline for comparison
"""

import argparse
import random
import signal
import statistics
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class ContractionTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise ContractionTimeout()


def n_qubits_of(qasm_path):
    for line in Path(qasm_path).read_text().splitlines():
        if line.strip().startswith("qreg"):
            return int(line.split("[", 1)[1].split("]", 1)[0])
    raise ValueError(f"no qreg line found in {qasm_path}")


def build_circuit(qasm_path, timeout_s):
    import quimb.tensor as qtn

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            circ = qtn.Circuit.from_openqasm2_file(str(qasm_path))
        return circ, time.perf_counter() - start, None
    except ContractionTimeout:
        return None, timeout_s, "timeout"
    except Exception as e:
        return None, time.perf_counter() - start, f"error: {e}"
    finally:
        signal.alarm(0)


def time_amplitude(circ, bitstring, timeout_s):
    """bitstring is in model/CSV (little-endian) convention; reversed for quimb."""
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)
    start = time.perf_counter()
    try:
        bs_quimb = bitstring[::-1]
        amp = circ.amplitude(bs_quimb, simplify_sequence="DC", simplify_atol=0.0)
        return time.perf_counter() - start, abs(amp) ** 2, None
    except ContractionTimeout:
        return timeout_s, None, "timeout"
    except Exception as e:
        return time.perf_counter() - start, None, f"error: {e}"
    finally:
        signal.alarm(0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qasm", help="benchmark a single circuit instead of a whole tier")
    p.add_argument("--difficulty", default="very_hard", help="tier under qasm_data/ (default: very_hard)")
    p.add_argument("--n-bitstrings", type=int, default=10, help="random bitstrings sampled per circuit")
    p.add_argument("--timeout", type=int, default=120, help="per-call timeout in seconds (build and each amplitude)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)

    if args.qasm:
        paths = [Path(args.qasm)]
    else:
        tier_dir = REPO_ROOT / "qasm_data" / args.difficulty
        paths = sorted(tier_dir.glob("*.qasm"), key=n_qubits_of)

    header = f"{'circuit':<24} {'qubits':>6} {'build_s':>9} {'amp_mean_s':>11} {'amp_median_s':>13} {'amp_max_s':>10} {'fail':>8}"
    print(header)
    print("-" * len(header))

    for qasm_path in paths:
        n = n_qubits_of(qasm_path)

        circ, build_s, build_err = build_circuit(qasm_path, args.timeout)
        if build_err:
            print(f"{qasm_path.stem:<24} {n:>6} {'BUILD FAIL':>9}   {build_err}")
            continue

        bitstrings = ["".join(random.choice("01") for _ in range(n)) for _ in range(args.n_bitstrings)]
        times = []
        failures = 0
        for bs in bitstrings:
            elapsed, amp_sq, err = time_amplitude(circ, bs, args.timeout)
            if err:
                failures += 1
                print(f"    [{qasm_path.stem}] contraction failed after {elapsed:.1f}s: {err}", file=sys.stderr)
            else:
                times.append(elapsed)

        if times:
            mean_s, median_s, max_s = statistics.mean(times), statistics.median(times), max(times)
        else:
            mean_s = median_s = max_s = float("nan")

        print(f"{qasm_path.stem:<24} {n:>6} {build_s:>9.2f} {mean_s:>11.2f} {median_s:>13.2f} {max_s:>10.2f} "
              f"{failures:>4}/{args.n_bitstrings}")


if __name__ == "__main__":
    main()
