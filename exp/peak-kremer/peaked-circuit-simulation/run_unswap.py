"""
Usage:
    python run_unswap.py path/to/circuit.qasm [options]

Examples:
    python run_unswap.py ../../../qasm_data/moderate/challenge-8_27.qasm
    python run_unswap.py circuit.qasm --max-bond 4096 --cutoff 0.002 --samples 5000 --top 20
"""

import argparse
import sys
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(description="MPO unswap simulation of a QASM circuit")
    p.add_argument("qasm", help="Path to the .qasm file")
    p.add_argument("--max-bond",   type=int,   default=8192,  help="MPO max bond dimension (default: 8192)")
    p.add_argument("--cutoff",     type=float, default=0.002,  help="SVD truncation cutoff (default: 0.002)")
    p.add_argument("--max-its",    type=int,   default=20,    help="Max unswap iterations (default: 20)")
    p.add_argument("--threshold",  type=float, default=1e6,   help="Element count that triggers unswapping (default: 1e6)")
    p.add_argument("--samples",    type=int,   default=10000, help="MPS samples to draw (default: 10000)")
    p.add_argument("--top",        type=int,   default=10,    help="Top-k bitstrings to print (default: 10)")
    p.add_argument("--seed",       type=int,   default=123,   help="Random seed (default: 123)")
    p.add_argument("--mps-bond",   type=int,   default=None,  help="MPS max bond (defaults to --max-bond)")
    return p.parse_args()


def main():
    args = parse_args()
    mps_bond = args.mps_bond or args.max_bond

    from unswap import simulate_qasm

    print(f"Loading: {args.qasm}")
    print(f"Params : max_bond={args.max_bond}, cutoff={args.cutoff}, max_its={args.max_its}, threshold={args.threshold:.0e}")
    print(f"Sampling {args.samples} shots, reporting top {args.top}\n")

    pbar = tqdm(total=None, unit="u", desc="Compressing", dynamic_ncols=True)
    _last = [0]

    def _progress(consumed, total):
        if pbar.total is None:
            pbar.total = total
            pbar.refresh()
        delta = consumed - _last[0]
        if delta > 0:
            pbar.update(delta)
        _last[0] = consumed

    results = simulate_qasm(
        args.qasm,
        n_samples=args.samples,
        top_k=args.top,
        compress_kwargs=dict(
            max_bond=args.max_bond,
            cutoff=args.cutoff,
            max_its=args.max_its,
            unswap_threshold=args.threshold,
            seed=args.seed,
        ),
        mps_kwargs=dict(
            max_bond=mps_bond,
            cutoff=args.cutoff,
        ),
        progress_callback=_progress,
    )
    pbar.close()

    print(f"\n{'Rank':<6} {'Bitstring':<60} {'Count':>6}  {'Prob':>7}")
    print("-" * 80)
    total = sum(c for _, c in results)
    for rank, (bs, count) in enumerate(results, 1):
        print(f"{rank:<6} {bs:<60} {count:>6}  {count/args.samples:>6.2%}")

    print(f"\nTop bitstring: {results[0][0]}")


if __name__ == "__main__":
    main()
