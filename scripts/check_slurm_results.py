"""
Compare top bitstrings from SLURM unswap job output files against CSV ground truth.

Usage:
    # Check all .out files in current directory:
    python scripts/check_slurm_results.py

    # Check specific files or a glob:
    python scripts/check_slurm_results.py slurm-unswap-*.out

    # Check files from a specific difficulty tier only:
    python scripts/check_slurm_results.py --difficulty easy slurm-unswap-*.out
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

DIFFICULTY_ORDER = ["very_easy", "easy", "moderate", "hard", "very_hard"]


def load_ground_truth() -> dict[str, str]:
    """Return {challenge_name: bitstring} from all result CSVs (last success wins)."""
    gt: dict[str, str] = {}
    for csv_path in RESULTS_DIR.glob("*_bitstrings.csv"):
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                if row.get("status") == "success":
                    gt[row["challenge"]] = row["bitstring"]
    return gt


def parse_out_file(path: Path) -> tuple[str | None, str | None]:
    """Return (challenge_name, top_bitstring) extracted from a SLURM .out file."""
    challenge = None
    bitstring = None
    for line in path.read_text(errors="replace").splitlines():
        # SLURM script prints: "Circuit : /path/to/qasm_data/easy/challenge-8_11.qasm"
        m = re.search(r"Circuit\s*:\s*(.+\.qasm)", line)
        if m:
            challenge = Path(m.group(1).strip()).stem  # e.g. "challenge-8_11"
        # run_unswap.py prints: "Top bitstring: 01001110"
        m = re.search(r"Top bitstring:\s*([01]+)", line)
        if m:
            bitstring = m.group(1).strip()
    return challenge, bitstring


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate unswap SLURM outputs against CSV ground truth")
    parser.add_argument("outfiles", nargs="*", help=".out files to check (default: slurm-unswap-*.out)")
    parser.add_argument("--difficulty", help="Filter to a specific difficulty tier")
    args = parser.parse_args()

    if args.outfiles:
        paths = [Path(p) for p in args.outfiles]
    else:
        paths = sorted(Path(".").glob("slurm-unswap-*.out"))

    if not paths:
        print("No .out files found. Pass paths explicitly or run from the directory containing slurm-*.out files.")
        sys.exit(1)

    gt = load_ground_truth()

    passed = []
    failed = []
    skipped = []

    for path in paths:
        challenge, bitstring = parse_out_file(path)

        if challenge is None:
            skipped.append((path.name, "no 'Circuit :' line found (job may still be running or crashed early)"))
            continue
        if bitstring is None:
            skipped.append((path.name, f"{challenge} — no 'Top bitstring:' line (job incomplete or failed)"))
            continue
        if challenge not in gt:
            skipped.append((path.name, f"{challenge} — not in any results CSV"))
            continue

        expected = gt[challenge]
        if bitstring == expected:
            passed.append((challenge, bitstring, path.name))
        else:
            failed.append((challenge, bitstring, expected, path.name))

    print(f"\n{'='*70}")
    print(f"  Unswap result check — {len(paths)} file(s)")
    print(f"{'='*70}")

    if passed:
        print(f"\n  PASS ({len(passed)})")
        for challenge, bs, fname in passed:
            print(f"    [OK] {challenge:35s}  {bs}  ({fname})")

    if failed:
        print(f"\n  FAIL ({len(failed)})")
        for challenge, got, expected, fname in failed:
            print(f"    [XX] {challenge:35s}  got      {got}")
            print(f"         {' '*35}  expected {expected}  ({fname})")

    if skipped:
        print(f"\n  SKIPPED ({len(skipped)})")
        for name, reason in skipped:
            print(f"    [--] {name}: {reason}")

    print(f"\n  Result: {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")
    print(f"{'='*70}\n")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
