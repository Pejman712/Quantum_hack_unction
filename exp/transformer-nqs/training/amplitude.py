"""
Circuit amplitude evaluators for VMC training.

Each evaluator has the signature:
    fn(qasm_path, bitstring) -> float   (log |⟨x|ψ⟩|²)

Returning -inf for zero-amplitude bitstrings is correct — the REINFORCE
baseline naturally ignores them and they receive zero advantage gradient.

Qubit ordering convention
-------------------------
The model and CSV files use Qiskit little-endian convention: character i of
the bitstring string is the state of qubit i.  quimb uses big-endian: the
first character is the most significant qubit.  All evaluators here accept
bitstrings in model/CSV convention and reverse internally.
"""

import math
import warnings


def quimb_amplitude(qasm_path, bitstring):
    """
    Compute log|⟨x|ψ⟩|² via quimb tensor-network contraction.

    Efficient for circuits with limited entanglement (treewidth ≪ n).
    Uses `simplify_sequence='DC'` to avoid a zero-division bug in quimb's
    default 'ADCRS' simplification pipeline.

    Parameters
    ----------
    qasm_path : str or Path
    bitstring : str — in model/CSV convention (q[0] is leftmost character)

    Returns
    -------
    float — log|A(x)|², or -inf if the amplitude is zero or an error occurs
    """
    try:
        import quimb.tensor as qtn
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            circ = qtn.Circuit.from_openqasm2_file(str(qasm_path))
        # quimb big-endian: reverse the bitstring
        bs_quimb = bitstring[::-1]
        amp = circ.amplitude(bs_quimb, simplify_sequence="DC", simplify_atol=0.0)
        amp_sq = abs(amp) ** 2
        if amp_sq == 0.0:
            return float("-inf")
        return math.log(amp_sq)
    except Exception:
        return float("-inf")


def make_peaked_mock(target_bitstring, peak_prob=0.9, other_prob=1e-4):
    """
    Return a mock amplitude function that assigns `peak_prob` to one bitstring
    and `other_prob` to everything else.  Useful for unit tests.

    The reward decays with Hamming distance from the target so the model
    gets a gradient signal even when it hasn't found the exact target yet.

    Parameters
    ----------
    target_bitstring : str
    peak_prob        : float — |A(target)|²
    other_prob       : float — |A(x≠target)|² baseline

    Returns
    -------
    amplitude_fn : callable(qasm_path, bitstring) -> float
    """
    n = len(target_bitstring)

    def _fn(qasm_path, bitstring):
        hamming = sum(a != b for a, b in zip(bitstring, target_bitstring))
        # Interpolate: prob = peak * exp(-3 * hamming/n) + other_prob floor
        p = peak_prob * math.exp(-3 * hamming / max(n, 1))
        p = max(p, other_prob)
        return math.log(p)

    return _fn
