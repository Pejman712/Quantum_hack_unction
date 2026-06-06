
from quantum_hack.majority_bitstring import weighted_majority_bitstring


def test_weighted_majority_bitstring_basic():
    """
    Test that the function correctly computes the weighted majority bitstring.

    The majority bit at each position is chosen based on the total probability
    mass assigned to `0` and `1`, not by simply counting how many bitstrings
    contain each bit.
    """
    results = [
        ("111", 0.4),
        ("101", 0.3),
        ("000", 0.3),
    ]

    assert weighted_majority_bitstring(results) == "101"
