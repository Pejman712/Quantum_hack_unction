
def weighted_majority_bitstring(results: list[tuple[str, float]]) -> str:
    """
    Compute the weighted majority bitstring from a list of bitstrings and probabilities.

    For each bit position, the function sums the probabilities of all bitstrings
    that have a `1` in that position and compares it with the summed probabilities
    of all bitstrings that have a `0` in the same position. The output bit is `1`
    if the total probability for `1` is greater than or equal to the total
    probability for `0`; otherwise, the output bit is `0`.

    If the input list is empty, the function returns an empty string.

    Args:
        results: A list of tuples where each tuple contains a bitstring and its
            associated probability.

    Returns:
        A bitstring formed by choosing the weighted majority bit at each position.

    Raises:
        ValueError: If not all bitstrings have the same length.

    Example:
        >>> weighted_majority_bitstring([
        ...     ("111", 0.4),
        ...     ("101", 0.3),
        ...     ("000", 0.3),
        ... ])
        '101'
    """

    if not results:
        return ""

    n = len(results[0][0])

    if any(len(bitstring) != n for bitstring, _ in results):
        raise ValueError("All bitstrings must have the same length")

    majority_bits = []

    for i in range(n):
        prob_1 = sum(prob for bitstring, prob in results if bitstring[i] == "1")
        prob_0 = sum(prob for bitstring, prob in results if bitstring[i] == "0")

        # Tie-breaking rule: choose 1
        majority_bits.append("1" if prob_1 >= prob_0 else "0")

    return "".join(majority_bits)

# results = [
#     ("111110011011011111011001011110101111", 0.0020),
#     ("110111011011111101011001011110101111", 0.0017),
#     ("111110011001011101011001011110101111", 0.0015),
#     ("111110011001111101011001011110101111", 0.0012),
#     ("111110011011111101011001011110101111", 0.0012),
#     ("110110011011011101011001011110101111", 0.0010),
#     ("111110011011111101011000011110101111", 0.0010),
#     ("110110011000011101011001011110101111", 0.0010),
#     ("111110011000011111011001011110101111", 0.0010),
#     ("110110011001011111011001011110101111", 0.0010),
# ]

# majority = weighted_majority_bitstring(results)
# print(majority)
