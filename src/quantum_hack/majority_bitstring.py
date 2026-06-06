
def weighted_majority_bitstring(results: list[tuple[str, float]]) -> str:
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