def bitstring_to_tokens(bitstring):
    """'01101' → [0, 1, 1, 0, 1]"""
    return [int(b) for b in bitstring]


def tokens_to_bitstring(tokens):
    """[0, 1, 1, 0, 1] → '01101'"""
    return "".join(str(t) for t in tokens)
