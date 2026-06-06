from quantum_hack.peak.result import PeakResult, score_confidence


def test_confidence_zero_when_not_local_max():
    # A bitstring beaten by a neighbour cannot be the peak, regardless of magnitude.
    assert score_confidence(
        probability=0.9, num_qubits=8, is_local_max=False, agreeing_methods=3
    ) == 0.0


def test_confidence_rises_with_local_max_status():
    untested = score_confidence(
        probability=0.5, num_qubits=8, is_local_max=None, agreeing_methods=1
    )
    confirmed = score_confidence(
        probability=0.5, num_qubits=8, is_local_max=True, agreeing_methods=1
    )
    assert confirmed > untested > 0.0


def test_confidence_monotonic_in_probability():
    low = score_confidence(probability=0.05, num_qubits=8, is_local_max=True, agreeing_methods=1)
    high = score_confidence(probability=0.8, num_qubits=8, is_local_max=True, agreeing_methods=1)
    assert high > low


def test_confidence_rises_with_consensus():
    one = score_confidence(probability=0.6, num_qubits=8, is_local_max=True, agreeing_methods=1)
    two = score_confidence(probability=0.6, num_qubits=8, is_local_max=True, agreeing_methods=2)
    assert two > one


def test_confidence_at_uniform_probability_is_zero():
    # probability == 2**-n contributes no magnitude signal.
    assert score_confidence(
        probability=2**-8, num_qubits=8, is_local_max=True, agreeing_methods=2
    ) == 0.0


def test_peak_result_round_trips_through_dict():
    result = PeakResult(
        bitstring="0101",
        probability=0.97,
        method="statevector",
        num_qubits=4,
        confidence=0.8,
        is_local_max=True,
        neighbor_gap=0.5,
        agreeing_methods=("statevector", "marginal"),
        notes="exact peak",
    )
    restored = PeakResult.from_dict(result.to_dict())
    assert restored == result
    assert isinstance(result.to_dict()["agreeing_methods"], list)
