"""Tests for the GPU cluster module and GPU simulation backends.

GPU-specific paths are tested with ``gpu_available()`` guards; on CPU-only CI
machines these tests become no-ops.  The CPU fallback paths (which run on all
machines) are always exercised.
"""

from pathlib import Path

import pytest
from qiskit import QuantumCircuit

from quantum_hack.gpu import (
    GpuClusterConfig,
    cluster_info,
    count_gpus,
    gpu_available,
    make_mps_simulator,
    make_statevector_simulator,
)
from quantum_hack.simulation import (
    auto_simulation,
    gpu_mps_simulation,
    gpu_statevector_simulation,
)

SAMPLE_QASM = Path("qasm_data/very_easy/challenge-8_1.qasm")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deterministic_circuit() -> QuantumCircuit:
    """3-qubit deterministic circuit whose only outcome is |001>."""
    qc = QuantumCircuit(3)
    qc.x(0)
    return qc


# ---------------------------------------------------------------------------
# GpuClusterConfig unit tests
# ---------------------------------------------------------------------------

class TestGpuClusterConfig:
    def test_auto_detect_returns_config(self):
        cfg = GpuClusterConfig.auto_detect()
        assert isinstance(cfg, GpuClusterConfig)
        assert cfg.num_gpus >= 0

    def test_cpu_only_has_no_gpu(self):
        cfg = GpuClusterConfig.cpu_only()
        assert not cfg.has_gpu
        assert not cfg.is_multi_gpu

    def test_multi_gpu_flag(self):
        cfg = GpuClusterConfig(num_gpus=4, blocking_qubits=22)
        assert cfg.is_multi_gpu

    def test_single_gpu_not_multi_gpu(self):
        cfg = GpuClusterConfig(num_gpus=1)
        assert not cfg.is_multi_gpu

    def test_str_cpu_only(self):
        assert "CPU" in str(GpuClusterConfig.cpu_only())

    def test_str_gpu(self):
        cfg = GpuClusterConfig(num_gpus=2, blocking_qubits=22, custatevec_enable=True)
        s = str(cfg)
        assert "gpus=2" in s
        assert "cuStateVec" in s


# ---------------------------------------------------------------------------
# cluster_info / count_gpus
# ---------------------------------------------------------------------------

def test_cluster_info_keys():
    info = cluster_info()
    assert "num_gpus" in info
    assert "mpi_world_size" in info
    assert "gpu_available" in info


def test_count_gpus_non_negative():
    assert count_gpus() >= 0


def test_gpu_available_consistent():
    assert gpu_available() == (count_gpus() > 0)


# ---------------------------------------------------------------------------
# AerSimulator factories (CPU fallback always exercised)
# ---------------------------------------------------------------------------

class TestSimulatorFactories:
    def test_make_statevector_cpu_fallback(self):
        cfg = GpuClusterConfig.cpu_only()
        sim = make_statevector_simulator(cfg)
        assert sim is not None

    def test_make_mps_cpu_fallback(self):
        cfg = GpuClusterConfig.cpu_only()
        sim = make_mps_simulator(cfg, bond_dim=64)
        assert sim is not None

    @pytest.mark.skipif(not gpu_available(), reason="no CUDA GPU present")
    def test_make_statevector_gpu(self):
        cfg = GpuClusterConfig.auto_detect()
        sim = make_statevector_simulator(cfg)
        assert sim is not None

    @pytest.mark.skipif(not gpu_available(), reason="no CUDA GPU present")
    def test_make_mps_gpu(self):
        cfg = GpuClusterConfig.auto_detect()
        sim = make_mps_simulator(cfg)
        assert sim is not None


# ---------------------------------------------------------------------------
# gpu_statevector_simulation — CPU fallback path
# ---------------------------------------------------------------------------

class TestGpuStatevectorSimulation:
    def test_cpu_fallback_deterministic(self):
        cfg = GpuClusterConfig.cpu_only()
        bitstring, prob = gpu_statevector_simulation(_deterministic_circuit(), config=cfg)
        assert bitstring == "001"
        assert prob == pytest.approx(1.0)

    def test_cpu_fallback_does_not_mutate_input(self):
        cfg = GpuClusterConfig.cpu_only()
        qc = _deterministic_circuit()
        qc.measure_all()
        before = len(qc.data)
        gpu_statevector_simulation(qc, config=cfg)
        assert len(qc.data) == before

    @pytest.mark.skipif(not gpu_available(), reason="no CUDA GPU present")
    def test_gpu_deterministic(self):
        cfg = GpuClusterConfig.auto_detect()
        bitstring, prob = gpu_statevector_simulation(_deterministic_circuit(), config=cfg)
        assert bitstring == "001"
        assert prob == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# gpu_mps_simulation — CPU fallback path
# ---------------------------------------------------------------------------

class TestGpuMpsSimulation:
    def test_cpu_fallback_deterministic(self):
        cfg = GpuClusterConfig.cpu_only()
        bitstring, prob = gpu_mps_simulation(_deterministic_circuit(), config=cfg, shots=256)
        assert bitstring == "001"
        assert prob == pytest.approx(1.0)

    def test_cpu_fallback_does_not_mutate_input(self):
        cfg = GpuClusterConfig.cpu_only()
        qc = _deterministic_circuit()
        before = len(qc.data)
        gpu_mps_simulation(qc, config=cfg, shots=64)
        assert len(qc.data) == before

    @pytest.mark.skipif(not gpu_available(), reason="no CUDA GPU present")
    def test_gpu_deterministic(self):
        cfg = GpuClusterConfig.auto_detect()
        bitstring, prob = gpu_mps_simulation(_deterministic_circuit(), config=cfg, shots=256)
        assert bitstring == "001"
        assert prob == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# auto_simulation — selects backend based on qubit count + GPU availability
# ---------------------------------------------------------------------------

class TestAutoSimulation:
    def test_small_circuit_cpu(self):
        cfg = GpuClusterConfig.cpu_only()
        bitstring, prob = auto_simulation(_deterministic_circuit(), config=cfg)
        assert bitstring == "001"
        assert prob == pytest.approx(1.0)

    def test_large_circuit_falls_back_to_mps(self):
        # Build a 25-qubit identity circuit (no gates), all-zero outcome expected.
        cfg = GpuClusterConfig.cpu_only()
        qc = QuantumCircuit(25)
        bitstring, prob = auto_simulation(qc, config=cfg, shots=128, sv_qubit_limit=20)
        assert len(bitstring) == 25
        assert set(bitstring) <= {"0", "1"}

    @pytest.mark.skipif(not gpu_available(), reason="no CUDA GPU present")
    def test_small_circuit_gpu(self):
        cfg = GpuClusterConfig.auto_detect()
        bitstring, prob = auto_simulation(_deterministic_circuit(), config=cfg)
        assert bitstring == "001"
        assert prob == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.skipif(not SAMPLE_QASM.exists(), reason="sample QASM not present")
    def test_on_real_circuit_cpu(self):
        cfg = GpuClusterConfig.cpu_only()
        qc = QuantumCircuit.from_qasm_file(str(SAMPLE_QASM))
        bitstring, prob = auto_simulation(qc, config=cfg)
        assert len(bitstring) == qc.num_qubits
        assert set(bitstring) <= {"0", "1"}
        assert 0.0 <= prob <= 1.0
