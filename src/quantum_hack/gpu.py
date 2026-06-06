"""GPU cluster detection and AerSimulator factory for quantum circuit simulation.

Supports three tiers of GPU acceleration:
* Single GPU via cuStateVec (``cuStatevec_enable=True``).
* Multi-GPU single-node via Aer blocking mode (``blocking_enable=True``).
* Multi-node GPU cluster via MPI (requires ``qiskit-aer-gpu`` built with MPI
  and ``mpi4py`` installed).

All public helpers fall back gracefully when no GPU is available.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def count_gpus() -> int:
    """Return the number of CUDA-capable GPUs visible to nvidia-smi, or 0."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        return len(lines)
    except Exception:
        return 0


def gpu_available() -> bool:
    """Return ``True`` if at least one CUDA GPU is accessible."""
    return count_gpus() > 0


def _mpi_world_size() -> int:
    """Return the MPI world size, or 1 if mpi4py is not installed / not in MPI context."""
    try:
        from mpi4py import MPI  # type: ignore[import]
        return int(MPI.COMM_WORLD.Get_size())
    except ImportError:
        return 1


def _optimal_blocking_qubits(num_gpus: int) -> int:
    """Return a sensible blocking_qubits value for the given number of GPUs.

    ``blocking_qubits=N`` means each GPU chunk holds ``2**N`` complex128 amplitudes
    (16 bytes each), so N=23 -> 128 MB per chunk.  Reduce by 1 for each doubling
    of the GPU count so the total GPU memory budget stays roughly constant.
    """
    import math
    if num_gpus <= 1:
        return 23
    return max(17, 23 - int(math.log2(num_gpus)))


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class GpuClusterConfig:
    """Describes the GPU resources available for circuit simulation.

    Attributes:
        num_gpus: Number of GPUs to use (0 means CPU-only mode).
        blocking_qubits: log2 of the statevector chunk size per GPU; only
            used when ``num_gpus > 1``.
        custatevec_enable: Whether to enable NVIDIA cuStateVec for single-GPU
            statevector acceleration.
        mpi_enable: Whether to use MPI for multi-node distribution.
    """

    num_gpus: int = 0
    blocking_qubits: int = 23
    custatevec_enable: bool = True
    mpi_enable: bool = False

    # ------------------------------------------------------------------ #
    # Factories                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def auto_detect(cls) -> GpuClusterConfig:
        """Probe the environment and return the best available configuration."""
        n = count_gpus()
        mpi = _mpi_world_size() > 1
        return cls(
            num_gpus=n,
            blocking_qubits=_optimal_blocking_qubits(n),
            custatevec_enable=True,
            mpi_enable=mpi,
        )

    @classmethod
    def cpu_only(cls) -> GpuClusterConfig:
        """Return a config that disables all GPU acceleration."""
        return cls(num_gpus=0, custatevec_enable=False, mpi_enable=False)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def has_gpu(self) -> bool:
        """``True`` when at least one GPU is configured."""
        return self.num_gpus > 0

    @property
    def is_multi_gpu(self) -> bool:
        """``True`` when blocking / multi-GPU mode will be used."""
        return self.num_gpus > 1

    def __str__(self) -> str:
        if not self.has_gpu:
            return "GpuClusterConfig(CPU-only)"
        parts = [f"gpus={self.num_gpus}"]
        if self.custatevec_enable:
            parts.append("cuStateVec=on")
        if self.is_multi_gpu:
            parts.append(f"blocking_qubits={self.blocking_qubits}")
        if self.mpi_enable:
            parts.append("MPI=on")
        return f"GpuClusterConfig({', '.join(parts)})"


# ---------------------------------------------------------------------------
# AerSimulator factories
# ---------------------------------------------------------------------------

def make_statevector_simulator(
    config: GpuClusterConfig | None = None,
) -> AerSimulator:
    """Return an :class:`~qiskit_aer.AerSimulator` configured for GPU statevector.

    Automatically falls back to CPU statevector when ``config.has_gpu`` is
    ``False`` (e.g. no CUDA GPUs detected).

    Args:
        config: GPU cluster configuration.  ``None`` triggers auto-detection.

    Returns:
        AerSimulator: Simulator ready for statevector simulation.
    """
    if config is None:
        config = GpuClusterConfig.auto_detect()

    if not config.has_gpu:
        return AerSimulator(method="statevector")

    kwargs: dict = {
        "method": "statevector",
        "device": "GPU",
        "cuStateVec_enable": config.custatevec_enable,
    }

    if config.is_multi_gpu:
        kwargs["blocking_enable"] = True
        kwargs["blocking_qubits"] = config.blocking_qubits

    if config.mpi_enable:
        kwargs["mpi"] = True

    return AerSimulator(**kwargs)


def make_mps_simulator(
    config: GpuClusterConfig | None = None,
    bond_dim: int = 128,
) -> AerSimulator:
    """Return an :class:`~qiskit_aer.AerSimulator` for MPS simulation on GPU.

    Falls back to CPU MPS when no GPU is available.

    Args:
        config: GPU cluster configuration.  ``None`` triggers auto-detection.
        bond_dim: Maximum MPS bond dimension.

    Returns:
        AerSimulator: Simulator ready for MPS simulation.
    """
    if config is None:
        config = GpuClusterConfig.auto_detect()

    base_kwargs: dict = {
        "method": "matrix_product_state",
        "matrix_product_state_max_bond_dimension": bond_dim,
    }

    if config.has_gpu:
        base_kwargs["device"] = "GPU"

    if config.mpi_enable and config.has_gpu:
        base_kwargs["mpi"] = True

    return AerSimulator(**base_kwargs)


def cluster_info() -> dict:
    """Return a summary dict of detected GPU cluster resources.

    Returns:
        dict: Keys ``num_gpus`` (int), ``mpi_world_size`` (int), and
        ``gpu_available`` (bool).
    """
    n = count_gpus()
    return {
        "num_gpus": n,
        "mpi_world_size": _mpi_world_size(),
        "gpu_available": n > 0,
    }
