import matplotlib
import pytest

# Use a non-interactive backend so circuit drawing works headless (e.g. in CI).
matplotlib.use("Agg")


@pytest.fixture
def bond_dim(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--bond-dim")


@pytest.fixture
def device(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--device")
