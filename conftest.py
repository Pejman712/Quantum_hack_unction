import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--bond-dim",
        type=int,
        default=128,
        help="MPS max bond dimension for validation tests (default: 128).",
    )
    parser.addoption(
        "--device",
        default="CPU",
        help="Aer device for validation tests: CPU (default) or GPU.",
    )
