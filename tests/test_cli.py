from pathlib import Path

from quantum_hack.cli import DEFAULT_QASM, build_parser, main
from quantum_hack.viz import DEFAULT_OUTPUT_DIR

QASM_BELL = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
"""


def test_build_parser_defaults():
    args = build_parser().parse_args([])
    assert args.qasm == DEFAULT_QASM
    assert args.out_dir == DEFAULT_OUTPUT_DIR
    assert args.timestamp is True


def test_build_parser_custom_arguments():
    args = build_parser().parse_args(
        ["foo.qasm", "--out-dir", "drawings", "--no-timestamp"]
    )
    assert args.qasm == Path("foo.qasm")
    assert args.out_dir == Path("drawings")
    assert args.timestamp is False


def test_main_writes_three_drawings(tmp_path, capsys):
    qasm = tmp_path / "bell.qasm"
    qasm.write_text(QASM_BELL)
    out_dir = tmp_path / "out"

    main([str(qasm), "--out-dir", str(out_dir), "--no-timestamp"])

    for name in ("circuit", "transpiled_circuit", "optimized_circuit"):
        png = out_dir / f"{name}.png"
        assert png.exists() and png.stat().st_size > 0

    out = capsys.readouterr().out
    assert out.count("Saved ") == 3
