"""Driver-level tests that do not require IREE."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import compile_litert

ROOT = Path(__file__).resolve().parents[1]

STATIC_MODULE = """
module {
  func.func @main(%x: tensor<1x8x2x4xf32>) -> tensor<1x8x2x4xf32> {
    return %x : tensor<1x8x2x4xf32>
  }
}
"""

DYNAMIC_MODULE = """
module {
  func.func @main(%x: tensor<?x4xf32>) -> tensor<?x4xf32> {
    return %x : tensor<?x4xf32>
  }
}
"""

ATTENTION_MODULE = """
module {
  func.func @attention(%q: tensor<1x8x2x4xf32>, %k: tensor<1x8x2x4xf32>,
                       %v: tensor<1x8x2x4xf32>) -> tensor<1x8x2x4xf32> {
    %0 = tensor.empty() : tensor<1x8x2x4xf32>
    return %0 : tensor<1x8x2x4xf32>
  }
}
"""


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def test_deep_merge() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"c": 4}, "e": 5}
    assert compile_litert.deep_merge(base, override) == {
        "a": {"b": 1, "c": 4},
        "d": 3,
        "e": 5,
    }


@pytest.mark.parametrize(
    "value, expected",
    [(0, "0 B"), (512, "512 B"), (1024, "1.0 KiB"), (1536, "1.5 KiB"), (1 << 20, "1.0 MiB")],
)
def test_human_bytes(value: int, expected: str) -> None:
    assert compile_litert.human_bytes(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [("abc", "abc"), ("a b", "'a b'"), ("a'b", "'a'\\''b'"), ("", "''")],
)
def test_shlex_quote(value: str, expected: str) -> None:
    assert compile_litert.shlex_quote(value) == expected


def test_substitutions_reject_unknown_placeholder() -> None:
    substitutions = compile_litert._Substitutions(tile_bytes=1)
    assert substitutions["tile_bytes"] == 1
    with pytest.raises(KeyError, match="unknown placeholder"):
        substitutions["missing"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_print_config_default(capsys) -> None:
    assert compile_litert.main(["--print-config"]) == 0
    config = json.loads(capsys.readouterr().out)
    assert config["target"]["name"] == "coral-npu"
    assert config["memory"]["tile_bytes"] == 256 * 1024


def test_coralnpu_profile_uses_hal_device(capsys) -> None:
    assert compile_litert.main(["--print-config", "--profile", "coralnpu"]) == 0
    config = json.loads(capsys.readouterr().out)
    flags = config["target"]["compile_flags"]
    assert "--iree-hal-target-device=coralnpu" in flags
    assert not any("riscv32" in flag for flag in flags)


def test_coralnpu_config_file(capsys) -> None:
    config_path = ROOT / "scripts" / "coralnpu.json"
    assert compile_litert.main(["--print-config", "--config", str(config_path)]) == 0
    config = json.loads(capsys.readouterr().out)
    assert "--iree-hal-target-device=coralnpu" in config["target"]["compile_flags"]
    assert config["memory"]["tile_bytes"] == 262144


def test_config_overrides_profile_fields(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "npu.json"
    config_path.write_text(
        json.dumps({"target": {"iree_compile": "/opt/coral/iree-compile"}})
    )
    assert (
        compile_litert.main(
            [
                "--print-config",
                "--profile",
                "coralnpu",
                "--config",
                str(config_path),
            ]
        )
        == 0
    )
    config = json.loads(capsys.readouterr().out)
    assert config["target"]["iree_compile"] == "/opt/coral/iree-compile"
    assert "--iree-hal-target-device=coralnpu" in config["target"]["compile_flags"]


def test_load_config_cli_overrides() -> None:
    args = compile_litert.build_parser().parse_args(
        [
            "model.tflite",
            "--budget-bytes",
            "1024",
            "--safety-factor",
            "2.0",
            "--q-tile",
            "2",
            "--kv-tile",
            "4",
            "--layout",
            "bhsd",
            "--no-causal",
            "--iree-compile",
            "/tmp/iree-compile",
        ]
    )
    config = compile_litert.load_config(args)
    assert config["memory"]["tile_bytes"] == 1024
    assert config["memory"]["safety_factor"] == 2.0
    assert config["attention"]["q_tile"] == 2
    assert config["attention"]["kv_tile"] == 4
    assert config["attention"]["layout"] == "bhsd"
    assert config["attention"]["causal"] is False
    assert config["target"]["iree_compile"] == "/tmp/iree-compile"


def test_load_config_missing_file() -> None:
    with pytest.raises(SystemExit, match="config not found"):
        compile_litert.main(["--print-config", "--config", "/definitely/missing.json"])


# ---------------------------------------------------------------------------
# Bundle extraction and ingest
# ---------------------------------------------------------------------------


def test_extract_tflite_from_zip(tmp_path: Path) -> None:
    bundle = tmp_path / "model.litertlm"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("metadata.json", "{}")
        archive.writestr("model.tflite", b"TFL3payload")
    target = compile_litert.extract_tflite(bundle, tmp_path)
    assert target.name == "extracted_model.tflite"
    assert target.read_bytes() == b"TFL3payload"


def test_extract_tflite_from_marker(tmp_path: Path) -> None:
    bundle = tmp_path / "model.bin"
    bundle.write_bytes(b"\x00\x00\x00\x00TFL3payload")
    target = compile_litert.extract_tflite(bundle, tmp_path)
    assert target.read_bytes() == b"\x00\x00\x00\x00TFL3payload"


def test_extract_tflite_rejects_unknown_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "model.bin"
    bundle.write_bytes(b"not a model")
    with pytest.raises(SystemExit, match="no TFLite flatbuffer"):
        compile_litert.extract_tflite(bundle, tmp_path)


def test_ingest_mlir(tmp_path: Path) -> None:
    model = write(tmp_path, "model.mlir", STATIC_MODULE)
    raw = tmp_path / "01_tosa.mlir"
    text = compile_litert.ingest(model, raw, compile_litert.DEFAULTS, dry_run=True)
    assert text == STATIC_MODULE
    assert raw.read_text() == STATIC_MODULE


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def test_stop_after_bind_binds_dynamic_extent(tmp_path: Path, capsys) -> None:
    model = write(tmp_path, "model.mlir", DYNAMIC_MODULE)
    out = tmp_path / "out"
    assert (
        compile_litert.main(
            [
                str(model),
                "--seq-len",
                "8",
                "--out-dir",
                str(out),
                "--stop-after",
                "bind",
            ]
        )
        == 0
    )
    static = (out / "02_static.mlir").read_text()
    assert "tensor<8x4xf32>" in static
    assert "tensor<?x4xf32>" not in static
    # Argument, function result, and return type: three textual extents.
    assert "bound 3 dynamic extent(s) to 8" in capsys.readouterr().out


def test_dynamic_graph_requires_seq_len(tmp_path: Path) -> None:
    model = write(tmp_path, "model.mlir", DYNAMIC_MODULE)
    with pytest.raises(SystemExit, match="pass --seq-len"):
        compile_litert.main(
            [str(model), "--out-dir", str(tmp_path / "out"), "--stop-after", "bind"]
        )


def test_stop_after_attention_writes_artifacts(tmp_path: Path, capsys) -> None:
    model = write(tmp_path, "model.mlir", ATTENTION_MODULE)
    out = tmp_path / "out"
    assert (
        compile_litert.main(
            [
                str(model),
                "--out-dir",
                str(out),
                "--stop-after",
                "attention",
                "--attention-func",
                "attention",
                "--q-tile",
                "2",
                "--kv-tile",
                "2",
                "--budget-bytes",
                str(64 * 1024),
            ]
        )
        == 0
    )
    tiled = (out / "03_tiled.mlir").read_text()
    kernel = (out / "mhsa_tiled.mlir").read_text()
    plan = json.loads((out / "tile_plan.json").read_text())

    assert "func.func @mhsa_tiled_attention" in tiled
    assert "func.call @mhsa_tiled_attention" in tiled
    assert "func.func @mhsa_tiled_attention" in kernel
    assert plan["function"] == "attention"
    assert plan["tiles"]["q_tile"] == 2
    assert plan["tiles"]["kv_tile"] == 2
    assert plan["tiles"]["unrolled_steps"] == 16  # 4 query tiles x 4 KV tiles
    assert plan["budget"]["fits"] is True
    assert "function   @attention" in capsys.readouterr().out


def test_skip_attention_leaves_graph_untiled(tmp_path: Path) -> None:
    model = write(tmp_path, "model.mlir", ATTENTION_MODULE)
    out = tmp_path / "out"
    assert (
        compile_litert.main(
            [
                str(model),
                "--out-dir",
                str(out),
                "--stop-after",
                "attention",
                "--skip-attention",
            ]
        )
        == 0
    )
    assert (out / "03_tiled.mlir").read_text() == (out / "02_static.mlir").read_text()
    assert not (out / "mhsa_tiled.mlir").exists()
    assert not (out / "tile_plan.json").exists()


def test_missing_attention_function_is_actionable(tmp_path: Path) -> None:
    model = write(tmp_path, "model.mlir", STATIC_MODULE)
    with pytest.raises(SystemExit, match="no multi-head attention function found"):
        compile_litert.main(
            [
                str(model),
                "--out-dir",
                str(tmp_path / "out"),
                "--stop-after",
                "attention",
            ]
        )


def test_allow_untiled_attention(tmp_path: Path, capsys) -> None:
    model = write(tmp_path, "model.mlir", STATIC_MODULE)
    out = tmp_path / "out"
    assert (
        compile_litert.main(
            [
                str(model),
                "--out-dir",
                str(out),
                "--stop-after",
                "attention",
                "--allow-untiled-attention",
            ]
        )
        == 0
    )
    assert "leaving the imported attention graph untiled" in capsys.readouterr().out


def test_attention_func_not_found(tmp_path: Path) -> None:
    model = write(tmp_path, "model.mlir", ATTENTION_MODULE)
    with pytest.raises(SystemExit, match="attention function @missing not found"):
        compile_litert.main(
            [
                str(model),
                "--out-dir",
                str(tmp_path / "out"),
                "--stop-after",
                "attention",
                "--attention-func",
                "missing",
            ]
        )


def test_budget_too_small_fails_before_compile(tmp_path: Path) -> None:
    model = write(tmp_path, "model.mlir", ATTENTION_MODULE)
    with pytest.raises(SystemExit, match="cannot fit @attention"):
        compile_litert.main(
            [
                str(model),
                "--out-dir",
                str(tmp_path / "out"),
                "--stop-after",
                "attention",
                "--budget-bytes",
                "1",
            ]
        )


def test_list_functions(tmp_path: Path, capsys) -> None:
    model = write(tmp_path, "model.mlir", ATTENTION_MODULE)
    assert (
        compile_litert.main(
            [
                str(model),
                "--out-dir",
                str(tmp_path / "out"),
                "--list-functions",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "@attention" in output
    assert "<- attention" in output


def test_report_tensors(tmp_path: Path, capsys) -> None:
    model = write(tmp_path, "model.mlir", STATIC_MODULE)
    assert (
        compile_litert.main(
            [
                str(model),
                "--out-dir",
                str(tmp_path / "out"),
                "--stop-after",
                "bind",
                "--report-tensors",
            ]
        )
        == 0
    )
    assert "largest tensor types" in capsys.readouterr().out


def test_print_plan(tmp_path: Path, capsys) -> None:
    model = write(tmp_path, "model.mlir", ATTENTION_MODULE)
    assert (
        compile_litert.main(
            [
                str(model),
                "--out-dir",
                str(tmp_path / "out"),
                "--stop-after",
                "attention",
                "--print-plan",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"tiles"' in output
    assert '"budget"' in output


def test_dry_run_compile_does_not_invoke_tools(tmp_path: Path, capsys) -> None:
    model = write(tmp_path, "model.mlir", STATIC_MODULE)
    out = tmp_path / "out"
    assert (
        compile_litert.main(
            [
                str(model),
                "--out-dir",
                str(out),
                "--stop-after",
                "compile",
                "--skip-attention",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "iree-opt --verify-each" in output
    assert "iree-compile" in output
    assert not list(out.glob("*.vmfb"))


def test_dry_run_uses_coralnpu_profile(tmp_path: Path, capsys) -> None:
    model = write(tmp_path, "model.mlir", STATIC_MODULE)
    assert (
        compile_litert.main(
            [
                str(model),
                "--out-dir",
                str(tmp_path / "out"),
                "--stop-after",
                "compile",
                "--skip-attention",
                "--profile",
                "coralnpu",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "--iree-hal-target-device=coralnpu" in output
    assert "riscv32-unknown-elf" not in output


def test_missing_input_model(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="input model not found"):
        compile_litert.main([str(tmp_path / "missing.tflite"), "--dry-run"])
