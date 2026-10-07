# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "corpus"
    / "race"
    / "scripts"
    / "regenerate_asm.py"
)
SPEC = importlib.util.spec_from_file_location("race_regenerate_asm", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fake_hipcc(tmp_path: Path, monkeypatch, body: str) -> None:
    root = tmp_path / "rocm"
    compiler = root / "bin" / "hipcc"
    compiler.parent.mkdir(parents=True)
    compiler.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\nimport sys\n{body}\n"
    )
    compiler.chmod(0o755)
    monkeypatch.setenv("ROCM_PATH", str(root))


@pytest.mark.parametrize("returncode", [0, 1])
def test_failed_compilation_preserves_existing_assembly(tmp_path, monkeypatch, returncode):
    _fake_hipcc(tmp_path, monkeypatch, f"sys.exit({returncode})")
    monkeypatch.chdir(tmp_path)
    kernel = tmp_path / "kernel.hip"
    kernel.write_text("// source\n")
    out = tmp_path / "asm" / "gfx950" / "kernel.s"
    out.parent.mkdir(parents=True)
    out.write_text("previous pinned assembly\n")

    with pytest.raises(MODULE.CompileError):
        MODULE.compile_device_asm(kernel, "gfx950", out, [])

    assert out.read_text() == "previous pinned assembly\n"


@pytest.mark.parametrize("auto_named", [False, True])
def test_compilation_replaces_output_after_normalization(tmp_path, monkeypatch, auto_named):
    output_expr = (
        "Path('kernel-hip-amdgcn-amd-amdhsa-gfx950.s')"
        if auto_named
        else "Path(sys.argv[sys.argv.index('-o') + 1])"
    )
    _fake_hipcc(
        tmp_path,
        monkeypatch,
        f"{output_expr}.write_text('__hip_cuid_0123456789abcdef\\n')",
    )
    monkeypatch.chdir(tmp_path)
    kernel = Path("kernel.hip")
    kernel.write_text("// source\n")
    out = Path("asm/gfx950/kernel.s")
    out.parent.mkdir(parents=True)
    out.write_text("previous pinned assembly\n")

    MODULE.compile_device_asm(kernel, "gfx950", out, [])

    assert out.read_text() == "__hip_cuid_corpus\n"
    assert not Path("kernel-hip-amdgcn-amd-amdhsa-gfx950.s").exists()


@pytest.mark.parametrize("symlink", [False, True])
def test_compilation_preserves_existing_caller_output(tmp_path, monkeypatch, symlink):
    _fake_hipcc(tmp_path, monkeypatch, "sys.exit(1)")
    monkeypatch.chdir(tmp_path)
    kernel = Path("kernel.hip")
    kernel.write_text("// source\n")
    unrelated = Path("kernel-hip-amdgcn-amd-amdhsa-gfx950.s")
    if symlink:
        unrelated.symlink_to("missing-target")
    else:
        unrelated.write_text("another invocation's output\n")

    with pytest.raises(MODULE.CompileError):
        MODULE.compile_device_asm(kernel, "gfx950", Path("asm/gfx950/kernel.s"), [])

    if symlink:
        assert unrelated.is_symlink()
        assert unrelated.readlink() == Path("missing-target")
    else:
        assert unrelated.read_text() == "another invocation's output\n"


@pytest.mark.parametrize(
    ("committed", "fresh", "expected"),
    [
        ({"kernel.s": "same"}, {"kernel.s": "same"}, 0),
        ({"kernel.s": "old"}, {"kernel.s": "new"}, 1),
        ({}, {"kernel.s": "new"}, 1),
        ({"kernel.s": "same", "obsolete.s": "old"}, {"kernel.s": "same"}, 1),
        ({"obsolete.s": "old"}, {}, 1),
    ],
)
def test_drift_checks_both_artifact_inventories(tmp_path, monkeypatch, committed, fresh, expected):
    asm_dir = tmp_path / "committed"
    fresh_dir = tmp_path / "fresh"
    monkeypatch.setattr(MODULE, "ASM_DIR", asm_dir)
    for directory, files in ((asm_dir, committed), (fresh_dir, fresh)):
        for name, contents in files.items():
            path = directory / "gfx950" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)

    assert MODULE._report_drift(["gfx950"], fresh_dir) == expected


def test_regenerate_retains_compiler_diagnostic(tmp_path, monkeypatch):
    _fake_hipcc(
        tmp_path,
        monkeypatch,
        "sys.stderr.write('fatal error: rocwmma/rocwmma.hpp not found\\n'); sys.exit(1)",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MODULE, "KERNELS_DIR", tmp_path)
    monkeypatch.setattr(MODULE, "load_cases", lambda: [{"kernel": "kernel"}])
    (tmp_path / "kernel.hip").write_text("// source\n")

    written, failures = MODULE.regenerate(["gfx950"], [], tmp_path / "asm")

    assert written == 0
    assert len(failures) == 1
    assert "kernel.hip (gfx950)" in failures[0]
    assert "fatal error: rocwmma/rocwmma.hpp not found" in failures[0]


def test_cli_preserves_quoted_include_path(tmp_path, monkeypatch):
    _fake_hipcc(
        tmp_path,
        monkeypatch,
        "assert '-Iheaders with spaces' in sys.argv, sys.argv\n"
        "Path(sys.argv[sys.argv.index('-o') + 1]).write_text('assembly\\n')",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MODULE, "KERNELS_DIR", tmp_path)
    monkeypatch.setattr(MODULE, "load_cases", lambda: [{"kernel": "kernel"}])
    monkeypatch.setattr(MODULE, "ASM_DIR", tmp_path / "asm")
    monkeypatch.setattr(MODULE.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--check", "--target", "gfx950", '--cxxflags=-I"headers with spaces"'],
    )
    (tmp_path / "kernel.hip").write_text("// source\n")
    committed = tmp_path / "asm" / "gfx950" / "kernel.s"
    committed.parent.mkdir(parents=True)
    committed.write_text("assembly\n")

    assert MODULE.main() == 0


def test_cli_reports_unmatched_flag_quote(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), '--cxxflags=-I"unterminated'])
    monkeypatch.setattr(
        MODULE, "regenerate", lambda *args: pytest.fail("invalid quoting reached compilation")
    )

    with pytest.raises(SystemExit) as error:
        MODULE.main()

    assert error.value.code == 2
    assert "--cxxflags: No closing quotation" in capsys.readouterr().err
