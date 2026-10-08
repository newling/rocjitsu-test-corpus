# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "corpus" / "race" / "scripts" / "check_cases.py"
spec = importlib.util.spec_from_file_location("check_cases", SCRIPT)
assert spec is not None and spec.loader is not None
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_committed_inventory():
    assert checker.main() == 0


@pytest.fixture
def inventory(monkeypatch, tmp_path):
    kernels, assembly = tmp_path / "kernels", tmp_path / "asm"
    kernels.mkdir()
    (kernels / "k.hip").write_text("// Kernel source\n")
    targets = ["gfx950", "gfx1250"]
    for target in targets:
        (assembly / target).mkdir(parents=True)
        (assembly / target / "k.s").write_text("s_endpgm\n")
    monkeypatch.setattr(checker, "KERNELS_DIR", kernels)
    monkeypatch.setattr(checker, "ASM_DIR", assembly)
    return [{"id": "k", "kernel": "k", "requires": []}], targets


def test_portable_case_owns_each_target(inventory):
    cases, targets = inventory
    assert checker.check(cases, targets) == []


def test_target_specific_case_and_optional_header(inventory):
    cases, targets = inventory
    cases[0].update(requires=["gfx1250"], headers=["rocwmma/rocwmma.hpp"])
    (checker.ASM_DIR / "gfx950" / "k.s").unlink()
    assert checker.check(cases, targets) == []


def test_mutation_fields_are_not_part_of_inventory(inventory):
    cases, targets = inventory
    cases[0]["mutate"] = "wait"
    assert "cases.toml [k]: unknown field 'mutate'" in checker.check(cases, targets)


def test_duplicate_case_id_and_kernel(inventory):
    cases, targets = inventory
    errors = checker.check(cases + cases, targets)
    assert "cases.toml [k]: duplicate id" in errors
    assert "cases.toml [k]: kernel k is already used by another case" in errors


def test_missing_source(inventory):
    cases, targets = inventory
    (checker.KERNELS_DIR / "k.hip").unlink()
    assert "cases.toml [k]: kernels/k.hip is missing" in checker.check(cases, targets)


def test_missing_assembly(inventory):
    cases, targets = inventory
    (checker.ASM_DIR / "gfx950" / "k.s").unlink()
    assert "cases.toml [k]: asm/gfx950/k.s is missing" in checker.check(cases, targets)


def test_required_target_has_no_directory(inventory):
    cases, targets = inventory
    cases[0]["requires"] = ["gfx9999"]
    assert (
        "cases.toml [k]: requires gfx9999, which has no asm/gfx9999/ directory"
        in checker.check(cases, targets)
    )


def test_assembly_on_unsupported_case_target(inventory):
    cases, targets = inventory
    cases[0]["requires"] = ["gfx1250"]
    assert "asm/gfx950/k.s: no case builds k for gfx950" in checker.check(cases, targets)


def test_obsolete_assembly_has_no_owner(inventory):
    cases, targets = inventory
    (checker.ASM_DIR / "gfx950" / "obsolete.s").write_text("s_endpgm\n")
    assert (
        "asm/gfx950/obsolete.s: no case builds obsolete for gfx950"
        in checker.check(cases, targets)
    )


def test_main_rejects_unsupported_schema(monkeypatch, tmp_path, capsys):
    manifest = tmp_path / "cases.toml"
    manifest.write_text("[corpus]\nschema = 2\n")
    monkeypatch.setattr(checker, "CASES_PATH", manifest)
    assert checker.main() == 2
    assert "unsupported corpus schema" in capsys.readouterr().err
