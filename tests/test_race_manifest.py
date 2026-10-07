# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import re
import tomllib


CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus" / "race"
CASES = tomllib.loads((CORPUS_ROOT / "cases.toml").read_text())["case"]


def test_manifest_matches_pinned_assembly_inventory():
    assert len({case["id"] for case in CASES}) == len(CASES)
    assert {case["kernel"] for case in CASES} == {
        path.stem for path in (CORPUS_ROOT / "kernels").glob("*.hip")
    }
    for directory in (CORPUS_ROOT / "asm").iterdir():
        if not directory.is_dir():
            continue
        expected = {
            case["kernel"]
            for case in CASES
            if not case["requires"] or directory.name in case["requires"]
        }
        assert {path.stem for path in directory.glob("*.s")} == expected


def test_gfx1250_declared_waits_exist_in_pinned_assembly():
    # The manifest currently uses gfx1250 counter names even for portable
    # cases. Validate those names here; a gfx950 mapping needs its own contract.
    missing = {}
    for case in CASES:
        if case["requires"] and "gfx1250" not in case["requires"]:
            continue
        assembly = (CORPUS_ROOT / "asm" / "gfx1250" / f"{case['kernel']}.s").read_text()
        present = {
            counter
            for mnemonic in re.findall(r"^\s+s_wait_([a-z_]+)\s", assembly, re.MULTILINE)
            for counter in mnemonic.split("_")
        }
        if absent := set(case["waits"]) - present:
            missing[case["id"]] = sorted(absent)

    assert not missing, f"Declared wait counters absent from gfx1250 assembly: {missing}"


def test_wait_exemptions_name_existing_pinned_sites():
    for case in CASES:
        for exemption in case.get("exempt", []):
            if exemption["kind"] != "wait":
                continue
            for directory in (CORPUS_ROOT / "asm").iterdir():
                if not directory.is_dir():
                    continue
                if case["requires"] and directory.name not in case["requires"]:
                    continue
                assembly = (directory / f"{case['kernel']}.s").read_text()
                sites = [
                    mnemonic
                    for mnemonic in re.findall(r"^\s+(s_wait\w+)\s+\S+", assembly, re.MULTILINE)
                    if mnemonic != "s_wait_xcnt"
                ]
                ordinal = exemption["ordinal"]
                assert 0 <= ordinal < len(sites), (case["id"], directory.name, ordinal)
                assert sites[ordinal] == exemption["site"], (case["id"], directory.name, ordinal)
