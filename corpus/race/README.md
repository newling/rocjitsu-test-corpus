# Race baseline assembly

The corpus pins device assembly so instruction-dependent tests have stable
inputs across compiler updates. Regeneration is a deliberate change: review the
resulting instruction streams and revalidate downstream mutation expectations.
Mutation generation and its semantic expectations are introduced separately.

`kernels/<kernel>.hip` contains the HIP source and host harness;
`asm/<target>/<kernel>.s` contains the pinned device assembly. `cases.toml` is
the shared inventory: `id` is the stable case selector, `kernel` names the
source, and `requires` restricts the target set. An empty `requires` means every
target under `asm/`. Optional `headers` list additional header prerequisites.

## Check the inventory

With Python 3.11 or later, run this command from the repository root:

```bash
python corpus/race/scripts/check_cases.py
```

It checks case fields and uniqueness, required target directories, source and
assembly presence, and assembly ownership. It needs no HIP compiler. The
inventory check and regeneration regressions also run through pytest:

```bash
python -m pytest tests/test_check_cases.py tests/test_regenerate_asm.py
```

## Regenerate with a HIP compiler

Set `ROCM_PATH` to a ROCm SDK root, or make `rocm-sdk path --root` available.
The script also recognizes `/opt/rocm`. It selects `bin/hipcc`, then
`llvm/bin/clang++`, with a final `hipcc` lookup on `PATH`. The SDK needs the
selected targets' compiler support and device libraries. The `wmma-rocwmma`
case also needs rocWMMA headers.

```bash
python corpus/race/scripts/regenerate_asm.py --target gfx950 --target gfx1250
python corpus/race/scripts/check_cases.py
```

Omit `--target` to use the existing assembly targets and targets explicitly
declared in the manifest. Supply additional compiler flags with `--cxxflags`:

```bash
python corpus/race/scripts/regenerate_asm.py --target gfx1250 \
  --cxxflags='-I"third party/rocwmma/include"'
```

Flags use shell-style quoting without executing a shell. Relative `-I`,
`-isystem`, `-iquote`, and `-include` paths are interpreted from the caller's
directory before compilation moves to a private temporary directory. Header
preflight examines the SDK's `include/` and explicit `-I`, `-isystem`, and
`-iquote` directories; supply external rocWMMA headers through one of these.

Each successful compilation atomically replaces its pinned file. A failed
compilation preserves that file and reports the compiler diagnostic; earlier
successful replacements in the same run remain. Update mode skips cases with
missing declared headers and preserves their pinned files, so inspect skip
messages before committing a regeneration.

## Compare without updating the pinned files

```bash
python corpus/race/scripts/regenerate_asm.py --check
```

This regenerates into temporary storage and compares the selected case/target
inventory and assembly text. Missing or obsolete files count as drift.
Exit status 0 means the complete check matched; 1 reports drift; 2 reports
compilation failure, skipped cases, or no selected cases. Temporary output is
removed on completion, including failed checks.

The compiler's `.ident` line remains in each artifact. A change only to that
line is reported separately and does not fail the comparison. All other text,
including directives and metadata, participates in comparison after the random
`__hip_cuid_<hash>` symbol is normalized. Host linking uses the separate fatbin
symbol from the host object; pinned device assembly still needs compatible
assembler/linker tools and a HIP compiler for the host harness.

The compiler is discovered locally, not downloaded or version-enforced. For
exact reproduction, use the compiler revision recorded in `.ident`, matching
HIP/rocWMMA headers, and the same compiler flags. These commands are manual
tools; this repository does not currently configure them as CI gates. A
consumer can invoke them in CI with a pinned SDK and explicit prerequisites.
