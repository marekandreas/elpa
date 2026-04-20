#!/usr/bin/env python3
"""Generate a filtered Windows .def file for ELPA DLL exports.

Scans all .obj files under a given directory, extracts defined EXTERNAL
symbols via ``dumpbin /SYMBOLS``, and writes a .def file containing only
the public API symbols.

The allowlist is intentionally narrow — only symbols that consumers
(QuantumATK, test_package) and the test suite actually need are exported.
This was determined by scanning all 947 test executables with
``dumpbin /imports`` and collecting the union of referenced symbols.

Categories exported:
  * ``elpa_*``              — C public API (elpa.h / elpa_generic.h)
  * ``ELPA_mp_*``           — Fortran ELPA module (allocate/deallocate)
  * ``ELPA_API_mp_*``       — Fortran API module (strerr, value helpers)
  * ``FTIMINGS_mp_TIMER_*`` — timing infrastructure (used by test harness)
  * ``CUDA_FUNCTIONS_mp_*`` — GPU memory management (used by GPU tests)

Everything else (>1600 internal Fortran module procedures, SIMD kernels,
CRT leaks, Cannon's helpers) is kept private.

Usage:
    python generate_exports_def.py <obj_dir> <output.def> [dumpbin_path]

``obj_dir`` is searched recursively for ``*.obj`` files.
``dumpbin_path`` defaults to ``dumpbin`` (expected on PATH via VS tools).
"""

import re
import subprocess
import sys
from pathlib import Path

# Symbols matching ANY of these patterns (anchored) are KEPT.
# Order: most-specific first for clarity; evaluation is short-circuit OR.
_KEEP_PATTERNS = [
    re.compile(r"^elpa_"),  # C public API (lowercase)
    re.compile(r"^ELPA_mp_"),  # Fortran ELPA module
    re.compile(r"^ELPA_API_mp_"),  # Fortran ELPA_API module
    re.compile(r"^FTIMINGS_mp_TIMER_"),  # timing infrastructure
    re.compile(r"^CUDA_FUNCTIONS_mp_CUDA_"),  # CUDA memory management
    re.compile(r"^CUDA_FUNCTIONS_mp_CUBLAS_"),  # cuBLAS helpers
]


def _should_keep(symbol: str) -> bool:
    """Return True if *symbol* belongs to the public ELPA ABI."""
    return any(p.search(symbol) for p in _KEEP_PATTERNS)


def _extract_defined_externals(obj_files: list[Path], dumpbin: str) -> set[str]:
    """Return the set of defined EXTERNAL symbol names from *obj_files*."""
    symbols: set[str] = set()
    # Process in batches to stay within the Windows command-line length limit.
    batch_size = 40
    for i in range(0, len(obj_files), batch_size):
        batch = obj_files[i : i + batch_size]
        result = subprocess.run(
            [dumpbin, "/SYMBOLS"] + [str(f) for f in batch],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"dumpbin /SYMBOLS failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
        for line in result.stdout.splitlines():
            # Typical dumpbin /SYMBOLS line for a defined external:
            #   004 00000000 SECT2  notype ()    External     | elpa_allocate
            # UNDEF means the symbol is referenced but not defined here.
            if "External" in line and "UNDEF" not in line:
                parts = line.split("|")
                if len(parts) >= 2:
                    sym = parts[-1].strip()
                    if sym:
                        symbols.add(sym)
    return symbols


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <obj_dir> <output.def> [dumpbin]", file=sys.stderr)
        sys.exit(1)

    obj_dir = Path(sys.argv[1])
    def_file = Path(sys.argv[2])
    dumpbin = sys.argv[3] if len(sys.argv) > 3 else "dumpbin"

    obj_files = sorted(obj_dir.rglob("*.obj"))
    if not obj_files:
        print(f"ERROR: No .obj files found under {obj_dir}", file=sys.stderr)
        sys.exit(1)

    all_symbols = _extract_defined_externals(obj_files, dumpbin)
    keep = {s for s in all_symbols if _should_keep(s)}

    def_file.parent.mkdir(parents=True, exist_ok=True)
    with open(def_file, "w") as f:
        f.write("EXPORTS\n")
        for sym in sorted(keep):
            f.write(f"    {sym}\n")

    print(
        f"generate_exports_def: {len(keep)} exports "
        f"(filtered from {len(all_symbols)} defined externals, "
        f"excluded {len(all_symbols) - len(keep)})"
    )


if __name__ == "__main__":
    main()
