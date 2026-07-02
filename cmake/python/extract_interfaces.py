#!/usr/bin/env python3
"""Extract C and Fortran interface declarations from ELPA source files.

ELPA source files contain specially-marked comment lines that declare C and
Fortran interfaces.  The autotools build extracts them with grep+sed.  This
script replaces that pipeline for CMake.

Markers
-------
  !c>   – C interface declarations  (for elpa_generated.h)
  !c_o> – C interface (optional error arg variant)
  !c_no> – C interface (non-optional error arg variant)
  !f>   – Fortran interface from C sources
  #!f>  – Fortran interface from C preprocessor lines
  !pf>  – Public Fortran interface from C sources

Usage
-----
  python extract_interfaces.py --marker '!c>' --marker '!c_o>' --marker '!c_no>' \
         -o elpa/elpa_generated.h src/elpa_impl.F90 src/elpa_api.F90 ...

Each matching source line has the marker prefix stripped and the remainder
is appended to the output file.
"""

import argparse
import pathlib


def extract(
    sources: list[pathlib.Path], markers: list[str], output: pathlib.Path
) -> None:
    """Scan *sources* for lines starting with any of *markers* and write
    the extracted text (with markers stripped) to *output*.

    Markers are processed in order: first ALL lines for the first marker
    from all sources, then ALL lines for the second marker, etc.  This
    matches autotools behaviour where extract_interface is called once per
    marker."""
    lines: list[str] = []
    for marker in markers:
        for src in sources:
            text = src.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                stripped = line.lstrip()
                if stripped.startswith(marker):
                    lines.append(stripped[len(marker) :])
    # Patch complex-type macros for portability:
    #   C++ mode:  std::complex<T>  (elpa.h already includes <complex>)
    #   C/Windows: double _Complex   (clang-cl / MSVC with /std:c11)
    #   C/Linux:   double complex    (standard C99/C11)
    patched_lines: list[str] = []
    idx = 0
    while idx < len(lines):
        current = lines[idx].strip()
        next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
        if (
            current == "#define double_complex double complex"
            and next_line == "#define float_complex float complex"
        ):
            indent = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())]
            patched_lines.extend(
                [
                    f"{indent}#ifdef __cplusplus",
                    f"{indent}  #define double_complex std::complex<double>",
                    f"{indent}  #define float_complex std::complex<float>",
                    f"{indent}#elif defined(_WIN32)",
                    f"{indent}  #define double_complex double _Complex",
                    f"{indent}  #define float_complex float _Complex",
                    f"{indent}#else",
                    f"{indent}  #define double_complex double complex",
                    f"{indent}  #define float_complex float complex",
                    f"{indent}#endif",
                ]
            )
            idx += 2
            continue

        patched_lines.append(lines[idx])
        idx += 1

    content = "\n".join(patched_lines) + ("\n" if patched_lines else "")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--marker",
        "-m",
        action="append",
        required=True,
        help="Comment marker to extract (may be repeated)",
    )
    ap.add_argument(
        "--output", "-o", required=True, type=pathlib.Path, help="Output header file"
    )
    ap.add_argument(
        "sources", nargs="+", type=pathlib.Path, help="Source files to scan"
    )
    args = ap.parse_args()
    extract(args.sources, args.marker, args.output)


if __name__ == "__main__":
    main()
