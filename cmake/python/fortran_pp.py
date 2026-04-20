#!/usr/bin/env python3
"""Traditional C preprocessor for ELPA Fortran sources.

Replaces ``cpp -P -traditional`` on platforms where a GNU-compatible
``cpp`` is not available (notably Windows without MSYS2).

Key behaviour that distinguishes *traditional* preprocessing from
standard (ANSI) C preprocessing:

* Macro expansion is purely textual — identifiers inside literal
  tokens such as ``1.0_rk`` are expanded when ``_rk`` is a defined
  macro.  An ANSI preprocessor treats ``1.0_rk`` as one pp-number
  and never expands ``_rk``.

* ``#include`` resolves quoted paths relative to the file that
  contains the directive, then falls back to ``-I`` search paths.

Supports both object-like macros (``#define FOO bar``) and
function-like macros (``#define FOO(x, y) x + y``).

Usage::

    python fortran_pp.py -DHAVE_CONFIG_H -DFOO=BAR -I/inc -o out.F90 in.F90
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Macro table
# ---------------------------------------------------------------------------
class _FuncMacro:
    """A function-like macro with parameter names and a body template."""

    __slots__ = ("params", "body")

    def __init__(self, params: list[str], body: str) -> None:
        self.params = params
        self.body = body


class MacroTable:
    """Stores object-like and function-like macro definitions."""

    def __init__(self) -> None:
        # Object-like: name → str value
        # Function-like: name → _FuncMacro
        self._defs: dict[str, str | _FuncMacro] = {}

    def define(self, name: str, value: str = "") -> None:
        self._defs[name] = value

    def define_func(self, name: str, params: list[str], body: str) -> None:
        self._defs[name] = _FuncMacro(params, body)

    def undef(self, name: str) -> None:
        self._defs.pop(name, None)

    def defined(self, name: str) -> bool:
        return name in self._defs

    def get(self, name: str) -> str | _FuncMacro | None:
        return self._defs.get(name)

    # Regex for C identifiers.
    _IDENT_RE = re.compile(r"[A-Za-z_]\w*")

    def expand(self, text: str, *, line_number: int = 0) -> str:
        """Textual macro expansion (traditional mode).

        Scans *text* for identifiers and replaces each one that is a
        defined macro with its value.  Traditional mode expands inside
        tokens like ``1.0_rk`` (``_rk`` is at an identifier boundary
        after the digit) but does NOT replace substrings of larger
        identifiers (``C_REAL_DATATYPE`` is one identifier, so
        ``REAL_DATATYPE`` is not matched inside it).

        Function-like macros are expanded when followed by ``(``.
        ``__LINE__`` is replaced with the current source line number.
        """
        for _ in range(20):  # expansion depth limit
            parts: list[str] = []
            pos = 0
            changed = False

            while pos < len(text):
                m = self._IDENT_RE.search(text, pos)
                if m is None:
                    parts.append(text[pos:])
                    break

                ident = m.group()

                # __LINE__ pseudo-macro
                if ident == "__LINE__":
                    parts.append(text[pos : m.start()])
                    parts.append(str(line_number))
                    pos = m.end()
                    changed = True
                    continue

                defn = self._defs.get(ident)
                if defn is None:
                    parts.append(text[pos : m.end()])
                    pos = m.end()
                    continue

                if isinstance(defn, _FuncMacro):
                    # Function-like macro — must be followed by '('
                    rest = text[m.end() :]
                    rest_stripped = rest.lstrip()
                    if not rest_stripped.startswith("("):
                        # Not an invocation — output as-is
                        parts.append(text[pos : m.end()])
                        pos = m.end()
                        continue
                    # Find the opening paren in original text
                    ws_len = len(rest) - len(rest_stripped)
                    paren_start = m.end() + ws_len
                    args_str, end_pos = _extract_paren_args(text, paren_start)
                    if args_str is None:
                        parts.append(text[pos : m.end()])
                        pos = m.end()
                        continue
                    args = _split_args(args_str)
                    # Substitute params in body using identifier-aware replacement
                    body = defn.body
                    for param, arg in zip(defn.params, args):
                        body = _replace_param(body, param, arg)
                    parts.append(text[pos : m.start()])
                    parts.append(body)
                    pos = end_pos
                    changed = True
                else:
                    # Object-like macro
                    parts.append(text[pos : m.start()])
                    parts.append(defn)
                    pos = m.end()
                    changed = True

            if not changed:
                break
            text = "".join(parts)
        return text


def _replace_param(body: str, param: str, arg: str) -> str:
    """Replace *param* with *arg* in *body* at identifier boundaries only."""
    # Use regex to match the param as a complete identifier
    return re.sub(
        r"(?<![A-Za-z0-9_])" + re.escape(param) + r"(?![A-Za-z0-9_])", arg, body
    )


def _extract_paren_args(text: str, start: int) -> tuple[str | None, int]:
    """Extract the text between balanced parentheses starting at *start*.

    Respects string literals (both ``"..."`` and ``'...'``).
    Returns (inner_text, position_after_closing_paren) or (None, 0).
    """
    assert text[start] == "("
    depth = 0
    in_str: str | None = None
    for i in range(start, len(text)):
        ch = text[i]
        if in_str is not None:
            if ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    return None, 0


def _split_args(args_str: str) -> list[str]:
    """Split comma-separated macro arguments respecting nested parens and strings."""
    args: list[str] = []
    depth = 0
    in_str: str | None = None  # current string delimiter or None
    current: list[str] = []
    for ch in args_str:
        if in_str is not None:
            current.append(ch)
            if ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
            current.append(ch)
            continue
        if ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            current.append(ch)
    args.append("".join(current).strip())
    return args


# ---------------------------------------------------------------------------
# Conditional evaluation
# ---------------------------------------------------------------------------
_DEFINED_RE = re.compile(r"defined\s*\(\s*(\w+)\s*\)")
_IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b")


def _eval_condition(expr: str, macros: MacroTable) -> bool:
    """Evaluate a ``#if`` / ``#elif`` expression."""

    # Replace defined(X) with 1/0
    def _replace_defined(m: re.Match) -> str:
        return "1" if macros.defined(m.group(1)) else "0"

    expr = _DEFINED_RE.sub(_replace_defined, expr)

    # Replace remaining identifiers with their macro value or 0
    def _replace_ident(m: re.Match) -> str:
        val = macros.get(m.group(1))
        if val is None:
            return "0"
        if isinstance(val, _FuncMacro):
            return "0"
        # Try to resolve to an integer
        try:
            int(val)
            return val
        except ValueError:
            return "0"

    expr = _IDENT_RE.sub(_replace_ident, expr)

    # Normalise C logical operators to Python
    expr = expr.replace("||", " or ").replace("&&", " and ")
    expr = re.sub(r"(?<![!=<>])!(?!=)", " not ", expr)

    try:
        return bool(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
_DIRECTIVE_RE = re.compile(r"^\s*#\s*(\w+)\s*(.*)")
# Matches: name(params) body  OR  name body  OR  name
_DEFINE_FUNC_RE = re.compile(r"(\w+)\(([^)]*)\)\s*(.*)")
_DEFINE_OBJ_RE = re.compile(r"(\w+)(?:\s+(.*))?")
_INCLUDE_QUOTED_RE = re.compile(r'"([^"]+)"')
_INCLUDE_ANGLE_RE = re.compile(r"<([^>]+)>")


def _join_continuations(lines: list[str]) -> list[str]:
    """Join lines ending with backslash (line continuation)."""
    result: list[str] = []
    buf: list[str] = []
    for line in lines:
        if line.rstrip().endswith("\\"):
            buf.append(line.rstrip()[:-1])
        else:
            if buf:
                buf.append(line)
                result.append("".join(buf))
                buf = []
            else:
                result.append(line)
    if buf:
        result.append("".join(buf))
    return result


def preprocess(
    source: Path,
    include_dirs: list[Path],
    macros: MacroTable,
    *,
    _depth: int = 0,
    _included_files: set[Path] | None = None,
) -> list[str]:
    """Preprocess *source* and return a list of output lines.

    *_included_files* collects the resolved paths of every file opened
    during preprocessing (the root source plus all transitively
    ``#include``d files).  The caller can inspect this set after the
    call to produce a dependency file.
    """
    if _depth > 50:
        print(f"fortran_pp: include depth exceeded at {source}", file=sys.stderr)
        return []

    if _included_files is not None:
        _included_files.add(source)

    raw_lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = _join_continuations(raw_lines)
    result: list[str] = []

    # Conditional stack: list of (active, else_seen) tuples.
    # *active* means the current branch is being processed.
    cond_stack: list[tuple[bool, bool]] = []

    def _active() -> bool:
        return all(c[0] for c in cond_stack)

    # Track source line numbers (approximate — continuations collapse lines)
    line_number = 0
    for raw_line in lines:
        line_number += 1
        # C null directive: a line containing only '#' (and optional whitespace).
        # It is a legal no-op in C preprocessing.  _DIRECTIVE_RE requires \w+
        # after '#' so it does NOT match a bare '#' — without this guard the
        # bare '#' falls through to the normal-line path and is emitted
        # verbatim, causing ifort warning #5117.  Consume silently.
        if raw_line.strip() == "#":
            continue
        m = _DIRECTIVE_RE.match(raw_line)
        if m:
            directive, rest = m.group(1), m.group(2).strip()

            # --- Conditional directives (always processed) ---
            if directive == "ifdef":
                cond_stack.append((macros.defined(rest) if _active() else False, False))
                continue
            if directive == "ifndef":
                cond_stack.append(
                    (not macros.defined(rest) if _active() else False, False)
                )
                continue
            if directive == "if":
                cond_stack.append(
                    (_eval_condition(rest, macros) if _active() else False, False)
                )
                continue
            if directive == "elif":
                if cond_stack:
                    prev_active, _ = cond_stack[-1]
                    # elif is taken only if no prior branch was taken and
                    # all enclosing conditions are active.
                    parent_active = all(c[0] for c in cond_stack[:-1])
                    if prev_active:
                        # A prior branch was taken — skip this and all
                        # subsequent elif/else.
                        cond_stack[-1] = (False, True)
                    elif parent_active:
                        cond_stack[-1] = (_eval_condition(rest, macros), False)
                continue
            if directive == "else":
                if cond_stack:
                    prev_active, else_seen = cond_stack[-1]
                    parent_active = all(c[0] for c in cond_stack[:-1])
                    if else_seen or prev_active:
                        cond_stack[-1] = (False, True)
                    else:
                        cond_stack[-1] = (parent_active, True)
                continue
            if directive == "endif":
                if cond_stack:
                    cond_stack.pop()
                continue

            # --- Non-conditional directives (only when active) ---
            if not _active():
                continue

            if directive == "define":
                # Try function-like macro: name(params) body
                fm = _DEFINE_FUNC_RE.match(rest)
                if fm:
                    name = fm.group(1)
                    params = [p.strip() for p in fm.group(2).split(",") if p.strip()]
                    body = fm.group(3).rstrip()
                    macros.define_func(name, params, body)
                else:
                    # Object-like macro: name value
                    dm = _DEFINE_OBJ_RE.match(rest)
                    if dm:
                        macros.define(dm.group(1), (dm.group(2) or "").rstrip())
                continue

            if directive == "undef":
                macros.undef(rest.split()[0] if rest else "")
                continue

            if directive == "include":
                # Try quoted include first, then angle-bracket
                im = _INCLUDE_QUOTED_RE.search(rest)
                if not im:
                    im = _INCLUDE_ANGLE_RE.search(rest)
                if im:
                    inc_path = _resolve_include(im.group(1), source, include_dirs)
                    if inc_path:
                        result.extend(
                            preprocess(
                                inc_path,
                                include_dirs,
                                macros,
                                _depth=_depth + 1,
                                _included_files=_included_files,
                            )
                        )
                    else:
                        print(
                            f"fortran_pp: cannot find include '{im.group(1)}' "
                            f"from {source}",
                            file=sys.stderr,
                        )
                continue

            if directive == "error":
                print(f"fortran_pp: #error {rest} in {source}", file=sys.stderr)
                sys.exit(1)

            # Unknown directive — skip silently
            continue

        # --- Normal line ---
        if _active():
            result.append(macros.expand(raw_line, line_number=line_number))

    return result


def _resolve_include(
    name: str, referrer: Path, include_dirs: list[Path]
) -> Path | None:
    """Resolve a quoted include path."""
    # First: relative to the file containing the #include
    candidate = referrer.parent / name
    if candidate.is_file():
        return candidate.resolve()
    # Then: search -I directories
    for d in include_dirs:
        candidate = d / name
        if candidate.is_file():
            return candidate.resolve()
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Traditional C preprocessor for ELPA Fortran sources."
    )
    ap.add_argument("source", type=Path, help="Input .F90 file")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output file")
    ap.add_argument(
        "-I",
        dest="include_dirs",
        action="append",
        default=[],
        help="Include search directory (repeatable)",
    )
    ap.add_argument(
        "-D",
        dest="defines",
        action="append",
        default=[],
        help="Define a macro: -DNAME or -DNAME=VALUE (repeatable)",
    )
    ap.add_argument(
        "--depfile",
        type=Path,
        default=None,
        help="Write a Makefile-format dependency file listing all included headers",
    )
    args = ap.parse_args()

    macros = MacroTable()
    for d in args.defines:
        if "=" in d:
            k, v = d.split("=", 1)
            macros.define(k, v)
        else:
            macros.define(d, "1")

    inc_dirs = [Path(p).resolve() for p in args.include_dirs]
    included_files: set[Path] = set()
    output_lines = preprocess(
        args.source.resolve(), inc_dirs, macros, _included_files=included_files
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    if args.depfile is not None:
        # Write Makefile-format depfile: target: dep1 dep2 ...
        # Spaces in paths are escaped with backslash.
        def _escape(p: Path) -> str:
            return str(p).replace("\\", "/").replace(" ", "\\ ")

        deps = " ".join(_escape(f) for f in sorted(included_files))
        args.depfile.parent.mkdir(parents=True, exist_ok=True)
        args.depfile.write_text(f"{_escape(args.output)}: {deps}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
