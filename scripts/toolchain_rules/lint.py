"""Scan Bend sources for patterns the official checker rejects.

Run from the repository root before a checker run:
  python -B scripts/toolchain_rules/lint.py FILE_OR_DIR [...]
It only reads the files and takes seconds. Each finding names its rule in
docs/toolchain_rules.md. It looks at spelling only: it finds neither usage
errors nor type mismatches.
"""
import re
import sys
from pathlib import Path

DEF = re.compile(r"^(?:@unsafe\s+)?def\s+([\w.]+)\??\(", re.M)
LAW = re.compile(r"^law\s+([\w.]+):\n((?:  for .*\n)+)", re.M)
IMPORT = re.compile(r"^import\s+(\S+\.bend)\s+as\s+(\w+)\s*$", re.M)
CALL = re.compile(r"(?<![\w.])((?:[A-Z]\w*\.)?[a-z_]\w*)\(")
BIG_WORD = re.compile(r"U32\.to_nat\(\s*(\d{6,})\s*\)")
CLOSED_LIMIT = re.compile(r"U32\.to_nat\(\s*[\w.]+\(\s*\w+\.limits\(\)\s*\)\s*\)")
BIG_NAT = re.compile(r"(?<![\w.])(\d{6,})n\b")
MATCH = re.compile(r"^\s*match\s+(.*):\s*$")
LET = re.compile(r"^(\s*)\+?[a-z_]\w*(?:\s+\+?[a-z_]\w*)*\s*(?<![=!<>])=(?!=)\s*(.*)$")
DESTRUCTURE = re.compile(r"^(\s*)(\([^=]*\)|[A-Z][\w.]*\{[^=]*\})\s*(?<![=!<>])=(?!=)\s*(.+)$")
NAME = re.compile(r"^[a-z_]\w*$")
BARE_CTOR_LET = re.compile(r"^\s*\+?[a-z_]\w*\s*(?<![=!<>])=(?!=)\s*[A-Z][\w.]*\{")
LIST_PATTERN = re.compile(r"^\s*case\s+.*[\w}]\s+\[\]")


# Each rule: (what was found, why the checker rejects it, what to do).
RULES = {
    "A2": ("a match or destructuring after a let",
           "The checker compiles a body into a case tree. At the first let, every parameter not yet "
           "destructed becomes a lambda and can no longer be matched (can't be matched in this position).",
           "Move every match and destructuring of parameters before the first let. To branch on a computed "
           "value, put the rest of the body in another def that takes the value as a parameter."),
    "A3": ("a match or destructuring of a computed value",
           "A match applies to a parameter or a bound field, not to a call result or another expression "
           "(a match cannot scrutinize a computed value).",
           "Write a helper def that takes the value as a parameter and matches it there."),
    "A7": ("a pattern followed by []",
           "`name []` parses as an array read, not as two patterns (expected : a term / observed : ']').",
           "Write list patterns as Con{head,tail} and Nil{}."),
    "C1": ("a ~ parameter that is not leading",
           "Template parameters come first in a def's parameter list (only leading binders take ~).",
           "Move every ~ parameter to the front and reorder the arguments at the call sites."),
    "C2": ("a call whose number of ~ arguments differs from the declaration",
           "Only parameters the callee declares with ~ take ~. An extra ~ fails in the parser "
           "(expected : a term / observed : '~'), often reported on a nearby line.",
           "Write ~ on exactly the callee's leading ~ parameters, or on none."),
    "D4": ("a let of a constructor without a type annotation",
           "A constructor needs a target type to be checked; alone on the right of a let it cannot be inferred.",
           "Write +x = {Ctor{...} : Type}."),
    "E1": ("a large Nat in a type or expression",
           "Nat is unary. When two types containing a closed large Nat in different spellings are compared, "
           "the checker unfolds it; the recursion depth grows with the value and the stack overflows "
           "(the machine stack overflowed), with no location. Computing with a large Nat literal takes time "
           "proportional to its value even when it does not overflow.",
           "Keep the quantity abstract: make it a parameter (for example +limit: U32) and use it only through "
           "equations and inequalities. Compare concrete numbers on U32 (U32.is_le)."),
    "E1?": ("a default machine limit converted to Nat (a risk, not always a failure)",
            "U32.to_nat(Machine.capacity(Default.limits())) is a closed large Nat that can be computed. "
            "Comparing two types that spell it differently overflows the stack, with no location.",
            "Follow the convolution proofs: take +limits: Machine.Limits, write Machine.capacity(limits) in types, "
            "and substitute Default.limits() only at the top public entry. See count_within in "
            "lib/proofs/bounded_u32_arithmetic.bend and lib/proofs/checked_word_completion.bend."),
}


def split_top(text):
    """Split an argument list at top-level commas."""
    parts, depth, current = [], 0, ""
    for char in text:
        if char in "([{<" and not (char == "<" and current.endswith(" ")):
            depth += 1
        elif char in ")]}>" and depth > 0 and not (char == ">" and current.endswith(("=", "-", " "))):
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current.strip())
    return parts


def close_paren(text, start):
    """Index just past the parenthesis that closes the one at start - 1."""
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return -1


TEMPLATES = {}


def templates(path):
    """Leading ~ parameter count of every def in a file."""
    path = path.resolve()
    if path not in TEMPLATES:
        found = {}
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        for match in DEF.finditer(text):
            end = close_paren(text, match.end())
            if end < 0:
                continue
            params = split_top(text[match.end():end])
            leading = 0
            for param in params:
                if not param.startswith("~"):
                    break
                leading += 1
            late = [param for param in params[leading:] if param.startswith("~")]
            found[match.group(1)] = (leading, bool(late), text.count("\n", 0, match.start()) + 1)
        # A law declares its ~ parameters as leading `for ~name: T` lines.
        for match in LAW.finditer(text):
            binders = [line.strip() for line in match.group(2).splitlines()]
            leading = 0
            for binder in binders:
                if not binder.startswith("for ~"):
                    break
                leading += 1
            late = [binder for binder in binders[leading:] if binder.startswith("for ~")]
            found[match.group(1)] = (leading, bool(late), text.count("\n", 0, match.start()) + 1)
        TEMPLATES[path] = found
    return TEMPLATES[path]


def lint(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    findings = []

    def report(line, rule, message):
        findings.append((line, rule, message))

    aliases = {alias: (path.parent / target) for target, alias in IMPORT.findall(text)}
    own = templates(path)
    for name, (leading, late, line) in own.items():
        if late:
            report(line, "C1", f"def {name}")

    for match in CALL.finditer(text):
        end = close_paren(text, match.end())
        if end < 0:
            continue
        args = split_top(text[match.end():end])
        marked = [arg.startswith("~") for arg in args]
        count = sum(marked)
        if count == 0:
            continue
        line = text.count("\n", 0, match.start()) + 1
        if lines[line - 1].lstrip().startswith(("def ", "@unsafe def ")):
            continue
        target = match.group(1)
        if "." in target:
            alias, name = target.split(".", 1)
            table = templates(aliases[alias]) if alias in aliases else None
        else:
            name, table = target, own
        if table is None or name not in table:
            continue
        leading = table[name][0]
        if count != leading or not all(marked[:count]):
            report(line, "C2", f"{target} declares {leading} ~ parameters; the call writes {count}")

    let_indent = None
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if stripped.startswith(("def ", "type ", "law ", "import ", "@unsafe")):
            let_indent = None
        if let_indent is not None and indent < let_indent:
            let_indent = None
        for match in BIG_WORD.finditer(line):
            report(number, "E1", f"U32.to_nat({match.group(1)})")
        if CLOSED_LIMIT.search(line):
            report(number, "E1?", "")
        for match in BIG_NAT.finditer(line):
            report(number, "E1", f"{match.group(1)}n")
        header = MATCH.match(line)
        if header:
            if re.search(r"[({\[]", header.group(1)):
                report(number, "A3", "the scrutinee is an expression")
            elif let_indent is not None:
                report(number, "A2", "a match after a let")
            continue
        destructure = DESTRUCTURE.match(line)
        if destructure and not stripped.startswith(("case ", "%")):
            value = destructure.group(3).strip()
            if not NAME.match(value):
                report(number, "A3", "the destructured value is an expression")
            elif let_indent is not None:
                report(number, "A2", "a destructuring after a let")
            continue
        if LIST_PATTERN.match(line):
            report(number, "A7", "")
        let = LET.match(line)
        if let and not stripped.startswith(("case ", "%", "def ")):
            if BARE_CTOR_LET.match(line):
                report(number, "D4", "")
            if let_indent is None or indent < let_indent:
                let_indent = indent
    return findings


def main():
    targets = [Path(arg) for arg in sys.argv[1:]]
    if not targets:
        sys.exit("Usage: lint.py FILE_OR_DIR [...]")
    files = []
    for target in targets:
        files.extend(sorted(target.rglob("*.bend")) if target.is_dir() else [target])
    groups = {}
    for path in files:
        for line, rule, detail in sorted(lint(path)):
            groups.setdefault(rule, {}).setdefault(path.as_posix(), []).append((line, detail))
    total = 0
    for rule in sorted(groups):
        found, why, fix = RULES[rule]
        count = sum(len(hits) for hits in groups[rule].values())
        total += count
        print(f"[{rule}] {found}: {count}")
        print(f"  why: {why}")
        print(f"  fix: {fix}")
        for name, hits in groups[rule].items():
            lines = ", ".join(str(line) for line, _ in hits[:12]) + (f" ... ({len(hits)} in all)" if len(hits) > 12 else "")
            print(f"    {name}: line {lines}")
            details = sorted({detail for _, detail in hits if detail})
            if details:
                print(f"      {'; '.join(details[:4])}")
        print()
    print(f"{len(files)} files scanned, {total} findings. Rules: docs/toolchain_rules.md")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
