"""Re-verify docs/toolchain_rules.md against the pinned official Bend.

Run from the repository root: python -B scripts/toolchain_rules/verify.py
It checks that the source passages the rules cite still exist, that the
checker's set of error messages is unchanged, and that every probe in
probes.py still behaves as recorded. After reviewing a new Bend version and
updating the rules, run it with --snapshot to record that version.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

from probes import CHECKER, CODEGEN

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCES = ROOT / '.tools/bend/bend2'
BUILD = ROOT / 'out/toolchain_rules'
CALL = re.compile(r'(?:parse_fail|Err)\(')
TEXT = re.compile(r'"((?:[^"\\]|\\.){6,})"')


def diagnostics():
    """Every string literal on a line that raises a checker or parser error."""
    found = set()
    for line in (SOURCES / 'bend.ts').read_text(encoding='utf-8').splitlines():
        if CALL.search(line):
            found.update(TEXT.findall(line))
    return sorted(found)


def launch(name, source, *arguments):
    path = BUILD / f'{name}.bend'
    path.write_text('import Base\n' + source, encoding='utf-8', newline='\n')
    result = subprocess.run(['node', 'scripts/bend_launcher.mjs', path.relative_to(ROOT).as_posix(), *arguments],
                            cwd=ROOT, capture_output=True, text=True, timeout=300)
    return result.stdout + result.stderr


def check_probe(name, expected, source):
    output = launch(name, source, '--check-only')
    passed = 'ALL PROOFS CHECK' in output
    if (passed and expected == 'pass') or (not passed and expected != 'pass' and expected in output):
        return []
    first = next((line for line in output.splitlines() if line.strip()), 'no output')
    return [f'probe {name}: expected {expected!r}, got {first!r}']


def check_codegen(name, present, absent, shared, source):
    """Compile one probe to C and compare the markers the rules rely on."""
    target = BUILD / f'{name}.c'
    target.unlink(missing_ok=True)
    output = launch(name, source, '-o', target.relative_to(ROOT).as_posix())
    if not target.exists():
        return [f'codegen {name}: no C was produced ({output.strip()[:160]})']
    text = target.read_text(encoding='utf-8', errors='replace')
    found = [f'codegen {name}: expected {needle!r} in the C' for needle in present if needle not in text]
    found += [f'codegen {name}: did not expect {needle!r} in the C' for needle in absent if needle in text]
    table = re.search(r'CID_T\[\]\[2\] = \{(.*)\};', text)
    flags = [int(flag) for flag in re.findall(r'\{\s*\d+\s*,\s*(\d+)\s*\}', table.group(1))] if table else []
    actual = 'none' if sum(flags) == 0 else 'all' if sum(flags) == len(flags) else 'some'
    if actual != shared:
        found.append(f'codegen {name}: expected {shared} constructors shared, got {sum(flags)} of {len(flags)}')
    return found


def main():
    snapshot = '--snapshot' in sys.argv
    BUILD.mkdir(parents=True, exist_ok=True)
    failures = []
    pin = json.loads((ROOT / 'scripts/toolchain.json').read_text(encoding='utf-8'))
    current = dict(tag=pin['tag'], commit=pin['commit'])
    reviewed_path = HERE / 'reviewed.json'
    reviewed = json.loads(reviewed_path.read_text(encoding='utf-8'))
    if reviewed != current and not snapshot:
        failures.append(f"the rules were reviewed for {reviewed['tag']}; the pin is now {current['tag']}")
    anchors = json.loads((HERE / 'anchors.json').read_text(encoding='utf-8'))
    for name, entries in anchors.items():
        text = (SOURCES / name).read_text(encoding='utf-8')
        failures += [f'anchor missing in {name}: {rule}' for rule, needle in entries.items() if needle not in text]
    messages = diagnostics()
    listing = HERE / 'diagnostics.txt'
    if not snapshot:
        before = listing.read_text(encoding='utf-8').splitlines()
        failures += [f'new checker message: {line}' for line in sorted(set(messages) - set(before))]
        failures += [f'removed checker message: {line}' for line in sorted(set(before) - set(messages))]
    for name, (expected, source) in CHECKER.items():
        failures += check_probe(name, expected, source)
    for name, (present, absent, shared, source) in CODEGEN.items():
        failures += check_codegen(name, present, absent, shared, source)
    if snapshot:
        reviewed_path.write_text(json.dumps(current, indent=2) + '\n', encoding='utf-8', newline='\n')
        listing.write_text('\n'.join(messages) + '\n', encoding='utf-8', newline='\n')
        print(f"snapshot written for {current['tag']}: {len(messages)} checker messages")
    count = sum(len(entries) for entries in anchors.values())
    print(f"{current['tag']}: {count} anchors, {len(messages)} checker messages, "
          f'{len(CHECKER)} checker probes, {len(CODEGEN)} codegen probes')
    for failure in failures:
        print('DIFFERENT:', failure)
    print('RULES HOLD' if not failures else f'{len(failures)} differences: reread the sources and update docs/toolchain_rules.md')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
