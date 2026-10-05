"""Check Bend files bottom-up and report the ones that fail on their own.

Run from the repository root:
  python -B scripts/toolchain_rules/sweep.py OUT.json [--skip PREVIOUS.json] [FILE_OR_DIR ...]
Use it after a Bend upgrade, when `python run.py proofs` fails and its first error
hides the others. It starts the checker once per file, so a sweep of the
whole repository takes much longer than the single proof run.
A file is checked only after every file it imports has passed, so each
reported failure is in that file, not in something it imports. A file whose
only complaint is an open law (TODO) counts as passed for its importers,
because another file fills it.
Files above a failure wait. When nothing else is ready, the failed files are
checked again, so a fix made while the sweep runs unblocks them. The sweep
ends when everything passed or a recheck fixes nothing.
--skip takes the files a previous report passed as already checked; use it
after editing only files that the previous report did not pass.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
IMPORT = re.compile(r"^import\s+(\S+\.bend)\s+as\s+\w+\s*$", re.M)
OPEN_LAW = re.compile(r"Error: \d+ TODOs? found")


def imports(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    found = []
    for target in IMPORT.findall(text):
        other = (path.parent / target).resolve()
        if other.exists():
            found.append(other)
    return found


def check(path):
    started = time.time()
    try:
        result = subprocess.run(["node", "scripts/bend_launcher.mjs", path.relative_to(ROOT).as_posix(), "--check-only"],
                                cwd=ROOT, capture_output=True, text=True, timeout=1200)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        output = "TIMEOUT"
    return "ALL PROOFS CHECK" in output, output, time.time() - started


def main():
    out = Path(sys.argv[1])
    arguments = sys.argv[2:]
    skipped = set()
    if arguments[:1] == ["--skip"]:
        previous = json.loads(Path(arguments[1]).read_text(encoding="utf-8"))
        skipped = {(ROOT / name).resolve() for name in previous["passed"] + previous.get("open_laws", [])}
        arguments = arguments[2:]
    targets = [Path(arg) for arg in arguments] or [Path("lib"), Path("convolution"), Path("scripts/check_official.bend")]
    files = []
    for target in targets:
        target = (ROOT / target).resolve()
        files.extend(sorted(target.rglob("*.bend")) if target.is_dir() else [target])
    graph = {}
    pending = list(files)
    while pending:
        path = pending.pop()
        if path not in graph:
            graph[path] = imports(path)
            pending.extend(graph[path])
    state = {path: "ok" for path in graph if path in skipped}
    report = {"passed": sorted(path.relative_to(ROOT).as_posix() for path in state), "open_laws": [], "failed": {}, "blocked": []}
    remaining = set(graph) - set(state)

    def record(path):
        passed, output, seconds = check(path)
        name = path.relative_to(ROOT).as_posix()
        report["failed"].pop(name, None)
        if passed:
            state[path] = "ok"
            report["passed"].append(name)
        elif OPEN_LAW.search(output) and output.count("Error:") == 1:
            state[path] = "ok"
            report["open_laws"].append(name)
        else:
            state[path] = "bad"
            lines = [line for line in output.splitlines() if line.strip() and "is available" not in line]
            report["failed"][name] = lines[:40]
            print(f"FAIL {name} ({seconds:.0f}s)", flush=True)
        report["blocked"] = sorted(other.relative_to(ROOT).as_posix() for other in remaining)
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    while remaining:
        ready = sorted(path for path in remaining if all(state.get(dep) == "ok" for dep in graph[path]))
        if not ready:
            bad = sorted(path for path in graph if state.get(path) == "bad")
            for path in bad:
                record(path)
            if all(state.get(path) == "bad" for path in bad):
                break
            continue
        for path in ready:
            remaining.discard(path)
            record(path)
    report["blocked"] = sorted(path.relative_to(ROOT).as_posix() for path in remaining)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"passed {len(report['passed'])}, failed {len(report['failed'])}, blocked {len(report['blocked'])}", flush=True)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
