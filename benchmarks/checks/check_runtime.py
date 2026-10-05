from pathlib import Path
import subprocess
import json
import argparse

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / 'bend.cmd').is_file()
)
cases = json.loads((ROOT / 'out/results/integration/cases.json').read_text())
allrows = []
ap = argparse.ArgumentParser()
ap.add_argument(
    '--entry', choices=['convolution', 'scalar_convolution', 'array_api', 'array_api_zero_workspace'], required=True
)
args = ap.parse_args()
binary = args.entry
for threads in [1, 2, 4, 8]:
    p = subprocess.run(
        [str(ROOT / 'out/build/integration' / binary), '--threads', str(threads), '--gpu', 'off'],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    (ROOT / f'out/results/integration/runtime-{args.entry}-{threads}.log').write_text(
        p.stdout + p.stderr
    )
    results = {}
    current = None
    for line in p.stdout.splitlines():
        if line.startswith('CASE '):
            _, i, h, w = line.split()
            current = int(i)
            assert current not in results
            results[current] = dict(height=int(h), width=int(w), accepted=int(h) > 0, bits=[])
        elif line == 'END':
            current = None
        elif current is not None:
            results[current]['bits'].append(int(line, 16))
    assert len(results) == len(cases) * 2
    for i, c in enumerate(cases):
        for side in [0, 1]:
            assert results[2 * i + side] == c['expected'], (
                threads,
                c['name'],
                side,
                results[2 * i + side],
                c['expected'],
            )
    allrows.append(
        dict(threads=threads, cases=len(cases), executions=len(results), all_bitwise=True)
    )
(ROOT / f'out/results/integration/runtime-{args.entry}.json').write_text(
    json.dumps(allrows, indent=2)
)
print(allrows)
