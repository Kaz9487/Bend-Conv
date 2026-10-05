"""Collect the current measured results into one record.

Run after the convolution and full-model CPU benchmark workflows. Timings are
copied from their result files; nothing is inferred from source.
"""
import argparse
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def git(*command):
    return subprocess.check_output(['git', *command], cwd=ROOT, text=True).strip()


RESULTS = ['out/results/cpu_convolution/results.json',
           'out/results/cpu_convolution/allocation_profile.json',
           'out/results/cpu_convolution/measurement.json',
           'out/results/backends/summary.json',
           'out/results/backends/bend_cpu/timing.json',
           'out/results/backends/bend_cpu_4t/timing.json',
           'out/results/backends/numpy_1t/timing.json',
           'out/results/backends/numpy_4t/timing.json',
           'out/results/backends/torch_cpu_1t/timing.json',
           'out/results/backends/torch_cpu_4t/timing.json']
GENERATED = ['out/build/cpu_convolution/bend_convolution.c',
             'out/models/bus_640.c', 'out/models/bus_640_cpu_4t.c']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label', help='Name of the record: out/results/baselines/LABEL/evidence.json')
    args = parser.parse_args()
    evidence = {name: json.loads((ROOT / name).read_text()) for name in RESULTS}
    summary = evidence['out/results/backends/summary.json']
    assert {row['name'] for row in summary['backends']} == {
        'bend_cpu', 'bend_cpu_4t', 'numpy_1t', 'numpy_4t', 'torch_cpu_1t', 'torch_cpu_4t'}
    assert all(row['passed'] for row in summary['backends'])
    pin = json.loads((ROOT / 'scripts/toolchain.json').read_text())
    generated = {}
    for name in GENERATED:
        path = ROOT / name
        build = json.loads(path.with_suffix('.c.build.json').read_text())
        assert build['compiler_commit'] == pin['commit'], f'Rebuild with the pinned compiler: {name}'
        generated[name] = dict(bytes=path.stat().st_size, build=build)
    for name in ['bend_cpu', 'bend_cpu_4t']:
        timing = evidence[f'out/results/backends/{name}/timing.json']
        assert timing['process_protocol'].startswith('same process')
        assert timing['warmups'] == 2 and len(timing['seconds']) == 7
    proofs = {}
    for package in ['lib', 'convolution']:
        files = sorted((ROOT / package / 'proofs').glob('*.bend'))
        proofs[package] = dict(files=len(files), lines=sum(len(p.read_text().splitlines()) for p in files))
    result = dict(
        label=args.label,
        captured_at=datetime.now(timezone.utc).isoformat(),
        commit=git('rev-parse', 'HEAD'),
        uncommitted_changes=bool(git('status', '--porcelain', '--untracked-files=no')),
        source_sizes=generated, proofs=proofs, evidence=evidence,
        scope='Measured latency and allocation requests; source bytes and proof lines are static counts, not performance.',
    )
    output = ROOT / 'out/results/baselines' / args.label / 'evidence.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(output)


if __name__ == '__main__':
    main()
