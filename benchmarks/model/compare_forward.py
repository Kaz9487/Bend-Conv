"""Alternate preserved baseline and candidate C with identical compiler flags.

This is final acceptance, not a split-depth or thread-grid calibration.
"""

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def statistics_ms(values):
    quartiles = statistics.quantiles(values, n=4, method='inclusive')
    return dict(median_ms=statistics.median(values) * 1000,
                q1_ms=quartiles[0] * 1000, q3_ms=quartiles[2] * 1000,
                iqr_ms=(quartiles[2] - quartiles[0]) * 1000)


def baseline_stem(parser, folder, requested, explicit, threads):
    stem = explicit or requested
    if not re.fullmatch(r'[a-z][a-z0-9_]*', stem):
        parser.error('--baseline-stem must be a file stem in snake_case')
    if not explicit and not all((folder / f'{stem}.{suffix}').is_file() for suffix in ('c', 'json')):
        stem = 'bus_640'
    for suffix in ('c', 'json', 'c.build.json'):
        if not (folder / f'{stem}.{suffix}').is_file():
            parser.error(f'Missing baseline {stem}.{suffix}')
    graph = json.loads((folder / f'{stem}.json').read_text())
    dynamic = False
    entry = folder / f'{stem}.bend'
    if entry.is_file():
        dynamic = bool(re.search(r'\bIO\.thread_count\s*\(\s*\)', entry.read_text()))
    if graph.get('threads') != threads and not dynamic:
        parser.error('A baseline with different graph threads requires the original IO.thread_count() entry')
    return stem, graph, dynamic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-models', required=True, type=Path)
    parser.add_argument('--baseline-stem', help='Preserved dynamic-worker entry; default: thread-specific stem, then bus_640')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--threads', type=int, choices=[1, 4], required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = 'bus_640' + (f'_cpu_{args.threads}t' if args.threads != 1 else '')
    folders = dict(before=args.baseline_models.resolve(), after=ROOT / 'out/models')
    before_stem, baseline_graph, dynamic = baseline_stem(
        parser, folders['before'], stem, args.baseline_stem, args.threads)
    candidate_graph = json.loads((folders['after'] / f'{stem}.json').read_text())
    if set(baseline_graph['snapshots']) != set(candidate_graph['snapshots']):
        parser.error('Baseline and candidate must expose the same complete snapshot set')
    stems = dict(before=before_stem, after=stem)
    binaries, provenance = {}, {}
    for name, folder in folders.items():
        source = folder / f'{stems[name]}.c'
        metadata = folder / f'{stems[name]}.c.build.json'
        build = json.loads(metadata.read_text())
        toolchain = json.loads((ROOT / 'scripts/toolchain.json').read_text())
        # A baseline may come from an earlier pin: that is the toolchain upgrade comparison.
        if name == 'after' and build['compiler_commit'] != toolchain['commit']:
            parser.error('The candidate must use the pinned official compiler')
        subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/check_generated_c.py'), str(source)], check=True)
        binary = output / name
        flags = ['-O3', '-march=native', '-ffp-contract=off', '-std=c11']
        subprocess.run([os.environ.get('CC', 'clang'), *flags, str(source), '-lpthread', '-lm', '-o', str(binary)], check=True)
        binaries[name] = binary
        provenance[name] = dict(source=os.path.relpath(source, ROOT), stem=stems[name], runtime_threads=args.threads,
                                graph_threads=(baseline_graph if name == 'before' else candidate_graph).get('threads'),
                                dynamic_workers=dynamic if name == 'before' else None,
                                flags=flags, build=build)

    def run(name, label, repetitions, dump):
        folder = output / label
        folder.mkdir(exist_ok=True)
        subprocess.run([str(binaries[name]), '--threads', str(args.threads), '--gpu', 'off'],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
                       env=dict(os.environ, BEND_OUTDIR=str(folder), BEND_DUMP=str(int(dump)),
                                BEND_REPETITIONS=str(repetitions)))
        return folder

    snapshots = baseline_graph['snapshots']
    before = run('before', 'validation_before', 1, True)
    after = run('after', 'validation_after', 1, True)
    for label in snapshots:
        assert (before / f'{label}.bin').read_bytes() == (after / f'{label}.bin').read_bytes(), label
    rounds = {name: [] for name in folders}
    for number in range(5):
        order = ['before', 'after'] if number % 2 == 0 else ['after', 'before']
        for name in order:
            folder = run(name, f'round_{number}_{name}', 9, False)
            rows = [json.loads(line) for line in (folder / 'native_samples.jsonl').read_text().splitlines()]
            assert [row['sample'] for row in rows] == list(range(9)), 'Incomplete samples'
            values = [row['seconds'] for row in rows[2:]]
            rounds[name].append(dict(round=number, order=order, samples_seconds=values, **statistics_ms(values)))
    result = dict(threads=args.threads, bitwise_snapshots=len(snapshots),
                  protocol='5 alternating rounds; each run 2 warmups then 7 samples; same process',
                  scope='Whole-version comparison includes band layout, GEMM, planner and graph ownership changes; no single-factor attribution.',
                  provenance=provenance, rounds=rounds,
                  aggregate={name: statistics_ms([value for row in rows for value in row['samples_seconds']])
                             for name, rows in rounds.items()})
    (output / 'comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['aggregate'], indent=2))


if __name__ == '__main__':
    main()
