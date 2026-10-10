"""Compare preserved and current handwritten YOLO models with identical flags.

Validation saves every layer. Timed runs use the model's forward clock with
layer tracing disabled: two warmups and seven samples in each process.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASES = ('bus_320', 'bus_640', 'zidane_320', 'zidane_640')
FLAGS = ['-O3', '-march=native', '-ffp-contract=off', '-std=c11']
WARMUPS, SAMPLES, ROUNDS = 2, 7, 5


def statistics_ms(values):
    quartiles = statistics.quantiles(values, n=4, method='inclusive')
    return dict(
        median_ms=statistics.median(values) * 1000,
        q1_ms=quartiles[0] * 1000,
        q3_ms=quartiles[2] * 1000,
        iqr_ms=(quartiles[2] - quartiles[0]) * 1000,
    )


# The generated-operation diagnostics import this selector; their graph schema
# and marker protocol remain separate from the handwritten forward comparison.
def baseline_stem(parser, folder, requested, explicit, threads):
    stem = explicit or requested
    if not re.fullmatch(r'[a-z][a-z0-9_]*', stem):
        parser.error('--baseline-stem must be a file stem in snake_case')
    if not explicit and not all(
        (folder / f'{stem}.{suffix}').is_file() for suffix in ('c', 'json')
    ):
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
        parser.error(
            'A baseline with different graph threads requires the original IO.thread_count() entry'
        )
    return stem, graph, dynamic


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def handwritten_source(folder, stem, toolchain):
    if not re.fullmatch(r'[a-z][a-z0-9_]*', stem):
        raise ValueError('Use a snake_case handwritten model stem')
    source = folder / f'{stem}.c'
    receipt = folder / f'{stem}.c.build.json'
    build = json.loads(receipt.read_text())
    entry = build.get('source', '').replace('\\', '/')
    if not entry.endswith('examples/yolov5/yolov5n.bend'):
        raise ValueError(
            'The C build record must identify examples/yolov5/yolov5n.bend; generated graphs are not handwritten baselines'
        )
    if build.get('compiler_commit') != toolchain['commit']:
        raise ValueError('Both models must use the pinned official Bend compiler')
    return source, dict(
        build=build, c_sha256=fingerprint(source), build_sha256=fingerprint(receipt)
    )


def compare_snapshots(before, after, shapes):
    expected = set(shapes)
    names = {p.stem for p in before.glob('*.npy')}
    other = {p.stem for p in after.glob('*.npy')}
    if not expected <= names or names != other:
        raise ValueError(
            'The two runs must contain every reference snapshot and the same complete NPY set'
        )
    for name in sorted(names):
        left = np.load(before / f'{name}.npy', allow_pickle=False)
        right = np.load(after / f'{name}.npy', allow_pickle=False)
        if left.dtype != np.dtype('<f4') or right.dtype != left.dtype or right.shape != left.shape:
            raise ValueError(f'Snapshot dtype/shape differs: {name}')
        if name in shapes and list(left.shape) != shapes[name]:
            raise ValueError(f'Snapshot shape differs from the reference manifest: {name}')
        if left.tobytes() != right.tobytes():
            raise ValueError(f'Snapshot bits differ: {name}')
    return sorted(names)


def run_model(binary, folder, case, threads, repetitions, layers):
    folder.mkdir()
    environment = dict(os.environ)
    for key in (
        'YOLO_SIZE',
        'YOLO_INPUT',
        'YOLO_OUTPUT',
        'YOLO_LAYERS',
        'YOLO_EXTRA',
        'BEND_OUTDIR',
        'BEND_DUMP',
        'BEND_REPETITIONS',
    ):
        environment.pop(key, None)
    environment.update(
        YOLO_SIZE=case.rsplit('_', 1)[1],
        YOLO_INPUT=str(ROOT / 'out/results' / case / 'input.bin'),
        YOLO_OUTPUT=str(folder),
        YOLO_EXTRA=str(repetitions - 1),
    )
    if layers:
        environment['YOLO_LAYERS'] = '1'
    with (folder / 'stdout.log').open('w') as stdout, (folder / 'stderr.log').open('w') as stderr:
        subprocess.run(
            [str(binary), '--threads', str(threads), '--gpu', 'off'],
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=True,
        )
    text = (folder / 'stdout.log').read_text()
    lines = text.splitlines()
    values = [int(match[1]) for line in lines if (match := re.fullmatch(r'forward (\d+) ms', line))]
    if len(values) != repetitions or any(
        line.startswith('forward ') and not re.fullmatch(r'forward \d+ ms', line) for line in lines
    ):
        raise ValueError(f'Expected exactly {repetitions} forward timing receipts in {folder.name}')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--case', choices=CASES, default='bus_640')
    parser.add_argument('--baseline-models', required=True, type=Path)
    parser.add_argument(
        '--baseline-stem',
        default='yolov5n',
        help='Preserved handwritten C stem (no graph JSON needed)',
    )
    parser.add_argument(
        '--output', required=True, type=Path, help='New or empty directory under out/'
    )
    parser.add_argument('--threads', type=int, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    output = args.output.resolve()
    if output == ROOT / 'out' or not output.is_relative_to((ROOT / 'out').resolve()):
        parser.error('--output must be a subdirectory under out/')
    if output.exists() and any(output.iterdir()):
        parser.error(
            '--output must be empty; use a new directory to exclude stale snapshots and reports'
        )
    toolchain = json.loads((ROOT / 'scripts/toolchain.json').read_text())
    sources = {
        'before': handwritten_source(args.baseline_models.resolve(), args.baseline_stem, toolchain),
        'after': handwritten_source(ROOT / 'out/models', 'yolov5n', toolchain),
    }
    compiler = os.environ.get('CC', 'clang-19')
    compiler_version = subprocess.check_output([compiler, '--version'], text=True)
    if not re.search(r'\bclang version 19\.', compiler_version):
        parser.error('Set CC=clang-19')
    input_path = ROOT / 'out/results' / args.case / 'input.bin'
    side = int(args.case.rsplit('_', 1)[1])
    if input_path.stat().st_size != 4 * 3 * side * side:
        parser.error('The input must contain raw FP32 [1,3,size,size] values')
    input_hash = fingerprint(input_path)
    reference = ROOT / 'out/results' / args.case / 'reference.json'
    reference_hash = fingerprint(reference)
    shapes = json.loads(reference.read_text())['outputs']
    required = {f'layer_{i:02}' for i in range(24)} | {f'head_{i}' for i in range(3)} | {'pred'}
    if not required <= set(shapes):
        parser.error('Reference metadata must describe all 24 layers, three heads and pred')
    weights = ROOT / 'out/weights/named'
    names = (weights / 'names.txt').read_text().splitlines()
    if not names or len(names) != len(set(names)):
        parser.error('Weights must have a nonempty, unique names.txt; run prepare_reference.py')
    if any(not re.fullmatch(r'[\w.]+', name) for name in names):
        parser.error('Weight names must be flat tensor names')
    names_hash = fingerprint(weights / 'names.txt')
    weight_hashes = {name: fingerprint(weights / f'{name}.npy') for name in names}
    output.mkdir(parents=True, exist_ok=True)
    provenance, binaries = {}, {}
    for name, (source, record) in sources.items():
        subprocess.run(
            [sys.executable, '-B', str(ROOT / 'scripts/check_generated_c.py'), str(source)],
            check=True,
        )
        binary = output / name
        subprocess.run(
            [compiler, *FLAGS, str(source), '-lpthread', '-lm', '-o', str(binary)], check=True
        )
        binaries[name] = binary
        provenance[name] = dict(
            record,
            source=os.path.relpath(source, ROOT),
            runtime_threads=args.threads,
            compiler_flags=FLAGS,
        )
    validated = {}
    for name in sources:
        folder = output / f'validation_{name}'
        run_model(binaries[name], folder, args.case, args.threads, 1, True)
        validated[name] = folder
    snapshots = compare_snapshots(validated['before'], validated['after'], shapes)
    print(f'Validated {len(snapshots)} bitwise snapshots', flush=True)
    rounds = {name: [] for name in sources}
    for number in range(ROUNDS):
        order = ['before', 'after'] if number % 2 == 0 else ['after', 'before']
        for name in order:
            folder = output / f'round_{number}_{name}'
            values = run_model(
                binaries[name], folder, args.case, args.threads, WARMUPS + SAMPLES, False
            )
            prediction = np.load(folder / 'pred.npy', allow_pickle=False)
            expected = np.load(validated[name] / 'pred.npy', allow_pickle=False)
            if (
                prediction.dtype != expected.dtype
                or prediction.shape != expected.shape
                or prediction.tobytes() != expected.tobytes()
            ):
                raise ValueError(
                    f'Timed run prediction differs from validation: {name}, round {number}'
                )
            seconds = [value / 1000 for value in values[WARMUPS:]]
            rounds[name].append(
                dict(
                    round=number,
                    order=order,
                    all_forward_ms=values,
                    samples_seconds=seconds,
                    **statistics_ms(seconds),
                )
            )
            print(
                f'round {number + 1}/{ROUNDS} {name}: {statistics.median(values[WARMUPS:])} ms',
                flush=True,
            )
    if (
        fingerprint(input_path) != input_hash
        or fingerprint(reference) != reference_hash
        or fingerprint(weights / 'names.txt') != names_hash
        or any(
            fingerprint(weights / f'{name}.npy') != digest for name, digest in weight_hashes.items()
        )
    ):
        raise ValueError('Input, reference or weights changed while comparing')
    result = dict(
        case=args.case,
        threads=args.threads,
        bitwise_snapshots=len(snapshots),
        snapshot_names=snapshots,
        entry='examples/yolov5/yolov5n.bend',
        protocol='5 alternating rounds; each process runs 2 warmups then 7 samples with YOLO_EXTRA=8; YOLO_LAYERS unset for timing',
        clock='forward N ms printed by the model; millisecond resolution; file output after the forward is outside the timer',
        scope="Whole handwritten forward; includes the model's ownership operations within forward. Local reference timings only; no operation-level or single-factor attribution.",
        provenance=provenance,
        compiler=compiler_version,
        inputs=dict(
            input_sha256=input_hash,
            reference_sha256=reference_hash,
            names_sha256=names_hash,
            weights=weight_hashes,
        ),
        rounds=rounds,
        aggregate={
            name: statistics_ms([value for row in rows for value in row['samples_seconds']])
            for name, rows in rounds.items()
        },
    )
    (output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['aggregate'], indent=2))


if __name__ == '__main__':
    main()
