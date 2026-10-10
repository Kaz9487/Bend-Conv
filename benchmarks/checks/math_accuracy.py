"""Measure the elementwise mathematical functions against MPFR on every FP32 input.

The public functions run in a compiled Bend program over blocks of consecutive
32-bit words; math_accuracy.c compares each result with MPFR. Linux, with
libmpfr-dev installed. Blocks run side by side, one per job; a finished block
is kept, so an interrupted run resumes.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'out/results/math_accuracy'
BUILD = ROOT / 'out/build/math_accuracy'
SOURCE = ROOT / 'out/checks/math_accuracy.bend'
FUNCTIONS = ['sigmoid', 'silu', 'tanh', 'gelu_tanh', 'exp', 'log']


def program(count, names):
    steps = ''.join(
        f'    {name}_input : F.Tensor() <- F.load_raw("input.bin",[{count}])\n'
        f'    {name}_saved : F.Tensor() <- F.save_raw("{name}.bin",F.{name}({name}_input))\n'
        f'    {name}_checked : F.Tensor() <- F.expect({name}_saved)\n'
        for name in names
    )
    return (
        'import Base\nimport ../../lib/tensor_f32.bend as F\n\n'
        'def main() -> IO(Unit):\n  do IO<Unit>:\n' + steps + '    IO.print("done")\n'
    )


def build(cc, count, names):
    for folder in (OUT / 'blocks', BUILD, SOURCE.parent):
        folder.mkdir(parents=True, exist_ok=True)
    SOURCE.write_text(program(count, names))
    subprocess.run([sys.executable, '-B', 'scripts/toolchain_rules/lint.py', str(SOURCE)], cwd=ROOT, check=True)
    generated = BUILD / 'math_accuracy.c'
    subprocess.run(['node', 'scripts/bend_launcher.mjs', str(SOURCE), '-o', str(generated)], cwd=ROOT, check=True)
    subprocess.run(
        [cc, '-O3', '-march=native', '-ffp-contract=off', '-std=c11', str(generated), '-lpthread', '-lm',
         '-o', str(BUILD / 'functions')],
        cwd=ROOT, check=True,
    )
    subprocess.run(
        [cc, '-O2', '-std=c11', 'benchmarks/checks/math_accuracy.c', '-lmpfr', '-lgmp', '-lm',
         '-o', str(BUILD / 'compare')],
        cwd=ROOT, check=True,
    )


def measure(block, bits, names):
    base, count = block << bits, 1 << bits
    compare = str(BUILD / 'compare')
    work = OUT / 'work' / str(block)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    subprocess.run([compare, 'input', str(base), str(count), 'input.bin'], cwd=work, check=True)
    subprocess.run([str(BUILD / 'functions'), '--threads', '1', '--gpu', 'off'], cwd=work, check=True,
                   capture_output=True)
    path = OUT / 'blocks' / f'{bits}_{block:05d}.json'
    kept = json.loads(path.read_text()) if path.is_file() else []
    rows = [row for row in kept if row['function'] not in names]
    for name in names:
        result = subprocess.run(
            [compare, 'check', name, str(base), str(count), 'input.bin', f'{name}.bin'],
            cwd=work, check=True, capture_output=True, text=True,
        )
        rows.append(json.loads(result.stdout))
    shutil.rmtree(work)
    rows.sort(key=lambda row: FUNCTIONS.index(row['function']))
    path.write_text(json.dumps(rows) + '\n')
    print(f'block {block}: ' + ', '.join(f'{row["function"]} {row["largest_ulp"]:.3f}' for row in rows), flush=True)


def summarize(bits):
    blocks = sorted((OUT / 'blocks').glob(f'{bits}_*.json'))
    totals = {}
    for path in blocks:
        for row in json.loads(path.read_text()):
            total = totals.setdefault(row['function'], dict(
                finite=0, largest_ulp=0.0, largest_word=0, steps=[0] * 5, undecided=0, nan_wrong=0,
                infinity_wrong=0, zero_sign_wrong=0, plus_infinity=None, minus_infinity=None))
            for key in ('finite', 'undecided', 'nan_wrong', 'infinity_wrong', 'zero_sign_wrong'):
                total[key] += row[key]
            total['steps'] = [a + b for a, b in zip(total['steps'], row['steps'])]
            if row['largest_ulp'] > total['largest_ulp']:
                total['largest_ulp'], total['largest_word'] = row['largest_ulp'], row['largest_word']
            first, last = row['base'], row['base'] + row['count'] - 1
            if first <= 0x7f800000 <= last:
                total['plus_infinity'] = row['plus_infinity']
            if first <= 0xff800000 <= last:
                total['minus_infinity'] = row['minus_infinity']
    inputs = len(blocks) << bits
    summary = dict(inputs=inputs, complete=inputs == 1 << 32, functions=totals)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = [
        f'{inputs} of {1 << 32} input words.', '',
        '| Function | Largest error (ULP) | At input | Correctly rounded | 1 step | 2 steps | 3 steps | 4 or more |',
        '|---|---:|---|---:|---:|---:|---:|---:|',
    ]
    for name, total in totals.items():
        value = struct.unpack('<f', struct.pack('<I', total['largest_word']))[0]
        share = total['steps'][0] / max(1, total['finite'])
        lines.append(
            f'| `{name}` | {total["largest_ulp"]:.3f} | `{value!r}` | {share:.4%} | '
            + ' | '.join(str(step) for step in total['steps'][1:]) + ' |')
    problems = {
        name: {key: total[key] for key in ('undecided', 'nan_wrong', 'infinity_wrong', 'zero_sign_wrong') if total[key]}
        for name, total in totals.items()
    }
    lines += ['', 'Other findings: ' + json.dumps({name: found for name, found in problems.items() if found})]
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block-bits', type=int, default=24, help='log2 of the inputs in one block')
    parser.add_argument('--first', type=int, default=0, help='first block')
    parser.add_argument('--last', type=int, help='one past the last block; every block by default')
    parser.add_argument('--jobs', type=int, default=os.cpu_count(), help='blocks measured at the same time')
    parser.add_argument('--functions', nargs='+', choices=FUNCTIONS,
                        help='measure these again in every block and keep the other results')
    args = parser.parse_args()
    cc = os.environ.get('CC') or shutil.which('clang-19') or shutil.which('clang')
    if not cc:
        raise SystemExit('The accuracy measurement needs clang on PATH or CC.')
    last = args.last if args.last is not None else 1 << (32 - args.block_bits)
    names = args.functions or FUNCTIONS
    build(cc, 1 << args.block_bits, names)
    blocks = [
        block for block in range(args.first, last)
        if args.functions or not (OUT / 'blocks' / f'{args.block_bits}_{block:05d}.json').is_file()
    ]
    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(lambda block: measure(block, args.block_bits, names), blocks))
    summarize(args.block_bits)


if __name__ == '__main__':
    main()
