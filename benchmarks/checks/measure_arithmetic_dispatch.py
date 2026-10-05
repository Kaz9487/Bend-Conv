"""Compare a templated arithmetic record with a direct function template.

Both use the same owner-preserving elementwise loop and allocate the same two
arrays before the chain. No foreign arithmetic or timing hooks are introduced.
Times cover the whole process, including allocation, checksum and teardown.
Generated source, native C, assembly, binaries and evidence live under out/.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def source(element, count, rounds, variant):
    depth = (count - 1).bit_length()
    initial, increment, zero = ('1.0', '2.0', '0.0') if element == 'F32' else ('1', '2', '0')
    operation = (f'~arithmetic_add(~{element},~{element}.arithmetic())'
                 if variant == 'record' else f'~{element}.add')
    return f'''import Base
import ../../lib/traversal_array.bend as Transform

type Arithmetic<-Element: Data> is Type:
  Arithmetic{{add: Element -> Element -> Element,sub: Element -> Element -> Element,
    mul: Element -> Element -> Element,maximum: Element -> Element -> Element,minimum: Element -> Element -> Element}}

def {element}.arithmetic() -> Arithmetic<{element}>:
  Arithmetic{{left => right => {element}.add(left,right),left => right => {element}.sub(left,right),
    left => right => {element}.mul(left,right),left => right => {element}.max(left,right),left => right => {element}.min(left,right)}}

def apply_add(~Element: Data,rules: Arithmetic<Element>,left: Element,right: Element) -> Element:
  Arithmetic{{add,sub,mul,maximum,minimum}} = rules
  add(left,right)

def arithmetic_add(~Element: Data,~rules: Arithmetic<Element>,left: Element,right: Element) -> Element:
  apply_add(~Element,rules,left,right)

def step(count: Nat,left: Array<{element}>,right: Array<{element}>) -> Array<{element}> & Array<{element}>:
  Transform.finish(~Array<{element}>,~{element},~Unit,
    Transform.update(~Array<{element}>,~{element},~{element},~Unit,~Transform.read_index(~{element}),
      {operation},~Transform.index,count,Transform.State{{right,left,0,Unit{{}}}}))

def chain_step(index: U32,count: Nat,result: Array<{element}> & Array<{element}>) -> Array<{element}> & Array<{element}>:
  (right,left) = result
  step(count,left,right)

def chain_output(result: Array<{element}> & Array<{element}>) -> Array<{element}>:
  (right,output) = result
  output

def checksum(result: Array<{element}> & {element}) -> {element}:
  (output,value) = result
  value

def main() -> {element}:
  checksum(Transform.fold(~Array<{element}>,~{element},~{element},~Unit,~Transform.read_index(~{element}),~{element}.add,
    {count}n,0,chain_output(Transform.indexed_fold(~(Array<{element}> & Array<{element}>),~Nat,~chain_step,
      {rounds}n,0,{count}n,(Array.new({element},{depth}n,{increment}),Array.new({element},{depth}n,{initial})))),{zero},Unit{{}}))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', nargs='+', type=int, default=[4096, 65536, 1048576])
    parser.add_argument('--rounds', type=int, default=32)
    parser.add_argument('--samples', type=int, default=7)
    args = parser.parse_args()
    if not args.sizes or min(args.sizes) < 1 or args.rounds < 1 or args.samples < 3:
        parser.error('Use positive sizes/rounds and at least three samples')
    cc = os.environ.get('CC') or shutil.which('clang')
    if not cc:
        parser.error('Set CC to a Clang-compatible compiler')
    directory = ROOT / 'out/checks'
    report = ROOT / 'out/results/arithmetic_dispatch'
    directory.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    (report / 'summary.json').unlink(missing_ok=True)
    flags = ['-O3', '-march=native', '-ffp-contract=off', '-std=c11']
    rows = []
    for element in ['F32', 'U32']:
        for count in args.sizes:
            binaries, builds = {}, {}
            for variant in ['direct', 'record']:
                path = directory / f'arithmetic_{element.lower()}_{count}_{variant}.bend'
                path.write_text(source(element, count, args.rounds, variant))
                c = path.with_suffix('.c')
                build = subprocess.run(
                    ['node', 'scripts/bend_launcher.mjs', str(path), '-o', str(c)],
                    cwd=ROOT, env=dict(os.environ, BEND_NO_TELEMETRY='1'),
                    capture_output=True, text=True)
                (report / f'{path.stem}.log').write_text(build.stdout + build.stderr)
                build.check_returncode()
                binary = path.with_suffix('')
                subprocess.run([cc, *flags, str(c), '-lpthread', '-lm', '-o', str(binary)], check=True)
                assembly = path.with_suffix('.s')
                subprocess.run([cc, *flags, '-S', str(c), '-o', str(assembly)], check=True)
                binaries[variant] = binary
                builds[variant] = dict(bend=json.loads(Path(str(c) + '.build.json').read_text()))
            samples = {variant: [] for variant in binaries}
            outputs = {}
            for iteration in range(args.samples + 2):
                order = list(binaries) if iteration % 2 == 0 else list(reversed(binaries))
                for variant in order:
                    start = time.perf_counter_ns()
                    run = subprocess.run([str(binaries[variant]), '--threads', '1', '--gpu', 'off'],
                                         capture_output=True, text=True, check=True)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    outputs.setdefault(variant, run.stdout)
                    assert outputs[variant] == run.stdout, 'Unstable checksum'
                    assert re.fullmatch(r'\s*[0-9]+(?:\.[0-9]+)?\s*', run.stdout), run.stdout
                    if element == 'U32':
                        assert int(run.stdout) == (count * (1 + 2 * args.rounds)) % 2**32
                    if iteration >= 2:
                        samples[variant].append(elapsed)
                assert outputs['direct'] == outputs['record'], 'Dispatch changed the checksum'
            medians = {variant: statistics.median(values) for variant, values in samples.items()}
            row = dict(element=element, elements=count, rounds=args.rounds,
                       samples_ms=samples, median_ms=medians,
                       record_over_direct=medians['record'] / medians['direct'],
                       checksum=outputs['direct'].strip(), builds=builds)
            rows.append(row)
            print(json.dumps({key: value for key, value in row.items() if key != 'builds'}), flush=True)
    summary = dict(platform=platform.platform(),
                   compiler=subprocess.check_output([cc, '--version'], text=True),
                   flags=flags, threads=1, rows=rows,
                   timing_scope='Process wall time including initialization and checksum; '
                                'two warmups, alternating variant order, seven default samples',
                   validation_scope='Same printed checksum, not a full elementwise bitwise oracle; '
                                    'generated loop and dispatch must also be inspected')
    (report / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
