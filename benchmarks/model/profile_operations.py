"""Time every actual model operation and group the time by general operation class.

Graph operations have explicit begin/end effects around the public tensor call
and its ownership copies. Snapshot gathering is outside this interval.
Host copies made by Tensor.get_buffer are timed separately. A diagnostic copy of the
official generated C carries the timers; its saved tensors must match the normal
example run bitwise. The dependency graph gives the critical path, which bounds the
speedup available from running independent operations at the same time.
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'benchmarks'))
sys.path.insert(0, str(ROOT / 'examples/yolov5'))
from allocation_instrumentation import replace_once  # noqa: E402
from yolo_graph import input_slots  # noqa: E402

WARMUPS = 2
SAMPLES = 7

DECLARATIONS = """static unsigned profile_operation;
static u64 profile_mark, profile_sample, profile_operation_ns[@PHASES@], profile_copy_ns[@PHASES@];
"""
TIMED_COPY = """{u64 profile_start=io_tick();Term profile_copy=blk_copy(environment,tensor_slots[tensor_id]);
if(profile_operation<@PHASES@)profile_copy_ns[profile_operation]+=io_tick()-profile_start;return profile_copy;}"""
SAVE_START = """if(profile_operation<@PHASES@)profile_operation_ns[profile_operation]=io_tick()-profile_mark;
profile_operation++;"""
BEGIN = """profile_operation=0;profile_mark=tensor_started;
for(unsigned j=0;j<@PHASES@;j++)profile_copy_ns[j]=0;"""
END = r"""if(profile_operation!=@PHASES@){fprintf(stderr,"operation count mismatch\n");exit(2);}
{char profile_path[4096];snprintf(profile_path,sizeof(profile_path),"%s/operation_times.jsonl",tensor_outdir());
FILE*profile=fopen(profile_path,profile_sample?"a":"w");if(!profile)exit(2);
fprintf(profile,"{\"sample\":%llu,\"forward_ns\":%llu,\"operation_ns\":[",
(unsigned long long)profile_sample,(unsigned long long)(elapsed*1e9));
for(unsigned j=0;j<@PHASES@;j++)fprintf(profile,"%s%llu",j?",":"",(unsigned long long)profile_operation_ns[j]);
fprintf(profile,"],\"copy_ns\":[");
for(unsigned j=0;j<@PHASES@;j++)fprintf(profile,"%s%llu",j?",":"",(unsigned long long)profile_copy_ns[j]);
fprintf(profile,"]}\n");fclose(profile);profile_sample++;}"""


def instrument_timing(source, phases):
    def fill(text):
        return text.replace('@PHASES@', str(phases))

    operation_begin = 'Term tensor_operation_begin_run(Env environment, Term *arguments, IoWork *work) {'
    source = replace_once(source, operation_begin, fill(DECLARATIONS) + operation_begin)
    source = replace_once(
        source, 'return blk_copy(environment, tensor_slots[tensor_id]);', fill(TIMED_COPY)
    )
    operation_end = 'Term tensor_operation_end_run(Env environment, Term *arguments, IoWork *work) {'
    source = replace_once(source, operation_begin, operation_begin + 'profile_mark=io_tick();')
    source = replace_once(source, operation_end, operation_end + fill(SAVE_START))
    begin = 'tensor_dump_seconds=0;tensor_started=io_tick();'
    source = replace_once(source, begin, begin + fill(BEGIN))
    elapsed = 'double elapsed=(io_tick()-tensor_started)/1e9;'
    return replace_once(source, elapsed, elapsed + fill(END))


def classify(operation):
    """General operation class and output size class; never a model layer name."""
    kind = operation['kind']
    if kind == 'conv':
        positions = operation['output_height'] * operation['output_width']
        size = 'large' if positions >= 16384 else 'medium' if positions >= 1024 else 'small'
        kernel = operation['kernel_size']
        return f'conv {kernel}x{kernel} stride {operation["stride"]}', size
    names = dict(
        add='elementwise add',
        concat='channel concat',
        pool='max pool',
        up='nearest upsample',
        decode='detection decode',
    )
    return names[kind], None


def work(operation, tensors):
    """Multiply-adds for convolution; payload bytes read and written otherwise."""
    if operation['kind'] == 'conv':
        return 'multiply_adds', (
            operation['output_channels']
            * operation['output_height']
            * operation['output_width']
            * operation['input_channels']
            * operation['kernel_size'] ** 2
        )
    reads = [slot for slot in input_slots(operation) if tensors[slot]['path'] is None]
    elements = sum(tensors[slot]['count'] for slot in reads) + tensors[operation['output']]['count']
    return 'payload_bytes', 4 * elements


def critical_path(operations, compute_ns):
    """Longest dependency chain when every independent operation could run at once."""
    producer = {operation['output']: index for index, operation in enumerate(operations)}
    finish, previous = [], []
    for index, operation in enumerate(operations):
        dependencies = [producer[slot] for slot in input_slots(operation) if slot in producer]
        assert all(dependency < index for dependency in dependencies), 'Graph is not in execution order'
        start, parent = max(((finish[d], d) for d in dependencies), default=(0, None))
        finish.append(start + compute_ns[index])
        previous.append(parent)
    last = max(range(len(operations)), key=finish.__getitem__)
    path = []
    while last is not None:
        path.append(last)
        last = previous[last]
    return finish[path[0]], path[::-1]


def summarize(rows, key):
    groups = {}
    for row in rows:
        name = key(row)
        if name is None:
            continue
        group = groups.setdefault(name, dict(operations=0, ms=0.0, copy_ms=0.0, work=0, work_unit=row['work_unit']))
        group['operations'] += 1
        group['ms'] += row['ms']
        group['copy_ms'] += row['copy_ms']
        group['work'] += row['work']
    for group in groups.values():
        compute_seconds = (group['ms'] - group['copy_ms']) / 1e3
        if group['work_unit'] == 'multiply_adds':
            group['gmac_per_second'] = group['work'] / compute_seconds / 1e9
        else:
            group['gb_per_second'] = group['work'] / compute_seconds / 1e9
    return dict(sorted(groups.items(), key=lambda item: -item[1]['ms']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['bus_320', 'bus_640', 'zidane_320', 'zidane_640'], default='bus_640')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    stem = args.case + (f'_cpu_{args.threads}t' if args.threads != 1 else '')
    backend = 'native' + (f'_{args.threads}t' if args.threads != 1 else '')
    generated = ROOT / f'out/models/{stem}.c'
    subprocess.run([sys.executable, '-B', str(ROOT / 'benchmarks/convolution/report_plans.py'), str(ROOT / f'out/models/{stem}.json')], check=True)
    graph = json.loads((ROOT / f'out/models/{stem}.json').read_text())
    operations, tensors = graph['ops'], graph['tensors']

    directory = ROOT / 'out/build/model_profile'
    directory.mkdir(parents=True, exist_ok=True)
    instrumented = directory / f'{stem}_timing.c'
    instrumented.write_text(instrument_timing(generated.read_text(), len(operations)))
    subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/check_generated_c.py'), str(instrumented)], check=True)
    executable = directory / f'{stem}_timing'
    compiler = os.environ.get('CC') or shutil.which('clang')
    if not compiler:
        parser.error('Set CC to a Clang-compatible compiler')
    subprocess.run(
        [compiler, '-O3', '-march=native', '-ffp-contract=off', '-std=c11', str(instrumented),
         '-lpthread', '-lm', '-o', str(executable)],
        check=True,
    )

    report = ROOT / 'out/results/model_operations' / stem
    report.mkdir(parents=True, exist_ok=True)
    command = [str(executable), '--threads', str(args.threads), '--gpu', 'off']
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, BEND_OUTDIR=str(report), BEND_DUMP='1', BEND_REPETITIONS='1'))
    for label in graph['snapshots']:
        actual = report / f'{label}.bin'
        expected = ROOT / f'out/results/{args.case}/{backend}/{label}.bin'
        assert actual.read_bytes() == expected.read_bytes(), f'Instrumentation changed {label}'
        actual.unlink()
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, BEND_OUTDIR=str(report), BEND_DUMP='0',
                            BEND_REPETITIONS=str(WARMUPS + SAMPLES)))
    lines = (report / 'operation_times.jsonl').read_text().splitlines()
    samples = [json.loads(line) for line in lines]
    assert [sample['sample'] for sample in samples] == list(range(WARMUPS + SAMPLES)), 'Incomplete samples'
    samples = samples[WARMUPS:]
    for name in ['operation_times.jsonl', 'native_run.json', 'native_samples.jsonl']:
        (report / name).unlink()

    rows = []
    for index, operation in enumerate(operations):
        category, size = classify(operation)
        unit, amount = work(operation, tensors)
        rows.append(dict(
            operation=index,
            category=category,
            size_class=size,
            output_shape=tensors[operation['output']]['shape'],
            split_depth=operation.get('split_depth'),
            ms=statistics.median(sample['operation_ns'][index] for sample in samples) / 1e6,
            copy_ms=statistics.median(sample['copy_ns'][index] for sample in samples) / 1e6,
            work_unit=unit,
            work=amount,
        ))
    compute_ns = [1e6 * (row['ms'] - row['copy_ms']) for row in rows]
    critical_ns, path = critical_path(operations, compute_ns)
    forward_ms = statistics.median(sample['forward_ns'] for sample in samples) / 1e6
    operation_ms = sum(row['ms'] for row in rows)
    copy_ms = sum(row['copy_ms'] for row in rows)
    result = dict(
        case=args.case,
        threads=args.threads,
        protocol=f'{WARMUPS} warmups and {SAMPLES} samples in one process; per-operation medians; no dumps',
        samples=samples,
        snapshot_outputs_bitwise=True,
        forward_ms=forward_ms,
        operation_sum_ms=operation_ms,
        ownership_copy_ms=copy_ms,
        categories=summarize(rows, lambda row: row['category']),
        convolution_size_classes=summarize(
            rows, lambda row: f'{row["category"]}, {row["size_class"]}' if row['size_class'] else None),
        critical_path=dict(
            compute_ms=critical_ns / 1e6,
            sequential_compute_ms=sum(compute_ns) / 1e6,
            speedup_bound=sum(compute_ns) / critical_ns,
            operations=path,
        ),
        operations=rows,
        scope='Operation time runs from the end of the previous save to the start of this one and '
              'includes IO dispatch; ownership copies are the timed Tensor.get_buffer block copies. The critical '
              'path excludes those copies and assumes unlimited workers without memory contention.',
    )
    (report / 'results.json').write_text(json.dumps(result, indent=2))
    print(f'{stem}: forward {forward_ms:.1f} ms; operations {operation_ms:.1f} ms; ownership copies {copy_ms:.1f} ms')
    for name, group in result['categories'].items():
        rate = group.get('gmac_per_second') or group.get('gb_per_second')
        unit = 'GMAC/s' if 'gmac_per_second' in group else 'GB/s'
        print(f'  {name:22} {group["operations"]:3} ops {group["ms"]:8.2f} ms '
              f'({100 * group["ms"] / operation_ms:4.1f}%) {rate:7.2f} {unit}')
    bound = result['critical_path']
    print(f'  critical path {bound["compute_ms"]:.1f} of {bound["sequential_compute_ms"]:.1f} ms '
          f'(inter-operation bound {bound["speedup_bound"]:.2f}x)')


if __name__ == '__main__':
    main()
