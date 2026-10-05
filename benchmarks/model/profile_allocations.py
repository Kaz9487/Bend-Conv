"""Measure heap/copy requests of an official model C; discard timings.

The original and diagnostic binaries read identical inputs from the repository
root and must produce bitwise equal snapshots.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'benchmarks'))
from allocation_instrumentation import instrument, replace_once, write_counts  # noqa: E402
from compare_forward import baseline_stem  # noqa: E402

FLAGS = ['-O3', '-march=native', '-ffp-contract=off', '-std=c11']
SEMANTIC_SCOPE = 'operand_acquisition_and_computation; snapshot_gather_and_state_save_io_excluded'


def instrument_operations(source, operations):
    phases = len(operations)
    if not phases:
        raise ValueError('Allocation profiling requires at least one graph operation')
    outputs = [operation['output'] for operation in operations]
    if any(type(value) is not int or not 0 <= value < 2**32 for value in outputs):
        raise ValueError('Graph operation output ids must be U32 values')
    source = instrument(source, phases)
    anchors = []

    def replace(needle, replacement, label):
        nonlocal source
        source = replace_once(source, needle, replacement)
        anchors.append(dict(label=label, matched=1, source_anchor=needle))

    # Official Array primitives keep these native producers even when their
    # Bend callers inline. Observe real allocations after BLK_ALLOC, separately
    # from generic heap requests and owner-level clone interpretations.
    producers = [('blk_new', 'l', 'arr ? c : buf_wcls(c)', 'c'),
                 ('blk_copy', 'dst', 'cls', 'blk_cls(a)'),
                 ('blk_node', 'n', 'arr ? c + 1 : c', 'c + 1'),
                 ('blk_half', 'n', 'cw', 'c')]
    producer_receipts = []
    for producer, variable, heap_class, payload_class in producers:
        matches = list(re.finditer(r'\b(?:INLINE|OUTLINE)\s+Term\s+' + producer
                                   + r'\s*\([^{};]*\)\s*\{', source))
        if len(matches) != 1:
            raise ValueError(f'Expected one official native Array producer: {producer}')
        start, opening = matches[0].start(), matches[0].end() - 1
        depth, closing = 0, None
        # Lex comments/strings as whole tokens so their braces are ignored.
        tokens = re.finditer(r'/\*[\s\S]*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]',
                             source[opening:])
        for token in tokens:
            if token.group() == '{':
                depth += 1
            elif token.group() == '}':
                depth -= 1
                if depth == 0:
                    closing = opening + token.end()
                    break
        if closing is None:
            raise ValueError(f'Unclosed native producer: {producer}')
        body = source[start:closing]
        allocation = f'BLK_ALLOC({variable}, {heap_class})'
        slot = len(producer_receipts)
        hook = (allocation + '\n#if !DEVICE\n'
                f'if(profile_phase<{phases}){{'
                f'__atomic_fetch_add(&profile_array_calls[profile_phase][{slot}],1,__ATOMIC_RELAXED);'
                f'__atomic_fetch_add(&profile_array_payload[profile_phase][{slot}],'
                f'(1ull<<({payload_class}))*(arr?8:4),__ATOMIC_RELAXED);'
                f'__atomic_fetch_add(&profile_array_heap[profile_phase][{slot}],'
                f'(1ull<<({heap_class}))*8,__ATOMIC_RELAXED);'
                '}\n#endif\n')
        revised = replace_once(body, allocation, hook)
        source = source[:start] + revised + source[closing:]
        producer_receipts.append(dict(producer=producer, allocation_anchor=allocation,
                                      matched=1, 
                                      payload_class=payload_class, heap_class=heap_class))

    phase_declaration = f'static unsigned profile_phase = {phases};'
    replace(phase_declaration, phase_declaration
            + '\nstatic unsigned profile_operation;'
            + f'\nstatic unsigned long long profile_array_calls[{phases}][4],'
            f'profile_array_payload[{phases}][4],profile_array_heap[{phases}][4];'
            + '\nstatic const uint32_t profile_output_ids[]={'
            + ','.join(str(value) for value in outputs) + '};', 'actual output id receipt')
    explicit_begin = bool(re.search(r'\bTerm\s+tensor_operation_begin_run\s*\([^{};]*\)\s*\{', source))
    explicit_end = bool(re.search(r'\bTerm\s+tensor_operation_end_run\s*\([^{};]*\)\s*\{', source))
    if explicit_begin != explicit_end:
        raise ValueError('A model must emit both operation begin/end effects or neither')
    validate_id = (f'if(profile_operation>={phases} || arguments[0]!=profile_output_ids[profile_operation])'
                   '{fprintf(stderr,"allocation operation id/count mismatch\\n");exit(2);}')
    if explicit_begin:
        replace('tensor_dump_seconds=0;tensor_started=io_tick();',
                f'profile_operation=0;profile_phase={phases};tensor_dump_seconds=0;tensor_started=io_tick();',
                'forward begins with counters paused')
        anchor = 'Term tensor_operation_begin_run(Env environment,Term* arguments,IoWork* work){'
        replace(anchor, anchor + validate_id + f'if(profile_phase!={phases})'
                '{fprintf(stderr,"allocation operation begin while active\\n");exit(2);}'
                'profile_phase=profile_operation;', 'explicit operation begin validates and resumes')
        anchor = 'Term tensor_operation_end_run(Env environment,Term* arguments,IoWork* work){'
        replace(anchor, anchor + validate_id + 'if(profile_phase!=profile_operation)'
                '{fprintf(stderr,"allocation operation end while paused\\n");exit(2);}'
                f'profile_operation++;profile_phase={phases};', 'explicit operation end validates and pauses')
        mode = 'explicit_effects'
        protocol = 'Actual operation begin-to-end, including operand acquisition and computation; counters pause between operations before snapshot gather/state-save IO.'
    else:
        replace('tensor_dump_seconds=0;tensor_started=io_tick();',
                'profile_operation=0;profile_phase=0;tensor_dump_seconds=0;tensor_started=io_tick();',
                'forward begins first operation')
        anchor = 'static Term tensor_save_run(Env environment,Term* arguments,IoWork* work){'
        replace(anchor, anchor + validate_id + 'if(profile_phase!=profile_operation)'
                '{fprintf(stderr,"allocation save while paused\\n");exit(2);}'
                f'profile_operation++;profile_phase={phases};', 'save entry validates and pauses')
        anchor = 'free(label);return term_pak(CID_UNIT,0);'
        replace(anchor, 'free(label);profile_phase=profile_operation;return term_pak(CID_UNIT,0);',
                'save return resumes next operation')
        mode = 'save_pause_resume'
        protocol = 'Forward begin/previous save return to current save entry; unmarked dense operands/computation are counted and state-save C/IO is paused. Bend/IO dispatch bookkeeping can differ from explicit effects.'
    anchor = 'Term tensor_end_run(Env environment,Term* arguments,IoWork* work){'
    replace(anchor, anchor + f'if(profile_operation!={phases} || profile_phase!={phases})'
            '{fprintf(stderr,"allocation operation count mismatch\\n");exit(2);}'
            'char profile_path[4096];snprintf(profile_path,sizeof(profile_path),"%s/allocation_counts.json",tensor_outdir());'
            + write_counts('profile_path', phases)
            + 'snprintf(profile_path,sizeof(profile_path),"%s/array_counts.json",tensor_outdir());'
            + array_counts('profile_path', phases), 'forward count receipt and stopped counters')
    return source, dict(mode=mode, public_begin_effect_emitted=explicit_begin,
                        public_end_effect_emitted=explicit_end, semantic_scope=SEMANTIC_SCOPE,
                        protocol=protocol, phases=phases, output_ids=outputs,
                        comparison_scope='Compare native Array producer calls/payload/heap bytes at the common semantic scope. Generic heap requests and control/IO dispatch bookkeeping have different actual boundary coverage and are reported separately.',
                        anchors=anchors, array_producers=producer_receipts, counter_anchors=[
                            dict(source_anchor='INLINE u64 heap_alloc(Env e, u32 cls) {', matched=1),
                            dict(source_anchor='OUTLINE Term blk_copy(Env e, Term a) {', matched=1)])


def array_counts(filename, phases):
    return f'''FILE*arrays=fopen({filename},"w");if(!arrays)exit(2);
fprintf(arrays,"[");for(unsigned j=0;j<{phases};j++){{
fprintf(arrays,"%s[",j?",":"");for(unsigned k=0;k<4;k++)fprintf(arrays,
"%s{{\\"calls\\":%llu,\\"payload_bytes\\":%llu,\\"heap_bytes\\":%llu}}",
k?",":"",profile_array_calls[j][k],profile_array_payload[j][k],profile_array_heap[j][k]);
fprintf(arrays,"]");}}fprintf(arrays,"]\\n");fclose(arrays);'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='bus_640')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--models', type=Path, default=ROOT / 'out/models')
    parser.add_argument('--model-stem', help='Preserved entry stem; default: thread-specific stem, then case')
    parser.add_argument('--output', type=Path, help='Report directory under out/; default: out/results/model_allocations/STEM')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if not re.fullmatch(r'[a-z][a-z0-9_]*', args.case):
        parser.error('--case must use snake_case')
    models = args.models.resolve()
    requested = args.case + (f'_cpu_{args.threads}t' if args.threads != 1 else '')
    selected = args.model_stem or (requested if all((models / f'{requested}.{suffix}').is_file()
                                  for suffix in ('c', 'json')) else args.case)
    stem, graph, dynamic = baseline_stem(parser, models, requested, selected, args.threads)
    if graph['case'] != args.case:
        parser.error('Selected graph belongs to a different case')
    generated, entry = models / f'{stem}.c', models / f'{stem}.bend'
    build_path = generated.with_suffix('.c.build.json')
    build = json.loads(build_path.read_text())
    if build['compiler_commit'] != json.loads((ROOT / 'scripts/toolchain.json').read_text())['commit']:
        parser.error('The model must use the pinned official compiler')
    source = generated.read_text()
    limits = re.findall(r'^#define\s+CUBE_T\s+(\d+)\s*$', source, re.M)
    if len(limits) != 1 or args.threads > int(limits[0]):
        parser.error('Requested threads must fit the actual emitted CUBE_T worker capacity')
    source, instrumentation = instrument_operations(source, graph['ops'])
    report = (args.output or ROOT / 'out/results/model_allocations' / stem).resolve()
    if not report.is_relative_to((ROOT / 'out').resolve()):
        parser.error('--output must stay under this repository out/ directory')
    if report.exists():
        shutil.rmtree(report)
    report.mkdir(parents=True)
    instrumented = report / 'diagnostic.c'
    instrumented.write_text(source, newline='\n')
    original = report / 'original.c'
    original.write_bytes(generated.read_bytes())
    (report / 'original.bend').write_bytes(entry.read_bytes())
    (report / 'original.c.build.json').write_bytes(build_path.read_bytes())
    (report / 'graph.json').write_bytes((models / f'{stem}.json').read_bytes())
    subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/check_generated_c.py'),
                    str(original), str(instrumented)], check=True)
    compiler = os.environ.get('CC') or shutil.which('clang')
    if not compiler:
        parser.error('Set CC to a Clang-compatible compiler')
    binaries, commands = {}, {}
    for name, path in (('normal', original), ('diagnostic', instrumented)):
        executable = report / name
        command = [compiler, *FLAGS, str(path), '-lpthread', '-lm', '-o', str(executable)]
        subprocess.run(command, check=True)
        binaries[name], commands[name] = executable, command

    def run(name):
        folder = report / ('validation_' + name)
        folder.mkdir()
        environment = dict(os.environ, BEND_OUTDIR=str(folder), BEND_DUMP='1', BEND_REPETITIONS='1')
        environment.pop('BEND_TRACE_FILE', None)
        subprocess.run([str(binaries[name]), '--threads', str(args.threads), '--gpu', 'off'],
                       cwd=ROOT, env=environment, check=True, stdout=subprocess.DEVNULL)
        samples = [json.loads(line) for line in (folder / 'native_samples.jsonl').read_text().splitlines()]
        if [sample['sample'] for sample in samples] != [0]:
            raise ValueError('Allocation run must complete exactly one forward')
        # The native IO effects write wall-time artifacts; neither binary's
        # instrumented/validation time is performance evidence here.
        for filename in ('native_run.json', 'native_samples.jsonl'):
            (folder / filename).unlink()
        return folder

    normal, diagnostic = run('normal'), run('diagnostic')
    snapshots = {}
    for label in graph['snapshots']:
        expected, actual = normal / f'{label}.bin', diagnostic / f'{label}.bin'
        if expected.read_bytes() != actual.read_bytes():
            raise ValueError(f'Instrumentation changed {label}')
        snapshots[label] = dict(bytes=actual.stat().st_size, bitwise_equal=True)
    counts_path = diagnostic / 'allocation_counts.json'
    counts = json.loads(counts_path.read_text())
    arrays_path = diagnostic / 'array_counts.json'
    arrays = json.loads(arrays_path.read_text())
    rows, totals, array_totals = [], {}, {}
    for index, (operation, count, array_row) in enumerate(zip(graph['ops'], counts, arrays, strict=True)):
        if set(count) != {'allocation_requests', 'requested_bytes', 'clone_calls', 'clone_bytes'}:
            raise ValueError('Unexpected counter fields')
        if any(type(value) is not int or value < 0 for value in count.values()):
            raise ValueError('Allocation counters must be nonnegative integers')
        kind = operation['kind']
        if len(array_row) != len(instrumentation['array_producers']):
            raise ValueError('Native Array producer count changed')
        array_observation = {}
        for producer, values in zip(instrumentation['array_producers'], array_row, strict=True):
            if set(values) != {'calls', 'payload_bytes', 'heap_bytes'} or any(
                    type(value) is not int or value < 0 for value in values.values()):
                raise ValueError('Invalid native Array producer counters')
            name = producer['producer']
            array_observation[name] = values
            total = array_totals.setdefault(name, {key: 0 for key in values})
            for key, value in values.items():
                total[key] += value
        rows.append(dict(operation=index, kind=kind, tensor=operation['output'],
                         shape=graph['tensors'][operation['output']]['shape'],
                         strategy=operation.get('partition', {}).get('strategy'), counts=count,
                         array_producers=array_observation))
        total = totals.setdefault(kind, {key: 0 for key in count})
        for key, value in count.items():
            total[key] += value
    result = dict(case=args.case, threads=args.threads,
                  snapshot_outputs_bitwise=True, snapshots=snapshots, totals=totals, operations=rows,
                  array_totals=array_totals,
                  semantic_scope=SEMANTIC_SCOPE,
                  instrumentation=instrumentation,
                  provenance=dict(stem=stem, models=str(models),
                                  runtime_threads=args.threads, graph_threads=graph.get('threads'), dynamic_workers=dynamic,
                                  build=build, compiler_flags=FLAGS, build_commands=commands,
                                  compiler=subprocess.check_output([compiler, '--version'], text=True)),
                  scope='Measured heap requests and four official native Array block producers. Producer calls '
                        'are counted after successful BLK_ALLOC; payload bytes include capacity padding and use '
                        '4-byte BUF or 8-byte ARR slots; heap bytes include size-class rounding. blk_copy is a '
                        'native block producer, not a claim about which semantic owner requested a clone. '
                        'These are cumulative allocations, not host malloc, live bytes or peak RSS. '
                        'Initial loading and snapshot/state-save IO are excluded. Actual boundary modes are recorded '
                        'separately; compare Array producers at the common operand acquisition/computation scope. '
                        'Generic heap request attribution can differ across save-boundary and effect control dispatch and is '
                        'not presented as a directly comparable semantic Array count. '
                        'Per-operation strategy is graph metadata, not a measured planner decision. '
                        'All timings discarded. Whole-version changes are not single-factor kernel/layout attribution.')
    (report / 'results.json').write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(stem, json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()
