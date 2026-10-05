"""Task timeline of an official model C diagnostic copy; computation unchanged.

Phase entries and dispatch segments are separate from complete task
claims and block copies. No inclusive semantic-function time is invented.
"""

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'benchmarks'))
from allocation_instrumentation import replace_once  # noqa: E402
from compare_forward import baseline_stem  # noqa: E402

PHASES = ('unclassified', 'band_execute', 'band_run', 'halo_prepare', 'halo_assemble',
          'band_gather', 'dense_partition', 'dense_finish', 'range_copy', 'dense_pipeline')
KINDS = dict(turn=1, wake=2, claim=3, finish=4, forward_begin=5, forward_end=6,
             operation_begin=7, operation_end=8, dispatch=9, dispatch_span=10,
             block_copy=11, host_grow=12, windowed_entry=13)
EVENT = np.dtype([('t0', '<u8'), ('t1', '<u8'), ('kind', '<u4'), ('worker', '<u4'),
                  ('sample', '<u4'), ('operation', '<u4'), ('phase', '<u4'), ('fid', '<u4'), ('value', '<u8')])

HEADER = r'''
#if !DEVICE
typedef struct { uint64_t t0,t1; uint32_t kind,worker,sample,operation,phase,fid; uint64_t value; } TraceEvent;
_Static_assert(sizeof(TraceEvent)==48,"unexpected timeline event layout");
static TraceEvent trace_events[@CAP@];
static uint32_t trace_count,trace_active,trace_sample,trace_operation=UINT32_MAX,trace_next_sample;
static _Thread_local uint32_t trace_worker=UINT32_MAX,trace_fid,trace_phase_id;
static _Thread_local uint64_t trace_phase_start;
static uint64_t trace_now(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
  return (uint64_t)ts.tv_sec*1000000000ull+(uint64_t)ts.tv_nsec;
}
static unsigned trace_phase(unsigned fid) {
  switch(fid) { @PHASE_CASES@ default:return 0; }
}
static void trace_add(unsigned kind,uint64_t t0,uint64_t t1,unsigned phase,unsigned fid,uint64_t value) {
  if(!__atomic_load_n(&trace_active,__ATOMIC_ACQUIRE))return;
  unsigned slot=__atomic_fetch_add(&trace_count,1,__ATOMIC_RELAXED);
  if(slot<@CAP@)trace_events[slot]=(TraceEvent){t0,t1,kind,trace_worker,
    __atomic_load_n(&trace_sample,__ATOMIC_RELAXED),__atomic_load_n(&trace_operation,__ATOMIC_RELAXED),phase,fid,value};
}
static void trace_point(unsigned kind,uint64_t value) {
  uint64_t t=trace_now();trace_add(kind,t,t,trace_phase_id,trace_fid,value);
}
static void trace_flush(void) {
  if(trace_phase_id)trace_add(10,trace_phase_start,trace_now(),trace_phase_id,trace_fid,0);
  trace_phase_id=0;
}
static void trace_dispatch(unsigned fid) {
  unsigned phase=trace_phase(fid);
  if(phase!=trace_phase_id){trace_flush();trace_phase_id=phase;trace_phase_start=trace_now();}
  trace_fid=fid;
  if(phase)trace_point(9,0);
}
static void trace_forward_begin(void) {
  trace_phase_id=0;
  __atomic_store_n(&trace_sample,trace_next_sample++,__ATOMIC_RELAXED);
  __atomic_store_n(&trace_operation,UINT32_MAX,__ATOMIC_RELAXED);
  __atomic_store_n(&trace_active,1,__ATOMIC_RELEASE);trace_point(5,0);
}
static void trace_operation_begin(void) {
  trace_flush();
  unsigned previous=__atomic_load_n(&trace_operation,__ATOMIC_RELAXED);
  __atomic_store_n(&trace_operation,previous==UINT32_MAX?0:previous+1,__ATOMIC_RELAXED);
  trace_point(7,0);
}
static void trace_operation_end(void) { trace_flush();trace_point(8,0); }
static void trace_forward_end(void) {
  trace_flush();trace_point(6,0);__atomic_store_n(&trace_active,0,__ATOMIC_RELEASE);
}
__attribute__((destructor)) static void trace_dump(void) {
  const char* path=getenv("BEND_TRACE_FILE");if(!path)return;
  if(__BYTE_ORDER__!=__ORDER_LITTLE_ENDIAN__){fprintf(stderr,"timeline requires little-endian output\n");return;}
  unsigned count=__atomic_load_n(&trace_count,__ATOMIC_ACQUIRE),kept=count<@CAP@?count:@CAP@;
  FILE* file=fopen(path,"wb");if(!file){perror(path);return;}
  if(fwrite(trace_events,sizeof(TraceEvent),kept,file)!=kept){perror(path);}
  fclose(file);char meta[4096];snprintf(meta,sizeof(meta),"%s.json",path);
  file=fopen(meta,"w");if(!file){perror(meta);return;}
  fprintf(file,"{\"events\":%u,\"capacity\":%u,\"overflow\":%s,\"event_bytes\":48}\n",kept,@CAP@,count>@CAP@?"true":"false");fclose(file);
}
#endif
'''




def phase_of(name):
    if re.search(r'_KERNEL_BAND_PIPELINE_EXECUTE(?:_|$)', name):
        return 'band_execute'
    if re.search(r'_KERNEL_BAND_PIPELINE_(?:RUN|WINDOWED)(?:_|$)', name):
        return 'band_run'
    if re.search(r'_TENSOR_BAND_WINDOW_(?:PREPARE|EXTRACT)(?:_|$)', name):
        return 'halo_prepare'
    if re.search(r'_TENSOR_BAND_WINDOW_ASSEMBL(?:E|ED)(?:_|$)', name):
        return 'halo_assemble'
    if re.search(r'_TENSOR_BAND_GATHER(?:_|$)', name):
        return 'band_gather'
    if re.search(r'_TRAVERSAL_PARTITION_RUN(?:_|$)', name):
        return 'dense_partition'
    if re.search(r'_KERNEL_PIPELINE_FINISH(?:_|$)', name):
        return 'dense_finish'
    if re.search(r'_(?:TENSOR_BAND_RANGE|STORAGE_COPY_RANGE)(?:_|$)', name):
        return 'range_copy'
    if re.search(r'_KERNEL_PIPELINE_(?:RUN|RUN_PACKED)(?:_|$)', name):
        return 'dense_pipeline'
    return 'unclassified'


def block_end(source, start):
    """Balanced literal C block; macros themselves are never expanded."""
    depth, index, quote = 0, start, None
    while index < len(source):
        char = source[index]
        if quote:
            if char == '\\':
                index += 2
                continue
            if char == quote:
                quote = None
        elif source.startswith('//', index):
            index = source.find('\n', index)
            if index < 0:
                break
            continue
        elif source.startswith('/*', index):
            ending = source.find('*/', index + 2)
            if ending < 0:
                raise ValueError('Unclosed C comment')
            index = ending + 2
            continue
        elif char in ('"', "'"):
            quote = char
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    raise ValueError('Unclosed C block')


def instrument(source, capacity):
    anchors = []

    def replace(needle, replacement, label):
        nonlocal source
        source = replace_once(source, needle, replacement)
        anchors.append(dict(label=label, matched=1, source_anchor=needle))

    declaration_items = re.findall(r'^#define\s+(FID_\w+)\s+(\d+)\s*$', source, re.M)
    declarations = dict((name, int(value)) for name, value in declaration_items)
    if len(declarations) != len(declaration_items):
        raise ValueError('Repeated emitted FID declaration')
    case_pattern = re.compile(r'\bWL_CASE\((FID_\w+)\)\s*\{')
    cases = list(case_pattern.finditer(source))
    identifiers = [case.group(1) for case in cases]
    if len(set(identifiers)) != len(identifiers) or not cases:
        raise ValueError('Every emitted WL_CASE must have one unique source identifier')
    if any(name not in declarations for name in identifiers):
        raise ValueError('WL_CASE identifier lacks its emitted numeric declaration')
    catalog = [dict(name=name, fid=declarations[name], phase=phase_of(name)) for name in identifiers]
    if len({item['fid'] for item in catalog}) != len(catalog):
        raise ValueError('Different WL_CASE identifiers share a numeric declaration')
    phase_cases = ' '.join(f'case {item["fid"]}:return {PHASES.index(item["phase"])};' for item in catalog if item['phase'] != 'unclassified')
    # All host WL entries set the current dispatch identity. Spans stop on a
    # change of phase or work_loop return. Synchronous inline C work belongs to
    # its current segment; this is not semantic-function exclusive time.
    insertions = [(case.end(), '\n#if !DEVICE\n    trace_dispatch(' + case.group(1) + ');\n#endif\n') for case in cases]
    windowed = [item for item in catalog if re.search(r'_KERNEL_BAND_PIPELINE_WINDOWED(?:_\d+)?$', item['name'])]
    inline = []
    if windowed:
        names = {item['name'] for item in windowed}
        insertions.extend((case.end(), '\n#if !DEVICE\n    trace_point(13,0);\n#endif\n')
                          for case in cases if case.group(1) in names)
    else:
        for index, case in enumerate(cases):
            if not re.search(r'_KERNEL_BAND_PIPELINE_RUN(?:_\d+)?$', case.group(1)):
                continue
            # WL_OPEN contributes a macro brace. Use the next emitted WL_CASE
            # as the outer slice boundary and literal balancing for each guard.
            if index + 1 == len(cases):
                raise ValueError('Band run case lacks a following emitted case boundary')
            body = source[case.end():cases[index + 1].start()]
            rows = list(re.finditer(r'if\s*\(\s*term_aux\(\s*\w+\s*\)\s*==\s*CID_\w+TENSOR_BAND_ROWS\s*\)\s*\{', body))
            # The marker is optional: a run that splits its Rows cases has no single guard to mark.
            if len(rows) != 1 or len(re.findall(r'\bWL_OPEN\b', body)) != 1:
                continue
            leaves = list(re.finditer(r'if\s*\(\s*term_aux\(\s*\w+\s*\)\s*==\s*CID_\w+TRAVERSAL_PARTITION_LEAF\s*\)\s*\{', body))
            if len(leaves) != 2:
                raise ValueError('Inlined windowed requires exactly the core and halo Leaf guards')
            row_end = block_end(body, rows[0].end() - 1)
            core_end = block_end(body, leaves[0].end() - 1)
            halo_end = block_end(body, leaves[1].end() - 1)
            if not rows[0].end() <= leaves[0].start() < leaves[1].start() < halo_end <= core_end <= row_end:
                raise ValueError('Windowed guards are not nested Rows/core Leaf/halo Leaf')
            position = case.end() + leaves[1].end()
            insertions.append((position, '\n#if !DEVICE\n          trace_point(13,0);\n#endif\n'))
            inline.append(dict(name=case.group(1), core_halo_leaf_guards=2, rows_guards=1,
                               marker='inside nested halo Leaf guard of Rows/core Leaf branch'))
    for position, insertion in sorted(insertions, reverse=True):
        source = source[:position] + insertion + source[position:]
    header = HEADER.replace('@CAP@', str(capacity)).replace('@PHASE_CASES@', phase_cases)
    replace('// Types\n// =====', header + '\n// Types\n// =====', 'host trace declarations')
    replace('Term r = work_loop(e, stk, t, seq);', 'Term r = work_loop(e, stk, t, seq);\n#if !DEVICE\n    trace_flush();\n#endif', 'work_loop return flush')
    replace('static void* pool_work(void* arg) {', 'static void* pool_work(void* arg) {\n  trace_worker=(unsigned)(uintptr_t)arg;', 'worker identity')
    # Bend 2.0.35: the host claims units too (it keeps the worker id of no pool thread), one
    # claim is one unit of rings, and a thread reports its finished units once per turn.
    replace('POOL_WAIT(a32_load(&pool_row) >> 24 == seen, pool_wake)',
            'POOL_WAIT(a32_load(&pool_row) >> 24 == seen, pool_wake)\n    trace_point(2,a32_load(&pool_row)>>23&1);', 'worker wake')
    replace('a32_acq(&pool_row); if (c >> 23 & 1) {',
            'a32_acq(&pool_row);\n    uint64_t trace_claim_start=trace_now();unsigned trace_steps=0;\n    if (c >> 23 & 1) {', 'ring claim begin')
    replace('monk_step(e, stk, rg, put0, rg, 0, NULL); } } } n += 1;',
            'monk_step(e,stk,rg,put0,rg,0,NULL);trace_steps++;\n        }\n      }\n    }\n'
            '    trace_add(3,trace_claim_start,trace_now(),0,trace_fid,((uint64_t)(c>>23&1)<<32)|trace_steps);\n    n += 1;', 'ring claim end')
    replace('if (n != 0 && a32_sub_rel(&pool_done, n) == n) {',
            'trace_flush();trace_point(4,c>>23&1);\n      if(n!=0&&a32_sub_rel(&pool_done,n)==n){', 'worker finish')
    replace('OUTLINE void pool_turn(bool grow, u32 rows) { static u32 turn;',
            'OUTLINE void pool_turn(bool grow,u32 rows){uint64_t trace_turn_start=trace_now();static u32 turn;', 'pool turn begin')
    replace('POOL_WAIT(a32_load_acq(&pool_done) != 0, pool_join) }',
            'POOL_WAIT(a32_load_acq(&pool_done) != 0, pool_join)\n  trace_add(1,trace_turn_start,trace_now(),0,trace_fid,grow);\n}', 'pool turn end')
    replace('u32 cur = row_grow((Env){ H, ALC[0] }, io_stk, 0, CUBE_G, want);',
            'uint64_t trace_host_start=trace_now();u32 cur=row_grow((Env){H,ALC[0]},io_stk,0,CUBE_G,want);trace_add(12,trace_host_start,trace_now(),0,trace_fid,0);', 'host frontier growth')
    replace('OUTLINE Term blk_copy(Env e, Term a) {',
            'OUTLINE Term blk_copy(Env e,Term a){uint64_t trace_copy_start=trace_now();', 'block copy begin')
    replace('blk_fill(e, dst, blk_loc(e.mem, a), 1ull << cls, arr); return term_blk(arr, blk_cls(a), dst);',
            'blk_fill(e,dst,blk_loc(e.mem,a),1ull<<cls,arr);trace_add(11,trace_copy_start,trace_now(),trace_phase_id,trace_fid,(1ull<<cls)*8);return term_blk(arr,blk_cls(a),dst);', 'block copy end and actual size class')
    replace('tensor_dump_seconds=0;tensor_started=io_tick();',
            'tensor_dump_seconds=0;tensor_started=io_tick();trace_forward_begin();', 'forward begin')
    replace('double elapsed=(io_tick()-tensor_started)/1e9;',
            'trace_forward_end();double elapsed=(io_tick()-tensor_started)/1e9;', 'forward end')
    begin = 'Term tensor_operation_begin_run(Env environment,Term* arguments,IoWork* work){'
    end = 'Term tensor_operation_end_run(Env environment,Term* arguments,IoWork* work){'
    begin_emitted = 'tensor_operation_begin_run' in source
    end_emitted = 'tensor_operation_end_run' in source
    if begin_emitted != end_emitted:
        raise ValueError('Operation effects must emit both begin and end')
    emitted_effects = begin_emitted and end_emitted
    explicit = emitted_effects
    if explicit:
        replace(begin, begin + 'trace_operation_begin();', 'explicit operation begin')
        replace(end, end + 'trace_operation_end();', 'explicit operation end')
        protocol = 'explicit begin/end includes operand acquisition and computation; snapshot gather and state-save IO excluded'
    else:
        replace('static Term tensor_save_run(Env environment,Term* arguments,IoWork* work){',
                'static Term tensor_save_run(Env environment,Term* arguments,IoWork* work){trace_operation_end();', 'save boundary end')
        replace('free(label);return term_pak(CID_UNIT,0);',
                'free(label);trace_operation_begin();return term_pak(CID_UNIT,0);', 'save boundary begin')
        replace('tensor_started=io_tick();trace_forward_begin();',
                'tensor_started=io_tick();trace_forward_begin();trace_operation_begin();', 'first operation')
        protocol = 'unmarked dense operation from previous save return to current save entry; dense expect_storage transfers its owner; state-save IO excluded'
    availability = {phase: [item['name'] for item in catalog if item['phase'] == phase] for phase in PHASES[1:]}
    return source, dict(anchors=anchors, case_count=len(cases), declaration_count=len(declarations),
                        case_identifiers=catalog, phases=availability,
                        windowed=dict(emitted=[item['name'] for item in windowed], inlined=inline,
                                      absent=not windowed and not inline, inclusive_ns=None),
                        operation_effects_emitted=emitted_effects,
                        operation_boundaries='explicit' if explicit else 'save_boundaries',
                        mode='explicit_effects' if explicit else 'save_pause_resume',
                        public_begin_effect_emitted=begin_emitted,
                        public_end_effect_emitted=end_emitted,
                        semantic_scope='operand_acquisition_and_computation; snapshot_gather_and_state_save_io_excluded',
                        operation_protocol=protocol)


def overlap(intervals, first, last):
    return [(max(int(left), first), min(int(right), last)) for left, right in intervals if left < last and first < right]


def partial_claim_time(claims, first, last, workers):
    edges = []
    for claim in claims:
        if int(claim['value']) >> 32:
            for left, right in overlap([(claim['t0'], claim['t1'])], first, last):
                edges.extend([(left, 1), (right, -1)])
    active, previous, total = 0, first, 0
    for time, change in sorted(edges):
        if 0 < active < workers:
            total += time - previous
        active += change
        previous = time
    return total


def read_events(path):
    meta = json.loads(path.with_name(path.name + '.json').read_text())
    assert not meta['overflow'], 'Trace overflow: increase --event-capacity and rerun this affected diagnostic'
    events = np.fromfile(path, dtype=EVENT)
    assert EVENT.itemsize == meta['event_bytes'] == 48 and len(events) == meta['events']
    assert path.stat().st_size == len(events) * EVENT.itemsize, 'Truncated trace event'
    return events[np.argsort(events['t0'], kind='stable')]


def analyze(path, graph, threads, instrumentation):
    events = read_events(path)
    source_identifiers = {item['fid']: item['name'] for item in instrumentation['case_identifiers']}
    beginnings = events[events['kind'] == KINDS['forward_begin']]
    endings = events[events['kind'] == KINDS['forward_end']]
    assert list(beginnings['sample']) == list(range(9)) and list(endings['sample']) == list(range(9))
    samples = []
    for repetition in range(9):
        sample = events[events['sample'] == repetition]
        starts = sample[sample['kind'] == KINDS['operation_begin']]
        ends = sample[sample['kind'] == KINDS['operation_end']]
        # Without markers, a final save starts an empty trailing interval, never another operation.
        starts = starts[starts['operation'] < len(graph['ops'])]
        assert list(starts['operation']) == list(range(len(graph['ops'])))
        assert list(ends['operation']) == list(range(len(graph['ops'])))
        rows = []
        for index, operation in enumerate(graph['ops']):
            lo, hi = int(starts[index]['t0']), int(ends[index]['t0'])
            assert lo <= hi
            mine = sample[(sample['t0'] >= lo) & (sample['t1'] <= hi)]
            turns = mine[mine['kind'] == KINDS['turn']]
            claims = mine[mine['kind'] == KINDS['claim']]
            clones = mine[mine['kind'] == KINDS['block_copy']]
            copy_dispatch = []
            for fid in sorted(set(int(event['fid']) for event in clones)):
                group = clones[clones['fid'] == fid]
                copy_dispatch.append(dict(last_dispatch_identifier=source_identifiers.get(fid), fid=fid,
                                          calls=len(group), bytes=sum(int(event['value']) for event in group),
                                          ms=sum(int(event['t1']) - int(event['t0']) for event in group) / 1e6))
            phases = {}
            for phase_id, phase in enumerate(PHASES[1:], 1):
                spans = mine[(mine['kind'] == KINDS['dispatch_span']) & (mine['phase'] == phase_id)]
                entries = mine[(mine['kind'] == KINDS['dispatch']) & (mine['phase'] == phase_id)]
                phases[phase] = dict(present_in_source=bool(instrumentation['phases'][phase]), entries=len(entries),
                                     dispatch_segment_ms=sum(int(event['t1']) - int(event['t0']) for event in spans) / 1e6 if len(spans) else None,
                                     inclusive_ms=None)
            wake_envelope = 0
            partial = 0
            for turn in turns:
                wakes = mine[(mine['kind'] == KINDS['wake']) & (mine['t0'] >= turn['t0']) & (mine['t0'] <= turn['t1'])]
                if len(wakes):
                    wake_envelope += int(wakes['t0'].max()) - int(turn['t0'])
                if turn['value']:
                    partial += partial_claim_time(claims, int(turn['t0']), int(turn['t1']), threads)
            busy = {str(worker): sum(int(claim['t1']) - int(claim['t0']) for claim in claims if claim['worker'] == worker) / 1e6 for worker in range(threads)}
            rows.append(dict(operation=index, kind=operation['kind'], ms=(hi - lo) / 1e6,
                             grow_ms=sum(int(t['t1']) - int(t['t0']) for t in turns if t['value']) / 1e6,
                             work_ms=sum(int(t['t1']) - int(t['t0']) for t in turns if not t['value']) / 1e6,
                             host_grow_ms=sum(int(e['t1']) - int(e['t0']) for e in mine if e['kind'] == KINDS['host_grow']) / 1e6,
                             grow_partial_claim_ms=partial / 1e6, wake_envelope_ms=wake_envelope / 1e6,
                             worker_claim_ms=busy, block_copy_calls=len(clones), block_copy_bytes=sum(int(e['value']) for e in clones),
                             block_copy_ms=sum(int(e['t1']) - int(e['t0']) for e in clones) / 1e6,
                             block_copy_by_last_dispatch=copy_dispatch,
                             windowed_entries=int((mine['kind'] == KINDS['windowed_entry']).sum())
                                 if not instrumentation['windowed']['absent'] else None, phases=phases))
        samples.append(dict(sample=repetition, forward_ms=(int(endings[repetition]['t0']) - int(beginnings[repetition]['t0'])) / 1e6, operations=rows))
    measured = samples[2:]
    categories = {}
    fields = ('ms', 'grow_ms', 'work_ms', 'host_grow_ms', 'grow_partial_claim_ms', 'wake_envelope_ms', 'block_copy_ms', 'block_copy_calls', 'block_copy_bytes', 'windowed_entries')
    for kind in sorted({operation['kind'] for operation in graph['ops']}):
        selected = [[row for row in sample['operations'] if row['kind'] == kind] for sample in measured]
        categories[kind] = {field: statistics.median(sum(row[field] for row in rows) for rows in selected)
                            if all(row[field] is not None for rows in selected for row in rows) else None for field in fields}
        categories[kind]['phases'] = {}
        for phase in PHASES[1:]:
            values = [sum(row['phases'][phase]['dispatch_segment_ms'] or 0 for row in rows)
                      if any(row['phases'][phase]['dispatch_segment_ms'] is not None for row in rows) else None for rows in selected]
            categories[kind]['phases'][phase] = dict(
                present_in_source=bool(instrumentation['phases'][phase]),
                dispatch_segment_median_ms=statistics.median(values) if all(value is not None for value in values) else None,
                entries_median=statistics.median(sum(row['phases'][phase]['entries'] for row in rows) for rows in selected),
                inclusive_ms=None)
    return dict(protocol='same process: 2 warmups and 7 samples', samples=samples, categories=categories,
                forward_median_ms=statistics.median(sample['forward_ms'] for sample in measured),
                unavailable=dict(root_construction_ms=None, semantic_leaf_ms=None, halo_assembly_inclusive_ms=None,
                                 parameter_owner_clone_ms=None, leaf_count=None),
                interpretation='Diagnostic timings include trace overhead. Claims describe ring occupancy, not CPU residency. '
                               'Wake envelopes overlap grow/work. Dispatch segments include synchronous inline C until the next family transition or work_loop return; '
                               'they are not semantic self time or inclusive semantic calls. '
                               'Block copies and dispatch/claim intervals overlap; do not add them. '
                               'Block-copy attribution uses the last real dispatch identifier, not an inferred parameter owner. '
                               'Root construction, semantic leaf time/count, and inclusive halo/parameter-owner durations have no complete anchors. '
                               'Absent/inlined phases and unavailable inclusive durations are null, not zero. Snapshot IO follows the recorded operation protocol.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, default=ROOT / 'out/models')
    parser.add_argument('--model-stem', help='Explicit preserved dynamic-worker entry')
    parser.add_argument('--case', default='bus_640')
    parser.add_argument('--threads', type=int, choices=(1, 4), required=True)
    parser.add_argument('--output', type=Path, required=True)
    # 48 bytes per event; the four-thread bus_640 trace overflows a quarter of this.
    parser.add_argument('--event-capacity', type=int, default=4194304)
    args = parser.parse_args()
    if not 1 <= args.event_capacity < 2**31:
        parser.error('--event-capacity must fit the trace counter')
    if not re.fullmatch(r'[a-z][a-z0-9_]*', args.case):
        parser.error('--case must use snake_case')
    models = args.models.resolve()
    requested = args.case + (f'_cpu_{args.threads}t' if args.threads != 1 else '')
    selected = args.model_stem or (requested if all((models / f'{requested}.{suffix}').is_file() for suffix in ('c', 'json')) else args.case)
    stem, graph, dynamic = baseline_stem(parser, models, requested, selected, args.threads)
    assert graph['case'] == args.case
    original = models / f'{stem}.c'
    build = json.loads(original.with_suffix('.c.build.json').read_text())
    toolchain = json.loads((ROOT / 'scripts/toolchain.json').read_text())
    assert build['compiler_commit'] == toolchain['commit'], 'Use the pinned official compiler'
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    source, instrumentation = instrument(original.read_text(), args.event_capacity)
    output.mkdir(parents=True)
    diagnostic = output / 'diagnostic.c'
    diagnostic.write_text(source, newline='\n')
    (output / 'original.c').write_bytes(original.read_bytes())
    (output / 'original.c.build.json').write_text(json.dumps(build, indent=2) + '\n', newline='\n')
    (output / 'graph.json').write_bytes((models / f'{stem}.json').read_bytes())
    entry = models / f'{stem}.bend'
    if entry.is_file():
        (output / 'original.bend').write_bytes(entry.read_bytes())
    subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/check_generated_c.py'), str(original), str(diagnostic)], check=True)
    flags = ['-O3', '-march=native', '-ffp-contract=off', '-std=c11']
    compiler = os.environ.get('CC', 'clang')
    binaries = {}
    for name, path in (('normal', original), ('diagnostic', diagnostic)):
        executable = output / name
        subprocess.run([compiler, *flags, str(path), '-lpthread', '-lm', '-o', str(executable)], check=True)
        binaries[name] = executable

    def run(name, folder, repetitions, dump, trace=None):
        folder.mkdir()
        environment = dict(os.environ, BEND_OUTDIR=str(folder), BEND_DUMP=str(int(dump)), BEND_REPETITIONS=str(repetitions))
        if trace:
            environment['BEND_TRACE_FILE'] = str(trace)
        else:
            environment.pop('BEND_TRACE_FILE', None)
        with (folder / 'stdout.log').open('w', newline='\n') as stdout:
            subprocess.run([str(binaries[name]), '--threads', str(args.threads), '--gpu', 'off'], cwd=ROOT,
                           env=environment, check=True, stdout=stdout)

    normal, validated = output / 'validation_normal', output / 'validation_diagnostic'
    run('normal', normal, 1, True)
    run('diagnostic', validated, 1, True, output / 'validation_trace.bin')
    validation_events = read_events(output / 'validation_trace.bin')
    assert sum(validation_events['kind'] == KINDS['forward_begin']) == 1
    assert sum(validation_events['kind'] == KINDS['forward_end']) == 1
    for label in graph['snapshots']:
        assert (normal / f'{label}.bin').read_bytes() == (validated / f'{label}.bin').read_bytes(), label
    trace = output / 'trace.bin'
    run('diagnostic', output / 'samples', 9, False, trace)
    native_samples = [json.loads(line) for line in (output / 'samples/native_samples.jsonl').read_text().splitlines()]
    assert [row['sample'] for row in native_samples] == list(range(9)), 'Incomplete native sample receipt'
    result = analyze(trace, graph, args.threads, instrumentation)
    result['native_samples'] = native_samples
    result['provenance'] = dict(case=args.case, stem=stem, threads=args.threads, dynamic_workers=dynamic,
                              model_source=str(original), build=build, compiler_flags=flags,
                              compiler=subprocess.check_output([compiler, '--version'], text=True),
                              snapshot_outputs_bitwise=len(graph['snapshots']),
                              trace_capacity=args.event_capacity, trace_capacity_bytes=args.event_capacity * EVENT.itemsize,
                              comparison_scope='Whole-version 3.5/3.6/planner/ownership differences; no single-factor attribution.')
    result['instrumentation'] = instrumentation
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', newline='\n')
    print(output / 'summary.json')


if __name__ == '__main__':
    main()
