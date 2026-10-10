"""Report executed handwritten plans or replay generated-model metadata.

Handwritten mode observes executed convolution plans in generated C and checks
its prediction against an uninstrumented run. Graph mode replays layout metadata
through the Bend planners. Neither mode selects plans or times kernels.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]




def shape_literal(shape):
    if any(not isinstance(value, int) or not 0 <= value < 2**32 for value in shape):
        raise ValueError(f'Invalid U32 shape: {shape}')
    return '[' + ','.join(map(str, shape)) + ']'


def geometry(shape):
    shape_literal(shape)
    if len(shape) == 3:
        batch, channels, height, width = (1, *shape)
    elif len(shape) == 4:
        batch, channels, height, width = shape
    else:
        raise ValueError(f'Band geometry requires CHW or NCHW: {shape}')
    return f'Bands.Geometry{{{batch},{channels},{height},{width}}}'


def generate_entry(graph):
    workers = graph['threads']
    if not isinstance(workers, int) or not 1 <= workers < 2**32:
        raise ValueError('Graph threads must be a positive U32')
    tensors = graph['tensors']
    for tensor in tensors:
        shape_literal(tensor['shape'])
        if math.prod(tensor['shape']) != tensor['count'] or not 0 <= tensor['count'] < 2**32:
            raise ValueError('Tensor count must match shape and fit a U32 argument')
    source = ['import Base',
              'import ../../../benchmarks/model/plan_observation.bend as Observation',
              'import ../../../lib/tensor_band.bend as Bands',
              'import ../../../lib/tensor_band_window.bend as Windows',
              'import ../../../lib/math_activation.bend as Activation',
              'def main() -> IO(Unit):', '  do IO<Unit>:']
    available = {index for index, tensor in enumerate(tensors) if tensor['path'] is not None}
    for index, tensor in enumerate(tensors):
        source.append(f'    Observation.emit_tensor_count({index},{shape_literal(tensor["shape"])})')
    for index in sorted(available):
        source.append(f'    +state_{index} : Observation.Receipt = Observation.dense()')
    descriptions = []
    for index, operation in enumerate(graph['ops']):
        kind, output = operation['kind'], operation['output']
        result_shape = tensors[output]['shape']

        def state(slot):
            if slot not in available:
                raise ValueError(f'Operation {index} reads a tensor before its producer: {slot}')
            return f'state_{slot}'

        description = dict(operation=index, kind=kind, output=output, shape=result_shape,
                           graph_operation={key: value for key, value in operation.items()
                                            if key not in ('partition', 'split_depth')})
        if kind == 'conv':
            incoming = tensors[operation['input']]
            weights, bias = tensors[operation['weights']], tensors[operation['bias']]
            if operation['activation'] not in (0, 1):
                raise ValueError('This generated model schema encodes identity or SiLU convolution')
            activation = 'Activation.SiLU{}' if operation['activation'] else 'Activation.Identity{}'
            expression = (f'Observation.checked_convolution({shape_literal(incoming["shape"])},'
                          f'{shape_literal(weights["shape"])},{shape_literal(bias["shape"])},'
                          f'{operation["stride"]},{operation["padding"]},{weights["count"]},{bias["count"]},'
                          f'{activation},{workers},{state(operation["input"])})')
            description['activation'] = 'silu' if operation['activation'] else 'identity'
        elif kind == 'add':
            left, right = operation['left'], operation['right']
            if tensors[left]['shape'] != tensors[right]['shape'] or tensors[left]['shape'] != result_shape:
                raise ValueError('Generated add requires equal public shapes')
            expression = f'Observation.combined({geometry(result_shape)},{workers},{state(left)},{state(right)})'
        elif kind == 'concat':
            slots = operation['inputs']
            if not slots:
                raise ValueError('Public concat requires a nonempty parts list')
            plans = [f'Observation.layout({state(slot)})' for slot in slots]
            # The model invokes public concat(axis=1); rank-three decode concat
            # is not a channel concat and therefore retains the dense route.
            expression = (f'Observation.concatenated({shape_literal(result_shape)},1,'
                          '[' + ','.join(plans) + '])')
            description['public_arguments'] = dict(axis=1, source='generate_yolo_graph emits F.concat(parts,1)')
            description['layout_selection'] = 'all inputs banded with compatible plans and nonempty output; otherwise dense append'
            description['parallel_scope'] = 'per-input aligned append; no whole-operation parallel choice'
        elif kind == 'pool':
            incoming = tensors[operation['input']]
            kernel, stride, padding = (operation[key] for key in ('kernel', 'stride', 'padding'))
            if any(type(value) is not int or not 0 <= value < 2**32 for value in (kernel, stride, padding)):
                raise ValueError('Pool metadata must record U32 public call arguments')
            description['public_arguments'] = dict(kernel=kernel, stride=stride, padding=padding,
                                                   source='graph metadata recorded by the same public call arguments')
            expression = (f'Observation.pooled_public({shape_literal(incoming["shape"])},'
                          f'{geometry(incoming["shape"])},{kernel},{stride},{padding},{workers},{state(operation["input"])})')
        elif kind == 'up':
            incoming = tensors[operation['input']]
            ih, iw = incoming['shape'][-2:]
            oh, ow = result_shape[-2:]
            if not ih or not iw or oh % ih or ow % iw or oh // ih != ow // iw:
                raise ValueError('Generated nearest upsample must have one positive spatial scale')
            scale = operation['scale']
            if type(scale) is not int or not 0 < scale < 2**32 or oh // ih != scale:
                raise ValueError('Graph shape differs from its recorded public upsample scale')
            description['public_arguments'] = dict(scale=scale, source='graph metadata recorded by the public call')
            expression = (f'Observation.upsample_public({shape_literal(incoming["shape"])},'
                          f'{geometry(incoming["shape"])},{workers},{scale},{state(operation["input"])})')
        elif kind == 'decode':
            operations = operation['operations']
            if type(operations) is not float or not 0.0 <= operations < 2.0**24:
                raise ValueError('Decode metadata must record the operation count of the public call')
            description['public_arguments'] = dict(operations=operations, source='graph metadata recorded by the public call')
            expression = (f'Observation.decoded({geometry(tensors[operation["input"]]["shape"])},{workers},{operations},'
                          f'{state(operation["input"])})')
            description['gather_boundary'] = 'public reshape after the indexed decode, before transpose'
        else:
            raise ValueError(f'Unsupported graph operation: {kind}')
        if output in available:
            raise ValueError('An operation output must have one producer')
        source.extend([f'    +state_{output} : Observation.Receipt = {expression}', f'    Observation.emit({index},state_{output})'])
        if kind == 'concat' and len(result_shape) == 4:
            for part, slot in enumerate(operation['inputs']):
                source.append(f'    Observation.emit_append({index},{part},Observation.append_parallel('
                              f'{geometry(tensors[slot]["shape"])},{workers},{state(slot)},state_{output}))')
        available.add(output)
        descriptions.append(description)
    source.append('    return Unit{}')
    return '\n'.join(source) + '\n', descriptions


# The handwritten model reports the plans it actually executes. Instrumentation
# observes named dispatch arguments without changing their ownership or planners.
PLAN_HEADER = r'''
static char plan_name[256];
static unsigned plan_index;
static void plan_label(Env e, Term name) {
  unsigned n=0;
  while(term_aux(name)==CID_SCON && n+1<sizeof(plan_name)) {
    u64 p=term_peek(e.mem,name);
    plan_name[n++]=(char)e.mem[p]; name=e.mem[p+1];
  }
  plan_name[n]=0;
}
static void plan_tree(Env e, Term tree, unsigned depth, unsigned *leaves, unsigned *maximum) {
  u64 p=term_peek(e.mem,tree);
  if(term_aux(tree)==CID_______LIB_TRAVERSAL_PARTITION_LEAF) {
    ++*leaves; if(depth>*maximum)*maximum=depth;
  } else {
    plan_tree(e,e.mem[p],depth+1,leaves,maximum);
    plan_tree(e,e.mem[p+1],depth+1,leaves,maximum);
  }
}
static void plan_emit(const char *route, unsigned depth, unsigned leaves, unsigned parallel,
                      unsigned ci,unsigned co,unsigned h,unsigned w,unsigned k,unsigned s,unsigned p) {
  fprintf(stderr,"PLAN {\"convolution\":%u,\"name\":\"%s\",\"route\":\"%s\","
    "\"depth\":%u,\"leaves\":%u,\"parallel\":%s,\"dimensions\":[%u,%u,%u,%u,%u,%u,%u]}\n",
    plan_index++,plan_name,route,depth,leaves,parallel?"true":"false",ci,co,h,w,k,s,p);
}
'''


def instrument_handwritten(source):
    def dispatch(match):
        symbol = match['symbol'].removeprefix('FID_').lstrip('_')
        # Bind argument names from this build, not native function IDs or lines.
        arguments = {}
        for variable, name in re.findall(r'\b((_\w+)_\d+) = r\d+;', match[0]):
            arguments.setdefault(name[1:], []).append(variable)
        extra = ''
        if symbol in ('CONV', 'LOGITS'):
            extra = f'plan_label(e,{arguments["name"][0]});'
        elif re.fullmatch(r'LIB_TENSOR_CONVOLUTION_BAND_OUTPUT_\d+', symbol):
            g = arguments['geometry']
            extra = ('unsigned plan_leaves=0,plan_depth=0;'
                     f'plan_tree(e,{arguments["input"][0]},0,&plan_leaves,&plan_depth);'
                     'plan_emit("bands",plan_depth,plan_leaves,'
                     + arguments['parallel'][0] + ','
                     + ','.join(f'(unsigned){value}' for value in g[2:9]) + ');')
        elif re.fullmatch(r'LIB_KERNEL_PIPELINE_RUN_PACKED_\d+', symbol):
            w = arguments['work']
            extra = (f'unsigned plan_depth={arguments["depth"][0]},plan_limit=0;'
                     f'for(unsigned n=({w[7]}+7)/8;n>1;n>>=1)++plan_limit;'
                     'if(plan_depth>plan_limit)plan_depth=plan_limit;'
                     'plan_emit("dense",plan_depth,1u<<plan_depth,plan_depth>0,'
                     f'{w[8]}/({w[3]}*{w[3]}),'
                     + ','.join(w[:6]) + ');')
        return match[0] + ('\n    ' + extra if extra else '')

    marker = source.index('static Term work_loop(')
    source = source[:marker] + PLAN_HEADER + '\n' + source[marker:]
    return re.sub(r'WL_CASE\((?P<symbol>FID_\w+)\)\s*\{[\s\S]*?\bWL_OPEN', dispatch, source)


def handwritten_plans(args):
    output = args.output.resolve()
    directory = output.parent / output.stem
    directory.mkdir(parents=True, exist_ok=True)
    generated = args.model_c.resolve() if args.model_c else directory / 'yolov5n.c'
    if not args.model_c:
        subprocess.run([sys.executable, str(ROOT / 'scripts/toolchain_rules/lint.py'),
                        'examples/yolov5/yolov5n.bend'], cwd=ROOT, check=True)
        subprocess.run(['node', str(ROOT / 'scripts/bend_launcher.mjs'),
                        'examples/yolov5/yolov5n.bend', '-o', str(generated)], cwd=ROOT, check=True)
    observed_c = directory / 'observed.c'
    observed_c.write_text(instrument_handwritten(generated.read_text()))
    compiler = os.environ.get('CC') or shutil.which('clang-19') or shutil.which('clang')
    flags = ['-O3', '-march=native', '-ffp-contract=off', '-std=c11']
    runs = []
    for label, source in [('plain', generated), ('observed', observed_c)]:
        binary = directory / label
        subprocess.run([compiler, *flags, str(source), '-lpthread', '-lm', '-o', str(binary)],
                       cwd=ROOT, check=True)
    for threads in args.threads:
        for label in ('plain', 'observed'):
            folder = directory / f'{label}_{threads}t'
            folder.mkdir(exist_ok=True)
            env = dict(os.environ)
            env.pop('YOLO_LAYERS', None)
            env.update(YOLO_SIZE=str(args.size), YOLO_INPUT=str(args.input.resolve()),
                       YOLO_OUTPUT=str(folder), YOLO_EXTRA='0')
            result = subprocess.run([str(directory / label), '--threads', str(threads), '--gpu', 'off'],
                                    cwd=ROOT, env=env, capture_output=True, text=True, check=True)
            (folder / 'stdout.log').write_text(result.stdout)
            (folder / 'stderr.log').write_text(result.stderr)
        before = (directory / f'plain_{threads}t/pred.npy').read_bytes()
        after = (directory / f'observed_{threads}t/pred.npy').read_bytes()
        if before != after:
            raise ValueError(f'Instrumented prediction differs at {threads} threads')
        rows = [json.loads(line.removeprefix('PLAN ')) for line in result.stderr.splitlines()
                if line.startswith('PLAN ')]
        runs.append(dict(threads=threads, convolutions=rows, prediction_bitwise=True))
    output.write_text(json.dumps(dict(model='examples/yolov5/yolov5n.bend', size=args.size,
        compiler_flags=flags, runs=runs,
        scope='Executed handwritten convolution plans; depth is the effective dense split or '
              'maximum band tree depth, leaves counts actual bands or dense partitions. '
              'Parallel is the scheduling choice, not measured CPU occupancy. '
              'Dimensions are ci, co, h, w, k, stride, padding. No timing estimates.'), indent=2) + '\n')
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('graph', nargs='?', type=Path, help='Current generated model JSON; thread count selects the replay')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--operations', type=Path, help='Optional matching profile_operations results.json')
    parser.add_argument('--handwritten', action='store_true', help='Observe the handwritten YOLO convolution plans')
    parser.add_argument('--model-c', type=Path, help='Preserved handwritten C; otherwise build the current checkout')
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--input', type=Path, default=ROOT / 'out/results/bus_640/input.bin')
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8])
    args = parser.parse_args()
    if args.handwritten:
        handwritten_plans(args)
        return
    if args.graph is None:
        parser.error('Supply a graph or --handwritten')
    graph_path = args.graph.resolve()
    graph = json.loads(graph_path.read_text())
    generated = graph_path.with_suffix('.c')
    build = json.loads(generated.with_suffix('.c.build.json').read_text())
    toolchain = json.loads((ROOT / 'scripts/toolchain.json').read_text())
    assert build['compiler_commit'] == toolchain['commit'], 'Model must use the pinned official compiler'
    output = args.output.resolve()
    if not re.fullmatch(r'[a-z][a-z0-9_]*', output.stem):
        parser.error('Use a snake_case report file stem')
    directory = ROOT / 'out/build/model_planning'
    directory.mkdir(parents=True, exist_ok=True)
    entry, descriptions = generate_entry(graph)
    source = directory / f'{output.stem}.bend'
    source.write_text(entry)
    diagnostic_c = source.with_suffix('.c')
    subprocess.run([sys.executable, str(ROOT / 'scripts/toolchain_rules/lint.py'), str(source)],
                   cwd=ROOT, check=True)
    subprocess.run(['node', str(ROOT / 'scripts/bend_launcher.mjs'), str(source), '-o', str(diagnostic_c)], check=True)
    executable = source.with_suffix('')
    flags = ['-O3', '-ffp-contract=off', '-std=c11']
    subprocess.run([os.environ.get('CC') or shutil.which('clang-19') or shutil.which('clang'), *flags, str(diagnostic_c), '-lpthread', '-lm', '-o', str(executable)], check=True)
    observed = subprocess.run([str(executable), '--threads', '1', '--gpu', 'off'], cwd=ROOT,
                              check=True, capture_output=True, text=True)
    records = [json.loads(line) for line in observed.stdout.splitlines()]
    counts = [record for record in records if 'tensor' in record]
    assert [record['tensor'] for record in counts] == list(range(len(graph['tensors'])))
    for count, tensor in zip(counts, graph['tensors'], strict=True):
        if count['count'] != tensor['count']:
            raise ValueError('Actual public Tensor.count_of rejected or differs from a graph tensor')
    rows = [record for record in records if 'part' not in record and 'tensor' not in record]
    append_records = [record for record in records if 'part' in record]
    assert [row['operation'] for row in rows] == list(range(len(descriptions)))
    for row, description in zip(rows, descriptions, strict=True):
        if row['route'].startswith('invalid_'):
            raise ValueError(f'Actual public convolution guard rejected operation {row["operation"]}')
        row.update(description)
        if row['route'] not in ('band_initial', 'dense_initial', 'dense_convolution'):
            row['initial_or_dense_depth'] = None
        if row['kind'] == 'concat':
            row['parallel'] = None
            row['append_parallel'] = [record for record in append_records if record['operation'] == row['operation']]
    measured = None
    if args.operations:
        measured = json.loads(args.operations.read_text())
        assert measured['threads'] == graph['threads'] and measured['snapshot_outputs_bitwise']
        assert len(measured['operations']) == len(rows)
        for row, actual in zip(rows, measured['operations'], strict=True):
            assert row['operation'] == actual['operation']
            row['measured_ms'] = actual['ms']
            row['estimated_error_fraction'] = (row['estimated_ns'] / 1e6 - actual['ms']) / actual['ms'] if row['estimated_ns'] is not None and actual['ms'] else None
    result = dict(case=graph['case'], threads=graph['threads'],
                  compiler_flags=flags, tensor_counts=counts, operations=rows,
                  scope='Actual Bend planner metadata replay on ready dense parameters and valid graph owners; estimates are unmeasured. '
                        'Layout maps reuse Windows.window; binary alignment and loop flags reuse the public schedulers. '
                        'Concat reuses the actual compatible_plans/first_plan and geometry/count/zero guards; '
                        'mixed or different plans use dense append. Comparisons cost metadata traversal; '
                        'per-input parallel decisions do not model a whole-concat time. '
                        'Non-convolution times and multi-image dense times are null. Snapshot gathers outside operations are excluded. '
                        'This observes planner choices, not an execution trace or proof of runtime metadata.',
                  measured_profile=str(args.operations) if measured is not None else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
