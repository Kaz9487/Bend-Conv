"""Replay actual public band planner choices on a generated model's metadata.

The Bend diagnostic calls the real planners. Python connects tensor layouts in
graph order and records estimates; it neither selects plans nor times kernels.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('graph', type=Path, help='Current generated model JSON; thread count selects the replay')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--operations', type=Path, help='Optional matching profile_operations results.json')
    args = parser.parse_args()
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
    subprocess.run(['node', str(ROOT / 'scripts/bend_launcher.mjs'), str(source), '-o', str(diagnostic_c)], check=True)
    executable = source.with_suffix('')
    flags = ['-O3', '-ffp-contract=off', '-std=c11']
    subprocess.run([os.environ.get('CC', 'clang'), *flags, str(diagnostic_c), '-lpthread', '-lm', '-o', str(executable)], check=True)
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
