"""Emit a complete Bend forward from the shared YOLO graph.

Every operation, convolution included, is a public tensor call
(lib/tensor_f32.bend). Convolution fuses SiLU into its single output store.
"""

import argparse
from collections import Counter
import json
import re
import struct
import sys
from pathlib import Path
from yolo_graph import build_graph, input_slots

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def shape_literal(shape):
    return '[' + ','.join(str(extent) for extent in shape) + ']'


# Scalar operations of one decoded element, for the band schedule: the
# sigmoid (40), six index divisions and remainders, the field match and the
# costliest field (two doublings, two multiplies and the anchor lookup).
DECODE_OPERATIONS = 54.0


def decode_function(name, operation):
    """Per-head decode as a direct template over the logits in NCHW order.

    Index i of [1, 255, height, width] has channel i / (height * width), row
    (i % (height * width)) / width and column i % width; the anchor is channel
    / 85 and the field channel % 85. Decoding before the reshape keeps the
    logits' band layout, so the leaves decode in parallel. The arithmetic is
    the reference YOLOv5 decode in the same order.
    """
    height, width, stride = operation['input_height'], operation['input_width'], float(operation['stride'])
    widths = [pair[0] for pair in operation['anchors']]
    heights = [pair[1] for pair in operation['anchors']]

    def table(values):
        return '\n'.join(
            [f'    case {index}: {float(value)}' for index, value in enumerate(values[:-1])]
            + [f'    case _: {float(values[-1])}']
        )

    return [
        f'def {name}_anchor_width(anchor: U32) -> F32:',
        '  match anchor:',
        table(widths),
        '',
        f'def {name}_anchor_height(anchor: U32) -> F32:',
        '  match anchor:',
        table(heights),
        '',
        f'def {name}_field(+score: F32,field: U32,+column: U32,+row: U32,+anchor: U32) -> F32:',
        '  match field:',
        '    case 0:',
        f'      ((score * 2.0 - 0.5 + U32.to_f32(column)) * {stride} : F32)',
        '    case 1:',
        f'      ((score * 2.0 - 0.5 + U32.to_f32(row)) * {stride} : F32)',
        '    case 2:',
        f'      (score * 2.0 * (score * 2.0) * {name}_anchor_width(anchor) : F32)',
        '    case 3:',
        f'      (score * 2.0 * (score * 2.0) * {name}_anchor_height(anchor) : F32)',
        '    case _:',
        '      score',
        '',
        f'def {name}(index: U32,value: F32) -> F32:',
        '  +index = index',
        f'  +channel = (index / {width * height} : U32)',
        f'  +cell = (index % {width * height} : U32)',
        f'  {name}_field((1.0 / (1.0 + F32.exp(F32.neg(value))) : F32),(channel % 85 : U32),(cell % {width} : U32),'
        f'(cell / {width} : U32),(channel / 85 : U32))',
        '',
    ]


def emit_model(graph, target):
    tensors = graph['tensors']
    lines = [
        'import Base',
        'import ../../lib/tensor_f32.bend as F',
        'import ../../lib/math_activation.bend as Act',
        '',
    ]
    for name, arguments, result in [
        ('load', 'path: String, id: U32, count: U32', 'Unit'),
        ('buffer_like', 'x: F.Storage()', 'Unit'),
        ('get_buffer', 'id: U32, length: Nat', 'F.Storage()'),
        ('take_buffer', 'id: U32, length: Nat', 'F.Storage()'),
        ('save_buffer', 'id: U32, x: F.Storage(), label: String', 'Unit'),
        ('begin', '', 'Unit'),
        ('end', '', 'Unit'),
        ('repetitions', '', 'U32'),
        ('dump_enabled', '', 'Bool'),
        ('operation_begin', 'id: U32', 'Unit'),
        ('operation_end', 'id: U32', 'Unit'),
    ]:
        lines += [
            f'def Tensor.{name}({arguments}) -> IO({result}):',
            '  import "../../examples/yolov5/tensor_io.c"',
            '',
        ]

    initialization = [
        f'Tensor.load("{tensor["path"]}", {index}, {tensor["count"]})'
        for index, tensor in enumerate(tensors)
        if tensor['path']
    ]
    decoders = {}
    for operation in graph['ops']:
        if operation['kind'] == 'decode':
            name = f'decode_{len(decoders)}'
            decoders[operation['output']] = name
            lines += decode_function(name, operation)
    graph['memory'] = dict(
        policy='Bend retains public Tensor owners between operations; copies use F.copy and snapshots gather only when requested.'
    )

    def function(name, body, parameters=''):
        lines.extend([f'def {name}({parameters}) -> IO(Unit):', '  do IO<Unit>:'])
        lines.extend('    ' + statement for statement in body)
        lines.append('')

    initializers = []
    for start in range(0, len(initialization), 16):
        name = f'initialize_{start // 16}'
        function(name, initialization[start : start + 16])
        initializers.append(name)

    remaining_reads = Counter(slot for operation in graph['ops'] for slot in input_slots(operation))
    available = {index for index, tensor in enumerate(tensors) if tensor['path']}
    live = set()
    ownership = Counter()
    labels = {index: label for label, index in graph['snapshots'].items()}
    lines += '''def copied(pair: F.Tensor() & F.Tensor(),body: F.Tensor() -> F.Tensor() -> IO(Unit)) -> IO(Unit):
  (left,right) = pair
  body(left,right)

def snapshot_pair(id: U32,label: String,pair: F.Tensor() & F.Tensor()) -> IO(F.Tensor()):
  (tensor,snapshot) = pair
  do IO<F.Tensor()>:
    storage : F.Storage() <- F.expect_storage(snapshot)
    Tensor.save_buffer(id,storage,label)
    IO.pure(F.Tensor(),tensor)

def snapshot(enabled: Bool,id: U32,label: String,tensor: F.Tensor()) -> IO(F.Tensor()):
  match enabled:
    case False{}: IO.pure(F.Tensor(),tensor)
    case True{}: snapshot_pair(id,label,F.copy(tensor))

'''.splitlines()
    functions = []
    for index, operation in enumerate(graph['ops']):
        kind, output = operation['kind'], operation['output']
        count = tensors[output]['count']
        parameters = ['+workers: U32', '+dump: Bool'] + [f'tensor_{slot}: F.Tensor()' for slot in sorted(live)]
        body = []
        copies = []

        def read(slot, variable):
            assert slot in available and remaining_reads[slot] > 0, 'Read of unavailable tensor'
            remaining_reads[slot] -= 1
            transfer = remaining_reads[slot] == 0
            action = 'take' if transfer else 'get'
            if transfer:
                available.remove(slot)
            ownership[action] += 1
            if not tensors[slot]['path']:
                if transfer:
                    live.remove(slot)
                    return f'tensor_{slot}'
                copies.append((slot, variable))
                ownership['bend_copy'] += 1
                return variable
            ownership['host_' + action] += 1
            body.append(f'{variable} : F.Storage() <- Tensor.{action}_buffer({slot},{tensors[slot]["count"]}n)')
            # Tensors carry the runtime's worker count, read once in main.
            return f'F.with_workers(F.from_storage({shape_literal(tensors[slot]["shape"])},{variable}),workers)'

        label = labels.get(output, '')
        if kind == 'conv':
            source = read(operation['input'], 'input')
            weights = read(operation['weights'], 'weights')
            bias = read(operation['bias'], 'bias')
            activation = 'Act.silu()' if operation['activation'] else 'Act.identity()'
            lane = '!' if target == 'cuda' else ''
            expression = (
                f'F.conv2d{lane}({source},{weights},{bias},{operation["stride"]},{operation["padding"]},{activation})'
            )
        else:
            if kind == 'concat':
                parts = [read(source, f'part_{part}') for part, source in enumerate(operation['inputs'])]
                sequence = f'F.single({parts[-1]})'
                for part in reversed(parts[:-1]):
                    sequence = f'F.cons({part},{sequence})'
                expression = f'F.concat({sequence},1)'
            elif kind == 'add':
                left = read(operation['left'], 'left')
                right = read(operation['right'], 'right')
                expression = f'F.add({left},{right})'
            elif kind == 'up':
                scale = 2
                operation['scale'] = scale
                expression = f'F.upsample_nearest({read(operation["input"], "input")},{scale})'
            elif kind == 'pool':
                kernel, stride, padding = 5, 1, 2
                operation.update(kernel=kernel, stride=stride, padding=padding)
                expression = f'F.max_pool2d_with({read(operation["input"], "input")},{kernel},{stride},{padding})'
            elif kind == 'decode':
                logits = read(operation['input'], 'input')
                height, width = operation['input_height'], operation['input_width']
                operation['operations'] = DECODE_OPERATIONS
                expression = (
                    f'F.reshape(F.transpose(F.reshape(F.apply_indexed(~{decoders[output]},{DECODE_OPERATIONS},{logits}),[3,85,{height},{width}]),[0,2,3,1]),'
                    f'{shape_literal(tensors[output]["shape"])})'
                )
            else:
                raise ValueError(kind)
        body.append(f'tensor_{output} : F.Tensor() <- F.expect({expression})')
        body.append(f'Tensor.operation_end({output})')
        if remaining_reads[output]:
            live.add(output)
            if label:
                body.append(f'tensor_{output} : F.Tensor() <- snapshot(dump,{output},"{label}",tensor_{output})')
        else:
            # Terminal observations remain real I/O even with dumps disabled.
            body.append(f'output : F.Storage() <- F.expect_storage(tensor_{output})')
            body.append(f'Tensor.save_buffer({output},output,"{label}")')
        assert output not in available, 'Output unexpectedly aliases a live input'
        available.add(output)
        if index + 1 < len(graph['ops']):
            arguments = ['workers', 'dump'] + [f'tensor_{slot}' for slot in sorted(live)]
            body.append(f'operation_{index + 1}({",".join(arguments)})')
        else:
            assert not live, 'Unconsumed graph owners'
            body.append('Tensor.end()')
        definition = [f'def operation_body_{index}({",".join(parameters)}) -> IO(Unit):']
        indentation = '  '
        for slot, variable in copies:
            definition.append(indentation + f'copied(F.copy(tensor_{slot}),tensor_{slot} => {variable} =>')
            indentation += '  '
        definition.append(indentation + 'do IO<Unit>:')
        definition += [indentation + '  ' + statement for statement in body]
        definition[-1] += ')' * len(copies)
        call_arguments = ['workers', 'dump'] + [parameter.split(':')[0] for parameter in parameters[2:]]
        definition += ['', f'def operation_{index}({",".join(parameters)}) -> IO(Unit):',
                       '  do IO<Unit>:', f'    Tensor.operation_begin({output})',
                       f'    operation_body_{index}({",".join(call_arguments)})']
        functions.append(definition + [''])

    assert not any(remaining_reads.values()), 'Graph use accounting is incomplete'
    graph['ownership'] = dict(ownership)
    for definition in reversed(functions):
        lines += definition
    function(
        'forward',
        [f'{name}()' for name in initializers]
        + ['Tensor.begin()']
        + ['operation_0(workers,dump)'],
        '+workers: U32,+dump: Bool',
    )
    lines += [
        'def repeat_forward(count: Nat,+workers: U32,+dump: Bool) -> IO(Unit):',
        '  match count:',
        '    case 0n: IO.pure(Unit,Unit{})',
        '    case 1n+rest:',
        '      do IO<Unit>:',
        '        forward(workers,dump)',
        '        repeat_forward(rest,workers,dump)',
        '',
        'def main() -> IO(Unit):',
        '  do IO<Unit>:',
        '    count : U32 <- Tensor.repetitions()',
        '    workers : U32 <- IO.thread_count()',
        '    dump : Bool <- Tensor.dump_enabled()',
        '    like : F.Storage() <- F.expect_storage(F.full([1],0.0))',
        '    Tensor.buffer_like(like)',
        '    repeat_forward(U32.to_nat(count),workers,dump)',
        '',
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='bus_640')
    parser.add_argument('--target', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument(
        '--threads',
        type=int,
        default=None,
        help='Runtime CPU thread count this build is named for; the program is identical, '
        'because it reads the worker pool size with IO.thread_count',
    )
    args = parser.parse_args()
    if args.threads is not None:
        if args.target != 'cpu':
            parser.error('--threads names a CPU build; do not combine it with --target cuda')
        if args.threads < 1:
            parser.error('--threads must be positive')
    if not re.fullmatch(r'(?:bus|zidane)_[0-9]+', args.case):
        parser.error('case must identify a prepared bus or zidane input')
    meta = json.loads((ROOT / 'out/weights/model.json').read_text())
    reference = json.loads((ROOT / 'out/results' / args.case / 'reference.json').read_text())
    graph = build_graph(meta, reference, args.case)
    output = ROOT / 'out/models'
    output.mkdir(parents=True, exist_ok=True)
    (ROOT / 'out/results' / args.case / 'native').mkdir(exist_ok=True)
    stem = args.case + ('_cuda' if args.target == 'cuda' else '')
    if args.threads is not None and args.threads > 1:
        stem += f'_cpu_{args.threads}t'
    (output / f'{stem}.bend').write_text(emit_model(graph, args.target), encoding='utf-8')
    graph.update(backend='lib/tensor_f32.bend', target=args.target, threads=args.threads or 1)
    (output / f'{stem}.json').write_text(json.dumps(graph, indent=2), encoding='utf-8')
    print(f'{stem}: {len(graph["ops"])} operations; conv2d splits within the runtime worker count')


if __name__ == '__main__':
    main()
