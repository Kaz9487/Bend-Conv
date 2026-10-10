"""Emit the YOLO graph as a Bend program with a host timer around every operation.

The measurement tools use it; the model itself is examples/yolov5/yolov5n.bend.
Every operation is a public tensor call (lib/tensor_f32.bend).
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


RUNTIME = '''# The model is a list of operations run by one loop. A tensor between operations
# waits in the environment under its number; its last reader takes it.
type Read is Data:
  Read{slot: U32,take: Bool}

type Work is Data:
  Conv{input: Read,host: Bool,weights: U32,bias: U32,channels: U32,height: U32,width: U32,outputs: U32,kernel: U32,stride: U32,padding: U32,silu: Bool}
  Concat{first: Read,rest: List<&2,Read>}
  Add{left: Read,right: Read}
  Up{input: Read}
  Pool{input: Read}
  Decode{input: Read,head: U32}

type Op is Data:
  Op{output: U32,keep: Bool,labelled: Bool,work: Work}

type Env is Type:
  Empty{}
  Slot{id: U32,tensor: F.Tensor(),rest: Env}

def Loaded() -> Type:
  F.Tensor() & Env

def kept(+id: U32,pair: F.Tensor() & F.Tensor(),rest: Env) -> Loaded():
  (left,right) = pair
  (right,Slot{id,left,rest})

def taken(take: Bool,+id: U32,tensor: F.Tensor(),rest: Env) -> Loaded():
  match take:
    case True{}: (tensor,rest)
    case False{}: kept(id,F.copy(tensor),rest)

def skipped(+id: U32,tensor: F.Tensor(),pair: Loaded()) -> Loaded():
  (result,rest) = pair
  (result,Slot{id,tensor,rest})

# The generator reads only tensors that are waiting, so the empty cases are not reached.
def scan(rest: Env,same: Bool,+id: U32,tensor: F.Tensor(),+slot: U32,+take: Bool) -> Loaded():
  match rest same:
    case Empty{} True{}: taken(take,id,tensor,Empty{})
    case Slot{+next,other,more} True{}: taken(take,id,tensor,Slot{next,other,more})
    case Empty{} False{}: (F.full([1],0.0),Slot{id,tensor,Empty{}})
    case Slot{+next,other,more} False{}: skipped(id,tensor,scan(more,U32.is_eq(next,slot),next,other,slot,take))

def fetch(env: Env,+slot: U32,+take: Bool) -> Loaded():
  match env:
    case Empty{}: (F.full([1],0.0),Empty{})
    case Slot{+id,tensor,rest}: scan(rest,U32.is_eq(id,slot),id,tensor,slot,take)

def read(source: Read,env: Env) -> Loaded():
  Read{+slot,+take} = source
  fetch(env,slot,take)

def hosted(take: Bool,+slot: U32,+channels: U32,+height: U32,+width: U32,env: Env,+workers: U32) -> IO(Loaded()):
  match take:
    case True{}:
      do IO<Loaded()>:
        input : F.Storage() <- Tensor.take_buffer(slot,U32.to_nat((channels * height * width : U32)))
        IO.pure(Loaded(),(F.with_workers(F.from_storage([1,channels,height,width],input),workers),env))
    case False{}:
      do IO<Loaded()>:
        input : F.Storage() <- Tensor.get_buffer(slot,U32.to_nat((channels * height * width : U32)))
        IO.pure(Loaded(),(F.with_workers(F.from_storage([1,channels,height,width],input),workers),env))

def hosted_read(input: Read,+channels: U32,+height: U32,+width: U32,env: Env,+workers: U32) -> IO(Loaded()):
  Read{+slot,take} = input
  hosted(take,slot,channels,height,width,env,workers)

def source(input: Read,host: Bool,+channels: U32,+height: U32,+width: U32,env: Env,+workers: U32) -> IO(Loaded()):
  match host:
    case True{}: hosted_read(input,channels,height,width,env,workers)
    case False{}: IO.pure(Loaded(),read(input,env))

def convolved_silu(loaded: Loaded(),weights: F.Tensor(),bias: F.Tensor(),stride: U32,padding: U32) -> Loaded():
  (input,env) = loaded
  (CONV2D(input,weights,bias,stride,padding,Act.silu()),env)

def convolved_identity(loaded: Loaded(),weights: F.Tensor(),bias: F.Tensor(),stride: U32,padding: U32) -> Loaded():
  (input,env) = loaded
  (CONV2D(input,weights,bias,stride,padding,Act.identity()),env)

def convolved(silu: Bool,loaded: Loaded(),weights: F.Tensor(),bias: F.Tensor(),stride: U32,padding: U32) -> Loaded():
  match silu:
    case True{}: convolved_silu(loaded,weights,bias,stride,padding)
    case False{}: convolved_identity(loaded,weights,bias,stride,padding)

def joined(tensor: F.Tensor(),pair: F.Parts() & Env) -> F.Parts() & Env:
  (parts,env) = pair
  (F.cons(tensor,parts),env)

def parts_after(rest: List<&2,Read>,loaded: Loaded()) -> F.Parts() & Env:
  match rest:
    case []:
      (tensor,env) = loaded
      (F.single(tensor),env)
    case next <> more:
      (tensor,env) = loaded
      joined(tensor,parts_after(more,read(next,env)))

def concatenated(pair: F.Parts() & Env) -> Loaded():
  (parts,env) = pair
  (F.concat(parts,1),env)

def added_right(left: F.Tensor(),loaded: Loaded()) -> Loaded():
  (right,env) = loaded
  (F.add(left,right),env)

def added(right: Read,loaded: Loaded()) -> Loaded():
  (left,env) = loaded
  added_right(left,read(right,env))

def upsampled(loaded: Loaded()) -> Loaded():
  (input,env) = loaded
  (F.upsample_nearest(input,2),env)

def pooled(loaded: Loaded()) -> Loaded():
  (input,env) = loaded
  (F.max_pool2d_with(input,5,1,2),env)

def decoded(head: U32,loaded: Loaded()) -> Loaded():
  (input,env) = loaded
  (decode_head(head,input),env)

def compute(work: Work,env: Env,+workers: U32) -> IO(Loaded()):
  match work:
    case Conv{input,host,+weights,+bias,+channels,+height,+width,+outputs,+kernel,stride,padding,silu}:
      do IO<Loaded()>:
        loaded : Loaded() <- source(input,host,channels,height,width,env,workers)
        w : F.Storage() <- Tensor.take_buffer(weights,U32.to_nat((outputs * channels * kernel * kernel : U32)))
        b : F.Storage() <- Tensor.take_buffer(bias,U32.to_nat(outputs))
        # Tensors carry the runtime's worker count, read once in main.
        IO.pure(Loaded(),convolved(silu,loaded,F.with_workers(F.from_storage([outputs,channels,kernel,kernel],w),workers),
          F.with_workers(F.from_storage([outputs],b),workers),stride,padding))
    case Concat{first,rest}: IO.pure(Loaded(),concatenated(parts_after(rest,read(first,env))))
    case Add{left,right}: IO.pure(Loaded(),added(right,read(left,env)))
    case Up{input}: IO.pure(Loaded(),upsampled(read(input,env)))
    case Pool{input}: IO.pure(Loaded(),pooled(read(input,env)))
    case Decode{input,head}: IO.pure(Loaded(),decoded(head,read(input,env)))

def snapshot_pair(+id: U32,pair: F.Tensor() & F.Tensor()) -> IO(F.Tensor()):
  (tensor,snapshot) = pair
  do IO<F.Tensor()>:
    storage : F.Storage() <- F.expect_storage(snapshot)
    Tensor.save_buffer(id,storage,label(id))
    IO.pure(F.Tensor(),tensor)

def snapshot(enabled: Bool,+id: U32,tensor: F.Tensor()) -> IO(F.Tensor()):
  match enabled:
    case False{}: IO.pure(F.Tensor(),tensor)
    case True{}: snapshot_pair(id,F.copy(tensor))

# A tensor with later readers waits in the environment; a terminal one is
# real I/O even with dumps disabled.
def settle(keep: Bool,snapshots: Bool,+output: U32,tensor: F.Tensor(),env: Env) -> IO(Env):
  match keep:
    case True{}:
      do IO<Env>:
        waiting : F.Tensor() <- snapshot(snapshots,output,tensor)
        IO.pure(Env,Slot{output,waiting,env})
    case False{}:
      do IO<Env>:
        storage : F.Storage() <- F.expect_storage(tensor)
        Tensor.save_buffer(output,storage,label(output))
        IO.pure(Env,env)

def finish(keep: Bool,snapshots: Bool,+output: U32,loaded: Loaded()) -> IO(Env):
  (result,env) = loaded
  do IO<Env>:
    tensor : F.Tensor() <- F.expect(result)
    Tensor.operation_end(output)
    settle(keep,snapshots,output,tensor,env)

def step(op: Op,env: Env,+workers: U32,+dump: Bool) -> IO(Env):
  Op{+output,keep,labelled,work} = op
  do IO<Env>:
    Tensor.operation_begin(output)
    loaded : Loaded() <- compute(work,env,workers)
    finish(keep,dump && labelled,output,loaded)

def release(env: Env) -> IO(Unit):
  match env:
    case Empty{}: Tensor.end()
    case Slot{+id,tensor,rest}:
      do IO<Unit>:
        storage : F.Storage() <- F.expect_storage(tensor)
        Tensor.save_buffer(id,storage,label(id))
        release(rest)

def run(operations: List<&2,Op>,env: Env,+workers: U32,+dump: Bool) -> IO(Unit):
  match operations:
    case []: release(env)
    case op <> rest:
      do IO<Unit>:
        next : Env <- step(op,env,workers,dump)
        run(rest,next,workers,dump)

'''


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
            '  import "../../benchmarks/model/tensor_io.c"',
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

    def boolean(value):
        return 'True{}' if value else 'False{}'

    def read(slot):
        """One use of a tensor: its last use takes the owner, an earlier one copies it."""
        assert slot in available and remaining_reads[slot] > 0, 'Read of unavailable tensor'
        remaining_reads[slot] -= 1
        transfer = remaining_reads[slot] == 0
        action = 'take' if transfer else 'get'
        if transfer:
            available.remove(slot)
        ownership[action] += 1
        if tensors[slot]['path']:
            ownership['host_' + action] += 1
        elif transfer:
            live.remove(slot)
        else:
            ownership['bend_copy'] += 1
        return f'Read{{{slot},{boolean(transfer)}}}'

    def host(slot):
        assert tensors[slot]['path'] and remaining_reads[slot] == 1, 'Weights and biases are host tensors read once'
        read(slot)
        return slot

    records = []
    for operation in graph['ops']:
        kind, output = operation['kind'], operation['output']
        assert kind == 'conv' or not any(tensors[slot]['path'] for slot in input_slots(operation)), 'Only convolutions read host tensors'
        if kind == 'conv':
            source = operation['input']
            channels, outputs, kernel = operation['input_channels'], operation['output_channels'], operation['kernel_size']
            assert tensors[source]['shape'] == [1, channels, operation['input_height'], operation['input_width']]
            assert tensors[operation['weights']]['shape'] == [outputs, channels, kernel, kernel]
            hosted = boolean(tensors[source]['path'])
            work = (
                f'Conv{{{read(source)},{hosted},{host(operation["weights"])},{host(operation["bias"])},'
                f'{channels},{operation["input_height"]},{operation["input_width"]},{outputs},'
                f'{kernel},{operation["stride"]},{operation["padding"]},{boolean(operation["activation"])}}}'
            )
        elif kind == 'concat':
            parts = [read(source) for source in operation['inputs']]
            work = f'Concat{{{parts[0]},[{",".join(parts[1:])}]}}'
        elif kind == 'add':
            left = read(operation['left'])
            work = f'Add{{{left},{read(operation["right"])}}}'
        elif kind == 'up':
            operation['scale'] = 2
            work = f'Up{{{read(operation["input"])}}}'
        elif kind == 'pool':
            operation.update(kernel=5, stride=1, padding=2)
            work = f'Pool{{{read(operation["input"])}}}'
        elif kind == 'decode':
            operation['operations'] = DECODE_OPERATIONS
            work = f'Decode{{{read(operation["input"])},{decoders[output][len("decode_"):]}}}'
        else:
            raise ValueError(kind)
        keep = remaining_reads[output] > 0
        if keep:
            live.add(output)
        assert output not in available, 'Output unexpectedly aliases a live input'
        available.add(output)
        records.append(f'Op{{{output},{boolean(keep)},{boolean(output in labels)},{work}}}')
    assert not live, 'Unconsumed graph owners'
    assert not any(remaining_reads.values()), 'Graph use accounting is incomplete'
    graph['ownership'] = dict(ownership)

    heads = sorted(decoders.items(), key=lambda item: item[1])
    for output, name in heads:
        operation = next(o for o in graph['ops'] if o['output'] == output)
        height, width = operation['input_height'], operation['input_width']
        lines += [
            f'def {name}_head(logits: F.Tensor()) -> F.Tensor():',
            f'  F.reshape(F.transpose(F.reshape(F.apply_indexed(~{name},{DECODE_OPERATIONS},logits),[3,85,{height},{width}]),[0,2,3,1]),'
            f'{shape_literal(tensors[output]["shape"])})',
            '',
        ]
    lines += ['def decode_head(head: U32,logits: F.Tensor()) -> F.Tensor():', '  match head:']
    lines += [f'    case {name[len("decode_"):]}: {name}_head(logits)' for _, name in heads[:-1]]
    lines += [f'    case _: {heads[-1][1]}_head(logits)', '']
    lines += ['def label(id: U32) -> String:', '  match id:']
    lines += [f'    case {index}: "{label}"' for index, label in sorted(labels.items())]
    lines += ['    case _: ""', '']
    lines += RUNTIME.replace('CONV2D', 'F.conv2d!' if target == 'cuda' else 'F.conv2d').splitlines()
    lines += ['def operations() -> List<&2,Op>:', '  [' + ',\n    '.join(records) + ']', '']
    function(
        'forward',
        [f'{name}()' for name in initializers]
        + ['Tensor.begin()']
        + ['run(operations(),Empty{},workers,dump)'],
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
