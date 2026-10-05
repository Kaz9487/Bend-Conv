"""Compile programs written against the public F module and compare them with NumPy.

Every case uses only lib/tensor_f32.bend and lib/math_activation.bend. The
cases check what the proofs do not reach: the generated C, scheduling on 1
and 4 threads, Base F32 value semantics and the storage I/O boundary.
"""
from pathlib import Path
import argparse
import json
import os
import re
import shutil
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'out/checks'
REPORT = ROOT / 'out/results/public_api'


def generate():
    cases = []
    helpers = '''import Base
import ../../lib/tensor_f32.bend as F
import ../../lib/math_activation.bend as Act

def same_words(left: List<&2,U32>,right: List<&2,U32>) -> Bool:
  match left right:
    case Nil{} Nil{}: True{}
    case Con{x,xs} Con{y,ys}: U32.is_eq(x,y) && same_words(xs,ys)
    case left right: False{}

def from_bits(word: U32) -> F32:
  U32{bits} = word
  F32{bits}

def to_bits(value: F32) -> U32:
  F32{bits} = value
  U32{bits}

def bit_values(values: List<&2,F32>) -> List<&2,U32>:
  match values:
    case Nil{}: Nil{}
    case Con{value,tail}: to_bits(value) <> bit_values(tail)

def public_values(expected: List<&2,U32>,result: Result<&2,&2,F.Problem(),List<&2,F32>>) -> Bool:
  match result:
    case Done{values}: same_words(expected,bit_values(values))
    case Fail{problem}: False{}

def public_observed(+shape: List<&2,U32>,+expected: List<&2,U32>,observed: F.Tensor() & List<&2,U32>) -> Bool:
  (tensor,actual) = observed
  same_words(shape,actual) && public_values(expected,F.to_list(tensor))

def check_public(+shape: List<&2,U32>,+expected: List<&2,U32>,tensor: F.Tensor()) -> Bool:
  public_observed(shape,expected,F.shape(tensor))

def failure_text(result: Result<&2,&2,F.Problem(),List<&2,F32>>) -> String:
  match result:
    case Done{values}: ""
    case Fail{problem}: F.describe(problem)

def check_failed(message: String,tensor: F.Tensor()) -> Bool:
  String.eq(message,failure_text(F.to_list(tensor)))

def cleared(result: Result<&1,&1,F.Tensor() & F.Problem(),F.Tensor()>) -> Bool:
  match result:
    case Done{tensor}: False{}
    case Fail{(tensor,problem)}: public_values([0,0],F.to_list(tensor))

def from_bit_list(bits: List<&2,U32>) -> List<&2,F32>:
  match bits:
    case Nil{}: Nil{}
    case Con{word,tail}: from_bits(word) <> from_bit_list(tail)

def f32_tensor(+shape: List<&2,U32>,bits: List<&2,U32>) -> F.Tensor():
  F.from_list(shape,from_bit_list(bits))

# Storage leaves the tensor for foreign I/O and comes back under the same shape.
def reloaded(+shape: List<&2,U32>,result: Result<&2,&1,F.Problem(),F.Storage()>) -> F.Tensor():
  match result:
    case Done{storage}: F.from_storage(shape,storage)
    case Fail{problem}: F.zeros([0])

def plus_index(index: U32,value: F32) -> F32:
  (value + U32.to_f32(index) : F32)

def convolution_activation(value: F32) -> F32:
  F32.max((value * 0.5 + 0.25 : F32),0.0)

'''

    def case(name, expression):
        cases.append((name, f'def case_{len(cases)}() -> Bool:\n  {expression}\n'))

    def bits(values):
        return np.asarray(values, dtype=np.float32).view(np.uint32).tolist()

    def tensor(shape, values):
        return f'f32_tensor({list(shape)},{bits(values)})'

    def public(name, expression, expected):
        expected = np.asarray(expected, dtype=np.float32)
        case(name, f'check_public({list(expected.shape)},{bits(expected.ravel())},{expression})')

    def failing(name, expression, message):
        case(name, f'check_failed("{message}",{expression})')

    def ordered_max(x, y):
        return np.where(x < y, y, x)

    def ordered_min(x, y):
        return np.where(x < y, x, y)

    left = np.array([[1.25, -2.5, 0.0, -0.0], [3.0, np.nan, 0.5, 8.0]], dtype=np.float32)
    right = np.array([[2.0, -4.0, -0.0, 0.0], [-3.0, 0.25, np.nan, -2.0]], dtype=np.float32)
    with np.errstate(invalid='ignore', divide='ignore'):
        results = dict(add=left + right, sub=left - right, mul=left * right, div=left / right,
            maximum=ordered_max(left, right), minimum=ordered_min(left, right))
    for operation, expected in results.items():
        public(f'public_{operation}', f'F.{operation}({tensor(left.shape, left.ravel())},{tensor(right.shape, right.ravel())})', expected)
    failing('public_shape_mismatch', f'F.add({tensor([2,4], left.ravel())},{tensor([4,2], right.ravel())})',
        'add: shapes [2, 4] and [4, 2] do not match')
    failing('public_failure_propagates', f'F.relu(F.mul({tensor([2,4], left.ravel())},F.reshape(F.zeros([2,3]),[4,2])))',
        'reshape: shapes [2, 3] and [4, 2] do not match')
    case('public_check_clears', 'cleared(F.check(F.add(F.zeros([2]),F.zeros([3]))))')
    grid = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    public('public_transpose', f'F.transpose({tensor(grid.shape, grid.ravel())},[2,0,1])', grid.transpose(2, 0, 1))
    public('public_slice_step', f'F.slice({tensor(grid.shape, grid.ravel())},2,1,2,2)', grid[:, :, 1:4:2])
    part = -np.arange(6, dtype=np.float32).reshape(2, 3, 1)
    written = grid.copy(); written[:, :, 2:3] = part
    public('public_set_slice', f'F.set_slice({tensor(grid.shape, grid.ravel())},2,2,{tensor(part.shape, part.ravel())})', written)
    a, b, c = grid[:, :1], grid[:, 1:], -grid[:, :2]
    public('public_concat_axis1', f'F.concat(F.cons({tensor(a.shape, a.ravel())},F.cons({tensor(b.shape, b.ravel())},F.single({tensor(c.shape, c.ravel())}))),1)',
        np.concatenate([a, b, c], axis=1))
    public('public_concat_axis0', f'F.concat(F.cons({tensor(grid.shape, grid.ravel())},F.single({tensor(grid.shape, -grid.ravel())})),0)',
        np.concatenate([grid, -grid], axis=0))
    failing('public_concat_mismatch', f'F.concat(F.cons({tensor(a.shape, a.ravel())},F.single({tensor([2,2,3], np.zeros(12))})),1)',
        'concat: invalid argument for shape [2, 1, 4]')
    public('public_upsample', f'F.upsample_nearest({tensor(grid.shape, grid.ravel())},2)', grid.repeat(2, axis=1).repeat(2, axis=2))
    pooled = np.array([[1, 5, 2, np.nan, 3], [-np.inf, 4, 9, 0, 1], [7, 0, 6, 2, -1], [3, 3, 8, -5, 2]], dtype=np.float32)[None]
    for kernel, stride, padding in [(3, 2, 1), (2, 1, 0), (5, 1, 2)]:
        padded = np.pad(pooled, ((0, 0), (padding, padding), (padding, padding)), constant_values=-np.inf)
        rows = (padded.shape[1] - kernel) // stride + 1
        columns = (padded.shape[2] - kernel) // stride + 1
        expected = np.full((1, rows, columns), -np.inf, dtype=np.float32)
        for ky in range(kernel):
            for kx in range(kernel):
                window = padded[:, ky:ky + stride * (rows - 1) + 1:stride, kx:kx + stride * (columns - 1) + 1:stride]
                expected = ordered_max(expected, window)
        public(f'public_max_pool_{kernel}_{stride}_{padding}', f'F.max_pool2d_with({tensor(pooled.shape, pooled.ravel())},{kernel},{stride},{padding})', expected)
    public('public_apply_indexed', f'F.apply_indexed(~plus_index,Act.unknown_operations(),{tensor(grid.shape, grid.ravel())})', grid + np.arange(24, dtype=np.float32).reshape(grid.shape))
    public('public_scalar', f'F.mul_scalar(F.add_scalar({tensor(grid.shape, grid.ravel())},0.5),F32.neg(2.0))', (grid + np.float32(0.5)) * np.float32(-2.0))
    total = np.float32(0)
    for value in grid.ravel():
        total = np.float32(total + value)
    public('public_sum', f'F.sum({tensor(grid.shape, grid.ravel())})', total)
    public('public_mean', f'F.mean({tensor(grid.shape, grid.ravel())})', total / np.float32(grid.size))
    signed = grid - np.float32(7.5)
    public('public_max', f'F.max({tensor(signed.shape, signed.ravel())})', signed.max())
    public('public_min', f'F.min({tensor(signed.shape, signed.ravel())})', signed.min())
    public('public_div_scalar', f'F.div_scalar({tensor(grid.shape, grid.ravel())},3.0)', grid / np.float32(3.0))
    public('public_arange', 'F.reshape(F.arange(2.5,24),[2,3,4])', (np.arange(24, dtype=np.float32) + np.float32(2.5)).reshape(2, 3, 4))
    public('public_arange_empty', 'F.arange(1.0,0)', np.zeros((0,), dtype=np.float32))
    public('public_ones', 'F.ones([2,3])', np.ones((2, 3), dtype=np.float32))
    public('public_to_list_or', 'F.from_list([2],F.to_list_or(F.add_scalar(F.ones([2]),1.0),[]))', np.full((2,), 2, dtype=np.float32))
    public('public_to_list_or_failed', 'F.from_list([1],F.to_list_or(F.add(F.zeros([2]),F.zeros([3])),[7.0]))', np.full((1,), 7, dtype=np.float32))
    public('public_item_or', 'F.full([],F.item_or(F.sum(F.ones([3])),0.0))', np.float32(3))
    public('public_item_or_failed', 'F.full([],F.item_or(F.sum(F.add(F.zeros([2]),F.zeros([3]))),9.0))', np.float32(9))
    even = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
    public('public_max_pool_default', 'F.max_pool2d(F.reshape(F.arange(0.0,32),[2,4,4]),2)', even.reshape(2, 2, 2, 2, 2).max(axis=(2, 4)))
    failing('public_pool_name', f'F.max_pool2d_with({tensor(grid.shape, grid.ravel())},0,1,0)', 'max_pool2d: invalid argument for shape [2, 3, 4]')
    public('public_storage_roundtrip', f'reloaded({list(grid.shape)},F.into_storage(F.add_scalar({tensor(grid.shape, grid.ravel())},1.0)))', grid + np.float32(1.0))
    public('public_reshape', f'F.reshape({tensor(grid.shape, grid.ravel())},[4,6])', grid.reshape(4, 6))
    public('public_copy', f'F.then(F.copy({tensor(grid.shape, grid.ravel())}),kept => copied => F.sub(kept,copied))', np.zeros_like(grid))
    public('public_then_shape', f'F.then_shape(F.shape({tensor(grid.shape, grid.ravel())}),owner => extents => F.add(owner,F.full(extents,0.5)))', grid + np.float32(0.5))

    # Broadcasting follows NumPy: the result reuses the left storage when only
    # the right operand is stretched, and is new storage otherwise.
    values = ((np.arange(24, dtype=np.float32) % 11) - np.float32(4)) / np.float32(7)
    cube = values.reshape(2, 3, 4)
    row = np.array([0.5, -1.25, 3.0, 1 / 3], dtype=np.float32)
    column = np.array([[2.0], [-0.75], [1 / 9]], dtype=np.float32)
    plane = (np.arange(8, dtype=np.float32) / np.float32(3)).reshape(2, 1, 4)
    for name, operation, x, y, expected in [
        ('right_row', 'add', cube, row, cube + row),
        ('right_column', 'mul', cube, column, cube * column),
        ('right_scalar', 'sub', cube, np.float32(0.3), cube - np.float32(0.3)),
        ('left_row', 'sub', row, cube, row - cube),
        ('left_column', 'maximum', column, cube, ordered_max(np.broadcast_to(column, cube.shape), cube)),
        ('both', 'add', column, row, column + row),
        ('both_rank', 'mul', plane, column, plane * column),
        ('empty', 'add', np.zeros((0, 3), dtype=np.float32), row[None, :3], np.zeros((0, 3), dtype=np.float32)),
    ]:
        public('public_broadcast_' + name, f'F.{operation}({tensor(np.shape(x), np.ravel(x))},{tensor(np.shape(y), np.ravel(y))})', expected)
    failing('public_broadcast_mismatch', f'F.add({tensor(cube.shape, cube.ravel())},{tensor([3], row[:3])})', 'add: shapes [2, 3, 4] and [3] do not match')
    public('public_broadcast_to', f'F.broadcast_to({tensor(plane.shape, plane.ravel())},[2,3,4])', np.broadcast_to(plane, (2, 3, 4)))
    public('public_broadcast_to_rank', f'F.broadcast_to({tensor(row.shape, row)},[3,2,4])', np.broadcast_to(row, (3, 2, 4)))
    failing('public_broadcast_to_mismatch', f'F.broadcast_to({tensor([3], row[:3])},[2,4])', 'broadcast_to: shapes [3] and [2, 4] do not match')

    # An axis reduction takes the elements of the axis in increasing index.
    def fold_axis(x, axis, operation, initial):
        accumulator = np.full(np.delete(x.shape, axis), initial, dtype=np.float32)
        for index in range(x.shape[axis]):
            accumulator = operation(accumulator, np.take(x, index, axis=axis)).astype(np.float32)
        return accumulator

    special = cube.copy()
    special[0, 1, 2], special[1, 0, 0], special[1, 2, 3] = np.nan, -np.inf, np.inf
    for axis in range(3):
        call = tensor(cube.shape, cube.ravel())
        total = fold_axis(cube, axis, np.add, 0.0)
        public(f'public_sum_axis_{axis}', f'F.sum_axis({call},{axis})', total)
        public(f'public_mean_axis_{axis}', f'F.mean_axis({call},{axis})', total / np.float32(cube.shape[axis]))
        call = tensor(special.shape, special.ravel())
        with np.errstate(invalid='ignore'):
            public(f'public_max_axis_{axis}', f'F.max_axis({call},{axis})', fold_axis(special, axis, ordered_max, -np.inf))
            public(f'public_min_axis_{axis}', f'F.min_axis({call},{axis})', fold_axis(special, axis, ordered_min, np.inf))
    public('public_sum_axis_empty', 'F.sum_axis(F.zeros([2,0,3]),1)', np.zeros((2, 3), dtype=np.float32))
    failing('public_sum_axis_range', f'F.sum_axis({tensor(cube.shape, cube.ravel())},3)', 'sum_axis: invalid argument for shape [2, 3, 4]')

    widths = ((0, 0), (1, 2), (2, 0))
    before, after = [list(side) for side in zip(*widths)]
    call = tensor(cube.shape, cube.ravel())
    public('public_pad', f'F.pad({call},{before},{after},F32.neg(1.5))', np.pad(cube, widths, constant_values=np.float32(-1.5)))
    public('public_pad_edge', f'F.pad_edge({call},{before},{after})', np.pad(cube, widths, mode='edge'))
    public('public_pad_edge_all', f'F.pad_edge({call},[1,2,3],[2,0,1])', np.pad(cube, ((1, 2), (2, 0), (3, 1)), mode='edge'))
    failing('public_pad_rank', f'F.pad({call},[1,1],[1,1],0.0)', 'pad: invalid argument for shape [2, 3, 4]')
    failing('public_pad_edge_empty', 'F.pad_edge(F.zeros([2,0]),[0,1],[0,1])', 'pad_edge: invalid argument for shape [2, 0]')

    # A flow of the new operations: each channel of a CHW map is centred on
    # its mean, scaled and shifted, then pooled with replicated borders.
    feature = (((np.arange(90, dtype=np.float32) * np.float32(7)) % 23) - np.float32(9)).reshape(3, 5, 6) / np.float32(5)
    gain = np.array([1.5, -0.25, 2 / 3], dtype=np.float32).reshape(3, 1, 1)
    shift = np.array([0.125, 1.0, -0.5], dtype=np.float32).reshape(3, 1, 1)
    mean = fold_axis(fold_axis(feature, 2, np.add, 0.0) / np.float32(6), 1, np.add, 0.0) / np.float32(5)
    normalized = (feature - mean.reshape(3, 1, 1)) * gain + shift
    bordered = np.pad(normalized, ((0, 0), (1, 1), (1, 1)), mode='edge')
    pooled_flow = np.full(normalized.shape, -np.inf, dtype=np.float32)
    for ky in range(3):
        for kx in range(3):
            pooled_flow = ordered_max(pooled_flow, bordered[:, ky:ky + 5, kx:kx + 6])
    flow = ('F.then(F.copy(' + tensor(feature.shape, feature.ravel()) + '),kept => copied => '
        'F.max_pool2d_with(F.pad_edge(F.add(F.mul(F.sub(kept,F.reshape(F.mean_axis(F.mean_axis(copied,2),1),[3,1,1])),'
        + tensor(gain.shape, gain.ravel()) + '),' + tensor(shift.shape, shift.ravel()) + '),[0,1,1],[0,1,1]),3,1,0))')
    public('public_flow_channel_normalize', flow, pooled_flow)

    # Each product and addition rounds separately, in (C, KH, KW) order.
    # np.matmul/einsum could reorder or contract this reduction.
    def convolution_oracle(x, weights, bias, stride, padding, custom):
        batched = x.ndim == 4
        source = x if batched else x[None]
        n, channels, height, width = source.shape
        rows, _, kernel, _ = weights.shape
        oh = (height + 2 * padding - kernel) // stride + 1
        ow = (width + 2 * padding - kernel) // stride + 1
        output = np.zeros((n, rows, oh, ow), dtype=np.float32)
        for image in range(n):
            for row in range(rows):
                for y in range(oh):
                    for z in range(ow):
                        accumulator = np.float32(0.0)
                        for channel in range(channels):
                            for ky in range(kernel):
                                for kx in range(kernel):
                                    iy, ix = y * stride + ky - padding, z * stride + kx - padding
                                    value = source[image, channel, iy, ix] if 0 <= iy < height and 0 <= ix < width else np.float32(0.0)
                                    product = np.float32(value * weights[row, channel, ky, kx])
                                    accumulator = np.float32(accumulator + product)
                        value = np.float32(accumulator + bias[row])
                        if custom:
                            value = np.float32(np.float32(value * np.float32(0.5)) + np.float32(0.25))
                            value = ordered_max(value, np.float32(0.0))
                        output[image, row, y, z] = value
        return output if batched else output[0]

    for name, dims, rows, kernel, stride, padding, custom in [
        ('chw_tail', (2, 3, 5), 7, 1, 1, 0, False),
        ('chw_aligned', (2, 3, 8), 6, 3, 1, 1, False),
        ('batch_custom', (2, 2, 5, 7), 7, 3, 2, 1, True),
        ('batch_identity', (2, 1, 2, 5), 1, 1, 1, 0, False),
        ('empty_batch', (0, 2, 3, 5), 7, 1, 1, 0, False),
    ]:
        x = ((np.arange(np.prod(dims), dtype=np.float32) % 17) - np.float32(8)) / np.float32(7)
        x = x.reshape(dims)
        weight_shape = (rows, dims[-3], kernel, kernel)
        weights = ((np.arange(np.prod(weight_shape), dtype=np.float32) % 13) - np.float32(6)) / np.float32(11)
        weights = weights.reshape(weight_shape)
        bias = np.arange(rows, dtype=np.float32) / np.float32(19)
        call, last = ('F.conv2d_with(~convolution_activation,4.0,', '') if custom else ('F.conv2d(', ',Act.identity()')
        expression = f'{call}{tensor(dims, x.ravel())},{tensor(weight_shape, weights.ravel())},{tensor([rows], bias)},{stride},{padding}{last})'
        public('public_conv2d_' + name, expression, convolution_oracle(x, weights, bias, stride, padding, custom))
    for name, expression, shape, reason in [
        ('rank', 'F.conv2d(F.zeros([2,3]),F.zeros([1,2,1,1]),F.zeros([1]),1,0,Act.identity())', [2,3], 'invalid argument for shape {}'),
        ('machine_limit', 'F.conv2d(F.zeros([2,3,5]),F.zeros([1,2,1,1]),F.zeros([1]),65536,0,Act.identity())', [2,3,5], 'shape {} is invalid or exceeds the machine capacity'),
    ]:
        failing('public_conv2d_reject_' + name, expression, 'conv2d: ' + reason.format(shape))
    failing('public_conv2d_sticky', 'F.conv2d(F.reshape(F.zeros([2]),[3]),F.zeros([1,1,1,1]),F.zeros([1]),1,0,Act.identity())',
        'reshape: shapes [2] and [3] do not match')
    source = helpers + '\n'.join(body for _, body in cases)
    source += '\ndef main() -> List<&2,Bool>:\n  [' + ','.join(f'case_{i}()' for i in range(len(cases))) + ']\n'
    OUT.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    (OUT/'public_api.bend').write_text(source)
    (REPORT/'cases.json').write_text(json.dumps([name for name,_ in cases],indent=2))
    return len(cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-only',action='store_true')
    args = parser.parse_args()
    count = generate()
    (REPORT/'status.json').unlink(missing_ok=True)
    if args.generate_only:
        print(f'Generated {count} NumPy-backed runtime cases.')
        return
    env = dict(os.environ, BEND_NO_TELEMETRY='1')
    c = ROOT/'out/build/integration/public_api.c'
    c.parent.mkdir(parents=True,exist_ok=True)
    build = subprocess.run(['node','scripts/bend_launcher.mjs',str(OUT/'public_api.bend'),'-o',str(c)],cwd=ROOT,env=env,capture_output=True,text=True)
    (REPORT/'build.log').write_text(build.stdout+build.stderr)
    build.check_returncode()
    cc = os.environ.get('CC') or shutil.which('clang-19') or shutil.which('clang')
    if not cc:
        raise SystemExit('Native public API checks need clang on PATH or CC.')
    subprocess.run([cc,'-O3','-ffp-contract=off','-std=c11',str(c),'-lpthread','-lm','-o',str(c.with_suffix(''))],check=True)
    for threads in [1,4]:
        result = subprocess.run([str(c.with_suffix('')),'--threads',str(threads),'--gpu','off'],capture_output=True,text=True,check=True)
        (REPORT/f'runtime-{threads}.log').write_text(result.stdout+result.stderr)
        flags = re.findall(r'(True|False)\{\}',result.stdout)
        assert len(flags)==count, result.stdout
        names = json.loads((REPORT/'cases.json').read_text())
        failures = [name for name,flag in zip(names,flags) if flag!='True']
        assert not failures, failures
    status = dict(cases=count,threads=[1,4],all_passed=True,numpy_version=np.__version__,
        build=json.loads(Path(str(c)+'.build.json').read_text()))
    (REPORT/'status.json').write_text(json.dumps(status,indent=2))
    print(f'{count} public API cases passed on 1 and 4 threads.')


if __name__ == '__main__':
    main()
