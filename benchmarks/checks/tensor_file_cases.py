"""Check tensor file writes against NumPy at chunk boundaries and on failures."""

import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'out/results/tensor_file'
BUILD = ROOT / 'out/build/tensor_file'
SOURCE = ROOT / 'out/checks'

HELPERS = """import Base
import ../../lib/tensor_f32.bend as F
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_band.bend as Bands

def metadata(pair: F.Tensor() & U32) -> IO(Unit):
  (tensor,workers) = pair
  do IO<Unit>:
    checked : F.Tensor() <- F.expect(tensor)
    IO.print(U32.show(workers))

def round_trip(+name: String,tensor: F.Tensor()) -> IO(Unit):
  do IO<Unit>:
    raw : F.Tensor() <- F.save_raw("out/results/tensor_file/" ++ name ++ ".bin",F.with_workers(tensor,3))
    npy : F.Tensor() <- F.save_npy("out/results/tensor_file/" ++ name ++ ".npy",raw)
    returned : F.Tensor() <- F.save_raw("out/results/tensor_file/" ++ name ++ "_returned.bin",npy)
    metadata(F.workers(returned))

def dense(+name: String,+shape: List<&2,U32>) -> IO(Unit):
  do IO<Unit>:
    tensor : F.Tensor() <- F.load_raw("out/results/tensor_file/" ++ name ++ "_input.bin",shape)
    round_trip(name,tensor)

def partitioned(result: Result<&2,&1,F.Problem(),F.Storage()>) -> F.Tensor():
  match result:
    case Fail{problem}: F.zeros([0])
    case Done{buffer}:
      Tensors.Banded{Bands.split(~F32,1n,Bands.Geometry{2,3,173,17},0.0,buffer),Bands.Geometry{2,3,173,17},[2,3,173,17],0.0,Tensors.Usable{},3}

def banded() -> IO(Unit):
  do IO<Unit>:
    tensor : F.Tensor() <- F.load_raw("out/results/tensor_file/banded_input.bin",[2,3,173,17])
    round_trip("banded",partitioned(F.into_storage(tensor)))

def problem(result: Result<&1,&1,F.Tensor() & F.Problem(),F.Tensor()>) -> String:
  match result:
    case Done{tensor}: "unexpected success"
    case Fail{(tensor,problem)}: F.describe(problem)

def save_failures() -> IO(Unit):
  do IO<Unit>:
    raw : F.Tensor() <- F.save_raw("out/results/tensor_file/untouched.bin",F.add(F.zeros([2]),F.zeros([3])))
    IO.print(problem(F.check(raw)))
    npy : F.Tensor() <- F.save_npy("out/results/tensor_file/untouched.npy",F.add(F.zeros([2]),F.zeros([3])))
    IO.print(problem(F.check(npy)))
    raw : F.Tensor() <- F.save_raw("out/results/tensor_file/missing/raw.bin",F.zeros([1]))
    IO.print(problem(F.check(raw)))
    npy : F.Tensor() <- F.save_npy("out/results/tensor_file/missing/array.npy",F.zeros([1]))
    IO.print(problem(F.check(npy)))

def weights() -> IO(Unit):
  do IO<Unit>:
    store : F.Weights() <- F.load_weights("out/results/tensor_file/weights_input")
    saved : F.Weights() <- F.save_weights("out/results/tensor_file/weights_output",store)
    again : F.Weights() <- F.save_weights("out/results/tensor_file/weights_returned",saved)
    IO.pure(Unit,Unit{})
"""


def build(cc, name, text):
    source = SOURCE / f'{name}.bend'
    source.write_text(text)
    subprocess.run(
        [sys.executable, '-B', 'scripts/toolchain_rules/lint.py', str(source)],
        cwd=ROOT,
        check=True,
    )
    c = BUILD / f'{name}.c'
    result = subprocess.run(
        ['node', 'scripts/bend_launcher.mjs', str(source), '-o', str(c)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    (OUT / f'{name}_build.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        print('\n'.join(line[:300] for line in (result.stdout + result.stderr).splitlines()))
    result.check_returncode()
    binary = c.with_suffix('')
    subprocess.run(
        [
            cc,
            '-O3',
            '-ffp-contract=off',
            '-std=c11',
            str(c),
            '-lpthread',
            '-lm',
            '-o',
            str(binary),
        ],
        cwd=ROOT,
        check=True,
    )
    return binary


def run(binary, **kwargs):
    result = subprocess.run(
        [str(binary), '--threads', '1', '--gpu', 'off'],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout.strip().splitlines()


def main():
    cc = os.environ.get('CC') or shutil.which('clang-19') or shutil.which('clang')
    if not cc:
        raise SystemExit('Tensor file checks need clang on PATH or CC.')
    for directory in [OUT, BUILD, SOURCE]:
        directory.mkdir(parents=True, exist_ok=True)
    (OUT / 'status.json').unlink(missing_ok=True)
    chunk = int(
        re.search(r'def chunk\(\) -> U32:\s+(\d+)', (ROOT / 'lib/tensor_file.bend').read_text())[1]
    )
    shapes = {
        'empty': [2, 0, 3],
        'scalar': [],
        'below': [chunk - 1],
        'exact': [chunk],
        'tail': [chunk + 1],
        'multiple': [2 * chunk],
        'remainder': [2 * chunk + 37],
        'banded': [2, 3, 173, 17],
    }
    # Distinct payloads include both zeros, subnormals, infinities and NaNs.
    patterns = np.array(
        [
            0,
            0x80000000,
            1,
            0x807FFFFF,
            0x3F800000,
            0xC0200000,
            0x7F800000,
            0xFF800000,
            0x7FC12345,
            0xFFA54321,
        ],
        dtype=np.uint32,
    )
    expected = {}
    for name, shape in shapes.items():
        size = int(np.prod(shape, dtype=np.int64))
        words = np.arange(size, dtype=np.uint32) * np.uint32(2654435761)
        positions = np.arange(size) % 19 < len(patterns)
        words[positions] = patterns[(np.arange(size) % 19)[positions]]
        expected[name] = words.reshape(shape)
        words.astype('<u4').tofile(OUT / f'{name}_input.bin')
    for suffix in ['bin', 'npy']:
        (OUT / f'untouched.{suffix}').write_bytes(b'must remain untouched')
    if (OUT / 'missing').exists():
        raise SystemExit(
            'Remove the unexpected out/results/tensor_file/missing directory before testing.'
        )
    for folder in ['weights_input', 'weights_output', 'weights_returned']:
        (OUT / folder).mkdir(exist_ok=True)
    (OUT / 'weights_input/names.txt').write_text('remainder\nempty\n')
    for name in ['remainder', 'empty']:
        np.save(
            OUT / f'weights_input/{name}.npy', expected[name].view(np.float32), allow_pickle=False
        )
    calls = [f'    dense("{name}",{shape})' for name, shape in shapes.items() if name != 'banded']
    program = (
        HELPERS
        + '\ndef main() -> IO(Unit):\n  do IO<Unit>:\n'
        + '\n'.join(calls)
        + '\n    banded()\n    weights()\n    save_failures()\n'
    )
    output = run(build(cc, 'tensor_file_cases', program))
    (OUT / 'runtime.log').write_text('\n'.join(output) + '\n')
    assert output[: len(shapes)] == ['3'] * len(shapes), output
    errors = output[len(shapes) :]
    assert errors[:2] == ['add: shapes [2] and [3] do not match'] * 2, errors
    assert errors[2:] == [
        'save: file out/results/tensor_file/missing/raw.bin cannot be created',
        'save_npy: file out/results/tensor_file/missing/array.npy cannot be created',
    ], errors
    for name, words in expected.items():
        for suffix in ['.bin', '_returned.bin']:
            actual = np.fromfile(OUT / (name + suffix), dtype='<u4')
            assert np.array_equal(actual, words.ravel()), (name, suffix)
        actual = np.load(OUT / f'{name}.npy', allow_pickle=False)
        assert actual.dtype == np.dtype('<f4') and actual.shape == words.shape, name
        assert np.array_equal(actual.view(np.uint32), words), name
    for suffix in ['bin', 'npy']:
        assert (OUT / f'untouched.{suffix}').read_bytes() == b'must remain untouched'
    for folder in ['weights_output', 'weights_returned']:
        assert (OUT / folder / 'names.txt').read_text().splitlines() == ['remainder', 'empty']
        for name in ['remainder', 'empty']:
            actual = np.load(OUT / folder / f'{name}.npy', allow_pickle=False)
            assert actual.shape == expected[name].shape
            assert np.array_equal(actual.view(np.uint32), expected[name]), (folder, name)

    # A partial write fails after one full chunk; a zero-byte limit fails the NPY header.
    import resource

    fault_source = (
        HELPERS
        + f"""
def main() -> IO(Unit):
  do IO<Unit>:
    raw : F.Tensor() <- F.save_raw("out/results/tensor_file/limited.bin",F.zeros([{2 * chunk + 37}]))
    IO.print(problem(F.check(raw)))
    npy : F.Tensor() <- F.save_npy("out/results/tensor_file/limited.npy",F.zeros([{2 * chunk + 37}]))
    IO.print(problem(F.check(npy)))
"""
    )
    fault_binary = build(cc, 'tensor_file_faults', fault_source)
    for limit in [0, chunk * 4 + 141]:

        def restrict_size():
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))

        errors = run(fault_binary, preexec_fn=restrict_size, restore_signals=False)
        assert errors == [
            'save: file out/results/tensor_file/limited.bin cannot be written',
            'save_npy: file out/results/tensor_file/limited.npy cannot be written',
        ], errors
        assert (OUT / 'limited.bin').stat().st_size == limit
        assert (OUT / 'limited.npy').stat().st_size == limit
    status = dict(
        chunk=chunk,
        shapes=shapes,
        bitwise=True,
        retained_workers=3,
        weights_round_trip=True,
        failed_input_untouched=True,
        open_failure=True,
        header_failure=True,
        partial_write_failure=True,
        compiler=Path(cc).name,
        numpy=np.__version__,
    )
    (OUT / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(
        f'PASS: {len(shapes)} bitwise raw/NPY cases, returned tensors, weights, and file failures'
    )


if __name__ == '__main__':
    main()
