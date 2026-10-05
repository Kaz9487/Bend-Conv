"""Current official Bend, packed C twin and NumPy on the same FP32 convolutions."""

from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
from numpy_convolution import prepare
from report_plans import Shape, plans

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reporting import write_report

BUILD = ROOT / 'out/build/cpu_convolution'
REPORT = ROOT / 'out/results/cpu_convolution'


def main():
    toolchain = json.loads((ROOT / 'scripts/toolchain.json').read_text())
    build = json.loads((BUILD / 'bend_convolution.c.build.json').read_text())
    assert build['compiler_commit'] == toolchain['commit'], (
        'Rebuild with the pinned official compiler'
    )
    environment = dict(os.environ, CONV_REPS='9')
    records = []
    cases = list(prepare(REPORT))
    requests = [(Shape(*np.fromfile(REPORT / case / 'config.bin', np.uint32)[:9].tolist()), threads, None)
                for case in cases for threads in [1, 4]]
    decisions = dict(zip(((case, threads) for case in cases for threads in [1, 4]), plans(requests), strict=True))
    for case in cases:
        folder = REPORT / case
        for threads in [1, 4]:
            config = np.fromfile(folder / 'config.bin', np.uint32)
            config[9] = threads
            config.tofile(folder / 'config.bin')
            samples, outputs = {}, {}
            order = ['bend', 'packed_c', 'packed_c_scalar', 'numpy']
            if threads in [2, 8]:
                order.reverse()
            for backend in order:
                if backend == 'numpy':
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).with_name('numpy_convolution.py')),
                            '--numpy',
                            str(folder),
                            '--threads',
                            str(threads),
                        ],
                        env=environment,
                        check=True,
                        timeout=180,
                    )
                    samples[backend] = json.loads((folder / f'numpy_{threads}.json').read_text())[
                        'samples'
                    ]
                    outputs[backend] = np.fromfile(folder / 'numpy.bin', np.float32)
                    continue
                is_c = backend.startswith('packed_c')
                label = 'c' if is_c else 'bend'
                sample_file = folder / f'{label}_samples.jsonl'
                sample_file.write_text('')
                command = [
                    str(BUILD / (backend if is_c else 'bend_convolution'))
                ]
                command += (
                    [str(threads)]
                    if is_c
                    else ['--threads', str(threads), '--gpu', 'off']
                )
                subprocess.run(
                    command,
                    cwd=folder,
                    env=environment,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    timeout=180,
                )
                rows = [json.loads(line) for line in sample_file.read_text().splitlines()]
                assert len(rows) == 9, (backend, len(rows))
                samples[backend] = rows[2:]
                outputs[backend] = np.fromfile(folder / f'{label}.bin', np.float32)
            assert np.array_equal(
                outputs['bend'].view(np.uint32), outputs['packed_c'].view(np.uint32)
            )
            assert np.array_equal(
                outputs['packed_c'].view(np.uint32), outputs['packed_c_scalar'].view(np.uint32)
            )
            assert np.isfinite(outputs['bend']).all()
            assert np.allclose(outputs['bend'], outputs['numpy'], atol=3e-4, rtol=3e-4)
            records.append(
                dict(
                    case=case,
                    threads=threads,
                    order=order,
                    plan=decisions[case, threads],
                    samples=samples,
                    compiler_commit=toolchain['commit'],
                    native_bitwise_c=True,
                    numpy_close=True,
                    median_ms={
                        name: {
                            key: float(np.median([row[key] for row in values])) for key in values[0]
                        }
                        for name, values in samples.items()
                    },
                )
            )
            (REPORT / 'results.json').write_text(json.dumps(records, indent=2))
            print(
                case,
                threads,
                {
                    name: round(value['total_ms'], 3)
                    for name, value in records[-1]['median_ms'].items()
                },
                flush=True,
            )
    (REPORT / 'measurement.json').write_text(
        json.dumps(
            dict(
                compiler_commit=toolchain['commit'],
                protocol='FP32; 2 warmups and 7 measured samples; 1/4 threads; Bend automatic planning; no forced-depth calibration',
                c_backend='ordered packed C; channel partition differs from Bend spatial partition',
                c_scalar_flags='-fno-vectorize -fno-slp-vectorize',
                attribution='C vectorization control isolates compiler vectorization within C only; Bend/C also differ in ownership, allocation and partitioning.',
            ),
            indent=2,
        )
    )
    write_measurement_report(records, toolchain)


def write_measurement_report(records, toolchain):
    sections = [
        ('# CPU convolution results', '# CPU 卷積結果'),
        f'Bend {toolchain["tag"]} · `{toolchain["commit"]}`',
        (
            '[Measurement protocol](../../../benchmarks/convolution/README.md). Medians in milliseconds; the estimate is the Bend planner prediction for the selected plan:',
            '[量測流程](../../../benchmarks/convolution/README.zh-TW.md)。中位數，單位毫秒；估計值是 Bend 規劃器對所選方案的預測：',
        ),
        (
            '| Case | Threads | Bend total | Estimate | Error | Packed C total | NumPy total | Reorder | Pack+GEMM | Gather |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
            '| 案例 | 執行緒 | Bend 總時間 | 估計 | 誤差 | Packed C 總時間 | NumPy 總時間 | 重排 | 打包+GEMM | 收集 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        ),
    ]
    for record in records:
        timing = record['median_ms']
        bend = timing['bend']
        estimate = record['plan']['estimated_ns'] / 1e6
        error = (estimate - bend['total_ms']) / bend['total_ms']
        sections.append(
            f'| {record["case"]} | {record["threads"]} | {bend["total_ms"]:.3f} | {estimate:.3f} | {error:+.0%} | {timing["packed_c"]["total_ms"]:.3f} | {timing["numpy"]["total_ms"]:.3f} | {bend["reorder_ms"]:.3f} | {bend["compute_ms"]:.3f} | {bend["gather_ms"]:.3f} |'
        )
    sections.append(
        (
            '[Samples](results.json) · [Conditions](measurement.json)',
            '[取樣](results.json) · [條件](measurement.json)',
        )
    )
    write_report(REPORT / 'README.md', sections)


if __name__ == '__main__':
    main()
