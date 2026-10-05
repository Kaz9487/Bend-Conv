"""Run in WSL. One binary per backend; all shapes are runtime data.

Thread controls must be applied before NumPy import. Workers are subprocesses so
OpenBLAS configuration cannot leak across measurements. Disk I/O is not timed.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / 'bend.cmd').is_file()
)
REPORT = ROOT / 'out/results/cpu_convolution'


def numpy_worker(folder, threads):
    os.environ.update(OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads))
    import numpy as np
    from threadpoolctl import threadpool_info

    # Initialize the BLAS worker pool outside the timer, like Bend's runtime.
    warm = np.ones((64, 64), np.float32)
    warm @ warm
    (
        input_channel_count,
        output_channel_count,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        output_height,
        output_width,
        _,
    ) = np.fromfile(folder / 'config.bin', np.uint32)[:10].tolist()
    input_values = np.fromfile(folder / 'input.bin', np.float32).reshape(
        input_channel_count, input_height, input_width
    )
    weight = np.fromfile(folder / 'weight.bin', np.float32).reshape(output_channel_count, -1)
    bias = np.fromfile(folder / 'bias.bin', np.float32)
    rows = []
    reps = int(os.environ.get('CONV_REPS', '1'))
    for rep in range(reps):
        input_values = np.fromfile(folder / 'input.bin', np.float32).reshape(
            input_channel_count, input_height, input_width
        )
        weight = np.fromfile(folder / 'weight.bin', np.float32).reshape(output_channel_count, -1)
        bias = np.fromfile(folder / 'bias.bin', np.float32)
        packing_start = time.perf_counter_ns()
        padded_input = np.pad(input_values, ((0, 0), (padding, padding), (padding, padding)))
        windows = np.lib.stride_tricks.sliding_window_view(
            padded_input, (kernel_size, kernel_size), axis=(1, 2)
        )[:, ::stride, ::stride]
        patches = np.ascontiguousarray(windows.transpose(1, 2, 0, 3, 4)).reshape(
            output_height * output_width, -1
        )
        kernel_start = time.perf_counter_ns()
        output_values = patches @ weight.T
        gather_start = time.perf_counter_ns()
        output_values = np.ascontiguousarray((output_values + bias).T).reshape(
            output_channel_count, output_height, output_width
        )
        finished = time.perf_counter_ns()
        if reps == 1 or rep >= 2:
            rows.append(
                dict(
                    packing_ms=(kernel_start - packing_start) / 1e6,
                    kernel_ms=(gather_start - kernel_start) / 1e6,
                    gather_ms=(finished - gather_start) / 1e6,
                    total_ms=(finished - packing_start) / 1e6,
                )
            )
        output_values.tofile(folder / 'numpy.bin')
        del output_values, patches, windows, padded_input
    (folder / f'numpy_{threads}.json').write_text(
        json.dumps(dict(samples=rows, pools=threadpool_info(), version=np.__version__), indent=2)
    )


def prepare(output=REPORT):
    import numpy as np

    # Shapes used by official YOLOv5n; seeded inputs isolate the operator.
    cases = [
        ('cv1_640', 32, 16, 160, 160, 1, 1, 0, 'model_2_cv1_conv'),
        ('down1_640', 16, 32, 320, 320, 3, 2, 1, 'model_1_conv'),
        ('down5_640', 64, 128, 80, 80, 3, 2, 1, 'model_5_conv'),
        ('down7_160_tail', 128, 256, 10, 10, 3, 2, 1, 'model_7_conv'),
        ('detect_320_tail', 256, 255, 10, 10, 1, 1, 0, 'model_24_m_2'),
        ('stem_640', 3, 16, 640, 640, 6, 2, 2, 'model_0_conv'),
    ]
    rng = np.random.default_rng(20260918)
    for (
        name,
        input_channel_count,
        output_channel_count,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        prefix,
    ) in cases:
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        output_height = (input_height + 2 * padding - kernel_size) // stride + 1
        output_width = (input_width + 2 * padding - kernel_size) // stride + 1
        np.array(
            [
                input_channel_count,
                output_channel_count,
                input_height,
                input_width,
                kernel_size,
                stride,
                padding,
                output_height,
                output_width,
                0,
            ],
            np.uint32,
        ).tofile(folder / 'config.bin')
        rng.normal(0, 0.25, (input_channel_count, input_height, input_width)).astype(
            np.float32
        ).tofile(folder / 'input.bin')
        for kind in ['weight', 'bias']:
            data = (ROOT / f'out/weights/{prefix}_{kind}.bin').read_bytes()
            assert len(data) == 4 * (
                output_channel_count * input_channel_count * kernel_size * kernel_size
                if kind == 'weight'
                else output_channel_count
            )
            (folder / f'{kind}.bin').write_bytes(data)
        (folder / 'case.json').write_text(
            json.dumps(
                dict(
                    name=name,
                    shape=[
                        input_channel_count,
                        output_channel_count,
                        input_height,
                        input_width,
                        kernel_size,
                        stride,
                        padding,
                        output_height,
                        output_width,
                    ],
                    weight_source=prefix,
                    seed=20260918,
                ),
                indent=2,
            )
        )
    return [gather_start[0] for gather_start in cases]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--numpy', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    numpy_worker(args.numpy, args.threads)
