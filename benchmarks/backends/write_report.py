"""Render the validated backend evidence in two languages."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reporting import write_report

REPORT = ROOT / 'out/results/backends'
parser = argparse.ArgumentParser()
parser.add_argument('--cpu-only', action='store_true')
args = parser.parse_args()
summary = json.loads((REPORT / 'summary.json').read_text())
rows = {row['name']: row for row in summary['backends']}
required = ['numpy_1t', 'numpy_4t', 'bend_cpu', 'bend_cpu_4t', 'torch_cpu_1t', 'torch_cpu_4t']
if not args.cpu_only:
    required += ['c_naive', 'cuda_naive', 'torch_cuda_1t']
assert all(name in rows and rows[name]['passed'] for name in required)
sections = [
    ('# YOLOv5n backend results', '# YOLOv5n 後端結果'),
    (
        'FP32 forward computation on the official pretrained model. [Protocol and reproduction](../../benchmarks/backends/README.md). Raw settings, samples and comparisons are linked below.',
        '官方預訓練模型的 FP32 forward 計算。[量測流程與重現方式](../../benchmarks/backends/README.zh-TW.md)。原始設定、樣本與比較結果見下方連結。',
    ),
    (
        '| Backend | Median (ms) | Min–max (ms) | Tensors | NMS |\n|---|---:|---:|---:|---|',
        '| 後端 | 中位數 (ms) | 最小–最大 (ms) | Tensor 數 | NMS |\n|---|---:|---:|---:|---|',
    ),
]
for name in required:
    row = rows[name]
    values = row['timing']['seconds']
    sections.append(
        f'| [{name}](backends/{name}/comparison.json) | {row["median"] * 1000:.3f} | {min(values) * 1000:.3f}–{max(values) * 1000:.3f} | {len(row["checks"])} | PASS |'
    )
if not args.cpu_only:
    build = json.loads((REPORT / 'bend_cuda/bend_build.json').read_text())
    pin = json.loads((ROOT / 'scripts/toolchain.json').read_text())
    assert build['compiler_commit'] == pin['commit']
    exit_code = int((REPORT / 'bend_cuda/exit_code.txt').read_text())
    sections.append(('## Bend CUDA', '## Bend CUDA', '## Bend CUDA'))
    if exit_code:
        sections.append(
            f'`exit_code={exit_code}`\n\n```text\n'
            + (REPORT / 'bend_cuda/stderr.log').read_text().strip()
            + '\n```'
        )
    else:
        row = rows['bend_cuda']
        assert row['passed']
        sections.append(
            f'`median_ms={row["median"] * 1000:.3f}` · [comparison.json](backends/bend_cuda/comparison.json)'
        )
    sections.append(
        (
            '[Device capabilities](backends/cuda_capabilities.json)',
            '[裝置能力](backends/cuda_capabilities.json)',
        )
    )
sections.append(
    (
        '[Sources and numerical results](backends/summary.json) · [CPU environment](backends/cpu.txt) · [Python environment](backends/requirements-linux.txt)',
        '[來源與數值結果](backends/summary.json) · [CPU 環境](backends/cpu.txt) · [Python 環境](backends/requirements-linux.txt)',
    )
)
write_report(ROOT / 'out/results/backend_comparison.md', sections)
print(ROOT / 'out/results/backend_comparison.md')
