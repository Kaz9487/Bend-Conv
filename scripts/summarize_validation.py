"""Index the existing validation results and report which ones passed."""

from datetime import datetime, timezone
import json
from pathlib import Path
from reporting import write_report

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'out/results'


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def pinned_build(build):
    assert build['compiler_commit'] == read('scripts/toolchain.json')['commit'], 'Rebuild with the pinned compiler'


def official_proofs():
    result = read('out/results/matrix/proof-status.json')
    assert result['native_to_matrix_proved'] and result['native_all_split_depths_proved']
    assert result['all_component_entries_proved'] and result['frozen_contracts_verified']


def integration():
    for entry in ['convolution', 'scalar_convolution', 'array_api', 'array_api_zero_workspace']:
        result = read(f'out/results/integration/runtime-{entry}.json')
        assert [row['threads'] for row in result] == [1, 2, 4, 8]
        assert all(row['all_bitwise'] and row['cases'] for row in result)
        pinned_build(read(f'out/build/integration/{entry}.c.build.json'))


def convolution():
    result = read('out/results/cpu_convolution/results.json')
    assert len(result) == 12 and all(
        row['native_bitwise_c'] and row['numpy_close'] for row in result
    )
    pinned_build(read('out/build/cpu_convolution/bend_convolution.c.build.json'))
    assert read('out/results/cpu_convolution/allocation_profile.json')['rows']


def models():
    cases = [(case, 1) for case in ['bus_320', 'bus_640', 'zidane_320', 'zidane_640']]
    for case, threads in cases + [('bus_640', 4)]:
        suffix = f'_{threads}t' if threads != 1 else ''
        result = read(f'out/results/{case}/comparison_native{suffix}.json')
        assert result['passed'] and result['nms_pass'], case
        pinned_build(result['bend_build'])


def model_comparison():
    result = read('out/results/backends/summary.json')
    rows = {row['name']: row for row in result['backends']}
    for name in ['numpy_1t', 'numpy_4t', 'bend_cpu', 'bend_cpu_4t', 'torch_cpu_1t', 'torch_cpu_4t']:
        row = rows[name]
        assert row['passed'] and row['nms_pass'], name
        assert len(row['timing']['seconds']) == 7 and row['timing']['warmups'] == 2, name


def main():
    checks = [
        (
            ('Official roots and shared contracts', '官方根定理與共用契約'),
            official_proofs,
            '../matrix/proof-status.json',
        ),
        (('Runtime integration', '執行時整合'), integration, '../integration'),
        (
            ('Convolution timing and allocation', '卷積計時與配置量'),
            convolution,
            '../cpu_convolution/README.md',
        ),
        (
            ('YOLO layers, predictions and NMS', 'YOLO 各層、預測與 NMS'),
            models,
            '../bus_640/comparison_native_4t.json',
        ),
        (
            ('Full-model CPU comparison', '完整模型 CPU 比較'),
            model_comparison,
            '../backends/summary.json',
        ),
    ]
    rows = []
    sections = [
        ('# Validation index', '# 驗收索引'),
        (
            '`python run.py summary` reads the existing results and lists which checks passed. Each row links its result.',
            '`python run.py summary` 讀取既有結果並列出哪些檢查通過。每列連到對應結果。',
        ),
    ]
    for labels, check, link in checks:
        try:
            check()
            row = dict(name=labels[0], status='passed', evidence=link)
        except (AssertionError, OSError, KeyError, ValueError) as error:
            row = dict(name=labels[0], status='missing_or_failed', evidence=link, reason=str(error))
        rows.append(row)
        sections.append(tuple(f'- [{label}]({link}): `{row["status"]}`' for label in labels))
        if 'reason' in row:
            sections.append('```text\n' + row['reason'] + '\n```')
    summary = dict(
        collected_utc=datetime.now(timezone.utc).isoformat(),
        checks=rows,
        all_passed=all(row['status'] == 'passed' for row in rows),
    )
    destination = REPORT / 'validation'
    destination.mkdir(exist_ok=True)
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    sections.append(
        (
            '[Detailed results](summary.json) · [Proof coverage](../../../docs/proofs.md)',
            '[詳細結果](summary.json) · [證明範圍](../../../docs/proofs.zh-TW.md)',
        )
    )
    write_report(destination / 'README.md', sections)
    if not summary['all_passed']:
        raise SystemExit('A required result is missing or failed; see out/results/validation/summary.json')
    print('All required results are present and passed.')


if __name__ == '__main__':
    main()
