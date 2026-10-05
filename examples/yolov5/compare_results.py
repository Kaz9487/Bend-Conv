"""Validate the independently executed Bend outputs and render detections."""

import argparse
import json
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.yolov5.image_pipeline import nms, draw, iou_one

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / 'bend.cmd').is_file()
)


def run(case, threads=1):
    if threads < 1:
        raise ValueError('threads must be positive')
    backend = 'native' + (f'_{threads}t' if threads != 1 else '')
    stem = case + (f'_cpu_{threads}t' if threads != 1 else '')
    folder = ROOT / 'out/results' / case
    ref = json.loads((folder / 'reference.json').read_text())
    run = json.loads((folder / backend / 'native_run.json').read_text())
    native_meta = json.loads((ROOT / 'out/models' / f'{stem}.json').read_text())
    run.update(macs=native_meta['macs'], weights_sha256=native_meta['weights_sha256'])
    meta = json.loads((ROOT / 'out/weights/model.json').read_text())
    checks = []
    for name, shape in ref['outputs'].items():
        bp = folder / backend / f'{name}.bin'
        bend = np.fromfile(bp, dtype='<f4').reshape(shape)
        for reference_backend in ['torch', 'numpy']:
            p = folder / reference_backend / f'{name}.bin'
            if not p.exists():
                continue
            expected = np.fromfile(p, dtype='<f4').reshape(shape)
            diff = np.abs(bend - expected)
            atol = 0.01 if name == 'pred' else 0.0003
            rtol = 0.0003
            passed = bool(
                np.isfinite(bend).all() and np.allclose(bend, expected, atol=atol, rtol=rtol)
            )
            checks.append(
                dict(
                    name=name,
                    against=reference_backend,
                    shape=shape,
                    atol=atol,
                    rtol=rtol,
                    max_abs=float(diff.max()),
                    mean_abs=float(diff.mean()),
                    rmse=float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))),
                    passed=passed,
                )
            )
            print(
                name, reference_backend, 'max_abs', float(diff.max()), 'PASS' if passed else 'FAIL'
            )
    pred = np.fromfile(folder / backend / 'pred.bin', dtype='<f4').reshape(ref['outputs']['pred'])
    det = nms(pred)
    torch_det = np.array(
        json.loads((folder / 'torch/nms.json').read_text()), dtype=np.float32
    ).reshape(-1, 6)
    matches = []
    available = set(range(len(det)))
    for row in torch_det:
        candidates = [i for i in available if det[i, 5] == row[5]]
        if not candidates:
            break
        overlaps = iou_one(row[:4], det[candidates, :4])
        j = candidates[int(overlaps.argmax())]
        available.remove(j)
        matches.append(
            dict(
                class_id=int(row[5]),
                name=meta['names'][int(row[5])],
                iou=float(overlaps.max()),
                box_max_abs=float(np.abs(row[:4] - det[j, :4]).max()),
                score_abs=float(abs(row[4] - det[j, 4])),
            )
        )
    nms_pass = len(matches) == len(torch_det) == len(det) and all(
        m['iou'] > 0.999 and m['box_max_abs'] < 0.05 and m['score_abs'] < 0.0005 for m in matches
    )
    report = dict(
        case=case,
        passed=all(c['passed'] for c in checks) and nms_pass,
        checks=checks,
        nms_pass=nms_pass,
        torch_detections=len(torch_det),
        bend_detections=len(det),
        detection_matches=matches,
        timing=dict(
            torch_seconds=ref['torch_seconds'],
            numpy_seconds=ref['numpy_seconds'],
            bend_seconds=run['seconds'],
        ),
        backend=run['backend'],
        macs=run['macs'],
        weights_sha256=run['weights_sha256'],
    )
    build_path = folder / backend / 'bend_build.json'
    report['bend_build'] = json.loads(build_path.read_text()) if build_path.exists() else None
    (folder / backend / 'nms.json').write_text(json.dumps(det.tolist(), indent=2))
    (folder / f'comparison_{backend}.json').write_text(json.dumps(report, indent=2))
    draw(
        ROOT / 'data/images' / Path(ref['image']).name,
        det,
        ref['preprocess'],
        meta['names'],
        folder / f'{backend}_detections.jpg',
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    'passed',
                    'nms_pass',
                    'torch_detections',
                    'bend_detections',
                    'timing',
                    'detection_matches',
                )
            },
            indent=2,
        )
    )
    if not report['passed']:
        raise SystemExit(f'Bend comparison failed; inspect comparison_{backend}.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('case', nargs='?', default='bus_640')
    p.add_argument('--threads', type=int, default=1)
    a = p.parse_args()
    run(a.case, a.threads)
