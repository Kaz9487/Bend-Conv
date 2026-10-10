"""One accuracy gate for every executed backend, independently from timing."""

import argparse
import json
import statistics
import sys
from pathlib import Path
import numpy as np

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / 'bend.cmd').is_file()
)
sys.path.insert(0, str(ROOT))
from examples.yolov5.image_pipeline import nms, iou_one

refdir = ROOT / 'out/results/bus_640'
ref = json.loads((refdir / 'reference.json').read_text())
base = ROOT / 'out/results/backends'
parser = argparse.ArgumentParser()
parser.add_argument('--backends', nargs='+', help='Validate only these freshly measured backends')
args = parser.parse_args()
if args.backends:
    for name in args.backends:
        if Path(name).name != name or not (base / name / 'timing.json').is_file():
            raise SystemExit('Missing backend timing: ' + name)
rows = []
for folder in sorted(base.iterdir()):
    if args.backends and folder.name not in args.backends:
        continue
    if not folder.is_dir() or not (folder / 'timing.json').exists():
        continue
    timing = json.loads((folder / 'timing.json').read_text())
    checks = []
    for name, shape in ref['outputs'].items():
        path = folder / f'{name}.bin'
        saved = folder / f'{name}.npy'
        if not path.exists() and not saved.exists():
            # NumPy's public runner returns 24 layers and predictions, not raw heads.
            assert folder.name.startswith('numpy') and name.startswith('head_'), path
            continue
        actual = np.load(saved) if saved.exists() else np.fromfile(path, dtype='<f4').reshape(shape)
        assert list(actual.shape) == shape, name
        expected = np.fromfile(refdir / 'torch' / f'{name}.bin', dtype='<f4').reshape(shape)
        old = np.load(refdir / 'native' / f'{name}.npy')
        assert list(old.shape) == shape, name
        checks.append(
            dict(
                name=name,
                passed=bool(
                    np.isfinite(actual).all()
                    and np.allclose(
                        actual, expected, atol=0.01 if name == 'pred' else 3e-4, rtol=3e-4
                    )
                ),
                max_abs=float(np.abs(actual - expected).max()),
                native_bitwise=actual.tobytes() == old.tobytes(),
            )
        )
    pred = np.load(folder / 'pred.npy') if (folder / 'pred.npy').exists() else np.fromfile(folder / 'pred.bin', dtype='<f4').reshape(ref['outputs']['pred'])
    det = nms(pred)
    expected = np.array(
        json.loads((refdir / 'torch/nms.json').read_text()), dtype=np.float32
    ).reshape(-1, 6)
    available = set(range(len(det)))
    matches = []
    for row in expected:
        candidates = [i for i in available if det[i, 5] == row[5]]
        if not candidates:
            break
        overlap = iou_one(row[:4], det[candidates, :4])
        j = candidates[int(overlap.argmax())]
        available.remove(j)
        matches.append(
            dict(
                iou=float(overlap.max()),
                box_max_abs=float(np.abs(row[:4] - det[j, :4]).max()),
                score_abs=float(abs(row[4] - det[j, 4])),
            )
        )
    nms_pass = len(matches) == len(expected) == len(det) and all(
        m['iou'] > 0.999 and m['box_max_abs'] < 0.05 and m['score_abs'] < 0.0005 for m in matches
    )
    result = dict(
        name=folder.name,
        passed=all(c['passed'] for c in checks) and nms_pass,
        checks=checks,
        nms_pass=nms_pass,
        matches=matches,
        detections=len(det),
        timing=timing,
        median=statistics.median(timing['seconds']),
    )
    (folder / 'comparison.json').write_text(json.dumps(result, indent=2))
    rows.append(result)
pin = json.loads((ROOT / 'scripts/toolchain.json').read_text())
for row in rows:
    if row['name'].startswith('bend_'):
        build = json.loads((base / row['name'] / 'bend_build.json').read_text())
        assert build['compiler_commit'] == pin['commit'], 'Rebuild with the pinned Bend compiler'
(base / 'summary.json').write_text(json.dumps(dict(backends=rows), indent=2))
for row in rows:
    print(
        row['name'],
        row['median'],
        'PASS' if row['passed'] else 'FAIL',
        len(row['checks']),
        'tensors',
        row['detections'],
        'detections',
    )
assert all(r['passed'] for r in rows)
