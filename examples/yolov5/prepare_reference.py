"""Download provenance, official FP32 reference, and architecture/weight export."""

import os

os.environ['YOLOv5_AUTOINSTALL'] = 'false'
os.environ['MPLCONFIGDIR'] = str(
    __import__('pathlib').Path(__file__).resolve().parents[2] / '.tools/matplotlib'
)
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / 'bend.cmd').is_file()
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.tools/yolov5-official'))
import torch
import numpy as np
from examples.yolov5.image_pipeline import preprocess, nms, draw
from examples.yolov5.numpy_yolo import NumpyYoloModel


def run(size=640, image='bus'):
    torch.set_num_threads(4)
    weights = ROOT / 'data/weights/yolov5n-v7.0.pt'
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    model = ckpt['model'].float().eval()
    assert len(model.names) == 80, 'Expected official COCO 80 classes'
    from models.common import Conv

    assert all(isinstance(m.act, torch.nn.SiLU) for m in model.modules() if isinstance(m, Conv))
    model.fuse()
    export = ROOT / 'out/weights'
    export.mkdir(parents=True, exist_ok=True)
    detect = model.model[-1]
    names = [model.names[i] for i in range(80)]
    meta = dict(
        nc=80,
        names=names,
        strides=detect.stride.tolist(),
        anchors_pixel=(detect.anchors * detect.stride[:, None, None]).tolist(),
        source='https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt',
        sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        source_commit='915bbf294bb74c859f0b41f1c23bc395014ea679',
        torch_version=torch.__version__,
        layers=[],
        weights={},
    )
    # The same weights under their PyTorch names, for the Bend model.
    named = export / 'named'
    named.mkdir(exist_ok=True)
    for name, t in model.state_dict().items():
        if not (name.endswith('.weight') or name.endswith('.bias')):
            continue
        v = t.detach().numpy().astype('<f4')
        fn = name.replace('.', '_') + '.bin'
        v.tofile(export / fn)
        np.save(named / f'{name}.npy', v)
        meta['weights'][name] = {'file': fn, 'shape': list(v.shape)}
    (named / 'names.txt').write_text('\n'.join(meta['weights']), encoding='utf-8', newline='\n')
    for i, m in enumerate(model.model):
        typ = m.__class__.__name__
        spec = {'i': i, 'f': m.f, 'type': typ}
        if typ == 'Conv':
            spec['conv'] = {'stride': list(m.conv.stride), 'padding': list(m.conv.padding)}
        elif typ == 'C3':
            spec.update(n=len(m.m), shortcut=m.m[0].add)
        elif typ == 'SPPF':
            spec['k'] = m.m.kernel_size
        meta['layers'].append(spec)
    (export / 'model.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    output = ROOT / 'out/results' / f'{image}_{size}'
    output.mkdir(parents=True, exist_ok=True)
    ref_dir = output / 'torch'
    ref_dir.mkdir(exist_ok=True)
    np_dir = output / 'numpy'
    np_dir.mkdir(exist_ok=True)
    image_path = ROOT / 'data/images' / f'{image}.jpg'
    x, info = preprocess(image_path, size)
    x.tofile(output / 'input.bin')
    tensors = {}
    handles = []
    for i, m in enumerate(model.model):
        if i < 24:
            handles.append(
                m.register_forward_hook(
                    lambda mod, inp, out, i=i: tensors.__setitem__(
                        f'layer_{i:02}', out.detach().cpu().numpy().copy()
                    )
                )
            )
    for i, m in enumerate(detect.m):
        handles.append(
            m.register_forward_hook(
                lambda mod, inp, out, i=i: tensors.__setitem__(
                    f'head_{i}', out.detach().cpu().numpy().copy()
                )
            )
        )
    with torch.inference_mode():
        t = time.perf_counter()
        pred = model(torch.from_numpy(x))[0].numpy()
        torch_sec = time.perf_counter() - t
    for h in handles:
        h.remove()
    tensors['pred'] = pred
    manifest = {}
    for name, a in tensors.items():
        a.astype('<f4').tofile(ref_dir / f'{name}.bin')
        manifest[name] = list(a.shape)
    gold = NumpyYoloModel(export)
    t = time.perf_counter()
    outputs = gold.forward(x, return_all_outputs=True)
    numpy_sec = time.perf_counter() - t
    checks = []
    for i, a in enumerate(outputs):
        name = f'layer_{i:02}' if i < 24 else 'pred'
        a.astype('<f4').tofile(np_dir / f'{name}.bin')
        r = tensors[name]
        diff = np.abs(a - r)
        atol = 2e-3 if name == 'pred' else 2e-4
        passed = bool(np.allclose(a, r, atol=atol, rtol=2e-4))
        checks.append(
            dict(
                name=name,
                shape=list(a.shape),
                dtype=str(a.dtype),
                max_abs=float(diff.max()),
                mean_abs=float(diff.mean()),
                passed=passed,
            )
        )
        print(
            name, a.shape, a.dtype, 'max_abs=', diff.max(), 'PASS' if passed else 'FAIL', flush=True
        )
    # Independent torchvision NMS on the official Torch predictions.
    from utils.general import non_max_suppression

    official_nms = non_max_suppression(torch.from_numpy(pred.copy()), 0.25, 0.45)[0].numpy()
    official_nms.astype('<f4').tofile(ref_dir / 'nms.bin')
    (ref_dir / 'nms.json').write_text(json.dumps(official_nms.tolist()))
    detections = nms(outputs[-1])
    (np_dir / 'nms.json').write_text(json.dumps(detections.tolist()))
    draw(image_path, detections, info, names, output / 'numpy_detections.jpg')
    report = dict(
        image=str(image_path),
        size=size,
        preprocess=info,
        input_shape=list(x.shape),
        outputs=manifest,
        torch_seconds=torch_sec,
        numpy_seconds=numpy_sec,
        checks=checks,
        torch_detections=len(official_nms),
        numpy_detections=len(detections),
        nms_close=bool(
            official_nms.shape == detections.shape
            and np.allclose(official_nms, detections, atol=0.02, rtol=2e-4)
        ),
        weights_sha256=meta['sha256'],
    )
    (output / 'reference.json').write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    'torch_seconds',
                    'numpy_seconds',
                    'torch_detections',
                    'numpy_detections',
                    'nms_close',
                )
            },
            indent=2,
        )
    )
    if not all(c['passed'] for c in checks) or not report['nms_close']:
        raise SystemExit('Reference comparison failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--size', type=int, default=640)
    p.add_argument('--image', default='bus', choices=['bus', 'zidane'])
    a = p.parse_args()
    if a.size < 32 or a.size % 32:
        p.error('size must be a positive multiple of 32')
    run(a.size, a.image)
