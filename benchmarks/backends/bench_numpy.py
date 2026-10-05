import os

os.environ['OPENBLAS_NUM_THREADS'] = os.environ.get('BENCH_THREADS', '1')
os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS']
import json
import sys
import time
import statistics
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / 'bend.cmd').is_file()
)
sys.path.insert(0, str(ROOT))
from examples.yolov5.numpy_yolo import NumpyYoloModel

case = 'bus_640'
ref = json.loads((ROOT / 'out/results' / case / 'reference.json').read_text())
model = NumpyYoloModel(ROOT / 'out/weights')
x = np.fromfile(ROOT / 'out/results' / case / 'input.bin', dtype='<f4').reshape(ref['input_shape'])
folder = ROOT / 'out/results/backends' / f'numpy_{os.environ["OPENBLAS_NUM_THREADS"]}t'
folder.mkdir(parents=True, exist_ok=True)
out = model.forward(x, return_all_outputs=True)
for i, v in enumerate(out):
    v.astype('<f4').tofile(folder / ('pred.bin' if i == 24 else f'layer_{i:02}.bin'))
del out
for _ in range(2):
    model.forward(x, return_all_outputs=True)
times = []
for _ in range(7):
    start = time.perf_counter()
    out = model.forward(x, return_all_outputs=True)
    times.append(time.perf_counter() - start)
    del out
report = dict(
    backend='NumPy CPU',
    numpy=np.__version__,
    threadpools=threadpool_info(),
    warmups=2,
    seconds=times,
    median=statistics.median(times),
)
(folder / 'timing.json').write_text(json.dumps(report, indent=2))
print(folder.name, report['median'])
