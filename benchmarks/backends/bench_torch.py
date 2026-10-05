"""PyTorch eager execution of the exported official fused graph; no torch.compile."""

import argparse
import json
import statistics
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / 'bend.cmd').is_file()
)
parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
parser.add_argument('--threads', type=int, default=1)
parser.add_argument('--case', default='bus_640')
args = parser.parse_args()
torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('highest')
graph = json.loads((ROOT / 'out/backends' / f'{args.case}.json').read_text())
device = torch.device(args.device)
initial = {
    tensor_index: torch.from_numpy(
        np.fromfile(ROOT / tensor['path'], dtype='<f4').reshape(tensor['shape'])
    ).to(device)
    for tensor_index, tensor in enumerate(graph['tensors'])
    if tensor['path']
}
decode_constants = {}
for operation in graph['ops']:
    if operation['kind'] == 'decode':
        height, width = operation['input_height'], operation['input_width']
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), -1).float().reshape(1, 1, height, width, 2) - 0.5
        anchors = torch.tensor(operation['anchors'], device=device, dtype=torch.float32).reshape(
            1, 3, 1, 1, 2
        )
        decode_constants[operation['output']] = (grid, anchors)


@torch.inference_mode()
def forward():
    tensors = dict(initial)
    for operation in graph['ops']:
        kind = operation['kind']
        output_index = operation['output']
        if kind == 'conv':
            values = F.conv2d(
                tensors[operation['input']],
                tensors[operation['weights']],
                tensors[operation['bias']],
                stride=operation['stride'],
                padding=operation['padding'],
            )
            tensors[output_index] = F.silu(values) if operation['activation'] else values
        elif kind == 'concat':
            tensors[output_index] = torch.cat(
                [tensors[input_index] for input_index in operation['inputs']], dim=1
            )
        elif kind == 'add':
            tensors[output_index] = tensors[operation['left']] + tensors[operation['right']]
        elif kind == 'pool':
            tensors[output_index] = F.max_pool2d(tensors[operation['input']], 5, 1, 2)
        elif kind == 'up':
            tensors[output_index] = F.interpolate(
                tensors[operation['input']], scale_factor=2, mode='nearest'
            )
        else:
            height, width = operation['input_height'], operation['input_width']
            values = (
                tensors[operation['input']]
                .reshape(1, 3, 85, height, width)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
                .sigmoid()
            )
            grid, anchors = decode_constants[output_index]
            xy = (values[..., :2] * 2 + grid) * operation['stride']
            wh = (values[..., 2:4] * 2) ** 2 * anchors
            tensors[output_index] = torch.cat((xy, wh, values[..., 4:]), -1).reshape(1, -1, 85)
    return tensors


folder = ROOT / 'out/results/backends' / f'torch_{args.device}_{args.threads}t'
folder.mkdir(parents=True, exist_ok=True)
validation = forward()
for label, tensor_index in graph['snapshots'].items():
    validation[tensor_index].cpu().numpy().astype('<f4').tofile(folder / f'{label}.bin')
del validation
for _ in range(2):
    forward()
wall = []
events = []
if args.device == 'cuda':
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
for _ in range(7):
    if args.device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    if args.device == 'cuda':
        begin.record()
    result = forward()
    if args.device == 'cuda':
        end.record()
        torch.cuda.synchronize()
        events.append(begin.elapsed_time(end) / 1000)
    wall.append(time.perf_counter() - start)
    del result
# Untimed CPU-side profiler confirms the actual backend dispatch.
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    forward()
keys = sorted({e.key for e in prof.key_averages()})
report = dict(
    backend='PyTorch ' + args.device,
    torch=torch.__version__,
    cuda=torch.version.cuda,
    cudnn=torch.backends.cudnn.version(),
    device=torch.cuda.get_device_name() if args.device == 'cuda' else 'CPU',
    threads=args.threads,
    tf32=False,
    autocast=False,
    compile=False,
    cudnn_benchmark=True,
    warmups=2,
    seconds=wall,
    cuda_event_seconds=events,
    median=statistics.median(wall),
    operator_names=keys,
    graph='official exported BN-fused YOLOv5n graph, PyTorch eager functional operators',
)
(folder / 'timing.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: values for k, values in report.items() if k != 'operator_names'}, indent=2))
if args.device == 'cuda':
    assert any('cudnn' in k for k in keys), 'cuDNN dispatch was not observed'
