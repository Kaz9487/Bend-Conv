"""Emit handwritten C/CUDA wiring from the shared YOLOv5n operation graph."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'examples/yolov5'))
from yolo_graph import build_graph


def emit_header(graph):
    tensors = graph['tensors']
    calls = []
    for operation in graph['ops']:
        kind, output = operation['kind'], operation['output']
        count = tensors[output]['count']
        if kind == 'conv':
            arguments = ','.join(
                str(operation[key])
                for key in (
                    'input_channels',
                    'output_channels',
                    'input_height',
                    'input_width',
                    'kernel_size',
                    'stride',
                    'padding',
                    'output_height',
                    'output_width',
                    'activation',
                )
            )
            calls.append(
                f'convolve(tensors[{operation["input"]}],tensors[{operation["weights"]}],tensors[{operation["bias"]}],tensors[{output}],scratch,{arguments});'
            )
        elif kind == 'concat':
            offset = 0
            for source in operation['inputs']:
                length = tensors[source]['count']
                calls.append(
                    f'memcpy(tensors[{output}]+{offset},tensors[{source}],{length}*sizeof(float));'
                )
                offset += length
        elif kind == 'add':
            calls.append(
                f'add_arrays(tensors[{operation["left"]}],tensors[{operation["right"]}],tensors[{output}],{count});'
            )
        elif kind in ('pool', 'up'):
            _, channels, height, width = tensors[operation['input']]['shape']
            function = 'max_pool' if kind == 'pool' else 'upsample_nearest'
            calls.append(
                f'{function}(tensors[{operation["input"]}],tensors[{output}],{channels},{height},{width});'
            )
        elif kind == 'decode':
            anchors = ','.join(
                str(float(value)) + 'f' for pair in operation['anchors'] for value in pair
            )
            calls.append(
                f'decode_predictions(tensors[{operation["input"]}],tensors[{output}],{operation["input_height"]},{operation["input_width"]},{float(operation["stride"])}f,(float[]){{{anchors}}});'
            )
        else:
            raise ValueError(f'Unsupported operation: {kind}')

    loads, allocations, releases = [], [], []
    for index, tensor in enumerate(tensors):
        if tensor['path']:
            loads.append(f'tensors[{index}]=load_floats("{tensor["path"]}",{tensor["count"]});')
        else:
            allocations.append(f'tensors[{index}]=allocate_floats({tensor["count"]});')
            releases.append(f'free(tensors[{index}]);tensors[{index}]=NULL;')
    maximum_patch_elements = max(
        operation['output_height']
        * operation['output_width']
        * operation['input_channels']
        * operation['kernel_size'] ** 2
        for operation in graph['ops']
        if operation['kind'] == 'conv'
    )
    functions = [
        ('load_graph', loads),
        ('alloc_graph', allocations + [f'scratch=allocate_floats({maximum_patch_elements});']),
        ('free_graph', releases + ['free(scratch);']),
        ('forward_graph', calls),
    ]
    header = f'static float* tensors[{len(tensors)}];\nstatic float* scratch;\n'
    for name, statements in functions:
        header += f'static void {name}(void){{\n' + '\n'.join(statements) + '\n}\n'
    snapshots = [
        f'dump_tensor(dir,"{name}",tensors[{index}],{tensors[index]["count"]});'
        for name, index in graph['snapshots'].items()
    ]
    return header + 'static void dump_graph(const char* dir){\n' + '\n'.join(snapshots) + '\n}\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='bus_640')
    args = parser.parse_args()
    model = json.loads((ROOT / 'out/weights/model.json').read_text())
    reference = json.loads((ROOT / 'out/results' / args.case / 'reference.json').read_text())
    graph = build_graph(model, reference, args.case)
    output = ROOT / 'out/backends'
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{args.case}.json').write_text(json.dumps(graph, indent=2))
    (output / f'{args.case}.h').write_text(emit_header(graph))
    print(args.case, len(graph['ops']), 'ops', len(graph['tensors']), 'tensors')


if __name__ == '__main__':
    main()
