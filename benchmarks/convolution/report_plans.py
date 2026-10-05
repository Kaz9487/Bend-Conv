"""Read the real Bend planner without timing kernels or calibrating constants.

The generated entry contains shapes/resources only. Python never selects a plan.
Run on Linux after graph generation; optionally attach decisions to graph JSON.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Shape:
    input_channels: int
    output_channels: int
    input_height: int
    input_width: int
    kernel_size: int
    stride: int
    padding: int
    output_height: int
    output_width: int

    def __post_init__(self):
        values = asdict(self)
        if any(type(value) is not int for value in values.values()):
            raise ValueError('Convolution dimensions must be integers')
        if self.padding < 0 or any(value < 1 for key, value in values.items() if key != 'padding'):
            raise ValueError('Dimensions must be positive and padding nonnegative')
        expected = tuple(
            (extent + 2 * self.padding - self.kernel_size) // self.stride + 1
            for extent in (self.input_height, self.input_width)
        )
        if expected != (self.output_height, self.output_width):
            raise ValueError('Output dimensions do not match convolution parameters')


def capacity(elements):
    return 1 << (max(1, elements) - 1).bit_length()


EMIT = '''import Base
import ../../../convolution/planning.bend as Planning
import ../../../convolution/shape.bend as Shapes
import ../../../lib/kernel_planning.bend as KernelPlanning

def emit(plan: KernelPlanning.Plan) -> IO(Unit):
  KernelPlanning.Plan{packed,depth,workspace,time} = plan
  do IO<Unit>:
    IO.write(Bool.pick(String,packed,"packed ","direct "))
    IO.write(Nat.show(U32.to_nat(depth)))
    IO.write(" ")
    IO.write(Nat.show(workspace))
    IO.write(" ")
    IO.write(F32.show(time))
    IO.write("\\n")

def main() -> IO(Unit):
  do IO<Unit>:
'''


def workspace_literal(value):
    if value is None:
        return 'None{}'
    # Eight guarded partitions have less than 2^44 bytes of Array workspace.
    # Saturating a larger ceiling avoids native Nat overflow without excluding
    # any feasible plan. Bend literal tokens themselves are limited to U32.
    value = min(value, 1 << 44)
    high, low = divmod(value, 1 << 31)
    return f'Some{{Nat.add(Nat.mul({high}n,2147483648n),{low}n)}}'


def plans(requests):
    directory = ROOT / 'out/build/planning_report'
    directory.mkdir(parents=True, exist_ok=True)
    source = EMIT
    for shape, workers, budget in requests:
        values = ','.join(map(str, asdict(shape).values()))
        owners = [capacity(shape.input_channels * shape.input_height * shape.input_width),
                  capacity(shape.output_channels)]
        limit = workspace_literal(budget)
        source += f'    emit(Planning.plan(Shapes.Shape{{{values}}},{workers},{limit},KernelPlanning.Owners{{{",".join(map(str, owners))}}}))\n'
    entry = directory / 'plans.bend'
    entry.write_text(source)
    generated = directory / 'plans.c'
    subprocess.run(['node', str(ROOT / 'scripts/bend_launcher.mjs'), str(entry), '-o', str(generated)], check=True)
    executable = directory / 'plans'
    subprocess.run([os.environ.get('CC', 'clang'), '-O3', '-ffp-contract=off', '-std=c11',
                    str(generated), '-lpthread', '-lm', '-o', str(executable)], check=True)
    result = subprocess.run([str(executable), '--threads', '1', '--gpu', 'off'], check=True, capture_output=True, text=True)
    records = []
    for line, (shape, workers, budget) in zip(result.stdout.splitlines(), requests, strict=True):
        strategy, depth, workspace, time = line.split()
        depth, workspace = int(depth), int(workspace)
        assert strategy == 'packed' or workspace == 0
        assert budget is None or workspace <= budget
        # One worker never splits; several take up to five tasks each (XNNPACK).
        assert 1 << depth <= (1 if workers <= 1 else 5 * workers)
        records.append(dict(strategy=strategy, split_depth=depth, tasks=1 << depth,
                            worker_limit=workers, workspace_budget=budget,
                            workspace_bytes=workspace, estimated_ns=float(time)))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('graph', type=Path)
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text())
    operations = [operation for operation in graph['ops'] if operation['kind'] == 'conv']
    # Generated models give F.conv2d the runtime worker count (IO.thread_count),
    # which is the thread count the build runs with, and no workspace budget.
    requests = [(Shape(*(operation[key] for key in Shape.__dataclass_fields__)), graph['threads'], None)
                for operation in operations]
    for operation, plan in zip(operations, plans(requests), strict=True):
        operation['partition'] = plan
        operation['split_depth'] = plan['split_depth']
    args.graph.write_text(json.dumps(graph, indent=2))


if __name__ == '__main__':
    main()
