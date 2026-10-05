"""Generate each integration entry directly from the same case data."""

import json
from pathlib import Path
from convolution_cases import convolution_cases, oracle, flist

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'out/checks'
REPORT = ROOT / 'out/results/integration'
OUTPUT.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)
cases = convolution_cases()
for case in cases:
    case['expected'] = oracle(case)
(REPORT / 'cases.json').write_text(json.dumps(cases, indent=2))

helpers = """def input(+xs: List<&2,F32>) -> DenseArray.Buffer<F32>:
  DenseArray.from_list(~F32,xs,DenseArray.allocation_depth(Bool.pick(U32,List.is_empty(&2,F32,xs),1,U32.from_nat(Spec.length(xs)))),0.0,Spec.length(xs))

def convert(+channels: Nat,outcome: Tensor.Outcome) -> Spec.ConvOutcome:
  match outcome:
    case Tensor.Rejected{}: Spec.Rejected{}
    case Tensor.Unsupported{}: Spec.Rejected{}
    case Tensor.Accepted{+height,+width,values}:
      Spec.Accepted{height,width,DenseArray.read_list(F32,Nat.mul(channels,Nat.mul(height,width)),0,values)}

def candidate(+problem: Spec.Problem) -> Spec.ConvOutcome:
  Spec.Problem{shape,+xs,+ws,+bs} = problem
  convert(Spec.outputs(problem),Tensor.infer(shape,Convolution.automatic(4,None{}),input(xs),input(ws),input(bs)))

"""
for entry, module in [
    ('convolution', 'inference'),
    ('scalar_convolution', 'scalar_kernel'),
    ('array_api', 'array_api'),
    ('array_api_zero_workspace', 'array_api'),
]:
    source = 'import Base\nimport ../../convolution/specification.bend as Spec\nimport ../../benchmarks/checks/result_io.bend as Output\n'
    source += (
        f'import ../../convolution/{module}.bend as '
        + ('Tensor' if entry.startswith('array_api') else 'Candidate')
        + '\n'
    )
    if entry.startswith('array_api'):
        source += (
            'import ../../convolution/scalar_kernel.bend as Scalar\nimport ../../convolution/inference.bend as Convolution\nimport ../../lib/storage_buffer.bend as DenseArray\n\n'
            + (helpers.replace('None{}', 'Some{0n}') if entry.endswith('zero_workspace') else helpers)
        )
    for index, case in enumerate(cases):
        shape = ','.join(f'{value}n' for value in case['shape'])
        values = ','.join(flist(case[key]) for key in ['x', 'w', 'b'])
        source += f'def case_{index}() -> Spec.Problem:\n  Spec.Problem{{Spec.Shape{{{shape}}},{values}}}\n\n'
    source += 'def main() -> IO(Unit):\n  do IO<Unit>:\n'
    call = 'candidate' if entry.startswith('array_api') else 'Candidate.infer'
    for index in range(len(cases)):
        source += f'    Output.dump({2 * index},Spec.reference(case_{index}()))\n    Output.dump({2 * index + 1},{call}(case_{index}()))\n'
    (OUTPUT / f'{entry}.bend').write_text(source, encoding='utf-8')
print(
    f'{len(cases)} existing cases; native convolution, scalar fallback and Array convolution API.'
)

