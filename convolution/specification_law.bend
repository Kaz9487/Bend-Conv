# Contract v1. Written before implementation; changes require a new version
# and an explanation, not weakening a failed proof until it passes.
import Base
import ./specification.bend as S
import ./reference.bend as I

law ordered_reduction:
  for +products: List<&2,S.Product>
  for +acc: F32
  {I.sum(products,acc) == S.sum(S.multiply(products),acc) : F32}

law job_values:
  for +jobs: List<&2,S.DotJob>
  {I.evaluate(jobs) == S.evaluate(jobs) : List<&2,F32>}

law convolution:
  for +p: S.Problem
  {I.infer(p) == S.reference(p) : S.ConvOutcome}
