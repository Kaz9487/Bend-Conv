# Contract-first candidate: stream products and output jobs.
# Geometry/validation are shared with the frozen semantic specification.
import Base
import ./specification.bend as ConvSpec
import ./ordered_reference.bend as OrderedReference

law sum:
  for products: List<&2,ConvSpec.Product>
  for acc: F32
  F32

law evaluate:
  for jobs: List<&2,ConvSpec.DotJob>
  List<&2,F32>

law infer:
  for p: ConvSpec.Problem
  ConvSpec.ConvOutcome

def sum(products,acc):
  match products:
    case Nil{}:
      acc
    case ConvSpec.Product{x,w} <> tail:
      sum(tail,(acc + x * w : F32))

def evaluate(jobs):
  match jobs:
    case Nil{}:
      Nil{}
    case ConvSpec.DotJob{products,bias} <> tail:
      (sum(products,0.0) + bias : F32) <> evaluate(tail)

def finish(ok: Bool, +p: ConvSpec.Problem) -> ConvSpec.ConvOutcome:
  match ok:
    case False{}:
      ConvSpec.Rejected{}
    case True{}:
      ConvSpec.Accepted{ConvSpec.out_height(p),ConvSpec.out_width(p),OrderedReference.values(p)}

def infer(p):
  +p = p
  finish(ConvSpec.valid(p),p)
