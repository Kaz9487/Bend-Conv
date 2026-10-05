import Base
import ../specification.bend as ConvSpec
import ../specification_law.bend as SpecificationLaw
import ../reference.bend as Reference
import ../ordered_reference.bend as OrderedReference
import ./ordered_reference_refinement.bend as ReferenceLaws

def SpecificationLaw.ordered_reduction(products,acc):
  match products:
    case Nil{}:
      {==}
    case ConvSpec.Product{x,w} <> tail:
      SpecificationLaw.ordered_reduction(tail,(acc + x * w : F32))

def SpecificationLaw.job_values(jobs):
  match jobs:
    case Nil{}:
      {==}
    case ConvSpec.DotJob{+products,+bias} <> +tail:
      %SpecificationLaw.ordered_reduction(products,0.0) : {(Reference.sum(products,
            0.0) + bias : F32) <> Reference.evaluate(tail) == (_ + bias : F32) <> ConvSpec.evaluate(tail) : List<&2,F32>}
      %SpecificationLaw.job_values(tail) : {(Reference.sum(products,
            0.0) + bias : F32) <> Reference.evaluate(tail) == (Reference.sum(products,0.0) + bias : F32) <> _ : List<&2,F32>}
      {==}

law validated:
  for +ok: Bool
  for +p: ConvSpec.Problem
  {Reference.finish(ok,p) == ConvSpec.finish(ok,p) : ConvSpec.ConvOutcome}

def validated(ok,p):
  match ok:
    case False{}:
      {==}
    case True{}:
      %SpecificationLaw.job_values(ConvSpec.jobs(p)) : {ConvSpec.Accepted{ConvSpec.out_height(p),ConvSpec.out_width(p),
          OrderedReference.values(p)} == ConvSpec.Accepted{ConvSpec.out_height(p),ConvSpec.out_width(p),_} : ConvSpec.ConvOutcome}
      %ReferenceLaws.output_channels(ConvSpec.outputs(p),0n,p) : {ConvSpec.Accepted{ConvSpec.out_height(p),ConvSpec.out_width(p),
          OrderedReference.values(p)} == ConvSpec.Accepted{ConvSpec.out_height(p),ConvSpec.out_width(p),_} : ConvSpec.ConvOutcome}
      {==}

def SpecificationLaw.convolution(p):
  validated(ConvSpec.valid(p),p)
