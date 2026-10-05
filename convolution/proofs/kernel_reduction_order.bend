import ../../lib/proofs/ordered_reduction_split.bend as Ordered
import ../../lib/proofs/storage_shape.bend as Shape
# The actual kernel tap list follows the frozen matrix K axis, in the same order.
# Address equality here includes U32 wrap semantics; memory bounds are separate.
import Base
import ../../lib/proofs/traversal_reduction_refinement.bend as ReductionLaws
import ../../lib/math_sequence.bend as Sequence
import ../matrix_specification.bend as MatrixSpec
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ./matrix_reduction_order.bend as MatrixOrder
import ./packed_reduction_samples.bend as ReductionSamples

# A view of the same storage through matrix keys; it performs no new packing.
def ReindexedContext() -> Data:
  Shape.Both<MatrixOrder.TapLayout,ReductionSamples.Context>

def context(storage: ReductionSamples.Context,kernel: Nat,input_base: Nat,weight_base: Nat) -> ReindexedContext():
  Shape.Both{MatrixOrder.TapLayout{kernel,input_base,weight_base},storage}

def matrix_input() -> ReindexedContext() -> Sequence.Index(8n) -> MatrixSpec.Reduction -> F32:
  Ordered.mapped_sample(~MatrixSpec.Reduction,~ReductionLaws.AddressPair,~Sequence.Index(8n),~ReductionSamples.Context,~MatrixOrder.TapLayout,~MatrixOrder.mapped_tap,~ReductionSamples.input)

def matrix_weight() -> ReindexedContext() -> Bool -> MatrixSpec.Reduction -> F32:
  Ordered.mapped_sample(~MatrixSpec.Reduction,~ReductionLaws.AddressPair,~Bool,~ReductionSamples.Context,~MatrixOrder.TapLayout,~MatrixOrder.mapped_tap,~ReductionSamples.weight)