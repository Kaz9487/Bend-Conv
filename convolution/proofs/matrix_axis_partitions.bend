# Instantiate the generic row-split proof on the convolution's actual axes.
# Column and reduction split are also instantiated in PROOF.bend.
import Base
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ../matrix_specification.bend as MatrixSpec
import ../specification.bend as ConvSpec
import ../../lib/proofs/ordered_reduction_split.bend as Ordered

law output_channel_tiles:
  for +a: List<&2,Nat>
  for +b: List<&2,Nat>
  for +ks: List<&2,MatrixSpec.Reduction>
  for +cs: List<&2,MatrixSpec.Pixel>
  for +p: ConvSpec.Problem
  {OrderedMatrix.matmul(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,~ConvSpec.bias_at,
      OrderedMatrix.cat(Nat,a,b),ks,cs,p) == OrderedMatrix.cat(F32,OrderedMatrix.matmul(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,
        ~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,~ConvSpec.bias_at,a,ks,cs,p),OrderedMatrix.matmul(~Nat,~MatrixSpec.Reduction,
        ~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,~ConvSpec.bias_at,b,ks,cs,p)) : List<&2,F32>}
def output_channel_tiles(a,b,ks,cs,p):
  Ordered.row_split(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,
    ~ConvSpec.bias_at,a,b,ks,cs,p)
