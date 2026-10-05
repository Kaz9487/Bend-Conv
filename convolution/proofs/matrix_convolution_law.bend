# Obligations fixed before implementation of their proofs.
import Base
import ../../lib/ordered_matrix.bend as O
import ../matrix_specification.bend as C
import ../specification.bend as S

law convolution:
  for +p: S.Problem
  {C.infer(p) == S.reference(p) : S.ConvOutcome}
