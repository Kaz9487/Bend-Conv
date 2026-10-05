# Actual one-axis descriptor contract. Successful computation evidence remains
# explicit; the scalar producer obligations belong to descriptor refinement.
import Base
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes

def vector(count: U32) -> Views.View:
  Views.View{[Views.Axis{count,1}],0}

def or_false(+value: Bool) -> {value || False{} == value : Bool}:
  match value:
    case False{}: {==}
    case True{}: {==}

def zero_flag(+count: U32) -> {Shapes.has_zero([count]) == U32.is_zero(count) : Bool}:
  or_false(U32.is_zero(count))

type Descriptor<-count: U32> is Data:
  Descriptor{counted: {Views.numel(vector(count)) == Some{count} : Maybe<&2,U32>},
    contiguous: {Views.is_contiguous(vector(count)) == True{} : Bool},
    fits: {Views.fits(vector(count),count) == True{} : Bool}}
