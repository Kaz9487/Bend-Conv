# Conditional scalar-to-descriptor implications. The actual successful
# computations are premises; no unconditional caller acceptance is claimed.
import Base
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ./tensor_vector_view_refinement.bend as Vector
import ./tensor_shape_refinement.bend as ShapeLaws
import ./boolean_logic.bend as Boolean
import ./checked_word_unit_products.bend as UnitProducts
import ./tensor_band_combine_leaf.bend as BandCombineLeaf
import ./tensor_band_shape_refinement.bend as BandShape
import ../tensor_band.bend as Bands

def product_single(+start: U32,+count: U32,+output: U32,accepted: {Shapes.checked_multiply(start,count) == Some{output} : Maybe<&2,U32>}) ->
  {Shapes.product([count],Some{start}) == Some{output} : Maybe<&2,U32>}:
  %Equal.sym(Maybe<&2,U32>,Shapes.checked_multiply(start,count),Some{output},accepted) : {Shapes.product([],_) == Some{output} : Maybe<&2,U32>}
  {==}

def counted_when(+count: U32,empty: Bool,classified: {U32.is_zero(count) == empty : Bool},
  one_left: {Shapes.checked_multiply(1,count) == Some{count} : Maybe<&2,U32>}) ->
  {Shapes.count_nonempty(empty,[count]) == Some{count} : Maybe<&2,U32>}:
  match empty:
    case True{}:
      %Equal.sym(U32,count,0,ShapeLaws.native_equal(count,0,classified)) : {Some{0} == Some{_} : Maybe<&2,U32>}
      {==}
    case False{}: product_single(1,count,count,one_left)

def counted(+count: U32,one_left: {Shapes.checked_multiply(1,count) == Some{count} : Maybe<&2,U32>}) -> {Views.numel(Vector.vector(count)) == Some{count} : Maybe<&2,U32>}:
  %Equal.sym(Bool,Shapes.has_zero([count]),U32.is_zero(count),Vector.zero_flag(count)) :
    {Shapes.count_nonempty(_,[count]) == Some{count} : Maybe<&2,U32>}
  counted_when(count,U32.is_zero(count),{==},one_left)

def contiguous_when(+count: U32,empty: Bool,one_right: {Shapes.checked_multiply(count,1) == Some{count} : Maybe<&2,U32>}) ->
  {Views.contiguous_nonempty(empty,[Views.Axis{count,1}]) == True{} : Bool}:
  match empty:
    case True{}: {==}
    case False{}:
      %Equal.sym(Bool,U32.is_le(count,1) || True{},True{},Boolean.or_true_right(U32.is_le(count,1))) :
        {Views.has_count(Views.contiguous_product(_,count,1)) == True{} : Bool}
      %Equal.sym(Maybe<&2,U32>,Shapes.checked_multiply(count,1),Some{count},one_right) :
        {Views.has_count(_) == True{} : Bool}
      {==}

def contiguous(+count: U32,one_right: {Shapes.checked_multiply(count,1) == Some{count} : Maybe<&2,U32>}) -> {Views.is_contiguous(Vector.vector(count)) == True{} : Bool}:
  contiguous_when(count,Shapes.has_zero([count]),one_right)

def descriptor(+count: U32) -> Vector.Descriptor<count>:
  Vector.Descriptor{counted(count,UnitProducts.one_left(count)),contiguous(count,UnitProducts.one_right(count)),BandCombineLeaf.fits(count,UnitProducts.one_right((count - 1 : U32)))}

def from_count(+geometry: Bands.Geometry,+rows: U32,+length: Nat,evidence: BandShape.Count<geometry,rows,length>) -> Vector.Descriptor<Bands.local_count(geometry,rows)>:
  BandShape.Count{exact,allocatable} = evidence
  descriptor(Bands.local_count(geometry,rows))
