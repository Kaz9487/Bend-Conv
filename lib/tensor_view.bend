# Rank-independent forward-strided metadata. The Array owner is passed separately.
# A vector, matrix and tensor use the same descriptor and lookup operation.
import Base
import ./storage_buffer.bend as Storage
import ./tensor_shape.bend as Shapes
import ./machine_limits.bend as Machine

type Axis is Data:
  Axis{extent: U32,stride: U32}

type View is Data:
  View{axes: List<&2,Axis>,offset: U32}

# Successful addresses are below logical_length. The checked product and the
# remaining-room comparison prevent multiplication and addition from wrapping.
def address_when(valid: Bool,offset: U32,increment: U32) -> Maybe<&2,U32>:
  match valid:
    case False{}: None{}
    case True{}: Some{(offset + increment : U32)}

def finish_address(valid: Bool,offset: U32) -> Maybe<&2,U32>:
  match valid:
    case False{}: None{}
    case True{}: Some{offset}

def advance_product(product: Maybe<&2,U32>,offset: U32,room: U32) -> Maybe<&2,U32>:
  match product:
    case None{}: None{}
    case Some{+increment}: address_when(U32.is_le(increment,room),offset,increment)

def advance_inside(inside: Bool,+offset: U32,coordinate: U32,+stride: U32,logical_length: U32) -> Maybe<&2,U32>:
  match inside:
    case False{}: None{}
    case True{}: advance_product(Shapes.checked_multiply(coordinate,stride),offset,(logical_length - 1 - offset : U32))

def advance(+offset: U32,coordinate: U32,stride: U32,+logical_length: U32) -> Maybe<&2,U32>:
  advance_inside(U32.is_lt(offset,logical_length),offset,coordinate,stride,logical_length)

# The recursive step consumes one axis and one coordinate; no flattening/copy.
def locate(axes: List<&2,Axis>,coordinates: List<&2,U32>,address: Maybe<&2,U32>,+logical_length: U32) -> Maybe<&2,U32>:
  match axes coordinates address:
    case Nil{} Nil{} Some{+offset}:
      finish_address(U32.is_lt(offset,logical_length),offset)
    case Con{Axis{+extent,stride},tail} Con{+coordinate,rest} Some{+offset}:
      locate(tail,rest,advance_inside(U32.is_lt(coordinate,extent) && U32.is_lt(offset,logical_length),offset,coordinate,stride,logical_length),logical_length)
    case axes coordinates address: None{}

def address(view: View,coordinates: List<&2,U32>,logical_length: U32) -> Maybe<&2,U32>:
  View{axes,offset} = view
  locate(axes,coordinates,Some{offset},logical_length)

def axis_extents(axes: List<&2,Axis>) -> List<&2,U32>:
  match axes:
    case Nil{}: Nil{}
    case Con{Axis{extent,stride},tail}: extent <> axis_extents(tail)

def shape(view: View) -> Shapes.Shape:
  View{axes,offset} = view
  Shapes.Shape{axis_extents(axes)}

def numel(view: View) -> Maybe<&2,U32>:
  Shapes.numel(shape(view))

def rank(view: View) -> Nat:
  Shapes.rank(shape(view))

def extent(view: View,axis: Nat) -> Maybe<&2,U32>:
  Shapes.extent(shape(view),axis)

def build_axes(reversed: List<&2,U32>,stride: Maybe<&2,U32>,axes: List<&2,Axis>,offset: U32) -> Maybe<&2,View>:
  match reversed stride:
    case Nil{} Some{value}: Some{View{axes,offset}}
    case Con{+size,tail} Some{+value}:
      build_axes(tail,Shapes.checked_multiply(value,size),Axis{size,value} <> axes,offset)
    case reversed None{}: None{}

def contiguous_count(count: Maybe<&2,U32>,extents: List<&2,U32>,offset: U32) -> Maybe<&2,View>:
  match count:
    case None{}: None{}
    case Some{+value}:
      build_axes(List.reverse(&2,U32,extents),Some{Bool.pick(U32,U32.is_zero(value),0,1)},Nil{},offset)

def contiguous(+dimensions: Shapes.Shape,offset: U32) -> Maybe<&2,View>:
  Shapes.Shape{extents} = dimensions
  contiguous_count(Shapes.numel(dimensions),extents,offset)

def contiguous_product(valid: Bool,extent: U32,inner_count: U32) -> Maybe<&2,U32>:
  match valid:
    case False{}: None{}
    case True{}: Shapes.checked_multiply(extent,inner_count)

def contiguous_axis(+extent: U32,stride: U32,inner: Maybe<&2,U32>) -> Maybe<&2,U32>:
  match inner:
    case None{}: None{}
    case Some{+count}: contiguous_product(U32.is_le(extent,1) || U32.is_eq(stride,count),extent,count)

# Validate strides from the innermost axis outward. This avoids allocating a
# second canonical View just to compare its metadata. Empty views bypass it.
def contiguous_size(axes: List<&2,Axis>) -> Maybe<&2,U32>:
  match axes:
    case []: Some{1}
    case Axis{extent,stride} <> tail: contiguous_axis(extent,stride,contiguous_size(tail))

def has_count(count: Maybe<&2,U32>) -> Bool:
  match count:
    case None{}: False{}
    case Some{value}: True{}

def contiguous_nonempty(empty: Bool,+axes: List<&2,Axis>) -> Bool:
  match empty:
    case True{}: True{}
    case False{}: has_count(contiguous_size(axes))

def is_contiguous(view: View) -> Bool:
  View{+axes,offset} = view
  contiguous_nonempty(Shapes.has_zero(axis_extents(axes)),axes)

def last_coordinates(axes: List<&2,Axis>) -> List<&2,U32>:
  match axes:
    case Nil{}: Nil{}
    case Con{Axis{size,stride},tail}: (size - 1 : U32) <> last_coordinates(tail)

def found(address: Maybe<&2,U32>) -> Bool:
  match address:
    case None{}: False{}
    case Some{value}: True{}

def fits_nonempty(empty: Bool,+view: View,+logical_length: U32) -> Bool:
  match empty:
    case True{}:
      View{axes,offset} = view
      U32.is_le(offset,logical_length)
    case False{}:
      View{+axes,offset} = view
      found(address(View{axes,offset},last_coordinates(axes),logical_length))

# Forward strides are nonnegative: the last coordinate is the maximum address.
# Empty views may point exactly one past the logical end, and perform no reads.
def axis_extents_of(view: View) -> List<&2,U32>:
  View{axes,offset} = view
  axis_extents(axes)

def fits(view: View,logical_length: U32) -> Bool:
  +view = view
  fits_nonempty(Shapes.has_zero(axis_extents_of(view)),view,logical_length)

def contains_index(indices: List<&2,Nat>,+axis: Nat) -> Bool:
  match indices:
    case Nil{}: False{}
    case Con{index,tail}: Nat.is_eq(index,axis) || contains_index(tail,axis)

def unique_axes(order: List<&2,Nat>) -> Bool:
  match order:
    case Nil{}: True{}
    case Con{axis,+tail}: Bool.not(contains_index(tail,axis)) && unique_axes(tail)

def prepend_axis(axis: Maybe<&2,Axis>,view: Maybe<&2,View>) -> Maybe<&2,View>:
  match axis view:
    case Some{item} Some{View{axes,offset}}: Some{View{item <> axes,offset}}
    case axis view: None{}

def reorder(order: List<&2,Nat>,+axes: List<&2,Axis>,offset: U32) -> Maybe<&2,View>:
  match order:
    case Nil{}: Some{View{Nil{},offset}}
    case Con{index,tail}:
      prepend_axis(List.get(&2,Axis,axes,index),reorder(tail,axes,offset))

def permute_when(valid: Bool,view: View,order: List<&2,Nat>) -> Maybe<&2,View>:
  match valid:
    case False{}: None{}
    case True{}:
      View{axes,offset} = view
      reorder(order,axes,offset)

# Order must contain each axis exactly once, including singleton axes.
def permute(+view: View,+order: List<&2,Nat>) -> Maybe<&2,View>:
  permute_when(Nat.is_eq(rank(view),List.length(&2,Nat,order)) && unique_axes(order),view,order)

def slice_step(zero: Bool,+start: U32,+count: U32,+step: U32,+size: U32) -> Bool:
  match zero:
    case True{}: False{}
    case False{}:
      U32.is_le(start,size) && (U32.is_zero(count) ||
        (U32.is_lt(start,size) && U32.is_le((count - 1 : U32),((size - 1 - start) / step : U32))))

def sliced_axis(valid: Bool,count: U32,stride: Maybe<&2,U32>,offset: Maybe<&2,U32>,tail: List<&2,Axis>) -> Maybe<&2,View>:
  match valid stride offset:
    case True{} Some{stride} Some{offset}: Some{View{Axis{count,stride} <> tail,offset}}
    case valid stride offset: None{}

def add_offset(shift: Maybe<&2,U32>,+offset: U32) -> Maybe<&2,U32>:
  match shift:
    case None{}: None{}
    case Some{+value}: address_when(U32.is_le(value,(Machine.word_max() - offset : U32)),offset,value)

def slice_axes(axes: List<&2,Axis>,axis: Nat,+start: U32,+count: U32,+step: U32,+offset: U32) -> Maybe<&2,View>:
  match axes axis:
    case Con{Axis{size,+stride},tail} 0n:
      sliced_axis(slice_step(U32.is_zero(step),start,count,step,size),count,Shapes.checked_multiply(stride,step),
        add_offset(Shapes.checked_multiply(start,stride),offset),tail)
    case Con{head,tail} 1n+rest:
      prepend_axis(Some{head},slice_axes(tail,rest,start,count,step,offset))
    case Nil{} axis: None{}

# Explicit start/count/positive step. The resulting view is validated against
# the actual Buffer at execution time; arithmetic overflow is rejected here.
def slice(view: View,axis: Nat,start: U32,count: U32,step: U32) -> Maybe<&2,View>:
  View{axes,offset} = view
  slice_axes(axes,axis,start,count,step,offset)

# One axis read as extent elements: unchanged when the extents agree, the
# single element repeated (stride 0) when the axis has extent 1.
def stretched_axis(same: Bool,single: Bool,size: U32,stride: U32,extent: U32) -> Maybe<&2,Axis>:
  match same single:
    case True{} single: Some{Axis{size,stride}}
    case False{} True{}: Some{Axis{extent,0}}
    case False{} False{}: None{}

def stretched_prepend(axis: Maybe<&2,Axis>,rest: Maybe<&2,List<&2,Axis>>) -> Maybe<&2,List<&2,Axis>>:
  match axis rest:
    case Some{axis} Some{rest}: Some{axis <> rest}
    case axis rest: None{}

def stretch_axes(axes: List<&2,Axis>,target: List<&2,U32>) -> Maybe<&2,List<&2,Axis>>:
  match axes target:
    case Nil{} Nil{}: Some{[]}
    case Con{Axis{+size,stride},tail} Con{+extent,rest}:
      stretched_prepend(stretched_axis(U32.is_eq(size,extent),U32.is_eq(size,1),size,stride,extent),stretch_axes(tail,rest))
    case axes target: None{}

# The axes with leading axes of extent 1 added up to rank; the same elements.
def with_rank(+axes: List<&2,Axis>,rank: Nat) -> List<&2,Axis>:
  List.append(&2,Axis,List.replicate(Axis,Nat.sub(rank,List.length(&2,Axis,axes)),Axis{1,0}),axes)

def stretched_view(axes: Maybe<&2,List<&2,Axis>>,offset: U32) -> Maybe<&2,View>:
  match axes:
    case Some{axes}: Some{View{axes,offset}}
    case None{}: None{}

# The view read with the extents target under NumPy broadcasting. No element
# moves: an axis of extent 1 is repeated with stride 0, and so is an axis the
# view lacks.
def stretch(view: View,+target: List<&2,U32>) -> Maybe<&2,View>:
  View{axes,offset} = view
  stretched_view(stretch_axes(with_rank(axes,List.length(&2,U32,target)),target),offset)
