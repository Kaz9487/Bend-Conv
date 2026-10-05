# Storage-independent extents. View derives this descriptor from its axes.
import Base
import ./machine_limits.bend as Machine
import ./math_checked_word.bend as CheckedWord

type Shape is Data:
  Shape{extents: List<&2,U32>}

def rank(shape: Shape) -> Nat:
  Shape{extents} = shape
  List.length(&2,U32,extents)

def extent(shape: Shape,axis: Nat) -> Maybe<&2,U32>:
  Shape{extents} = shape
  List.get(&2,U32,extents,axis)

def same_extents(left: List<&2,U32>,right: List<&2,U32>) -> Bool:
  match left right:
    case Nil{} Nil{}: True{}
    case Con{x,xs} Con{y,ys}: U32.is_eq(x,y) && same_extents(xs,ys)
    case left right: False{}

def same(left: Shape,right: Shape) -> Bool:
  Shape{xs} = left
  Shape{ys} = right
  same_extents(xs,ys)

def has_zero(extents: List<&2,U32>) -> Bool:
  match extents:
    case Nil{}: False{}
    case Con{x,xs}: U32.is_zero(x) || has_zero(xs)

def checked_multiply(left: U32,right: U32) -> Maybe<&2,U32>:
  CheckedWord.multiply(left,right)

def product(extents: List<&2,U32>,accumulator: Maybe<&2,U32>) -> Maybe<&2,U32>:
  match extents accumulator:
    case Nil{} result: result
    case Con{x,xs} Some{value}: product(xs,checked_multiply(value,x))
    case Con{x,xs} None{}: None{}

def count_nonempty(empty: Bool,extents: List<&2,U32>) -> Maybe<&2,U32>:
  match empty:
    case True{}: Some{0}
    case False{}: product(extents,Some{1})

# Scalar [] has one element. Any zero extent makes an empty shape, even when
# an earlier partial product would overflow. Nonempty overflow is rejected.
def numel(shape: Shape) -> Maybe<&2,U32>:
  Shape{+extents} = shape
  count_nonempty(has_zero(extents),extents)

# NumPy broadcasting of one axis: equal extents, or one of them is 1.
def broadcast_extent(same: Bool,left_one: Bool,right_one: Bool,left: U32,right: U32) -> Maybe<&2,U32>:
  match same left_one right_one:
    case True{} left_one right_one: Some{left}
    case False{} True{} right_one: Some{right}
    case False{} False{} True{}: Some{left}
    case False{} False{} False{}: None{}

def prepend_extent(extent: Maybe<&2,U32>,rest: Maybe<&2,List<&2,U32>>) -> Maybe<&2,List<&2,U32>>:
  match extent rest:
    case Some{extent} Some{rest}: Some{extent <> rest}
    case extent rest: None{}

def broadcast_same_rank(left: List<&2,U32>,right: List<&2,U32>) -> Maybe<&2,List<&2,U32>>:
  match left right:
    case Nil{} Nil{}: Some{[]}
    case Con{+x,xs} Con{+y,ys}:
      prepend_extent(broadcast_extent(U32.is_eq(x,y),U32.is_eq(x,1),U32.is_eq(y,1),x,y),broadcast_same_rank(xs,ys))
    case left right: None{}

# The extents with leading axes of extent 1 added up to rank.
def with_rank(+extents: List<&2,U32>,rank: Nat) -> List<&2,U32>:
  List.append(&2,U32,List.replicate(U32,Nat.sub(rank,List.length(&2,U32,extents)),1),extents)

# The common extents of two operands under NumPy broadcasting: the extents are
# aligned at the last axis, a missing axis counts as 1, and on each axis the
# extents are equal or one of them is 1.
def broadcast(+left: List<&2,U32>,+right: List<&2,U32>) -> Maybe<&2,List<&2,U32>>:
  broadcast_same_rank(with_rank(left,List.length(&2,U32,right)),with_rank(right,List.length(&2,U32,left)))

# The extents without one axis.
def without_axis(+extents: List<&2,U32>,+axis: Nat) -> List<&2,U32>:
  List.append(&2,U32,List.take(&2,U32,extents,axis),List.drop(&2,U32,extents,1n+axis))

def sum_when(valid: Bool,value: U32) -> Maybe<&2,U32>:
  match valid:
    case True{}: Some{value}
    case False{}: None{}

def checked_add(+left: U32,+right: U32) -> Maybe<&2,U32>:
  sum_when(U32.is_le(right,(4294967295 - left : U32)),(left + right : U32))

def checked_add_to(left: Maybe<&2,U32>,right: U32) -> Maybe<&2,U32>:
  match left:
    case Some{left}: checked_add(left,right)
    case None{}: None{}

# Each extent with before elements added in front and after elements behind.
# The three lists have one entry per axis and no sum may wrap.
def padded(extents: List<&2,U32>,before: List<&2,U32>,after: List<&2,U32>) -> Maybe<&2,List<&2,U32>>:
  match extents before after:
    case Nil{} Nil{} Nil{}: Some{[]}
    case Con{extent,extents} Con{front,before} Con{back,after}:
      prepend_extent(checked_add_to(checked_add(front,extent),back),padded(extents,before,after))
    case extents before after: None{}
