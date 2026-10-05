# Clip a leaf once, then copy each channel without per-element predicates.
import Base
import ../../lib/kernel_gather.bend as Gather
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/word_subtraction.bend as Subtraction
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ../../lib/proofs/tensor_view_bounds.bend as ViewBounds
import ../../lib/proofs/u32_extrema.bend as U32Extrema

def not_less(+left: Nat,+right: Nat,order: {Nat.is_lt(left,right) == False{} : Bool}) -> {Nat.is_le(right,left) == True{} : Bool}:
  match left right:
    case left 0n: Order.zero_le(left)
    case 0n 1n+right: Empty.absurd({Nat.is_le(1n+right,0n) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},order)))
    case 1n+left 1n+right: not_less(left,right,order)

def minimum_inside_when(+choose_left: Bool,+left: U32,+right: U32,+column: Nat,
  comparison: {U32.is_lt(left,right) == choose_left : Bool},+inside: {Nat.is_lt(column,U32.to_nat(Bool.pick(U32,choose_left,left,right))) == True{} : Bool}) ->
  Shape.Both<{Nat.is_lt(column,U32.to_nat(left)) == True{} : Bool},{Nat.is_lt(column,U32.to_nat(right)) == True{} : Bool}>:
  match choose_left:
    case True{}:
      Shape.Both{inside,Bounds.nat_trans(column,U32.to_nat(left),U32.to_nat(right),inside,Bounds.strict_le(U32.to_nat(left),U32.to_nat(right),Bounds.as_nat_lt(left,right,comparison)))}
    case False{}:
      Shape.Both{Bounds.nat_trans(column,U32.to_nat(right),U32.to_nat(left),inside,
        not_less(U32.to_nat(left),U32.to_nat(right),%Comparison.lt(left,right) : {_ == False{} : Bool} comparison)),inside}

def minimum_covers_when(choose_left: Bool,+left: U32,+right: U32,+column: Nat,
  in_left: {Nat.is_lt(column,U32.to_nat(left)) == True{} : Bool},in_right: {Nat.is_lt(column,U32.to_nat(right)) == True{} : Bool}) ->
  {Nat.is_lt(column,U32.to_nat(Bool.pick(U32,choose_left,left,right))) == True{} : Bool}:
  match choose_left:
    case True{}: in_left
    case False{}: in_right

def difference(+offset: U32,+spatial: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +bounded: {U32.is_le(spatial,capacity) == True{} : Bool},+inside: {U32.is_lt(offset,spatial) == True{} : Bool}) ->
  {U32.to_nat((spatial - offset : U32)) == Nat.sub(U32.to_nat(spatial),U32.to_nat(offset)) : Nat}:
  Arithmetic.subtract(spatial,offset,capacity,capacity_ok,Bounds.strict_le(U32.to_nat(offset),U32.to_nat(spatial),Bounds.as_nat_lt(offset,spatial,inside)),
    Order.transitive(Nat.sub(U32.to_nat(spatial),U32.to_nat(offset)),U32.to_nat(spatial),U32.to_nat(capacity),Order.sub_le(U32.to_nat(spatial),U32.to_nat(offset)),Bounds.as_nat_le(spatial,capacity,bounded)))

def restored(+offset: U32,+spatial: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  bounded: {U32.is_le(spatial,capacity) == True{} : Bool},+inside: {U32.is_lt(offset,spatial) == True{} : Bool}) ->
  {Nat.add(U32.to_nat(offset),U32.to_nat((spatial - offset : U32))) == U32.to_nat(spatial) : Nat}:
  %Equal.sym(Nat,U32.to_nat((spatial - offset : U32)),Nat.sub(U32.to_nat(spatial),U32.to_nat(offset)),difference(offset,spatial,capacity,capacity_ok,bounded,inside)) :
    {Nat.add(U32.to_nat(offset),_) == U32.to_nat(spatial) : Nat}
  %Equal.sym(Nat,Nat.add(U32.to_nat(offset),Nat.sub(U32.to_nat(spatial),U32.to_nat(offset))),Nat.add(Nat.sub(U32.to_nat(spatial),U32.to_nat(offset)),U32.to_nat(offset)),
    Order.commutative(U32.to_nat(offset),Nat.sub(U32.to_nat(spatial),U32.to_nat(offset)))) : {_ == U32.to_nat(spatial) : Nat}
  Subtraction.restore(U32.to_nat(spatial),U32.to_nat(offset),Bounds.strict_le(U32.to_nat(offset),U32.to_nat(spatial),Bounds.as_nat_lt(offset,spatial,inside)))

def inside_remainder(+offset: U32,+spatial: U32,+span: U32,+column: Nat,capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  bounded: {U32.is_le(spatial,capacity) == True{} : Bool},guard: {U32.is_lt(offset,spatial) == True{} : Bool},
  facts: Shape.Both<{Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},{Nat.is_lt(column,U32.to_nat((spatial - offset : U32))) == True{} : Bool}>) ->
  Shape.Both<{Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},{Nat.is_lt(Nat.add(U32.to_nat(offset),column),U32.to_nat(spatial)) == True{} : Bool}>:
  Shape.Both{in_span,in_remainder} = facts
  Shape.Both{in_span,
    %restored(offset,spatial,capacity,capacity_ok,bounded,guard) : {Nat.is_lt(Nat.add(U32.to_nat(offset),column),_) == True{} : Bool}
    Order.offset_lt(U32.to_nat(offset),column,U32.to_nat((spatial - offset : U32)),in_remainder)}

def inside_when(+valid: Bool,+offset: U32,+spatial: U32,+span: U32,+column: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  bounded: {U32.is_le(spatial,capacity) == True{} : Bool},guard: {U32.is_lt(offset,spatial) == valid : Bool},
  inside: {Nat.is_lt(column,U32.to_nat(Gather.gather_width_when(valid,offset,spatial,span))) == True{} : Bool}) ->
  Shape.Both<{Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},{Nat.is_lt(Nat.add(U32.to_nat(offset),column),U32.to_nat(spatial)) == True{} : Bool}>:
  match valid:
    case False{}:
      Empty.absurd(Shape.Both<{Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},{Nat.is_lt(Nat.add(U32.to_nat(offset),column),U32.to_nat(spatial)) == True{} : Bool}>,
        WordEncoding.false_true(%ViewBounds.less_zero_false(column) : {_ == True{} : Bool} inside))
    case True{}:
      inside_remainder(offset,spatial,span,column,capacity,capacity_ok,bounded,guard,
        minimum_inside_when(U32.is_lt(span,(spatial - offset : U32)),span,(spatial - offset : U32),column,{==},
          %U32Extrema.minimum_pick(span,(spatial - offset : U32)) : {Nat.is_lt(column,U32.to_nat(_)) == True{} : Bool}
          inside))

def inside(+offset: U32,+spatial: U32,+span: U32,+column: Nat,capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  bounded: {U32.is_le(spatial,capacity) == True{} : Bool},inside: {Nat.is_lt(column,U32.to_nat(Gather.gather_width(offset,spatial,span))) == True{} : Bool}) ->
  Shape.Both<{Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},{Nat.is_lt(Nat.add(U32.to_nat(offset),column),U32.to_nat(spatial)) == True{} : Bool}>:
  inside_when(U32.is_lt(offset,spatial),offset,spatial,span,column,capacity,capacity_ok,bounded,{==},inside)

def covers(+offset: U32,+spatial: U32,+span: U32,+column: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  bounded: {U32.is_le(spatial,capacity) == True{} : Bool},in_span: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},
  +in_output: {Nat.is_lt(Nat.add(U32.to_nat(offset),column),U32.to_nat(spatial)) == True{} : Bool}) ->
  {Nat.is_lt(column,U32.to_nat(Gather.gather_width(offset,spatial,span))) == True{} : Bool}:
  +guard = {%Equal.sym(Bool,U32.is_lt(offset,spatial),Nat.is_lt(U32.to_nat(offset),U32.to_nat(spatial)),Comparison.lt(offset,spatial)) : {_ == True{} : Bool}
    Order.le_lt_trans(U32.to_nat(offset),Nat.add(U32.to_nat(offset),column),U32.to_nat(spatial),Order.le_add(U32.to_nat(offset),column),in_output) : {U32.is_lt(offset,spatial) == True{} : Bool}}
  %Equal.sym(Bool,U32.is_lt(offset,spatial),True{},guard) : {Nat.is_lt(column,U32.to_nat(Gather.gather_width_when(_,offset,spatial,span))) == True{} : Bool}
  %U32Extrema.pick_minimum(span,(spatial - offset : U32)) : {Nat.is_lt(column,U32.to_nat(_)) == True{} : Bool}
  minimum_covers_when(U32.is_lt(span,(spatial - offset : U32)),span,(spatial - offset : U32),column,in_span,
    Order.cancel_offset_lt(U32.to_nat(offset),column,U32.to_nat((spatial - offset : U32)),
      %Equal.sym(Nat,Nat.add(U32.to_nat(offset),U32.to_nat((spatial - offset : U32))),U32.to_nat(spatial),restored(offset,spatial,capacity,capacity_ok,bounded,guard)) :
        {Nat.is_lt(Nat.add(U32.to_nat(offset),column),_) == True{} : Bool}
      in_output))
