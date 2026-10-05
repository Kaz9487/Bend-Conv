# Complete checked shape products from exact natural counts and native bounds.
# Zero extents remain accepted by numel/count_of. Raw prefix/suffix products
# use a positive whole product, so their checked intermediate products fit.
import Base
import ../tensor.bend as Tensors
import ../tensor_shape.bend as Shapes
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_refinement.bend as Count
import ./checked_word_completion.bend as Completion
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_full_range.bend as FullRange
import ./u32_comparison.bend as Comparison
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding

def positive_natural(+value: Nat,nonzero: {Nat.is_eq(value,0n) == False{} : Bool}) -> {Nat.is_lt(0n,value) == True{} : Bool}:
  match value:
    case 0n: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},nonzero)))
    case 1n+rest: {==}

def positive_word(+value: U32,nonzero: {U32.is_zero(value) == False{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(value)) == True{} : Bool}:
  positive_natural(U32.to_nat(value),%Comparison.eq(value,0) : {_ == False{} : Bool} nonzero)

def positive_product(+left: Nat,+right: Nat,left_positive: {Nat.is_lt(0n,left) == True{} : Bool},right_positive: {Nat.is_lt(0n,right) == True{} : Bool}) ->
  {Nat.is_lt(0n,Nat.mul(left,right)) == True{} : Bool}:
  match left right:
    case 0n right: Empty.absurd({Nat.is_lt(0n,Nat.mul(0n,right)) == True{} : Bool},WordEncoding.false_true(left_positive))
    case 1n+left 0n: Empty.absurd({Nat.is_lt(0n,Nat.mul(1n+left,0n)) == True{} : Bool},WordEncoding.false_true(right_positive))
    case 1n+left 1n+right: {==}

def positive_elements(+extents: List<&2,U32>,nonempty: {Shapes.has_zero(extents) == False{} : Bool}) ->
  {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}:
  match extents:
    case []: {==}
    case +head <> +tail:
      +nonempty = nonempty
      positive_product(U32.to_nat(head),ShapeLaws.elements(tail),
        positive_word(head,Boolean.or_false_left(U32.is_zero(head),Shapes.has_zero(tail),nonempty)),
        positive_elements(tail,Boolean.or_false_right(U32.is_zero(head),Shapes.has_zero(tail),nonempty)))

def checked_product(+left: U32,+right: U32,+limit: U32,
  +maximum: {limit == Machine.word_max() : U32},+room: {Nat.is_le(Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(limit)) == True{} : Bool}) ->
  {Shapes.checked_multiply(left,right) == Some{U32.mul(left,right)} : Maybe<&2,U32>}:
  +exact = FullRange.multiply(left,right,limit,room)
  Completion.actual(left,right,limit,U32.mul(left,right),Equal.sym(Nat,U32.to_nat(U32.mul(left,right)),Nat.mul(U32.to_nat(left),U32.to_nat(right)),exact),
    %Equal.sym(Bool,U32.is_le(U32.mul(left,right),limit),Nat.is_le(U32.to_nat(U32.mul(left,right)),U32.to_nat(limit)),Comparison.le(U32.mul(left,right),limit)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.mul(left,right)),Nat.mul(U32.to_nat(left),U32.to_nat(right)),exact) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool} room,
    maximum)

def product_complete(+extents: List<&2,U32>,+accumulator: U32,+limit: U32,+maximum: {limit == Machine.word_max() : U32},
  +nonempty: {Shapes.has_zero(extents) == False{} : Bool},+room: {Nat.is_le(Nat.mul(U32.to_nat(accumulator),ShapeLaws.elements(extents)),U32.to_nat(limit)) == True{} : Bool}) ->
  {Shapes.product(extents,Some{accumulator}) == Some{U32.from_nat(Nat.mul(U32.to_nat(accumulator),ShapeLaws.elements(extents)))} : Maybe<&2,U32>}:
  match extents:
    case []:
      %Equal.sym(Nat,Nat.mul(U32.to_nat(accumulator),1n),U32.to_nat(accumulator),ShapeLaws.times_one(U32.to_nat(accumulator))) : {Some{accumulator} == Some{U32.from_nat(_)} : Maybe<&2,U32>}
      Equal.cong(U32,Maybe<&2,U32>,value => Some{value},accumulator,U32.from_nat(U32.to_nat(accumulator)),Equal.sym(U32,U32.from_nat(U32.to_nat(accumulator)),accumulator,Arithmetic.native_roundtrip(accumulator)))
    case +head <> +tail:
      +tail_nonempty = Boolean.or_false_right(U32.is_zero(head),Shapes.has_zero(tail),nonempty)
      +prefix_room = Order.transitive(Nat.mul(U32.to_nat(accumulator),U32.to_nat(head)),Nat.mul(U32.to_nat(accumulator),ShapeLaws.elements(head <> tail)),U32.to_nat(limit),
        %RowMajor.assoc(U32.to_nat(accumulator),U32.to_nat(head),ShapeLaws.elements(tail)) : {Nat.is_le(Nat.mul(U32.to_nat(accumulator),U32.to_nat(head)),_) == True{} : Bool}
        RowMajor.product_covers(Nat.mul(U32.to_nat(accumulator),U32.to_nat(head)),ShapeLaws.elements(tail),positive_elements(tail,tail_nonempty)),room)
      +product_exact = FullRange.multiply(accumulator,head,limit,prefix_room)
      %Equal.sym(Maybe<&2,U32>,Shapes.checked_multiply(accumulator,head),Some{U32.mul(accumulator,head)},checked_product(accumulator,head,limit,maximum,prefix_room)) :
        {Shapes.product(tail,_) == Some{U32.from_nat(Nat.mul(U32.to_nat(accumulator),ShapeLaws.elements(head <> tail)))} : Maybe<&2,U32>}
      %RowMajor.assoc(U32.to_nat(accumulator),U32.to_nat(head),ShapeLaws.elements(tail)) :
        {Shapes.product(tail,Some{U32.mul(accumulator,head)}) == Some{U32.from_nat(_)} : Maybe<&2,U32>}
      %product_exact :
        {Shapes.product(tail,Some{U32.mul(accumulator,head)}) == Some{U32.from_nat(Nat.mul(_,ShapeLaws.elements(tail)))} : Maybe<&2,U32>}
      product_complete(tail,U32.mul(accumulator,head),limit,maximum,tail_nonempty,
        %Equal.sym(Nat,U32.to_nat(U32.mul(accumulator,head)),Nat.mul(U32.to_nat(accumulator),U32.to_nat(head)),product_exact) : {Nat.is_le(Nat.mul(_,ShapeLaws.elements(tail)),U32.to_nat(limit)) == True{} : Bool}
        %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(accumulator),U32.to_nat(head)),ShapeLaws.elements(tail)),Nat.mul(U32.to_nat(accumulator),Nat.mul(U32.to_nat(head),ShapeLaws.elements(tail))),RowMajor.assoc(U32.to_nat(accumulator),U32.to_nat(head),ShapeLaws.elements(tail))) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool} room)

def numel_when(+extents: List<&2,U32>,+count: U32,+limit: U32,+maximum: {limit == Machine.word_max() : U32},empty: Bool,
  classified: {Shapes.has_zero(extents) == empty : Bool},+exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat},+within: {U32.is_le(count,limit) == True{} : Bool}) ->
  {Shapes.count_nonempty(empty,extents) == Some{count} : Maybe<&2,U32>}:
  match empty:
    case True{}:
      +zero = Equal.trans(Nat,U32.to_nat(count),ShapeLaws.elements(extents),0n,Equal.sym(Nat,ShapeLaws.elements(extents),U32.to_nat(count),exact),ShapeLaws.empty_product(extents,classified))
      %Arithmetic.native_roundtrip(count) : {Some{0} == Some{_} : Maybe<&2,U32>}
      %Equal.sym(Nat,U32.to_nat(count),0n,zero) : {Some{0} == Some{U32.from_nat(_)} : Maybe<&2,U32>}
      {==}
    case False{}:
      %Arithmetic.native_roundtrip(count) : {Shapes.product(extents,Some{1}) == Some{_} : Maybe<&2,U32>}
      %exact : {Shapes.product(extents,Some{1}) == Some{U32.from_nat(_)} : Maybe<&2,U32>}
      %Addition.zero_right(ShapeLaws.elements(extents)) : {Shapes.product(extents,Some{1}) == Some{U32.from_nat(_)} : Maybe<&2,U32>}
      product_complete(extents,1,limit,maximum,classified,
        %Equal.sym(Nat,Nat.add(ShapeLaws.elements(extents),0n),ShapeLaws.elements(extents),Addition.zero_right(ShapeLaws.elements(extents))) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
        %Equal.sym(Nat,ShapeLaws.elements(extents),U32.to_nat(count),exact) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool} Bounds.as_nat_le(count,limit,within))

def numel_bounded(+extents: List<&2,U32>,+count: U32,+limit: U32,+maximum: {limit == Machine.word_max() : U32},
  +exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat},within: {U32.is_le(count,limit) == True{} : Bool}) ->
  {Shapes.numel(Shapes.Shape{extents}) == Some{count} : Maybe<&2,U32>}:
  numel_when(extents,count,limit,maximum,Shapes.has_zero(extents),{==},exact,within)

def numel(+extents: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat}) ->
  {Shapes.numel(Shapes.Shape{extents}) == Some{count} : Maybe<&2,U32>}:
  numel_bounded(extents,count,Machine.word_max(),{==},exact,FullRange.maximum(count))

def count_of_bounded(+extents: List<&2,U32>,+count: U32,+limit: U32,+maximum: {limit == Machine.word_max() : U32},
  within_word: {U32.is_le(count,limit) == True{} : Bool},+capacity: U32,capacity_actual: {capacity == Machine.capacity(Default.limits()) : U32},
  +exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat},within_capacity: {U32.is_le(count,capacity) == True{} : Bool}) ->
  {Tensors.count_of(extents) == Some{count} : Maybe<&2,U32>}:
  %Equal.sym(Maybe<&2,U32>,Shapes.numel(Shapes.Shape{extents}),Some{count},numel_bounded(extents,count,limit,maximum,exact,within_word)) : {Tensors.allocatable(_) == Some{count} : Maybe<&2,U32>}
  %capacity_actual : {Tensors.within(U32.is_le(count,_),count) == Some{count} : Maybe<&2,U32>}
  %Equal.sym(Bool,U32.is_le(count,capacity),True{},within_capacity) : {Tensors.within(_,count) == Some{count} : Maybe<&2,U32>}
  {==}

def count_of(+extents: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat},
  within: {U32.is_le(count,Machine.capacity(Default.limits())) == True{} : Bool}) -> {Tensors.count_of(extents) == Some{count} : Maybe<&2,U32>}:
  count_of_bounded(extents,count,Machine.word_max(),{==},FullRange.maximum(count),Machine.capacity(Default.limits()),{==},exact,within)

def positive_not_zero_when(+extents: List<&2,U32>,empty: Bool,classified: {Shapes.has_zero(extents) == empty : Bool},positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) -> {Shapes.has_zero(extents) == False{} : Bool}:
  match empty:
    case False{}: classified
    case True{}:
      Empty.absurd({Shapes.has_zero(extents) == False{} : Bool},WordEncoding.false_true(
        %ShapeLaws.empty_product(extents,classified) : {Nat.is_lt(0n,_) == True{} : Bool} positive))

def positive_not_zero(+extents: List<&2,U32>,positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) -> {Shapes.has_zero(extents) == False{} : Bool}:
  positive_not_zero_when(extents,Shapes.has_zero(extents),{==},positive)

def elements_split(+extents: List<&2,U32>,+axis: Nat) ->
  {ShapeLaws.elements(extents) == Nat.mul(ShapeLaws.elements(List.take(&2,U32,extents,axis)),ShapeLaws.elements(List.drop(&2,U32,extents,axis))) : Nat}:
  match extents axis:
    case [] 0n: {==}
    case [] 1n+axis: {==}
    case +head <> +tail 0n: Equal.sym(Nat,Nat.add(ShapeLaws.elements(head <> tail),0n),ShapeLaws.elements(head <> tail),Addition.zero_right(ShapeLaws.elements(head <> tail)))
    case +head <> +tail 1n+ +previous:
      Equal.trans(Nat,ShapeLaws.elements(head <> tail),Nat.mul(U32.to_nat(head),Nat.mul(ShapeLaws.elements(List.take(&2,U32,tail,previous)),ShapeLaws.elements(List.drop(&2,U32,tail,previous)))),
        Nat.mul(Nat.mul(U32.to_nat(head),ShapeLaws.elements(List.take(&2,U32,tail,previous))),ShapeLaws.elements(List.drop(&2,U32,tail,previous))),
        Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(head),value),ShapeLaws.elements(tail),Nat.mul(ShapeLaws.elements(List.take(&2,U32,tail,previous)),ShapeLaws.elements(List.drop(&2,U32,tail,previous))),elements_split(tail,previous)),
        Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(head),ShapeLaws.elements(List.take(&2,U32,tail,previous))),ShapeLaws.elements(List.drop(&2,U32,tail,previous))),
          Nat.mul(U32.to_nat(head),Nat.mul(ShapeLaws.elements(List.take(&2,U32,tail,previous)),ShapeLaws.elements(List.drop(&2,U32,tail,previous)))),RowMajor.assoc(U32.to_nat(head),ShapeLaws.elements(List.take(&2,U32,tail,previous)),ShapeLaws.elements(List.drop(&2,U32,tail,previous)))))

def prefix_positive(+extents: List<&2,U32>,+axis: Nat,positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  {Nat.is_lt(0n,ShapeLaws.elements(List.take(&2,U32,extents,axis))) == True{} : Bool}:
  RowMajor.left_factor_positive(ShapeLaws.elements(List.take(&2,U32,extents,axis)),ShapeLaws.elements(List.drop(&2,U32,extents,axis)),
    %elements_split(extents,axis) : {Nat.is_lt(0n,_) == True{} : Bool} positive)

def suffix_positive(+extents: List<&2,U32>,+axis: Nat,positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  {Nat.is_lt(0n,ShapeLaws.elements(List.drop(&2,U32,extents,axis))) == True{} : Bool}:
  RowMajor.left_factor_positive(ShapeLaws.elements(List.drop(&2,U32,extents,axis)),ShapeLaws.elements(List.take(&2,U32,extents,axis)),
    %RowMajor.multiply_commutative(ShapeLaws.elements(List.take(&2,U32,extents,axis)),ShapeLaws.elements(List.drop(&2,U32,extents,axis))) : {Nat.is_lt(0n,_) == True{} : Bool}
    %elements_split(extents,axis) : {Nat.is_lt(0n,_) == True{} : Bool} positive)

def prefix_room(+extents: List<&2,U32>,+axis: Nat,+limit: Nat,+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},room: {Nat.is_le(ShapeLaws.elements(extents),limit) == True{} : Bool}) ->
  {Nat.is_le(ShapeLaws.elements(List.take(&2,U32,extents,axis)),limit) == True{} : Bool}:
  Order.transitive(ShapeLaws.elements(List.take(&2,U32,extents,axis)),ShapeLaws.elements(extents),limit,
    %Equal.sym(Nat,ShapeLaws.elements(extents),Nat.mul(ShapeLaws.elements(List.take(&2,U32,extents,axis)),ShapeLaws.elements(List.drop(&2,U32,extents,axis))),elements_split(extents,axis)) : {Nat.is_le(ShapeLaws.elements(List.take(&2,U32,extents,axis)),_) == True{} : Bool}
    RowMajor.product_covers(ShapeLaws.elements(List.take(&2,U32,extents,axis)),ShapeLaws.elements(List.drop(&2,U32,extents,axis)),suffix_positive(extents,axis,positive)),room)

def suffix_room(+extents: List<&2,U32>,+axis: Nat,+limit: Nat,+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},room: {Nat.is_le(ShapeLaws.elements(extents),limit) == True{} : Bool}) ->
  {Nat.is_le(ShapeLaws.elements(List.drop(&2,U32,extents,axis)),limit) == True{} : Bool}:
  Order.transitive(ShapeLaws.elements(List.drop(&2,U32,extents,axis)),ShapeLaws.elements(extents),limit,
    %Equal.sym(Nat,ShapeLaws.elements(extents),Nat.mul(ShapeLaws.elements(List.take(&2,U32,extents,axis)),ShapeLaws.elements(List.drop(&2,U32,extents,axis))),elements_split(extents,axis)) : {Nat.is_le(ShapeLaws.elements(List.drop(&2,U32,extents,axis)),_) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(ShapeLaws.elements(List.take(&2,U32,extents,axis)),ShapeLaws.elements(List.drop(&2,U32,extents,axis))),Nat.mul(ShapeLaws.elements(List.drop(&2,U32,extents,axis)),ShapeLaws.elements(List.take(&2,U32,extents,axis))),RowMajor.multiply_commutative(ShapeLaws.elements(List.take(&2,U32,extents,axis)),ShapeLaws.elements(List.drop(&2,U32,extents,axis)))) : {Nat.is_le(ShapeLaws.elements(List.drop(&2,U32,extents,axis)),_) == True{} : Bool}
    RowMajor.product_covers(ShapeLaws.elements(List.drop(&2,U32,extents,axis)),ShapeLaws.elements(List.take(&2,U32,extents,axis)),prefix_positive(extents,axis,positive)),room)
