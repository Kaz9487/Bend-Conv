# Successful nonempty shape counts equal the unbounded product of extents.
import Base
import ../tensor_shape.bend as Shapes
import ./checked_word_refinement.bend as Checked
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./boolean_logic.bend as Boolean
import ./nat_order.bend as Order
import ./u32_comparison.bend as Comparison
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_encoding.bend as WordEncoding

def elements(extents: List<&2,U32>) -> Nat:
  match extents:
    case []: 1n
    case extent <> tail: Nat.mul(U32.to_nat(extent),elements(tail))

law Count:
  for result: Maybe<&2,U32>
  for expected: Nat
  Data

def Count(result,expected):
  match result:
    case None{}: Unit
    case Some{value}: {expected == U32.to_nat(value) : Nat}

def checked_count(+result: Maybe<&2,U32>,+left: U32,+right: U32,
  computed: {Shapes.checked_multiply(left,right) == result : Maybe<&2,U32>}) -> Count(result,Nat.mul(U32.to_nat(left),U32.to_nat(right))):
  match result:
    case None{}: Unit{}
    case Some{value}: Checked.multiply_sound(left,right,value,computed)

def times_one(+value: Nat) -> {Nat.mul(value,1n) == value : Nat}:
  match value:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(Nat,Nat,value => 1n+value,Nat.mul(rest,1n),rest,times_one(rest))

def product(+extents: List<&2,U32>,+accumulator: Maybe<&2,U32>,+expected: Nat,previous: Count(accumulator,expected)) ->
  Count(Shapes.product(extents,accumulator),Nat.mul(expected,elements(extents))):
  match extents accumulator:
    case Nil{} None{}: Unit{}
    case Nil{} Some{+value}:
      %Equal.sym(Nat,Nat.mul(expected,1n),expected,times_one(expected)) : {_ == U32.to_nat(value) : Nat}
      previous
    case Con{extent,tail} None{}: Unit{}
    case Con{+extent,+tail} Some{+value}:
      %Equal.sym(Nat,expected,U32.to_nat(value),previous) : Count(Shapes.product(extent <> tail,Some{value}),Nat.mul(_,elements(extent <> tail)))
      %RowMajor.assoc(U32.to_nat(value),U32.to_nat(extent),elements(tail)) : Count(Shapes.product(extent <> tail,Some{value}),_)
      product(tail,Shapes.checked_multiply(value,extent),Nat.mul(U32.to_nat(value),U32.to_nat(extent)),
        checked_count(Shapes.checked_multiply(value,extent),value,extent,{==}))

def nonempty_count(+extents: List<&2,U32>) -> Count(Shapes.count_nonempty(False{},extents),elements(extents)):
  %Addition.zero_right(elements(extents)) : Count(Shapes.product(extents,Some{1}),_)
  product(extents,Some{1},1n,{==})

def nonempty_numel(+extents: List<&2,U32>,nonempty: {Shapes.has_zero(extents) == False{} : Bool}) ->
  Count(Shapes.numel(Shapes.Shape{extents}),elements(extents)):
  %Equal.sym(Bool,Shapes.has_zero(extents),False{},nonempty) : Count(Shapes.count_nonempty(_,extents),elements(extents))
  nonempty_count(extents)

def native_equal(+left: U32,+right: U32,equal: {U32.is_eq(left,right) == True{} : Bool}) -> {left == right : U32}:
  %Arithmetic.native_roundtrip(left) : {_ == right : U32}
  %Arithmetic.native_roundtrip(right) : {U32.from_nat(U32.to_nat(left)) == _ : U32}
  Equal.cong(Nat,U32,value => U32.from_nat(value),U32.to_nat(left),U32.to_nat(right),Order.equal_of_is_eq(U32.to_nat(left),U32.to_nat(right),{
    %Comparison.eq(left,right) : {_ == True{} : Bool}
    equal : {Nat.is_eq(U32.to_nat(left),U32.to_nat(right)) == True{} : Bool}}))

def same_extents(+left: List<&2,U32>,+right: List<&2,U32>,same: {Shapes.same_extents(left,right) == True{} : Bool}) ->
  {left == right : List<&2,U32>}:
  match left right:
    case Nil{} Nil{}: {==}
    case Nil{} Con{head,tail}: Empty.absurd({Nil{} == head <> tail : List<&2,U32>},WordEncoding.false_true(same))
    case Con{head,tail} Nil{}: Empty.absurd({head <> tail == Nil{} : List<&2,U32>},WordEncoding.false_true(same))
    case Con{+head,+tail} Con{+other,+rest}:
      +same = same
      %native_equal(head,other,Boolean.left(U32.is_eq(head,other),Shapes.same_extents(tail,rest),same)) : {head <> tail == _ <> rest : List<&2,U32>}
      %same_extents(tail,rest,Boolean.right(U32.is_eq(head,other),Shapes.same_extents(tail,rest),same)) : {head <> tail == head <> _ : List<&2,U32>}
      {==}

def count_zero(+count: U32,equal: {Some{0} == Some{count} : Maybe<&2,U32>}) -> {U32.to_nat(count) == 0n : Nat}:
  Equal.sym(Nat,0n,U32.to_nat(count),Equal.cong(U32,Nat,value => U32.to_nat(value),0,count,
    Equal.cong(Maybe<&2,U32>,U32,result => Checked.value(result),Some{0},Some{count},equal)))

def empty_product_selected(zero: Bool,+head: U32,+tail: List<&2,U32>,comparison: {U32.is_zero(head) == zero : Bool},empty: {Shapes.has_zero(head <> tail) == True{} : Bool},
  tail_case: {Shapes.has_zero(tail) == True{} : Bool} -> {elements(tail) == 0n : Nat}) -> {elements(head <> tail) == 0n : Nat}:
  match zero:
    case True{}:
      %Equal.sym(U32,head,0,native_equal(head,0,comparison)) : {elements(_ <> tail) == 0n : Nat}
      {==}
    case False{}:
      %Equal.sym(Nat,elements(tail),0n,tail_case({
        %comparison : {_ || Shapes.has_zero(tail) == True{} : Bool}
        empty : {Shapes.has_zero(tail) == True{} : Bool}})) : {Nat.mul(U32.to_nat(head),_) == 0n : Nat}
      RowMajor.multiply_zero_right(U32.to_nat(head))

def empty_product(+extents: List<&2,U32>,empty: {Shapes.has_zero(extents) == True{} : Bool}) -> {elements(extents) == 0n : Nat}:
  match extents:
    case Nil{}: Empty.absurd({1n == 0n : Nat},WordEncoding.false_true(empty))
    case +head <> +tail: empty_product_selected(U32.is_zero(head),head,tail,{==},empty,proof => empty_product(tail,proof))

def numel_selected(empty: Bool,+extents: List<&2,U32>,comparison: {Shapes.has_zero(extents) == empty : Bool}) -> Count(Shapes.count_nonempty(empty,extents),elements(extents)):
  match empty:
    case True{}: empty_product(extents,comparison)
    case False{}: nonempty_count(extents)

def numel(+extents: List<&2,U32>) -> Count(Shapes.numel(Shapes.Shape{extents}),elements(extents)):
  numel_selected(Shapes.has_zero(extents),extents,{==})
