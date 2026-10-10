# Machine address bounds for balanced storage, independent of element values.
import Base
import ./storage_shape.bend as Shape
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding
import ./word_power_of_two.bend as Power
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_comparison.bend as Comparison
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_subtraction.bend as Subtraction

def capacity(depth: Nat) -> U32:
  U32.shln(1,depth)

def shift_extent(depth: Nat) -> {WordEncoding.shift(depth,1n) == Shape.extent(depth) : Nat}:
  match depth:
    case 0n: {==}
    case 1n+rest:
      Equal.cong(Nat,Nat,Nat.double,WordEncoding.shift(rest,1n),Shape.extent(rest),shift_extent(rest))

def capacity_value(+depth: Nat,limit: {Nat.is_le(depth,31n) == True{} : Bool}) ->
  {U32.to_nat(capacity(depth)) == Shape.extent(depth) : Nat}:
  %Equal.sym(U32,capacity(depth),U32{Power.power(32n,depth)},Power.native_power(depth)) : {U32.to_nat(_) == Shape.extent(depth) : Nat}
  Equal.trans(Nat,Word.to_nat(32n,Power.power(32n,depth)),WordEncoding.shift(depth,1n),Shape.extent(depth),
    Power.power_value(32n,depth,Order.lt_succ(depth,31n,limit)),shift_extent(depth))

def natural_bound(+depth: Nat,+index: U32,+limit: {Nat.is_le(depth,31n) == True{} : Bool},bound: {U32.is_lt(index,capacity(depth)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(index),Shape.extent(depth)) == True{} : Bool}:
  %capacity_value(depth,limit) : {Nat.is_lt(U32.to_nat(index),_) == True{} : Bool}
  Bounds.as_nat_lt(index,capacity(depth),bound)

def child_limit_full(+depth: Nat,limit: {Nat.is_le(1n+depth,31n) == True{} : Bool}) -> {Nat.is_le(depth,31n) == True{} : Bool}:
  Bounds.strict_le(depth,31n,Bounds.predecessor_strict(depth,31n,limit))

def right_order(+depth: Nat,+index: U32,choice: {U32.is_lt(index,capacity(depth)) == False{} : Bool}) ->
  {Nat.is_le(U32.to_nat(capacity(depth)),U32.to_nat(index)) == True{} : Bool}:
  Order.ge_true(U32.to_nat(index),U32.to_nat(capacity(depth)),Order.comparison_not_less(Nat.cmp(U32.to_nat(index),U32.to_nat(capacity(depth))),
    %Comparison.lt(index,capacity(depth)) : {_ == False{} : Bool}
    choice))

def sub_value(+depth: Nat,+index: U32,+limit: {Nat.is_le(1n+depth,31n) == True{} : Bool},+bound: {U32.is_lt(index,capacity(1n+depth)) == True{} : Bool},
  choice: {U32.is_lt(index,capacity(depth)) == False{} : Bool}) -> {U32.to_nat((index - capacity(depth) : U32)) == Nat.sub(U32.to_nat(index),U32.to_nat(capacity(depth))) : Nat}:
  Arithmetic.subtract(index,capacity(depth),capacity(1n+depth),Power.native_power_bound(1n+depth,limit),right_order(depth,index,choice),
    Order.transitive(Nat.sub(U32.to_nat(index),U32.to_nat(capacity(depth))),U32.to_nat(index),U32.to_nat(capacity(1n+depth)),
      Order.sub_le(U32.to_nat(index),U32.to_nat(capacity(depth))),Bounds.strict_le(U32.to_nat(index),U32.to_nat(capacity(1n+depth)),Bounds.as_nat_lt(index,capacity(1n+depth),bound))))

def right_restores_index_full(+depth: Nat,+index: U32,+limit: {Nat.is_le(1n+depth,31n) == True{} : Bool},+bound: {U32.is_lt(index,capacity(1n+depth)) == True{} : Bool},
  +choice: {U32.is_lt(index,capacity(depth)) == False{} : Bool}) -> {U32.to_nat(index) == Nat.add(Shape.extent(depth),U32.to_nat((index - capacity(depth) : U32))) : Nat}:
  %Equal.sym(Nat,U32.to_nat((index - capacity(depth) : U32)),Nat.sub(U32.to_nat(index),U32.to_nat(capacity(depth))),sub_value(depth,index,limit,bound,choice)) :
    {U32.to_nat(index) == Nat.add(Shape.extent(depth),_) : Nat}
  %capacity_value(depth,child_limit_full(depth,limit)) : {U32.to_nat(index) == Nat.add(_,Nat.sub(U32.to_nat(index),U32.to_nat(capacity(depth)))) : Nat}
  Equal.sym(Nat,Nat.add(U32.to_nat(capacity(depth)),Nat.sub(U32.to_nat(index),U32.to_nat(capacity(depth)))),U32.to_nat(index),
    Equal.trans(Nat,Nat.add(U32.to_nat(capacity(depth)),Nat.sub(U32.to_nat(index),U32.to_nat(capacity(depth)))),
      Nat.add(Nat.sub(U32.to_nat(index),U32.to_nat(capacity(depth))),U32.to_nat(capacity(depth))),U32.to_nat(index),
      Order.commutative(U32.to_nat(capacity(depth)),Nat.sub(U32.to_nat(index),U32.to_nat(capacity(depth)))),Subtraction.restore(U32.to_nat(index),U32.to_nat(capacity(depth)),right_order(depth,index,choice))))

def right_stays_in_child_full(+depth: Nat,+index: U32,+limit: {Nat.is_le(1n+depth,31n) == True{} : Bool},+bound: {U32.is_lt(index,capacity(1n+depth)) == True{} : Bool},
  +choice: {U32.is_lt(index,capacity(depth)) == False{} : Bool}) -> {U32.is_lt((index - capacity(depth) : U32),capacity(depth)) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_lt((index - capacity(depth) : U32),capacity(depth)),Nat.is_lt(U32.to_nat((index - capacity(depth) : U32)),U32.to_nat(capacity(depth))),Comparison.lt((index - capacity(depth) : U32),capacity(depth))) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((index - capacity(depth) : U32)),Nat.sub(U32.to_nat(index),U32.to_nat(capacity(depth))),sub_value(depth,index,limit,bound,choice)) :
    {Nat.is_lt(_,U32.to_nat(capacity(depth))) == True{} : Bool}
  Order.sub_lt(U32.to_nat(index),U32.to_nat(capacity(depth)),U32.to_nat(capacity(depth)),right_order(depth,index,choice),
    %Equal.sym(Nat,U32.to_nat(capacity(depth)),Shape.extent(depth),capacity_value(depth,child_limit_full(depth,limit))) :
      {Nat.is_lt(U32.to_nat(index),Nat.add(_,_)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Shape.extent(depth),Shape.extent(depth)),Nat.double(Shape.extent(depth)),Addition.double(Shape.extent(depth))) : {Nat.is_lt(U32.to_nat(index),_) == True{} : Bool}
    natural_bound(1n+depth,index,limit,bound))

def below_one_value(+index: U32,bound: {U32.is_lt(index,1) == True{} : Bool}) -> {0n == U32.to_nat(index) : Nat}:
  U32{+word} = index
  %Equal.sym(Word(32n),word,Word.zero(32n),WordEncoding.below_one_zero(31n,word,bound)) : {0n == Word.to_nat(32n,_) : Nat}
  {==}

def equal_values(+left: U32,+right: U32,equation: {U32.to_nat(left) == U32.to_nat(right) : Nat}) -> {left == right : U32}:
  %Arithmetic.native_roundtrip(left) : {_ == right : U32}
  %Arithmetic.native_roundtrip(right) : {U32.from_nat(U32.to_nat(left)) == _ : U32}
  Equal.cong(Nat,U32,U32.from_nat,U32.to_nat(left),U32.to_nat(right),equation)

def equal_words(+left: U32,+right: U32,equation: {U32.is_eq(left,right) == True{} : Bool}) -> {left == right : U32}:
  equal_values(left,right,Order.equal_of_is_eq(U32.to_nat(left),U32.to_nat(right),
    %Comparison.eq(left,right) : {_ == True{} : Bool}
    equation))

def unequal_words(+left: U32,+right: U32,different: {U32.is_eq(left,right) == False{} : Bool},equal: {left == right : U32}) -> Empty:
  WordEncoding.false_true(Equal.trans(Bool,False{},U32.is_eq(left,right),True{},Equal.sym(Bool,U32.is_eq(left,right),False{},different),
    %equal : {U32.is_eq(left,_) == True{} : Bool}
    %Equal.sym(Bool,U32.is_eq(left,left),Nat.is_eq(U32.to_nat(left),U32.to_nat(left)),Comparison.eq(left,left)) : {_ == True{} : Bool}
    Order.equal_reflexive(U32.to_nat(left))))

def subtract_compared(+left: U32,+right: U32,+offset: U32,equal: Bool,
  compared: {U32.is_eq((left - offset : U32),(right - offset : U32)) == equal : Bool},different: {U32.is_eq(left,right) == False{} : Bool}) -> {equal == False{} : Bool}:
  match equal:
    case False{}: {==}
    case True{}:
      Empty.absurd({True{} == False{} : Bool},unequal_words(left,right,different,
        Equal.trans(U32,left,((left - offset : U32) + offset : U32),right,
          Equal.sym(U32,((left - offset : U32) + offset : U32),left,Subtraction.add_subtract(left,offset)),
          Equal.trans(U32,((left - offset : U32) + offset : U32),((right - offset : U32) + offset : U32),right,
            Equal.cong(U32,U32,value => (value + offset : U32),(left - offset : U32),(right - offset : U32),equal_words((left - offset : U32),(right - offset : U32),compared)),
            Subtraction.add_subtract(right,offset)))))

def subtract_distinct(+left: U32,+right: U32,+offset: U32,different: {U32.is_eq(left,right) == False{} : Bool}) ->
  {U32.is_eq((left - offset : U32),(right - offset : U32)) == False{} : Bool}:
  subtract_compared(left,right,offset,U32.is_eq((left - offset : U32),(right - offset : U32)),{==},different)

# Existing allocation proofs use the smaller, 30-bit allocation limit.
def child_limit(+depth: Nat,limit: {Nat.is_le(1n+depth,30n) == True{} : Bool}) -> {Nat.is_le(depth,30n) == True{} : Bool}:
  Bounds.strict_le(depth,30n,Bounds.predecessor_strict(depth,30n,limit))

def right_restores_index(+depth: Nat,+index: U32,+limit: {Nat.is_le(1n+depth,30n) == True{} : Bool},+bound: {U32.is_lt(index,capacity(1n+depth)) == True{} : Bool},
  +choice: {U32.is_lt(index,capacity(depth)) == False{} : Bool}) -> {U32.to_nat(index) == Nat.add(Shape.extent(depth),U32.to_nat((index - capacity(depth) : U32))) : Nat}:
  right_restores_index_full(depth,index,Order.successor(1n+depth,30n,limit),bound,choice)

def right_stays_in_child(+depth: Nat,+index: U32,+limit: {Nat.is_le(1n+depth,30n) == True{} : Bool},+bound: {U32.is_lt(index,capacity(1n+depth)) == True{} : Bool},
  +choice: {U32.is_lt(index,capacity(depth)) == False{} : Bool}) -> {U32.is_lt((index - capacity(depth) : U32),capacity(depth)) == True{} : Bool}:
  right_stays_in_child_full(depth,index,Order.successor(1n+depth,30n,limit),bound,choice)
