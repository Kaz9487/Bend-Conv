# Checked multiplication is exactly a bounded unbounded-natural product.
import Base
import ../math_checked_word.bend as CheckedWord
import ../machine_limits.bend as Machine
import ./nat_order.bend as Order
import ./word_encoding.bend as WordEncoding
import ./natural_addition.bend as Addition
import ./binary_division.bend as BinaryDivision
import ./u32_comparison.bend as Comparison
import ./u32_full_range.bend as FullRange
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor

def outside_monotone(+small: Nat,+large: Nat,+limit: Nat,
  ordered: {Nat.is_le(small,large) == True{} : Bool},outside: {Nat.is_le(small,limit) == False{} : Bool}) ->
  {Nat.is_le(large,limit) == False{} : Bool}:
  match small large limit:
    case 0n large 0n: Empty.absurd({Nat.is_le(large,0n) == False{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},outside)))
    case 0n large 1n+limit: Empty.absurd({Nat.is_le(large,1n+limit) == False{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},outside)))
    case 1n+small 0n limit: Empty.absurd({Nat.is_le(0n,limit) == False{} : Bool},WordEncoding.false_true(ordered))
    case 1n+small 1n+large 0n: {==}
    case 1n+small 1n+large 1n+limit: outside_monotone(small,large,limit,ordered,outside)

def step_bounded(+valid: Bool,+bit: Bool,+right: Nat,+previous: Nat,+limit: Nat,
  guard: {Nat.is_le(previous,limit) == valid : Bool}) ->
  {CheckedWord.multiply_step(bit,right,CheckedWord.bounded_result(valid,previous),limit) ==
    CheckedWord.bounded(Nat.add(Nat.double(previous),Bool.pick(Nat,bit,right,0n)),limit) : Maybe<&2,Nat>}:
  match valid:
    case True{}: {==}
    case False{}:
      %Equal.sym(Bool,Nat.is_le(Nat.add(Nat.double(previous),Bool.pick(Nat,bit,right,0n)),limit),False{},outside_monotone(previous,Nat.add(Nat.double(previous),Bool.pick(Nat,bit,right,0n)),limit,
        Order.transitive(previous,Nat.double(previous),Nat.add(Nat.double(previous),Bool.pick(Nat,bit,right,0n)),
          Order.le_double(previous),Order.le_add(Nat.double(previous),Bool.pick(Nat,bit,right,0n))),guard)) :
        {None{} == CheckedWord.bounded_result(_,Nat.add(Nat.double(previous),Bool.pick(Nat,bit,right,0n))) : Maybe<&2,Nat>}
      {==}

def digit_product(+bit: Bool,+previous: Nat,+right: Nat) ->
  {Nat.mul(Comparison.digit(bit,previous),right) == Nat.add(Nat.double(Nat.mul(previous,right)),Bool.pick(Nat,bit,right,0n)) : Nat}:
  match bit:
    case False{}:
      %Equal.sym(Nat,Nat.add(Nat.double(Nat.mul(previous,right)),0n),Nat.double(Nat.mul(previous,right)),Addition.zero_right(Nat.double(Nat.mul(previous,right)))) : {Nat.mul(Nat.double(previous),right) == _ : Nat}
      BinaryDivision.product_double(previous,right)
    case True{}:
      %BinaryDivision.product_double(previous,right) : {Nat.add(right,Nat.mul(Nat.double(previous),right)) == Nat.add(_,right) : Nat}
      Order.commutative(right,Nat.mul(Nat.double(previous),right))

def digit_value(+width: Nat,+bit: Bool,+tail: Word(width)) ->
  {Word.to_nat(1n+width,WCon{bit,tail}) == Comparison.digit(bit,Word.to_nat(width,tail)) : Nat}:
  match bit:
    case False{}: {==}
    case True{}: {==}

def multiply_bits(+width: Nat,+bits: Word(width),+right: Nat,+limit: Nat) ->
  {CheckedWord.multiply_bits(width,bits,right,limit) == CheckedWord.bounded(Nat.mul(Word.to_nat(width,bits),right),limit) : Maybe<&2,Nat>}:
  match width:
    case 0n:
      WNil{} = bits
      %Equal.sym(Bool,Nat.is_le(0n,limit),True{},Order.zero_le(limit)) :
        {Some{0n} == CheckedWord.bounded_result(_,0n) : Maybe<&2,Nat>}
      {==}
    case 1n+ +rest:
      WCon{+bit,+tail} = bits
      %Equal.sym(Nat,Word.to_nat(1n+rest,WCon{bit,tail}),Comparison.digit(bit,Word.to_nat(rest,tail)),digit_value(rest,bit,tail)) :
        {CheckedWord.multiply_bits(1n+rest,WCon{bit,tail},right,limit) == CheckedWord.bounded(Nat.mul(_,right),limit) : Maybe<&2,Nat>}
      %Equal.sym(Nat,Nat.mul(Comparison.digit(bit,Word.to_nat(rest,tail)),right),Nat.add(Nat.double(Nat.mul(Word.to_nat(rest,tail),right)),Bool.pick(Nat,bit,right,0n)),digit_product(bit,Word.to_nat(rest,tail),right)) :
        {CheckedWord.multiply_bits(1n+rest,WCon{bit,tail},right,limit) == CheckedWord.bounded(_,limit) : Maybe<&2,Nat>}
      %Equal.sym(Maybe<&2,Nat>,CheckedWord.multiply_bits(rest,tail,right,limit),CheckedWord.bounded(Nat.mul(Word.to_nat(rest,tail),right),limit),multiply_bits(rest,tail,right,limit)) :
        {CheckedWord.multiply_step(bit,right,_,limit) == CheckedWord.bounded(Nat.add(Nat.double(Nat.mul(Word.to_nat(rest,tail),right)),Bool.pick(Nat,bit,right,0n)),limit) : Maybe<&2,Nat>}
      step_bounded(Nat.is_le(Nat.mul(Word.to_nat(rest,tail),right),limit),bit,right,Nat.mul(Word.to_nat(rest,tail),right),limit,{==})

def encoded_product(+width: Nat,+bits: Word(width),+right: Nat,+limit: Nat) ->
  {CheckedWord.encoded(CheckedWord.multiply_bits(width,bits,right,limit)) == CheckedWord.encoded(CheckedWord.bounded(Nat.mul(Word.to_nat(width,bits),right),limit)) : Maybe<&2,U32>}:
  Equal.cong(Maybe<&2,Nat>,Maybe<&2,U32>,result => CheckedWord.encoded(result),CheckedWord.multiply_bits(width,bits,right,limit),
    CheckedWord.bounded(Nat.mul(Word.to_nat(width,bits),right),limit),multiply_bits(width,bits,right,limit))

def multiply_bounded(+left: U32,+right: U32,+limit: U32) ->
  {CheckedWord.multiply_bounded(left,right,limit) == CheckedWord.encoded(CheckedWord.bounded(Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(limit))) : Maybe<&2,U32>}:
  U32{+bits} = left
  encoded_product(32n,bits,U32.to_nat(right),U32.to_nat(limit))

def present(result: Maybe<&2,U32>) -> Bool:
  match result:
    case None{}: False{}
    case Some{value}: True{}

def value(result: Maybe<&2,U32>) -> U32:
  match result:
    case None{}: 0
    case Some{value}: value

def selected_sound(+valid: Bool,+product: Nat,+limit: U32,+output: U32,
  guard: {Nat.is_le(product,U32.to_nat(limit)) == valid : Bool},
  accepted: {CheckedWord.encoded(CheckedWord.bounded_result(valid,product)) == Some{output} : Maybe<&2,U32>}) ->
  {product == U32.to_nat(output) : Nat}:
  match valid:
    case False{}:
      Empty.absurd({product == U32.to_nat(output) : Nat},WordEncoding.false_true(
        Equal.cong(Maybe<&2,U32>,Bool,result => present(result),None{},Some{output},accepted)))
    case True{}:
      Equal.trans(Nat,product,U32.to_nat(U32.from_nat(product)),U32.to_nat(output),
        Equal.sym(Nat,U32.to_nat(U32.from_nat(product)),product,FullRange.roundtrip(product,limit,guard)),
        Equal.cong(U32,Nat,word => U32.to_nat(word),U32.from_nat(product),output,
          Equal.cong(Maybe<&2,U32>,U32,result => value(result),Some{U32.from_nat(product)},Some{output},accepted)))

def bounded_sound(+left: U32,+right: U32,+limit: U32,+output: U32,
  accepted: {CheckedWord.multiply_bounded(left,right,limit) == Some{output} : Maybe<&2,U32>}) ->
  {Nat.mul(U32.to_nat(left),U32.to_nat(right)) == U32.to_nat(output) : Nat}:
  selected_sound(Nat.is_le(Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(limit)),Nat.mul(U32.to_nat(left),U32.to_nat(right)),limit,output,{==},
    Equal.trans(Maybe<&2,U32>,CheckedWord.encoded(CheckedWord.bounded(Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(limit))),
      CheckedWord.multiply_bounded(left,right,limit),Some{output},
      Equal.sym(Maybe<&2,U32>,CheckedWord.multiply_bounded(left,right,limit),CheckedWord.encoded(CheckedWord.bounded(Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(limit))),multiply_bounded(left,right,limit)),accepted))

def equal_native(+left: U32,+right: U32,equal: {U32.to_nat(left) == U32.to_nat(right) : Nat}) -> {left == right : U32}:
  %Arithmetic.native_roundtrip(left) : {_ == right : U32}
  %Arithmetic.native_roundtrip(right) : {U32.from_nat(U32.to_nat(left)) == _ : U32}
  Equal.cong(Nat,U32,value => U32.from_nat(value),U32.to_nat(left),U32.to_nat(right),equal)

def classified_equal(+left: U32,+right: U32,equal: {U32.is_eq(left,right) == True{} : Bool}) -> {left == right : U32}:
  equal_native(left,right,Order.equal_of_is_eq(U32.to_nat(left),U32.to_nat(right),{
    %Comparison.eq(left,right) : {_ == True{} : Bool}
    equal : {Nat.is_eq(U32.to_nat(left),U32.to_nat(right)) == True{} : Bool}}))

def left_unit_product(+left: U32,+right: U32,unit: {U32.is_eq(left,1) == True{} : Bool}) ->
  {Nat.mul(U32.to_nat(left),U32.to_nat(right)) == U32.to_nat(right) : Nat}:
  %Equal.sym(U32,left,1,classified_equal(left,1,unit)) : {Nat.mul(U32.to_nat(_),U32.to_nat(right)) == U32.to_nat(right) : Nat}
  Addition.zero_right(U32.to_nat(right))

def right_unit_product(+left: U32,+right: U32,unit: {U32.is_eq(right,1) == True{} : Bool}) ->
  {Nat.mul(U32.to_nat(left),U32.to_nat(right)) == U32.to_nat(left) : Nat}:
  %Equal.sym(U32,right,1,classified_equal(right,1,unit)) : {Nat.mul(U32.to_nat(left),U32.to_nat(_)) == U32.to_nat(left) : Nat}
  Equal.trans(Nat,Nat.mul(U32.to_nat(left),1n),Nat.mul(1n,U32.to_nat(left)),U32.to_nat(left),RowMajor.multiply_commutative(U32.to_nat(left),1n),Addition.zero_right(U32.to_nat(left)))

def encoded_identity(+input: U32,+limit: U32,within: {U32.is_le(input,limit) == True{} : Bool}) ->
  {Some{input} == CheckedWord.encoded(CheckedWord.bounded(U32.to_nat(input),U32.to_nat(limit))) : Maybe<&2,U32>}:
  %Equal.sym(Bool,Nat.is_le(U32.to_nat(input),U32.to_nat(limit)),True{},Bounds.as_nat_le(input,limit,within)) :
    {Some{input} == CheckedWord.encoded(CheckedWord.bounded_result(_,U32.to_nat(input))) : Maybe<&2,U32>}
  Equal.cong(U32,Maybe<&2,U32>,word => Some{word},input,U32.from_nat(U32.to_nat(input)),Equal.sym(U32,U32.from_nat(U32.to_nat(input)),input,Arithmetic.native_roundtrip(input)))

def right_refines(unit: Bool,+left: U32,+right: U32,+limit: U32,
  classified: {U32.is_eq(right,1) == unit : Bool},+maximum: {limit == Machine.word_max() : U32}) ->
  {CheckedWord.multiply_right(unit,left,right) == CheckedWord.encoded(CheckedWord.bounded(Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(limit))) : Maybe<&2,U32>}:
  match unit:
    case True{}:
      %Equal.sym(Nat,Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(left),right_unit_product(left,right,classified)) : {Some{left} == CheckedWord.encoded(CheckedWord.bounded(_,U32.to_nat(limit))) : Maybe<&2,U32>}
      encoded_identity(left,limit,{
        %Equal.sym(U32,limit,Machine.word_max(),maximum) : {U32.is_le(left,_) == True{} : Bool}
        FullRange.maximum(left) : {U32.is_le(left,limit) == True{} : Bool}})
    case False{}:
      %maximum : {CheckedWord.multiply_bounded(left,right,_) == CheckedWord.encoded(CheckedWord.bounded(Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(limit))) : Maybe<&2,U32>}
      multiply_bounded(left,right,limit)

def left_refines(unit: Bool,+left: U32,+right: U32,+limit: U32,
  classified: {U32.is_eq(left,1) == unit : Bool},+maximum: {limit == Machine.word_max() : U32}) ->
  {CheckedWord.multiply_left(unit,left,right) == CheckedWord.encoded(CheckedWord.bounded(Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(limit))) : Maybe<&2,U32>}:
  match unit:
    case True{}:
      %Equal.sym(Nat,Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(right),left_unit_product(left,right,classified)) : {Some{right} == CheckedWord.encoded(CheckedWord.bounded(_,U32.to_nat(limit))) : Maybe<&2,U32>}
      encoded_identity(right,limit,{
        %Equal.sym(U32,limit,Machine.word_max(),maximum) : {U32.is_le(right,_) == True{} : Bool}
        FullRange.maximum(right) : {U32.is_le(right,limit) == True{} : Bool}})
    case False{}: right_refines(U32.is_eq(right,1),left,right,limit,{==},maximum)

def multiply(+left: U32,+right: U32,+limit: U32,maximum: {limit == Machine.word_max() : U32}) ->
  {CheckedWord.multiply(left,right) == CheckedWord.encoded(CheckedWord.bounded(Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(limit))) : Maybe<&2,U32>}:
  left_refines(U32.is_eq(left,1),left,right,limit,{==},maximum)

def accepted_value(+input: U32,+output: U32,accepted: {Some{input} == Some{output} : Maybe<&2,U32>}) -> {U32.to_nat(input) == U32.to_nat(output) : Nat}:
  Equal.cong(Maybe<&2,U32>,Nat,result => U32.to_nat(value(result)),Some{input},Some{output},accepted)

def right_sound(unit: Bool,+left: U32,+right: U32,+output: U32,
  classified: {U32.is_eq(right,1) == unit : Bool},accepted: {CheckedWord.multiply_right(unit,left,right) == Some{output} : Maybe<&2,U32>}) ->
  {Nat.mul(U32.to_nat(left),U32.to_nat(right)) == U32.to_nat(output) : Nat}:
  match unit:
    case True{}: Equal.trans(Nat,Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(left),U32.to_nat(output),right_unit_product(left,right,classified),accepted_value(left,output,accepted))
    case False{}: bounded_sound(left,right,Machine.word_max(),output,accepted)

def left_sound(unit: Bool,+left: U32,+right: U32,+output: U32,
  classified: {U32.is_eq(left,1) == unit : Bool},accepted: {CheckedWord.multiply_left(unit,left,right) == Some{output} : Maybe<&2,U32>}) ->
  {Nat.mul(U32.to_nat(left),U32.to_nat(right)) == U32.to_nat(output) : Nat}:
  match unit:
    case True{}: Equal.trans(Nat,Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(right),U32.to_nat(output),left_unit_product(left,right,classified),accepted_value(right,output,accepted))
    case False{}: right_sound(U32.is_eq(right,1),left,right,output,{==},accepted)

def multiply_sound(+left: U32,+right: U32,+output: U32,
  accepted: {CheckedWord.multiply(left,right) == Some{output} : Maybe<&2,U32>}) ->
  {Nat.mul(U32.to_nat(left),U32.to_nat(right)) == U32.to_nat(output) : Nat}:
  left_sound(U32.is_eq(left,1),left,right,output,{==},accepted)
