import ./storage_shape.bend as Shape
# The storage allocator follows the generic bit-length proof.
import Base
import ../storage_buffer.bend as Storage
import ./word_bit_length.bend as WordLength
import ./word_power_of_two.bend as Power

def LengthGoal(+index: U32) -> Data:
  Shape.Both<{Nat.is_le(Storage.bit_length(32n,index),30n) == True{} : Bool},{U32.is_lt(index,U32.shln(1,Storage.bit_length(32n,
          index))) == True{} : Bool}>

law native_counted_shifts:
  for +fuel: Nat
  for +word: Word(32n)
  {Storage.bit_length(fuel,U32{word}) == WordLength.counted_shifts(32n,fuel,word) : Nat}

def native_counted_shifts(fuel,word):
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      %native_counted_shifts(rest,Word.shr(32n,word)) : {Bool.pick(Nat,WordLength.is_zero(32n,word),0n,1n+Storage.bit_length(rest,
            U32.shr(U32{word}))) == Bool.pick(Nat,WordLength.is_zero(32n,word),0n,1n+_) : Nat}
      {==}

law native_length:
  for +word: Word(32n)
  {Storage.bit_length(32n,U32{word}) == WordLength.bit_length(32n,word) : Nat}

def native_length(word):
  Equal.trans(Nat,Storage.bit_length(32n,U32{word}),WordLength.counted_shifts(32n,32n,word),WordLength.bit_length(32n,word),
    native_counted_shifts(32n,word),WordLength.counted_shifts_match_length(32n,32n,word,{==}))

def bounded_word(+word: Word(32n),+bound: {U32.is_lt(U32{word},1073741824) == True{} : Bool}) -> LengthGoal(U32{word}):
  %Equal.sym(Nat,Storage.bit_length(32n,U32{word}),WordLength.bit_length(32n,word),native_length(word)) : Shape.Both<{Nat.is_le(_,
      30n) == True{} : Bool},{U32.is_lt(U32{word},U32.shln(1,_)) == True{} : Bool}>
  Shape.Both{
    WordLength.bounded_length(32n,30n,word,bound),
    %Equal.sym(U32,U32.shln(1,WordLength.bit_length(32n,word)),U32{Power.power(32n,WordLength.bit_length(32n,word))},
      Power.native_power(WordLength.bit_length(32n,word))) : {U32.is_lt(U32{word},_) == True{} : Bool}
    WordLength.length_covers_bounded_word(32n,30n,word,{==},bound)}

def width(+index: U32,bound: {U32.is_lt(index,1073741824) == True{} : Bool}) -> LengthGoal(index):
  match index:
    case U32{word}:
      bounded_word(word,bound)
