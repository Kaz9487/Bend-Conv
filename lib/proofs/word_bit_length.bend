# Bit-length induction and its zero-padding/iteration contracts.
import Base
import ./word_power_of_two.bend as Power
import ./nat_order.bend as Order
import ./word_encoding.bend as WordEncoding

def is_zero(+width: Nat,+word: Word(width)) -> Bool:
  Cmp.is_eq(Word.cmp(width,word,Word.zero(width)))

def bit_length(width: Nat,+word: Word(width)) -> Nat:
  match width:
    case 0n: 0n
    case 1n+ +rest:
      WCon{+bit,+tail} = word
      Bool.pick(Nat,is_zero(1n+rest,WCon{bit,tail}),0n,1n+bit_length(rest,tail))

def counted_shifts(+width: Nat,fuel: Nat,+word: Word(width)) -> Nat:
  match fuel:
    case 0n: 0n
    case 1n+rest:
      Bool.pick(Nat,is_zero(width,word),0n,1n+counted_shifts(width,rest,Word.shr(width,word)))

law padded_comparison:
  for +width: Nat
  for +word: Word(width)
  {Word.cmp(1n+width,Word.shr.pad(width,word),Word.zero(1n+width)) == Word.cmp(width,word,Word.zero(width)) : Cmp}

def padded_comparison(width,word):
  match width:
    case 0n:
      match word:
        case WNil{}: {==}
    case 1n+ +rest:
      match word:
        case WCon{+bit,+tail}:
          %padded_comparison(rest,tail) : {Word.cmp.fin(bit,False{},Word.cmp(1n+rest,Word.shr.pad(rest,tail),
                Word.zero(1n+rest))) == Word.cmp.fin(bit,False{},_) : Cmp}
          {==}

law padding_commutes_with_shift:
  for +width: Nat
  for +word: Word(width)
  {Word.shr(1n+width,Word.shr.pad(width,word)) == Word.shr.pad(width,Word.shr(width,word)) : Word(1n+width)}

def padding_commutes_with_shift(width,word):
  match width:
    case 0n:
      match word:
        case WNil{}: {==}
    case 1n+rest:
      match word:
        case WCon{bit,tail}: {==}

law counted_padding:
  for +width: Nat
  for +fuel: Nat
  for +word: Word(width)
  {counted_shifts(1n+width,fuel,Word.shr.pad(width,word)) == counted_shifts(width,fuel,word) : Nat}

def counted_padding(width,fuel,word):
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Cmp,Word.cmp(1n+width,Word.shr.pad(width,word),Word.zero(1n+width)),Word.cmp(width,word,Word.zero(width)),
        padded_comparison(width,word)) : {Bool.pick(Nat,Cmp.is_eq(_),0n,1n+counted_shifts(1n+width,rest,Word.shr(1n+width,Word.shr.pad(width,
              word)))) == counted_shifts(width,1n+rest,word) : Nat}
      %Equal.sym(Word(1n+width),Word.shr(1n+width,Word.shr.pad(width,word)),Word.shr.pad(width,Word.shr(width,word)),
        padding_commutes_with_shift(width,word)) : {Bool.pick(Nat,is_zero(width,word),0n,1n+counted_shifts(1n+width,rest,
            _)) == counted_shifts(width,1n+rest,word) : Nat}
      %counted_padding(width,rest,Word.shr(width,word)) : {Bool.pick(Nat,is_zero(width,word),0n,1n+counted_shifts(1n+width,rest,
            Word.shr.pad(width,Word.shr(width,word)))) == Bool.pick(Nat,is_zero(width,word),0n,1n+_) : Nat}
      {==}

law counted_empty:
  for +fuel: Nat
  {counted_shifts(0n,fuel,WNil{}) == 0n : Nat}

def counted_empty(fuel):
  match fuel:
    case 0n: {==}
    case 1n+rest: {==}

law counted_shifts_match_length:
  for +width: Nat
  for +fuel: Nat
  for +word: Word(width)
  for enough: {Nat.is_le(width,fuel) == True{} : Bool}
  {counted_shifts(width,fuel,word) == bit_length(width,word) : Nat}

def counted_shifts_match_length(width,fuel,word,enough):
  match width fuel:
    case 0n fuel:
      match word:
        case WNil{}: counted_empty(fuel)
    case 1n+width 0n:
      Empty.absurd({counted_shifts(1n+width,0n,word) == bit_length(1n+width,word) : Nat},WordEncoding.false_true(enough))
    case 1n+ +rest 1n+ +remaining:
      match word:
        case WCon{+bit,+tail}:
          %Equal.sym(Nat,counted_shifts(1n+rest,remaining,Word.shr.pad(rest,tail)),counted_shifts(rest,remaining,tail),counted_padding(rest,
              remaining,tail)) : {Bool.pick(Nat,is_zero(1n+rest,WCon{bit,tail}),0n,1n+_) == bit_length(1n+rest,WCon{bit,tail}) : Nat}
          %counted_shifts_match_length(rest,remaining,tail,enough) : {Bool.pick(Nat,is_zero(1n+rest,WCon{bit,tail}),0n,1n+counted_shifts(rest,
                remaining,tail)) == Bool.pick(Nat,is_zero(1n+rest,WCon{bit,tail}),0n,1n+_) : Nat}
          {==}

law zero_length:
  for +width: Nat
  {bit_length(width,Word.zero(width)) == 0n : Nat}

def zero_length(width):
  match width:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Cmp,Word.cmp(1n+rest,Word.zero(1n+rest),Word.zero(1n+rest)),EQ{},Power.compare_zero(1n+rest)) : {Bool.pick(Nat,Cmp.is_eq(_),
          0n,1n+bit_length(rest,Word.zero(rest))) == 0n : Nat}
      {==}

def tail_bound(+width: Nat,+depth: Nat,+bit: Bool,+tail: Word(width),bound: {Cmp.is_lt(Word.cmp(1n+width,WCon{bit,tail},Power.power(1n+width,
        1n+depth))) == True{} : Bool}) -> {Cmp.is_lt(Word.cmp(width,tail,Power.power(width,depth))) == True{} : Bool}:
  %WordEncoding.finish_lt_zero(bit,Word.cmp(width,tail,Power.power(width,depth))) : {_ == True{} : Bool}
  bound

def selected_length_bound(zero: Bool,+depth: Nat,+length: Nat,bound: {Nat.is_le(length,depth) == True{} : Bool}) -> {Nat.is_le(Bool.pick(Nat,
      zero,0n,1n+length),1n+depth) == True{} : Bool}:
  match zero:
    case True{}: {==}
    case False{}: bound

law bounded_length:
  for +width: Nat
  for +depth: Nat
  for +word: Word(width)
  for bound: {Cmp.is_lt(Word.cmp(width,word,Power.power(width,depth))) == True{} : Bool}
  {Nat.is_le(bit_length(width,word),depth) == True{} : Bool}

def bounded_length(width,depth,word,bound):
  match width:
    case 0n: Order.zero_le(depth)
    case 1n+ +rest:
      match depth:
        case 0n:
          %Equal.sym(Word(1n+rest),word,Word.zero(1n+rest),WordEncoding.below_one_zero(rest,word,bound)) : {Nat.is_le(bit_length(1n+rest,_),
              0n) == True{} : Bool}
          %Equal.sym(Nat,bit_length(1n+rest,Word.zero(1n+rest)),0n,zero_length(1n+rest)) : {Nat.is_le(_,0n) == True{} : Bool}
          {==}
        case 1n+ +previous:
          match word:
            case WCon{+bit,+tail}:
              selected_length_bound(is_zero(1n+rest,WCon{bit,tail}),previous,bit_length(rest,tail),bounded_length(rest,previous,tail,
                  tail_bound(rest,previous,bit,tail,bound)))

def selected_length_covers(+width: Nat,+bit: Bool,+tail: Word(width),zero: Bool,zero_test: {is_zero(1n+width,WCon{bit,tail}) == zero : Bool},
  tail_room: {Cmp.is_lt(Word.cmp(width,tail,Power.power(width,bit_length(width,tail)))) == True{} : Bool}) -> {Cmp.is_lt(Word.cmp(1n+width,
      WCon{bit,tail},Power.power(1n+width,Bool.pick(Nat,zero,0n,1n+bit_length(width,tail))))) == True{} : Bool}:
  match zero:
    case True{}:
      %Equal.sym(Word(1n+width),WCon{bit,tail},Word.zero(1n+width),WordEncoding.zero_reflect(1n+width,WCon{bit,tail},
          WordEncoding.cmp_eq(Word.cmp(1n+width,WCon{bit,tail},Word.zero(1n+width)),zero_test))) : {Cmp.is_lt(Word.cmp(1n+width,_,
            Power.power(1n+width,0n))) == True{} : Bool}
      %Power.positive_power(1n+width,0n,{==}) : {Cmp.is_lt(Word.cmp(1n+width,Word.zero(1n+width),Power.power(1n+width,
              0n))) == Cmp.is_lt(_) : Bool}
      {==}
    case False{}:
      %Equal.sym(Bool,Cmp.is_lt(Word.cmp.fin(bit,False{},Word.cmp(width,tail,Power.power(width,bit_length(width,tail))))),
        Cmp.is_lt(Word.cmp(width,tail,Power.power(width,bit_length(width,tail)))),WordEncoding.finish_lt_zero(bit,Word.cmp(width,tail,
            Power.power(width,bit_length(width,tail))))) : {_ == True{} : Bool}
      tail_room

law length_covers_bounded_word:
  for +width: Nat
  for +depth: Nat
  for +word: Word(width)
  for fits: {Nat.is_lt(depth,width) == True{} : Bool}
  for bound: {Cmp.is_lt(Word.cmp(width,word,Power.power(width,depth))) == True{} : Bool}
  {Cmp.is_lt(Word.cmp(width,word,Power.power(width,bit_length(width,word)))) == True{} : Bool}

def length_covers_bounded_word(width,depth,word,fits,bound):
  match width:
    case 0n:
      match depth:
        case 0n:
          Empty.absurd({Cmp.is_lt(Word.cmp(0n,word,Power.power(0n,bit_length(0n,word)))) == True{} : Bool},WordEncoding.false_true(fits))
        case 1n+rest:
          Empty.absurd({Cmp.is_lt(Word.cmp(0n,word,Power.power(0n,bit_length(0n,word)))) == True{} : Bool},WordEncoding.false_true(fits))
    case 1n+ +rest:
      match depth:
        case 0n:
          %Equal.sym(Word(1n+rest),word,Word.zero(1n+rest),WordEncoding.below_one_zero(rest,word,bound)) : {Cmp.is_lt(Word.cmp(1n+rest,_,
                Power.power(1n+rest,bit_length(1n+rest,_)))) == True{} : Bool}
          %Equal.sym(Nat,bit_length(1n+rest,Word.zero(1n+rest)),0n,zero_length(1n+rest)) : {Cmp.is_lt(Word.cmp(1n+rest,Word.zero(1n+rest),
                Power.power(1n+rest,_))) == True{} : Bool}
          %Power.positive_power(1n+rest,0n,{==}) : {Cmp.is_lt(Word.cmp(1n+rest,Word.zero(1n+rest),Power.power(1n+rest,
                  0n))) == Cmp.is_lt(_) : Bool}
          {==}
        case 1n+ +previous:
          match word:
            case WCon{+bit,+tail}:
              selected_length_covers(rest,bit,tail,is_zero(1n+rest,WCon{bit,tail}),{==},length_covers_bounded_word(rest,previous,tail,fits,
                  tail_bound(rest,previous,bit,tail,bound)))
