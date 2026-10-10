# Power-of-two capacities and masks, by bit-width induction.
import Base
import ./word_encoding.bend as WordEncoding
import ./word_arithmetic.bend as WordArithmetic

def power(+width: Nat,depth: Nat) -> Word(width):
  match width:
    case 0n:
      WNil{}
    case 1n+rest:
      match depth:
        case 0n:
          WCon{True{},Word.zero(rest)}
        case 1n+previous:
          WCon{False{},power(rest,previous)}

def mask(+width: Nat,depth: Nat) -> Word(width):
  match width:
    case 0n:
      WNil{}
    case 1n+rest:
      match depth:
        case 0n:
          Word.zero(1n+rest)
        case 1n+previous:
          WCon{True{},mask(rest,previous)}

law put_into_zero:
  for +width: Nat
  for +bit: Bool
  {Word.shl.put(width,bit,Word.zero(width)) == Bool.pick(Word(width),bit,power(width,0n),Word.zero(width)) : Word(width)}

def put_into_zero(width,bit):
  match width:
    case 0n:
      match bit:
        case False{}: {==}
        case True{}: {==}
    case 1n+ +rest:
      match bit:
        case False{}:
          %put_into_zero(rest,False{}) : {WCon{False{},Word.shl.put(rest,False{},Word.zero(rest))} == WCon{False{},_} : Word(1n+rest)}
          {==}
        case True{}:
          %put_into_zero(rest,False{}) : {WCon{True{},Word.shl.put(rest,False{},Word.zero(rest))} == WCon{True{},_} : Word(1n+rest)}
          {==}

law shift_power:
  for +width: Nat
  for +depth: Nat
  {Word.shl(width,power(width,depth)) == power(width,1n+depth) : Word(width)}

def shift_power(width,depth):
  match width:
    case 0n: {==}
    case 1n+ +rest:
      match depth:
        case 0n:
          %put_into_zero(rest,True{}) : {WCon{False{},Word.shl.put(rest,True{},Word.zero(rest))} == WCon{False{},_} : Word(1n+rest)}
          {==}
        case 1n+ +previous:
          %shift_power(rest,previous) : {WCon{False{},Word.shl.put(rest,False{},power(rest,previous))} == WCon{False{},_} : Word(1n+rest)}
          Equal.cong(Word(rest),Word(1n+rest),w => WCon{False{},w},Word.shl.put(rest,False{},power(rest,previous)),Word.shl(rest,power(rest,
                previous)),Equal.sym(Word(rest),Word.shl(rest,power(rest,previous)),Word.shl.put(rest,False{},power(rest,previous)),
              WordArithmetic.shift_put(rest,power(rest,previous))))

law native_power:
  for +depth: Nat
  {U32.shln(1,depth) == U32{power(32n,depth)} : U32}

def native_power(depth):
  match depth:
    case 0n: {==}
    case 1n+ +previous:
      %Equal.sym(U32,U32.shln(1,previous),U32{power(32n,previous)},native_power(previous)) : {U32.shl(_) == U32{power(32n,1n+previous)} : U32}
      Equal.cong(Word(32n),U32,w => U32{w},Word.shl(32n,power(32n,previous)),power(32n,1n+previous),shift_power(32n,previous))

law subtract_one:
  for +width: Nat
  for +word: Word(width)
  {Word.sub(width,word,power(width,0n)) == Word.adc(width,word,Word.zero(width),True{},False{}) : Word(width)}

def subtract_one(width,word):
  match width:
    case 0n: {==}
    case 1n+rest:
      match word:
        case WCon{False{},tail}: {==}
        case WCon{True{},tail}: {==}

law zero_subtract_zero:
  for +width: Nat
  {Word.adc(width,Word.zero(width),Word.zero(width),True{},True{}) == Word.zero(width) : Word(width)}

def zero_subtract_zero(width):
  match width:
    case 0n: {==}
    case 1n+ +rest:
      %zero_subtract_zero(rest) : {WCon{False{},Word.adc(rest,Word.zero(rest),Word.zero(rest),True{},True{})} == WCon{False{},_} : Word(1n+rest)}
      {==}

law decrement_power:
  for +width: Nat
  for +depth: Nat
  {Word.adc(width,power(width,depth),Word.zero(width),True{},False{}) == mask(width,depth) : Word(width)}

def decrement_power(width,depth):
  match width:
    case 0n: {==}
    case 1n+ +rest:
      match depth:
        case 0n:
          %zero_subtract_zero(rest) : {WCon{False{},Word.adc(rest,Word.zero(rest),Word.zero(rest),True{},True{})} == WCon{False{},_} : Word(1n+rest)}
          {==}
        case 1n+ +previous:
          %decrement_power(rest,previous) : {WCon{True{},Word.adc(rest,power(rest,previous),Word.zero(rest),True{},False{})} == WCon{True{},
              _} : Word(1n+rest)}
          {==}

law mask_preserves_bounded:
  for +width: Nat
  for +depth: Nat
  for +word: Word(width)
  for bound: {Cmp.is_lt(Word.cmp(width,word,power(width,depth))) == True{} : Bool}
  {Word.and(width,word,mask(width,depth)) == word : Word(width)}

def mask_preserves_bounded(width,depth,word,bound):
  match width:
    case 0n:
      match word:
        case WNil{}: {==}
    case 1n+ +rest:
      match depth:
        case 0n:
          %Equal.sym(Word(1n+rest),word,Word.zero(1n+rest),WordEncoding.below_one_zero(rest,word,bound)) : {Word.and(1n+rest,_,
              Word.zero(1n+rest)) == _ : Word(1n+rest)}
          WordEncoding.and_zero(1n+rest,Word.zero(1n+rest))
        case 1n+ +previous:
          match word:
            case WCon{False{},+tail}:
              %mask_preserves_bounded(rest,previous,tail,Equal.trans(Bool,Cmp.is_lt(Word.cmp(rest,tail,power(rest,previous))),
                  Cmp.is_lt(Word.cmp.fin(False{},False{},Word.cmp(rest,tail,power(rest,previous)))),True{},Equal.sym(Bool,
                    Cmp.is_lt(Word.cmp.fin(False{},False{},Word.cmp(rest,tail,power(rest,previous)))),Cmp.is_lt(Word.cmp(rest,tail,power(rest,
                      previous))),WordEncoding.finish_lt_zero(False{},Word.cmp(rest,tail,power(rest,previous)))),bound)) : {WCon{False{},
                  Word.and(rest,tail,mask(rest,previous))} == WCon{False{},_} : Word(1n+rest)}
              {==}
            case WCon{True{},+tail}:
              %mask_preserves_bounded(rest,previous,tail,Equal.trans(Bool,Cmp.is_lt(Word.cmp(rest,tail,power(rest,previous))),
                  Cmp.is_lt(Word.cmp.fin(True{},False{},Word.cmp(rest,tail,power(rest,previous)))),True{},Equal.sym(Bool,
                    Cmp.is_lt(Word.cmp.fin(True{},False{},Word.cmp(rest,tail,power(rest,previous)))),Cmp.is_lt(Word.cmp(rest,tail,power(rest,
                      previous))),WordEncoding.finish_lt_zero(True{},Word.cmp(rest,tail,power(rest,previous)))),bound)) : {WCon{True{},
                  Word.and(rest,tail,mask(rest,previous))} == WCon{True{},_} : Word(1n+rest)}
              {==}

def native_word_mask(+depth: Nat,+word: Word(32n),bound: {Cmp.is_lt(Word.cmp(32n,word,power(32n,
        depth))) == True{} : Bool}) -> {U32.and(U32{word},U32.sub(U32{power(32n,depth)},1)) == U32{word} : U32}:
  %Equal.sym(Word(32n),Word.sub(32n,power(32n,depth),power(32n,0n)),mask(32n,depth),Equal.trans(Word(32n),Word.sub(32n,power(32n,depth),
        power(32n,0n)),Word.adc(32n,power(32n,depth),Word.zero(32n),True{},False{}),mask(32n,depth),subtract_one(32n,power(32n,depth)),
      decrement_power(32n,depth))) : {U32{Word.and(32n,word,_)} == U32{word} : U32}
  Equal.cong(Word(32n),U32,w => U32{w},Word.and(32n,word,mask(32n,depth)),word,mask_preserves_bounded(32n,depth,word,bound))

law native_mask_preserves_bounded:
  for +depth: Nat
  for +index: U32
  {U32.is_lt(index,U32.shln(1,depth)) == True{} : Bool} -> {U32.and(index,U32.sub(U32.shln(1,depth),1)) == index : U32}

def native_mask_preserves_bounded(depth,index):
  match index:
    case U32{+word}:
      %Equal.sym(U32,U32.shln(1,depth),U32{power(32n,depth)},native_power(depth)) : {U32.is_lt(U32{word},
          _) == True{} : Bool} -> {U32.and(U32{word},U32.sub(_,1)) == U32{word} : U32}
      bound => native_word_mask(depth,word,bound)

law compare_zero:
  for +width: Nat
  {Word.cmp(width,Word.zero(width),Word.zero(width)) == EQ{} : Cmp}

def compare_zero(width):
  match width:
    case 0n: {==}
    case 1n+ +rest:
      %compare_zero(rest) : {Word.cmp.fin(False{},False{},Word.cmp(rest,Word.zero(rest),Word.zero(rest))) == Word.cmp.fin(False{},False{},_) : Cmp}
      {==}

law positive_power:
  for +width: Nat
  for +depth: Nat
  for fits: {Nat.is_lt(depth,width) == True{} : Bool}
  {Word.cmp(width,Word.zero(width),power(width,depth)) == LT{} : Cmp}

def positive_power(width,depth,fits):
  match width:
    case 0n:
      match depth:
        case 0n: Empty.absurd({Word.cmp(0n,WNil{},power(0n,0n)) == LT{} : Cmp},WordEncoding.false_true(fits))
        case 1n+rest: Empty.absurd({Word.cmp(0n,WNil{},power(0n,1n+rest)) == LT{} : Cmp},WordEncoding.false_true(fits))
    case 1n+ +rest:
      match depth:
        case 0n:
          %compare_zero(rest) : {Word.cmp.fin(False{},True{},Word.cmp(rest,Word.zero(rest),Word.zero(rest))) == Word.cmp.fin(False{},True{},_) : Cmp}
          {==}
        case 1n+ +previous:
          %positive_power(rest,previous,fits) : {Word.cmp.fin(False{},False{},Word.cmp(rest,Word.zero(rest),power(rest,
                  previous))) == Word.cmp.fin(False{},False{},_) : Cmp}
          {==}

law compare_equal_low_bits:
  for +comparison: Cmp
  {Word.cmp.fin(False{},False{},comparison) == comparison : Cmp}

def compare_equal_low_bits(comparison):
  match comparison:
    case LT{}: {==}
    case EQ{}: {==}
    case GT{}: {==}

law power_monotone:
  for +width: Nat
  for +lower: Nat
  for +upper: Nat
  for fits: {Nat.is_lt(upper,width) == True{} : Bool}
  for ordered: {Nat.is_le(lower,upper) == True{} : Bool}
  {Cmp.is_le(Word.cmp(width,power(width,lower),power(width,upper))) == True{} : Bool}

def power_monotone(width,lower,upper,fits,ordered):
  match width:
    case 0n: {==}
    case 1n+ +rest:
      match lower upper:
        case 0n 0n:
          %compare_zero(rest) : {Cmp.is_le(Word.cmp.fin(True{},True{},Word.cmp(rest,Word.zero(rest),
                  Word.zero(rest)))) == Cmp.is_le(Word.cmp.fin(True{},True{},_)) : Bool}
          {==}
        case 0n 1n+ +previous:
          %positive_power(rest,previous,fits) : {Cmp.is_le(Word.cmp.fin(True{},False{},Word.cmp(rest,Word.zero(rest),power(rest,
                  previous)))) == Cmp.is_le(Word.cmp.fin(True{},False{},_)) : Bool}
          {==}
        case 1n+previous 0n:
          Empty.absurd({Cmp.is_le(Word.cmp(1n+rest,power(1n+rest,1n+previous),power(1n+rest,0n))) == True{} : Bool},WordEncoding.false_true(ordered))
        case 1n+ +low 1n+ +high:
          %Equal.sym(Cmp,Word.cmp.fin(False{},False{},Word.cmp(rest,power(rest,low),power(rest,high))),Word.cmp(rest,power(rest,low),power(rest,
                high)),compare_equal_low_bits(Word.cmp(rest,power(rest,low),power(rest,high)))) : {Cmp.is_le(_) == True{} : Bool}
          power_monotone(rest,low,high,fits,ordered)

law power_value:
  for +width: Nat
  for +depth: Nat
  for fits: {Nat.is_lt(depth,width) == True{} : Bool}
  {Word.to_nat(width,power(width,depth)) == WordEncoding.shift(depth,1n) : Nat}

def power_value(width,depth,fits):
  match width:
    case 0n:
      match depth:
        case 0n: Empty.absurd({0n == WordEncoding.shift(0n,1n) : Nat},WordEncoding.false_true(fits))
        case 1n+rest: Empty.absurd({0n == WordEncoding.shift(1n+rest,1n) : Nat},WordEncoding.false_true(fits))
    case 1n+ +rest:
      match depth:
        case 0n:
          %WordEncoding.zero_nat(rest) : {1n+Nat.double(Word.to_nat(rest,Word.zero(rest))) == 1n+Nat.double(_) : Nat}
          {==}
        case 1n+ +previous:
          %power_value(rest,previous,fits) : {Nat.double(Word.to_nat(rest,power(rest,previous))) == Nat.double(_) : Nat}
          {==}

law pad_zero:
  for +width: Nat
  {Word.shr.pad(width,Word.zero(width)) == Word.zero(1n+width) : Word(1n+width)}

def pad_zero(width):
  match width:
    case 0n: {==}
    case 1n+ +rest:
      %pad_zero(rest) : {WCon{False{},Word.shr.pad(rest,Word.zero(rest))} == WCon{False{},_} : Word(2n+rest)}
      {==}

law pad_power:
  for +width: Nat
  for +depth: Nat
  for fits: {Nat.is_lt(depth,width) == True{} : Bool}
  {Word.shr.pad(width,power(width,depth)) == power(1n+width,depth) : Word(1n+width)}

def pad_power(width,depth,fits):
  match width:
    case 0n:
      match depth:
        case 0n: Empty.absurd({Word.shr.pad(0n,power(0n,0n)) == power(1n,0n) : Word(1n)},WordEncoding.false_true(fits))
        case 1n+rest: Empty.absurd({Word.shr.pad(0n,power(0n,1n+rest)) == power(1n,1n+rest) : Word(1n)},WordEncoding.false_true(fits))
    case 1n+ +rest:
      match depth:
        case 0n:
          %pad_zero(rest) : {WCon{True{},Word.shr.pad(rest,Word.zero(rest))} == WCon{True{},_} : Word(2n+rest)}
          {==}
        case 1n+ +previous:
          %pad_power(rest,previous,fits) : {WCon{False{},Word.shr.pad(rest,power(rest,previous))} == WCon{False{},_} : Word(2n+rest)}
          {==}

law native_half_power:
  for +depth: Nat
  for fits: {Nat.is_lt(depth,31n) == True{} : Bool}
  {U32.shr(U32.shln(1,1n+depth)) == U32.shln(1,depth) : U32}

def native_half_power(depth,fits):
  %Equal.sym(U32,U32.shln(1,1n+depth),U32{power(32n,1n+depth)},native_power(1n+depth)) : {U32.shr(_) == U32.shln(1,depth) : U32}
  %Equal.sym(U32,U32.shln(1,depth),U32{power(32n,depth)},native_power(depth)) : {U32.shr(U32{power(32n,1n+depth)}) == _ : U32}
  Equal.cong(Word(32n),U32,w => U32{w},Word.shr.pad(31n,power(31n,depth)),power(32n,depth),pad_power(31n,depth,fits))

law native_power_bound:
  for +depth: Nat
  for bound: {Nat.is_le(depth,31n) == True{} : Bool}
  {U32.is_le(U32.shln(1,depth),2147483648) == True{} : Bool}

def native_power_bound(depth,bound):
  %Equal.sym(U32,U32.shln(1,depth),U32{power(32n,depth)},native_power(depth)) : {U32.is_le(_,2147483648) == True{} : Bool}
  power_monotone(32n,depth,31n,{==},bound)

def native_allocatable(+depth: Nat,bound: {Nat.is_le(depth,30n) == True{} : Bool}) ->
  {U32.is_le(U32.shln(1,depth),1073741824) == True{} : Bool}:
  %Equal.sym(U32,U32.shln(1,depth),U32{power(32n,depth)},native_power(depth)) : {U32.is_le(_,1073741824) == True{} : Bool}
  power_monotone(32n,depth,30n,{==},bound)

def positive_double(value: Nat,positive: {Nat.is_lt(0n,Nat.double(value)) == True{} : Bool}) -> {Nat.is_lt(0n,value) == True{} : Bool}:
  match value:
    case 0n: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(positive))
    case 1n+rest: {==}

# A nonempty checked owner cannot have a power-of-two capacity truncated to
# zero. This recovers the full machine-depth bound from its existing guard.
def nonempty_power_depth(+width: Nat,+depth: Nat,positive: {Nat.is_lt(0n,Word.to_nat(width,power(width,depth))) == True{} : Bool}) ->
  {Nat.is_lt(depth,width) == True{} : Bool}:
  match width:
    case 0n: Empty.absurd({Nat.is_lt(depth,0n) == True{} : Bool},WordEncoding.false_true(positive))
    case 1n+ +rest:
      match depth:
        case 0n: {==}
        case 1n+ +previous:
          nonempty_power_depth(rest,previous,positive_double(Word.to_nat(rest,power(rest,previous)),positive))

def native_nonempty_depth(+depth: Nat,positive: {Nat.is_lt(0n,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}) ->
  {Nat.is_lt(depth,32n) == True{} : Bool}:
  nonempty_power_depth(32n,depth,
    %native_power(depth) : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool} positive)
