import Base
import ./natural_addition.bend as Addition
import ./u32_successor.bend as U32Successor
import ./word_arithmetic.bend as WordArithmetic
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_comparison.bend as Comparison
import ./nat_to_u32_bounds.bend as Bounds
import ./word_encoding.bend as WordEncoding

law out_pad:
  for +n: Nat
  for +c: Bool
  for +w: Word(n)
  {Word.shl.out(1n+n,c,Word.shr.pad(n,w)) == (False{},Word.shl.put(1n+n,c,Word.shr.pad(n,w))) : Bool & Word(1n+n)}
def out_pad(n,c,w):
  match n:
    case 0n:
      {==}
    case 1n+ +p:
      WCon{+bit,+tail} = w
      %Equal.sym(Bool & Word(1n+p),Word.shl.out(1n+p,bit,Word.shr.pad(p,tail)),(False{},Word.shl.put(1n+p,bit,Word.shr.pad(p,tail))),out_pad(p,
          bit,tail)) : {Word.shl.out.con(1n+p,c,_) == (False{},WCon{c,Word.shl.put(1n+p,bit,Word.shr.pad(p,tail))}) : Bool & Word(2n+p)}
      {==}

def Goal(+w: Word(32n),+c: Bool) -> Type:
  {U32.is_lt(U32{w},2147483648) == True{} : Bool} -> {Word.shl.out(32n,c,w) == (False{},Word.shl.put(32n,c,w)) : Bool & Word(32n)}

def parts(+c: Bool,+parts: WordEncoding.Split<31n,1n>) -> Goal(WordEncoding.join(31n,1n,parts),c):
  WordEncoding.Parts{+low,+high} = parts
  bound =>
    %Equal.sym(Word(1n),high,Word.zero(1n),WordEncoding.upper_zero(31n,0n,low,high,bound)) : {Word.shl.out(32n,c,WordEncoding.concat(31n,1n,low,
          _)) == (False{},Word.shl.put(32n,c,WordEncoding.concat(31n,1n,low,_))) : Bool & Word(32n)}
    %Equal.sym(Word(32n),WordEncoding.concat(31n,1n,low,Word.zero(1n)),Word.shr.pad(31n,low),U32Successor.concat_pad(31n,
        low)) : {Word.shl.out(32n,c,_) == (False{},Word.shl.put(32n,c,_)) : Bool & Word(32n)}
    out_pad(31n,c,low)

def no_carry(+w: Word(32n),+c: Bool) -> Goal(w,c):
  %WordEncoding.recover(31n,1n,w) : Goal(_,c)
  parts(c,WordEncoding.split(31n,1n,w))

def put_inc(+n: Nat,+w: Word(n)) -> {Word.shl.put(n,True{},w) == Word.inc(n,Word.shl(n,w)) : Word(n)}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      WCon{bit,tail} = w
      {==}

def encoded_shift(+w: Word(32n)) -> {U32{Word.shl(32n,w)} == U32.from_nat(Nat.double(Word.to_nat(32n,w))) : U32}:
  %WordArithmetic.double_word(32n,w) : {U32{_} == U32.from_nat(Nat.double(Word.to_nat(32n,w))) : U32}
  %Addition.double(Word.to_nat(32n,w)) : {U32{Word.add(32n,w,w)} == U32.from_nat(_) : U32}
  Arithmetic.native_add(U32{w},U32{w})

def encoded_put(+c: Bool,+w: Word(32n)) -> {U32{Word.shl.put(32n,c,w)} == U32.from_nat(Comparison.digit(c,Word.to_nat(32n,w))) : U32}:
  match c:
    case False{}:
      %WordArithmetic.shift_put(32n,w) : {U32{_} == U32.from_nat(Nat.double(Word.to_nat(32n,w))) : U32}
      encoded_shift(w)
    case True{}:
      %Equal.sym(Word(32n),Word.shl.put(32n,True{},w),Word.inc(32n,Word.shl(32n,w)),put_inc(32n,
          w)) : {U32{_} == U32.inc(U32.from_nat(Nat.double(Word.to_nat(32n,w)))) : U32}
      %encoded_shift(w) : {U32.inc(U32{Word.shl(32n,w)}) == U32.inc(_) : U32}
      {==}

def bounded_put(+c: Bool,+w: Word(32n),+cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},h: {Nat.is_le(Comparison.digit(c,
        Word.to_nat(32n,w)),U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat(U32{Word.shl.put(32n,c,w)}) == Comparison.digit(c,
    Word.to_nat(32n,w)) : Nat}:
  %Equal.sym(U32,U32{Word.shl.put(32n,c,w)},U32.from_nat(Comparison.digit(c,Word.to_nat(32n,w))),encoded_put(c,
      w)) : {U32.to_nat(_) == Comparison.digit(c,Word.to_nat(32n,w)) : Nat}
  Bounds.roundtrip(Comparison.digit(c,Word.to_nat(32n,w)),cap,cap_ok,h)

def put_pad(+n: Nat,+c: Bool,+w: Word(n)) -> {Word.shl.put(1n+n,c,Word.shr.pad(n,w)) == WCon{c,w} : Word(1n+n)}:
  match n:
    case 0n:
      WNil{} = w
      {==}
    case 1n+ +p:
      WCon{+b,+tail} = w
      %put_pad(p,b,tail) : {WCon{c,Word.shl.put(1n+p,b,Word.shr.pad(p,tail))} == WCon{c,_} : Word(2n+p)}
      {==}

def DoubleGoal(+w: Word(32n)) -> Type:
  {U32.is_lt(U32{w},2147483648) == True{} : Bool} -> {Word.to_nat(32n,Word.shl(32n,w)) == Nat.double(Word.to_nat(32n,w)) : Nat}

def double_parts(+parts: WordEncoding.Split<31n,1n>) -> DoubleGoal(WordEncoding.join(31n,1n,parts)):
  WordEncoding.Parts{+low,+high} = parts
  bound =>
    %Equal.sym(Word(1n),high,Word.zero(1n),WordEncoding.upper_zero(31n,0n,low,high,bound)) : {Word.to_nat(32n,Word.shl(32n,
          WordEncoding.concat(31n,1n,low,_))) == Nat.double(Word.to_nat(32n,WordEncoding.concat(31n,1n,low,_))) : Nat}
    %Equal.sym(Word(32n),WordEncoding.concat(31n,1n,low,Word.zero(1n)),Word.shr.pad(31n,low),U32Successor.concat_pad(31n,
        low)) : {Word.to_nat(32n,Word.shl(32n,_)) == Nat.double(Word.to_nat(32n,_)) : Nat}
    %Equal.sym(Word(32n),Word.shl(32n,Word.shr.pad(31n,low)),Word.shl.put(32n,False{},Word.shr.pad(31n,low)),WordArithmetic.shift_put(32n,
        Word.shr.pad(31n,low))) : {Word.to_nat(32n,_) == Nat.double(Word.to_nat(32n,Word.shr.pad(31n,low))) : Nat}
    %Equal.sym(Word(32n),Word.shl.put(32n,False{},Word.shr.pad(31n,low)),WCon{False{},low},put_pad(31n,False{},low)) : {Word.to_nat(32n,
        _) == Nat.double(Word.to_nat(32n,Word.shr.pad(31n,low))) : Nat}
    %U32Successor.pad_nat(31n,low) : {Nat.double(_) == Nat.double(Word.to_nat(32n,Word.shr.pad(31n,low))) : Nat}
    {==}

def double_word(+w: Word(32n)) -> DoubleGoal(w):
  %WordEncoding.recover(31n,1n,w) : DoubleGoal(_)
  double_parts(WordEncoding.split(31n,1n,w))

def double(+x: U32,h: {U32.is_lt(x,2147483648) == True{} : Bool}) -> {U32.to_nat(U32.shl(x)) == Nat.double(U32.to_nat(x)) : Nat}:
  match x:
    case U32{w}:
      double_word(w)(h)
