# Actual Base.U32 long division -> the proven Nat binary division algorithm.
import Base
import ./binary_division.bend as BinaryDivision
import ./u32_comparison.bend as Comparison
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_shift.bend as U32Shift
import ./word_encoding.bend as WordEncoding

def decode(+n: Nat,pair: Word(n) & U32) -> BinaryDivision.Quotient:
  (q,r) = pair
  BinaryDivision.Quotient{Word.to_nat(n,q),U32.to_nat(r)}

def nat_lt_native(+a: U32,+b: U32,h: {Nat.is_lt(U32.to_nat(a),U32.to_nat(b)) == True{} : Bool}) -> {U32.is_lt(a,b) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_lt(a,b),Nat.is_lt(U32.to_nat(a),U32.to_nat(b)),Comparison.lt(a,b)) : {_ == True{} : Bool}
  h

def nat_le_native(+a: U32,+b: U32,h: {Nat.is_le(U32.to_nat(a),U32.to_nat(b)) == True{} : Bool}) -> {U32.is_le(a,b) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(a,b),Nat.is_le(U32.to_nat(a),U32.to_nat(b)),Comparison.le(a,b)) : {_ == True{} : Bool}
  h

def native_ge_nat(+a: U32,+b: U32,h: {U32.is_ge(a,b) == True{} : Bool}) -> {Nat.is_ge(U32.to_nat(a),U32.to_nat(b)) == True{} : Bool}:
  %Comparison.ge(a,b) : {_ == True{} : Bool}
  h

def fin(+p: Nat,+q: Word(p),+s: U32,+b: U32,+cap: U32,+g: Bool,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  bound: {Nat.is_le(U32.to_nat(s),U32.to_nat(cap)) == True{} : Bool},guard: {U32.is_ge(s,b) == g : Bool}) -> {decode(1n+p,U32.divmod.go.fin(p,q,
      s,b,g)) == BinaryDivision.branch(g,Nat.double(Word.to_nat(p,q)),U32.to_nat(s),U32.to_nat(b)) : BinaryDivision.Quotient}:
  match g:
    case False{}:
      {==}
    case True{}:
      %Arithmetic.subtract(s,b,cap,cap_ok,Order.ge_true(U32.to_nat(s),U32.to_nat(b),native_ge_nat(s,b,guard)),
        Order.transitive(Nat.sub(U32.to_nat(s),U32.to_nat(b)),U32.to_nat(s),U32.to_nat(cap),Order.sub_le(U32.to_nat(s),U32.to_nat(b)),
          bound)) : {BinaryDivision.Quotient{1n+Nat.double(Word.to_nat(p,q)),U32.to_nat(U32.sub(s,
              b))} == BinaryDivision.Quotient{1n+Nat.double(Word.to_nat(p,q)),_} : BinaryDivision.Quotient}
      {==}

def shifted_bound(+s: U32,+value: Nat,+cap: U32,e: {U32.to_nat(s) == value : Nat},h: {Nat.is_le(value,
      U32.to_nat(cap)) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(s),U32.to_nat(cap)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(s),value,e) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
  h

law step:
  for +p: Nat
  for +bit: Bool
  for +b: U32
  for +cap: U32
  for +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool}
  for +divisor_bound: {Nat.is_le(Nat.double(U32.to_nat(b)),U32.to_nat(cap)) == True{} : Bool}
  for pair: Word(p) & U32
  for small: {Nat.is_lt(BinaryDivision.remainder(decode(p,pair)),U32.to_nat(b)) == True{} : Bool}
  {decode(1n+p,U32.divmod.go.rec(p,bit,b,pair)) == BinaryDivision.step(bit,U32.to_nat(b),decode(p,pair)) : BinaryDivision.Quotient}
def step(p,bit,b,cap,cap_ok,divisor_bound,pair,small):
  (+q,r) = pair
  U32{+rw} = r
  +small = small
  +s = {U32{Word.shl.put(32n,bit,rw)} : U32}
  +value = Comparison.digit(bit,Word.to_nat(32n,rw))
  +bound = Order.digit_bounded(Word.to_nat(32n,rw),U32.to_nat(b),bit,U32.to_nat(cap),small,divisor_bound)
  +numeric = U32Shift.bounded_put(bit,rw,cap,cap_ok,bound)
  +bcap = nat_le_native(b,cap,Order.transitive(U32.to_nat(b),Nat.double(U32.to_nat(b)),U32.to_nat(cap),Order.le_double(U32.to_nat(b)),
      divisor_bound))
  +rcap = Bounds.uint_trans(U32{rw},b,cap,nat_lt_native(U32{rw},b,small),bcap)
  %Equal.sym(Bool & Word(32n),Word.shl.out(32n,bit,rw),(False{},Word.shl.put(32n,bit,rw)),U32Shift.no_carry(rw,
      bit)(Bounds.uint_trans(U32{rw},cap,2147483648,rcap,cap_ok))) : {decode(1n+p,U32.divmod.go.shl(p,q,b,_)) == BinaryDivision.step(bit,
      U32.to_nat(b),BinaryDivision.Quotient{Word.to_nat(p,q),Word.to_nat(32n,rw)}) : BinaryDivision.Quotient}
  %Equal.sym(BinaryDivision.Quotient,decode(1n+p,U32.divmod.go.fin(p,q,s,b,U32.is_ge(s,b))),BinaryDivision.branch(U32.is_ge(s,b),
      Nat.double(Word.to_nat(p,q)),U32.to_nat(s),U32.to_nat(b)),fin(p,q,s,b,cap,U32.is_ge(s,b),cap_ok,shifted_bound(s,value,cap,numeric,bound),
      {==})) : {_ == BinaryDivision.step(bit,U32.to_nat(b),BinaryDivision.Quotient{Word.to_nat(p,q),Word.to_nat(32n,rw)}) : BinaryDivision.Quotient}
  %Equal.sym(Bool,U32.is_ge(s,b),Nat.is_ge(U32.to_nat(s),U32.to_nat(b)),Comparison.ge(s,b)) : {BinaryDivision.branch(_,
      Nat.double(Word.to_nat(p,q)),U32.to_nat(s),U32.to_nat(b)) == BinaryDivision.step(bit,U32.to_nat(b),BinaryDivision.Quotient{Word.to_nat(p,q),
        Word.to_nat(32n,rw)}) : BinaryDivision.Quotient}
  %Equal.sym(Nat,U32.to_nat(s),value,numeric) : {BinaryDivision.branch(Nat.is_ge(_,U32.to_nat(b)),Nat.double(Word.to_nat(p,q)),_,
      U32.to_nat(b)) == BinaryDivision.step(bit,U32.to_nat(b),BinaryDivision.Quotient{Word.to_nat(p,q),Word.to_nat(32n,rw)}) : BinaryDivision.Quotient}
  {==}

def transfer_bound(+prev: BinaryDivision.Quotient,+next: BinaryDivision.Quotient,+b: Nat,e: {prev == next : BinaryDivision.Quotient},
  h: {Nat.is_lt(BinaryDivision.remainder(next),b) == True{} : Bool}) -> {Nat.is_lt(BinaryDivision.remainder(prev),b) == True{} : Bool}:
  %Equal.sym(BinaryDivision.Quotient,prev,next,e) : {Nat.is_lt(BinaryDivision.remainder(_),b) == True{} : Bool}
  h

law go:
  for +n: Nat
  for +w: Word(n)
  for +b: U32
  for +cap: U32
  for +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool}
  for +divisor_bound: {Nat.is_le(Nat.double(U32.to_nat(b)),U32.to_nat(cap)) == True{} : Bool}
  for +positive: {Nat.is_lt(0n,U32.to_nat(b)) == True{} : Bool}
  {decode(n,U32.divmod.go(n,w,b)) == BinaryDivision.go(n,w,U32.to_nat(b)) : BinaryDivision.Quotient}
def go(n,w,b,cap,cap_ok,divisor_bound,positive):
  match n:
    case 0n:
      WNil{} = w
      {==}
    case 1n+ +p:
      WCon{+bit,+tail} = w
      +previous = go(p,tail,b,cap,cap_ok,divisor_bound,positive)
      %previous : {decode(1n+p,U32.divmod.go.rec(p,bit,b,U32.divmod.go(p,tail,b))) == BinaryDivision.step(bit,U32.to_nat(b),
          _) : BinaryDivision.Quotient}
      step(p,bit,b,cap,cap_ok,divisor_bound,U32.divmod.go(p,tail,b),transfer_bound(decode(p,U32.divmod.go(p,tail,b)),BinaryDivision.go(p,tail,
            U32.to_nat(b)),U32.to_nat(b),previous,BinaryDivision.go_bound(p,tail,U32.to_nat(b),positive)))

def quotient(result: BinaryDivision.Quotient) -> Nat:
  BinaryDivision.Quotient{q,r} = result
  q

def quotient_decode(pair: Word(32n) & U32) -> {U32.to_nat(U32.div.fin(pair)) == quotient(decode(32n,pair)) : Nat}:
  (q,r) = pair
  {==}

def remainder_decode(pair: Word(32n) & U32) -> {U32.to_nat(U32.mod.fin(pair)) == BinaryDivision.remainder(decode(32n,pair)) : Nat}:
  (q,r) = pair
  {==}

def quotient_pair(+result: BinaryDivision.Quotient) -> {Pair.fst(Nat,Nat,BinaryDivision.pair(result)) == quotient(result) : Nat}:
  BinaryDivision.Quotient{q,r} = result
  {==}

def remainder_pair(+result: BinaryDivision.Quotient) -> {Pair.snd(Nat,Nat,BinaryDivision.pair(result)) == BinaryDivision.remainder(result) : Nat}:
  BinaryDivision.Quotient{q,r} = result
  {==}

def natural_quotient(+w: Word(32n),+b: Nat,+positive: {Nat.is_lt(0n,b) == True{} : Bool}) -> {Nat.div(Word.to_nat(32n,w),
    b) == quotient(BinaryDivision.go(32n,w,b)) : Nat}:
  %Equal.sym(Nat & Nat,Nat.divmod(Word.to_nat(32n,w),b),BinaryDivision.pair(BinaryDivision.go(32n,w,b)),BinaryDivision.natural(32n,w,b,
      positive)) : {Pair.fst(Nat,Nat,_) == quotient(BinaryDivision.go(32n,w,b)) : Nat}
  quotient_pair(BinaryDivision.go(32n,w,b))

def natural_remainder(+w: Word(32n),+b: Nat,+positive: {Nat.is_lt(0n,b) == True{} : Bool}) -> {Nat.mod(Word.to_nat(32n,w),
    b) == BinaryDivision.remainder(BinaryDivision.go(32n,w,b)) : Nat}:
  %Equal.sym(Nat & Nat,Nat.divmod(Word.to_nat(32n,w),b),BinaryDivision.pair(BinaryDivision.go(32n,w,b)),BinaryDivision.natural(32n,w,b,
      positive)) : {Pair.snd(Nat,Nat,_) == BinaryDivision.remainder(BinaryDivision.go(32n,w,b)) : Nat}
  remainder_pair(BinaryDivision.go(32n,w,b))

def nat_nonzero(+b: Nat,h: {Nat.is_lt(0n,b) == True{} : Bool}) -> {Nat.is_eq(b,0n) == False{} : Bool}:
  match b:
    case 0n:
      Empty.absurd({Nat.is_eq(0n,0n) == False{} : Bool},WordEncoding.false_true(h))
    case 1n+p:
      {==}

def nonzero(+b: U32,h: {Nat.is_lt(0n,U32.to_nat(b)) == True{} : Bool}) -> {U32.is_zero(b) == False{} : Bool}:
  %Equal.sym(Bool,U32.is_eq(b,0),Nat.is_eq(U32.to_nat(b),U32.to_nat(0)),Comparison.eq(b,0)) : {_ == False{} : Bool}
  nat_nonzero(U32.to_nat(b),h)

def divide(+a: U32,+b: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+divisor_bound: {Nat.is_le(Nat.double(U32.to_nat(b)),
      U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(b)) == True{} : Bool}) -> {U32.to_nat(U32.div(a,
      b)) == Nat.div(U32.to_nat(a),U32.to_nat(b)) : Nat}:
  U32{+w} = a
  U32{+bw} = b
  +b = {U32{bw} : U32}
  %Equal.sym(Bool,U32.is_zero(b),False{},nonzero(b,positive)) : {U32.to_nat(U32.div.if(w,b,_)) == Nat.div(Word.to_nat(32n,w),U32.to_nat(b)) : Nat}
  %Equal.sym(Nat,Nat.div(Word.to_nat(32n,w),U32.to_nat(b)),quotient(BinaryDivision.go(32n,w,U32.to_nat(b))),natural_quotient(w,U32.to_nat(b),
      positive)) : {U32.to_nat(U32.div.fin(U32.divmod.go(32n,w,b))) == _ : Nat}
  %go(32n,w,b,cap,cap_ok,divisor_bound,positive) : {U32.to_nat(U32.div.fin(U32.divmod.go(32n,w,b))) == quotient(_) : Nat}
  quotient_decode(U32.divmod.go(32n,w,b))

def modulo(+a: U32,+b: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+divisor_bound: {Nat.is_le(Nat.double(U32.to_nat(b)),
      U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(b)) == True{} : Bool}) -> {U32.to_nat(U32.mod(a,
      b)) == Nat.mod(U32.to_nat(a),U32.to_nat(b)) : Nat}:
  U32{+w} = a
  U32{+bw} = b
  +b = {U32{bw} : U32}
  %Equal.sym(Bool,U32.is_zero(b),False{},nonzero(b,positive)) : {U32.to_nat(U32.mod.if(w,b,_)) == Nat.mod(Word.to_nat(32n,w),U32.to_nat(b)) : Nat}
  %Equal.sym(Nat,Nat.mod(Word.to_nat(32n,w),U32.to_nat(b)),BinaryDivision.remainder(BinaryDivision.go(32n,w,U32.to_nat(b))),natural_remainder(w,
      U32.to_nat(b),positive)) : {U32.to_nat(U32.mod.fin(U32.divmod.go(32n,w,b))) == _ : Nat}
  %go(32n,w,b,cap,cap_ok,divisor_bound,positive) : {U32.to_nat(U32.mod.fin(U32.divmod.go(32n,w,b))) == BinaryDivision.remainder(_) : Nat}
  remainder_decode(U32.divmod.go(32n,w,b))
