# Natural-number binary long division, proved against the frozen Base.divmod.
import Base
import ./natural_addition.bend as Addition
import ./binary_digit.bend as BinaryDigit
import ./u32_comparison.bend as Comparison
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./word_multiplication.bend as Multiplication
import ./word_subtraction.bend as Subtraction
import ./word_encoding.bend as WordEncoding

type Quotient is Data:
  Quotient{quotient: Nat,remainder: Nat}

def total(result: Quotient,b: Nat) -> Nat:
  Quotient{q,r} = result
  Nat.add(Nat.mul(q,b),r)

def remainder(result: Quotient) -> Nat:
  Quotient{q,r} = result
  r

def pair(result: Quotient) -> Nat & Nat:
  Quotient{q,r} = result
  (q,r)

def branch(g: Bool,+q: Nat,s: Nat,b: Nat) -> Quotient:
  match g:
    case False{}:
      Quotient{q,s}
    case True{}:
      Quotient{1n+q,Nat.sub(s,b)}

def step(bit: Bool,+b: Nat,prev: Quotient) -> Quotient:
  Quotient{q,r} = prev
  +s = Comparison.digit(bit,r)
  branch(Nat.is_ge(s,b),Nat.double(q),s,b)

def go(n: Nat,w: Word(n),+b: Nat) -> Quotient:
  match n:
    case 0n:
      Quotient{0n,0n}
    case 1n+p:
      WCon{bit,tail} = w
      step(bit,b,go(p,tail,b))

def rotate(+a: Nat,+b: Nat,+c: Nat) -> {Nat.add(Nat.add(a,b),c) == Nat.add(b,Nat.add(c,a)) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(a,b),c),Nat.add(a,Nat.add(b,c)),Nat.add(b,Nat.add(c,a)),Addition.associative(a,b,c),Equal.trans(Nat,
      Nat.add(a,Nat.add(b,c)),Nat.add(Nat.add(b,c),a),Nat.add(b,Nat.add(c,a)),Order.commutative(a,Nat.add(b,c)),Addition.associative(b,c,a)))

def product_right(+a: Nat,+b: Nat) -> {Nat.mul(a,Nat.double(b)) == Nat.double(Nat.mul(a,b)) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+ +p:
      %Equal.sym(Nat,Nat.double(Nat.add(b,Nat.mul(p,b))),Nat.add(Nat.double(b),Nat.double(Nat.mul(p,b))),WordEncoding.double_add(b,Nat.mul(p,
            b))) : {Nat.add(Nat.double(b),Nat.mul(p,Nat.double(b))) == _ : Nat}
      %product_right(p,b) : {Nat.add(Nat.double(b),Nat.mul(p,Nat.double(b))) == Nat.add(Nat.double(b),_) : Nat}
      {==}

def product_double(+a: Nat,+b: Nat) -> {Nat.mul(Nat.double(a),b) == Nat.double(Nat.mul(a,b)) : Nat}:
  %Equal.sym(Nat,Nat.mul(Nat.double(a),b),Nat.mul(a,Nat.double(b)),Multiplication.double_mul(a,b)) : {_ == Nat.double(Nat.mul(a,b)) : Nat}
  product_right(a,b)

def even_total(+q: Nat,+r: Nat,+b: Nat) -> {Nat.add(Nat.mul(Nat.double(q),b),Nat.double(r)) == Nat.double(Nat.add(Nat.mul(q,b),r)) : Nat}:
  %Equal.sym(Nat,Nat.mul(Nat.double(q),b),Nat.double(Nat.mul(q,b)),product_double(q,b)) : {Nat.add(_,
      Nat.double(r)) == Nat.double(Nat.add(Nat.mul(q,b),r)) : Nat}
  Equal.sym(Nat,Nat.double(Nat.add(Nat.mul(q,b),r)),Nat.add(Nat.double(Nat.mul(q,b)),Nat.double(r)),WordEncoding.double_add(Nat.mul(q,b),r))

def digit_total(+bit: Bool,+q: Nat,+r: Nat,+b: Nat) -> {Nat.add(Nat.mul(Nat.double(q),b),Comparison.digit(bit,r)) == Comparison.digit(bit,
    Nat.add(Nat.mul(q,b),r)) : Nat}:
  match bit:
    case False{}:
      even_total(q,r,b)
    case True{}:
      %Equal.sym(Nat,Nat.add(Nat.mul(Nat.double(q),b),1n+Nat.double(r)),1n+Nat.add(Nat.mul(Nat.double(q),b),Nat.double(r)),
        Addition.successor_right(Nat.mul(Nat.double(q),b),Nat.double(r))) : {_ == 1n+Nat.double(Nat.add(Nat.mul(q,b),r)) : Nat}
      %even_total(q,r,b) : {1n+Nat.add(Nat.mul(Nat.double(q),b),Nat.double(r)) == 1n+_ : Nat}
      {==}

def branch_total(+g: Bool,+q: Nat,+s: Nat,+b: Nat,guard: {Nat.is_ge(s,b) == g : Bool}) -> {total(branch(g,q,s,b),b) == Nat.add(Nat.mul(q,b),s) : Nat}:
  match g:
    case False{}:
      {==}
    case True{}:
      %Equal.sym(Nat,Nat.add(Nat.add(b,Nat.mul(q,b)),Nat.sub(s,b)),Nat.add(Nat.mul(q,b),Nat.add(Nat.sub(s,b),b)),rotate(b,Nat.mul(q,b),Nat.sub(s,
            b))) : {_ == Nat.add(Nat.mul(q,b),s) : Nat}
      %Subtraction.restore(s,b,Order.ge_true(s,b,guard)) : {Nat.add(Nat.mul(q,b),Nat.add(Nat.sub(s,b),b)) == Nat.add(Nat.mul(q,b),_) : Nat}
      {==}

def step_total(+bit: Bool,+b: Nat,+prev: Quotient) -> {total(step(bit,b,prev),b) == Comparison.digit(bit,total(prev,b)) : Nat}:
  Quotient{+q,+r} = prev
  %Equal.sym(Nat,total(branch(Nat.is_ge(Comparison.digit(bit,r),b),Nat.double(q),Comparison.digit(bit,r),b),b),
    Nat.add(Nat.mul(Nat.double(q),b),Comparison.digit(bit,r)),branch_total(Nat.is_ge(Comparison.digit(bit,r),b),Nat.double(q),
      Comparison.digit(bit,r),b,{==})) : {_ == Comparison.digit(bit,Nat.add(Nat.mul(q,b),r)) : Nat}
  digit_total(bit,q,r,b)

def sum_bound(+s: Nat,+b: Nat,h: {Nat.is_lt(s,Nat.double(b)) == True{} : Bool}) -> {Nat.is_lt(s,Nat.add(b,b)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(b,b),Nat.double(b),Addition.double(b)) : {Nat.is_lt(s,_) == True{} : Bool}
  h

def branch_bound(+g: Bool,+q: Nat,+s: Nat,+b: Nat,guard: {Nat.is_ge(s,b) == g : Bool},bound: {Nat.is_lt(s,
      Nat.double(b)) == True{} : Bool}) -> {Nat.is_lt(remainder(branch(g,q,s,b)),b) == True{} : Bool}:
  match g:
    case False{}:
      Order.ge_false(s,b,guard)
    case True{}:
      Order.sub_lt(s,b,b,Order.ge_true(s,b,guard),sum_bound(s,b,bound))

def step_bound(+bit: Bool,+b: Nat,+prev: Quotient,h: {Nat.is_lt(remainder(prev),b) == True{} : Bool}) -> {Nat.is_lt(remainder(step(bit,b,prev)),
    b) == True{} : Bool}:
  Quotient{+q,+r} = prev
  branch_bound(Nat.is_ge(Comparison.digit(bit,r),b),Nat.double(q),Comparison.digit(bit,r),b,{==},Order.digit_lt(r,b,bit,h))

law go_total:
  for +n: Nat
  for +w: Word(n)
  for +b: Nat
  {total(go(n,w,b),b) == Word.to_nat(n,w) : Nat}
def go_total(n,w,b):
  match n:
    case 0n:
      {==}
    case 1n+ +p:
      WCon{+bit,+tail} = w
      %Equal.sym(Nat,Word.to_nat(1n+p,WCon{bit,tail}),Comparison.digit(bit,Word.to_nat(p,tail)),BinaryDigit.word_digit(p,bit,
          tail)) : {total(step(bit,b,go(p,tail,b)),b) == _ : Nat}
      %go_total(p,tail,b) : {total(step(bit,b,go(p,tail,b)),b) == Comparison.digit(bit,_) : Nat}
      step_total(bit,b,go(p,tail,b))

law go_bound:
  for +n: Nat
  for +w: Word(n)
  for +b: Nat
  for positive: {Nat.is_lt(0n,b) == True{} : Bool}
  {Nat.is_lt(remainder(go(n,w,b)),b) == True{} : Bool}
def go_bound(n,w,b,positive):
  match n:
    case 0n:
      positive
    case 1n+ +p:
      WCon{+bit,+tail} = w
      step_bound(bit,b,go(p,tail,b),go_bound(p,tail,b,positive))

def finish(+a: Nat,+b: Nat,+result: Quotient,e: {total(result,b) == a : Nat},positive: {Nat.is_lt(0n,b) == True{} : Bool},
  bounded: {Nat.is_lt(remainder(result),b) == True{} : Bool}) -> {Nat.divmod(a,b) == pair(result) : Nat & Nat}:
  Quotient{+q,+r} = result
  Division.unique(a,b,q,r,Equal.sym(Nat,Nat.add(Nat.mul(q,b),r),a,e),positive,bounded)

def natural(+n: Nat,+w: Word(n),+b: Nat,+positive: {Nat.is_lt(0n,b) == True{} : Bool}) -> {Nat.divmod(Word.to_nat(n,w),b) == pair(go(n,w,
      b)) : Nat & Nat}:
  finish(Word.to_nat(n,w),b,go(n,w,b),go_total(n,w,b),positive,go_bound(n,w,b,positive))
