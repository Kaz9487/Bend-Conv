# Safe Nat -> U32 -> Nat conversion over the candidate's finite envelope.
import Base
import ./u32_successor.bend as U32Successor
import ./u32_comparison.bend as Comparison
import ./word_encoding.bend as WordEncoding

def predecessor_strict(+a: Nat,+b: Nat,h: {Nat.is_le(1n+a,b) == True{} : Bool}) -> {Nat.is_lt(a,b) == True{} : Bool}:
  match a b:
    case 0n 0n:
      Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(h))
    case 0n 1n+bp:
      {==}
    case 1n+ap 0n:
      Empty.absurd({Nat.is_lt(1n+ap,0n) == True{} : Bool},WordEncoding.false_true(h))
    case 1n+ap 1n+bp:
      predecessor_strict(ap,bp,h)

def strict_le(+a: Nat,+b: Nat,h: {Nat.is_lt(a,b) == True{} : Bool}) -> {Nat.is_le(a,b) == True{} : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+bp:
      {==}
    case 1n+ap 0n:
      Empty.absurd({Nat.is_le(1n+ap,0n) == True{} : Bool},WordEncoding.false_true(h))
    case 1n+ap 1n+bp:
      strict_le(ap,bp,h)

def add_one(+i: U32) -> {U32.add(i,1) == U32.inc(i) : U32}:
  match i:
    case U32{+w}:
      %U32Successor.add_one(32n,w) : {U32{Word.add(32n,w,Word.inc(32n,Word.zero(32n)))} == U32{_} : U32}
      {==}

def native_bound(+n: Nat,+cap: U32,e: {U32.to_nat(U32.from_nat(n)) == n : Nat},h: {Nat.is_lt(n,
      U32.to_nat(cap)) == True{} : Bool}) -> {U32.is_lt(U32.from_nat(n),cap) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(n),cap),Nat.is_lt(U32.to_nat(U32.from_nat(n)),U32.to_nat(cap)),Comparison.lt(U32.from_nat(n),
      cap)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(n)),n,e) : {Nat.is_lt(_,U32.to_nat(cap)) == True{} : Bool}
  h

def nat_trans(+a: Nat,+b: Nat,+c: Nat,ab: {Nat.is_lt(a,b) == True{} : Bool},bc: {Nat.is_le(b,c) == True{} : Bool}) -> {Nat.is_lt(a,
    c) == True{} : Bool}:
  match a b c:
    case 0n 0n c:
      Empty.absurd({Nat.is_lt(0n,c) == True{} : Bool},WordEncoding.false_true(ab))
    case 0n 1n+b 0n:
      Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(bc))
    case 0n 1n+b 1n+c:
      {==}
    case 1n+a 0n c:
      Empty.absurd({Nat.is_lt(1n+a,c) == True{} : Bool},WordEncoding.false_true(ab))
    case 1n+a 1n+b 0n:
      Empty.absurd({Nat.is_lt(1n+a,0n) == True{} : Bool},WordEncoding.false_true(bc))
    case 1n+a 1n+b 1n+c:
      nat_trans(a,b,c,ab,bc)

def as_nat_lt(+a: U32,+b: U32,h: {U32.is_lt(a,b) == True{} : Bool}) -> {Nat.is_lt(U32.to_nat(a),U32.to_nat(b)) == True{} : Bool}:
  %Comparison.lt(a,b) : {_ == True{} : Bool}
  h

def as_nat_le(+a: U32,+b: U32,h: {U32.is_le(a,b) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(a),U32.to_nat(b)) == True{} : Bool}:
  %Comparison.le(a,b) : {_ == True{} : Bool}
  h

def uint_trans(+a: U32,+b: U32,+c: U32,ab: {U32.is_lt(a,b) == True{} : Bool},bc: {U32.is_le(b,c) == True{} : Bool}) -> {U32.is_lt(a,
    c) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_lt(a,c),Nat.is_lt(U32.to_nat(a),U32.to_nat(c)),Comparison.lt(a,c)) : {_ == True{} : Bool}
  nat_trans(U32.to_nat(a),U32.to_nat(b),U32.to_nat(c),as_nat_lt(a,b,ab),as_nat_le(b,c,bc))

law roundtrip:
  for +n: Nat
  for +cap: U32
  for +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool}
  for h: {Nat.is_le(n,U32.to_nat(cap)) == True{} : Bool}
  {U32.to_nat(U32.from_nat(n)) == n : Nat}
def roundtrip(n,cap,cap_ok,h):
  match n:
    case 0n:
      {==}
    case 1n+ +p:
      +small = predecessor_strict(p,U32.to_nat(cap),h)
      +previous = roundtrip(p,cap,cap_ok,strict_le(p,U32.to_nat(cap),small))
      %previous : {U32.to_nat(U32.inc(U32.from_nat(p))) == 1n+_ : Nat}
      %add_one(U32.from_nat(p)) : {U32.to_nat(_) == 1n+U32.to_nat(U32.from_nat(p)) : Nat}
      U32Successor.successor(U32.from_nat(p))(uint_trans(U32.from_nat(p),cap,2147483648,native_bound(p,cap,previous,small),cap_ok))
