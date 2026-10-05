# Characterize Base.Nat.divmod by quotient/remainder and remainder bounds.
import Base
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding

def lt_zero(+n: Nat) -> {Nat.is_lt(n,0n) == False{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      {==}

def lt_succ_le(+n: Nat,+m: Nat,h: {Nat.is_lt(n,1n+m) == True{} : Bool}) -> {Nat.is_le(n,m) == True{} : Bool}:
  match n m:
    case 0n 0n:
      {==}
    case 0n 1n+m:
      {==}
    case 1n+ +n 0n:
      Empty.absurd({Nat.is_le(1n+n,0n) == True{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(n,0n),True{},Equal.sym(Bool,
              Nat.is_lt(n,0n),False{},lt_zero(n)),h)))
    case 1n+n 1n+m:
      lt_succ_le(n,m,h)

def go_small(+n: Nat,+m: Nat,+d: Nat,+r: Nat,h: {Nat.is_le(n,m) == True{} : Bool}) -> {Nat.divmod.go(n,m,d,r) == (d,Nat.add(n,r)) : Nat & Nat}:
  match n:
    case 0n:
      {==}
    case 1n+ +p:
      match m:
        case 0n:
          Empty.absurd({Nat.divmod.go(1n+p,0n,d,r) == (d,Nat.add(1n+p,r)) : Nat & Nat},WordEncoding.false_true(h))
        case 1n+ +q:
          %Addition.successor_right(p,r) : {Nat.divmod.go(p,q,d,1n+r) == (d,_) : Nat & Nat}
          go_small(p,q,d,1n+r,h)

def small(+r: Nat,+b: Nat,h: {Nat.is_lt(r,b) == True{} : Bool}) -> {Nat.divmod(r,b) == (0n,r) : Nat & Nat}:
  match b:
    case 0n:
      Empty.absurd({Nat.divmod(r,0n) == (0n,r) : Nat & Nat},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(r,0n),True{},
            Equal.sym(Bool,Nat.is_lt(r,0n),False{},lt_zero(r)),h)))
    case 1n+ +p:
      %Addition.zero_right(r) : {Nat.divmod.go(r,p,0n,0n) == (0n,_) : Nat & Nat}
      go_small(r,p,0n,0n,lt_succ_le(r,p,h))

def bump(pair: Nat & Nat) -> Nat & Nat:
  (q,r) = pair
  (1n+q,r)

def bumped_quotient(pair: Nat & Nat) -> {Pair.fst(Nat,Nat,bump(pair)) == 1n+Pair.fst(Nat,Nat,pair) : Nat}:
  (quotient,remainder) = pair
  {==}

law bump_go:
  for +n: Nat
  for +m: Nat
  for +d: Nat
  for +r: Nat
  {Nat.divmod.go(n,m,1n+d,r) == bump(Nat.divmod.go(n,m,d,r)) : Nat & Nat}
def bump_go(n,m,d,r):
  match n:
    case 0n:
      {==}
    case 1n+ +p:
      match m:
        case 0n:
          bump_go(p,r,1n+d,0n)
        case 1n+q:
          bump_go(p,q,d,1n+r)

law cycle:
  for +m: Nat
  for +tail: Nat
  for +d: Nat
  for +r: Nat
  {Nat.divmod.go(Nat.add(1n+m,tail),m,d,r) == Nat.divmod.go(tail,Nat.add(m,r),1n+d,0n) : Nat & Nat}
def cycle(m,tail,d,r):
  match m:
    case 0n:
      {==}
    case 1n+ +p:
      %Addition.successor_right(p,r) : {Nat.divmod.go(Nat.add(1n+p,tail),p,d,1n+r) == Nat.divmod.go(tail,_,1n+d,0n) : Nat & Nat}
      cycle(p,tail,d,1n+r)

def add_divisor(+a: Nat,+b: Nat,h: {Nat.is_lt(0n,b) == True{} : Bool}) -> {Nat.divmod(Nat.add(b,a),b) == bump(Nat.divmod(a,b)) : Nat & Nat}:
  match b:
    case 0n:
      Empty.absurd({Nat.divmod(a,0n) == bump(Nat.divmod(a,0n)) : Nat & Nat},WordEncoding.false_true(h))
    case 1n+ +p:
      %Equal.sym(Nat & Nat,Nat.divmod.go(Nat.add(1n+p,a),p,0n,0n),Nat.divmod.go(a,Nat.add(p,0n),1n,0n),cycle(p,a,0n,
          0n)) : {_ == bump(Nat.divmod.go(a,p,0n,0n)) : Nat & Nat}
      %Equal.sym(Nat,Nat.add(p,0n),p,Addition.zero_right(p)) : {Nat.divmod.go(a,_,1n,0n) == bump(Nat.divmod.go(a,p,0n,0n)) : Nat & Nat}
      bump_go(a,p,0n,0n)

law quotient_remainder:
  for +q: Nat
  for +r: Nat
  for +b: Nat
  for +positive: {Nat.is_lt(0n,b) == True{} : Bool}
  for +bounded: {Nat.is_lt(r,b) == True{} : Bool}
  {Nat.divmod(Nat.add(Nat.mul(q,b),r),b) == (q,r) : Nat & Nat}
def quotient_remainder(q,r,b,positive,bounded):
  match q:
    case 0n:
      small(r,b,bounded)
    case 1n+ +p:
      %Equal.sym(Nat,Nat.add(Nat.add(b,Nat.mul(p,b)),r),Nat.add(b,Nat.add(Nat.mul(p,b),r)),Addition.associative(b,Nat.mul(p,b),
          r)) : {Nat.divmod(_,b) == (1n+p,r) : Nat & Nat}
      %Equal.sym(Nat & Nat,Nat.divmod(Nat.add(b,Nat.add(Nat.mul(p,b),r)),b),bump(Nat.divmod(Nat.add(Nat.mul(p,b),r),b)),
        add_divisor(Nat.add(Nat.mul(p,b),r),b,positive)) : {_ == (1n+p,r) : Nat & Nat}
      %Equal.sym(Nat & Nat,Nat.divmod(Nat.add(Nat.mul(p,b),r),b),(p,r),quotient_remainder(p,r,b,positive,bounded)) : {bump(_) == (1n+p,r) : Nat & Nat}
      {==}

def unique(+a: Nat,+b: Nat,+q: Nat,+r: Nat,e: {a == Nat.add(Nat.mul(q,b),r) : Nat},positive: {Nat.is_lt(0n,b) == True{} : Bool},
  bounded: {Nat.is_lt(r,b) == True{} : Bool}) -> {Nat.divmod(a,b) == (q,r) : Nat & Nat}:
  %Equal.sym(Nat,a,Nat.add(Nat.mul(q,b),r),e) : {Nat.divmod(_,b) == (q,r) : Nat & Nat}
  quotient_remainder(q,r,b,positive,bounded)
