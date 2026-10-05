import Base

def digit(bit: Bool,n: Nat) -> Nat:
  match bit:
    case False{}:
      Nat.double(n)
    case True{}:
      1n+Nat.double(n)

law digits:
  for +a: Nat
  for +b: Nat
  for +ab: Bool
  for +bb: Bool
  {Nat.cmp(digit(ab,a),digit(bb,b)) == Word.cmp.fin(ab,bb,Nat.cmp(a,b)) : Cmp}
def digits(a,b,ab,bb):
  match a b ab bb:
    case 0n 0n False{} False{}:
      {==}
    case 0n 0n False{} True{}:
      {==}
    case 0n 0n True{} False{}:
      {==}
    case 0n 0n True{} True{}:
      {==}
    case 0n 1n+bp False{} False{}:
      {==}
    case 0n 1n+bp False{} True{}:
      {==}
    case 0n 1n+bp True{} False{}:
      {==}
    case 0n 1n+bp True{} True{}:
      {==}
    case 1n+ap 0n False{} False{}:
      {==}
    case 1n+ap 0n False{} True{}:
      {==}
    case 1n+ap 0n True{} False{}:
      {==}
    case 1n+ap 0n True{} True{}:
      {==}
    case 1n+ap 1n+bp False{} False{}:
      digits(ap,bp,False{},False{})
    case 1n+ap 1n+bp False{} True{}:
      digits(ap,bp,False{},True{})
    case 1n+ap 1n+bp True{} False{}:
      digits(ap,bp,True{},False{})
    case 1n+ap 1n+bp True{} True{}:
      digits(ap,bp,True{},True{})

law word:
  for +n: Nat
  for +a: Word(n)
  for +b: Word(n)
  {Word.cmp(n,a,b) == Nat.cmp(Word.to_nat(n,a),Word.to_nat(n,b)) : Cmp}
def word(n,a,b):
  match n:
    case 0n:
      {==}
    case 1n+ +p:
      match a b:
        case WCon{False{},+at} WCon{False{},+bt}:
          %Equal.sym(Cmp,Nat.cmp(digit(False{},Word.to_nat(p,at)),digit(False{},Word.to_nat(p,bt))),Word.cmp.fin(False{},False{},
              Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt))),digits(Word.to_nat(p,at),Word.to_nat(p,bt),False{},False{})) : {Word.cmp.fin(False{},
              False{},Word.cmp(p,at,bt)) == _ : Cmp}
          %word(p,at,bt) : {Word.cmp.fin(False{},False{},Word.cmp(p,at,bt)) == Word.cmp.fin(False{},False{},_) : Cmp}
          {==}
        case WCon{False{},+at} WCon{True{},+bt}:
          %Equal.sym(Cmp,Nat.cmp(digit(False{},Word.to_nat(p,at)),digit(True{},Word.to_nat(p,bt))),Word.cmp.fin(False{},True{},
              Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt))),digits(Word.to_nat(p,at),Word.to_nat(p,bt),False{},True{})) : {Word.cmp.fin(False{},
              True{},Word.cmp(p,at,bt)) == _ : Cmp}
          %word(p,at,bt) : {Word.cmp.fin(False{},True{},Word.cmp(p,at,bt)) == Word.cmp.fin(False{},True{},_) : Cmp}
          {==}
        case WCon{True{},+at} WCon{False{},+bt}:
          %Equal.sym(Cmp,Nat.cmp(digit(True{},Word.to_nat(p,at)),digit(False{},Word.to_nat(p,bt))),Word.cmp.fin(True{},False{},
              Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt))),digits(Word.to_nat(p,at),Word.to_nat(p,bt),True{},False{})) : {Word.cmp.fin(True{},
              False{},Word.cmp(p,at,bt)) == _ : Cmp}
          %word(p,at,bt) : {Word.cmp.fin(True{},False{},Word.cmp(p,at,bt)) == Word.cmp.fin(True{},False{},_) : Cmp}
          {==}
        case WCon{True{},+at} WCon{True{},+bt}:
          %Equal.sym(Cmp,Nat.cmp(digit(True{},Word.to_nat(p,at)),digit(True{},Word.to_nat(p,bt))),Word.cmp.fin(True{},True{},
              Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt))),digits(Word.to_nat(p,at),Word.to_nat(p,bt),True{},True{})) : {Word.cmp.fin(True{},
              True{},Word.cmp(p,at,bt)) == _ : Cmp}
          %word(p,at,bt) : {Word.cmp.fin(True{},True{},Word.cmp(p,at,bt)) == Word.cmp.fin(True{},True{},_) : Cmp}
          {==}

law uint:
  for +a: U32
  for +b: U32
  {U32.cmp(a,b) == Nat.cmp(U32.to_nat(a),U32.to_nat(b)) : Cmp}
def uint(a,b):
  match a b:
    case U32{x} U32{y}:
      word(32n,x,y)

def lt(+a: U32,+b: U32) -> {U32.is_lt(a,b) == Nat.is_lt(U32.to_nat(a),U32.to_nat(b)) : Bool}:
  %uint(a,b) : {Cmp.is_lt(U32.cmp(a,b)) == Cmp.is_lt(_) : Bool}
  {==}

def le(+a: U32,+b: U32) -> {U32.is_le(a,b) == Nat.is_le(U32.to_nat(a),U32.to_nat(b)) : Bool}:
  %uint(a,b) : {Cmp.is_le(U32.cmp(a,b)) == Cmp.is_le(_) : Bool}
  {==}

def eq(+a: U32,+b: U32) -> {U32.is_eq(a,b) == Nat.is_eq(U32.to_nat(a),U32.to_nat(b)) : Bool}:
  %uint(a,b) : {Cmp.is_eq(U32.cmp(a,b)) == Cmp.is_eq(_) : Bool}
  {==}

def ge(+a: U32,+b: U32) -> {U32.is_ge(a,b) == Nat.is_ge(U32.to_nat(a),U32.to_nat(b)) : Bool}:
  %uint(a,b) : {Cmp.is_ge(U32.cmp(a,b)) == Cmp.is_ge(_) : Bool}
  {==}
