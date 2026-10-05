import Base
import ./natural_addition.bend as Addition
import ./word_arithmetic.bend as WordArithmetic
import ./word_encoding.bend as WordEncoding

law undo:
  for +n: Nat
  for +a: Word(n)
  for +b: Word(n)
  for +f: Bool
  for +c: Bool
  {Word.adc(n,Word.adc(n,a,b,f,c),b,Bool.not(f),Bool.not(c)) == a : Word(n)}
def undo(n,a,b,f,c):
  match n:
    case 0n:
      WNil{} = a
      {==}
    case 1n+ +p:
      match a b f c:
        case WCon{False{},+at} WCon{False{},+bt} False{} False{}:
          %undo(p,at,bt,False{},False{}) : {WCon{False{},Word.adc(p,Word.adc(p,at,bt,False{},False{}),bt,True{},True{})} == WCon{False{},
              _} : Word(1n+p)}
          {==}
        case WCon{False{},+at} WCon{False{},+bt} False{} True{}:
          %undo(p,at,bt,False{},False{}) : {WCon{False{},Word.adc(p,Word.adc(p,at,bt,False{},False{}),bt,True{},True{})} == WCon{False{},
              _} : Word(1n+p)}
          {==}
        case WCon{False{},+at} WCon{False{},+bt} True{} False{}:
          %undo(p,at,bt,True{},False{}) : {WCon{False{},Word.adc(p,Word.adc(p,at,bt,True{},False{}),bt,False{},True{})} == WCon{False{},
              _} : Word(1n+p)}
          {==}
        case WCon{False{},+at} WCon{False{},+bt} True{} True{}:
          %undo(p,at,bt,True{},True{}) : {WCon{False{},Word.adc(p,Word.adc(p,at,bt,True{},True{}),bt,False{},False{})} == WCon{False{},
              _} : Word(1n+p)}
          {==}
        case WCon{False{},+at} WCon{True{},+bt} False{} False{}:
          %undo(p,at,bt,False{},False{}) : {WCon{False{},Word.adc(p,Word.adc(p,at,bt,False{},False{}),bt,True{},True{})} == WCon{False{},
              _} : Word(1n+p)}
          {==}
        case WCon{False{},+at} WCon{True{},+bt} False{} True{}:
          %undo(p,at,bt,False{},True{}) : {WCon{False{},Word.adc(p,Word.adc(p,at,bt,False{},True{}),bt,True{},False{})} == WCon{False{},
              _} : Word(1n+p)}
          {==}
        case WCon{False{},+at} WCon{True{},+bt} True{} False{}:
          %undo(p,at,bt,True{},False{}) : {WCon{False{},Word.adc(p,Word.adc(p,at,bt,True{},False{}),bt,False{},True{})} == WCon{False{},
              _} : Word(1n+p)}
          {==}
        case WCon{False{},+at} WCon{True{},+bt} True{} True{}:
          %undo(p,at,bt,True{},False{}) : {WCon{False{},Word.adc(p,Word.adc(p,at,bt,True{},False{}),bt,False{},True{})} == WCon{False{},
              _} : Word(1n+p)}
          {==}
        case WCon{True{},+at} WCon{False{},+bt} False{} False{}:
          %undo(p,at,bt,False{},False{}) : {WCon{True{},Word.adc(p,Word.adc(p,at,bt,False{},False{}),bt,True{},True{})} == WCon{True{},
              _} : Word(1n+p)}
          {==}
        case WCon{True{},+at} WCon{False{},+bt} False{} True{}:
          %undo(p,at,bt,False{},True{}) : {WCon{True{},Word.adc(p,Word.adc(p,at,bt,False{},True{}),bt,True{},False{})} == WCon{True{},_} : Word(1n+p)}
          {==}
        case WCon{True{},+at} WCon{False{},+bt} True{} False{}:
          %undo(p,at,bt,True{},True{}) : {WCon{True{},Word.adc(p,Word.adc(p,at,bt,True{},True{}),bt,False{},False{})} == WCon{True{},_} : Word(1n+p)}
          {==}
        case WCon{True{},+at} WCon{False{},+bt} True{} True{}:
          %undo(p,at,bt,True{},True{}) : {WCon{True{},Word.adc(p,Word.adc(p,at,bt,True{},True{}),bt,False{},False{})} == WCon{True{},_} : Word(1n+p)}
          {==}
        case WCon{True{},+at} WCon{True{},+bt} False{} False{}:
          %undo(p,at,bt,False{},True{}) : {WCon{True{},Word.adc(p,Word.adc(p,at,bt,False{},True{}),bt,True{},False{})} == WCon{True{},_} : Word(1n+p)}
          {==}
        case WCon{True{},+at} WCon{True{},+bt} False{} True{}:
          %undo(p,at,bt,False{},True{}) : {WCon{True{},Word.adc(p,Word.adc(p,at,bt,False{},True{}),bt,True{},False{})} == WCon{True{},_} : Word(1n+p)}
          {==}
        case WCon{True{},+at} WCon{True{},+bt} True{} False{}:
          %undo(p,at,bt,True{},False{}) : {WCon{True{},Word.adc(p,Word.adc(p,at,bt,True{},False{}),bt,False{},True{})} == WCon{True{},_} : Word(1n+p)}
          {==}
        case WCon{True{},+at} WCon{True{},+bt} True{} True{}:
          %undo(p,at,bt,True{},True{}) : {WCon{True{},Word.adc(p,Word.adc(p,at,bt,True{},True{}),bt,False{},False{})} == WCon{True{},_} : Word(1n+p)}
          {==}

def subtract_add(+a: U32,+b: U32) -> {U32.sub(U32.add(a,b),b) == a : U32}:
  match a b:
    case U32{+x} U32{+y}:
      %undo(32n,x,y,False{},False{}) : {U32{Word.sub(32n,Word.add(32n,x,y),y)} == U32{_} : U32}
      {==}

def add_subtract(+a: U32,+b: U32) -> {U32.add(U32.sub(a,b),b) == a : U32}:
  match a b:
    case U32{+x} U32{+y}:
      %undo(32n,x,y,True{},True{}) : {U32{Word.add(32n,Word.sub(32n,x,y),y)} == U32{_} : U32}
      {==}

def encoded_subtract_add(+a: Nat,+b: Nat) -> {U32.sub(U32.from_nat(Nat.add(a,b)),U32.from_nat(b)) == U32.from_nat(a) : U32}:
  %WordArithmetic.native_add(a,b) : {U32.sub(_,U32.from_nat(b)) == U32.from_nat(a) : U32}
  subtract_add(U32.from_nat(a),U32.from_nat(b))

def restore(+a: Nat,+b: Nat,h: {Nat.is_le(b,a) == True{} : Bool}) -> {Nat.add(Nat.sub(a,b),b) == a : Nat}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+b:
      Empty.absurd({Nat.add(Nat.sub(0n,1n+b),1n+b) == 0n : Nat},WordEncoding.false_true(h))
    case 1n+ +a 0n:
      Addition.zero_right(1n+a)
    case 1n+ +a 1n+ +b:
      %Equal.sym(Nat,Nat.add(Nat.sub(a,b),1n+b),1n+Nat.add(Nat.sub(a,b),b),Addition.successor_right(Nat.sub(a,b),b)) : {_ == 1n+a : Nat}
      %restore(a,b,h) : {1n+Nat.add(Nat.sub(a,b),b) == 1n+_ : Nat}
      {==}

def encoded_subtract(+a: Nat,+b: Nat,h: {Nat.is_le(b,a) == True{} : Bool}) -> {U32.sub(U32.from_nat(a),
    U32.from_nat(b)) == U32.from_nat(Nat.sub(a,b)) : U32}:
  %restore(a,b,h) : {U32.sub(U32.from_nat(_),U32.from_nat(b)) == U32.from_nat(Nat.sub(a,b)) : U32}
  encoded_subtract_add(Nat.sub(a,b),b)
