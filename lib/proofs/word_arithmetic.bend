# Modular arithmetic identities used before applying bounded Nat conversion.
import Base
import ./natural_addition.bend as Addition
import ./u32_successor.bend as U32Successor

law carry:
  for +n: Nat
  for +a: Word(n)
  for +b: Word(n)
  {Word.adc(n,a,b,False{},True{}) == Word.inc(n,Word.add(n,a,b)) : Word(n)}
def carry(n,a,b):
  match n:
    case 0n:
      {==}
    case 1n+ +p:
      match a b:
        case WCon{False{},at} WCon{False{},bt}:
          {==}
        case WCon{False{},+at} WCon{True{},+bt}:
          %carry(p,at,bt) : {WCon{False{},Word.adc(p,at,bt,False{},True{})} == WCon{False{},_} : Word(1n+p)}
          {==}
        case WCon{True{},+at} WCon{False{},+bt}:
          %carry(p,at,bt) : {WCon{False{},Word.adc(p,at,bt,False{},True{})} == WCon{False{},_} : Word(1n+p)}
          {==}
        case WCon{True{},at} WCon{True{},bt}:
          {==}

law inc_left:
  for +n: Nat
  for +a: Word(n)
  for +b: Word(n)
  {Word.add(n,Word.inc(n,a),b) == Word.inc(n,Word.add(n,a,b)) : Word(n)}
def inc_left(n,a,b):
  match n:
    case 0n:
      {==}
    case 1n+ +p:
      match a b:
        case WCon{False{},at} WCon{False{},bt}:
          {==}
        case WCon{False{},+at} WCon{True{},+bt}:
          %carry(p,at,bt) : {WCon{False{},Word.adc(p,at,bt,False{},True{})} == WCon{False{},_} : Word(1n+p)}
          {==}
        case WCon{True{},+at} WCon{False{},+bt}:
          %inc_left(p,at,bt) : {WCon{False{},Word.add(p,Word.inc(p,at),bt)} == WCon{False{},_} : Word(1n+p)}
          {==}
        case WCon{True{},+at} WCon{True{},+bt}:
          %Equal.sym(Word(p),Word.adc(p,at,bt,False{},True{}),Word.inc(p,Word.add(p,at,bt)),carry(p,at,bt)) : {WCon{True{},Word.add(p,Word.inc(p,
                  at),bt)} == WCon{True{},_} : Word(1n+p)}
          %inc_left(p,at,bt) : {WCon{True{},Word.add(p,Word.inc(p,at),bt)} == WCon{True{},_} : Word(1n+p)}
          {==}

def encode(a: Nat,+width: Nat) -> Word(width):
  match a:
    case 0n:
      Word.zero(width)
    case 1n+p:
      Word.inc(width,encode(p,width))

law encode_add:
  for +a: Nat
  for +b: Nat
  for +width: Nat
  {Word.add(width,encode(a,width),encode(b,width)) == encode(Nat.add(a,b),width) : Word(width)}
def encode_add(a,b,width):
  match a:
    case 0n:
      %Equal.sym(Word(width),Word.add(width,Word.zero(width),encode(b,width)),Word.add(width,encode(b,width),Word.zero(width)),
        Word.add_comm(width,Word.zero(width),encode(b,width))) : {_ == encode(b,width) : Word(width)}
      U32Successor.adc_zero(width,encode(b,width))
    case 1n+ +p:
      %Equal.sym(Word(width),Word.add(width,Word.inc(width,encode(p,width)),encode(b,width)),Word.inc(width,Word.add(width,encode(p,width),
            encode(b,width))),inc_left(width,encode(p,width),encode(b,width))) : {_ == Word.inc(width,encode(Nat.add(p,b),width)) : Word(width)}
      %encode_add(p,b,width) : {Word.inc(width,Word.add(width,encode(p,width),encode(b,width))) == Word.inc(width,_) : Word(width)}
      {==}

law double_carry:
  for +n: Nat
  for +w: Word(n)
  for +c: Bool
  {Word.adc(n,w,w,False{},c) == Word.shl.put(n,c,w) : Word(n)}
def double_carry(n,w,c):
  match n:
    case 0n:
      {==}
    case 1n+ +p:
      match w c:
        case WCon{False{},+t} False{}:
          %double_carry(p,t,False{}) : {WCon{False{},Word.adc(p,t,t,False{},False{})} == WCon{False{},_} : Word(1n+p)}
          {==}
        case WCon{False{},+t} True{}:
          %double_carry(p,t,False{}) : {WCon{True{},Word.adc(p,t,t,False{},False{})} == WCon{True{},_} : Word(1n+p)}
          {==}
        case WCon{True{},+t} False{}:
          %double_carry(p,t,True{}) : {WCon{False{},Word.adc(p,t,t,False{},True{})} == WCon{False{},_} : Word(1n+p)}
          {==}
        case WCon{True{},+t} True{}:
          %double_carry(p,t,True{}) : {WCon{True{},Word.adc(p,t,t,False{},True{})} == WCon{True{},_} : Word(1n+p)}
          {==}

def shift_put(+n: Nat,+w: Word(n)) -> {Word.shl(n,w) == Word.shl.put(n,False{},w) : Word(n)}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      WCon{b,t} = w
      {==}

def double_word(+n: Nat,+w: Word(n)) -> {Word.add(n,w,w) == Word.shl(n,w) : Word(n)}:
  %Equal.sym(Word(n),Word.shl(n,w),Word.shl.put(n,False{},w),shift_put(n,w)) : {Word.add(n,w,w) == _ : Word(n)}
  double_carry(n,w,False{})

def encode_double(+a: Nat,+width: Nat) -> {encode(Nat.double(a),width) == Word.shl(width,encode(a,width)) : Word(width)}:
  %Addition.double(a) : {encode(_,width) == Word.shl(width,encode(a,width)) : Word(width)}
  %encode_add(a,a,width) : {_ == Word.shl(width,encode(a,width)) : Word(width)}
  double_word(width,encode(a,width))

law native_encoding:
  for +a: Nat
  {U32.from_nat(a) == U32{encode(a,32n)} : U32}
def native_encoding(a):
  match a:
    case 0n:
      {==}
    case 1n+ +p:
      %native_encoding(p) : {U32.inc(U32.from_nat(p)) == U32.inc(_) : U32}
      {==}

def native_add(+a: Nat,+b: Nat) -> {U32.add(U32.from_nat(a),U32.from_nat(b)) == U32.from_nat(Nat.add(a,b)) : U32}:
  %Equal.sym(U32,U32.from_nat(Nat.add(a,b)),U32{encode(Nat.add(a,b),32n)},native_encoding(Nat.add(a,b))) : {U32.add(U32.from_nat(a),
      U32.from_nat(b)) == _ : U32}
  %Equal.sym(U32,U32.from_nat(a),U32{encode(a,32n)},native_encoding(a)) : {U32.add(_,U32.from_nat(b)) == U32{encode(Nat.add(a,b),32n)} : U32}
  %Equal.sym(U32,U32.from_nat(b),U32{encode(b,32n)},native_encoding(b)) : {U32.add(U32{encode(a,32n)},_) == U32{encode(Nat.add(a,b),32n)} : U32}
  %encode_add(a,b,32n) : {U32{Word.add(32n,encode(a,32n),encode(b,32n))} == U32{_} : U32}
  {==}
