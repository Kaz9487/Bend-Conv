import Base
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding
import ./word_width.bend as WordWidth

law pad_nat:
  for +n: Nat
  for +w: Word(n)
  {Word.to_nat(1n+n,Word.shr.pad(n,w)) == Word.to_nat(n,w) : Nat}
def pad_nat(n,w):
  match n:
    case 0n:
      match w:
        case WNil{}: {==}
    case 1n+ +p:
      match w:
        case WCon{False{},+t}:
          %pad_nat(p,t) : {Nat.double(Word.to_nat(1n+p,Word.shr.pad(p,t))) == Nat.double(_) : Nat}
          {==}
        case WCon{True{},+t}:
          %pad_nat(p,t) : {1n+Nat.double(Word.to_nat(1n+p,Word.shr.pad(p,t))) == 1n+Nat.double(_) : Nat}
          {==}

law inc_nat:
  for +n: Nat
  for +w: Word(n)
  {Word.to_nat(1n+n,Word.inc(1n+n,Word.shr.pad(n,w))) == 1n+Word.to_nat(n,w) : Nat}
def inc_nat(n,w):
  match n:
    case 0n:
      match w:
        case WNil{}: {==}
    case 1n+ +p:
      match w:
        case WCon{False{},+t}:
          %pad_nat(p,t) : {1n+Nat.double(Word.to_nat(1n+p,Word.shr.pad(p,t))) == 1n+Nat.double(_) : Nat}
          {==}
        case WCon{True{},+t}:
          %inc_nat(p,t) : {Nat.double(Word.to_nat(1n+p,Word.inc(1n+p,Word.shr.pad(p,t)))) == Nat.double(_) : Nat}
          {==}

law adc_zero:
  for +n: Nat
  for +w: Word(n)
  {Word.adc(n,w,Word.zero(n),False{},False{}) == w : Word(n)}
def adc_zero(n,w):
  match n:
    case 0n:
      WNil{} = w
      {==}
    case 1n+ +p:
      match w:
        case WCon{False{},+t}:
          %adc_zero(p,t) : {WCon{False{},Word.adc(p,t,Word.zero(p),False{},False{})} == WCon{False{},_} : Word(1n+p)}
          {==}
        case WCon{True{},+t}:
          %adc_zero(p,t) : {WCon{True{},Word.adc(p,t,Word.zero(p),False{},False{})} == WCon{True{},_} : Word(1n+p)}
          {==}

law adc_one:
  for +n: Nat
  for +w: Word(n)
  {Word.adc(n,w,Word.zero(n),False{},True{}) == Word.inc(n,w) : Word(n)}
def adc_one(n,w):
  match n:
    case 0n:
      WNil{} = w
      {==}
    case 1n+ +p:
      match w:
        case WCon{False{},+t}:
          %adc_zero(p,t) : {WCon{True{},Word.adc(p,t,Word.zero(p),False{},False{})} == WCon{True{},_} : Word(1n+p)}
          {==}
        case WCon{True{},+t}:
          %adc_one(p,t) : {WCon{False{},Word.adc(p,t,Word.zero(p),False{},True{})} == WCon{False{},_} : Word(1n+p)}
          {==}

law add_one:
  for +n: Nat
  for +w: Word(n)
  {Word.add(n,w,Word.inc(n,Word.zero(n))) == Word.inc(n,w) : Word(n)}
def add_one(n,w):
  match n:
    case 0n:
      {==}
    case 1n+ +p:
      match w:
        case WCon{False{},+t}:
          %adc_zero(p,t) : {WCon{True{},Word.adc(p,t,Word.zero(p),False{},False{})} == WCon{True{},_} : Word(1n+p)}
          {==}
        case WCon{True{},+t}:
          %adc_one(p,t) : {WCon{False{},Word.adc(p,t,Word.zero(p),False{},True{})} == WCon{False{},_} : Word(1n+p)}
          {==}

law concat_pad:
  for +width: Nat
  for +word: Word(width)
  {WordWidth.transport(Nat.add(width,1n),1n+width,WordEncoding.concat(width,1n,word,Word.zero(1n)),
      Addition.successor_width(width)) == Word.shr.pad(width,word) : Word(1n+width)}

def concat_pad(width,word):
  match width:
    case 0n:
      match word:
        case WNil{}:
          {==}
    case 1n+ +rest:
      match word:
        case WCon{+bit,+tail}:
          Equal.trans(Word(2n+rest),
            WordWidth.transport(1n+Nat.add(rest,1n),2n+rest,WCon{bit,WordEncoding.concat(rest,1n,tail,Word.zero(1n))},
              Addition.successor_width(1n+rest)),
            WCon{bit,WordWidth.transport(Nat.add(rest,1n),1n+rest,WordEncoding.concat(rest,1n,tail,Word.zero(1n)),
                Addition.successor_width(rest))},
            WCon{bit,Word.shr.pad(rest,tail)},
            WordWidth.transport_cons(Nat.add(rest,1n),1n+rest,bit,WordEncoding.concat(rest,1n,tail,Word.zero(1n)),
              Addition.successor_width(rest)),
            Equal.cong(Word(1n+rest),Word(2n+rest),w => WCon{bit,w},WordWidth.transport(Nat.add(rest,1n),1n+rest,WordEncoding.concat(rest,1n,
                  tail,Word.zero(1n)),Addition.successor_width(rest)),Word.shr.pad(rest,tail),concat_pad(rest,tail)))

def padded_successor(+low: Word(31n)) -> {U32.to_nat(U32.add(U32{Word.shr.pad(31n,low)},1)) == 1n+U32.to_nat(U32{Word.shr.pad(31n,low)}) : Nat}:
  %Equal.sym(Word(32n),Word.add(32n,Word.shr.pad(31n,low),Word.inc(32n,Word.zero(32n))),Word.inc(32n,Word.shr.pad(31n,low)),add_one(32n,
      Word.shr.pad(31n,low))) : {Word.to_nat(32n,_) == 1n+Word.to_nat(32n,Word.shr.pad(31n,low)) : Nat}
  %Equal.sym(Nat,Word.to_nat(32n,Word.shr.pad(31n,low)),Word.to_nat(31n,low),pad_nat(31n,low)) : {Word.to_nat(32n,Word.inc(32n,Word.shr.pad(31n,
          low))) == 1n+_ : Nat}
  inc_nat(31n,low)

def Goal(+i: U32) -> Type:
  {U32.is_lt(i,2147483648) == True{} : Bool} -> {U32.to_nat((i + 1 : U32)) == 1n+U32.to_nat(i) : Nat}

def parts(+parts: WordEncoding.Split<31n,1n>) -> Goal(U32{WordEncoding.join(31n,1n,parts)}):
  WordEncoding.Parts{+low,+high} = parts
  bound =>
    %Equal.sym(Word(1n),high,Word.zero(1n),WordEncoding.upper_zero(31n,0n,low,high,bound)) : {U32.to_nat(U32.add(U32{WordEncoding.concat(31n,1n,
            low,_)},1)) == 1n+U32.to_nat(U32{WordEncoding.concat(31n,1n,low,_)}) : Nat}
    %Equal.sym(Word(32n),WordEncoding.concat(31n,1n,low,Word.zero(1n)),Word.shr.pad(31n,low),concat_pad(31n,low)) : {U32.to_nat(U32.add(U32{_},
          1)) == 1n+U32.to_nat(U32{_}) : Nat}
    padded_successor(low)

def word(+w: Word(32n)) -> Goal(U32{w}):
  %WordEncoding.recover(31n,1n,w) : Goal(U32{_})
  parts(WordEncoding.split(31n,1n,w))

law successor:
  for +i: U32
  Goal(i)
def successor(i):
  match i:
    case U32{w}:
      word(w)
