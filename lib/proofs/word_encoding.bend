# General-width bit-vector arithmetic used by the native-address obligation.
import Base

def concat(+n: Nat, +m: Nat, w: Word(n), tail: Word(m)) -> Word(Nat.add(n,m)):
  match n:
    case 0n:
      match w:
        case WNil{}:
          tail
    case 1n+p:
      match w:
        case WCon{b,rest}:
          WCon{b,concat(p,m,rest,tail)}

def shift(n: Nat, x: Nat) -> Nat:
  match n:
    case 0n:
      x
    case 1n+p:
      Nat.double(shift(p,x))

law double_add:
  for +a: Nat
  for +b: Nat
  {Nat.double(Nat.add(a,b)) == Nat.add(Nat.double(a),Nat.double(b)) : Nat}

def double_add(a,b):
  match a:
    case 0n:
      {==}
    case 1n+ +p:
      %double_add(p,b) : {2n+Nat.double(Nat.add(p,b)) == 2n+_ : Nat}
      {==}

law zero_nat:
  for +n: Nat
  {Word.to_nat(n,Word.zero(n)) == 0n : Nat}

def zero_nat(n):
  match n:
    case 0n:
      {==}
    case 1n+ +p:
      %zero_nat(p) : {Nat.double(Word.to_nat(p,Word.zero(p))) == Nat.double(_) : Nat}
      {==}

law finish_lt_zero:
  for +b: Bool
  for +c: Cmp
  {Cmp.is_lt(Word.cmp.fin(b,False{},c)) == Cmp.is_lt(c) : Bool}

def finish_lt_zero(b,c):
  match b c:
    case False{} LT{}:
      {==}
    case False{} EQ{}:
      {==}
    case False{} GT{}:
      {==}
    case True{} LT{}:
      {==}
    case True{} EQ{}:
      {==}
    case True{} GT{}:
      {==}

law compare_high_only:
  for +n: Nat
  for +m: Nat
  for +w: Word(n)
  for +top: Word(m)
  for +other: Word(m)
  {Cmp.is_lt(Word.cmp(Nat.add(n,m),concat(n,m,w,top),concat(n,m,Word.zero(n),other))) == Cmp.is_lt(Word.cmp(m,top,other)) : Bool}

def compare_high_only(n,m,w,top,other):
  match n:
    case 0n:
      match w:
        case WNil{}:
          {==}
    case 1n+ +p:
      match w:
        case WCon{+b,+rest}:
          %compare_high_only(p,m,rest,top,other) : {Cmp.is_lt(Word.cmp.fin(b,False{},Word.cmp(Nat.add(p,m),concat(p,m,rest,top),concat(p,m,
                  Word.zero(p),other)))) == _ : Bool}
          finish_lt_zero(b,Word.cmp(Nat.add(p,m),concat(p,m,rest,top),concat(p,m,Word.zero(p),other)))

law nonnegative:
  for +n: Nat
  for +w: Word(n)
  {Cmp.is_lt(Word.cmp(n,w,Word.zero(n))) == False{} : Bool}

def nonnegative(n,w):
  match n:
    case 0n:
      match w:
        case WNil{}:
          {==}
    case 1n+ +p:
      match w:
        case WCon{+b,+rest}:
          %nonnegative(p,rest) : {Cmp.is_lt(Word.cmp.fin(b,False{},Word.cmp(p,rest,Word.zero(p)))) == _ : Bool}
          finish_lt_zero(b,Word.cmp(p,rest,Word.zero(p)))

def truth(b: Bool) -> Type:
  match b:
    case False{}:
      Empty
    case True{}:
      Unit

def false_true(e: {False{} == True{} : Bool}) -> Empty:
  %Equal.sym(Bool,False{},True{},e) : truth(_)
  Unit{}

law finish_one:
  for +b: Bool
  for +c: Cmp
  for nonneg: {Cmp.is_lt(c) == False{} : Bool}
  {Cmp.is_lt(Word.cmp.fin(b,True{},c)) == Cmp.is_eq(Word.cmp.fin(b,False{},c)) : Bool}

def finish_one(b,c,nonneg):
  match b c:
    case False{} LT{}:
      Empty.absurd({True{} == False{} : Bool},false_true(Equal.sym(Bool,True{},False{},nonneg)))
    case False{} EQ{}:
      {==}
    case False{} GT{}:
      {==}
    case True{} LT{}:
      Empty.absurd({True{} == False{} : Bool},false_true(Equal.sym(Bool,True{},False{},nonneg)))
    case True{} EQ{}:
      {==}
    case True{} GT{}:
      {==}

law less_one_is_zero:
  for +n: Nat
  for +w: Word(1n+n)
  {Cmp.is_lt(Word.cmp(1n+n,w,WCon{True{},Word.zero(n)})) == Cmp.is_eq(Word.cmp(1n+n,w,Word.zero(1n+n))) : Bool}

def less_one_is_zero(n,w):
  match w:
    case WCon{b,+rest}:
      finish_one(b,Word.cmp(n,rest,Word.zero(n)),nonnegative(n,rest))

type Split<-n: Nat,-m: Nat> is Data:
  Parts{low: Word(n),high: Word(m)}

def join(+n: Nat, +m: Nat, parts: Split<n,m>) -> Word(Nat.add(n,m)):
  Parts{low,high} = parts
  concat(n,m,low,high)

def prepend(-n: Nat, -m: Nat, b: Bool, parts: Split<n,m>) -> Split<1n+n,m>:
  Parts{low,high} = parts
  Parts{WCon{b,low},high}

def split(+n: Nat, +m: Nat, w: Word(Nat.add(n,m))) -> Split<n,m>:
  match n:
    case 0n:
      Parts{WNil{},w}
    case 1n+ +p:
      match w:
        case WCon{b,rest}:
          prepend(p,m,b,split(p,m,rest))

law join_prepend:
  for +n: Nat
  for +m: Nat
  for +b: Bool
  for +parts: Split<n,m>
  {WCon{b,join(n,m,parts)} == join(1n+n,m,prepend(n,m,b,parts)) : Word(1n+Nat.add(n,m))}

def join_prepend(n,m,b,parts):
  Parts{low,high} = parts
  {==}

law recover:
  for +n: Nat
  for +m: Nat
  for +w: Word(Nat.add(n,m))
  {join(n,m,split(n,m,w)) == w : Word(Nat.add(n,m))}

def recover(n,m,w):
  match n:
    case 0n:
      {==}
    case 1n+ +p:
      match w:
        case WCon{+b,+rest}:
          %recover(p,m,rest) : {join(1n+p,m,prepend(p,m,b,split(p,m,rest))) == WCon{b,_} : Word(1n+Nat.add(p,m))}
          Equal.sym(Word(1n+Nat.add(p,m)),WCon{b,join(p,m,split(p,m,rest))},join(1n+p,m,prepend(p,m,b,split(p,m,rest))),join_prepend(p,m,b,
              split(p,m,rest)))

law cmp_eq:
  for +c: Cmp
  for yes: {Cmp.is_eq(c) == True{} : Bool}
  {c == EQ{} : Cmp}

def cmp_eq(c,yes):
  match c:
    case LT{}:
      Empty.absurd({LT{} == EQ{} : Cmp},false_true(yes))
    case EQ{}:
      {==}
    case GT{}:
      Empty.absurd({GT{} == EQ{} : Cmp},false_true(yes))

law finish_eq_parts:
  for +b: Bool
  for +c: Cmp
  for eq: {Word.cmp.fin(b,False{},c) == EQ{} : Cmp}
  {b == False{} : Bool} & {c == EQ{} : Cmp}

def finish_eq_parts(b,c,eq):
  match b c:
    case False{} LT{}:
      Empty.absurd({False{} == False{} : Bool} & {LT{} == EQ{} : Cmp},false_true(Equal.cong(Cmp,Bool,c => Cmp.is_eq(c),LT{},EQ{},eq)))
    case False{} EQ{}:
      ({==},{==})
    case False{} GT{}:
      Empty.absurd({False{} == False{} : Bool} & {GT{} == EQ{} : Cmp},false_true(Equal.cong(Cmp,Bool,c => Cmp.is_eq(c),GT{},EQ{},eq)))
    case True{} LT{}:
      Empty.absurd({True{} == False{} : Bool} & {LT{} == EQ{} : Cmp},false_true(Equal.cong(Cmp,Bool,c => Cmp.is_eq(c),LT{},EQ{},eq)))
    case True{} EQ{}:
      Empty.absurd({True{} == False{} : Bool} & {EQ{} == EQ{} : Cmp},false_true(Equal.cong(Cmp,Bool,c => Cmp.is_eq(c),GT{},EQ{},eq)))
    case True{} GT{}:
      Empty.absurd({True{} == False{} : Bool} & {GT{} == EQ{} : Cmp},false_true(Equal.cong(Cmp,Bool,c => Cmp.is_eq(c),GT{},EQ{},eq)))

law zero_reflect:
  for +n: Nat
  for +w: Word(n)
  for eq: {Word.cmp(n,w,Word.zero(n)) == EQ{} : Cmp}
  {w == Word.zero(n) : Word(n)}

def zero_step(+p: Nat, +b: Bool, +rest: Word(p), proof: {b == False{} : Bool} & {Word.cmp(p,rest,Word.zero(p)) == EQ{} : Cmp}, ih: {Word.cmp(p,
      rest,Word.zero(p)) == EQ{} : Cmp} -> {rest == Word.zero(p) : Word(p)}) -> {WCon{b,rest} == Word.zero(1n+p) : Word(1n+p)}:
  (head,tail) = proof
  %Equal.sym(Bool,b,False{},head) : {WCon{_,rest} == Word.zero(1n+p) : Word(1n+p)}
  %ih(tail) : {WCon{False{},rest} == WCon{False{},_} : Word(1n+p)}
  {==}

def zero_reflect(n,w,eq):
  match n:
    case 0n:
      match w:
        case WNil{}:
          {==}
    case 1n+ +p:
      match w:
        case WCon{+b,+rest}:
          zero_step(p,b,rest,finish_eq_parts(b,Word.cmp(p,rest,Word.zero(p)),eq),e => zero_reflect(p,rest,e))

law below_one_zero:
  for +n: Nat
  for +w: Word(1n+n)
  for below: {Cmp.is_lt(Word.cmp(1n+n,w,WCon{True{},Word.zero(n)})) == True{} : Bool}
  {w == Word.zero(1n+n) : Word(1n+n)}

def below_one_zero(n,w,below):
  zero_reflect(1n+n,w,cmp_eq(Word.cmp(1n+n,w,Word.zero(1n+n)),Equal.trans(Bool,Cmp.is_eq(Word.cmp(1n+n,w,Word.zero(1n+n))),
        Cmp.is_lt(Word.cmp(1n+n,w,WCon{True{},Word.zero(n)})),True{},Equal.sym(Bool,Cmp.is_lt(Word.cmp(1n+n,w,WCon{True{},Word.zero(n)})),
          Cmp.is_eq(Word.cmp(1n+n,w,Word.zero(1n+n))),less_one_is_zero(n,w)),below)))

law and_zero:
  for +n: Nat
  for +w: Word(n)
  {Word.and(n,w,Word.zero(n)) == Word.zero(n) : Word(n)}

def and_zero(n,w):
  match n:
    case 0n:
      match w:
        case WNil{}:
          {==}
    case 1n+ +p:
      match w:
        case WCon{False{},+rest}:
          %and_zero(p,rest) : {WCon{False{},Word.and(p,rest,Word.zero(p))} == WCon{False{},_} : Word(1n+p)}
          {==}
        case WCon{True{},+rest}:
          %and_zero(p,rest) : {WCon{False{},Word.and(p,rest,Word.zero(p))} == WCon{False{},_} : Word(1n+p)}
          {==}

law upper_zero:
  for +n: Nat
  for +m: Nat
  for +low: Word(n)
  for +high: Word(1n+m)
  for bound: {Cmp.is_lt(Word.cmp(Nat.add(n,1n+m),concat(n,1n+m,low,high),concat(n,1n+m,Word.zero(n),WCon{True{},Word.zero(m)}))) == True{} : Bool}
  {high == Word.zero(1n+m) : Word(1n+m)}

def upper_zero(n,m,low,high,bound):
  below_one_zero(m,high,Equal.trans(Bool,Cmp.is_lt(Word.cmp(1n+m,high,WCon{True{},Word.zero(m)})),Cmp.is_lt(Word.cmp(Nat.add(n,1n+m),concat(n,
          1n+m,low,high),concat(n,1n+m,Word.zero(n),WCon{True{},Word.zero(m)}))),True{},Equal.sym(Bool,Cmp.is_lt(Word.cmp(Nat.add(n,1n+m),
          concat(n,1n+m,low,high),concat(n,1n+m,Word.zero(n),WCon{True{},Word.zero(m)}))),Cmp.is_lt(Word.cmp(1n+m,high,WCon{True{},
          Word.zero(m)})),compare_high_only(n,1n+m,low,high,WCon{True{},Word.zero(m)})),bound))
