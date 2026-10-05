# Natural division by a positive divisor: the quotient/remainder identity and
# monotone quotients, plus monotone truncated subtraction.
import Base
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./word_encoding.bend as WordEncoding

type Witness<-value: Nat,-divisor: Nat> is Data:
  Witness{quotient: Nat,remainder: Nat,exact: {value == Nat.add(Nat.mul(quotient,divisor),remainder) : Nat},
    small: {Nat.is_lt(remainder,divisor) == True{} : Bool}}

def not_below(+value: Nat,+limit: Nat,+below: {Nat.is_lt(value,limit) == True{} : Bool},+outside: {Nat.is_lt(1n+value,limit) == False{} : Bool}) ->
  {1n+value == limit : Nat}:
  match value limit:
    case 0n 0n: Empty.absurd({1n == 0n : Nat},WordEncoding.false_true(below))
    case 1n+rest 0n: Empty.absurd({2n+rest == 0n : Nat},WordEncoding.false_true(below))
    case 0n 1n+ +rest:
      match rest:
        case 0n: {==}
        case 1n+more: Empty.absurd({1n == 2n+more : Nat},WordEncoding.false_true(Equal.sym(Bool,True{},False{},outside)))
    case 1n+ +left 1n+ +right:
      Equal.cong(Nat,Nat,n => 1n+n,1n+left,right,not_below(left,right,below,outside))

def advanced(+value: Nat,+divisor: Nat,+quotient: Nat,+remainder: Nat,+inside: Bool,
  +exact: {value == Nat.add(Nat.mul(quotient,divisor),remainder) : Nat},+small: {Nat.is_lt(remainder,divisor) == True{} : Bool},
  +positive: {Nat.is_lt(0n,divisor) == True{} : Bool},+next: {Nat.is_lt(1n+remainder,divisor) == inside : Bool}) -> Witness<1n+value,divisor>:
  match inside:
    case True{}:
      Witness{quotient,1n+remainder,
        %Equal.sym(Nat,value,Nat.add(Nat.mul(quotient,divisor),remainder),exact) : {1n+_ == Nat.add(Nat.mul(quotient,divisor),1n+remainder) : Nat}
        Equal.sym(Nat,Nat.add(Nat.mul(quotient,divisor),1n+remainder),1n+Nat.add(Nat.mul(quotient,divisor),remainder),
          Addition.successor_right(Nat.mul(quotient,divisor),remainder)),
        next}
    case False{}:
      +full = not_below(remainder,divisor,small,next)
      Witness{1n+quotient,0n,
        %Equal.sym(Nat,value,Nat.add(Nat.mul(quotient,divisor),remainder),exact) : {1n+_ == Nat.add(Nat.add(divisor,Nat.mul(quotient,divisor)),0n) : Nat}
        %Equal.sym(Nat,Nat.add(Nat.add(divisor,Nat.mul(quotient,divisor)),0n),Nat.add(divisor,Nat.mul(quotient,divisor)),Addition.zero_right(Nat.add(divisor,Nat.mul(quotient,divisor)))) :
          {1n+Nat.add(Nat.mul(quotient,divisor),remainder) == _ : Nat}
        %Order.commutative(Nat.mul(quotient,divisor),divisor) : {1n+Nat.add(Nat.mul(quotient,divisor),remainder) == _ : Nat}
        %full : {1n+Nat.add(Nat.mul(quotient,divisor),remainder) == Nat.add(Nat.mul(quotient,divisor),_) : Nat}
        Equal.sym(Nat,Nat.add(Nat.mul(quotient,divisor),1n+remainder),1n+Nat.add(Nat.mul(quotient,divisor),remainder),
          Addition.successor_right(Nat.mul(quotient,divisor),remainder)),
        positive}

def successor(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},+previous: Witness<value,divisor>) -> Witness<1n+value,divisor>:
  Witness{+quotient,+remainder,+exact,+small} = previous
  advanced(value,divisor,quotient,remainder,Nat.is_lt(1n+remainder,divisor),exact,small,positive,{==})

def witness(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) -> Witness<value,divisor>:
  match value:
    case 0n: Witness{0n,0n,{==},positive}
    case 1n+ +rest: successor(rest,divisor,positive,witness(rest,divisor,positive))

def quotient_exact_of(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},+found: Witness<value,divisor>) ->
  {value == Nat.add(Nat.mul(Nat.div(value,divisor),divisor),Nat.mod(value,divisor)) : Nat}:
  Witness{+quotient,+remainder,+exact,+small} = found
  %Equal.sym(Nat & Nat,Nat.divmod(value,divisor),(quotient,remainder),Division.unique(value,divisor,quotient,remainder,exact,positive,small)) :
    {value == Nat.add(Nat.mul(Pair.fst(Nat,Nat,_),divisor),Pair.snd(Nat,Nat,_)) : Nat}
  exact

def quotient_exact(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {value == Nat.add(Nat.mul(Nat.div(value,divisor),divisor),Nat.mod(value,divisor)) : Nat}:
  quotient_exact_of(value,divisor,positive,witness(value,divisor,positive))

def remainder_small_of(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},+found: Witness<value,divisor>) ->
  {Nat.is_lt(Nat.mod(value,divisor),divisor) == True{} : Bool}:
  Witness{+quotient,+remainder,+exact,+small} = found
  %Equal.sym(Nat & Nat,Nat.divmod(value,divisor),(quotient,remainder),Division.unique(value,divisor,quotient,remainder,exact,positive,small)) :
    {Nat.is_lt(Pair.snd(Nat,Nat,_),divisor) == True{} : Bool}
  small

def remainder_small(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mod(value,divisor),divisor) == True{} : Bool}:
  remainder_small_of(value,divisor,positive,witness(value,divisor,positive))

def product_below(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Nat.div(value,divisor),divisor),value) == True{} : Bool}:
  %Equal.sym(Nat,value,Nat.add(Nat.mul(Nat.div(value,divisor),divisor),Nat.mod(value,divisor)),quotient_exact(value,divisor,positive)) :
    {Nat.is_le(Nat.mul(Nat.div(value,divisor),divisor),_) == True{} : Bool}
  Order.le_add(Nat.mul(Nat.div(value,divisor),divisor),Nat.mod(value,divisor))

def next_above(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.is_lt(value,Nat.mul(1n+Nat.div(value,divisor),divisor)) == True{} : Bool}:
  +quotient = Nat.div(value,divisor)
  +remainder = Nat.mod(value,divisor)
  %Equal.sym(Nat,value,Nat.add(Nat.mul(quotient,divisor),remainder),quotient_exact(value,divisor,positive)) :
    {Nat.is_lt(_,Nat.add(divisor,Nat.mul(quotient,divisor))) == True{} : Bool}
  %Order.commutative(Nat.mul(quotient,divisor),divisor) :
    {Nat.is_lt(Nat.add(Nat.mul(quotient,divisor),remainder),_) == True{} : Bool}
  %Order.commutative(remainder,Nat.mul(quotient,divisor)) : {Nat.is_lt(_,Nat.add(Nat.mul(quotient,divisor),divisor)) == True{} : Bool}
  %Order.commutative(divisor,Nat.mul(quotient,divisor)) : {Nat.is_lt(Nat.add(remainder,Nat.mul(quotient,divisor)),_) == True{} : Bool}
  Order.add_lt(remainder,divisor,Nat.mul(quotient,divisor),remainder_small(value,divisor,positive))

def quotient_le(+smaller: Nat,+larger: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},
  +ordered: {Nat.is_le(smaller,larger) == True{} : Bool}) -> {Nat.is_le(Nat.div(smaller,divisor),Nat.div(larger,divisor)) == True{} : Bool}:
  Division.lt_succ_le(Nat.div(smaller,divisor),Nat.div(larger,divisor),
    RowMajor.cancel_multiply_lt(Nat.div(smaller,divisor),1n+Nat.div(larger,divisor),divisor,
      Order.le_lt_trans(Nat.mul(Nat.div(smaller,divisor),divisor),larger,Nat.mul(1n+Nat.div(larger,divisor),divisor),
        Order.transitive(Nat.mul(Nat.div(smaller,divisor),divisor),smaller,larger,product_below(smaller,divisor,positive),ordered),
        next_above(larger,divisor,positive))))

def subtract_le(+smaller: Nat,+larger: Nat,+amount: Nat,+ordered: {Nat.is_le(smaller,larger) == True{} : Bool}) ->
  {Nat.is_le(Nat.sub(smaller,amount),Nat.sub(larger,amount)) == True{} : Bool}:
  match smaller larger amount:
    case 0n 0n 0n: {==}
    case 0n 0n 1n+rest: {==}
    case 0n 1n+above 0n: {==}
    case 0n 1n+ +above 1n+ +rest: Order.zero_le(Nat.sub(above,rest))
    case 1n+left 0n 0n: Empty.absurd({Nat.is_le(1n+left,0n) == True{} : Bool},WordEncoding.false_true(ordered))
    case 1n+left 0n 1n+rest: Empty.absurd({Nat.is_le(Nat.sub(left,rest),0n) == True{} : Bool},WordEncoding.false_true(ordered))
    case 1n+left 1n+right 0n: ordered
    case 1n+ +left 1n+ +right 1n+ +rest: subtract_le(left,right,rest,ordered)
