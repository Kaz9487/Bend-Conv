# Minimum and maximum laws used by interval intersection. Proofs follow the
# official structural definitions; no bounded dimension is enumerated.
import Base
import ./nat_order.bend as Order
import ./word_encoding.bend as WordEncoding

def successor_pick(choice: Bool,+left: Nat,+right: Nat) ->
  {1n+Bool.pick(Nat,choice,left,right) == Bool.pick(Nat,choice,1n+left,1n+right) : Nat}:
  match choice:
    case False{}: {==}
    case True{}: {==}

def minimum_choice(+left: Nat,+right: Nat) -> {Nat.min(left,right) == Bool.pick(Nat,Nat.is_lt(left,right),left,right) : Nat}:
  match left right:
    case 0n 0n: {==}
    case 0n 1n+right: {==}
    case 1n+left 0n: {==}
    case 1n+ +left 1n+ +right:
      Equal.trans(Nat,1n+Nat.min(left,right),1n+Bool.pick(Nat,Nat.is_lt(left,right),left,right),Bool.pick(Nat,Nat.is_lt(left,right),1n+left,1n+right),
        Equal.cong(Nat,Nat,n => 1n+n,Nat.min(left,right),Bool.pick(Nat,Nat.is_lt(left,right),left,right),minimum_choice(left,right)),successor_pick(Nat.is_lt(left,right),left,right))

def maximum_choice(+left: Nat,+right: Nat) -> {Nat.max(left,right) == Bool.pick(Nat,Nat.is_lt(left,right),right,left) : Nat}:
  match left right:
    case 0n 0n: {==}
    case 0n 1n+right: {==}
    case 1n+left 0n: {==}
    case 1n+ +left 1n+ +right:
      Equal.trans(Nat,1n+Nat.max(left,right),1n+Bool.pick(Nat,Nat.is_lt(left,right),right,left),Bool.pick(Nat,Nat.is_lt(left,right),1n+right,1n+left),
        Equal.cong(Nat,Nat,n => 1n+n,Nat.max(left,right),Bool.pick(Nat,Nat.is_lt(left,right),right,left),maximum_choice(left,right)),successor_pick(Nat.is_lt(left,right),right,left))

def minimum_left(+left: Nat,+right: Nat) -> {Nat.is_le(Nat.min(left,right),left) == True{} : Bool}:
  match left right:
    case 0n right: {==}
    case 1n+left 0n: {==}
    case 1n+left 1n+right: minimum_left(left,right)

def minimum_right(+left: Nat,+right: Nat) -> {Nat.is_le(Nat.min(left,right),right) == True{} : Bool}:
  match left right:
    case 0n right: Order.zero_le(right)
    case 1n+left 0n: {==}
    case 1n+left 1n+right: minimum_right(left,right)

def maximum_left(+left: Nat,+right: Nat) -> {Nat.is_le(left,Nat.max(left,right)) == True{} : Bool}:
  match left right:
    case 0n right: Order.zero_le(right)
    case 1n+ +left 0n: Order.reflexive(left)
    case 1n+left 1n+right: maximum_left(left,right)

def maximum_right(+left: Nat,+right: Nat) -> {Nat.is_le(right,Nat.max(left,right)) == True{} : Bool}:
  match left right:
    case 0n +right: Order.reflexive(right)
    case 1n+left 0n: {==}
    case 1n+left 1n+right: maximum_right(left,right)

def pick_upper(choice: Bool,+left: Nat,+right: Nat,+upper: Nat,left_ok: {Nat.is_le(left,upper) == True{} : Bool},right_ok: {Nat.is_le(right,upper) == True{} : Bool}) ->
  {Nat.is_le(Bool.pick(Nat,choice,left,right),upper) == True{} : Bool}:
  match choice:
    case False{}: right_ok
    case True{}: left_ok

def pick_lower(choice: Bool,+left: Nat,+right: Nat,+lower: Nat,left_ok: {Nat.is_le(lower,left) == True{} : Bool},right_ok: {Nat.is_le(lower,right) == True{} : Bool}) ->
  {Nat.is_le(lower,Bool.pick(Nat,choice,left,right)) == True{} : Bool}:
  match choice:
    case False{}: right_ok
    case True{}: left_ok

def pick_strict_lower(choice: Bool,+left: Nat,+right: Nat,+lower: Nat,left_ok: {Nat.is_lt(lower,left) == True{} : Bool},right_ok: {Nat.is_lt(lower,right) == True{} : Bool}) ->
  {Nat.is_lt(lower,Bool.pick(Nat,choice,left,right)) == True{} : Bool}:
  match choice:
    case False{}: right_ok
    case True{}: left_ok

def maximum_upper(+left: Nat,+right: Nat,+upper: Nat,left_ok: {Nat.is_le(left,upper) == True{} : Bool},right_ok: {Nat.is_le(right,upper) == True{} : Bool}) ->
  {Nat.is_le(Nat.max(left,right),upper) == True{} : Bool}:
  %Equal.sym(Nat,Nat.max(left,right),Bool.pick(Nat,Nat.is_lt(left,right),right,left),maximum_choice(left,right)) : {Nat.is_le(_,upper) == True{} : Bool}
  pick_upper(Nat.is_lt(left,right),right,left,upper,right_ok,left_ok)

def minimum_lower(+left: Nat,+right: Nat,+lower: Nat,left_ok: {Nat.is_le(lower,left) == True{} : Bool},right_ok: {Nat.is_le(lower,right) == True{} : Bool}) ->
  {Nat.is_le(lower,Nat.min(left,right)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.min(left,right),Bool.pick(Nat,Nat.is_lt(left,right),left,right),minimum_choice(left,right)) : {Nat.is_le(lower,_) == True{} : Bool}
  pick_lower(Nat.is_lt(left,right),left,right,lower,left_ok,right_ok)

def minimum_strict_lower(+left: Nat,+right: Nat,+lower: Nat,left_ok: {Nat.is_lt(lower,left) == True{} : Bool},right_ok: {Nat.is_lt(lower,right) == True{} : Bool}) ->
  {Nat.is_lt(lower,Nat.min(left,right)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.min(left,right),Bool.pick(Nat,Nat.is_lt(left,right),left,right),minimum_choice(left,right)) : {Nat.is_lt(lower,_) == True{} : Bool}
  pick_strict_lower(Nat.is_lt(left,right),left,right,lower,left_ok,right_ok)

def minimum_monotone(+left: Nat,+right: Nat,+limit: Nat,ordered: {Nat.is_le(left,right) == True{} : Bool}) ->
  {Nat.is_le(Nat.min(limit,left),Nat.min(limit,right)) == True{} : Bool}:
  minimum_lower(limit,right,Nat.min(limit,left),minimum_left(limit,left),Order.transitive(Nat.min(limit,left),left,right,minimum_right(limit,left),ordered))

def clipped_end(+first: Nat,+end: Nat) -> {Nat.add(first,Nat.sub(end,first)) == Nat.max(first,end) : Nat}:
  match first end:
    case 0n +end: Order.subtract_zero(end)
    case 1n+ +first 0n: Order.commutative(1n+first,0n)
    case 1n+ +first 1n+ +end:
      Equal.cong(Nat,Nat,value => 1n+value,Nat.add(first,Nat.sub(end,first)),Nat.max(first,end),clipped_end(first,end))
