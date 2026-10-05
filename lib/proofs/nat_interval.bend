import Base
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./nat_extrema.bend as Extrema

def shifted_inside(+begin: Nat,+end: Nat,+query: Nat,+lower: {Nat.is_le(begin,query) == True{} : Bool},+upper: {Nat.is_lt(query,end) == True{} : Bool}) ->
  {Nat.is_lt(Nat.sub(query,begin),Nat.sub(end,begin)) == True{} : Bool}:
  +ordered = Bounds.strict_le(begin,end,Order.le_lt_trans(begin,query,end,lower,upper))
  Order.cancel_offset_lt(begin,Nat.sub(query,begin),Nat.sub(end,begin),
    %Equal.sym(Nat,Nat.add(begin,Nat.sub(query,begin)),query,Order.restore_shift(begin,query,lower)) : {Nat.is_lt(_,Nat.add(begin,Nat.sub(end,begin))) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(begin,Nat.sub(end,begin)),end,Order.restore_shift(begin,end,ordered)) : {Nat.is_lt(query,_) == True{} : Bool} upper)

def shifted_span(+first: Nat,+rows: Nat,+begin: Nat,+end: Nat,lower: {Nat.is_le(first,begin) == True{} : Bool},
  ordered: {Nat.is_le(begin,end) == True{} : Bool},upper: {Nat.is_le(end,Nat.add(first,rows)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.sub(begin,first),Nat.sub(end,begin)),rows) == True{} : Bool}:
  Order.cancel_offset_le(first,Nat.add(Nat.sub(begin,first),Nat.sub(end,begin)),rows,
    %Addition.associative(first,Nat.sub(begin,first),Nat.sub(end,begin)) : {Nat.is_le(_,Nat.add(first,rows)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(first,Nat.sub(begin,first)),begin,Order.restore_shift(first,begin,lower)) : {Nat.is_le(Nat.add(_,Nat.sub(end,begin)),Nat.add(first,rows)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(begin,Nat.sub(end,begin)),end,Order.restore_shift(begin,end,ordered)) : {Nat.is_le(_,Nat.add(first,rows)) == True{} : Bool} upper)

# A query lies strictly before or at/after the end of a half-open interval.
type Outside<-start: Nat,-length: Nat,-query: Nat> is Data:
  Before{bound: {Nat.is_lt(query,start) == True{} : Bool}}
  After{bound: {Nat.is_le(Nat.add(start,length),query) == True{} : Bool}}

def outside_head(+start: Nat,+remaining: Nat,+query: Nat,outside: Outside<start,1n+remaining,query>) ->
  {Nat.is_eq(query,start) == False{} : Bool}:
  match outside:
    case Before{bound}: Order.less_not_equal(query,start,bound)
    case After{bound}:
      %Equal.sym(Bool,Nat.is_eq(query,start),Nat.is_eq(start,query),Order.equal_symmetric(query,start)) : {_ == False{} : Bool}
      Order.less_not_equal(start,query,Order.room_index(start,remaining,query,bound))

def outside_tail(+start: Nat,+remaining: Nat,+query: Nat,outside: Outside<start,1n+remaining,query>) -> Outside<1n+start,remaining,query>:
  match outside:
    case Before{bound}: Before{Order.lt_succ(query,start,Bounds.strict_le(query,start,bound))}
    case After{bound}: After{Order.advance_room(start,remaining,query,bound)}
