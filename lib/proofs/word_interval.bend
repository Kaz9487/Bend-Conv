# The real unsigned intersection either copies zero rows or stays within both
# source intervals. Endpoint addition and each subtraction have explicit room.
import Base
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_extrema.bend as U32Extrema
import ./nat_extrema.bend as Extrema
import ./u32_comparison.bend as Comparison
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_interval.bend as Interval
import ./natural_addition.bend as Addition
import ./strided_range_bounds.bend as Strided

type Span<-first: U32,-rows: U32,-begin: U32,-end: U32> is Data:
  Span{lower: {Nat.is_le(U32.to_nat(first),U32.to_nat(begin)) == True{} : Bool},
    shift_exact: {U32.to_nat((begin - first : U32)) == Nat.sub(U32.to_nat(begin),U32.to_nat(first)) : Nat},
    count_exact: {U32.to_nat((end - begin : U32)) == Nat.sub(U32.to_nat(end),U32.to_nat(begin)) : Nat},
    fits: {Nat.is_le(Nat.add(U32.to_nat((begin - first : U32)),U32.to_nat((end - begin : U32))),U32.to_nat(rows)) == True{} : Bool}}

def span(+first: U32,+rows: U32,+begin: U32,+end: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +lower: {Nat.is_le(U32.to_nat(first),U32.to_nat(begin)) == True{} : Bool},+ordered: {Nat.is_le(U32.to_nat(begin),U32.to_nat(end)) == True{} : Bool},
  upper: {Nat.is_le(U32.to_nat(end),Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(end),U32.to_nat(capacity)) == True{} : Bool}) -> Span<first,rows,begin,end>:
  +shift_exact = Arithmetic.subtract(begin,first,capacity,capacity_ok,lower,
    Order.transitive(Nat.sub(U32.to_nat(begin),U32.to_nat(first)),U32.to_nat(begin),U32.to_nat(capacity),Order.sub_le(U32.to_nat(begin),U32.to_nat(first)),Order.transitive(U32.to_nat(begin),U32.to_nat(end),U32.to_nat(capacity),ordered,room)))
  +count_exact = Arithmetic.subtract(end,begin,capacity,capacity_ok,ordered,
    Order.transitive(Nat.sub(U32.to_nat(end),U32.to_nat(begin)),U32.to_nat(end),U32.to_nat(capacity),Order.sub_le(U32.to_nat(end),U32.to_nat(begin)),room))
  Span{lower,shift_exact,count_exact,
    %Equal.sym(Nat,U32.to_nat((begin - first : U32)),Nat.sub(U32.to_nat(begin),U32.to_nat(first)),shift_exact) : {Nat.is_le(Nat.add(_,U32.to_nat((end - begin : U32))),U32.to_nat(rows)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat((end - begin : U32)),Nat.sub(U32.to_nat(end),U32.to_nat(begin)),count_exact) : {Nat.is_le(Nat.add(Nat.sub(U32.to_nat(begin),U32.to_nat(first)),_),U32.to_nat(rows)) == True{} : Bool}
    Interval.shifted_span(U32.to_nat(first),U32.to_nat(rows),U32.to_nat(begin),U32.to_nat(end),lower,ordered,upper)}

type Intersection<-first: U32,-rows: U32,-other_first: U32,-other_rows: U32,-begin: U32,-end: U32> is Data:
  EmptyIntersection{empty: {(end - begin : U32) == 0 : U32}}
  NonemptyIntersection{positive: {Nat.is_lt(U32.to_nat(begin),U32.to_nat(end)) == True{} : Bool},
    source: Span<first,rows,begin,end>,target: Span<other_first,other_rows,begin,end>}

def rows_positive(+first: U32,+rows: U32,+begin: U32,+end: U32,positive: {Nat.is_lt(U32.to_nat(begin),U32.to_nat(end)) == True{} : Bool},span: Span<first,rows,begin,end>) ->
  {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool}:
  Span{lower,shift_exact,count_exact,fits} = span
  Bounds.nat_trans(0n,U32.to_nat((end - begin : U32)),U32.to_nat(rows),
    %Equal.sym(Nat,U32.to_nat((end - begin : U32)),Nat.sub(U32.to_nat(end),U32.to_nat(begin)),count_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.positive_difference(U32.to_nat(end),U32.to_nat(begin),positive),
    Strided.count_in_stride(U32.to_nat((begin - first : U32)),U32.to_nat((end - begin : U32)),U32.to_nat(rows),fits))

def row_inside(+first: U32,+rows: U32,+begin: U32,+end: U32,+row: Nat,span: Span<first,rows,begin,end>,
  inside: {Nat.is_lt(row,U32.to_nat((end - begin : U32))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat((begin - first : U32)),row),U32.to_nat(rows)) == True{} : Bool}:
  Span{lower,shift_exact,count_exact,fits} = span
  Bounds.nat_trans(Nat.add(U32.to_nat((begin - first : U32)),row),Nat.add(U32.to_nat((begin - first : U32)),U32.to_nat((end - begin : U32))),U32.to_nat(rows),
    Order.offset_lt(U32.to_nat((begin - first : U32)),row,U32.to_nat((end - begin : U32)),inside),fits)

def located_row(+first: U32,+rows: U32,+begin: U32,+end: U32,+global_row: Nat,+row: Nat,span: Span<first,rows,begin,end>,
  located: {global_row == Nat.add(U32.to_nat(begin),row) : Nat}) ->
  {global_row == Nat.add(U32.to_nat(first),Nat.add(U32.to_nat((begin - first : U32)),row)) : Nat}:
  Span{lower,shift_exact,count_exact,fits} = span
  +begin_exact = Equal.trans(Nat,Nat.add(U32.to_nat(first),U32.to_nat((begin - first : U32))),Nat.add(U32.to_nat(first),Nat.sub(U32.to_nat(begin),U32.to_nat(first))),U32.to_nat(begin),
    Equal.cong(Nat,Nat,shift => Nat.add(U32.to_nat(first),shift),U32.to_nat((begin - first : U32)),Nat.sub(U32.to_nat(begin),U32.to_nat(first)),shift_exact),Order.restore_shift(U32.to_nat(first),U32.to_nat(begin),lower))
  Equal.trans(Nat,global_row,Nat.add(U32.to_nat(begin),row),Nat.add(U32.to_nat(first),Nat.add(U32.to_nat((begin - first : U32)),row)),located,
    %begin_exact : {Nat.add(_,row) == Nat.add(U32.to_nat(first),Nat.add(U32.to_nat((begin - first : U32)),row)) : Nat}
    Addition.associative(U32.to_nat(first),U32.to_nat((begin - first : U32)),row))

def subtract_self(+value: U32) -> {(value - value : U32) == 0 : U32}:
  Equal.trans(U32,(value - value : U32),U32.from_nat(Nat.sub(U32.to_nat(value),U32.to_nat(value))),0,Arithmetic.native_subtract(value,value,Order.reflexive(U32.to_nat(value))),
    Equal.cong(Nat,U32,U32.from_nat,Nat.sub(U32.to_nat(value),U32.to_nat(value)),0n,
      %Addition.zero_right(U32.to_nat(value)) : {Nat.sub(_,U32.to_nat(value)) == 0n : Nat}
      Order.subtract_shift(U32.to_nat(value),0n)))

def selected(choice: Bool,+first: U32,+rows: U32,+other_first: U32,+other_rows: U32,+begin: U32,+limit: U32,+capacity: U32,
  compared: {U32.is_lt(begin,limit) == choice : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  source_lower: {Nat.is_le(U32.to_nat(first),U32.to_nat(begin)) == True{} : Bool},target_lower: {Nat.is_le(U32.to_nat(other_first),U32.to_nat(begin)) == True{} : Bool},
  source_upper: {Nat.is_le(U32.to_nat(limit),Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool},
  target_upper: {Nat.is_le(U32.to_nat(limit),Nat.add(U32.to_nat(other_first),U32.to_nat(other_rows))) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(limit),U32.to_nat(capacity)) == True{} : Bool}) -> Intersection<first,rows,other_first,other_rows,begin,U32.max(begin,limit)>:
  match choice:
    case False{}:
      %U32Extrema.pick_maximum(begin,limit) : Intersection<first,rows,other_first,other_rows,begin,_>
      %Equal.sym(Bool,U32.is_lt(begin,limit),False{},compared) : Intersection<first,rows,other_first,other_rows,begin,Bool.pick(U32,_,limit,begin)>
      EmptyIntersection{subtract_self(begin)}
    case True{}:
      +compared = compared
      +positive = {
        %Comparison.lt(begin,limit) : {_ == True{} : Bool} compared : {Nat.is_lt(U32.to_nat(begin),U32.to_nat(limit)) == True{} : Bool}}
      +ordered = Bounds.strict_le(U32.to_nat(begin),U32.to_nat(limit),positive)
      %U32Extrema.pick_maximum(begin,limit) : Intersection<first,rows,other_first,other_rows,begin,_>
      %Equal.sym(Bool,U32.is_lt(begin,limit),True{},compared) : Intersection<first,rows,other_first,other_rows,begin,Bool.pick(U32,_,limit,begin)>
      NonemptyIntersection{positive,span(first,rows,begin,limit,capacity,capacity_ok,source_lower,ordered,source_upper,room),
        span(other_first,other_rows,begin,limit,capacity,capacity_ok,target_lower,ordered,target_upper,room)}

def intersection(+first: U32,+rows: U32,+other_first: U32,+other_rows: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +source_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(capacity)) == True{} : Bool},
  target_room: {Nat.is_le(Nat.add(U32.to_nat(other_first),U32.to_nat(other_rows)),U32.to_nat(capacity)) == True{} : Bool}) ->
  Intersection<first,rows,other_first,other_rows,U32.max(first,other_first),U32.max(U32.max(first,other_first),U32.min((first + rows : U32),(other_first + other_rows : U32)))>:
  +begin = U32.max(first,other_first)
  +limit = U32.min((first + rows : U32),(other_first + other_rows : U32))
  +begin_exact = U32Extrema.maximum(first,other_first)
  +source_end = Arithmetic.add(first,rows,capacity,capacity_ok,source_room)
  +target_end = Arithmetic.add(other_first,other_rows,capacity,capacity_ok,target_room)
  +limit_exact = U32Extrema.minimum((first + rows : U32),(other_first + other_rows : U32))
  +source_upper = {
    %Equal.sym(Nat,U32.to_nat(limit),Nat.min(U32.to_nat((first + rows : U32)),U32.to_nat((other_first + other_rows : U32))),limit_exact) : {Nat.is_le(_,Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}
    %source_end : {Nat.is_le(Nat.min(U32.to_nat((first + rows : U32)),U32.to_nat((other_first + other_rows : U32))),_) == True{} : Bool}
    Extrema.minimum_left(U32.to_nat((first + rows : U32)),U32.to_nat((other_first + other_rows : U32))) : {Nat.is_le(U32.to_nat(limit),Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}}
  selected(U32.is_lt(begin,limit),first,rows,other_first,other_rows,begin,limit,capacity,{==},capacity_ok,
    %Equal.sym(Nat,U32.to_nat(begin),Nat.max(U32.to_nat(first),U32.to_nat(other_first)),begin_exact) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool} Extrema.maximum_left(U32.to_nat(first),U32.to_nat(other_first)),
    %Equal.sym(Nat,U32.to_nat(begin),Nat.max(U32.to_nat(first),U32.to_nat(other_first)),begin_exact) : {Nat.is_le(U32.to_nat(other_first),_) == True{} : Bool} Extrema.maximum_right(U32.to_nat(first),U32.to_nat(other_first)),source_upper,
    %Equal.sym(Nat,U32.to_nat(limit),Nat.min(U32.to_nat((first + rows : U32)),U32.to_nat((other_first + other_rows : U32))),limit_exact) : {Nat.is_le(_,Nat.add(U32.to_nat(other_first),U32.to_nat(other_rows))) == True{} : Bool}
    %target_end : {Nat.is_le(Nat.min(U32.to_nat((first + rows : U32)),U32.to_nat((other_first + other_rows : U32))),_) == True{} : Bool}
    Extrema.minimum_right(U32.to_nat((first + rows : U32)),U32.to_nat((other_first + other_rows : U32))),
    Order.transitive(U32.to_nat(limit),Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(capacity),source_upper,source_room))

type Point<-first: U32,-rows: U32,-begin: U32,-end: U32,-local: Nat> is Data:
  Point{row: Nat,inside: {Nat.is_lt(row,U32.to_nat((end - begin : U32))) == True{} : Bool},
    local_exact: {Nat.add(U32.to_nat((begin - first : U32)),row) == local : Nat},span: Span<first,rows,begin,end>}

def located_point(+first: U32,+rows: U32,+begin: U32,+end: U32,+local: Nat,+span: Span<first,rows,begin,end>,
  +lower: {Nat.is_le(U32.to_nat(begin),Nat.add(U32.to_nat(first),local)) == True{} : Bool},
  upper: {Nat.is_lt(Nat.add(U32.to_nat(first),local),U32.to_nat(end)) == True{} : Bool}) -> Point<first,rows,begin,end,local>:
  Span{first_lower,shift_exact,count_exact,fits} = span
  +query = Nat.add(U32.to_nat(first),local)
  +row = Nat.sub(query,U32.to_nat(begin))
  +located = Equal.sym(Nat,Nat.add(U32.to_nat(begin),row),query,Order.restore_shift(U32.to_nat(begin),query,lower))
  Point{row,
    %Equal.sym(Nat,U32.to_nat((end - begin : U32)),Nat.sub(U32.to_nat(end),U32.to_nat(begin)),count_exact) : {Nat.is_lt(row,_) == True{} : Bool}
    Interval.shifted_inside(U32.to_nat(begin),U32.to_nat(end),query,lower,upper),
    Order.cancel_sum(U32.to_nat(first),Nat.add(U32.to_nat((begin - first : U32)),row),local,
      Equal.sym(Nat,query,Nat.add(U32.to_nat(first),Nat.add(U32.to_nat((begin - first : U32)),row)),located_row(first,rows,begin,end,query,row,span,located))),span}

def overlap_point(+first: U32,+rows: U32,+other_first: U32,+other_rows: U32,+local: Nat,+other_local: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  source_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(capacity)) == True{} : Bool},
  +target_room: {Nat.is_le(Nat.add(U32.to_nat(other_first),U32.to_nat(other_rows)),U32.to_nat(capacity)) == True{} : Bool},
  source_inside: {Nat.is_lt(local,U32.to_nat(rows)) == True{} : Bool},target_inside: {Nat.is_lt(other_local,U32.to_nat(other_rows)) == True{} : Bool},
  +located: {Nat.add(U32.to_nat(first),local) == Nat.add(U32.to_nat(other_first),other_local) : Nat}) ->
  Point<other_first,other_rows,U32.max(first,other_first),U32.max(U32.max(first,other_first),U32.min((first + rows : U32),(other_first + other_rows : U32))),other_local>:
  +begin = U32.max(first,other_first)
  +limit = U32.min((first + rows : U32),(other_first + other_rows : U32))
  +query = Nat.add(U32.to_nat(other_first),other_local)
  +source_end = Arithmetic.add(first,rows,capacity,capacity_ok,source_room)
  +target_end = Arithmetic.add(other_first,other_rows,capacity,capacity_ok,target_room)
  +begin_exact = U32Extrema.maximum(first,other_first)
  +limit_exact = U32Extrema.minimum((first + rows : U32),(other_first + other_rows : U32))
  +lower = {
    %Equal.sym(Nat,U32.to_nat(begin),Nat.max(U32.to_nat(first),U32.to_nat(other_first)),begin_exact) : {Nat.is_le(_,query) == True{} : Bool}
    Extrema.maximum_upper(U32.to_nat(first),U32.to_nat(other_first),query,
      %located : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool} Order.le_add(U32.to_nat(first),local),Order.le_add(U32.to_nat(other_first),other_local)) : {Nat.is_le(U32.to_nat(begin),query) == True{} : Bool}}
  +upper = {
    %Equal.sym(Nat,U32.to_nat(limit),Nat.min(U32.to_nat((first + rows : U32)),U32.to_nat((other_first + other_rows : U32))),limit_exact) : {Nat.is_lt(query,_) == True{} : Bool}
    Extrema.minimum_strict_lower(U32.to_nat((first + rows : U32)),U32.to_nat((other_first + other_rows : U32)),query,
      %Equal.sym(Nat,U32.to_nat((first + rows : U32)),Nat.add(U32.to_nat(first),U32.to_nat(rows)),source_end) : {Nat.is_lt(query,_) == True{} : Bool}
      %located : {Nat.is_lt(_,Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool} Order.offset_lt(U32.to_nat(first),local,U32.to_nat(rows),source_inside),
      %Equal.sym(Nat,U32.to_nat((other_first + other_rows : U32)),Nat.add(U32.to_nat(other_first),U32.to_nat(other_rows)),target_end) : {Nat.is_lt(query,_) == True{} : Bool}
      Order.offset_lt(U32.to_nat(other_first),other_local,U32.to_nat(other_rows),target_inside)) : {Nat.is_lt(query,U32.to_nat(limit)) == True{} : Bool}}
  +positive = Order.le_lt_trans(U32.to_nat(begin),query,U32.to_nat(limit),lower,upper)
  +compared = {
    %Equal.sym(Bool,U32.is_lt(begin,limit),Nat.is_lt(U32.to_nat(begin),U32.to_nat(limit)),Comparison.lt(begin,limit)) : {_ == True{} : Bool} positive : {U32.is_lt(begin,limit) == True{} : Bool}}
  +target_upper = {
    %Equal.sym(Nat,U32.to_nat(limit),Nat.min(U32.to_nat((first + rows : U32)),U32.to_nat((other_first + other_rows : U32))),limit_exact) : {Nat.is_le(_,Nat.add(U32.to_nat(other_first),U32.to_nat(other_rows))) == True{} : Bool}
    %target_end : {Nat.is_le(Nat.min(U32.to_nat((first + rows : U32)),U32.to_nat((other_first + other_rows : U32))),_) == True{} : Bool}
    Extrema.minimum_right(U32.to_nat((first + rows : U32)),U32.to_nat((other_first + other_rows : U32))) : {Nat.is_le(U32.to_nat(limit),Nat.add(U32.to_nat(other_first),U32.to_nat(other_rows))) == True{} : Bool}}
  %U32Extrema.pick_maximum(begin,limit) : Point<other_first,other_rows,begin,_,other_local>
  %Equal.sym(Bool,U32.is_lt(begin,limit),True{},compared) : Point<other_first,other_rows,begin,Bool.pick(U32,_,limit,begin),other_local>
  located_point(other_first,other_rows,begin,limit,other_local,
    span(other_first,other_rows,begin,limit,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat(begin),Nat.max(U32.to_nat(first),U32.to_nat(other_first)),begin_exact) : {Nat.is_le(U32.to_nat(other_first),_) == True{} : Bool} Extrema.maximum_right(U32.to_nat(first),U32.to_nat(other_first)),
      Bounds.strict_le(U32.to_nat(begin),U32.to_nat(limit),positive),target_upper,
      Order.transitive(U32.to_nat(limit),Nat.add(U32.to_nat(other_first),U32.to_nat(other_rows)),U32.to_nat(capacity),target_upper,target_room)),lower,upper)
