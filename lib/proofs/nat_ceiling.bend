# Ceiling division is the least multiple covering a value. This contract is
# independent of window geometry and of any machine-word representation.
import Base
import ./word_encoding.bend as WordEncoding
import ./nat_order.bend as Order
import ./nat_division_order.bend as DivisionOrder
import ./nat_division.bend as Division
import ./row_major_index.bend as RowMajor
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition

def rounded(quotient: Nat,remainder: Nat) -> Nat:
  match remainder:
    case 0n: quotient
    case 1n+rest: 1n+quotient

def ceiling(+value: Nat,+divisor: Nat) -> Nat:
  rounded(Nat.div(value,divisor),Nat.mod(value,divisor))

def rounded_choice(+quotient: Nat,remainder: Nat) ->
  {rounded(quotient,remainder) == Nat.add(quotient,Bool.pick(Nat,Nat.is_eq(remainder,0n),0n,1n)) : Nat}:
  match remainder:
    case 0n: Equal.sym(Nat,Nat.add(quotient,0n),quotient,Addition.zero_right(quotient))
    case 1n+rest:
      %Equal.sym(Nat,Nat.add(quotient,1n),1n+Nat.add(quotient,0n),Addition.successor_right(quotient,0n)) : {1n+quotient == _ : Nat}
      Equal.cong(Nat,Nat,value => 1n+value,quotient,Nat.add(quotient,0n),Equal.sym(Nat,Nat.add(quotient,0n),quotient,Addition.zero_right(quotient)))

def cancel_product_le(+left: Nat,+right: Nat,+factor: Nat,positive: {Nat.is_lt(0n,factor) == True{} : Bool},ordered: {Nat.is_le(Nat.mul(left,factor),Nat.mul(right,factor)) == True{} : Bool}) ->
  {Nat.is_le(left,right) == True{} : Bool}:
  Division.lt_succ_le(left,right,RowMajor.cancel_multiply_lt(left,1n+right,factor,
    Order.le_lt_trans(Nat.mul(left,factor),Nat.mul(right,factor),Nat.mul(1n+right,factor),ordered,
      RowMajor.multiply_lt(right,1n+right,factor,Order.lt_succ(right,right,Order.reflexive(right)),positive))))

def rounded_covers(+quotient: Nat,remainder: Nat,+divisor: Nat,small: {Nat.is_lt(remainder,divisor) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(quotient,divisor),remainder),Nat.mul(rounded(quotient,remainder),divisor)) == True{} : Bool}:
  match remainder:
    case 0n:
      %Equal.sym(Nat,Nat.add(Nat.mul(quotient,divisor),0n),Nat.mul(quotient,divisor),Addition.zero_right(Nat.mul(quotient,divisor))) : {Nat.is_le(_,Nat.mul(quotient,divisor)) == True{} : Bool}
      Order.reflexive(Nat.mul(quotient,divisor))
    case 1n+ +rest:
      %Order.commutative(Nat.mul(quotient,divisor),divisor) : {Nat.is_le(Nat.add(Nat.mul(quotient,divisor),1n+rest),_) == True{} : Bool}
      Order.offset_le(Nat.mul(quotient,divisor),1n+rest,divisor,Bounds.strict_le(1n+rest,divisor,small))

def rounded_minimal(+quotient: Nat,remainder: Nat,+divisor: Nat,+upper: Nat,positive: {Nat.is_lt(0n,divisor) == True{} : Bool},
  bound: {Nat.is_le(Nat.add(Nat.mul(quotient,divisor),remainder),Nat.mul(upper,divisor)) == True{} : Bool}) ->
  {Nat.is_le(rounded(quotient,remainder),upper) == True{} : Bool}:
  match remainder:
    case 0n:
      cancel_product_le(quotient,upper,divisor,positive,
        %Addition.zero_right(Nat.mul(quotient,divisor)) : {Nat.is_le(_,Nat.mul(upper,divisor)) == True{} : Bool} bound)
    case 1n+ +rest:
      Order.strict_follow(quotient,upper,RowMajor.cancel_multiply_lt(quotient,upper,divisor,
        Bounds.nat_trans(Nat.mul(quotient,divisor),Nat.add(Nat.mul(quotient,divisor),1n+rest),Nat.mul(upper,divisor),
          %Addition.zero_right(Nat.mul(quotient,divisor)) : {Nat.is_lt(_,Nat.add(Nat.mul(quotient,divisor),1n+rest)) == True{} : Bool}
          Order.offset_lt(Nat.mul(quotient,divisor),0n,1n+rest,{==}),bound)))

def covers(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.is_le(value,Nat.mul(ceiling(value,divisor),divisor)) == True{} : Bool}:
  %Equal.sym(Nat,value,Nat.add(Nat.mul(Nat.div(value,divisor),divisor),Nat.mod(value,divisor)),DivisionOrder.quotient_exact(value,divisor,positive)) :
    {Nat.is_le(_,Nat.mul(ceiling(value,divisor),divisor)) == True{} : Bool}
  rounded_covers(Nat.div(value,divisor),Nat.mod(value,divisor),divisor,DivisionOrder.remainder_small(value,divisor,positive))

def minimal(+value: Nat,+divisor: Nat,+upper: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},bound: {Nat.is_le(value,Nat.mul(upper,divisor)) == True{} : Bool}) ->
  {Nat.is_le(ceiling(value,divisor),upper) == True{} : Bool}:
  rounded_minimal(Nat.div(value,divisor),Nat.mod(value,divisor),divisor,upper,positive,
    %DivisionOrder.quotient_exact(value,divisor,positive) : {Nat.is_le(_,Nat.mul(upper,divisor)) == True{} : Bool} bound)

def monotone(+left: Nat,+right: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},ordered: {Nat.is_le(left,right) == True{} : Bool}) ->
  {Nat.is_le(ceiling(left,divisor),ceiling(right,divisor)) == True{} : Bool}:
  minimal(left,divisor,ceiling(right,divisor),positive,Order.transitive(left,right,Nat.mul(ceiling(right,divisor),divisor),ordered,covers(right,divisor,positive)))

def bounded(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.is_le(ceiling(value,divisor),value) == True{} : Bool}:
  minimal(value,divisor,value,positive,RowMajor.product_covers(value,divisor,positive))

# For positive extents the last occupied block is floor((extent - 1)/span).
def successor_formula(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {ceiling(1n+value,divisor) == 1n+Nat.div(value,divisor) : Nat}:
  +quotient = Nat.div(value,divisor)
  +count = ceiling(1n+value,divisor)
  Order.antisymmetric(count,1n+quotient,
    minimal(1n+value,divisor,1n+quotient,positive,Order.strict_follow(value,Nat.mul(1n+quotient,divisor),DivisionOrder.next_above(value,divisor,positive))),
    Order.strict_follow(quotient,count,RowMajor.cancel_multiply_lt(quotient,count,divisor,
      Order.le_lt_trans(Nat.mul(quotient,divisor),value,Nat.mul(count,divisor),DivisionOrder.product_below(value,divisor,positive),
        Bounds.predecessor_strict(value,Nat.mul(count,divisor),covers(1n+value,divisor,positive))))))

def added_formula(+value: Nat,+divisor: Nat,+nonempty: {Nat.is_lt(0n,value) == True{} : Bool},+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.div(Nat.sub(Nat.add(value,divisor),1n),divisor) == ceiling(value,divisor) : Nat}:
  match value:
    case 0n: Empty.absurd({Nat.div(Nat.sub(Nat.add(0n,divisor),1n),divisor) == ceiling(0n,divisor) : Nat},WordEncoding.false_true(nonempty))
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.sub(Nat.add(rest,divisor),0n),Nat.add(rest,divisor),Order.subtract_zero(Nat.add(rest,divisor))) : {Nat.div(_,divisor) == ceiling(1n+rest,divisor) : Nat}
      %Equal.sym(Nat,Nat.add(rest,divisor),Nat.add(divisor,rest),Order.commutative(rest,divisor)) : {Nat.div(_,divisor) == ceiling(1n+rest,divisor) : Nat}
      %Equal.sym(Nat & Nat,Nat.divmod(Nat.add(divisor,rest),divisor),Division.bump(Nat.divmod(rest,divisor)),Division.add_divisor(rest,divisor,positive)) : {Pair.fst(Nat,Nat,_) == ceiling(1n+rest,divisor) : Nat}
      Equal.trans(Nat,Pair.fst(Nat,Nat,Division.bump(Nat.divmod(rest,divisor))),1n+Nat.div(rest,divisor),ceiling(1n+rest,divisor),
        Division.bumped_quotient(Nat.divmod(rest,divisor)),Equal.sym(Nat,ceiling(1n+rest,divisor),1n+Nat.div(rest,divisor),successor_formula(rest,divisor,positive)))

def previous_below(+value: Nat,+divisor: Nat,+nonempty: {Nat.is_lt(0n,value) == True{} : Bool},+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mul(Nat.sub(ceiling(value,divisor),1n),divisor),value) == True{} : Bool}:
  match value:
    case 0n: Empty.absurd({Nat.is_lt(Nat.mul(Nat.sub(ceiling(0n,divisor),1n),divisor),0n) == True{} : Bool},WordEncoding.false_true(nonempty))
    case 1n+ +rest:
      %Equal.sym(Nat,ceiling(1n+rest,divisor),1n+Nat.div(rest,divisor),successor_formula(rest,divisor,positive)) : {Nat.is_lt(Nat.mul(Nat.sub(_,1n),divisor),1n+rest) == True{} : Bool}
      %Equal.sym(Nat,Nat.sub(Nat.div(rest,divisor),0n),Nat.div(rest,divisor),Order.subtract_zero(Nat.div(rest,divisor))) : {Nat.is_lt(Nat.mul(_,divisor),1n+rest) == True{} : Bool}
      Order.lt_succ(Nat.mul(Nat.div(rest,divisor),divisor),rest,DivisionOrder.product_below(rest,divisor,positive))
