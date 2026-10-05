# A genuine natural count greater than one discharges both native equality
# flags. The result is independent of tensor shapes and kernel scheduling.
import Base
import ./u32_comparison.bend as Comparison
import ./nat_order.bend as Order

def zero_false_from_positive(+count: U32,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}) ->
  {U32.is_eq(count,0) == False{} : Bool}:
  %Equal.sym(Bool,U32.is_eq(count,0),Nat.is_eq(U32.to_nat(count),0n),Comparison.eq(count,0)) : {_ == False{} : Bool}
  %Equal.sym(Bool,Nat.is_eq(U32.to_nat(count),0n),Nat.is_eq(0n,U32.to_nat(count)),Order.equal_symmetric(U32.to_nat(count),0n)) : {_ == False{} : Bool}
  Order.less_not_equal(0n,U32.to_nat(count),positive)

def zero_false(+count: U32,+bound: {Nat.is_lt(1n,U32.to_nat(count)) == True{} : Bool}) ->
  {U32.is_eq(count,0) == False{} : Bool}:
  %Equal.sym(Bool,U32.is_eq(count,0),Nat.is_eq(U32.to_nat(count),0n),Comparison.eq(count,0)) : {_ == False{} : Bool}
  %Equal.sym(Bool,Nat.is_eq(U32.to_nat(count),0n),Nat.is_eq(0n,U32.to_nat(count)),Order.equal_symmetric(U32.to_nat(count),0n)) : {_ == False{} : Bool}
  Order.less_not_equal(0n,U32.to_nat(count),Order.le_lt_trans(0n,1n,U32.to_nat(count),Order.zero_le(1n),bound))

def one_false(+count: U32,+bound: {Nat.is_lt(1n,U32.to_nat(count)) == True{} : Bool}) ->
  {U32.is_eq(count,1) == False{} : Bool}:
  %Equal.sym(Bool,U32.is_eq(count,1),Nat.is_eq(U32.to_nat(count),1n),Comparison.eq(count,1)) : {_ == False{} : Bool}
  %Equal.sym(Bool,Nat.is_eq(U32.to_nat(count),1n),Nat.is_eq(1n,U32.to_nat(count)),Order.equal_symmetric(U32.to_nat(count),1n)) : {_ == False{} : Bool}
  Order.less_not_equal(1n,U32.to_nat(count),bound)
