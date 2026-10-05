# Exact native multiply/add for nonnegative coordinates, without reassociation
# of any FP32 operation. All arithmetic here is Nat/U32 indexing only.
import Base
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./row_major_index.bend as RowMajor
import ./word_arithmetic.bend as WordArithmetic
import ./natural_addition.bend as Addition
import ./u32_comparison.bend as Comparison
import ./word_multiplication.bend as Multiplication

def multiply_known_values(+left: U32,+right: U32,+a: Nat,+b: Nat,left_exact: {U32.to_nat(left) == a : Nat},right_exact: {U32.to_nat(right) == b : Nat}) ->
  {(left * right : U32) == U32.from_nat(Nat.mul(a,b)) : U32}:
  %left_exact : {(left * right : U32) == U32.from_nat(Nat.mul(_,b)) : U32}
  %right_exact : {(left * right : U32) == U32.from_nat(Nat.mul(U32.to_nat(left),_)) : U32}
  Multiplication.native_multiply(left,right)

def equal_words(+left: U32,+right: U32,equal: {left == right : U32}) -> {U32.is_eq(left,right) == True{} : Bool}:
  %equal : {U32.is_eq(left,_) == True{} : Bool}
  %Equal.sym(Bool,U32.is_eq(left,left),Nat.is_eq(U32.to_nat(left),U32.to_nat(left)),Comparison.eq(left,left)) : {_ == True{} : Bool}
  Order.equal_reflexive(U32.to_nat(left))

# Products used as tensor lengths. A positive final extent makes the prefix
# no larger than the complete allocation, so the same bound protects both.
def multiply_three(+a: U32,+b: U32,+c: U32,+capacity: U32,+capacity_bound: {U32.is_le(capacity,2147483648) == True{} : Bool},
  positive_last: {Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool},
  +length_bound: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(a),U32.to_nat(b)),U32.to_nat(c)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((a * b * c : U32)) == Nat.mul(Nat.mul(U32.to_nat(a),U32.to_nat(b)),U32.to_nat(c)) : Nat}:
  +prefix_exact = Arithmetic.multiply(a,b,capacity,capacity_bound,
    Order.transitive(Nat.mul(U32.to_nat(a),U32.to_nat(b)),Nat.mul(Nat.mul(U32.to_nat(a),U32.to_nat(b)),U32.to_nat(c)),U32.to_nat(capacity),
      RowMajor.product_covers(Nat.mul(U32.to_nat(a),U32.to_nat(b)),U32.to_nat(c),positive_last),length_bound))
  %prefix_exact : {U32.to_nat((a * b * c : U32)) == Nat.mul(_,U32.to_nat(c)) : Nat}
  Arithmetic.multiply((a * b : U32),c,capacity,capacity_bound,
    %Equal.sym(Nat,U32.to_nat((a * b : U32)),Nat.mul(U32.to_nat(a),U32.to_nat(b)),prefix_exact) :
      {Nat.is_le(Nat.mul(_,U32.to_nat(c)),U32.to_nat(capacity)) == True{} : Bool}
    length_bound)

def product(+a: Nat,+b: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},ha: {Nat.is_le(a,U32.to_nat(cap)) == True{} : Bool},
  hb: {Nat.is_le(b,U32.to_nat(cap)) == True{} : Bool},h: {Nat.is_le(Nat.mul(a,b),
      U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((U32.from_nat(a) * U32.from_nat(b) : U32)) == Nat.mul(a,b) : Nat}:
  +ea = Bounds.roundtrip(a,cap,cap_ok,ha)
  +eb = Bounds.roundtrip(b,cap,cap_ok,hb)
  %ea : {U32.to_nat((U32.from_nat(a) * U32.from_nat(b) : U32)) == Nat.mul(_,b) : Nat}
  %eb : {U32.to_nat((U32.from_nat(a) * U32.from_nat(b) : U32)) == Nat.mul(U32.to_nat(U32.from_nat(a)),_) : Nat}
  Arithmetic.multiply(U32.from_nat(a),U32.from_nat(b),cap,cap_ok,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(a)),a,ea) : {Nat.is_le(Nat.mul(_,U32.to_nat(U32.from_nat(b))),U32.to_nat(cap)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(b)),b,eb) : {Nat.is_le(Nat.mul(a,_),U32.to_nat(cap)) == True{} : Bool}
    h)

def madd(+a: Nat,+b: Nat,+c: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},ha: {Nat.is_le(a,
      U32.to_nat(cap)) == True{} : Bool},hb: {Nat.is_le(b,U32.to_nat(cap)) == True{} : Bool},hc: {Nat.is_le(c,U32.to_nat(cap)) == True{} : Bool},
  +h: {Nat.is_le(Nat.add(Nat.mul(a,b),c),
      U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((U32.from_nat(a) * U32.from_nat(b) + U32.from_nat(c) : U32)) == Nat.add(Nat.mul(a,b),
    c) : Nat}:
  +ep = product(a,b,cap,cap_ok,ha,hb,Order.transitive(Nat.mul(a,b),Nat.add(Nat.mul(a,b),c),U32.to_nat(cap),Order.le_add(Nat.mul(a,b),c),h))
  +ec = Bounds.roundtrip(c,cap,cap_ok,hc)
  %ep : {U32.to_nat((U32.from_nat(a) * U32.from_nat(b) + U32.from_nat(c) : U32)) == Nat.add(_,c) : Nat}
  %ec : {U32.to_nat((U32.from_nat(a) * U32.from_nat(b) + U32.from_nat(c) : U32)) == Nat.add(U32.to_nat((U32.from_nat(a) * U32.from_nat(b) : U32)),_) : Nat}
  Arithmetic.add((U32.from_nat(a) * U32.from_nat(b) : U32),U32.from_nat(c),cap,cap_ok,
    %Equal.sym(Nat,U32.to_nat((U32.from_nat(a) * U32.from_nat(b) : U32)),Nat.mul(a,b),ep) : {Nat.is_le(Nat.add(_,U32.to_nat(U32.from_nat(c))),
        U32.to_nat(cap)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(c)),c,ec) : {Nat.is_le(Nat.add(Nat.mul(a,b),_),U32.to_nat(cap)) == True{} : Bool}
    h)

def rectangle(+a: Nat,+a_limit: Nat,+b: Nat,+c: Nat,+c_limit: Nat,+cap: Nat,ha: {Nat.is_lt(a,a_limit) == True{} : Bool},hc: {Nat.is_lt(c,
      c_limit) == True{} : Bool},bound: {Nat.is_le(Nat.add(Nat.mul(a_limit,b),c_limit),cap) == True{} : Bool}) -> {Nat.is_le(Nat.add(Nat.mul(a,
        b),c),cap) == True{} : Bool}:
  Order.transitive(Nat.add(Nat.mul(a,b),c),Nat.add(Nat.mul(a_limit,b),c),cap,Order.add_le(Nat.mul(a,b),Nat.mul(a_limit,b),c,
      RowMajor.mul_le(a,a_limit,b,Bounds.strict_le(a,a_limit,ha))),
    Order.transitive(Nat.add(Nat.mul(a_limit,b),c),Nat.add(Nat.mul(a_limit,b),c_limit),cap,
      %Order.commutative(c,Nat.mul(a_limit,b)) : {Nat.is_le(_,Nat.add(Nat.mul(a_limit,b),c_limit)) == True{} : Bool}
      %Order.commutative(c_limit,Nat.mul(a_limit,b)) : {Nat.is_le(Nat.add(c,Nat.mul(a_limit,b)),_) == True{} : Bool}
      Order.add_le(c,c_limit,Nat.mul(a_limit,b),Bounds.strict_le(c,c_limit,hc)),bound))

def left_bound(+a: Nat,+b: Nat,+c: Nat,+cap: Nat,positive: {Nat.is_lt(0n,b) == True{} : Bool},bound: {Nat.is_le(Nat.add(Nat.mul(a,b),c),
      cap) == True{} : Bool}) -> {Nat.is_le(a,cap) == True{} : Bool}:
  Order.transitive(a,Nat.mul(a,b),cap,RowMajor.product_covers(a,b,positive),Order.transitive(Nat.mul(a,b),Nat.add(Nat.mul(a,b),c),cap,
      Order.le_add(Nat.mul(a,b),c),bound))

def sum_bounds(+a: Nat,+b: Nat,+a_limit: Nat,+b_limit: Nat,ha: {Nat.is_le(a,a_limit) == True{} : Bool},hb: {Nat.is_le(b,
      b_limit) == True{} : Bool}) -> {Nat.is_le(Nat.add(a,b),Nat.add(a_limit,b_limit)) == True{} : Bool}:
  Order.transitive(Nat.add(a,b),Nat.add(a_limit,b),Nat.add(a_limit,b_limit),Order.add_le(a,a_limit,b,ha),
    %Order.commutative(b,a_limit) : {Nat.is_le(_,Nat.add(a_limit,b_limit)) == True{} : Bool}
    %Order.commutative(b_limit,a_limit) : {Nat.is_le(Nat.add(b,a_limit),_) == True{} : Bool}
    Order.add_le(b,b_limit,a_limit,hb))

def encoding(+u: U32,+n: Nat,e: {U32.to_nat(u) == n : Nat}) -> {u == U32.from_nat(n) : U32}:
  %e : {u == U32.from_nat(_) : U32}
  Equal.sym(U32,U32.from_nat(U32.to_nat(u)),u,Arithmetic.native_roundtrip(u))

def scaled_offset_identity(+column: Nat,+offset: Nat,+stride: Nat,+tap: Nat) ->
  {Nat.add(Nat.add(Nat.mul(column,stride),tap),Nat.mul(offset,stride)) == Nat.add(Nat.mul(Nat.add(column,offset),stride),tap) : Nat}:
  %Equal.sym(Nat,Nat.mul(Nat.add(column,offset),stride),Nat.add(Nat.mul(column,stride),Nat.mul(offset,stride)),RowMajor.distrib(column,offset,stride)) :
    {Nat.add(Nat.add(Nat.mul(column,stride),tap),Nat.mul(offset,stride)) == Nat.add(_,tap) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(column,stride),tap),Nat.mul(offset,stride)),Nat.add(Nat.mul(column,stride),Nat.add(tap,Nat.mul(offset,stride))),
    Addition.associative(Nat.mul(column,stride),tap,Nat.mul(offset,stride))) :
    {_ == Nat.add(Nat.add(Nat.mul(column,stride),Nat.mul(offset,stride)),tap) : Nat}
  %Equal.sym(Nat,Nat.add(tap,Nat.mul(offset,stride)),Nat.add(Nat.mul(offset,stride),tap),Order.commutative(tap,Nat.mul(offset,stride))) :
    {Nat.add(Nat.mul(column,stride),_) == Nat.add(Nat.add(Nat.mul(column,stride),Nat.mul(offset,stride)),tap) : Nat}
  Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(column,stride),Nat.mul(offset,stride)),tap),Nat.add(Nat.mul(column,stride),Nat.add(Nat.mul(offset,stride),tap)),
    Addition.associative(Nat.mul(column,stride),Nat.mul(offset,stride),tap))

# Moving a lane along a strided input row preserves the integer address.
# Padding subtraction may wrap; only the nonnegative coordinates need bounds.
def padded_scaled_offset(+column: Nat,+offset: Nat,+stride: Nat,+tap: Nat,+padding: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +column_bound: {Nat.is_le(Nat.add(column,offset),U32.to_nat(capacity)) == True{} : Bool},
  +stride_bound: {Nat.is_le(stride,U32.to_nat(capacity)) == True{} : Bool},
  +tap_bound: {Nat.is_le(tap,U32.to_nat(capacity)) == True{} : Bool},
  +coordinate_bound: {Nat.is_le(Nat.add(Nat.mul(Nat.add(column,offset),stride),tap),U32.to_nat(capacity)) == True{} : Bool}) ->
  {((U32.from_nat(column) * U32.from_nat(stride) + U32.from_nat(tap) - padding : U32) + U32.from_nat(offset) * U32.from_nat(stride) : U32) ==
    (U32.from_nat(Nat.add(column,offset)) * U32.from_nat(stride) + U32.from_nat(tap) - padding : U32) : U32}:
  +column_le_sum = Order.le_add(column,offset)
  +offset_le_sum = {%Order.commutative(offset,column) : {Nat.is_le(offset,_) == True{} : Bool} Order.le_add(offset,column) : {Nat.is_le(offset,Nat.add(column,offset)) == True{} : Bool}}
  +product_bound = Order.transitive(Nat.mul(Nat.add(column,offset),stride),Nat.add(Nat.mul(Nat.add(column,offset),stride),tap),U32.to_nat(capacity),Order.le_add(Nat.mul(Nat.add(column,offset),stride),tap),coordinate_bound)
  +base_bound = Order.transitive(Nat.add(Nat.mul(column,stride),tap),Nat.add(Nat.mul(Nat.add(column,offset),stride),tap),U32.to_nat(capacity),
    Order.add_le(Nat.mul(column,stride),Nat.mul(Nat.add(column,offset),stride),tap,RowMajor.mul_le(column,Nat.add(column,offset),stride,column_le_sum)),coordinate_bound)
  +base_exact = encoding((U32.from_nat(column) * U32.from_nat(stride) + U32.from_nat(tap) : U32),Nat.add(Nat.mul(column,stride),tap),
    madd(column,stride,tap,capacity,capacity_ok,Order.transitive(column,Nat.add(column,offset),U32.to_nat(capacity),column_le_sum,column_bound),stride_bound,tap_bound,base_bound))
  +lane_product_exact = encoding((U32.from_nat(offset) * U32.from_nat(stride) : U32),Nat.mul(offset,stride),
    product(offset,stride,capacity,capacity_ok,Order.transitive(offset,Nat.add(column,offset),U32.to_nat(capacity),offset_le_sum,column_bound),stride_bound,
      Order.transitive(Nat.mul(offset,stride),Nat.mul(Nat.add(column,offset),stride),U32.to_nat(capacity),RowMajor.mul_le(offset,Nat.add(column,offset),stride,offset_le_sum),product_bound)))
  +target_exact = encoding((U32.from_nat(Nat.add(column,offset)) * U32.from_nat(stride) + U32.from_nat(tap) : U32),Nat.add(Nat.mul(Nat.add(column,offset),stride),tap),
    madd(Nat.add(column,offset),stride,tap,capacity,capacity_ok,column_bound,stride_bound,tap_bound,coordinate_bound))
  %Equal.sym(U32,((U32.from_nat(column) * U32.from_nat(stride) + U32.from_nat(tap) - padding : U32) + U32.from_nat(offset) * U32.from_nat(stride) : U32),
    ((U32.from_nat(column) * U32.from_nat(stride) + U32.from_nat(tap) + U32.from_nat(offset) * U32.from_nat(stride) : U32) - padding : U32),
    Arithmetic.add_after_subtract((U32.from_nat(column) * U32.from_nat(stride) + U32.from_nat(tap) : U32),padding,(U32.from_nat(offset) * U32.from_nat(stride) : U32))) :
    {_ == (U32.from_nat(Nat.add(column,offset)) * U32.from_nat(stride) + U32.from_nat(tap) - padding : U32) : U32}
  %Equal.sym(U32,(U32.from_nat(column) * U32.from_nat(stride) + U32.from_nat(tap) : U32),U32.from_nat(Nat.add(Nat.mul(column,stride),tap)),base_exact) :
    {((_ + U32.from_nat(offset) * U32.from_nat(stride) : U32) - padding : U32) == (U32.from_nat(Nat.add(column,offset)) * U32.from_nat(stride) + U32.from_nat(tap) - padding : U32) : U32}
  %Equal.sym(U32,(U32.from_nat(offset) * U32.from_nat(stride) : U32),U32.from_nat(Nat.mul(offset,stride)),lane_product_exact) :
    {((U32.from_nat(Nat.add(Nat.mul(column,stride),tap)) + _ : U32) - padding : U32) == (U32.from_nat(Nat.add(column,offset)) * U32.from_nat(stride) + U32.from_nat(tap) - padding : U32) : U32}
  %Equal.sym(U32,(U32.from_nat(Nat.add(column,offset)) * U32.from_nat(stride) + U32.from_nat(tap) : U32),U32.from_nat(Nat.add(Nat.mul(Nat.add(column,offset),stride),tap)),target_exact) :
    {((U32.from_nat(Nat.add(Nat.mul(column,stride),tap)) + U32.from_nat(Nat.mul(offset,stride)) : U32) - padding : U32) == (_ - padding : U32) : U32}
  %Equal.sym(U32,(U32.from_nat(Nat.add(Nat.mul(column,stride),tap)) + U32.from_nat(Nat.mul(offset,stride)) : U32),U32.from_nat(Nat.add(Nat.add(Nat.mul(column,stride),tap),Nat.mul(offset,stride))),
    WordArithmetic.native_add(Nat.add(Nat.mul(column,stride),tap),Nat.mul(offset,stride))) :
    {(_ - padding : U32) == (U32.from_nat(Nat.add(Nat.mul(Nat.add(column,offset),stride),tap)) - padding : U32) : U32}
  %scaled_offset_identity(column,offset,stride,tap) :
    {(U32.from_nat(Nat.add(Nat.add(Nat.mul(column,stride),tap),Nat.mul(offset,stride))) - padding : U32) == (U32.from_nat(_) - padding : U32) : U32}
  {==}
