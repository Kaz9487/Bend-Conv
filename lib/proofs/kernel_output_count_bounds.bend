# Output counts are bounded from logical shape products before allocator
# readiness is used. The native formula keeps its actual multiplication order.
import Base
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor

type CountEvidence<-channels: U32,-rows: U32,-width: U32> is Data:
  CountEvidence{exact: {U32.to_nat((channels * (rows * width : U32) : U32)) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat},
    allocatable: {U32.is_le((channels * (rows * width : U32) : U32),1073741824) == True{} : Bool}}

def positions_room(+channels: U32,+rows: U32,+width: U32,+capacity: U32,
  channels_positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool},
  room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool}:
  +positions = Nat.mul(U32.to_nat(rows),U32.to_nat(width))
  Order.transitive(positions,Nat.mul(U32.to_nat(channels),positions),U32.to_nat(capacity),
    %RowMajor.multiply_commutative(positions,U32.to_nat(channels)) : {Nat.is_le(positions,_) == True{} : Bool}
    RowMajor.product_covers(positions,U32.to_nat(channels),channels_positive),room)

def exact(+channels: U32,+rows: U32,+width: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +channels_positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((channels * (rows * width : U32) : U32)) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat}:
  +positions = (rows * width : U32)
  +positions_exact = Arithmetic.multiply(rows,width,capacity,capacity_ok,positions_room(channels,rows,width,capacity,channels_positive,room))
  Equal.trans(Nat,U32.to_nat((channels * positions : U32)),Nat.mul(U32.to_nat(channels),U32.to_nat(positions)),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),
    Arithmetic.multiply(channels,positions,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat(positions),Nat.mul(U32.to_nat(rows),U32.to_nat(width)),positions_exact) : {Nat.is_le(Nat.mul(U32.to_nat(channels),_),U32.to_nat(capacity)) == True{} : Bool}
      room),
    Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(channels),value),U32.to_nat(positions),Nat.mul(U32.to_nat(rows),U32.to_nat(width)),positions_exact))

def allocatable(+channels: U32,+rows: U32,+width: U32,+capacity: U32,
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},capacity_allocatable: {U32.is_le(capacity,1073741824) == True{} : Bool},
  channels_positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.is_le((channels * (rows * width : U32) : U32),1073741824) == True{} : Bool}:
  Arithmetic.count_within((channels * (rows * width : U32) : U32),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),capacity,1073741824,
    exact(channels,rows,width,capacity,capacity_ok,channels_positive,room),room,capacity_allocatable)

def row_product_room(+channels: Nat,+rows: Nat,+height: Nat,+width: Nat,+capacity: Nat,
  rows_inside: {Nat.is_le(rows,height) == True{} : Bool},
  room: {Nat.is_le(Nat.mul(channels,Nat.mul(height,width)),capacity) == True{} : Bool}) -> {Nat.is_le(Nat.mul(channels,Nat.mul(rows,width)),capacity) == True{} : Bool}:
  Order.transitive(Nat.mul(channels,Nat.mul(rows,width)),Nat.mul(channels,Nat.mul(height,width)),capacity,
    %RowMajor.multiply_commutative(Nat.mul(rows,width),channels) : {Nat.is_le(_,Nat.mul(channels,Nat.mul(height,width))) == True{} : Bool}
    %RowMajor.multiply_commutative(Nat.mul(height,width),channels) : {Nat.is_le(Nat.mul(Nat.mul(rows,width),channels),_) == True{} : Bool}
    RowMajor.mul_le(Nat.mul(rows,width),Nat.mul(height,width),channels,RowMajor.mul_le(rows,height,width,rows_inside)),room)

def proved(+channels: U32,+rows: U32,+width: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},capacity_allocatable: {U32.is_le(capacity,1073741824) == True{} : Bool},
  +channels_positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),U32.to_nat(capacity)) == True{} : Bool}) -> CountEvidence<channels,rows,width>:
  CountEvidence{exact(channels,rows,width,capacity,capacity_ok,channels_positive,room),allocatable(channels,rows,width,capacity,capacity_ok,capacity_allocatable,channels_positive,room)}

def exact_of(+channels: U32,+rows: U32,+width: U32,evidence: CountEvidence<channels,rows,width>) ->
  {U32.to_nat((channels * (rows * width : U32) : U32)) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat}:
  CountEvidence{exact,allocatable} = evidence
  exact

def allocatable_of(+channels: U32,+rows: U32,+width: U32,evidence: CountEvidence<channels,rows,width>) ->
  {U32.is_le((channels * (rows * width : U32) : U32),1073741824) == True{} : Bool}:
  CountEvidence{exact,allocatable} = evidence
  allocatable
