# Whole-channel copies write the same subrange in successive channel planes.
# Separating the allocation base from the intra-plane offset also covers the
# last band of a gather, whose start need not be plane-aligned.
import Base
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./bounded_u32_arithmetic.bend as Arithmetic

def start(base: Nat,offset: Nat,stride: Nat,channel: Nat) -> Nat:
  Nat.add(Nat.add(base,offset),Nat.mul(channel,stride))

def scaled_span(+first: Nat,+rows: Nat,+height: Nat,+width: Nat,span: {Nat.is_le(Nat.add(first,rows),height) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(first,width),Nat.mul(rows,width)),Nat.mul(height,width)) == True{} : Bool}:
  %RowMajor.distrib(first,rows,width) : {Nat.is_le(_,Nat.mul(height,width)) == True{} : Bool}
  RowMajor.mul_le(Nat.add(first,rows),height,width,span)

def end_form(+base: Nat,+offset: Nat,+stride: Nat,+channel: Nat,+count: Nat) ->
  {Nat.add(start(base,offset,stride,channel),count) == Nat.add(base,Nat.add(Nat.mul(channel,stride),Nat.add(offset,count))) : Nat}:
  +plane = Nat.mul(channel,stride)
  Equal.trans(Nat,Nat.add(Nat.add(Nat.add(base,offset),plane),count),Nat.add(Nat.add(base,offset),Nat.add(plane,count)),Nat.add(base,Nat.add(plane,Nat.add(offset,count))),
    Addition.associative(Nat.add(base,offset),plane,count),
    Equal.trans(Nat,Nat.add(Nat.add(base,offset),Nat.add(plane,count)),Nat.add(base,Nat.add(offset,Nat.add(plane,count))),Nat.add(base,Nat.add(plane,Nat.add(offset,count))),
      Addition.associative(base,offset,Nat.add(plane,count)),
      Equal.cong(Nat,Nat,value => Nat.add(base,value),Nat.add(offset,Nat.add(plane,count)),Nat.add(plane,Nat.add(offset,count)),RowMajor.add_swap(offset,plane,count))))

def plane_end(+channel: Nat,+channels: Nat,+stride: Nat,+offset: Nat,+count: Nat,
  inside: {Nat.is_lt(channel,channels) == True{} : Bool},span: {Nat.is_le(Nat.add(offset,count),stride) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(channel,stride),Nat.add(offset,count)),Nat.mul(channels,stride)) == True{} : Bool}:
  Order.transitive(Nat.add(Nat.mul(channel,stride),Nat.add(offset,count)),Nat.mul(1n+channel,stride),Nat.mul(channels,stride),
    %Equal.sym(Nat,Nat.add(Nat.mul(channel,stride),Nat.add(offset,count)),Nat.add(Nat.add(offset,count),Nat.mul(channel,stride)),Order.commutative(Nat.mul(channel,stride),Nat.add(offset,count))) :
      {Nat.is_le(_,Nat.mul(1n+channel,stride)) == True{} : Bool}
    Order.add_le(Nat.add(offset,count),stride,Nat.mul(channel,stride),span),
    RowMajor.mul_le(1n+channel,channels,stride,Order.strict_follow(channel,channels,inside)))

def end_inside(+base: Nat,+offset: Nat,+stride: Nat,+count: Nat,+channels: Nat,+channel: Nat,+capacity: Nat,
  inside: {Nat.is_lt(channel,channels) == True{} : Bool},span: {Nat.is_le(Nat.add(offset,count),stride) == True{} : Bool},
  room: {Nat.is_le(Nat.add(base,Nat.mul(channels,stride)),capacity) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(start(base,offset,stride,channel),count),capacity) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(start(base,offset,stride,channel),count),Nat.add(base,Nat.add(Nat.mul(channel,stride),Nat.add(offset,count))),end_form(base,offset,stride,channel,count)) :
    {Nat.is_le(_,capacity) == True{} : Bool}
  Order.transitive(Nat.add(base,Nat.add(Nat.mul(channel,stride),Nat.add(offset,count))),Nat.add(base,Nat.mul(channels,stride)),capacity,
    Order.offset_le(base,Nat.add(Nat.mul(channel,stride),Nat.add(offset,count)),Nat.mul(channels,stride),plane_end(channel,channels,stride,offset,count,inside,span)),room)

def query_inside(+base: Nat,+offset: Nat,+stride: Nat,+count: Nat,+channels: Nat,+channel: Nat,+index: Nat,+capacity: Nat,
  channel_inside: {Nat.is_lt(channel,channels) == True{} : Bool},+index_inside: {Nat.is_lt(index,count) == True{} : Bool},
  span: {Nat.is_le(Nat.add(offset,count),stride) == True{} : Bool},room: {Nat.is_le(Nat.add(base,Nat.mul(channels,stride)),capacity) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(start(base,offset,stride,channel),index),capacity) == True{} : Bool}:
  Bounds.nat_trans(Nat.add(start(base,offset,stride,channel),index),Nat.add(start(base,offset,stride,channel),count),capacity,
    Order.offset_lt(start(base,offset,stride,channel),index,count,index_inside),end_inside(base,offset,stride,count,channels,channel,capacity,channel_inside,span,room))

def count_in_stride(+offset: Nat,+count: Nat,+stride: Nat,span: {Nat.is_le(Nat.add(offset,count),stride) == True{} : Bool}) ->
  {Nat.is_le(count,stride) == True{} : Bool}:
  Order.transitive(count,Nat.add(offset,count),stride,
    %Order.commutative(count,offset) : {Nat.is_le(count,_) == True{} : Bool}
    Order.le_add(count,offset),span)

def query_before(+base: Nat,+offset: Nat,+stride: Nat,+count: Nat,+target: Nat,+later: Nat,+index: Nat,
  ordered: {Nat.is_lt(target,later) == True{} : Bool},inside: {Nat.is_lt(index,count) == True{} : Bool},span: {Nat.is_le(Nat.add(offset,count),stride) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(start(base,offset,stride,target),index),start(base,offset,stride,later)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(start(base,offset,stride,target),index),Nat.add(Nat.add(base,offset),Nat.add(Nat.mul(target,stride),index)),
    Addition.associative(Nat.add(base,offset),Nat.mul(target,stride),index)) : {Nat.is_lt(_,start(base,offset,stride,later)) == True{} : Bool}
  Order.offset_lt(Nat.add(base,offset),Nat.add(Nat.mul(target,stride),index),Nat.mul(later,stride),
    RowMajor.radix(target,later,index,stride,ordered,Bounds.nat_trans(index,count,stride,inside,count_in_stride(offset,count,stride,span))))

def native_start(+destination: U32,+stride: U32,+channel: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  channel_bound: {Nat.is_le(channel,U32.to_nat(capacity)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(destination),Nat.mul(channel,U32.to_nat(stride))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((destination + U32.from_nat(channel) * stride : U32)) == Nat.add(U32.to_nat(destination),Nat.mul(channel,U32.to_nat(stride))) : Nat}:
  +channel_exact = Bounds.roundtrip(channel,capacity,capacity_ok,channel_bound)
  +product = Nat.mul(channel,U32.to_nat(stride))
  +product_exact = {
    %channel_exact : {U32.to_nat((U32.from_nat(channel) * stride : U32)) == Nat.mul(_,U32.to_nat(stride)) : Nat}
    Arithmetic.multiply(U32.from_nat(channel),stride,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(channel)),channel,channel_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool}
      Order.transitive(product,Nat.add(U32.to_nat(destination),product),U32.to_nat(capacity),
        %Order.commutative(product,U32.to_nat(destination)) : {Nat.is_le(product,_) == True{} : Bool} Order.le_add(product,U32.to_nat(destination)),room)) :
    {U32.to_nat((U32.from_nat(channel) * stride : U32)) == product : Nat}}
  %product_exact : {U32.to_nat((destination + U32.from_nat(channel) * stride : U32)) == Nat.add(U32.to_nat(destination),_) : Nat}
  Arithmetic.add(destination,(U32.from_nat(channel) * stride : U32),capacity,capacity_ok,
    %Equal.sym(Nat,U32.to_nat((U32.from_nat(channel) * stride : U32)),product,product_exact) : {Nat.is_le(Nat.add(U32.to_nat(destination),_),U32.to_nat(capacity)) == True{} : Bool}
    room)
