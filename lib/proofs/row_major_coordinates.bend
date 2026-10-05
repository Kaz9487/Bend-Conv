# Every bounded flat index has bounded channel, row and column coordinates.
import Base
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./nat_order.bend as Order

type Coordinates<-channels: Nat,-rows: Nat,-width: Nat,-position: Nat> is Data:
  Coordinates{channel: Nat,row: Nat,column: Nat,
    channel_inside: {Nat.is_lt(channel,channels) == True{} : Bool},row_inside: {Nat.is_lt(row,rows) == True{} : Bool},column_inside: {Nat.is_lt(column,width) == True{} : Bool},
    exact: {position == RowMajor.index(channel,rows,width,row,column) : Nat}}

def locate(+channels: Nat,+rows: Nat,+width: Nat,+position: Nat,+width_positive: {Nat.is_lt(0n,width) == True{} : Bool},
  +inside: {Nat.is_lt(position,Nat.mul(channels,Nat.mul(rows,width))) == True{} : Bool}) -> Coordinates<channels,rows,width,position>:
  +plane = Nat.mul(rows,width)
  +positive = RowMajor.left_factor_positive(plane,channels,
    %RowMajor.multiply_commutative(channels,plane) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.le_lt_trans(0n,position,Nat.mul(channels,plane),Order.zero_le(position),inside))
  +channel = Nat.div(position,plane)
  +within = Nat.mod(position,plane)
  +split = DivisionOrder.quotient_exact(position,plane,positive)
  +within_inside = DivisionOrder.remainder_small(position,plane,positive)
  +row = Nat.div(within,width)
  +column = Nat.mod(within,width)
  +pixel = DivisionOrder.quotient_exact(within,width,width_positive)
  Coordinates{channel,row,column,
    RowMajor.row_inside(channel,channels,plane,within,%split : {Nat.is_lt(_,Nat.mul(channels,plane)) == True{} : Bool} inside),
    RowMajor.row_inside(row,rows,width,column,%pixel : {Nat.is_lt(_,Nat.mul(rows,width)) == True{} : Bool} within_inside),DivisionOrder.remainder_small(within,width,width_positive),
    Equal.trans(Nat,position,Nat.add(Nat.mul(channel,plane),within),RowMajor.index(channel,rows,width,row,column),split,
      Equal.cong(Nat,Nat,index => Nat.add(Nat.mul(channel,plane),index),within,Nat.add(Nat.mul(row,width),column),pixel))}
