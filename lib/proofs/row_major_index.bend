# Mixed-radix NCHW indexing and its required bounds, on unbounded Nat.
import Base
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./nat_division.bend as Division
import ./word_encoding.bend as WordEncoding

def multiply_zero_right(+value: Nat) -> {Nat.mul(value,0n) == 0n : Nat}:
  match value:
    case 0n: {==}
    case 1n+rest: multiply_zero_right(rest)

def add_swap(+a: Nat,+b: Nat,+tail: Nat) -> {Nat.add(a,Nat.add(b,tail)) == Nat.add(b,Nat.add(a,tail)) : Nat}:
  Equal.trans(Nat,Nat.add(a,Nat.add(b,tail)),Nat.add(Nat.add(a,b),tail),Nat.add(b,Nat.add(a,tail)),
    Equal.sym(Nat,Nat.add(Nat.add(a,b),tail),Nat.add(a,Nat.add(b,tail)),Addition.associative(a,b,tail)),
    Equal.trans(Nat,Nat.add(Nat.add(a,b),tail),Nat.add(Nat.add(b,a),tail),Nat.add(b,Nat.add(a,tail)),
      Equal.cong(Nat,Nat,n => Nat.add(n,tail),Nat.add(a,b),Nat.add(b,a),Order.commutative(a,b)),Addition.associative(b,a,tail)))

def multiply_successor_right(+a: Nat,+b: Nat) -> {Nat.mul(a,1n+b) == Nat.add(a,Nat.mul(a,b)) : Nat}:
  match a:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.mul(rest,1n+b),Nat.add(rest,Nat.mul(rest,b)),multiply_successor_right(rest,b)) :
        {1n+Nat.add(b,_) == 1n+Nat.add(rest,Nat.add(b,Nat.mul(rest,b))) : Nat}
      Equal.cong(Nat,Nat,n => 1n+n,Nat.add(b,Nat.add(rest,Nat.mul(rest,b))),Nat.add(rest,Nat.add(b,Nat.mul(rest,b))),add_swap(b,rest,Nat.mul(rest,b)))

def multiply_commutative(+a: Nat,+b: Nat) -> {Nat.mul(a,b) == Nat.mul(b,a) : Nat}:
  match a:
    case 0n: Equal.sym(Nat,Nat.mul(b,0n),0n,multiply_zero_right(b))
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.mul(b,1n+rest),Nat.add(b,Nat.mul(b,rest)),multiply_successor_right(b,rest)) : {Nat.add(b,Nat.mul(rest,b)) == _ : Nat}
      Equal.cong(Nat,Nat,n => Nat.add(b,n),Nat.mul(rest,b),Nat.mul(b,rest),multiply_commutative(rest,b))

def subtract_common_prefix(+prefix: Nat,+a: Nat,+b: Nat) -> {Nat.sub(Nat.add(prefix,a),Nat.add(prefix,b)) == Nat.sub(a,b) : Nat}:
  match prefix:
    case 0n: {==}
    case 1n+rest: subtract_common_prefix(rest,a,b)

def subtract_products(+a: Nat,+b: Nat,+factor: Nat) -> {Nat.sub(Nat.mul(a,factor),Nat.mul(b,factor)) == Nat.mul(Nat.sub(a,b),factor) : Nat}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+b: Order.zero_subtract(Nat.mul(1n+b,factor))
    case 1n+ +a 0n:
      %Equal.sym(Nat,Nat.sub(1n+a,0n),1n+a,Order.subtract_zero(1n+a)) :
        {Nat.sub(Nat.mul(1n+a,factor),0n) == Nat.mul(_,factor) : Nat}
      Order.subtract_zero(Nat.mul(1n+a,factor))
    case 1n+ +a 1n+ +b:
      %Equal.sym(Nat,Nat.sub(Nat.add(factor,Nat.mul(a,factor)),Nat.add(factor,Nat.mul(b,factor))),Nat.sub(Nat.mul(a,factor),Nat.mul(b,factor)),
        subtract_common_prefix(factor,Nat.mul(a,factor),Nat.mul(b,factor))) : {_ == Nat.mul(Nat.sub(a,b),factor) : Nat}
      subtract_products(a,b,factor)

def left_factor_positive(+left: Nat,+right: Nat,positive: {Nat.is_lt(0n,Nat.mul(left,right)) == True{} : Bool}) -> {Nat.is_lt(0n,left) == True{} : Bool}:
  match left:
    case 0n: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(positive))
    case 1n+rest: {==}

def distrib(+a: Nat,+b: Nat,+c: Nat) -> {Nat.mul(Nat.add(a,b),c) == Nat.add(Nat.mul(a,c),Nat.mul(b,c)) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+ +p:
      %Equal.sym(Nat,Nat.add(Nat.add(c,Nat.mul(p,c)),Nat.mul(b,c)),Nat.add(c,Nat.add(Nat.mul(p,c),Nat.mul(b,c))),Addition.associative(c,
          Nat.mul(p,c),Nat.mul(b,c))) : {Nat.add(c,Nat.mul(Nat.add(p,b),c)) == _ : Nat}
      %distrib(p,b,c) : {Nat.add(c,Nat.mul(Nat.add(p,b),c)) == Nat.add(c,_) : Nat}
      {==}

def assoc(+a: Nat,+b: Nat,+c: Nat) -> {Nat.mul(Nat.mul(a,b),c) == Nat.mul(a,Nat.mul(b,c)) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+ +p:
      %Equal.sym(Nat,Nat.mul(Nat.add(b,Nat.mul(p,b)),c),Nat.add(Nat.mul(b,c),Nat.mul(Nat.mul(p,b),c)),distrib(b,Nat.mul(p,b),
          c)) : {_ == Nat.add(Nat.mul(b,c),Nat.mul(p,Nat.mul(b,c))) : Nat}
      %assoc(p,b,c) : {Nat.add(Nat.mul(b,c),Nat.mul(Nat.mul(p,b),c)) == Nat.add(Nat.mul(b,c),_) : Nat}
      {==}

def right_distrib(+factor: Nat,+left: Nat,+right: Nat) ->
  {Nat.mul(factor,Nat.add(left,right)) == Nat.add(Nat.mul(factor,left),Nat.mul(factor,right)) : Nat}:
  %Equal.sym(Nat,Nat.mul(factor,Nat.add(left,right)),Nat.mul(Nat.add(left,right),factor),multiply_commutative(factor,Nat.add(left,right))) :
    {_ == Nat.add(Nat.mul(factor,left),Nat.mul(factor,right)) : Nat}
  %multiply_commutative(left,factor) : {Nat.mul(Nat.add(left,right),factor) == Nat.add(_,Nat.mul(factor,right)) : Nat}
  %multiply_commutative(right,factor) : {Nat.mul(Nat.add(left,right),factor) == Nat.add(Nat.mul(left,factor),_) : Nat}
  distrib(left,right,factor)

def mul_le(+a: Nat,+b: Nat,+c: Nat,h: {Nat.is_le(a,b) == True{} : Bool}) -> {Nat.is_le(Nat.mul(a,c),Nat.mul(b,c)) == True{} : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+b:
      Order.zero_le(Nat.mul(1n+b,c))
    case 1n+a 0n:
      Empty.absurd({Nat.is_le(Nat.mul(1n+a,c),0n) == True{} : Bool},WordEncoding.false_true(h))
    case 1n+ +a 1n+ +b:
      %Order.commutative(Nat.mul(a,c),c) : {Nat.is_le(_,Nat.add(c,Nat.mul(b,c))) == True{} : Bool}
      %Order.commutative(Nat.mul(b,c),c) : {Nat.is_le(Nat.add(Nat.mul(a,c),c),_) == True{} : Bool}
      Order.add_le(Nat.mul(a,c),Nat.mul(b,c),c,mul_le(a,b,c,h))

def radix(+a: Nat,+limit: Nat,+b: Nat,+width: Nat,ab: {Nat.is_lt(a,limit) == True{} : Bool},bw: {Nat.is_lt(b,
      width) == True{} : Bool}) -> {Nat.is_lt(Nat.add(Nat.mul(a,width),b),Nat.mul(limit,width)) == True{} : Bool}:
  Bounds.nat_trans(Nat.add(Nat.mul(a,width),b),Nat.mul(1n+a,width),Nat.mul(limit,width),
    %Order.commutative(b,Nat.mul(a,width)) : {Nat.is_lt(_,Nat.add(width,Nat.mul(a,width))) == True{} : Bool}
    Order.add_lt(b,width,Nat.mul(a,width),bw),
    mul_le(1n+a,limit,width,Order.strict_follow(a,limit,ab)))

def cancel_multiply_lt(+left: Nat,+right: Nat,+factor: Nat,ordered: {Nat.is_lt(Nat.mul(left,factor),Nat.mul(right,factor)) == True{} : Bool}) ->
  {Nat.is_lt(left,right) == True{} : Bool}:
  match left right:
    case 0n 0n: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(ordered))
    case 0n 1n+right: {==}
    case 1n+ +left 0n:
      Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(Nat.mul(1n+left,factor),0n),True{},
        Equal.sym(Bool,Nat.is_lt(Nat.mul(1n+left,factor),0n),False{},Division.lt_zero(Nat.mul(1n+left,factor))),ordered)))
    case 1n+ +left 1n+ +right:
      cancel_multiply_lt(left,right,factor,Order.cancel_offset_lt(factor,Nat.mul(left,factor),Nat.mul(right,factor),ordered))

def row_inside(+row: Nat,+rows: Nat,+width: Nat,+column: Nat,inside: {Nat.is_lt(Nat.add(Nat.mul(row,width),column),Nat.mul(rows,width)) == True{} : Bool}) ->
  {Nat.is_lt(row,rows) == True{} : Bool}:
  cancel_multiply_lt(row,rows,width,Order.le_lt_trans(Nat.mul(row,width),Nat.add(Nat.mul(row,width),column),Nat.mul(rows,width),Order.le_add(Nat.mul(row,width),column),inside))

def multiply_lt(+left: Nat,+right: Nat,+factor: Nat,ordered: {Nat.is_lt(left,right) == True{} : Bool},positive: {Nat.is_lt(0n,factor) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mul(left,factor),Nat.mul(right,factor)) == True{} : Bool}:
  %Addition.zero_right(Nat.mul(left,factor)) : {Nat.is_lt(_,Nat.mul(right,factor)) == True{} : Bool}
  radix(left,right,0n,factor,ordered,positive)

def index(+input_channel_index: Nat,+height: Nat,+width: Nat,+iy: Nat,+ix: Nat) -> Nat:
  Nat.add(Nat.mul(input_channel_index,Nat.mul(height,width)),Nat.add(Nat.mul(iy,width),ix))

def expanded(+input_channel_index: Nat,+height: Nat,+width: Nat,+iy: Nat,+ix: Nat) -> {Nat.add(Nat.add(Nat.mul(Nat.mul(input_channel_index,height),width),Nat.mul(iy,width)),
    ix) == index(input_channel_index,height,width,iy,ix) : Nat}:
  %assoc(input_channel_index,height,width) : {Nat.add(Nat.add(Nat.mul(Nat.mul(input_channel_index,height),width),Nat.mul(iy,width)),ix) == Nat.add(_,Nat.add(Nat.mul(iy,width),
        ix)) : Nat}
  Addition.associative(Nat.mul(Nat.mul(input_channel_index,height),width),Nat.mul(iy,width),ix)

def mixed(+input_channel_index: Nat,+height: Nat,+width: Nat,+iy: Nat,+ix: Nat) -> {Nat.add(Nat.mul(Nat.add(Nat.mul(input_channel_index,height),iy),width),ix) == index(input_channel_index,height,
    width,iy,ix) : Nat}:
  %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(input_channel_index,height),iy),width),Nat.add(Nat.mul(Nat.mul(input_channel_index,height),width),Nat.mul(iy,width)),distrib(Nat.mul(input_channel_index,
        height),iy,width)) : {Nat.add(_,ix) == index(input_channel_index,height,width,iy,ix) : Nat}
  expanded(input_channel_index,height,width,iy,ix)

law in_bounds:
  for +input_channel_index: Nat
  for +input_channel_count: Nat
  for +height: Nat
  for +width: Nat
  for +iy: Nat
  for +ix: Nat
  for channel: {Nat.is_lt(input_channel_index,input_channel_count) == True{} : Bool}
  for row: {Nat.is_lt(iy,height) == True{} : Bool}
  for col: {Nat.is_lt(ix,width) == True{} : Bool}
  {Nat.is_lt(index(input_channel_index,height,width,iy,ix),Nat.mul(input_channel_count,Nat.mul(height,width))) == True{} : Bool}
def in_bounds(input_channel_index,input_channel_count,height,width,iy,ix,channel,row,col):
  %mixed(input_channel_index,height,width,iy,ix) : {Nat.is_lt(_,Nat.mul(input_channel_count,Nat.mul(height,width))) == True{} : Bool}
  %assoc(input_channel_count,height,width) : {Nat.is_lt(Nat.add(Nat.mul(Nat.add(Nat.mul(input_channel_index,height),iy),width),ix),_) == True{} : Bool}
  radix(Nat.add(Nat.mul(input_channel_index,height),iy),Nat.mul(input_channel_count,height),ix,width,radix(input_channel_index,input_channel_count,iy,height,channel,row),col)

def product_covers(+a: Nat,+b: Nat,positive: {Nat.is_lt(0n,b) == True{} : Bool}) -> {Nat.is_le(a,Nat.mul(a,b)) == True{} : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+b:
      {==}
    case 1n+a 0n:
      Empty.absurd({Nat.is_le(1n+a,Nat.mul(1n+a,0n)) == True{} : Bool},WordEncoding.false_true(positive))
    case 1n+ +a 1n+ +b:
      Order.transitive(a,Nat.mul(a,1n+b),Nat.add(b,Nat.mul(a,1n+b)),product_covers(a,1n+b,{==}),
        %Order.commutative(Nat.mul(a,1n+b),b) : {Nat.is_le(Nat.mul(a,1n+b),_) == True{} : Bool}
        Order.le_add(Nat.mul(a,1n+b),b))

def row_product(+input_channel_index: Nat,+input_channel_count: Nat,+height: Nat,+width: Nat,+cap: Nat,channel: {Nat.is_lt(input_channel_index,input_channel_count) == True{} : Bool},positive: {Nat.is_lt(0n,
      width) == True{} : Bool},bound: {Nat.is_le(Nat.mul(input_channel_count,Nat.mul(height,width)),cap) == True{} : Bool}) -> {Nat.is_le(Nat.mul(input_channel_index,height),
    cap) == True{} : Bool}:
  Order.transitive(Nat.mul(input_channel_index,height),Nat.mul(input_channel_count,height),cap,mul_le(input_channel_index,input_channel_count,height,Bounds.strict_le(input_channel_index,input_channel_count,channel)),
    Order.transitive(Nat.mul(input_channel_count,height),Nat.mul(input_channel_count,Nat.mul(height,width)),cap,
      %assoc(input_channel_count,height,width) : {Nat.is_le(Nat.mul(input_channel_count,height),_) == True{} : Bool}
      product_covers(Nat.mul(input_channel_count,height),width,positive),bound))
