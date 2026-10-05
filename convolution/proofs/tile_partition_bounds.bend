# Exact native split arithmetic, including empty children at excessive depth.
import Base
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/binary_division.bend as BinaryDivision
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/word_subtraction.bend as Subtraction
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/natural_addition.bend as Addition

def right_mul_le(+n: Nat,+a: Nat,+b: Nat,+h: {Nat.is_le(a,b) == True{} : Bool}) -> {Nat.is_le(Nat.mul(n,a),Nat.mul(n,b)) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      Encoded.sum_bounds(a,Nat.mul(p,a),b,Nat.mul(p,b),h,right_mul_le(p,a,b,h))

def product_parts(+r: BinaryDivision.Quotient,+a: Nat,+b: Nat,h: {BinaryDivision.total(r,
      b) == a : Nat}) -> {Nat.is_le(Nat.mul(U32Division.quotient(r),b),a) == True{} : Bool}:
  BinaryDivision.Quotient{+q,+rest} = r
  %h : {Nat.is_le(Nat.mul(q,b),_) == True{} : Bool}
  Order.le_add(Nat.mul(q,b),rest)

def quotient_product(+a: U32,+b: Nat,positive: {Nat.is_lt(0n,b) == True{} : Bool}) -> {Nat.is_le(Nat.mul(Nat.div(U32.to_nat(a),b),b),
    U32.to_nat(a)) == True{} : Bool}:
  U32{+w} = a
  %Equal.sym(Nat,Nat.div(Word.to_nat(32n,w),b),U32Division.quotient(BinaryDivision.go(32n,w,b)),U32Division.natural_quotient(w,b,
      positive)) : {Nat.is_le(Nat.mul(_,b),Word.to_nat(32n,w)) == True{} : Bool}
  product_parts(BinaryDivision.go(32n,w,b),Word.to_nat(32n,w),b,BinaryDivision.go_total(32n,w,b))

def half(+span: U32) -> U32:
  (span / 16 * 8 : U32)

def product_bound(+span: U32) -> {Nat.is_le(Nat.mul(U32.to_nat((span / 16 : U32)),8n),U32.to_nat(span)) == True{} : Bool}:
  Order.transitive(Nat.mul(U32.to_nat((span / 16 : U32)),8n),Nat.mul(U32.to_nat((span / 16 : U32)),16n),U32.to_nat(span),
    right_mul_le(U32.to_nat((span / 16 : U32)),8n,16n,{==}),
    %Equal.sym(Nat,U32.to_nat((span / 16 : U32)),Nat.div(U32.to_nat(span),16n),U32Division.divide(span,16,2147483648,{==},{==},
        {==})) : {Nat.is_le(Nat.mul(_,16n),U32.to_nat(span)) == True{} : Bool}
    quotient_product(span,16n,{==}))

def half_exact(+span: U32,+cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},bound: {Nat.is_le(U32.to_nat(span),
      U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat(half(span)) == Nat.mul(U32.to_nat((span / 16 : U32)),8n) : Nat}:
  Arithmetic.multiply((span / 16 : U32),8,cap,cap_ok,Order.transitive(Nat.mul(U32.to_nat((span / 16 : U32)),8n),U32.to_nat(span),
      U32.to_nat(cap),product_bound(span),bound))

def half_le(+span: U32,+cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},bound: {Nat.is_le(U32.to_nat(span),
      U32.to_nat(cap)) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(half(span)),U32.to_nat(span)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(half(span)),Nat.mul(U32.to_nat((span / 16 : U32)),8n),half_exact(span,cap,cap_ok,bound)) : {Nat.is_le(_,
      U32.to_nat(span)) == True{} : Bool}
  product_bound(span)

def remainder_exact(+span: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+bound: {Nat.is_le(U32.to_nat(span),
      U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((span - half(span) : U32)) == Nat.sub(U32.to_nat(span),U32.to_nat(half(span))) : Nat}:
  Arithmetic.subtract(span,half(span),cap,cap_ok,half_le(span,cap,cap_ok,bound),Order.transitive(Nat.sub(U32.to_nat(span),
        U32.to_nat(half(span))),U32.to_nat(span),U32.to_nat(cap),Order.sub_le(U32.to_nat(span),U32.to_nat(half(span))),bound))

def sum(+span: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+bound: {Nat.is_le(U32.to_nat(span),
      U32.to_nat(cap)) == True{} : Bool}) -> {Nat.add(U32.to_nat(half(span)),U32.to_nat((span - half(span) : U32))) == U32.to_nat(span) : Nat}:
  %Equal.sym(Nat,U32.to_nat((span - half(span) : U32)),Nat.sub(U32.to_nat(span),U32.to_nat(half(span))),remainder_exact(span,cap,cap_ok,
      bound)) : {Nat.add(U32.to_nat(half(span)),_) == U32.to_nat(span) : Nat}
  %Order.commutative(Nat.sub(U32.to_nat(span),U32.to_nat(half(span))),U32.to_nat(half(span))) : {_ == U32.to_nat(span) : Nat}
  Subtraction.restore(U32.to_nat(span),U32.to_nat(half(span)),half_le(span,cap,cap_ok,bound))

def span_room(+offset: U32,+span: U32,+cap: U32,room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),
      U32.to_nat(cap)) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(span),U32.to_nat(cap)) == True{} : Bool}:
  Order.transitive(U32.to_nat(span),Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(cap),
    %Order.commutative(U32.to_nat(span),U32.to_nat(offset)) : {Nat.is_le(U32.to_nat(span),_) == True{} : Bool}
    Order.le_add(U32.to_nat(span),U32.to_nat(offset)),room)

def left_room(+offset: U32,+span: U32,+cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(cap)) == True{} : Bool}) -> {Nat.is_le(Nat.add(U32.to_nat(offset),
      U32.to_nat(half(span))),U32.to_nat(cap)) == True{} : Bool}:
  Order.transitive(Nat.add(U32.to_nat(offset),U32.to_nat(half(span))),Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(cap),
    Encoded.sum_bounds(U32.to_nat(offset),U32.to_nat(half(span)),U32.to_nat(offset),U32.to_nat(span),
      Order.reflexive(U32.to_nat(offset)),half_le(span,cap,cap_ok,span_room(offset,span,cap,room))),room)

def offset_exact(+offset: U32,+span: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),
      U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((offset + half(span) : U32)) == Nat.add(U32.to_nat(offset),U32.to_nat(half(span))) : Nat}:
  Arithmetic.add(offset,half(span),cap,cap_ok,left_room(offset,span,cap,cap_ok,room))

def right_room(+offset: U32,+span: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),
      U32.to_nat(cap)) == True{} : Bool}) -> {Nat.is_le(Nat.add(U32.to_nat((offset + half(span) : U32)),U32.to_nat((span - half(span) : U32))),
    U32.to_nat(cap)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((offset + half(span) : U32)),Nat.add(U32.to_nat(offset),U32.to_nat(half(span))),offset_exact(offset,span,cap,cap_ok,
      room)) : {Nat.is_le(Nat.add(_,U32.to_nat((span - half(span) : U32))),U32.to_nat(cap)) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(offset),U32.to_nat(half(span))),U32.to_nat((span - half(span) : U32))),Nat.add(U32.to_nat(offset),
      Nat.add(U32.to_nat(half(span)),U32.to_nat((span - half(span) : U32)))),Addition.associative(U32.to_nat(offset),U32.to_nat(half(span)),
      U32.to_nat((span - half(span) : U32)))) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(U32.to_nat(half(span)),U32.to_nat((span - half(span) : U32))),U32.to_nat(span),sum(span,cap,cap_ok,span_room(offset,
        span,cap,room))) : {Nat.is_le(Nat.add(U32.to_nat(offset),_),U32.to_nat(cap)) == True{} : Bool}
  room

type BlockAligned<-value: U32> is Data:
  BlockAligned{blocks: Nat,exact: {U32.to_nat(value) == Nat.mul(blocks,8n) : Nat}}

# Splitting preserves a logical interval bound as well as machine capacity.
def left_interval_bound(+offset: U32,+span: U32,+cap: U32,+limit: Nat,
  cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(cap)) == True{} : Bool},
  range_bound: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),limit) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(half(span))),limit) == True{} : Bool}:
  Order.transitive(Nat.add(U32.to_nat(offset),U32.to_nat(half(span))),Nat.add(U32.to_nat(offset),U32.to_nat(span)),limit,
    Encoded.sum_bounds(U32.to_nat(offset),U32.to_nat(half(span)),U32.to_nat(offset),U32.to_nat(span),
      Order.reflexive(U32.to_nat(offset)),half_le(span,cap,cap_ok,span_room(offset,span,cap,room))),range_bound)

def right_interval_bound(+offset: U32,+span: U32,+cap: U32,+limit: Nat,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(cap)) == True{} : Bool},
  range_bound: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),limit) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat((offset + half(span) : U32)),U32.to_nat((span - half(span) : U32))),limit) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((offset + half(span) : U32)),Nat.add(U32.to_nat(offset),U32.to_nat(half(span))),offset_exact(offset,span,cap,cap_ok,room)) :
    {Nat.is_le(Nat.add(_,U32.to_nat((span - half(span) : U32))),limit) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(offset),U32.to_nat(half(span))),U32.to_nat((span - half(span) : U32))),
    Nat.add(U32.to_nat(offset),Nat.add(U32.to_nat(half(span)),U32.to_nat((span - half(span) : U32)))),
    Addition.associative(U32.to_nat(offset),U32.to_nat(half(span)),U32.to_nat((span - half(span) : U32)))) : {Nat.is_le(_,limit) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(U32.to_nat(half(span)),U32.to_nat((span - half(span) : U32))),U32.to_nat(span),
    sum(span,cap,cap_ok,span_room(offset,span,cap,room))) : {Nat.is_le(Nat.add(U32.to_nat(offset),_),limit) == True{} : Bool}
  range_bound

def half_alignment(+span: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  bound: {Nat.is_le(U32.to_nat(span),U32.to_nat(cap)) == True{} : Bool}) -> BlockAligned<half(span)>:
  BlockAligned{U32.to_nat((span / 16 : U32)),half_exact(span,cap,cap_ok,bound)}

def remainder_alignment(+span: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +bound: {Nat.is_le(U32.to_nat(span),U32.to_nat(cap)) == True{} : Bool},alignment: BlockAligned<span>) -> BlockAligned<(span - half(span) : U32)>:
  BlockAligned{+blocks,+exact} = alignment
  +half_blocks = U32.to_nat((span / 16 : U32))
  BlockAligned{Nat.sub(blocks,half_blocks),
    %Equal.sym(Nat,U32.to_nat((span - half(span) : U32)),Nat.sub(U32.to_nat(span),U32.to_nat(half(span))),remainder_exact(span,cap,cap_ok,bound)) :
      {_ == Nat.mul(Nat.sub(blocks,half_blocks),8n) : Nat}
    %Equal.sym(Nat,U32.to_nat(span),Nat.mul(blocks,8n),exact) : {Nat.sub(_,U32.to_nat(half(span))) == Nat.mul(Nat.sub(blocks,half_blocks),8n) : Nat}
    %Equal.sym(Nat,U32.to_nat(half(span)),Nat.mul(half_blocks,8n),half_exact(span,cap,cap_ok,bound)) :
      {Nat.sub(Nat.mul(blocks,8n),_) == Nat.mul(Nat.sub(blocks,half_blocks),8n) : Nat}
    RowMajor.subtract_products(blocks,half_blocks,8n)}

def right_alignment(+offset: U32,+span: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(cap)) == True{} : Bool},alignment: BlockAligned<offset>) -> BlockAligned<(offset + half(span) : U32)>:
  BlockAligned{+group,+exact} = alignment
  +half_group = U32.to_nat((span / 16 : U32))
  BlockAligned{Nat.add(group,half_group),
    %Equal.sym(Nat,U32.to_nat((offset + half(span) : U32)),Nat.add(U32.to_nat(offset),U32.to_nat(half(span))),offset_exact(offset,span,cap,cap_ok,room)) :
      {_ == Nat.mul(Nat.add(group,half_group),8n) : Nat}
    %Equal.sym(Nat,U32.to_nat(offset),Nat.mul(group,8n),exact) : {Nat.add(_,U32.to_nat(half(span))) == Nat.mul(Nat.add(group,half_group),8n) : Nat}
    %Equal.sym(Nat,U32.to_nat(half(span)),Nat.mul(half_group,8n),half_exact(span,cap,cap_ok,span_room(offset,span,cap,room))) :
      {Nat.add(Nat.mul(group,8n),_) == Nat.mul(Nat.add(group,half_group),8n) : Nat}
    Equal.sym(Nat,Nat.mul(Nat.add(group,half_group),8n),Nat.add(Nat.mul(group,8n),Nat.mul(half_group,8n)),RowMajor.distrib(group,half_group,8n))}
