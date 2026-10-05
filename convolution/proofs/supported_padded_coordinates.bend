import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
# Exact runtime spatial rounding, derived from the real fast guard.
import Base
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/natural_addition.bend as Addition
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/u32_shift.bend as U32Shift
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ../specification.bend as ConvSpec

def scalar(+limits: Machine.Limits,+p: ConvSpec.Problem,proof: Support.FastSupported<limits,
  p>) -> Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>:
  Support.FastSupported{s,a,b} = proof
  s

def product_has_right(+a: Nat,+b: Nat,positive: {Nat.is_lt(0n,a) == True{} : Bool}) -> {Nat.is_le(b,Nat.mul(a,b)) == True{} : Bool}:
  match a:
    case 0n:
      Empty.absurd({Nat.is_le(b,0n) == True{} : Bool},WordEncoding.false_true(positive))
    case 1n+p:
      Order.le_add(b,Nat.mul(p,b))

def scalar_output_bound(+limits: Machine.Limits,+p: ConvSpec.Problem,
  support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>) ->
  {Nat.is_le(Nat.mul(Guards.output_channel_count(p),Nat.mul(ConvSpec.out_height(p),ConvSpec.out_width(p))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Guards.Supported{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11} = support
  h9

def spatial_bound_facts(+limits: Machine.Limits,+p: ConvSpec.Problem,valid: Guards.Valid<p>,
  support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {Nat.is_le(Nat.mul(ConvSpec.out_height(p),ConvSpec.out_width(p)),
    U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Guards.Valid{v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10} = valid
  Guards.Supported{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11} = support
  Order.transitive(Nat.mul(ConvSpec.out_height(p),ConvSpec.out_width(p)),Nat.mul(Guards.output_channel_count(p),Nat.mul(ConvSpec.out_height(p),
        ConvSpec.out_width(p))),U32.to_nat(Machine.capacity(limits)),product_has_right(Guards.output_channel_count(p),
      Nat.mul(ConvSpec.out_height(p),ConvSpec.out_width(p)),v1),h9)

def padded_bound(+limits: Machine.Limits,+p: ConvSpec.Problem,valid: Guards.Valid<p>,
  support: Support.FastSupported<limits,p>) -> {Nat.is_le(Support.padded(p),
    U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  ConvSpec.Problem{pshape,+xs,+ws,+bs} = p
  ConvSpec.Shape{+d0,+d1,+d2,+d3,+d4,+d5,+d6} = pshape
  Guards.Valid{v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10} = valid
  Support.FastSupported{sc,packing,output} = support
  Order.transitive(Support.padded(p),Nat.mul(Support.padded(p),ConvSpec.outputs(p)),
    U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(Support.padded(p),ConvSpec.outputs(p),v1),output)

def spatial(+p: ConvSpec.Problem) -> U32:
  (U32.from_nat(ConvSpec.out_height(p)) * U32.from_nat(ConvSpec.out_width(p)) : U32)

def padded(+p: ConvSpec.Problem) -> U32:
  ((spatial(p) + 7) / 8 * 8 : U32)

def scalar_spatial_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,+valid: Guards.Valid<p>,
  +s: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>) ->
  {U32.to_nat(spatial(p)) == Nat.mul(ConvSpec.out_height(p),ConvSpec.out_width(p)) : Nat}:
  +eh = Bounds.roundtrip(ConvSpec.out_height(p),Machine.coordinate(limits),Limits.coordinate_fits(limits),
    SupportedCoordinateBounds.row_limit(limits,p,valid,s))
  +ew = Bounds.roundtrip(ConvSpec.out_width(p),Machine.coordinate(limits),Limits.coordinate_fits(limits),
    SupportedCoordinateBounds.col_limit(limits,p,valid,s))
  %eh : {U32.to_nat(spatial(p)) == Nat.mul(_,ConvSpec.out_width(p)) : Nat}
  %ew : {U32.to_nat(spatial(p)) == Nat.mul(U32.to_nat(U32.from_nat(ConvSpec.out_height(p))),_) : Nat}
  Arithmetic.multiply(U32.from_nat(ConvSpec.out_height(p)),U32.from_nat(ConvSpec.out_width(p)),Machine.capacity(limits),
    Limits.capacity_fits(limits),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_height(p))),ConvSpec.out_height(p),eh) : {Nat.is_le(Nat.mul(_,
          U32.to_nat(U32.from_nat(ConvSpec.out_width(p)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_width(p))),ConvSpec.out_width(p),ew) : {Nat.is_le(Nat.mul(ConvSpec.out_height(p),_),
        U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    spatial_bound_facts(limits,p,valid,s))

def spatial_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,+valid: Guards.Valid<p>,
  support: Support.FastSupported<limits,p>) -> {U32.to_nat(spatial(p)) == Nat.mul(ConvSpec.out_height(p),ConvSpec.out_width(p)) : Nat}:
  scalar_spatial_exact(limits,p,valid,scalar(limits,p,support))

def add_in_double(+n: Nat,+small: U32,+large: U32,relation: {U32.to_nat(large) == Nat.double(U32.to_nat(small)) : Nat},bound: {Nat.is_le(n,
      U32.to_nat(small)) == True{} : Bool},seven: {Nat.is_le(7n,U32.to_nat(small)) == True{} : Bool}) -> {Nat.is_le(Nat.add(n,7n),
    U32.to_nat(large)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(large),Nat.double(U32.to_nat(small)),relation) : {Nat.is_le(Nat.add(n,7n),_) == True{} : Bool}
  %Addition.double(U32.to_nat(small)) : {Nat.is_le(Nat.add(n,7n),_) == True{} : Bool}
  Encoded.sum_bounds(n,7n,U32.to_nat(small),U32.to_nat(small),bound,seven)

def add_seven_bound(+limits: Machine.Limits,+n: Nat,bound: {Nat.is_le(n,
      U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {Nat.is_le(Nat.add(n,7n),
    U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  add_in_double(n,Machine.capacity(limits),Machine.half_word(limits),Limits.doubled_capacity(limits),bound,Bounds.as_nat_le(7,
      Machine.capacity(limits),Limits.seven_within_capacity(limits)))

def added_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,+valid: Guards.Valid<p>,
  +support: Support.FastSupported<limits,p>) -> {U32.to_nat((spatial(p) + 7 : U32)) == Nat.add(Nat.mul(ConvSpec.out_height(p),
      ConvSpec.out_width(p)),7n) : Nat}:
  +e = spatial_exact(limits,p,valid,support)
  %e : {U32.to_nat((spatial(p) + 7 : U32)) == Nat.add(_,7n) : Nat}
  Arithmetic.add(spatial(p),7,Machine.half_word(limits),Limits.half_word_fits(limits),
    %Equal.sym(Nat,U32.to_nat(spatial(p)),Nat.mul(ConvSpec.out_height(p),ConvSpec.out_width(p)),e) : {Nat.is_le(Nat.add(_,7n),
        U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    add_seven_bound(limits,Nat.mul(ConvSpec.out_height(p),ConvSpec.out_width(p)),spatial_bound_facts(limits,p,valid,scalar(limits,p,support))))

def divided_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,valid: Guards.Valid<p>,
  support: Support.FastSupported<limits,
  p>) -> {U32.to_nat(((spatial(p) + 7) / 8 : U32)) == Nat.div(Nat.add(Nat.mul(ConvSpec.out_height(p),ConvSpec.out_width(p)),7n),8n) : Nat}:
  %added_exact(limits,p,valid,support) : {U32.to_nat(((spatial(p) + 7) / 8 : U32)) == Nat.div(_,8n) : Nat}
  U32Division.divide((spatial(p) + 7 : U32),8,Machine.half_word(limits),Limits.half_word_fits(limits),
    Bounds.as_nat_le(16,Machine.half_word(limits),Limits.sixteen_within_half(limits)),{==})

def spatial_within_padded(+limits: Machine.Limits,+p: ConvSpec.Problem,+valid: Guards.Valid<p>,
  +support: Support.FastSupported<limits,p>) ->
  {Nat.is_le(Nat.mul(ConvSpec.out_height(p),ConvSpec.out_width(p)),Support.padded(p)) == True{} : Bool}:
  +size = Nat.mul(ConvSpec.out_height(p),ConvSpec.out_width(p))
  +rounded = (spatial(p) + 7 : U32)
  +quotient = Nat.div(U32.to_nat(rounded),8n)
  +remainder = Nat.mod(U32.to_nat(rounded),8n)
  %added_exact(limits,p,valid,support) : {Nat.is_le(size,Nat.mul(Nat.div(_,8n),8n)) == True{} : Bool}
  Order.cancel_offset_le(7n,size,Nat.mul(quotient,8n),
    %Order.commutative(size,7n) : {Nat.is_le(_,Nat.add(7n,Nat.mul(quotient,8n))) == True{} : Bool}
    %Order.commutative(Nat.mul(quotient,8n),7n) : {Nat.is_le(Nat.add(size,7n),_) == True{} : Bool}
    %added_exact(limits,p,valid,support) : {Nat.is_le(_,Nat.add(Nat.mul(quotient,8n),7n)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(rounded),Nat.add(Nat.mul(quotient,8n),remainder),Radix.division_identity(rounded,8n,{==})) :
      {Nat.is_le(_,Nat.add(Nat.mul(quotient,8n),7n)) == True{} : Bool}
    Encoded.sum_bounds(Nat.mul(quotient,8n),remainder,Nat.mul(quotient,8n),7n,Order.reflexive(Nat.mul(quotient,8n)),
      Division.lt_succ_le(remainder,7n,Radix.remainder_inside(rounded,8n,{==}))))

def padded_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,+valid: Guards.Valid<p>,
  +support: Support.FastSupported<limits,p>) -> {U32.to_nat(padded(p)) == Support.padded(p) : Nat}:
  +e = divided_exact(limits,p,valid,support)
  %e : {U32.to_nat(padded(p)) == Nat.mul(_,8n) : Nat}
  Arithmetic.multiply(((spatial(p) + 7) / 8 : U32),8,Machine.capacity(limits),Limits.capacity_fits(limits),
    %Equal.sym(Nat,U32.to_nat(((spatial(p) + 7) / 8 : U32)),Nat.div(Nat.add(Nat.mul(ConvSpec.out_height(p),ConvSpec.out_width(p)),7n),8n),
      e) : {Nat.is_le(Nat.mul(_,8n),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    padded_bound(limits,p,valid,support))

def root_room(+limits: Machine.Limits,+p: ConvSpec.Problem,+valid: Guards.Valid<p>,
  +support: Support.FastSupported<limits,p>) -> {Nat.is_le(U32.to_nat(padded(p)),
    U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(padded(p)),Support.padded(p),padded_exact(limits,p,valid,support)) : {Nat.is_le(_,
      U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  padded_bound(limits,p,valid,support)
