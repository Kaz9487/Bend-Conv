import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
# Coordinates derived from the actual concrete supported/valid guards.
import Base
import ../../lib/proofs/natural_addition.bend as Addition
import ./convolution_guard_facts.bend as Guards
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/u32_shift.bend as U32Shift
import ../specification.bend as ConvSpec

def ci_bound(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),Machine.coordinate(limits)>) -> {Nat.is_le(Guards.input_channel_count(p),
    U32.to_nat(Machine.dimension(limits))) == True{} : Bool}:
  Guards.Supported{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11} = support
  h0

def ci_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {U32.to_nat(U32.from_nat(Guards.input_channel_count(p))) == Guards.input_channel_count(p) : Nat}:
  Bounds.roundtrip(Guards.input_channel_count(p),Machine.dimension(limits),Limits.dimension_fits(limits),ci_bound(limits,
      p,support))

def co_bound(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),Machine.coordinate(limits)>) -> {Nat.is_le(Guards.output_channel_count(p),
    U32.to_nat(Machine.dimension(limits))) == True{} : Bool}:
  Guards.Supported{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11} = support
  h1

def co_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {U32.to_nat(U32.from_nat(Guards.output_channel_count(p))) == Guards.output_channel_count(p) : Nat}:
  Bounds.roundtrip(Guards.output_channel_count(p),Machine.dimension(limits),Limits.dimension_fits(limits),co_bound(limits,
      p,support))

def height_bound(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),Machine.coordinate(limits)>) -> {Nat.is_le(Guards.height(p),
    U32.to_nat(Machine.dimension(limits))) == True{} : Bool}:
  Guards.Supported{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11} = support
  h2

def height_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {U32.to_nat(U32.from_nat(Guards.height(p))) == Guards.height(p) : Nat}:
  Bounds.roundtrip(Guards.height(p),Machine.dimension(limits),Limits.dimension_fits(limits),
    height_bound(limits,p,support))

def width_bound(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),Machine.coordinate(limits)>) -> {Nat.is_le(Guards.width(p),
    U32.to_nat(Machine.dimension(limits))) == True{} : Bool}:
  Guards.Supported{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11} = support
  h3

def width_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {U32.to_nat(U32.from_nat(Guards.width(p))) == Guards.width(p) : Nat}:
  Bounds.roundtrip(Guards.width(p),Machine.dimension(limits),Limits.dimension_fits(limits),
    width_bound(limits,p,support))

def kernel_bound(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),Machine.coordinate(limits)>) -> {Nat.is_le(Guards.kernel(p),
    U32.to_nat(Machine.dimension(limits))) == True{} : Bool}:
  Guards.Supported{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11} = support
  h4

def kernel_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {U32.to_nat(U32.from_nat(Guards.kernel(p))) == Guards.kernel(p) : Nat}:
  Bounds.roundtrip(Guards.kernel(p),Machine.dimension(limits),Limits.dimension_fits(limits),
    kernel_bound(limits,p,support))

def kernel_large(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),Machine.coordinate(limits)>) -> {Nat.is_le(Guards.kernel(p),
    U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  Order.transitive(Guards.kernel(p),U32.to_nat(Machine.dimension(limits)),U32.to_nat(Machine.coordinate(limits)),
    kernel_bound(limits,p,support),Bounds.as_nat_le(Machine.dimension(limits),Machine.coordinate(limits),
      Limits.dimension_within_coordinate(limits)))

def stride_bound(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),Machine.coordinate(limits)>) -> {Nat.is_le(Guards.stride(p),
    U32.to_nat(Machine.dimension(limits))) == True{} : Bool}:
  Guards.Supported{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11} = support
  h5

def stride_large(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),Machine.coordinate(limits)>) -> {Nat.is_le(Guards.stride(p),
    U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  Order.transitive(Guards.stride(p),U32.to_nat(Machine.dimension(limits)),U32.to_nat(Machine.coordinate(limits)),
    stride_bound(limits,p,support),Bounds.as_nat_le(Machine.dimension(limits),Machine.coordinate(limits),
      Limits.dimension_within_coordinate(limits)))

def pad_bound(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),Machine.coordinate(limits)>) -> {Nat.is_le(Guards.pad(p),
    U32.to_nat(Machine.dimension(limits))) == True{} : Bool}:
  Guards.Supported{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11} = support
  h6

def pad_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {U32.to_nat(U32.from_nat(Guards.pad(p))) == Guards.pad(p) : Nat}:
  Bounds.roundtrip(Guards.pad(p),Machine.dimension(limits),Limits.dimension_fits(limits),
    pad_bound(limits,p,support))

def row_envelope(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),Machine.coordinate(limits)>) -> {Nat.is_le(Nat.add(Nat.mul(ConvSpec.out_height(p),
        Guards.stride(p)),Guards.kernel(p)),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  Guards.Supported{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11} = support
  h10

def row_limit(+limits: Machine.Limits,+p: ConvSpec.Problem,valid: Guards.Valid<p>,
  support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {Nat.is_le(ConvSpec.out_height(p),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  Guards.Valid{v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10} = valid
  Encoded.left_bound(ConvSpec.out_height(p),Guards.stride(p),Guards.kernel(p),
    U32.to_nat(Machine.coordinate(limits)),v5,row_envelope(limits,p,support))

def row_coordinate(+limits: Machine.Limits,+p: ConvSpec.Problem,+o: Nat,+k: Nat,valid: Guards.Valid<p>,
  +support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,
  +ho: {Nat.is_lt(o,ConvSpec.out_height(p)) == True{} : Bool},+hk: {Nat.is_lt(k,
      Guards.kernel(p)) == True{} : Bool}) -> {U32.to_nat((U32.from_nat(o) * U32.from_nat(Guards.stride(p)) +
        U32.from_nat(k) : U32)) == Nat.add(Nat.mul(o,Guards.stride(p)),k) : Nat}:
  Encoded.madd(o,Guards.stride(p),k,Machine.coordinate(limits),Limits.coordinate_fits(limits),
    Bounds.strict_le(o,U32.to_nat(Machine.coordinate(limits)),Bounds.nat_trans(o,ConvSpec.out_height(p),
        U32.to_nat(Machine.coordinate(limits)),ho,row_limit(limits,p,valid,support))),
    stride_large(limits,p,support),
    Bounds.strict_le(k,U32.to_nat(Machine.coordinate(limits)),Bounds.nat_trans(k,Guards.kernel(p),
        U32.to_nat(Machine.coordinate(limits)),hk,kernel_large(limits,p,support))),
    Encoded.rectangle(o,ConvSpec.out_height(p),Guards.stride(p),k,Guards.kernel(p),
      U32.to_nat(Machine.coordinate(limits)),ho,hk,row_envelope(limits,p,support)))

def col_envelope(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),Machine.coordinate(limits)>) -> {Nat.is_le(Nat.add(Nat.mul(ConvSpec.out_width(p),
        Guards.stride(p)),Guards.kernel(p)),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  Guards.Supported{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11} = support
  h11

def col_limit(+limits: Machine.Limits,+p: ConvSpec.Problem,valid: Guards.Valid<p>,
  support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {Nat.is_le(ConvSpec.out_width(p),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  Guards.Valid{v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10} = valid
  Encoded.left_bound(ConvSpec.out_width(p),Guards.stride(p),Guards.kernel(p),
    U32.to_nat(Machine.coordinate(limits)),v5,col_envelope(limits,p,support))

def col_coordinate(+limits: Machine.Limits,+p: ConvSpec.Problem,+o: Nat,+k: Nat,valid: Guards.Valid<p>,
  +support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,
  +ho: {Nat.is_lt(o,ConvSpec.out_width(p)) == True{} : Bool},+hk: {Nat.is_lt(k,
      Guards.kernel(p)) == True{} : Bool}) -> {U32.to_nat((U32.from_nat(o) * U32.from_nat(Guards.stride(p)) +
        U32.from_nat(k) : U32)) == Nat.add(Nat.mul(o,Guards.stride(p)),k) : Nat}:
  Encoded.madd(o,Guards.stride(p),k,Machine.coordinate(limits),Limits.coordinate_fits(limits),
    Bounds.strict_le(o,U32.to_nat(Machine.coordinate(limits)),Bounds.nat_trans(o,ConvSpec.out_width(p),
        U32.to_nat(Machine.coordinate(limits)),ho,col_limit(limits,p,valid,support))),
    stride_large(limits,p,support),
    Bounds.strict_le(k,U32.to_nat(Machine.coordinate(limits)),Bounds.nat_trans(k,Guards.kernel(p),
        U32.to_nat(Machine.coordinate(limits)),hk,kernel_large(limits,p,support))),
    Encoded.rectangle(o,ConvSpec.out_width(p),Guards.stride(p),k,Guards.kernel(p),
      U32.to_nat(Machine.coordinate(limits)),ho,hk,col_envelope(limits,p,support)))

def height_padding(+limits: Machine.Limits,+p: ConvSpec.Problem,+support: Guards.Supported<p,
  Machine.dimension(limits),Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {Nat.is_le(Nat.add(Guards.height(p),Guards.pad(p)),
    U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  Order.transitive(Nat.add(Guards.height(p),Guards.pad(p)),Nat.add(U32.to_nat(Machine.dimension(limits)),
      U32.to_nat(Machine.dimension(limits))),U32.to_nat(Machine.coordinate(limits)),
    Encoded.sum_bounds(Guards.height(p),Guards.pad(p),U32.to_nat(Machine.dimension(limits)),
      U32.to_nat(Machine.dimension(limits)),height_bound(limits,p,support),pad_bound(limits,p,support)),
    %Equal.sym(Nat,Nat.add(U32.to_nat(Machine.dimension(limits)),U32.to_nat(Machine.dimension(limits))),
      Nat.double(U32.to_nat(Machine.dimension(limits))),
      Addition.double(U32.to_nat(Machine.dimension(limits)))) : {Nat.is_le(_,
        U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}
    %U32Shift.double(Machine.dimension(limits),Limits.dimension_below_half(limits)) : {Nat.is_le(_,
        U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}
    Bounds.as_nat_le(U32.shl(Machine.dimension(limits)),Machine.coordinate(limits),
      Limits.doubled_dimension_within_coordinate(limits)))

def width_padding(+limits: Machine.Limits,+p: ConvSpec.Problem,+support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),Machine.coordinate(limits)>) -> {Nat.is_le(Nat.add(Guards.width(p),
      Guards.pad(p)),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  Order.transitive(Nat.add(Guards.width(p),Guards.pad(p)),Nat.add(U32.to_nat(Machine.dimension(limits)),
      U32.to_nat(Machine.dimension(limits))),U32.to_nat(Machine.coordinate(limits)),
    Encoded.sum_bounds(Guards.width(p),Guards.pad(p),U32.to_nat(Machine.dimension(limits)),
      U32.to_nat(Machine.dimension(limits)),width_bound(limits,p,support),pad_bound(limits,p,support)),
    %Equal.sym(Nat,Nat.add(U32.to_nat(Machine.dimension(limits)),U32.to_nat(Machine.dimension(limits))),
      Nat.double(U32.to_nat(Machine.dimension(limits))),
      Addition.double(U32.to_nat(Machine.dimension(limits)))) : {Nat.is_le(_,
        U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}
    %U32Shift.double(Machine.dimension(limits),Limits.dimension_below_half(limits)) : {Nat.is_le(_,
        U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}
    Bounds.as_nat_le(U32.shl(Machine.dimension(limits)),Machine.coordinate(limits),
      Limits.doubled_dimension_within_coordinate(limits)))
