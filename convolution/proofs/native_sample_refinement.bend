import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
# Actual native pixel read equals frozen SPEC.sample at convolution coordinates.
# All allocation and arithmetic facts come from the actual runtime guards.
import Base
import ./input_storage.bend as Inputs
import ./concrete_support_guard.bend as Support
import ./convolution_guard_facts.bend as Guards
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./padded_pixel_address.bend as PaddedPixelAddress
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../specification.bend as ConvSpec
import ../scalar_kernel.bend as ScalarConv
import ../../lib/kernel_spatial_packing.bend as Packing

def logical(+xs: List<&2,F32>,+input_channel_index: Nat,+h: Nat,+w: Nat,+padded_row: Nat,+padded_column: Nat,+pad: Nat) -> F32:
  ConvSpec.sample_if((Nat.is_le(pad,padded_row) && Nat.is_lt(padded_row,Nat.add(pad,h))) && (Nat.is_le(pad,padded_column) && Nat.is_lt(padded_column,Nat.add(pad,w))),
    Nat.add(Nat.mul(input_channel_index,Nat.mul(h,w)),Nat.add(Nat.mul(Nat.sub(padded_row,pad),w),Nat.sub(padded_column,pad))),xs)

def regroup(+a: Bool,+b: Bool,+c: Bool,+d: Bool) -> {(a && b) && (c && d) == a && b && c && d : Bool}:
  match a b c:
    case False{} b c:
      {==}
    case True{} False{} c:
      {==}
    case True{} True{} False{}:
      {==}
    case True{} True{} True{}:
      {==}

def logical_sample(+p: ConvSpec.Problem,+input_channel_index: Nat,+output_row_index: Nat,+output_column_index: Nat,+kernel_row_index: Nat,+kernel_column_index: Nat) -> {logical(Guards.input(p),input_channel_index,
    Guards.height(p),Guards.width(p),Nat.add(Nat.mul(output_row_index,Guards.stride(p)),kernel_row_index),Nat.add(Nat.mul(output_column_index,
        Guards.stride(p)),kernel_column_index),Guards.pad(p)) == ConvSpec.sample(p,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index) : F32}:
  ConvSpec.Problem{shape,+xs,+ws,+bs} = p
  ConvSpec.Shape{+v0,+v1,+v2,+v3,+v4,+v5,+v6} = shape
  +padded_row = Nat.add(Nat.mul(output_row_index,v5),kernel_row_index)
  +padded_column = Nat.add(Nat.mul(output_column_index,v5),kernel_column_index)
  %regroup(Nat.is_le(v6,padded_row),Nat.is_lt(padded_row,Nat.add(v6,v2)),Nat.is_le(v6,padded_column),Nat.is_lt(padded_column,Nat.add(v6,v3))) : {logical(xs,input_channel_index,v2,v3,padded_row,padded_column,
      v6) == ConvSpec.sample_if(_,Nat.add(Nat.mul(input_channel_index,Nat.mul(v2,v3)),Nat.add(Nat.mul(Nat.sub(padded_row,v6),v3),Nat.sub(padded_column,v6))),xs) : F32}
  {==}

def value_exact(+xs: List<&2,F32>,+uic: U32, +uh: U32, +uw: U32, +upy: U32, +upx: U32, +upad: U32, +input_channel_index: Nat, +h: Nat, +w: Nat, +padded_row: Nat,
  +padded_column: Nat, +pad: Nat, eic: {U32.to_nat(uic) == input_channel_index : Nat}, eh: {U32.to_nat(uh) == h : Nat}, ew: {U32.to_nat(uw) == w : Nat},
  epy: {U32.to_nat(upy) == padded_row : Nat}, epx: {U32.to_nat(upx) == padded_column : Nat}, epad: {U32.to_nat(upad) == pad : Nat}) -> {PaddedPixelAddress.value(xs,
    uic,uh,uw,upy,upx,upad) == logical(xs,input_channel_index,h,w,padded_row,padded_column,pad) : F32}:
  %eic : {PaddedPixelAddress.value(xs,uic,uh,uw,upy,upx,upad) == logical(xs,_,h,w,padded_row,padded_column,pad) : F32}
  %eh : {PaddedPixelAddress.value(xs,uic,uh,uw,upy,upx,upad) == logical(xs,U32.to_nat(uic),_,w,padded_row,padded_column,pad) : F32}
  %ew : {PaddedPixelAddress.value(xs,uic,uh,uw,upy,upx,upad) == logical(xs,U32.to_nat(uic),U32.to_nat(uh),_,padded_row,padded_column,pad) : F32}
  %epy : {PaddedPixelAddress.value(xs,uic,uh,uw,upy,upx,upad) == logical(xs,U32.to_nat(uic),U32.to_nat(uh),U32.to_nat(uw),_,padded_column,pad) : F32}
  %epx : {PaddedPixelAddress.value(xs,uic,uh,uw,upy,upx,upad) == logical(xs,U32.to_nat(uic),U32.to_nat(uh),U32.to_nat(uw),U32.to_nat(upy),_,
      pad) : F32}
  %epad : {PaddedPixelAddress.value(xs,uic,uh,uw,upy,upx,upad) == logical(xs,U32.to_nat(uic),U32.to_nat(uh),U32.to_nat(uw),U32.to_nat(upy),
      U32.to_nat(upx),_) : F32}
  {==}

def channel_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,+input_channel_index: Nat,support: Guards.Supported<p,
  Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,h: {Nat.is_lt(input_channel_index,
      Guards.input_channel_count(p)) == True{} : Bool}) -> {U32.to_nat(U32.from_nat(input_channel_index)) == input_channel_index : Nat}:
  Bounds.roundtrip(input_channel_index,Machine.dimension(limits),Limits.dimension_fits(limits),Bounds.strict_le(input_channel_index,
      U32.to_nat(Machine.dimension(limits)),Bounds.nat_trans(input_channel_index,Guards.input_channel_count(p),
        U32.to_nat(Machine.dimension(limits)),h,SupportedCoordinateBounds.ci_bound(limits,p,support))))

def input_shape(+limits: Machine.Limits,+p: ConvSpec.Problem,valid: Guards.Valid<p>,
  +support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {ConvSpec.length(Guards.input(p)) ==
    Nat.mul(U32.to_nat(U32.from_nat(Guards.input_channel_count(p))),Nat.mul(U32.to_nat(U32.from_nat(Guards.height(p))),U32.to_nat(U32.from_nat(Guards.width(p))))) : Nat}:
  Guards.Valid{v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10} = valid
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.input_channel_count(p))),Guards.input_channel_count(p),SupportedCoordinateBounds.ci_exact(limits,p,
      support)) : {ConvSpec.length(Guards.input(p)) == Nat.mul(_,
      Nat.mul(U32.to_nat(U32.from_nat(Guards.height(p))),U32.to_nat(U32.from_nat(Guards.width(p))))) : Nat}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.height(p))),Guards.height(p),
    SupportedCoordinateBounds.height_exact(limits,p,
      support)) : {ConvSpec.length(Guards.input(p)) == Nat.mul(Guards.input_channel_count(p),Nat.mul(_,
        U32.to_nat(U32.from_nat(Guards.width(p))))) : Nat}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.width(p))),Guards.width(p),
    SupportedCoordinateBounds.width_exact(limits,p,
      support)) : {ConvSpec.length(Guards.input(p)) == Nat.mul(Guards.input_channel_count(p),
      Nat.mul(Guards.height(p),_)) : Nat}
  Guards.reflect(ConvSpec.length(Guards.input(p)),Nat.mul(Guards.input_channel_count(p),
      Nat.mul(Guards.height(p),Guards.width(p))),v8)

def width_positive(+limits: Machine.Limits,+p: ConvSpec.Problem,valid: Guards.Valid<p>,
  support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {Nat.is_lt(0n,U32.to_nat(U32.from_nat(Guards.width(p)))) == True{} : Bool}:
  Guards.Valid{v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10} = valid
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.width(p))),Guards.width(p),
    SupportedCoordinateBounds.width_exact(limits,p,support)) : {Nat.is_lt(0n,_) == True{} : Bool}
  v3

def height_padding(+limits: Machine.Limits,+p: ConvSpec.Problem,+support: Guards.Supported<p,
  Machine.dimension(limits),Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {Nat.is_le(Nat.add(U32.to_nat(U32.from_nat(Guards.height(p))),
      U32.to_nat(U32.from_nat(Guards.pad(p)))),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.height(p))),Guards.height(p),
    SupportedCoordinateBounds.height_exact(limits,p,support)) : {Nat.is_le(Nat.add(_,U32.to_nat(U32.from_nat(Guards.pad(p)))),
      U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.pad(p))),Guards.pad(p),SupportedCoordinateBounds.pad_exact(limits,
      p,support)) : {Nat.is_le(Nat.add(Guards.height(p),_),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}
  SupportedCoordinateBounds.height_padding(limits,p,support)

def width_padding(+limits: Machine.Limits,+p: ConvSpec.Problem,+support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {Nat.is_le(Nat.add(U32.to_nat(U32.from_nat(Guards.width(p))),
      U32.to_nat(U32.from_nat(Guards.pad(p)))),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.width(p))),Guards.width(p),
    SupportedCoordinateBounds.width_exact(limits,p,support)) : {Nat.is_le(Nat.add(_,U32.to_nat(U32.from_nat(Guards.pad(p)))),
      U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.pad(p))),Guards.pad(p),SupportedCoordinateBounds.pad_exact(limits,
      p,support)) : {Nat.is_le(Nat.add(Guards.width(p),_),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}
  SupportedCoordinateBounds.width_padding(limits,p,support)

law native_sample:
  for +limits: Machine.Limits
  for +p: ConvSpec.Problem
  for +storage_contract: Inputs.Contract<p>
  for +input_channel_index: Nat
  for +output_row_index: Nat
  for +output_column_index: Nat
  for +kernel_row_index: Nat
  for +kernel_column_index: Nat
  for valid: {ConvSpec.valid(p) == True{} : Bool}
  for supported: {ScalarConv.supported_with_limits(limits,p) == True{} : Bool}
  for +input_channel_in_bounds: {Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool}
  for +output_row_in_bounds: {Nat.is_lt(output_row_index,ConvSpec.out_height(p)) == True{} : Bool}
  for +output_column_in_bounds: {Nat.is_lt(output_column_index,ConvSpec.out_width(p)) == True{} : Bool}
  for +kernel_row_in_bounds: {Nat.is_lt(kernel_row_index,Guards.kernel(p)) == True{} : Bool}
  for +kernel_column_in_bounds: {Nat.is_lt(kernel_column_index,Guards.kernel(p)) == True{} : Bool}
  {Packing.pixel(Inputs.input_owner(p,storage_contract),
      ((U32.from_nat(output_row_index) * U32.from_nat(Guards.stride(p)) + U32.from_nat(kernel_row_index) : U32) - U32.from_nat(Guards.pad(p)) :
        U32),((U32.from_nat(output_column_index) * U32.from_nat(Guards.stride(p)) + U32.from_nat(kernel_column_index) : U32) -
        U32.from_nat(Guards.pad(p)) : U32),(U32.from_nat(input_channel_index) * U32.from_nat(Guards.height(p)) *
        U32.from_nat(Guards.width(p)) : U32),U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p))) == (Inputs.input_owner(p,storage_contract),ConvSpec.sample(p,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index)) : Array<F32> & F32}
def native_sample(limits,p,storage_contract,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index,valid,supported,input_channel_in_bounds,output_row_in_bounds,output_column_in_bounds,kernel_row_in_bounds,kernel_column_in_bounds):
  +v = Guards.valid(p,valid)
  +support = Support.scalar(limits,p,supported)
  +eic = channel_exact(limits,p,input_channel_index,support,input_channel_in_bounds)
  %logical_sample(p,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index) : {Packing.pixel(Inputs.input_owner(p,storage_contract),
      ((U32.from_nat(output_row_index) * U32.from_nat(Guards.stride(p)) + U32.from_nat(kernel_row_index) : U32) - U32.from_nat(Guards.pad(p)) :
        U32),((U32.from_nat(output_column_index) * U32.from_nat(Guards.stride(p)) + U32.from_nat(kernel_column_index) : U32) -
        U32.from_nat(Guards.pad(p)) : U32),(U32.from_nat(input_channel_index) * U32.from_nat(Guards.height(p)) *
        U32.from_nat(Guards.width(p)) : U32),U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p))) == (Inputs.input_owner(p,storage_contract),_) : Array<F32> & F32}
  %value_exact(Guards.input(p),U32.from_nat(input_channel_index),U32.from_nat(Guards.height(p)),
    U32.from_nat(Guards.width(p)),(U32.from_nat(output_row_index) * U32.from_nat(Guards.stride(p)) + U32.from_nat(kernel_row_index) : U32),
    (U32.from_nat(output_column_index) * U32.from_nat(Guards.stride(p)) + U32.from_nat(kernel_column_index) : U32),U32.from_nat(Guards.pad(p)),input_channel_index,
    Guards.height(p),Guards.width(p),Nat.add(Nat.mul(output_row_index,Guards.stride(p)),kernel_row_index),Nat.add(Nat.mul(output_column_index,
        Guards.stride(p)),kernel_column_index),Guards.pad(p),eic,SupportedCoordinateBounds.height_exact(limits,p,support),
    SupportedCoordinateBounds.width_exact(limits,p,support),SupportedCoordinateBounds.row_coordinate(limits,p,output_row_index,kernel_row_index,v,support,output_row_in_bounds,kernel_row_in_bounds),
    SupportedCoordinateBounds.col_coordinate(limits,p,output_column_index,kernel_column_index,v,support,output_column_in_bounds,kernel_column_in_bounds),SupportedCoordinateBounds.pad_exact(limits,p,
      support)) : {Packing.pixel(Inputs.input_owner(p,storage_contract),
      ((U32.from_nat(output_row_index) * U32.from_nat(Guards.stride(p)) + U32.from_nat(kernel_row_index) : U32) - U32.from_nat(Guards.pad(p)) :
        U32),((U32.from_nat(output_column_index) * U32.from_nat(Guards.stride(p)) + U32.from_nat(kernel_column_index) : U32) -
        U32.from_nat(Guards.pad(p)) : U32),(U32.from_nat(input_channel_index) * U32.from_nat(Guards.height(p)) *
        U32.from_nat(Guards.width(p)) : U32),U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p))) == (Inputs.input_owner(p,storage_contract),_) : Array<F32> & F32}
  PaddedPixelAddress.native_pixel(Guards.input(p),Inputs.input_sequence(p,storage_contract),U32.from_nat(input_channel_index),U32.from_nat(Guards.input_channel_count(p)),
    U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),
    (U32.from_nat(output_row_index) * U32.from_nat(Guards.stride(p)) + U32.from_nat(kernel_row_index) : U32),
    (U32.from_nat(output_column_index) * U32.from_nat(Guards.stride(p)) + U32.from_nat(kernel_column_index) : U32),U32.from_nat(Guards.pad(p)),
    Support.input_ready_facts(limits,p,v,support),input_shape(limits,p,v,support),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(input_channel_index)),input_channel_index,eic) : {Nat.is_lt(_,U32.to_nat(U32.from_nat(Guards.input_channel_count(p)))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.input_channel_count(p))),Guards.input_channel_count(p),SupportedCoordinateBounds.ci_exact(limits,p,
        support)) : {Nat.is_lt(input_channel_index,_) == True{} : Bool}
    input_channel_in_bounds,
    width_positive(limits,p,v,support),Machine.coordinate(limits),Limits.coordinate_fits(limits),height_padding(limits,p,
      support),width_padding(limits,p,support))
