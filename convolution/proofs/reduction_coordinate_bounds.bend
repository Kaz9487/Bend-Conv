import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
# Reduction extents and spatial width derived from actual scalar guards.
import Base
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./supported_padded_coordinates.bend as Padded
import ./native_weight_refinement.bend as NativeWeightRefinement
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ../../lib/proofs/binary_radix.bend as Radix
import ../specification.bend as ConvSpec

def inner(+p: ConvSpec.Problem) -> Nat:
  Nat.mul(Guards.input_channel_count(p),Nat.mul(Guards.kernel(p),Guards.kernel(p)))

def inner_bound(+limits: Machine.Limits,+p: ConvSpec.Problem,valid: Guards.Valid<p>,
  support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {Nat.is_le(inner(p),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Guards.Valid{v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10} = valid
  Order.transitive(inner(p),Nat.mul(Guards.output_channel_count(p),inner(p)),U32.to_nat(Machine.capacity(limits)),
    Padded.product_has_right(Guards.output_channel_count(p),inner(p),v1),
    %Guards.reflect(ConvSpec.length(Guards.weights(p)),Nat.mul(Guards.output_channel_count(p),inner(p)),
      v9) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    NativeWeightRefinement.weights_bound(limits,p,support))

def square_bound(+limits: Machine.Limits,+p: ConvSpec.Problem,+valid: Guards.Valid<p>,
  support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {Nat.is_le(Nat.mul(Guards.kernel(p),Guards.kernel(p)),
    U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Guards.Valid{v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10} = valid
  Order.transitive(Nat.mul(Guards.kernel(p),Guards.kernel(p)),inner(p),
    U32.to_nat(Machine.capacity(limits)),Padded.product_has_right(Guards.input_channel_count(p),
      Nat.mul(Guards.kernel(p),Guards.kernel(p)),v0),inner_bound(limits,p,valid,support))

def square_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,valid: Guards.Valid<p>,
  +support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {U32.to_nat((U32.from_nat(Guards.kernel(p)) *
    U32.from_nat(Guards.kernel(p)) : U32)) == Nat.mul(Guards.kernel(p),Guards.kernel(p)) : Nat}:
  Encoded.product(Guards.kernel(p),Guards.kernel(p),Machine.capacity(limits),
    Limits.capacity_fits(limits),NativeWeightRefinement.dimension(limits,Guards.kernel(p),
      SupportedCoordinateBounds.kernel_bound(limits,p,support)),NativeWeightRefinement.dimension(limits,Guards.kernel(p),
      SupportedCoordinateBounds.kernel_bound(limits,p,support)),square_bound(limits,p,valid,support))

def kernel_positive(+p: ConvSpec.Problem,valid: Guards.Valid<p>) -> {Nat.is_lt(0n,Guards.kernel(p)) == True{} : Bool}:
  Guards.Valid{v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10} = valid
  v4

def width_positive(+p: ConvSpec.Problem) -> {Nat.is_lt(0n,ConvSpec.out_width(p)) == True{} : Bool}:
  ConvSpec.Problem{shape,xs,ws,bs} = p
  ConvSpec.Shape{input_channel_count,output_channel_count,h,w,k,stride,pad} = shape
  {==}

def width_bound(+limits: Machine.Limits,+p: ConvSpec.Problem,valid: Guards.Valid<p>,
  support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {Nat.is_le(ConvSpec.out_width(p),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  ConvSpec.Problem{shape,xs,ws,bs} = p
  ConvSpec.Shape{input_channel_count,output_channel_count,h,w,k,stride,pad} = shape
  Order.transitive(ConvSpec.out_width(p),Nat.mul(ConvSpec.out_height(p),ConvSpec.out_width(p)),U32.to_nat(Machine.capacity(limits)),
    Padded.product_has_right(ConvSpec.out_height(p),ConvSpec.out_width(p),{==}),
    Padded.spatial_bound_facts(limits,p,valid,support))

def width_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,valid: Guards.Valid<p>,
  support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {U32.to_nat(U32.from_nat(ConvSpec.out_width(p))) == ConvSpec.out_width(p) : Nat}:
  Bounds.roundtrip(ConvSpec.out_width(p),Machine.capacity(limits),Limits.capacity_fits(limits),width_bound(limits,p,
      valid,support))

def reduction(+input_channel_index: Nat,+kernel_row_index: Nat,+kernel_column_index: Nat,+k: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(input_channel_index,k),kernel_row_index),k),kernel_column_index)

def reduction_inside(+p: ConvSpec.Problem,+input_channel_index: Nat,+kernel_row_index: Nat,+kernel_column_index: Nat,input_channel_in_bounds: {Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool},
  kernel_row_in_bounds: {Nat.is_lt(kernel_row_index,Guards.kernel(p)) == True{} : Bool},kernel_column_in_bounds: {Nat.is_lt(kernel_column_index,
      Guards.kernel(p)) == True{} : Bool}) -> {Nat.is_lt(reduction(input_channel_index,kernel_row_index,kernel_column_index,Guards.kernel(p)),
    inner(p)) == True{} : Bool}:
  %RowMajor.assoc(Guards.input_channel_count(p),Guards.kernel(p),Guards.kernel(p)) : {Nat.is_lt(reduction(input_channel_index,kernel_row_index,
        kernel_column_index,Guards.kernel(p)),_) == True{} : Bool}
  RowMajor.radix(Nat.add(Nat.mul(input_channel_index,Guards.kernel(p)),kernel_row_index),Nat.mul(Guards.input_channel_count(p),
      Guards.kernel(p)),kernel_column_index,Guards.kernel(p),RowMajor.radix(input_channel_index,Guards.input_channel_count(p),kernel_row_index,
      Guards.kernel(p),input_channel_in_bounds,kernel_row_in_bounds),kernel_column_in_bounds)

def position(+p: ConvSpec.Problem,output_row_index: Nat,output_column_index: Nat) -> Nat:
  Nat.add(Nat.mul(output_row_index,ConvSpec.out_width(p)),output_column_index)

def position_bound(+limits: Machine.Limits,+p: ConvSpec.Problem,+output_row_index: Nat,+output_column_index: Nat,valid: Guards.Valid<p>,
  support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,
  output_row_in_bounds: {Nat.is_lt(output_row_index,ConvSpec.out_height(p)) == True{} : Bool},output_column_in_bounds: {Nat.is_lt(output_column_index,
      ConvSpec.out_width(p)) == True{} : Bool}) -> {Nat.is_le(position(p,output_row_index,output_column_index),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Bounds.strict_le(position(p,output_row_index,output_column_index),U32.to_nat(Machine.capacity(limits)),Bounds.nat_trans(position(p,output_row_index,output_column_index),
      Nat.mul(ConvSpec.out_height(p),ConvSpec.out_width(p)),U32.to_nat(Machine.capacity(limits)),RowMajor.radix(output_row_index,
        ConvSpec.out_height(p),output_column_index,ConvSpec.out_width(p),output_row_in_bounds,output_column_in_bounds),Padded.spatial_bound_facts(limits,p,valid,support)))

def inner_positive(+p: ConvSpec.Problem,valid: Guards.Valid<p>) -> {Nat.is_lt(0n,inner(p)) == True{} : Bool}:
  Guards.Valid{channels,outputs,height,width,+kernel,stride,padding,extent,input,weights,bias} = valid
  Support.positive_mul(Guards.input_channel_count(p),Nat.mul(Guards.kernel(p),Guards.kernel(p)),
    channels,Support.positive_mul(Guards.kernel(p),Guards.kernel(p),kernel,kernel))

def native_inner(+p: ConvSpec.Problem) -> U32:
  (U32.from_nat(Guards.input_channel_count(p)) * U32.from_nat(Guards.kernel(p)) * U32.from_nat(Guards.kernel(p)) : U32)

def inner_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,+valid: Guards.Valid<p>,
  +support: Guards.Supported<p,Machine.dimension(limits),Machine.capacity(limits),
  Machine.coordinate(limits)>) -> {U32.to_nat(native_inner(p)) == inner(p) : Nat}:
  +channels = Guards.input_channel_count(p)
  +kernel = Guards.kernel(p)
  +upper = Machine.capacity(limits)
  +upper_ok = Limits.capacity_fits(limits)
  +product_bound =
    Order.transitive(Nat.mul(channels,kernel),Nat.mul(Nat.mul(channels,kernel),kernel),U32.to_nat(upper),
      RowMajor.product_covers(Nat.mul(channels,kernel),kernel,kernel_positive(p,valid)),
      %Equal.sym(Nat,Nat.mul(Nat.mul(channels,kernel),kernel),inner(p),RowMajor.assoc(channels,kernel,kernel)) : {Nat.is_le(_,
          U32.to_nat(upper)) == True{} : Bool}
      inner_bound(limits,p,valid,support))
  +encoded_product = Encoded.product(channels,kernel,upper,upper_ok,NativeWeightRefinement.dimension(limits,channels,
      SupportedCoordinateBounds.ci_bound(limits,p,support)),NativeWeightRefinement.dimension(limits,kernel,
      SupportedCoordinateBounds.kernel_bound(limits,p,support)),product_bound)
  %RowMajor.assoc(channels,kernel,kernel) : {U32.to_nat(native_inner(p)) == _ : Nat}
  %encoded_product : {U32.to_nat(native_inner(p)) == Nat.mul(_,kernel) : Nat}
  %SupportedCoordinateBounds.kernel_exact(limits,p,
    support) : {U32.to_nat(native_inner(p)) == Nat.mul(U32.to_nat((U32.from_nat(channels) * U32.from_nat(kernel) : U32)),_) : Nat}
  Arithmetic.multiply((U32.from_nat(channels) * U32.from_nat(kernel) : U32),U32.from_nat(kernel),upper,upper_ok,
    %Equal.sym(Nat,U32.to_nat((U32.from_nat(channels) * U32.from_nat(kernel) : U32)),Nat.mul(channels,kernel),
      encoded_product) : {Nat.is_le(Nat.mul(_,U32.to_nat(U32.from_nat(kernel))),U32.to_nat(upper)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(kernel)),kernel,SupportedCoordinateBounds.kernel_exact(limits,p,
        support)) : {Nat.is_le(Nat.mul(Nat.mul(channels,kernel),_),U32.to_nat(upper)) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(Nat.mul(channels,kernel),kernel),inner(p),RowMajor.assoc(channels,kernel,kernel)) : {Nat.is_le(_,
        U32.to_nat(upper)) == True{} : Bool}
    inner_bound(limits,p,valid,support))

# Direct sampling and packing decode the same linear reduction index. Keep
# this machine-arithmetic contract independent of either execution strategy.
type DecodedReduction<-index: U32,-kernel: U32,-channel: Nat,-row: Nat,-column: Nat> is Data:
  DecodedReduction{
    channel: {(index / (kernel * kernel) : U32) == U32.from_nat(channel) : U32},
    row: {(index / kernel % kernel : U32) == U32.from_nat(row) : U32},
    column: {(index % kernel : U32) == U32.from_nat(column) : U32}}

def decode_reduction(+limits: Machine.Limits,+problem: ConvSpec.Problem,+index: U32,
  +input_channel: Nat,+kernel_row: Nat,+kernel_column: Nat,
  +valid: Guards.Valid<problem>,
  +supported: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,
  +exact: {U32.to_nat(index) == reduction(input_channel,kernel_row,kernel_column,Guards.kernel(problem)) : Nat},
  +row_inside: {Nat.is_lt(kernel_row,Guards.kernel(problem)) == True{} : Bool},
  +column_inside: {Nat.is_lt(kernel_column,Guards.kernel(problem)) == True{} : Bool}) ->
  DecodedReduction<index,U32.from_nat(Guards.kernel(problem)),input_channel,kernel_row,kernel_column>:
  +kernel = Guards.kernel(problem)
  +native_kernel = U32.from_nat(kernel)
  +positive = kernel_positive(problem,valid)
  +kernel_exact = SupportedCoordinateBounds.kernel_exact(limits,problem,supported)
  +kernel_double = Limits.double_bound(limits,kernel,NativeWeightRefinement.dimension(limits,kernel,SupportedCoordinateBounds.kernel_bound(limits,problem,supported)))
  +half_word = Machine.half_word(limits)
  +half_word_fits = Limits.half_word_fits(limits)
  +row_quotient = Radix.quotient_word(index,native_kernel,Nat.add(Nat.mul(input_channel,kernel),kernel_row),kernel_column,kernel,
    half_word,half_word_fits,exact,kernel_exact,positive,column_inside,kernel_double)
  +channel_index = Equal.trans(Nat,U32.to_nat(index),reduction(input_channel,kernel_row,kernel_column,kernel),
    RowMajor.index(input_channel,kernel,kernel,kernel_row,kernel_column),exact,RowMajor.mixed(input_channel,kernel,kernel,kernel_row,kernel_column))
  DecodedReduction{
    Encoded.encoding((index / (native_kernel * native_kernel) : U32),input_channel,
      Radix.quotient_word(index,(native_kernel * native_kernel : U32),input_channel,Nat.add(Nat.mul(kernel_row,kernel),kernel_column),Nat.mul(kernel,kernel),
        half_word,half_word_fits,channel_index,square_exact(limits,problem,valid,supported),Support.positive_mul(kernel,kernel,positive,positive),
        RowMajor.radix(kernel_row,kernel,kernel_column,kernel,row_inside,column_inside),Limits.double_bound(limits,Nat.mul(kernel,kernel),square_bound(limits,problem,valid,supported)))),
    Encoded.encoding((index / native_kernel % native_kernel : U32),kernel_row,
      Radix.remainder_word((index / native_kernel : U32),native_kernel,input_channel,kernel_row,kernel,half_word,half_word_fits,
        row_quotient,kernel_exact,positive,row_inside,kernel_double)),
    Encoded.encoding((index % native_kernel : U32),kernel_column,
      Radix.remainder_word(index,native_kernel,Nat.add(Nat.mul(input_channel,kernel),kernel_row),kernel_column,kernel,half_word,half_word_fits,
        exact,kernel_exact,positive,column_inside,kernel_double))}

def weight_index(+output_channel_index: Nat,+input_channel_count: Nat,+input_channel_index: Nat,+kernel: Nat,+kernel_row_index: Nat,+kernel_column_index: Nat) -> {NativeWeightRefinement.weight_index(output_channel_index,input_channel_count,input_channel_index,kernel,kernel_row_index,kernel_column_index) == Nat.add(Nat.mul(output_channel_index,
      Nat.mul(input_channel_count,Nat.mul(kernel,kernel))),reduction(input_channel_index,kernel_row_index,kernel_column_index,kernel)) : Nat}:
  +area = Nat.mul(kernel,kernel)
  +tail = Nat.add(Nat.mul(kernel_row_index,kernel),kernel_column_index)
  +prefix = Nat.mul(output_channel_index,input_channel_count)
  %Equal.sym(Nat,NativeWeightRefinement.weight_index(output_channel_index,input_channel_count,input_channel_index,kernel,kernel_row_index,kernel_column_index),RowMajor.index(Nat.add(prefix,input_channel_index),kernel,kernel,kernel_row_index,kernel_column_index),RowMajor.mixed(Nat.add(prefix,input_channel_index),
      kernel,kernel,kernel_row_index,kernel_column_index)) : {_ == Nat.add(Nat.mul(output_channel_index,Nat.mul(input_channel_count,area)),reduction(input_channel_index,kernel_row_index,kernel_column_index,kernel)) : Nat}
  %Equal.sym(Nat,reduction(input_channel_index,kernel_row_index,kernel_column_index,kernel),RowMajor.index(input_channel_index,kernel,kernel,kernel_row_index,kernel_column_index),RowMajor.mixed(input_channel_index,kernel,kernel,kernel_row_index,
      kernel_column_index)) : {Nat.add(Nat.mul(Nat.add(prefix,input_channel_index),area),tail) == Nat.add(Nat.mul(output_channel_index,Nat.mul(input_channel_count,area)),_) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.add(prefix,input_channel_index),area),Nat.add(Nat.mul(prefix,area),Nat.mul(input_channel_index,area)),RowMajor.distrib(prefix,input_channel_index,area)) : {Nat.add(_,
      tail) == Nat.add(Nat.mul(output_channel_index,Nat.mul(input_channel_count,area)),Nat.add(Nat.mul(input_channel_index,area),tail)) : Nat}
  %RowMajor.assoc(output_channel_index,input_channel_count,area) : {Nat.add(Nat.add(Nat.mul(prefix,area),Nat.mul(input_channel_index,area)),tail) == Nat.add(_,Nat.add(Nat.mul(input_channel_index,area),tail)) : Nat}
  Addition.associative(Nat.mul(prefix,area),Nat.mul(input_channel_index,area),tail)
