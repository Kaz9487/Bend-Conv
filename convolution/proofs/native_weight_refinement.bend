import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
# The measured native weight and bias addresses denote frozen SPEC accessors.
import Base
import ./convolution_guard_facts.bend as Guards
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_order.bend as Order
import ../specification.bend as ConvSpec

def channel_index(+output_channel_index: Nat,+input_channel_count: Nat,input_channel_index: Nat) -> Nat:
  Nat.add(Nat.mul(output_channel_index,input_channel_count),input_channel_index)

def kernel_row_index(+output_channel_index: Nat,+input_channel_count: Nat,input_channel_index: Nat,+k: Nat,kernel_row: Nat) -> Nat:
  Nat.add(Nat.mul(channel_index(output_channel_index,input_channel_count,input_channel_index),k),kernel_row)

def weight_index(+output_channel_index: Nat,+input_channel_count: Nat,input_channel_index: Nat,+k: Nat,kernel_row: Nat,kernel_column_index: Nat) -> Nat:
  Nat.add(Nat.mul(kernel_row_index(output_channel_index,input_channel_count,input_channel_index,k,kernel_row),k),kernel_column_index)

def size_associate(+output_channel_count: Nat,+input_channel_count: Nat,+k: Nat) -> {Nat.mul(Nat.mul(Nat.mul(output_channel_count,input_channel_count),k),k) == Nat.mul(output_channel_count,Nat.mul(input_channel_count,Nat.mul(k,k))) : Nat}:
  %RowMajor.assoc(output_channel_count,input_channel_count,Nat.mul(k,k)) : {Nat.mul(Nat.mul(Nat.mul(output_channel_count,input_channel_count),k),k) == _ : Nat}
  RowMajor.assoc(Nat.mul(output_channel_count,input_channel_count),k,k)

def index_bound(+p: ConvSpec.Problem,+output_channel_index: Nat,+input_channel_index: Nat,+kernel_row: Nat,+kernel_column_index: Nat,valid: Guards.Valid<p>,output_channel_in_bounds: {Nat.is_lt(output_channel_index,
      Guards.output_channel_count(p)) == True{} : Bool},input_channel_in_bounds: {Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool},kernel_row_in_bounds: {Nat.is_lt(kernel_row,
      Guards.kernel(p)) == True{} : Bool},kernel_column_in_bounds: {Nat.is_lt(kernel_column_index,
      Guards.kernel(p)) == True{} : Bool}) -> {Nat.is_lt(weight_index(output_channel_index,Guards.input_channel_count(p),input_channel_index,Guards.kernel(p),
      kernel_row,kernel_column_index),ConvSpec.length(Guards.weights(p))) == True{} : Bool}:
  Guards.Valid{v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10} = valid
  %Equal.sym(Nat,ConvSpec.length(Guards.weights(p)),Nat.mul(Guards.output_channel_count(p),Nat.mul(Guards.input_channel_count(p),
        Nat.mul(Guards.kernel(p),Guards.kernel(p)))),
    Guards.reflect(ConvSpec.length(Guards.weights(p)),Nat.mul(Guards.output_channel_count(p),
        Nat.mul(Guards.input_channel_count(p),Nat.mul(Guards.kernel(p),Guards.kernel(p)))),
      v9)) : {Nat.is_lt(weight_index(output_channel_index,Guards.input_channel_count(p),input_channel_index,Guards.kernel(p),kernel_row,kernel_column_index),_) == True{} : Bool}
  %size_associate(Guards.output_channel_count(p),Guards.input_channel_count(p),Guards.kernel(p)) : {Nat.is_lt(weight_index(output_channel_index,
        Guards.input_channel_count(p),input_channel_index,Guards.kernel(p),kernel_row,kernel_column_index),_) == True{} : Bool}
  RowMajor.radix(kernel_row_index(output_channel_index,Guards.input_channel_count(p),input_channel_index,Guards.kernel(p),kernel_row),Nat.mul(Nat.mul(Guards.output_channel_count(p),
        Guards.input_channel_count(p)),Guards.kernel(p)),kernel_column_index,Guards.kernel(p),RowMajor.radix(channel_index(output_channel_index,
        Guards.input_channel_count(p),input_channel_index),Nat.mul(Guards.output_channel_count(p),Guards.input_channel_count(p)),kernel_row,Guards.kernel(p),
      RowMajor.radix(output_channel_index,Guards.output_channel_count(p),input_channel_index,Guards.input_channel_count(p),output_channel_in_bounds,input_channel_in_bounds),kernel_row_in_bounds),kernel_column_in_bounds)

def weights_bound(+limits: Machine.Limits,+p: ConvSpec.Problem,support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),Machine.coordinate(limits)>) -> {Nat.is_le(ConvSpec.length(Guards.weights(p)),
    U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Guards.Supported{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11} = support
  h8

def below_dimension(+limits: Machine.Limits,+n: Nat,+d: Nat,h: {Nat.is_lt(n,d) == True{} : Bool},bound: {Nat.is_le(d,
      U32.to_nat(Machine.dimension(limits))) == True{} : Bool}) -> {Nat.is_le(n,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(n,U32.to_nat(Machine.dimension(limits)),U32.to_nat(Machine.capacity(limits)),Bounds.strict_le(n,
      U32.to_nat(Machine.dimension(limits)),Bounds.nat_trans(n,d,U32.to_nat(Machine.dimension(limits)),h,bound)),
    Bounds.as_nat_le(Machine.dimension(limits),Machine.capacity(limits),Limits.dimension_within_capacity(limits)))

def dimension(+limits: Machine.Limits,+n: Nat,h: {Nat.is_le(n,
      U32.to_nat(Machine.dimension(limits))) == True{} : Bool}) -> {Nat.is_le(n,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(n,U32.to_nat(Machine.dimension(limits)),U32.to_nat(Machine.capacity(limits)),h,
    Bounds.as_nat_le(Machine.dimension(limits),Machine.capacity(limits),Limits.dimension_within_capacity(limits)))

def bias_index(+p: ConvSpec.Problem,+output_channel_index: Nat,v: Guards.Valid<p>,output_channel_in_bounds: {Nat.is_lt(output_channel_index,
      Guards.output_channel_count(p)) == True{} : Bool}) -> {Nat.is_lt(output_channel_index,ConvSpec.length(Guards.bias(p))) == True{} : Bool}:
  Guards.Valid{h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10} = v
  %Equal.sym(Nat,ConvSpec.length(Guards.bias(p)),Guards.output_channel_count(p),
    Guards.reflect(ConvSpec.length(Guards.bias(p)),Guards.output_channel_count(p),h10)) : {Nat.is_lt(output_channel_index,
      _) == True{} : Bool}
  output_channel_in_bounds
