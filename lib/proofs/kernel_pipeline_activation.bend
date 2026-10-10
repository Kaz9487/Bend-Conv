# Shared array/tree semantics and readback of the final activation pass.
import Base
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_allocation_depth.bend as Depth
import ./word_power_of_two.bend as Power
import ./nat_order.bend as Order
import ../storage_buffer.bend as Storage
import ../storage_allocation.bend as Allocation
import ../tensor.bend as Tensors
import ../kernel_pipeline.bend as Pipeline
import ../math_activation.bend as Activations
import ../traversal_array.bend as Traversal
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_certificate.bend as Certificate
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./tensor_band_map.bend as BandMap
import ./traversal_map_readback.bend as MapReadback
import ./traversal_lanes_certificate.bend as LanesCertificate

def storage(~Element: Data,buffer: Storage.Buffer<Element>) -> Array<Element>:
  Storage.Buffer{storage,length,certificate} = buffer
  storage

def allocated(~Element: Data,+fill: Element,+count: U32) ->
  {Tensors.buffer(~Element,fill,count) == Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)) : Storage.Buffer<Element>}:
  AllocationLaws.dispatch_same(Element,Allocation.allocated(~Element),Tensors.depth_for(count),fill,U32.to_nat(count))

def activation_array(~custom: F32 -> F32,activation: Activations.Activation,+count: U32,values: Array<F32>) -> Array<F32>:
  match activation:
    case Activations.Identity{}: values
    case Activations.ReLU{}: Traversal.map_in_place(~F32,~Unit,~Pipeline.activated_value(~Activations.relu_value),U32.to_nat(count),0,Unit{},values)
    case Activations.ReLU6{}: Traversal.map_in_place(~F32,~Unit,~Pipeline.activated_value(~Activations.relu6_value),U32.to_nat(count),0,Unit{},values)
    case Activations.LeakyReLU{slope}: Traversal.map_in_place(~F32,~F32,~Pipeline.leaky_relu_value,U32.to_nat(count),0,slope,values)
    case Activations.Sigmoid{}: Traversal.map_in_place(~F32,~Unit,~Pipeline.activated_value(~Activations.sigmoid_value),U32.to_nat(count),0,Unit{},values)
    case Activations.SiLU{}: Traversal.map_in_place(~F32,~Unit,~Pipeline.activated_value(~Activations.silu_value),U32.to_nat(count),0,Unit{},values)
    case Activations.Tanh{}: Traversal.map_in_place(~F32,~Unit,~Pipeline.activated_value(~Activations.tanh_value),U32.to_nat(count),0,Unit{},values)
    case Activations.HardSigmoid{}: Traversal.map_in_place(~F32,~Unit,~Pipeline.activated_value(~Activations.hard_sigmoid_value),U32.to_nat(count),0,Unit{},values)
    case Activations.HardSwish{}: Traversal.map_in_place(~F32,~Unit,~Pipeline.activated_value(~Activations.hard_swish_value),U32.to_nat(count),0,Unit{},values)
    case Activations.GELUTanh{}: Traversal.map_in_place(~F32,~Unit,~Pipeline.activated_value(~Activations.gelu_tanh_value),U32.to_nat(count),0,Unit{},values)
    case Activations.Custom{operations}: Traversal.map_in_place(~F32,~Unit,~Pipeline.activated_value(~custom),U32.to_nat(count),0,Unit{},values)

def activated(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,buffer: Storage.Buffer<F32>) ->
  {storage(~F32,Pipeline.activated(~custom,activation,count,buffer)) == activation_array(~custom,activation,count,storage(~F32,buffer)) : Array<F32>}:
  match activation buffer:
    case Activations.Identity{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.ReLU{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.ReLU6{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.LeakyReLU{slope} Storage.Buffer{values,length,certificate}: {==}
    case Activations.Sigmoid{} Storage.Buffer{values,length,Certificate.Certificate{depth,equation}}:
      LanesCertificate.certified(~Unit,~Pipeline.activated_value(~Activations.sigmoid_value),U32.to_nat(count),0,Unit{},values,depth,equation)
    case Activations.SiLU{} Storage.Buffer{values,length,Certificate.Certificate{depth,equation}}:
      LanesCertificate.certified(~Unit,~Pipeline.activated_value(~Activations.silu_value),U32.to_nat(count),0,Unit{},values,depth,equation)
    case Activations.Tanh{} Storage.Buffer{values,length,Certificate.Certificate{depth,equation}}:
      LanesCertificate.certified(~Unit,~Pipeline.activated_value(~Activations.tanh_value),U32.to_nat(count),0,Unit{},values,depth,equation)
    case Activations.HardSigmoid{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.HardSwish{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.GELUTanh{} Storage.Buffer{values,length,Certificate.Certificate{depth,equation}}:
      LanesCertificate.certified(~Unit,~Pipeline.activated_value(~Activations.gelu_tanh_value),U32.to_nat(count),0,Unit{},values,depth,equation)
    case Activations.Custom{operations} Storage.Buffer{values,length,certificate}: {==}

def activation_tree(~custom: F32 -> F32,activation: Activations.Activation,+count: U32,values: Tree.Tree<F32>) -> Tree.Tree<F32>:
  match activation:
    case Activations.Identity{}: values
    case Activations.ReLU{}: BandMap.refinement_values(~F32,~Unit,~Pipeline.activated_value(~Activations.relu_value),U32.to_nat(count),Unit{},values)
    case Activations.ReLU6{}: BandMap.refinement_values(~F32,~Unit,~Pipeline.activated_value(~Activations.relu6_value),U32.to_nat(count),Unit{},values)
    case Activations.LeakyReLU{slope}: BandMap.refinement_values(~F32,~F32,~Pipeline.leaky_relu_value,U32.to_nat(count),slope,values)
    case Activations.Sigmoid{}: BandMap.refinement_values(~F32,~Unit,~Pipeline.activated_value(~Activations.sigmoid_value),U32.to_nat(count),Unit{},values)
    case Activations.SiLU{}: BandMap.refinement_values(~F32,~Unit,~Pipeline.activated_value(~Activations.silu_value),U32.to_nat(count),Unit{},values)
    case Activations.Tanh{}: BandMap.refinement_values(~F32,~Unit,~Pipeline.activated_value(~Activations.tanh_value),U32.to_nat(count),Unit{},values)
    case Activations.HardSigmoid{}: BandMap.refinement_values(~F32,~Unit,~Pipeline.activated_value(~Activations.hard_sigmoid_value),U32.to_nat(count),Unit{},values)
    case Activations.HardSwish{}: BandMap.refinement_values(~F32,~Unit,~Pipeline.activated_value(~Activations.hard_swish_value),U32.to_nat(count),Unit{},values)
    case Activations.GELUTanh{}: BandMap.refinement_values(~F32,~Unit,~Pipeline.activated_value(~Activations.gelu_tanh_value),U32.to_nat(count),Unit{},values)
    case Activations.Custom{operations}: BandMap.refinement_values(~F32,~Unit,~Pipeline.activated_value(~custom),U32.to_nat(count),Unit{},values)

def activation_model(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,+values: Tree.Tree<F32>) ->
  {activation_array(~custom,activation,count,Tree.pack(F32,values)) == Tree.pack(F32,activation_tree(~custom,activation,count,values)) : Array<F32>}:
  match activation:
    case Activations.Identity{}: {==}
    case Activations.ReLU{}: BandMap.raw(~F32,~Unit,~Pipeline.activated_value(~Activations.relu_value),U32.to_nat(count),Unit{},values)
    case Activations.ReLU6{}: BandMap.raw(~F32,~Unit,~Pipeline.activated_value(~Activations.relu6_value),U32.to_nat(count),Unit{},values)
    case Activations.LeakyReLU{+slope}: BandMap.raw(~F32,~F32,~Pipeline.leaky_relu_value,U32.to_nat(count),slope,values)
    case Activations.Sigmoid{}: BandMap.raw(~F32,~Unit,~Pipeline.activated_value(~Activations.sigmoid_value),U32.to_nat(count),Unit{},values)
    case Activations.SiLU{}: BandMap.raw(~F32,~Unit,~Pipeline.activated_value(~Activations.silu_value),U32.to_nat(count),Unit{},values)
    case Activations.Tanh{}: BandMap.raw(~F32,~Unit,~Pipeline.activated_value(~Activations.tanh_value),U32.to_nat(count),Unit{},values)
    case Activations.HardSigmoid{}: BandMap.raw(~F32,~Unit,~Pipeline.activated_value(~Activations.hard_sigmoid_value),U32.to_nat(count),Unit{},values)
    case Activations.HardSwish{}: BandMap.raw(~F32,~Unit,~Pipeline.activated_value(~Activations.hard_swish_value),U32.to_nat(count),Unit{},values)
    case Activations.GELUTanh{}: BandMap.raw(~F32,~Unit,~Pipeline.activated_value(~Activations.gelu_tanh_value),U32.to_nat(count),Unit{},values)
    case Activations.Custom{operations}: BandMap.raw(~F32,~Unit,~Pipeline.activated_value(~custom),U32.to_nat(count),Unit{},values)

def activation_value(~custom: F32 -> F32,activation: Activations.Activation,value: F32) -> F32:
  match activation:
    case Activations.Identity{}: value
    case Activations.ReLU{}: Activations.relu_value(value)
    case Activations.ReLU6{}: Activations.relu6_value(value)
    case Activations.LeakyReLU{slope}: Pipeline.leaky_relu_value(slope,value)
    case Activations.Sigmoid{}: Activations.sigmoid_value(value)
    case Activations.SiLU{}: Activations.silu_value(value)
    case Activations.Tanh{}: Activations.tanh_value(value)
    case Activations.HardSigmoid{}: Activations.hard_sigmoid_value(value)
    case Activations.HardSwish{}: Activations.hard_swish_value(value)
    case Activations.GELUTanh{}: Activations.gelu_tanh_value(value)
    case Activations.Custom{operations}: custom(value)

def activated_cases(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,+values: Tree.Tree<F32>,+depth: Nat,
  +capacity_equal: {Tree.capacity(F32,values) == U32.shln(1,depth) : U32},+balanced: Shape.Balanced(F32,depth,values),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(U32.shln(1,depth),2147483648) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(count),U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
  +index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(F32,activation_tree(~custom,activation,count,values),U32.from_nat(index)) ==
    activation_value(~custom,activation,Tree.read(F32,values,U32.from_nat(index))) : F32}:
  match activation:
    case Activations.Identity{}: {==}
    case Activations.ReLU{}:
      MapReadback.inside(~F32,~Unit,~Pipeline.activated_value(~Activations.relu_value),U32.to_nat(count),0n,index,Unit{},values,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.ReLU6{}:
      MapReadback.inside(~F32,~Unit,~Pipeline.activated_value(~Activations.relu6_value),U32.to_nat(count),0n,index,Unit{},values,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.LeakyReLU{+slope}:
      MapReadback.inside(~F32,~F32,~Pipeline.leaky_relu_value,U32.to_nat(count),0n,index,slope,values,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.Sigmoid{}:
      MapReadback.inside(~F32,~Unit,~Pipeline.activated_value(~Activations.sigmoid_value),U32.to_nat(count),0n,index,Unit{},values,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.SiLU{}:
      MapReadback.inside(~F32,~Unit,~Pipeline.activated_value(~Activations.silu_value),U32.to_nat(count),0n,index,Unit{},values,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.Tanh{}:
      MapReadback.inside(~F32,~Unit,~Pipeline.activated_value(~Activations.tanh_value),U32.to_nat(count),0n,index,Unit{},values,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.HardSigmoid{}:
      MapReadback.inside(~F32,~Unit,~Pipeline.activated_value(~Activations.hard_sigmoid_value),U32.to_nat(count),0n,index,Unit{},values,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.HardSwish{}:
      MapReadback.inside(~F32,~Unit,~Pipeline.activated_value(~Activations.hard_swish_value),U32.to_nat(count),0n,index,Unit{},values,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.GELUTanh{}:
      MapReadback.inside(~F32,~Unit,~Pipeline.activated_value(~Activations.gelu_tanh_value),U32.to_nat(count),0n,index,Unit{},values,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.Custom{operations}:
      MapReadback.inside(~F32,~Unit,~Pipeline.activated_value(~custom),U32.to_nat(count),0n,index,Unit{},values,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)

def product_exact(+limits: Machine.Limits,+rows: U32,+width: U32,
  +room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat((rows * width : U32)) == Nat.mul(U32.to_nat(rows),U32.to_nat(width)) : Nat}:
  Arithmetic.multiply(rows,width,Machine.capacity(limits),Limits.capacity_fits(limits),room)

def product_ready(+limits: Machine.Limits,+rows: U32,+width: U32,
  +room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> Depth.Sized<(rows * width : U32)>:
  +product = (rows * width : U32)
  Depth.bounded(product,Arithmetic.count_within(product,Nat.mul(U32.to_nat(rows),U32.to_nat(width)),Machine.capacity(limits),1073741824,
    product_exact(limits,rows,width,room),room,Limits.capacity_allocatable(limits)))

def ready_depth(+count: U32,ready: Depth.Sized<count>) -> {Nat.is_le(Tensors.depth_for(count),30n) == True{} : Bool}:
  Depth.Sized{depth_ok,room} = ready
  depth_ok

def power_fits(+depth: Nat,+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool}) -> {U32.is_le(U32.shln(1,depth),2147483648) == True{} : Bool}:
  Power.native_power_bound(depth,Order.transitive(depth,30n,31n,depth_ok,{==}))
