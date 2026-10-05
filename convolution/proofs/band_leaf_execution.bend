# The convolution band leaf: its execution, numeric guards and readback.
import Base
import ../../lib/proofs/tensor_band_window.bend as TensorBandWindow
import ./gather_blocks.bend as Blocks
import ./microkernel_leaf_readback.bend as MicroLeaf
import ../../lib/kernel_gather.bend as Gather
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/kernel_tile.bend as Tile
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/kernel_gemm_refinement.bend as GemmLaws
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_band_copy.bend as BandCopy
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/tensor.bend as Tensors
import ./band_window_contents.bend as WindowContents
import ./gather_block_width.bend as Width
import ./microkernel_storage_execution.bend as Execution
import ./packing_storage_refinement.bend as PackingStorage
import ./prepared_packing_samples.bend as Prepared
import ../../lib/kernel_gemm_buffer.bend as GemmBuffer
import ../../lib/kernel_gather_buffer.bend as GatherBuffer
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/kernel_packing_buffer.bend as PackingBuffer
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/contiguous_copy.bend as Contiguous
import ../../lib/proofs/kernel_gemm_writes.bend as GemmWrites
import ../../lib/proofs/kernel_packing_certificate.bend as PackingCertificate
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_extrema.bend as Extrema
import ../../lib/proofs/storage_allocation_refinement.bend as AllocationLaws
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/tensor_band_map.bend as BandMap
import ../../lib/proofs/tensor_band_range.bend as BandRange
import ../../lib/proofs/traversal_certificate.bend as TraversalCertificate
import ../../lib/proofs/traversal_map_readback.bend as MapReadback
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/u32_extrema.bend as U32Extrema
import ../../lib/proofs/word_power_of_two.bend as Power
import ../../lib/storage_allocation.bend as Allocation
import ../../lib/storage_buffer.bend as Storage
import ../../lib/tensor_band_window.bend as BandWindow
import ../../lib/traversal_array.bend as Traversal

# The actual band leaf on certified Buffers computes the same arrays as the
# internal kernels on their storage: certificates never change contents.
# Tree models follow from the existing packing, GEMM, map and gather proofs.
def storage(~Element: Data,buffer: Storage.Buffer<Element>) -> Array<Element>:
  Storage.Buffer{storage,length,certificate} = buffer
  storage

def second(pair: Array<F32> & Array<Tile.Tile8>) -> Array<Tile.Tile8>:
  (input,output) = pair
  output

def allocated(~Element: Data,+fill: Element,+count: U32) ->
  {Tensors.buffer(~Element,fill,count) == Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)) : Storage.Buffer<Element>}:
  AllocationLaws.dispatch_same(Element,Allocation.allocated(~Element),Tensors.depth_for(count),fill,U32.to_nat(count))

# The activation pass over the first count values of an array.
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
    case Activations.Sigmoid{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.SiLU{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.Tanh{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.HardSigmoid{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.HardSwish{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.GELUTanh{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.Custom{operations} Storage.Buffer{values,length,certificate}: {==}

def gemm(+rows: U32,tiles: Storage.Buffer<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,output: Storage.Buffer<F32>,+columns: U32,+inner: U32) ->
  {storage(~F32,GemmBuffer.run(rows,tiles,weights,bias,output,columns,inner)) ==
    Gemm.run(rows,storage(~Tile.Tile8,tiles),storage(~Tile.Tile8,weights),storage(~F32,bias),storage(~F32,output),columns,inner) : Array<F32>}:
  match tiles weights bias output:
    case Storage.Buffer{a,al,ac} Storage.Buffer{b,bl,bc} Storage.Buffer{c,cl,cc} Storage.Buffer{d,dl,dc}: {==}

def fresh_output(+rows: U32,+span: U32) -> Array<F32>:
  Array.new(F32,Tensors.depth_for((rows * span : U32)),0.0)

def computed_array(~custom: F32 -> F32,activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,tiles: Array<Tile.Tile8>,weights: Array<Tile.Tile8>,bias: Array<F32>) -> Array<F32>:
  activation_array(~custom,activation,(rows * span : U32),Gemm.run(rows,tiles,weights,bias,fresh_output(rows,span),span,inner))

def computed_pair(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,w: Array<Tile.Tile8>,+wl: Nat,+wc: Certificate.Certificate<Tile.Tile8,w>,
  b: Array<F32>,+bl: Nat,+bc: Certificate.Certificate<F32,b>,+input_length: Nat,+output_length: Nat,result: Array<F32> & Array<Tile.Tile8>,
  certificates: TraversalCertificate.Certificates<Array<F32>,Tile.Tile8,Certificate.family(~F32),result>) ->
  {storage(~F32,Pipeline.computed(~custom,activation,rows,span,inner,Storage.Buffer{w,wl,wc},Storage.Buffer{b,bl,bc},PackingBuffer.buffers(input_length,output_length,result,certificates))) ==
    computed_array(~custom,activation,rows,span,inner,second(result),w,b) : Array<F32>}:
  match result certificates:
    case (input,output) TraversalCertificate.Certificates{source,target}:
      %Equal.sym(Storage.Buffer<F32>,Tensors.buffer(~F32,0.0,(rows * span : U32)),Storage.allocate(~F32,Tensors.depth_for((rows * span : U32)),0.0,U32.to_nat((rows * span : U32))),allocated(~F32,0.0,(rows * span : U32))) :
        {storage(~F32,Pipeline.activated(~custom,activation,(rows * span : U32),GemmBuffer.run(rows,Storage.Buffer{output,output_length,target},Storage.Buffer{w,wl,wc},Storage.Buffer{b,bl,bc},_,span,inner))) ==
          computed_array(~custom,activation,rows,span,inner,output,w,b) : Array<F32>}
      activated(~custom,activation,(rows * span : U32),GemmBuffer.run(rows,Storage.Buffer{output,output_length,target},Storage.Buffer{w,wl,wc},Storage.Buffer{b,bl,bc},
        Storage.allocate(~F32,Tensors.depth_for((rows * span : U32)),0.0,U32.to_nat((rows * span : U32))),span,inner))

def computed_result(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,
  +input_length: Nat,+output_length: Nat,result: Array<F32> & Array<Tile.Tile8>,
  certificates: TraversalCertificate.Certificates<Array<F32>,Tile.Tile8,Certificate.family(~F32),result>) ->
  {storage(~F32,Pipeline.computed(~custom,activation,rows,span,inner,weights,bias,PackingBuffer.buffers(input_length,output_length,result,certificates))) ==
    computed_array(~custom,activation,rows,span,inner,second(result),storage(~Tile.Tile8,weights),storage(~F32,bias)) : Array<F32>}:
  match weights bias:
    case Storage.Buffer{w,wl,wc} Storage.Buffer{b,bl,bc}:
      computed_pair(~custom,activation,rows,span,inner,w,wl,wc,b,bl,bc,input_length,output_length,result,certificates)

def packed_array(+call: Packing.Call,window: Array<F32>,+count: U32) -> Array<Tile.Tile8>:
  second(Packing.run(call,window,Array.new(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))))

def computed(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,+call: Packing.Call,+count: U32,
  window: Storage.Buffer<F32>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>) ->
  {storage(~F32,Pipeline.computed(~custom,activation,rows,span,inner,weights,bias,PackingBuffer.run(call,window,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count)))) ==
    computed_array(~custom,activation,rows,span,inner,packed_array(call,storage(~F32,window),count),storage(~Tile.Tile8,weights),storage(~F32,bias)) : Array<F32>}:
  match window:
    case Storage.Buffer{values,length,certificate}:
      +tile_depth = Tensors.depth_for(count)
      +tile_certificate = Certificate.allocated(Tile.Tile8,tile_depth,Tile.filled(0.0))
      %Equal.sym(Storage.Buffer<Tile.Tile8>,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count),Storage.allocate(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0),U32.to_nat(count)),allocated(~Tile.Tile8,Tile.filled(0.0),count)) :
        {storage(~F32,Pipeline.computed(~custom,activation,rows,span,inner,weights,bias,PackingBuffer.run(call,Storage.Buffer{values,length,certificate},_))) ==
          computed_array(~custom,activation,rows,span,inner,packed_array(call,values,count),storage(~Tile.Tile8,weights),storage(~F32,bias)) : Array<F32>}
      computed_result(~custom,activation,rows,span,inner,weights,bias,length,U32.to_nat(count),Packing.run(call,values,Array.new(Tile.Tile8,tile_depth,Tile.filled(0.0))),
        PackingCertificate.certificates(call,values,Array.new(Tile.Tile8,tile_depth,Tile.filled(0.0)),certificate,tile_certificate))

def finish_array(identity: Bool,values: Array<F32>,+rows: U32,+offset: U32,+span: U32,+positions: U32) -> Array<F32>:
  match identity:
    case True{}: values
    case False{}: Gather.gather_block(rows,offset,positions,span,values,Array.new(F32,Tensors.depth_for((rows * positions : U32)),0.0))

def execution_finished(+identity: Bool,input: Storage.Buffer<F32>,+rows: U32,+offset: U32,+span: U32,+positions: U32) ->
  {storage(~F32,Pipeline.finish_leaf(identity,input,rows,offset,span,positions)) == finish_array(identity,storage(~F32,input),rows,offset,span,positions) : Array<F32>}:
  match identity input:
    case True{} Storage.Buffer{values,length,certificate}: {==}
    case False{} Storage.Buffer{values,length,certificate}:
      %Equal.sym(Storage.Buffer<F32>,Tensors.buffer(~F32,0.0,(rows * positions : U32)),Storage.allocate(~F32,Tensors.depth_for((rows * positions : U32)),0.0,U32.to_nat((rows * positions : U32))),allocated(~F32,0.0,(rows * positions : U32))) :
        {storage(~F32,GatherBuffer.run(rows,offset,positions,span,Storage.Buffer{values,length,certificate},_)) ==
          finish_array(False{},values,rows,offset,span,positions) : Array<F32>}
      {==}

# Tree models of the same steps.
def packed_tree(+call: Packing.Call,window: Tree.Tree<F32>,+count: U32) -> Tree.Tree<Tile.Tile8>:
  Prepared.packed_output(Execution.packed(call,window,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))))

def second_packed(+packed: PackingStorage.Packed) -> {second(PackingStorage.pack_pair(packed)) == Tree.pack(Tile.Tile8,Prepared.packed_output(packed)) : Array<Tile.Tile8>}:
  PackingStorage.Packed{input,output} = packed
  {==}

def packed(+call: Packing.Call,+window: Tree.Tree<F32>,+count: U32) ->
  {packed_array(call,Tree.pack(F32,window),count) == Tree.pack(Tile.Tile8,packed_tree(call,window,count)) : Array<Tile.Tile8>}:
  +fresh = Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))
  %Equal.sym(Array<Tile.Tile8>,Array.new(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)),Tree.pack(Tile.Tile8,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))),StorageLaws.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))) : {second(Packing.run(call,Tree.pack(F32,window),_)) == Tree.pack(Tile.Tile8,packed_tree(call,window,count)) : Array<Tile.Tile8>}
  %Equal.sym(Array<F32> & Array<Tile.Tile8>,Packing.run(call,Tree.pack(F32,window),Tree.pack(Tile.Tile8,fresh)),PackingStorage.pack_pair(Execution.packed(call,window,fresh)),Execution.packed_matches(call,window,fresh)) : {second(_) == Tree.pack(Tile.Tile8,packed_tree(call,window,count)) : Array<Tile.Tile8>}
  second_packed(Execution.packed(call,window,fresh))

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

def gemm_tree(+rows: U32,+span: U32,+inner: U32,tiles: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>) -> Tree.Tree<F32>:
  GemmLaws.run(rows,tiles,weights,bias,MicroLeaf.allocated(rows,span),span,inner)

def computed_tree(~custom: F32 -> F32,activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,tiles: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>) -> Tree.Tree<F32>:
  activation_tree(~custom,activation,(rows * span : U32),gemm_tree(rows,span,inner,tiles,weights,bias))

def computed_model(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,+tiles: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  {computed_array(~custom,activation,rows,span,inner,Tree.pack(Tile.Tile8,tiles),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias)) ==
    Tree.pack(F32,computed_tree(~custom,activation,rows,span,inner,tiles,weights,bias)) : Array<F32>}:
  +output = MicroLeaf.allocated(rows,span)
  %Equal.sym(Array<F32>,Array.new(F32,Tensors.depth_for((rows * span : U32)),0.0),Tree.pack(F32,Tree.fresh(F32,Tensors.depth_for((rows * span : U32)),0.0)),StorageLaws.fresh(F32,Tensors.depth_for((rows * span : U32)),0.0)) :
    {activation_array(~custom,activation,(rows * span : U32),Gemm.run(rows,Tree.pack(Tile.Tile8,tiles),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),_,span,inner)) ==
      Tree.pack(F32,computed_tree(~custom,activation,rows,span,inner,tiles,weights,bias)) : Array<F32>}
  %Equal.sym(Array<F32>,Gemm.run(rows,Tree.pack(Tile.Tile8,tiles),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,output),span,inner),Tree.pack(F32,GemmLaws.run(rows,tiles,weights,bias,output,span,inner)),GemmLaws.run_matches(rows,tiles,weights,bias,output,span,inner)) :
    {activation_array(~custom,activation,(rows * span : U32),_) == Tree.pack(F32,computed_tree(~custom,activation,rows,span,inner,tiles,weights,bias)) : Array<F32>}
  activation_model(~custom,activation,(rows * span : U32),gemm_tree(rows,span,inner,tiles,weights,bias))

def finish_tree(identity: Bool,values: Tree.Tree<F32>,+rows: U32,+offset: U32,+span: U32,+positions: U32) -> Tree.Tree<F32>:
  match identity:
    case True{}: values
    case False{}: Writes.pure(F32,Blocks.block_writes(rows,offset,positions,span,values),Tree.fresh(F32,Tensors.depth_for((rows * positions : U32)),0.0))

def finish_model(+identity: Bool,+values: Tree.Tree<F32>,+rows: U32,+offset: U32,+span: U32,+positions: U32) ->
  {finish_array(identity,Tree.pack(F32,values),rows,offset,span,positions) == Tree.pack(F32,finish_tree(identity,values,rows,offset,span,positions)) : Array<F32>}:
  match identity:
    case True{}: {==}
    case False{}:
      +fresh = Tree.fresh(F32,Tensors.depth_for((rows * positions : U32)),0.0)
      %Equal.sym(Array<F32>,Array.new(F32,Tensors.depth_for((rows * positions : U32)),0.0),Tree.pack(F32,Tree.fresh(F32,Tensors.depth_for((rows * positions : U32)),0.0)),StorageLaws.fresh(F32,Tensors.depth_for((rows * positions : U32)),0.0)) :
        {Gather.gather_block(rows,offset,positions,span,Tree.pack(F32,values),_) == Tree.pack(F32,finish_tree(False{},values,rows,offset,span,positions)) : Array<F32>}
      Blocks.block(rows,offset,positions,span,values,fresh)

def leaf_tree(~custom: F32 -> F32,activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,+positions: U32,+call: Packing.Call,+count: U32,
  window: Tree.Tree<F32>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>) -> Tree.Tree<F32>:
  finish_tree(U32.is_eq(positions,span),computed_tree(~custom,activation,rows,span,inner,packed_tree(call,window,count),weights,bias),rows,0,span,positions)

# The actual band leaf computation on canonical window, weight and bias owners.
def leaf(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,+positions: U32,+call: Packing.Call,+count: U32,
  +window: Buffer.Model<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {storage(~F32,Pipeline.finish_leaf(U32.is_eq(positions,span),
      Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),
        PackingBuffer.run(call,Buffer.pack(~F32,window),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,0,span,positions)) ==
    Tree.pack(F32,leaf_tree(~custom,activation,rows,span,inner,positions,call,count,Buffer.tree(~F32,window),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias))) : Array<F32>}:
  Buffer.Model{+window_tree,+window_length,+window_certificate} = window
  Buffer.Model{+weight_tree,+weight_length,+weight_certificate} = weights
  Buffer.Model{+bias_tree,+bias_length,+bias_certificate} = bias
  +window_buffer = {Buffer.Model{window_tree,window_length,window_certificate} : Buffer.Model<F32>}
  +weight_buffer = {Buffer.Model{weight_tree,weight_length,weight_certificate} : Buffer.Model<Tile.Tile8>}
  +bias_buffer = {Buffer.Model{bias_tree,bias_length,bias_certificate} : Buffer.Model<F32>}
  +identity = U32.is_eq(positions,span)
  +tiles = packed_tree(call,window_tree,count)
  +computed_values = computed_tree(~custom,activation,rows,span,inner,tiles,weight_tree,bias_tree)
  +stored = {
    %Equal.sym(Array<Tile.Tile8>,packed_array(call,Tree.pack(F32,window_tree),count),Tree.pack(Tile.Tile8,tiles),packed(call,window_tree,count)) :
      {computed_array(~custom,activation,rows,span,inner,_,Tree.pack(Tile.Tile8,weight_tree),Tree.pack(F32,bias_tree)) == Tree.pack(F32,computed_values) : Array<F32>}
    computed_model(~custom,activation,rows,span,inner,tiles,weight_tree,bias_tree) :
    {computed_array(~custom,activation,rows,span,inner,packed_array(call,Tree.pack(F32,window_tree),count),Tree.pack(Tile.Tile8,weight_tree),Tree.pack(F32,bias_tree)) ==
      Tree.pack(F32,computed_values) : Array<F32>}}
  +computed_values_storage = Equal.trans(Array<F32>,
    storage(~F32,Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weight_buffer),Buffer.pack(~F32,bias_buffer),
      PackingBuffer.run(call,Buffer.pack(~F32,window_buffer),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count)))),
    computed_array(~custom,activation,rows,span,inner,packed_array(call,Tree.pack(F32,window_tree),count),Tree.pack(Tile.Tile8,weight_tree),Tree.pack(F32,bias_tree)),
    Tree.pack(F32,computed_values),
    computed(~custom,activation,rows,span,inner,call,count,Buffer.pack(~F32,window_buffer),Buffer.pack(~Tile.Tile8,weight_buffer),Buffer.pack(~F32,bias_buffer)),stored)
  Equal.trans(Array<F32>,
    storage(~F32,Pipeline.finish_leaf(identity,Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weight_buffer),Buffer.pack(~F32,bias_buffer),
      PackingBuffer.run(call,Buffer.pack(~F32,window_buffer),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,0,span,positions)),
    finish_array(identity,Tree.pack(F32,computed_values),rows,0,span,positions),Tree.pack(F32,finish_tree(identity,computed_values,rows,0,span,positions)),
    %computed_values_storage :
      {storage(~F32,Pipeline.finish_leaf(identity,Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weight_buffer),Buffer.pack(~F32,bias_buffer),
        PackingBuffer.run(call,Buffer.pack(~F32,window_buffer),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,0,span,positions)) ==
        finish_array(identity,_,rows,0,span,positions) : Array<F32>}
    execution_finished(identity,Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weight_buffer),Buffer.pack(~F32,bias_buffer),
      PackingBuffer.run(call,Buffer.pack(~F32,window_buffer),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,0,span,positions),
    finish_model(identity,computed_values,rows,0,span,positions))

# Buffer lengths through the leaf: every pass keeps its output owner's length.
def buffer_length(~Element: Data,buffer: Storage.Buffer<Element>) -> Nat:
  Storage.Buffer{storage,length,certificate} = buffer
  length

def contents_parts(~Element: Data,buffer: Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,buffer) == Buffer.Content{Tree.reflect(Element,storage(~Element,buffer)),buffer_length(~Element,buffer)} : Buffer.Content<Element>}:
  match buffer:
    case Storage.Buffer{values,length,certificate}: {==}

def activated_length(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,buffer: Storage.Buffer<F32>) ->
  {buffer_length(~F32,Pipeline.activated(~custom,activation,count,buffer)) == buffer_length(~F32,buffer) : Nat}:
  match activation buffer:
    case Activations.Identity{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.ReLU{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.ReLU6{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.LeakyReLU{slope} Storage.Buffer{values,length,certificate}: {==}
    case Activations.Sigmoid{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.SiLU{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.Tanh{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.HardSigmoid{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.HardSwish{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.GELUTanh{} Storage.Buffer{values,length,certificate}: {==}
    case Activations.Custom{operations} Storage.Buffer{values,length,certificate}: {==}

def computed_pair_length(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,w: Array<Tile.Tile8>,+wl: Nat,+wc: Certificate.Certificate<Tile.Tile8,w>,
  b: Array<F32>,+bl: Nat,+bc: Certificate.Certificate<F32,b>,+input_length: Nat,+output_length: Nat,result: Array<F32> & Array<Tile.Tile8>,
  certificates: TraversalCertificate.Certificates<Array<F32>,Tile.Tile8,Certificate.family(~F32),result>) ->
  {buffer_length(~F32,Pipeline.computed(~custom,activation,rows,span,inner,Storage.Buffer{w,wl,wc},Storage.Buffer{b,bl,bc},PackingBuffer.buffers(input_length,output_length,result,certificates))) ==
    U32.to_nat((rows * span : U32)) : Nat}:
  match result certificates:
    case (input,output) TraversalCertificate.Certificates{source,target}:
      %Equal.sym(Storage.Buffer<F32>,Tensors.buffer(~F32,0.0,(rows * span : U32)),Storage.allocate(~F32,Tensors.depth_for((rows * span : U32)),0.0,U32.to_nat((rows * span : U32))),allocated(~F32,0.0,(rows * span : U32))) :
        {buffer_length(~F32,Pipeline.activated(~custom,activation,(rows * span : U32),GemmBuffer.run(rows,Storage.Buffer{output,output_length,target},Storage.Buffer{w,wl,wc},Storage.Buffer{b,bl,bc},_,span,inner))) ==
          U32.to_nat((rows * span : U32)) : Nat}
      activated_length(~custom,activation,(rows * span : U32),GemmBuffer.run(rows,Storage.Buffer{output,output_length,target},Storage.Buffer{w,wl,wc},Storage.Buffer{b,bl,bc},
        Storage.allocate(~F32,Tensors.depth_for((rows * span : U32)),0.0,U32.to_nat((rows * span : U32))),span,inner))

def computed_length(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,+call: Packing.Call,+count: U32,
  window: Storage.Buffer<F32>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>) ->
  {buffer_length(~F32,Pipeline.computed(~custom,activation,rows,span,inner,weights,bias,PackingBuffer.run(call,window,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count)))) ==
    U32.to_nat((rows * span : U32)) : Nat}:
  match window weights bias:
    case Storage.Buffer{values,length,certificate} Storage.Buffer{w,wl,wc} Storage.Buffer{b,bl,bc}:
      +tile_depth = Tensors.depth_for(count)
      %Equal.sym(Storage.Buffer<Tile.Tile8>,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count),Storage.allocate(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0),U32.to_nat(count)),allocated(~Tile.Tile8,Tile.filled(0.0),count)) :
        {buffer_length(~F32,Pipeline.computed(~custom,activation,rows,span,inner,Storage.Buffer{w,wl,wc},Storage.Buffer{b,bl,bc},PackingBuffer.run(call,Storage.Buffer{values,length,certificate},_))) ==
          U32.to_nat((rows * span : U32)) : Nat}
      computed_pair_length(~custom,activation,rows,span,inner,w,wl,wc,b,bl,bc,length,U32.to_nat(count),Packing.run(call,values,Array.new(Tile.Tile8,tile_depth,Tile.filled(0.0))),
        PackingCertificate.certificates(call,values,Array.new(Tile.Tile8,tile_depth,Tile.filled(0.0)),certificate,Certificate.allocated(Tile.Tile8,tile_depth,Tile.filled(0.0))))

def finished_length(identity: Bool,input: Storage.Buffer<F32>,+rows: U32,+span: U32,+positions: U32,
  +guard: {U32.is_eq(positions,span) == identity : Bool},+input_length: {buffer_length(~F32,input) == U32.to_nat((rows * span : U32)) : Nat}) ->
  {buffer_length(~F32,Pipeline.finish_leaf(identity,input,rows,0,span,positions)) == U32.to_nat((rows * positions : U32)) : Nat}:
  match identity input:
    case True{} input:
      %Equal.sym(U32,positions,span,BandRange.same_word(positions,span,guard)) : {buffer_length(~F32,input) == U32.to_nat((rows * _ : U32)) : Nat}
      input_length
    case False{} Storage.Buffer{values,length,certificate}:
      %Equal.sym(Storage.Buffer<F32>,Tensors.buffer(~F32,0.0,(rows * positions : U32)),Storage.allocate(~F32,Tensors.depth_for((rows * positions : U32)),0.0,U32.to_nat((rows * positions : U32))),allocated(~F32,0.0,(rows * positions : U32))) :
        {buffer_length(~F32,GatherBuffer.run(rows,0,positions,span,Storage.Buffer{values,length,certificate},_)) == U32.to_nat((rows * positions : U32)) : Nat}
      {==}

# The actual band leaf as tree contents: its values and its logical length.
def leaf_contents(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,+positions: U32,+call: Packing.Call,+count: U32,
  +window: Buffer.Model<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Pipeline.finish_leaf(U32.is_eq(positions,span),
      Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),
        PackingBuffer.run(call,Buffer.pack(~F32,window),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,0,span,positions)) ==
    Buffer.Content{leaf_tree(~custom,activation,rows,span,inner,positions,call,count,Buffer.tree(~F32,window),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias)),
      U32.to_nat((rows * positions : U32))} : Buffer.Content<F32>}:
  +values = leaf_tree(~custom,activation,rows,span,inner,positions,call,count,Buffer.tree(~F32,window),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias))
  +stored = leaf(~custom,activation,rows,span,inner,positions,call,count,window,weights,bias)
  +length = finished_length(U32.is_eq(positions,span),
    Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),
      PackingBuffer.run(call,Buffer.pack(~F32,window),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,span,positions,{==},
    computed_length(~custom,activation,rows,span,inner,call,count,Buffer.pack(~F32,window),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)))
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Pipeline.finish_leaf(U32.is_eq(positions,span),Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),  PackingBuffer.run(call,Buffer.pack(~F32,window),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,0,span,positions)),Buffer.Content{Tree.reflect(F32,storage(~F32,Pipeline.finish_leaf(U32.is_eq(positions,span),Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),  PackingBuffer.run(call,Buffer.pack(~F32,window),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,0,span,positions))),buffer_length(~F32,Pipeline.finish_leaf(U32.is_eq(positions,span),Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),  PackingBuffer.run(call,Buffer.pack(~F32,window),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,0,span,positions))},contents_parts(~F32,Pipeline.finish_leaf(U32.is_eq(positions,span),Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),  PackingBuffer.run(call,Buffer.pack(~F32,window),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,0,span,positions))) : {_ == Buffer.Content{values,U32.to_nat((rows * positions : U32))} : Buffer.Content<F32>}
  %Equal.sym(Array<F32>,storage(~F32,Pipeline.finish_leaf(U32.is_eq(positions,span),Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),  PackingBuffer.run(call,Buffer.pack(~F32,window),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,0,span,positions)),Tree.pack(F32,values),stored) : {Buffer.Content{Tree.reflect(F32,_),buffer_length(~F32,Pipeline.finish_leaf(U32.is_eq(positions,span),Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),  PackingBuffer.run(call,Buffer.pack(~F32,window),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,0,span,positions))} == Buffer.Content{values,U32.to_nat((rows * positions : U32))} : Buffer.Content<F32>}
  %Equal.sym(Tree.Tree<F32>,Tree.reflect(F32,Tree.pack(F32,values)),values,StorageLaws.reflect_packed(F32,values)) : {Buffer.Content{_,buffer_length(~F32,Pipeline.finish_leaf(U32.is_eq(positions,span),Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),  PackingBuffer.run(call,Buffer.pack(~F32,window),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,0,span,positions))} == Buffer.Content{values,U32.to_nat((rows * positions : U32))} : Buffer.Content<F32>}
  %Equal.sym(Nat,buffer_length(~F32,Pipeline.finish_leaf(U32.is_eq(positions,span),Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),  PackingBuffer.run(call,Buffer.pack(~F32,window),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,0,span,positions)),U32.to_nat((rows * positions : U32)),length) : {Buffer.Content{values,_} == Buffer.Content{values,U32.to_nat((rows * positions : U32))} : Buffer.Content<F32>}
  {==}

# The same contents for any window owner, read through its observed model.
def leaf_contents_model(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,+positions: U32,+call: Packing.Call,+count: U32,-window: Storage.Buffer<F32>,+model: Buffer.Model<F32>,
  +equation: {Buffer.pack(~F32,model) == window : Storage.Buffer<F32>},+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Pipeline.finish_leaf(U32.is_eq(positions,span),
      Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),
        PackingBuffer.run(call,window,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,0,span,positions)) ==
    Buffer.Content{leaf_tree(~custom,activation,rows,span,inner,positions,call,count,Tree.reflect(F32,storage(~F32,window)),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias)),
      U32.to_nat((rows * positions : U32))} : Buffer.Content<F32>}:
  Buffer.Model{+tree,+length,+certificate} = model
  %equation : {Buffer.contents(~F32,Pipeline.finish_leaf(U32.is_eq(positions,span),
      Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),
        PackingBuffer.run(call,_,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,0,span,positions)) ==
    Buffer.Content{leaf_tree(~custom,activation,rows,span,inner,positions,call,count,Tree.reflect(F32,storage(~F32,_)),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias)),
      U32.to_nat((rows * positions : U32))} : Buffer.Content<F32>}
  %Equal.sym(Tree.Tree<F32>,Tree.reflect(F32,Tree.pack(F32,tree)),tree,StorageLaws.reflect_packed(F32,tree)) : {Buffer.contents(~F32,Pipeline.finish_leaf(U32.is_eq(positions,span),
      Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),
        PackingBuffer.run(call,Buffer.pack(~F32,Buffer.Model{tree,length,certificate}),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,0,span,positions)) ==
    Buffer.Content{leaf_tree(~custom,activation,rows,span,inner,positions,call,count,_,Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias)),
      U32.to_nat((rows * positions : U32))} : Buffer.Content<F32>}
  leaf_contents(~custom,activation,rows,span,inner,positions,call,count,Buffer.Model{tree,length,certificate},weights,bias)

def leaf_contents_observed(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,+positions: U32,+call: Packing.Call,+count: U32,-window: Storage.Buffer<F32>,observation: Buffer.Observation<F32,window>,
  +weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Pipeline.finish_leaf(U32.is_eq(positions,span),
      Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),
        PackingBuffer.run(call,window,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,0,span,positions)) ==
    Buffer.Content{leaf_tree(~custom,activation,rows,span,inner,positions,call,count,Tree.reflect(F32,storage(~F32,window)),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias)),
      U32.to_nat((rows * positions : U32))} : Buffer.Content<F32>}:
  Buffer.Observation{+model,+equation} = observation
  leaf_contents_model(~custom,activation,rows,span,inner,positions,call,count,window,model,equation,weights,bias)

def leaf_contents_buffer(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,+positions: U32,+call: Packing.Call,+count: U32,window: Storage.Buffer<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Pipeline.finish_leaf(U32.is_eq(positions,span),
      Pipeline.computed(~custom,activation,rows,span,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),
        PackingBuffer.run(call,window,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),rows,0,span,positions)) ==
    Buffer.Content{leaf_tree(~custom,activation,rows,span,inner,positions,call,count,Tree.reflect(F32,storage(~F32,window)),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias)),
      U32.to_nat((rows * positions : U32))} : Buffer.Content<F32>}:
  leaf_contents_observed(~custom,activation,rows,span,inner,positions,call,count,window,Buffer.observe(~F32,window),weights,bias)

# Pointwise readback of the band leaf's finish and activation passes. The
# padded GEMM tile keeps every channel's first positions values in order, and
# activation applies its scalar function at every computed position.
# The value one output position holds after the leaf's activation pass.
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

# A leaf-sized product fits the machine and has an allocatable fresh tree.
def product_exact(+limits: Machine.Limits,+rows: U32,+width: U32,
  +room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat((rows * width : U32)) == Nat.mul(U32.to_nat(rows),U32.to_nat(width)) : Nat}:
  Arithmetic.multiply(rows,width,Machine.capacity(limits),Limits.capacity_fits(limits),room)

def product_ready(+limits: Machine.Limits,+rows: U32,+width: U32,
  +room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> Depth.Ready<(rows * width : U32)>:
  +product = (rows * width : U32)
  Depth.bounded(product,Arithmetic.count_within(product,Nat.mul(U32.to_nat(rows),U32.to_nat(width)),Machine.capacity(limits),1073741824,
    product_exact(limits,rows,width,room),room,Limits.capacity_allocatable(limits)))

def ready_depth(+count: U32,ready: Depth.Ready<count>) -> {Nat.is_le(Tensors.depth_for(count),30n) == True{} : Bool}:
  Depth.Ready{depth_ok,room} = ready
  depth_ok

def power_fits(+depth: Nat,+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool}) -> {U32.is_le(U32.shln(1,depth),2147483648) == True{} : Bool}:
  Power.native_power_bound(depth,Order.transitive(depth,30n,31n,depth_ok,{==}))

# Gather at offset zero is the shared per-channel plane copy.
def gather_writes(count: Nat,+channel: U32,+width: U32,+spatial: U32,+span: U32,+input: Tree.Tree<F32>) ->
  {Blocks.writes(count,channel,width,0,spatial,span,input) == BandCopy.writes(~F32,count,channel,width,span,spatial,0,0,input) : List<&2,Writes.Assignment<F32>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %gather_writes(rest,(channel + 1 : U32),width,spatial,span,input) :
        {List.append(&2,Writes.Assignment<F32>,Contiguous.writes(~F32,U32.to_nat(width),input,0,(channel * span : U32),(channel * spatial + 0 : U32)),
            Blocks.writes(rest,(channel + 1 : U32),width,0,spatial,span,input)) ==
          List.append(&2,Writes.Assignment<F32>,Contiguous.writes(~F32,U32.to_nat(width),input,0,(0 + channel * span : U32),(0 + channel * spatial : U32)),_) :
          List<&2,Writes.Assignment<F32>>}
      %Equal.sym(U32,(0 + channel * span : U32),(channel * span : U32),Arithmetic.add_zero_left((channel * span : U32))) :
        {List.append(&2,Writes.Assignment<F32>,Contiguous.writes(~F32,U32.to_nat(width),input,0,(channel * span : U32),(channel * spatial + 0 : U32)),
            Blocks.writes(rest,(channel + 1 : U32),width,0,spatial,span,input)) ==
          List.append(&2,Writes.Assignment<F32>,Contiguous.writes(~F32,U32.to_nat(width),input,0,_,(0 + channel * spatial : U32)),
            Blocks.writes(rest,(channel + 1 : U32),width,0,spatial,span,input)) : List<&2,Writes.Assignment<F32>>}
      %Equal.sym(U32,(0 + channel * spatial : U32),(channel * spatial : U32),Arithmetic.add_zero_left((channel * spatial : U32))) :
        {List.append(&2,Writes.Assignment<F32>,Contiguous.writes(~F32,U32.to_nat(width),input,0,(channel * span : U32),(channel * spatial + 0 : U32)),
            Blocks.writes(rest,(channel + 1 : U32),width,0,spatial,span,input)) ==
          List.append(&2,Writes.Assignment<F32>,Contiguous.writes(~F32,U32.to_nat(width),input,0,(channel * span : U32),_),
            Blocks.writes(rest,(channel + 1 : U32),width,0,spatial,span,input)) : List<&2,Writes.Assignment<F32>>}
      %Equal.sym(U32,(channel * spatial + 0 : U32),(channel * spatial : U32),Arithmetic.add_zero((channel * spatial : U32))) :
        {List.append(&2,Writes.Assignment<F32>,Contiguous.writes(~F32,U32.to_nat(width),input,0,(channel * span : U32),_),
            Blocks.writes(rest,(channel + 1 : U32),width,0,spatial,span,input)) ==
          List.append(&2,Writes.Assignment<F32>,Contiguous.writes(~F32,U32.to_nat(width),input,0,(channel * span : U32),(channel * spatial : U32)),
            Blocks.writes(rest,(channel + 1 : U32),width,0,spatial,span,input)) : List<&2,Writes.Assignment<F32>>}
      {==}

# Gather copies at most the real positions of each channel.
def width_bound(valid: Bool,+positions: U32,+span: U32) ->
  {Nat.is_le(U32.to_nat(Gather.gather_width_when(valid,0,positions,span)),U32.to_nat(positions)) == True{} : Bool}:
  match valid:
    case False{}: Order.zero_le(U32.to_nat(positions))
    case True{}:
      %Equal.sym(Nat,U32.to_nat(U32.min(span,(positions - 0 : U32))),Nat.min(U32.to_nat(span),U32.to_nat((positions - 0 : U32))),U32Extrema.minimum(span,(positions - 0 : U32))) :
        {Nat.is_le(_,U32.to_nat(positions)) == True{} : Bool}
      %Equal.sym(U32,(positions - 0 : U32),positions,Arithmetic.subtract_zero(positions)) :
        {Nat.is_le(Nat.min(U32.to_nat(span),U32.to_nat(_)),U32.to_nat(positions)) == True{} : Bool}
      Extrema.minimum_right(U32.to_nat(span),U32.to_nat(positions))

def native_query(+region: BandCopy.Region,+channel: Nat,+index: Nat,+valid: BandCopy.Valid<region>,
  +channel_inside: {Nat.is_lt(channel,BandCopy.channels(region)) == True{} : Bool},+index_inside: {Nat.is_lt(index,U32.to_nat(BandCopy.count(region))) == True{} : Bool}) ->
  {BandCopy.readback_query(region,channel,index) == U32.from_nat(Nat.add(BandCopy.start(region,channel),index)) : U32}:
  +query = BandCopy.readback_query(region,channel,index)
  Equal.trans(U32,query,U32.from_nat(U32.to_nat(query)),U32.from_nat(Nat.add(BandCopy.start(region,channel),index)),
    Equal.sym(U32,U32.from_nat(U32.to_nat(query)),query,Arithmetic.native_roundtrip(query)),
    Equal.cong(Nat,U32,value => U32.from_nat(value),U32.to_nat(query),Nat.add(BandCopy.start(region,channel),index),
      BandCopy.readback_query_exact(region,channel,index,valid,channel_inside,index_inside)))

def gathered(+limits: Machine.Limits,+rows: U32,+span: U32,+positions: U32,+values: Tree.Tree<F32>,
  +ordered: {Nat.is_le(U32.to_nat(positions),U32.to_nat(span)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +channel: Nat,+column: Nat,+channel_inside: {Nat.is_lt(channel,U32.to_nat(rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(positions)) == True{} : Bool}) ->
  {Tree.read(F32,finish_tree(False{},values,rows,0,span,positions),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),column))) ==
    Tree.read(F32,values,U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(span)),column))) : F32}:
  +rows_nat = U32.to_nat(rows)
  +positions_nat = U32.to_nat(positions)
  +span_nat = U32.to_nat(span)
  +capacity = Machine.capacity(limits)
  +product = (rows * positions : U32)
  +depth = Tensors.depth_for(product)
  +fresh = Tree.fresh(F32,depth,0.0)
  +width = Gather.gather_width(0,positions,span)
  +positions_positive = Order.le_lt_trans(0n,column,positions_nat,Order.zero_le(column),column_inside)
  +rows_positive = Order.le_lt_trans(0n,channel,rows_nat,Order.zero_le(channel),channel_inside)
  +span_positive = Bounds.nat_trans(0n,positions_nat,span_nat,positions_positive,ordered)
  +product_le = {
    %RowMajor.multiply_commutative(positions_nat,rows_nat) : {Nat.is_le(_,Nat.mul(rows_nat,span_nat)) == True{} : Bool}
    %RowMajor.multiply_commutative(span_nat,rows_nat) : {Nat.is_le(Nat.mul(positions_nat,rows_nat),_) == True{} : Bool}
    RowMajor.mul_le(positions_nat,span_nat,rows_nat,ordered) : {Nat.is_le(Nat.mul(rows_nat,positions_nat),Nat.mul(rows_nat,span_nat)) == True{} : Bool}}
  +product_room = Order.transitive(Nat.mul(rows_nat,positions_nat),Nat.mul(rows_nat,span_nat),U32.to_nat(capacity),product_le,room)
  +ready = product_ready(limits,rows,positions,product_room)
  +depth_ok = ready_depth(product,ready)
  +output_room = {%product_exact(limits,rows,positions,product_room) : {Nat.is_le(_,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}
    Depth.allocation_room(product,ready) : {Nat.is_le(Nat.mul(rows_nat,positions_nat),U32.to_nat(U32.shln(1,depth))) == True{} : Bool}}
  +positions_room = {
    %RowMajor.multiply_commutative(positions_nat,rows_nat) : {Nat.is_le(positions_nat,_) == True{} : Bool}
    RowMajor.product_covers(positions_nat,rows_nat,rows_positive) : {Nat.is_le(positions_nat,Nat.mul(rows_nat,positions_nat)) == True{} : Bool}}
  +positions_capacity = {
    %Equal.sym(Bool,U32.is_le(positions,capacity),Nat.is_le(positions_nat,U32.to_nat(capacity)),Comparison.le(positions,capacity)) : {_ == True{} : Bool}
    Order.transitive(positions_nat,Nat.mul(rows_nat,positions_nat),U32.to_nat(capacity),positions_room,product_room) : {U32.is_le(positions,capacity) == True{} : Bool}}
  +width_le = width_bound(U32.is_lt(0,positions),positions,span)
  +index_inside = Width.covers(0,positions,span,column,capacity,Limits.capacity_fits(limits),positions_capacity,
    Bounds.nat_trans(column,positions_nat,span_nat,column_inside,ordered),column_inside)
  +target = {BandCopy.Region{0n,0n,rows_nat,width,positions,0,U32.shln(1,depth)} : BandCopy.Region}
  +source = {BandCopy.Region{0n,0n,rows_nat,width,span,0,capacity} : BandCopy.Region}
  +target_valid = {BandCopy.Valid{power_fits(depth,depth_ok),
      Order.transitive(rows_nat,Nat.mul(rows_nat,positions_nat),U32.to_nat(U32.shln(1,depth)),RowMajor.product_covers(rows_nat,positions_nat,positions_positive),output_room),
      {==},width_le,output_room} : BandCopy.Valid<target>}
  +source_valid = {BandCopy.Valid{Limits.capacity_fits(limits),
      Order.transitive(rows_nat,Nat.mul(rows_nat,span_nat),U32.to_nat(capacity),RowMajor.product_covers(rows_nat,span_nat,span_positive),room),
      {==},Order.transitive(U32.to_nat(width),positions_nat,span_nat,width_le,ordered),room} : BandCopy.Valid<source>}
  +copied = BandCopy.read_inside(~F32,rows_nat,0n,channel,column,target,span,0,values,fresh,depth,Shape.fresh(F32,depth,0.0),depth_ok,
    Shape.capacity(F32,depth,fresh,Shape.fresh(F32,depth,0.0)),target_valid,Order.reflexive(rows_nat),channel_inside,index_inside)
  %Equal.sym(List<&2,Writes.Assignment<F32>>,Blocks.writes(rows_nat,0,width,0,positions,span,values),BandCopy.writes(~F32,rows_nat,0,width,span,positions,0,0,values),
    gather_writes(rows_nat,0,width,positions,span,values)) :
    {Tree.read(F32,Writes.pure(F32,_,fresh),U32.from_nat(Nat.add(Nat.mul(channel,positions_nat),column))) ==
      Tree.read(F32,values,U32.from_nat(Nat.add(Nat.mul(channel,span_nat),column))) : F32}
  %native_query(target,channel,column,target_valid,channel_inside,index_inside) :
    {Tree.read(F32,Writes.pure(F32,BandCopy.writes(~F32,rows_nat,0,width,span,positions,0,0,values),fresh),_) ==
      Tree.read(F32,values,U32.from_nat(Nat.add(Nat.mul(channel,span_nat),column))) : F32}
  %native_query(source,channel,column,source_valid,channel_inside,index_inside) :
    {Tree.read(F32,Writes.pure(F32,BandCopy.writes(~F32,rows_nat,0,width,span,positions,0,0,values),fresh),BandCopy.readback_query(target,channel,column)) ==
      Tree.read(F32,values,_) : F32}
  copied

def finished_when(identity: Bool,+limits: Machine.Limits,+rows: U32,+span: U32,+positions: U32,+values: Tree.Tree<F32>,
  +guard: {U32.is_eq(positions,span) == identity : Bool},
  +ordered: {Nat.is_le(U32.to_nat(positions),U32.to_nat(span)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +channel: Nat,+column: Nat,+channel_inside: {Nat.is_lt(channel,U32.to_nat(rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(positions)) == True{} : Bool}) ->
  {Tree.read(F32,finish_tree(identity,values,rows,0,span,positions),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),column))) ==
    Tree.read(F32,values,U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(span)),column))) : F32}:
  match identity:
    case True{}:
      +same = Order.equal_of_is_eq(U32.to_nat(positions),U32.to_nat(span),
        Equal.trans(Bool,Nat.is_eq(U32.to_nat(positions),U32.to_nat(span)),U32.is_eq(positions,span),True{},
          Equal.sym(Bool,U32.is_eq(positions,span),Nat.is_eq(U32.to_nat(positions),U32.to_nat(span)),Comparison.eq(positions,span)),guard))
      %same : {Tree.read(F32,values,U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),column))) ==
        Tree.read(F32,values,U32.from_nat(Nat.add(Nat.mul(channel,_),column))) : F32}
      {==}
    case False{}: gathered(limits,rows,span,positions,values,ordered,room,channel,column,channel_inside,column_inside)

# The leaf's finish pass at local output position (channel, column).
def readback_finished(+limits: Machine.Limits,+rows: U32,+span: U32,+positions: U32,+values: Tree.Tree<F32>,
  +ordered: {Nat.is_le(U32.to_nat(positions),U32.to_nat(span)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +channel: Nat,+column: Nat,+channel_inside: {Nat.is_lt(channel,U32.to_nat(rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(positions)) == True{} : Bool}) ->
  {Tree.read(F32,finish_tree(U32.is_eq(positions,span),values,rows,0,span,positions),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),column))) ==
    Tree.read(F32,values,U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(span)),column))) : F32}:
  finished_when(U32.is_eq(positions,span),limits,rows,span,positions,values,{==},ordered,room,channel,column,channel_inside,column_inside)

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

# The GEMM output tree keeps the fresh allocation's shape.
def gemm_balanced(+rows: U32,+span: U32,+inner: U32,+tiles: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  Shape.Balanced(F32,Tensors.depth_for((rows * span : U32)),gemm_tree(rows,span,inner,tiles,weights,bias)):
  +depth = Tensors.depth_for((rows * span : U32))
  +writes = GemmWrites.all_writes(GemmWrites.job_count(rows,span),0,Gemm.Layout{rows,span,inner},tiles,weights,bias)
  %Equal.sym(Tree.Tree<F32>,GemmLaws.run(rows,tiles,weights,bias,MicroLeaf.allocated(rows,span),span,inner),Writes.pure(F32,writes,MicroLeaf.allocated(rows,span)),
    GemmWrites.run_writes(rows,tiles,weights,bias,MicroLeaf.allocated(rows,span),span,inner)) : Shape.Balanced(F32,depth,_)
  Writes.writes_preserve_balance(F32,writes,MicroLeaf.allocated(rows,span),depth,Shape.fresh(F32,depth,0.0))

# The whole band leaf at local output position (channel, column) is the
# activated GEMM value at that channel's padded position.
def leaf_read(~custom: F32 -> F32,+limits: Machine.Limits,+activation: Activations.Activation,+rows: U32,+span: U32,+inner: U32,+positions: U32,
  +tiles: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,
  +ordered: {Nat.is_le(U32.to_nat(positions),U32.to_nat(span)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +channel: Nat,+column: Nat,+channel_inside: {Nat.is_lt(channel,U32.to_nat(rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(positions)) == True{} : Bool}) ->
  {Tree.read(F32,finish_tree(U32.is_eq(positions,span),computed_tree(~custom,activation,rows,span,inner,tiles,weights,bias),rows,0,span,positions),
      U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),column))) ==
    activation_value(~custom,activation,Tree.read(F32,gemm_tree(rows,span,inner,tiles,weights,bias),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(span)),column)))) : F32}:
  +count = (rows * span : U32)
  +depth = Tensors.depth_for(count)
  +values = gemm_tree(rows,span,inner,tiles,weights,bias)
  +ready = product_ready(limits,rows,span,room)
  +depth_ok = ready_depth(count,ready)
  +balanced = gemm_balanced(rows,span,inner,tiles,weights,bias)
  +index = Nat.add(Nat.mul(channel,U32.to_nat(span)),column)
  +inside = {%Equal.sym(Nat,U32.to_nat(count),Nat.mul(U32.to_nat(rows),U32.to_nat(span)),product_exact(limits,rows,span,room)) : {Nat.is_lt(index,_) == True{} : Bool}
    RowMajor.radix(channel,U32.to_nat(rows),column,U32.to_nat(span),channel_inside,Bounds.nat_trans(column,U32.to_nat(positions),U32.to_nat(span),column_inside,ordered)) :
    {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}}
  Equal.trans(F32,
    Tree.read(F32,finish_tree(U32.is_eq(positions,span),computed_tree(~custom,activation,rows,span,inner,tiles,weights,bias),rows,0,span,positions),
      U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),column))),
    Tree.read(F32,computed_tree(~custom,activation,rows,span,inner,tiles,weights,bias),U32.from_nat(index)),
    activation_value(~custom,activation,Tree.read(F32,values,U32.from_nat(index))),
    readback_finished(limits,rows,span,positions,computed_tree(~custom,activation,rows,span,inner,tiles,weights,bias),ordered,room,channel,column,channel_inside,column_inside),
    activated_cases(~custom,activation,count,values,depth,Shape.capacity(F32,depth,values,balanced),balanced,depth_ok,power_fits(depth,depth_ok),
      Depth.allocation_room(count,ready),index,inside))

# Natural values of the native window arithmetic a band leaf computes: the
# window's first row and row count, its output offset and positions.
def sum_exact(+base: U32,+local: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(local)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((base + local : U32)) == Nat.add(U32.to_nat(base),U32.to_nat(local)) : Nat}:
  Arithmetic.add(base,local,capacity,capacity_ok,room)

# A local row placed after an image base and read back relative to it.
def prefix_removed(+base: U32,+local: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(local)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(((base + local : U32) - base : U32)) == U32.to_nat(local) : Nat}:
  +sum = (base + local : U32)
  +exact = sum_exact(base,local,capacity,capacity_ok,room)
  Arithmetic.remove_prefix(sum,base,U32.to_nat(local),capacity,capacity_ok,exact,
    %Equal.sym(Nat,U32.to_nat(sum),Nat.add(U32.to_nat(base),U32.to_nat(local)),exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    room)

def scaled_prefix(+base: U32,+local: U32,+width: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(local)),U32.to_nat(capacity)) == True{} : Bool},
  +product_room: {Nat.is_le(Nat.mul(U32.to_nat(local),U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((((base + local : U32) - base : U32) * width : U32)) == Nat.mul(U32.to_nat(local),U32.to_nat(width)) : Nat}:
  +relative = ((base + local : U32) - base : U32)
  +relative_exact = prefix_removed(base,local,capacity,capacity_ok,room)
  Equal.trans(Nat,U32.to_nat((relative * width : U32)),Nat.mul(U32.to_nat(relative),U32.to_nat(width)),Nat.mul(U32.to_nat(local),U32.to_nat(width)),
    Arithmetic.multiply(relative,width,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat(relative),U32.to_nat(local),relative_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool} product_room),
    Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(width)),U32.to_nat(relative),U32.to_nat(local),relative_exact))

def clipped_scaled(+last: U32,+first: U32,+width: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +last_room: {Nat.is_le(U32.to_nat(last),U32.to_nat(capacity)) == True{} : Bool},+first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(capacity)) == True{} : Bool},
  +product_room: {Nat.is_le(Nat.mul(Nat.sub(U32.to_nat(last),U32.to_nat(first)),U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((BandWindow.subtract_clipped(last,first) * width : U32)) == Nat.mul(Nat.sub(U32.to_nat(last),U32.to_nat(first)),U32.to_nat(width)) : Nat}:
  +count = BandWindow.subtract_clipped(last,first)
  +count_exact = TensorBandWindow.subtract_clipped(last,first,capacity,capacity_ok,last_room,first_room)
  Equal.trans(Nat,U32.to_nat((count * width : U32)),Nat.mul(U32.to_nat(count),U32.to_nat(width)),Nat.mul(Nat.sub(U32.to_nat(last),U32.to_nat(first)),U32.to_nat(width)),
    Arithmetic.multiply(count,width,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat(count),Nat.sub(U32.to_nat(last),U32.to_nat(first)),count_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool} product_room),
    Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(width)),U32.to_nat(count),Nat.sub(U32.to_nat(last),U32.to_nat(first)),count_exact))

# The window's first row: the first row read, moved down to a stride multiple.
def start_exact(+height: U32,+output_first: U32,+stride: U32,+padding: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +product_room: {Nat.is_le(Nat.mul(U32.to_nat(output_first),U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  +padding_room: {Nat.is_le(U32.to_nat(padding),U32.to_nat(capacity)) == True{} : Bool},
  +height_room: {Nat.is_le(U32.to_nat(height),U32.to_nat(capacity)) == True{} : Bool},
  +divisor_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool}) ->
  {U32.to_nat(BandWindow.start(height,output_first,stride,padding)) ==
    WindowContents.window_first(U32.to_nat(height),U32.to_nat(output_first),U32.to_nat(stride),U32.to_nat(padding)) : Nat}:
  +needed = TensorBandWindow.samples_first(height,output_first,stride,padding)
  +remainder = (needed % stride : U32)
  +needed_exact = TensorBandWindow.first_decoded(height,output_first,stride,padding,capacity,capacity_ok,product_room,padding_room)
  +needed_room = Order.transitive(U32.to_nat(needed),U32.to_nat(height),U32.to_nat(capacity),
    TensorBandWindow.minimum_below(height,BandWindow.subtract_clipped((output_first * stride : U32),padding)),height_room)
  +remainder_exact = U32Division.modulo(needed,stride,capacity,capacity_ok,divisor_room,positive)
  +remainder_room = {%Equal.sym(Nat,U32.to_nat(remainder),Nat.mod(U32.to_nat(needed),U32.to_nat(stride)),remainder_exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    Order.transitive(Nat.mod(U32.to_nat(needed),U32.to_nat(stride)),Nat.double(U32.to_nat(stride)),U32.to_nat(capacity),
      Order.transitive(Nat.mod(U32.to_nat(needed),U32.to_nat(stride)),U32.to_nat(stride),Nat.double(U32.to_nat(stride)),
        Bounds.strict_le(Nat.mod(U32.to_nat(needed),U32.to_nat(stride)),U32.to_nat(stride),DivisionOrder.remainder_small(U32.to_nat(needed),U32.to_nat(stride),positive)),
        Order.le_double(U32.to_nat(stride))),divisor_room) :
    {Nat.is_le(U32.to_nat(remainder),U32.to_nat(capacity)) == True{} : Bool}}
  +first_row = TensorBandWindow.needed_first(U32.to_nat(height),U32.to_nat(output_first),U32.to_nat(stride),U32.to_nat(padding))
  Equal.trans(Nat,U32.to_nat(BandWindow.subtract_clipped(needed,remainder)),Nat.sub(U32.to_nat(needed),U32.to_nat(remainder)),
    WindowContents.window_first(U32.to_nat(height),U32.to_nat(output_first),U32.to_nat(stride),U32.to_nat(padding)),
    TensorBandWindow.subtract_clipped(needed,remainder,capacity,capacity_ok,needed_room,remainder_room),
    Equal.trans(Nat,Nat.sub(U32.to_nat(needed),U32.to_nat(remainder)),Nat.sub(U32.to_nat(needed),Nat.mod(U32.to_nat(needed),U32.to_nat(stride))),
      WindowContents.window_first(U32.to_nat(height),U32.to_nat(output_first),U32.to_nat(stride),U32.to_nat(padding)),
      Equal.cong(Nat,Nat,value => Nat.sub(U32.to_nat(needed),value),U32.to_nat(remainder),Nat.mod(U32.to_nat(needed),U32.to_nat(stride)),remainder_exact),
      Equal.cong(Nat,Nat,value => Nat.sub(value,Nat.mod(value,U32.to_nat(stride))),U32.to_nat(needed),first_row,needed_exact)))

# The window's row count: the last row read minus the window's first row.
def rows_exact(+base: U32,+first: U32,+end: U32,+height: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +endpoint_room: {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(height)),U32.to_nat(capacity)) == True{} : Bool},
  +first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(height)) == True{} : Bool},+end_room: {Nat.is_le(U32.to_nat(end),U32.to_nat(height)) == True{} : Bool}) ->
  {U32.to_nat(BandWindow.subtract_clipped((base + end : U32),(base + first : U32))) == Nat.sub(U32.to_nat(end),U32.to_nat(first)) : Nat}:
  +upper = (base + end : U32)
  +lower = (base + first : U32)
  Equal.trans(Nat,U32.to_nat(BandWindow.subtract_clipped(upper,lower)),Nat.sub(U32.to_nat(upper),U32.to_nat(lower)),Nat.sub(U32.to_nat(end),U32.to_nat(first)),
    TensorBandWindow.subtract_clipped(upper,lower,capacity,capacity_ok,
      TensorBandWindow.shifted_room(base,end,height,capacity,capacity_ok,endpoint_room,end_room),TensorBandWindow.shifted_room(base,first,height,capacity,capacity_ok,endpoint_room,first_room)),
    %Equal.sym(Nat,U32.to_nat(upper),Nat.add(U32.to_nat(base),U32.to_nat(end)),TensorBandWindow.shifted(base,end,height,capacity,capacity_ok,endpoint_room,end_room)) : {Nat.sub(_,U32.to_nat(lower)) == Nat.sub(U32.to_nat(end),U32.to_nat(first)) : Nat}
    %Equal.sym(Nat,U32.to_nat(lower),Nat.add(U32.to_nat(base),U32.to_nat(first)),TensorBandWindow.shifted(base,first,height,capacity,capacity_ok,endpoint_room,first_room)) :
      {Nat.sub(Nat.add(U32.to_nat(base),U32.to_nat(end)),_) == Nat.sub(U32.to_nat(end),U32.to_nat(first)) : Nat}
    RowMajor.subtract_common_prefix(U32.to_nat(base),U32.to_nat(end),U32.to_nat(first)))
