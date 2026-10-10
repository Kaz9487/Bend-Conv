# The convolution band leaf: its execution, numeric guards and readback.
import Base
import ./blocked_pipeline_model.bend as Blocked
import ../../lib/proofs/kernel_pipeline_activation.bend as ActivationProofs
import ../../lib/proofs/tensor_band_window.bend as TensorBandWindow
import ../../lib/kernel_gather.bend as Gather
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/kernel_tile.bend as Tile
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/tensor.bend as Tensors
import ./band_window_contents.bend as WindowContents
import ../../lib/kernel_gather_buffer.bend as GatherBuffer
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/storage_buffer.bend as Storage
import ../../lib/tensor_band_window.bend as BandWindow

# The outer partition finish and shared blocked leaf on certified buffers.
def finish_array(identity: Bool,values: Array<F32>,+rows: U32,+offset: U32,+span: U32,+positions: U32) -> Array<F32>:
  match identity:
    case True{}: values
    case False{}: Gather.gather_block(rows,offset,positions,span,values,Array.new(F32,Tensors.depth_for((rows * positions : U32)),0.0))

def execution_finished(+identity: Bool,input: Storage.Buffer<F32>,+rows: U32,+offset: U32,+span: U32,+positions: U32) ->
  {ActivationProofs.storage(~F32,Pipeline.finish_leaf(identity,input,rows,offset,span,positions)) == finish_array(identity,ActivationProofs.storage(~F32,input),rows,offset,span,positions) : Array<F32>}:
  match identity input:
    case True{} Storage.Buffer{values,length,certificate}: {==}
    case False{} Storage.Buffer{values,length,certificate}:
      %Equal.sym(Storage.Buffer<F32>,Tensors.buffer(~F32,0.0,(rows * positions : U32)),Storage.allocate(~F32,Tensors.depth_for((rows * positions : U32)),0.0,U32.to_nat((rows * positions : U32))),ActivationProofs.allocated(~F32,0.0,(rows * positions : U32))) :
        {ActivationProofs.storage(~F32,GatherBuffer.run(rows,offset,positions,span,Storage.Buffer{values,length,certificate},_)) ==
          finish_array(False{},values,rows,offset,span,positions) : Array<F32>}
      {==}

# Tree models of the same steps.
def work(+rows: U32,+inner: U32,+positions: U32,call: Packing.Call) -> Pipeline.Work:
  Packing.Call{count,start,height,width,kernel,stride,padding,output_width,call_inner} = call
  Pipeline.Work{rows,height,width,kernel,stride,padding,output_width,positions,inner}

def pack_start(call: Packing.Call) -> U32:
  Packing.Call{count,start,height,width,kernel,stride,padding,output_width,inner} = call
  start

def leaf_tree(~custom: F32 -> F32,activation: Activations.Activation,+rows: U32,+inner: U32,+positions: U32,+call: Packing.Call,
  window: Tree.Tree<F32>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>) -> Tree.Tree<F32>:
  Blocked.run(~custom,activation,work(rows,inner,positions,call),pack_start(call),window,weights,bias)

# The actual band leaf computation on canonical window, weight and bias owners.
# Buffer lengths through the leaf: every pass keeps its output owner's length.
def buffer_length(~Element: Data,buffer: Storage.Buffer<Element>) -> Nat:
  Storage.Buffer{storage,length,certificate} = buffer
  length

def contents_parts(~Element: Data,buffer: Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,buffer) == Buffer.Content{Tree.reflect(Element,ActivationProofs.storage(~Element,buffer)),buffer_length(~Element,buffer)} : Buffer.Content<Element>}:
  match buffer:
    case Storage.Buffer{values,length,certificate}: {==}

def leaf_contents(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+inner: U32,+positions: U32,+call: Packing.Call,
  +window: Buffer.Model<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Pipeline.blocked(~custom,activation,work(rows,inner,positions,call),pack_start(call),Buffer.pack(~F32,window),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) ==
    Buffer.Content{leaf_tree(~custom,activation,rows,inner,positions,call,Buffer.tree(~F32,window),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias)),
      U32.to_nat((rows * positions : U32))} : Buffer.Content<F32>}:
  Packing.Call{+size,+start,+height,+width,+kernel,+stride,+padding,+output_width,+call_inner} = call
  +call = {Packing.Call{size,start,height,width,kernel,stride,padding,output_width,call_inner} : Packing.Call}
  Blocked.run_contents(~custom,activation,work(rows,inner,positions,call),pack_start(call),window,weights,bias)

# The same contents for any window owner, read through its observed model.
def leaf_contents_model(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+inner: U32,+positions: U32,+call: Packing.Call,-window: Storage.Buffer<F32>,+model: Buffer.Model<F32>,
  +equation: {Buffer.pack(~F32,model) == window : Storage.Buffer<F32>},+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Pipeline.blocked(~custom,activation,work(rows,inner,positions,call),pack_start(call),window,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) ==
    Buffer.Content{leaf_tree(~custom,activation,rows,inner,positions,call,Tree.reflect(F32,ActivationProofs.storage(~F32,window)),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias)),
      U32.to_nat((rows * positions : U32))} : Buffer.Content<F32>}:
  Buffer.Model{+tree,+length,+certificate} = model
  %equation : {Buffer.contents(~F32,Pipeline.blocked(~custom,activation,work(rows,inner,positions,call),pack_start(call),_,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) ==
    Buffer.Content{leaf_tree(~custom,activation,rows,inner,positions,call,Tree.reflect(F32,ActivationProofs.storage(~F32,_)),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias)),
      U32.to_nat((rows * positions : U32))} : Buffer.Content<F32>}
  %Equal.sym(Tree.Tree<F32>,Tree.reflect(F32,Tree.pack(F32,tree)),tree,StorageLaws.reflect_packed(F32,tree)) : {Buffer.contents(~F32,Pipeline.blocked(~custom,activation,work(rows,inner,positions,call),pack_start(call),Buffer.pack(~F32,Buffer.Model{tree,length,certificate}),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) ==
    Buffer.Content{leaf_tree(~custom,activation,rows,inner,positions,call,_,Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias)),
      U32.to_nat((rows * positions : U32))} : Buffer.Content<F32>}
  leaf_contents(~custom,activation,rows,inner,positions,call,Buffer.Model{tree,length,certificate},weights,bias)

def leaf_contents_observed(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+inner: U32,+positions: U32,+call: Packing.Call,-window: Storage.Buffer<F32>,observation: Buffer.Observation<F32,window>,
  +weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Pipeline.blocked(~custom,activation,work(rows,inner,positions,call),pack_start(call),window,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) ==
    Buffer.Content{leaf_tree(~custom,activation,rows,inner,positions,call,Tree.reflect(F32,ActivationProofs.storage(~F32,window)),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias)),
      U32.to_nat((rows * positions : U32))} : Buffer.Content<F32>}:
  Buffer.Observation{+model,+equation} = observation
  leaf_contents_model(~custom,activation,rows,inner,positions,call,window,model,equation,weights,bias)

def leaf_contents_buffer(~custom: F32 -> F32,+activation: Activations.Activation,+rows: U32,+inner: U32,+positions: U32,+call: Packing.Call,window: Storage.Buffer<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Pipeline.blocked(~custom,activation,work(rows,inner,positions,call),pack_start(call),window,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) ==
    Buffer.Content{leaf_tree(~custom,activation,rows,inner,positions,call,Tree.reflect(F32,ActivationProofs.storage(~F32,window)),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias)),
      U32.to_nat((rows * positions : U32))} : Buffer.Content<F32>}:
  leaf_contents_observed(~custom,activation,rows,inner,positions,call,window,Buffer.observe(~F32,window),weights,bias)

# Window coordinates cancel their shared prefix before scaling.
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
