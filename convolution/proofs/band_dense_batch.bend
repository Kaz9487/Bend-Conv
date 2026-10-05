# Dense batch output uses the shared finite Tensor contract. Empty public NCHW
# batches construct that contract directly. Actual image copies retain their
# destination allocation; bounded point readback connects each copied block.
import Base
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/math_activation.bend as Activations
import ../../lib/storage_buffer.bend as Storage
import ../../lib/storage_copy.bend as Copy
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_copy_refinement.bend as StorageCopyLaws
import ../../lib/proofs/contiguous_copy.bend as Contiguous
import ../../lib/proofs/contiguous_copy_readback.bend as CopyReadback
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/proofs/tensor_dense_contract.bend as Dense
import ../../lib/proofs/natural_addition.bend as Addition
import ./band_convolution_values.bend as BandConvolutionValues
import ./band_dense_single.bend as DenseSingle

def empty_shape(dimensions: KernelShape.Dimensions) -> {ShapeLaws.elements(DenseSingle.empty_output_shape(True{},dimensions)) == 0n : Nat}:
  KernelShape.Dimensions{ci,co,h,w,k,s,p} = dimensions
  {==}

def empty_dense(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+activation: Activations.Activation,+dimensions: KernelShape.Dimensions,
  -input: Storage.Buffer<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+workers: U32) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,
    TensorConvolution.execute_dense(~custom,activation,TensorConvolution.Geometry{0,True{},dimensions},input,Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers)):
  %Equal.sym(Tensors.Tensor<F32>,TensorConvolution.execute_dense(~custom,activation,TensorConvolution.Geometry{0,True{},dimensions},input,Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers),
    Tensors.Tensor{Tensors.buffer(~F32,0.0,0),DenseSingle.empty_output_shape(True{},dimensions),0.0,Tensors.Ready{},workers},DenseSingle.empty_dense(~custom,activation,True{},dimensions,input,weights,bias,workers)) :
    Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,_)
  Dense.empty_tensor(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,DenseSingle.empty_output_shape(True{},dimensions),0.0,workers,empty_shape(dimensions))

def empty_existing_layout(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+activation: Activations.Activation,+dimensions: KernelShape.Dimensions,+layout: Bands.Geometry,
  -source: Bands.Bands(~F32),+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+workers: U32) ->
  Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,
    TensorConvolution.execute_existing_layout(~custom,activation,TensorConvolution.uses_band_windows(TensorConvolution.Geometry{0,True{},dimensions}),TensorConvolution.Geometry{0,True{},dimensions},layout,source,Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers)):
  %Equal.sym(Tensors.Tensor<F32>,TensorConvolution.execute_existing_layout(~custom,activation,TensorConvolution.uses_band_windows(TensorConvolution.Geometry{0,True{},dimensions}),TensorConvolution.Geometry{0,True{},dimensions},layout,source,Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),workers),
    Tensors.Tensor{Tensors.buffer(~F32,0.0,0),DenseSingle.empty_output_shape(True{},dimensions),0.0,Tensors.Ready{},workers},DenseSingle.empty_existing_layout(~custom,activation,True{},dimensions,layout,source,weights,bias,workers)) :
    Dense.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,_)
  Dense.empty_tensor(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,DenseSingle.empty_output_shape(True{},dimensions),0.0,workers,empty_shape(dimensions))

def copy_content(owners: StorageCopyLaws.Owners<F32>) -> Buffer.Content<F32>:
  StorageCopyLaws.Owners{source,source_length,output,output_length} = owners
  Buffer.Content{output,output_length}

def output_projection(owners: Storage.Buffer<F32> & Storage.Buffer<F32>) ->
  {Buffer.contents(~F32,TensorConvolution.copied_output(owners)) == copy_content(StorageCopyLaws.observe(~F32,owners)) : Buffer.Content<F32>}:
  (Storage.Buffer{source,source_length,source_certificate},Storage.Buffer{output,length,certificate}) = owners
  {==}

def packed_owners(+input: Buffer.Model<F32>,+output: Buffer.Model<F32>) ->
  {StorageCopyLaws.observe(~F32,(Buffer.pack(~F32,input),Buffer.pack(~F32,output))) == StorageCopyLaws.Owners{Buffer.tree(~F32,input),Buffer.length(~F32,input),Buffer.tree(~F32,output),Buffer.length(~F32,output)} : StorageCopyLaws.Owners<F32>}:
  match input output:
    case Buffer.Model{+source,+source_length,source_certificate} Buffer.Model{+target,+target_length,target_certificate}:
      StorageCopyLaws.represented_values(~F32,source,source_length,target,target_length)

def copied_content(+count: U32,+source: U32,+destination: U32,+input: Buffer.Model<F32>,+output: Buffer.Model<F32>) -> Buffer.Content<F32>:
  Buffer.Content{Writes.pure(F32,Contiguous.writes(~F32,U32.to_nat(count),Buffer.tree(~F32,input),0,source,destination),Buffer.tree(~F32,output)),Buffer.length(~F32,output)}

def content_parts(+model: Buffer.Model<F32>) -> {Buffer.content(~F32,model) == Buffer.Content{Buffer.tree(~F32,model),Buffer.length(~F32,model)} : Buffer.Content<F32>}:
  Buffer.Model{tree,length,certificate} = model
  {==}

def copy_output_contents(+count: U32,+source: U32,+destination: U32,+input: Buffer.Model<F32>,+output: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))) == copied_content(count,source,destination,input,output) : Buffer.Content<F32>}:
  Equal.trans(Buffer.Content<F32>,Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))),
    copy_content(StorageCopyLaws.observe(~F32,Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))),copied_content(count,source,destination,input,output),
    output_projection(Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output))),
    Equal.cong(StorageCopyLaws.Owners<F32>,Buffer.Content<F32>,copy_content,
      StorageCopyLaws.observe(~F32,Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output))),
      StorageCopyLaws.model(~F32,count,source,destination,StorageCopyLaws.Owners{Buffer.tree(~F32,input),Buffer.length(~F32,input),Buffer.tree(~F32,output),Buffer.length(~F32,output)}),
      Equal.trans(StorageCopyLaws.Owners<F32>,StorageCopyLaws.observe(~F32,Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output))),
        StorageCopyLaws.model(~F32,count,source,destination,StorageCopyLaws.observe(~F32,(Buffer.pack(~F32,input),Buffer.pack(~F32,output)))),
        StorageCopyLaws.model(~F32,count,source,destination,StorageCopyLaws.Owners{Buffer.tree(~F32,input),Buffer.length(~F32,input),Buffer.tree(~F32,output),Buffer.length(~F32,output)}),
        StorageCopyLaws.range(~F32,count,source,destination,(Buffer.pack(~F32,input),Buffer.pack(~F32,output))),
        Equal.cong(StorageCopyLaws.Owners<F32>,StorageCopyLaws.Owners<F32>,owners => StorageCopyLaws.model(~F32,count,source,destination,owners),
          StorageCopyLaws.observe(~F32,(Buffer.pack(~F32,input),Buffer.pack(~F32,output))),StorageCopyLaws.Owners{Buffer.tree(~F32,input),Buffer.length(~F32,input),Buffer.tree(~F32,output),Buffer.length(~F32,output)},packed_owners(input,output)))))

def copy_output_ready(+count: U32,+source: U32,+destination: U32,+input: Buffer.Model<F32>,+output: Buffer.Model<F32>,ready: Buffer.Ready<F32,output>) ->
  Buffer.ContentReady<F32,Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output))))>:
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))),copied_content(count,source,destination,input,output),copy_output_contents(count,source,destination,input,output)) : Buffer.ContentReady<F32,_>
  Writes.content_ready(~F32,Contiguous.writes(~F32,U32.to_nat(count),Buffer.tree(~F32,input),0,source,destination),Buffer.tree(~F32,output),Buffer.length(~F32,output),
    %content_parts(output) : Buffer.ContentReady<F32,_>
    Buffer.content_ready(~F32,output,ready))

def copy_inside(+count: U32,+source: U32,+destination: U32,+input: Buffer.Model<F32>,+output: Buffer.Model<F32>,+index: Nat,
  ready: Buffer.Ready<F32,output>,room: {Nat.is_le(Nat.add(U32.to_nat(destination),U32.to_nat(count)),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool},
  inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Buffer.value(~F32,(destination + U32.from_nat(index) : U32),Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output))))) ==
    Tree.read(F32,Buffer.tree(~F32,input),(source + U32.from_nat(index) : U32)) : F32}:
  Buffer.Ready{+depth_ok,allocation_room} = ready
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,TensorConvolution.copied_output(Copy.range(~F32,count,source,destination,Buffer.pack(~F32,input),Buffer.pack(~F32,output)))),copied_content(count,source,destination,input,output),copy_output_contents(count,source,destination,input,output)) :
    {Buffer.value(~F32,(destination + U32.from_nat(index) : U32),_) == Tree.read(F32,Buffer.tree(~F32,input),(source + U32.from_nat(index) : U32)) : F32}
  CopyReadback.read_inside(~F32,U32.to_nat(count),0n,index,Buffer.tree(~F32,input),Buffer.tree(~F32,output),source,destination,Buffer.depth(~F32,output),
    Buffer.balanced(~F32,output),depth_ok,Buffer.capacity_bound(~F32,output,depth_ok),
    %Equal.sym(Nat,Nat.add(U32.to_nat(destination),0n),U32.to_nat(destination),Addition.zero_right(U32.to_nat(destination))) :
      {Nat.is_le(Nat.add(_,U32.to_nat(count)),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool}
    room,inside)
