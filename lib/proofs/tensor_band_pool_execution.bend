# Band pooling: window allocation and guards, the ordered dense pool on each
# assembled window, and its readback values.
import Base
import ../tensor_band.bend as Bands
import ../tensor_band_pool.bend as BandPool
import ../tensor_band_window.bend as BandWindow
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_band_model.bend as Model
import ../tensor_access.bend as Access
import ./nat_order.bend as Order
import ./tensor_allocation_depth.bend as Depth
import ./tensor_band_halo.bend as BandHalo
import ./tensor_band_window.bend as TensorBandWindow
import ./tensor_sweep_refinement.bend as SweepLaws
import ./tensor_sweep_values.bend as Sweeps
import ./tensor_band_sweep_values.bend as BandSweeps
import ./tensor_pool_dense_values.bend as PoolDense
import ./traversal_indexed_values.bend as IndexedValues
import ../tensor.bend as Tensors
import ./storage_shape.bend as Shape
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./row_major_index.bend as RowMajor
import ./storage_readback.bend as StorageReadback
import ./tensor_access_refinement.bend as AccessLaws
import ../traversal_array.bend as Traversal
import ../machine_limits.bend as Machine
import ./boolean_logic.bend as Boolean
import ./encoded_coordinates.bend as Encoded
import ./indexed_output_readback.bend as OutputReadback
import ./machine_limit_bounds.bend as Limits
import ./nat_to_u32_bounds.bend as Bounds
import ./ordered_fold_refinement.bend as Fold
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_certificate.bend as Certificate
import ./storage_refinement.bend as StorageLaws
import ./tensor_access_certificate.bend as AccessCertificate
import ./tensor_band_dense.bend as BandDense
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_values.bend as BandValues
import ./tensor_band_window_apply.bend as Apply
import ./tensor_operations_refinement.bend as OperationsLaws
import ./traversal_array_refinement.bend as ArrayLaws
import ./traversal_certificate.bend as TraversalCertificate
import ./traversal_map_readback.bend as MapReadback
import ./u32_comparison.bend as Comparison
import ../storage_buffer.bend as Storage
import ../tensor_operations.bend as Operations
import ../traversal_loop.bend as Loop
import ../traversal_partition.bend as Partition

# The band pool calls the existing ordered dense pool on its assembled window.
# This connects the real Buffer caller, preserving certificates and length.
# Window sample coverage and global Values are separate producer obligations.
def right_content(~Element: Data,owners: OperationsLaws.PairOwners(Element)) -> Buffer.Content<Element>:
  (left,OperationsLaws.Owner{storage,length}) = owners
  Buffer.Content{Tree.reflect(Element,storage),length}

def destination_content(~Element: Data,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,Bands.destination(~Element,owners)) == right_content(~Element,OperationsLaws.observe_pair(~Element,owners)) : Buffer.Content<Element>}:
  (left,Storage.Buffer{storage,length,certificate}) = owners
  {==}

def modeled_content(~Element: Data,+input_length: Nat,+output_length: Nat,+result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) ->
  {right_content(~Element,OperationsLaws.modeled_pair(~Element,input_length,output_length,result)) ==
    Buffer.Content{IndexedValues.output_tree(~Tree.Tree<Element>,~Element,result),output_length} : Buffer.Content<Element>}:
  AccessLaws.ModelResult{input,+output} = result
  Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,output_length},
    Tree.reflect(Element,Tree.pack(Element,output)),output,StorageLaws.reflect_packed(Element,output))

def packed_swept(~Element: Data,~operation: Element -> Element -> Element,+count: Nat,+context: Access.SweepContext<Element>,
  +input: Buffer.Model<Element>,+output: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,Bands.destination(~Element,Operations.swept(~Element,~operation,count,context,Buffer.pack(~Element,input),Buffer.pack(~Element,output)))) ==
    Sweeps.model(~Element,~operation,count,context,Buffer.content(~Element,input),Buffer.content(~Element,output)) : Buffer.Content<Element>}:
  Buffer.Model{+input_storage,+input_length,+input_certificate} = input
  Buffer.Model{+output_storage,+output_length,+output_certificate} = output
  +input = {Buffer.Model{input_storage,input_length,input_certificate} : Buffer.Model<Element>}
  +output = {Buffer.Model{output_storage,output_length,output_certificate} : Buffer.Model<Element>}
  +modeled = {SweepLaws.sweep_model(~Element,~operation,count,context,input_storage,output_storage) : AccessLaws.ModelResult<Tree.Tree<Element>,Element>}
  Equal.trans(Buffer.Content<Element>,
    Buffer.contents(~Element,Bands.destination(~Element,Operations.swept(~Element,~operation,count,context,Buffer.pack(~Element,input),Buffer.pack(~Element,output)))),
    right_content(~Element,OperationsLaws.modeled_pair(~Element,input_length,output_length,modeled)),Buffer.Content{IndexedValues.output_tree(~Tree.Tree<Element>,~Element,modeled),output_length},
    Equal.trans(Buffer.Content<Element>,
      Buffer.contents(~Element,Bands.destination(~Element,Operations.swept(~Element,~operation,count,context,Buffer.pack(~Element,input),Buffer.pack(~Element,output)))),
      right_content(~Element,OperationsLaws.observe_pair(~Element,Operations.swept(~Element,~operation,count,context,Buffer.pack(~Element,input),Buffer.pack(~Element,output)))),
      right_content(~Element,OperationsLaws.modeled_pair(~Element,input_length,output_length,modeled)),
      destination_content(~Element,Operations.swept(~Element,~operation,count,context,Buffer.pack(~Element,input),Buffer.pack(~Element,output))),
      Equal.cong(OperationsLaws.PairOwners(Element),Buffer.Content<Element>,owners => right_content(~Element,owners),
        OperationsLaws.observe_pair(~Element,Operations.swept(~Element,~operation,count,context,Buffer.pack(~Element,input),Buffer.pack(~Element,output))),OperationsLaws.modeled_pair(~Element,input_length,output_length,modeled),
        Sweeps.swept_packed(~Element,~operation,count,context,input,output))),
    modeled_content(~Element,input_length,output_length,modeled))

def swept_observed(~Element: Data,~operation: Element -> Element -> Element,+count: Nat,+context: Access.SweepContext<Element>,
  -input: Storage.Buffer<Element>,-output: Storage.Buffer<Element>,
  input_observation: Buffer.Observation<Element,input>,output_observation: Buffer.Observation<Element,output>) ->
  {Buffer.contents(~Element,Bands.destination(~Element,Operations.swept(~Element,~operation,count,context,input,output))) ==
    Sweeps.model(~Element,~operation,count,context,Buffer.contents(~Element,input),Buffer.contents(~Element,output)) : Buffer.Content<Element>}:
  Buffer.Observation{+input_model,input_equal} = input_observation
  Buffer.Observation{+output_model,output_equal} = output_observation
  %input_equal :
    {Buffer.contents(~Element,Bands.destination(~Element,Operations.swept(~Element,~operation,count,context,_,output))) ==
      Sweeps.model(~Element,~operation,count,context,Buffer.contents(~Element,_),Buffer.contents(~Element,output)) : Buffer.Content<Element>}
  %output_equal :
    {Buffer.contents(~Element,Bands.destination(~Element,Operations.swept(~Element,~operation,count,context,Buffer.pack(~Element,input_model),_))) ==
      Sweeps.model(~Element,~operation,count,context,Buffer.contents(~Element,Buffer.pack(~Element,input_model)),Buffer.contents(~Element,_)) : Buffer.Content<Element>}
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,input_model)),Buffer.content(~Element,input_model),Buffer.packed_contents(~Element,input_model)) :
    {Buffer.contents(~Element,Bands.destination(~Element,Operations.swept(~Element,~operation,count,context,Buffer.pack(~Element,input_model),Buffer.pack(~Element,output_model)))) ==
      Sweeps.model(~Element,~operation,count,context,_,Buffer.contents(~Element,Buffer.pack(~Element,output_model))) : Buffer.Content<Element>}
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,output_model)),Buffer.content(~Element,output_model),Buffer.packed_contents(~Element,output_model)) :
    {Buffer.contents(~Element,Bands.destination(~Element,Operations.swept(~Element,~operation,count,context,Buffer.pack(~Element,input_model),Buffer.pack(~Element,output_model)))) ==
      Sweeps.model(~Element,~operation,count,context,Buffer.content(~Element,input_model),_) : Buffer.Content<Element>}
  packed_swept(~Element,~operation,count,context,input_model,output_model)

def refinement_swept(~Element: Data,~operation: Element -> Element -> Element,+count: Nat,+context: Access.SweepContext<Element>,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,Bands.destination(~Element,Operations.swept(~Element,~operation,count,context,input,output))) ==
    Sweeps.model(~Element,~operation,count,context,Buffer.contents(~Element,input),Buffer.contents(~Element,output)) : Buffer.Content<Element>}:
  swept_observed(~Element,~operation,count,context,input,output,Buffer.observe(~Element,input),Buffer.observe(~Element,output))

# The model of a pooled window: the planes swept along the columns, then the
# output swept along the rows from origin.
def pooled_model(~Element: Data,~operation: Element -> Element -> Element,+pool: Access.Pool,+pad: Element,origin: U32,+channels: U32,
  input: Buffer.Content<Element>,middle: Buffer.Content<Element>,output: Buffer.Content<Element>) -> Buffer.Content<Element>:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  +pool = {Access.Pool{height,width,kernel,stride,padding,out_height,out_width} : Access.Pool}
  Sweeps.model(~Element,~operation,U32.to_nat((channels * out_height * out_width : U32)),Access.SweepContext{Access.pool_rows(pool,origin),pad},
    Sweeps.model(~Element,~operation,U32.to_nat((channels * height * out_width : U32)),Access.SweepContext{Access.pool_columns(pool),pad},input,middle),output)

def packed_pooled(~Element: Data,~operation: Element -> Element -> Element,+pool: Access.Pool,+pad: Element,+origin: U32,+channels: U32,
  +input: Buffer.Model<Element>,+middle: Buffer.Model<Element>,+output: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,BandPool.pooled(~Element,~operation,pool,pad,origin,channels,Buffer.pack(~Element,input),Buffer.pack(~Element,middle),Buffer.pack(~Element,output))) == pooled_model(~Element,~operation,pool,pad,origin,channels,Buffer.content(~Element,input),Buffer.content(~Element,middle),Buffer.content(~Element,output)) : Buffer.Content<Element>}:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  +pool = {Access.Pool{height,width,kernel,stride,padding,out_height,out_width} : Access.Pool}
  +columns = {Access.SweepContext{Access.pool_columns(pool),pad} : Access.SweepContext<Element>}
  +lines = {Access.SweepContext{Access.pool_rows(pool,origin),pad} : Access.SweepContext<Element>}
  +column_count = U32.to_nat((channels * height * out_width : U32))
  +row_count = U32.to_nat((channels * out_height * out_width : U32))
  Equal.trans(Buffer.Content<Element>,
    Buffer.contents(~Element,Bands.destination(~Element,Operations.swept(~Element,~operation,row_count,lines,Bands.destination(~Element,Operations.swept(~Element,~operation,column_count,columns,Buffer.pack(~Element,input),Buffer.pack(~Element,middle))),Buffer.pack(~Element,output)))),
    Sweeps.model(~Element,~operation,row_count,lines,Buffer.contents(~Element,Bands.destination(~Element,Operations.swept(~Element,~operation,column_count,columns,Buffer.pack(~Element,input),Buffer.pack(~Element,middle)))),Buffer.contents(~Element,Buffer.pack(~Element,output))),
    Sweeps.model(~Element,~operation,row_count,lines,Sweeps.model(~Element,~operation,column_count,columns,Buffer.content(~Element,input),Buffer.content(~Element,middle)),Buffer.content(~Element,output)),
    refinement_swept(~Element,~operation,row_count,lines,Bands.destination(~Element,Operations.swept(~Element,~operation,column_count,columns,Buffer.pack(~Element,input),Buffer.pack(~Element,middle))),Buffer.pack(~Element,output)),
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Bands.destination(~Element,Operations.swept(~Element,~operation,column_count,columns,Buffer.pack(~Element,input),Buffer.pack(~Element,middle)))),Sweeps.model(~Element,~operation,column_count,columns,Buffer.content(~Element,input),Buffer.content(~Element,middle)),packed_swept(~Element,~operation,column_count,columns,input,middle)) :
      {Sweeps.model(~Element,~operation,row_count,lines,_,Buffer.contents(~Element,Buffer.pack(~Element,output))) ==
        Sweeps.model(~Element,~operation,row_count,lines,Sweeps.model(~Element,~operation,column_count,columns,Buffer.content(~Element,input),Buffer.content(~Element,middle)),Buffer.content(~Element,output)) : Buffer.Content<Element>}
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,output)),Buffer.content(~Element,output),Buffer.packed_contents(~Element,output)) :
      {Sweeps.model(~Element,~operation,row_count,lines,Sweeps.model(~Element,~operation,column_count,columns,Buffer.content(~Element,input),Buffer.content(~Element,middle)),_) ==
        Sweeps.model(~Element,~operation,row_count,lines,Sweeps.model(~Element,~operation,column_count,columns,Buffer.content(~Element,input),Buffer.content(~Element,middle)),Buffer.content(~Element,output)) : Buffer.Content<Element>}
    {==})

def pooled_observed(~Element: Data,~operation: Element -> Element -> Element,+pool: Access.Pool,+pad: Element,+origin: U32,+channels: U32,
  -input: Storage.Buffer<Element>,-middle: Storage.Buffer<Element>,-output: Storage.Buffer<Element>,
  input_observation: Buffer.Observation<Element,input>,middle_observation: Buffer.Observation<Element,middle>,output_observation: Buffer.Observation<Element,output>) ->
  {Buffer.contents(~Element,BandPool.pooled(~Element,~operation,pool,pad,origin,channels,input,middle,output)) ==
    pooled_model(~Element,~operation,pool,pad,origin,channels,Buffer.contents(~Element,input),Buffer.contents(~Element,middle),Buffer.contents(~Element,output)) : Buffer.Content<Element>}:
  Buffer.Observation{+input_model,input_equal} = input_observation
  Buffer.Observation{+middle_model,middle_equal} = middle_observation
  Buffer.Observation{+output_model,output_equal} = output_observation
  %input_equal :
    {Buffer.contents(~Element,BandPool.pooled(~Element,~operation,pool,pad,origin,channels,_,middle,output)) ==
      pooled_model(~Element,~operation,pool,pad,origin,channels,Buffer.contents(~Element,_),Buffer.contents(~Element,middle),Buffer.contents(~Element,output)) : Buffer.Content<Element>}
  %middle_equal :
    {Buffer.contents(~Element,BandPool.pooled(~Element,~operation,pool,pad,origin,channels,Buffer.pack(~Element,input_model),_,output)) ==
      pooled_model(~Element,~operation,pool,pad,origin,channels,Buffer.contents(~Element,Buffer.pack(~Element,input_model)),Buffer.contents(~Element,_),Buffer.contents(~Element,output)) : Buffer.Content<Element>}
  %output_equal :
    {Buffer.contents(~Element,BandPool.pooled(~Element,~operation,pool,pad,origin,channels,Buffer.pack(~Element,input_model),Buffer.pack(~Element,middle_model),_)) ==
      pooled_model(~Element,~operation,pool,pad,origin,channels,Buffer.contents(~Element,Buffer.pack(~Element,input_model)),Buffer.contents(~Element,Buffer.pack(~Element,middle_model)),Buffer.contents(~Element,_)) : Buffer.Content<Element>}
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,input_model)),Buffer.content(~Element,input_model),Buffer.packed_contents(~Element,input_model)) :
    {Buffer.contents(~Element,BandPool.pooled(~Element,~operation,pool,pad,origin,channels,Buffer.pack(~Element,input_model),Buffer.pack(~Element,middle_model),Buffer.pack(~Element,output_model))) ==
      pooled_model(~Element,~operation,pool,pad,origin,channels,_,Buffer.contents(~Element,Buffer.pack(~Element,middle_model)),Buffer.contents(~Element,Buffer.pack(~Element,output_model))) : Buffer.Content<Element>}
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,middle_model)),Buffer.content(~Element,middle_model),Buffer.packed_contents(~Element,middle_model)) :
    {Buffer.contents(~Element,BandPool.pooled(~Element,~operation,pool,pad,origin,channels,Buffer.pack(~Element,input_model),Buffer.pack(~Element,middle_model),Buffer.pack(~Element,output_model))) ==
      pooled_model(~Element,~operation,pool,pad,origin,channels,Buffer.content(~Element,input_model),_,Buffer.contents(~Element,Buffer.pack(~Element,output_model))) : Buffer.Content<Element>}
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,output_model)),Buffer.content(~Element,output_model),Buffer.packed_contents(~Element,output_model)) :
    {Buffer.contents(~Element,BandPool.pooled(~Element,~operation,pool,pad,origin,channels,Buffer.pack(~Element,input_model),Buffer.pack(~Element,middle_model),Buffer.pack(~Element,output_model))) ==
      pooled_model(~Element,~operation,pool,pad,origin,channels,Buffer.content(~Element,input_model),Buffer.content(~Element,middle_model),_) : Buffer.Content<Element>}
  packed_pooled(~Element,~operation,pool,pad,origin,channels,input_model,middle_model,output_model)

def refinement_pooled(~Element: Data,~operation: Element -> Element -> Element,+pool: Access.Pool,+pad: Element,+origin: U32,+channels: U32,
  input: Storage.Buffer<Element>,middle: Storage.Buffer<Element>,output: Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,BandPool.pooled(~Element,~operation,pool,pad,origin,channels,input,middle,output)) ==
    pooled_model(~Element,~operation,pool,pad,origin,channels,Buffer.contents(~Element,input),Buffer.contents(~Element,middle),Buffer.contents(~Element,output)) : Buffer.Content<Element>}:
  pooled_observed(~Element,~operation,pool,pad,origin,channels,input,middle,output,Buffer.observe(~Element,input),Buffer.observe(~Element,middle),Buffer.observe(~Element,output))

def refinement_window_model(~Element: Data,~operation: Element -> Element -> Element,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+first: U32,+rows: U32,
  rows_read: BandWindow.RowWindow,core: Buffer.Content<Element>,above: Buffer.Content<Element>,below: Buffer.Content<Element>) -> Model.Content<Element>:
  BandWindow.Geometry{Bands.Geometry{batch,+channels,+height,+width},+kernel,+stride,+padding,+output_height,+output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{+output_first,+output_rows},+above_range,+below_range,+window} = rows_read
  BandWindow.Rows{+window_first,+window_rows} = window
  +middle_count = (channels * window_rows * output_width : U32)
  +count = (channels * output_rows * output_width : U32)
  Model.RowBand{pooled_model(~Element,~operation,Access.Pool{window_rows,width,kernel,stride,padding,output_rows,output_width},pad,((output_first - first / height * output_height) * stride + first / height * height - window_first : U32),channels,TensorBandWindow.assembly_content(~Element,BandWindow.input_geometry(geometry),fill,window,BandWindow.Rows{first,rows},above_range,below_range,core,above,below),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(middle_count),fill),U32.to_nat(middle_count)},Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)}),output_first,output_rows}

def refinement_windowed(~Element: Data,~operation: Element -> Element -> Element,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+first: U32,+rows: U32,
  +rows_read: BandWindow.RowWindow,+core: Buffer.Model<Element>,+above: Buffer.Model<Element>,+below: Buffer.Model<Element>) ->
  {Model.contents(~Element,BandPool.windowed(~Element,~operation,geometry,pad,fill,first,rows,rows_read,Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below))) == refinement_window_model(~Element,~operation,geometry,pad,fill,first,rows,rows_read,Buffer.content(~Element,core),Buffer.content(~Element,above),Buffer.content(~Element,below)) : Model.Content<Element>}:
  BandWindow.Geometry{Bands.Geometry{batch,+channels,+height,+width},+kernel,+stride,+padding,+output_height,+output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{+output_first,+output_rows},+above_range,+below_range,+window} = rows_read
  BandWindow.Rows{+window_first,+window_rows} = window
  +geometry = {BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,output_height,output_width} : BandWindow.Geometry}
  +middle_count = (channels * window_rows * output_width : U32)
  +count = (channels * output_rows * output_width : U32)
  +pool = {Access.Pool{window_rows,width,kernel,stride,padding,output_rows,output_width} : Access.Pool}
  +origin = ((output_first - first / height * output_height) * stride + first / height * height - window_first : U32)
  +core_content = Buffer.content(~Element,core)
  +above_content = Buffer.content(~Element,above)
  +below_content = Buffer.content(~Element,below)
  +assembled = TensorBandWindow.assembly_content(~Element,BandWindow.input_geometry(geometry),fill,window,BandWindow.Rows{first,rows},above_range,below_range,core_content,above_content,below_content)
  Equal.cong(Buffer.Content<Element>,Model.Content<Element>,buffer => Model.RowBand{buffer,output_first,output_rows},
    Buffer.contents(~Element,BandPool.pooled(~Element,~operation,pool,pad,origin,channels,BandWindow.assemble(~Element,fill,BandWindow.input_geometry(geometry),window,BandWindow.Rows{first,rows},above_range,below_range,Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below)),Tensors.buffer(~Element,fill,middle_count),Tensors.buffer(~Element,fill,count))),
    pooled_model(~Element,~operation,pool,pad,origin,channels,assembled,Buffer.Content{Tree.fresh(Element,Tensors.depth_for(middle_count),fill),U32.to_nat(middle_count)},Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)}),
    %TensorBandWindow.assemble_contents(~Element,BandWindow.input_geometry(geometry),fill,window,BandWindow.Rows{first,rows},above_range,below_range,core,above,below) :
      {Buffer.contents(~Element,BandPool.pooled(~Element,~operation,pool,pad,origin,channels,BandWindow.assemble(~Element,fill,BandWindow.input_geometry(geometry),window,BandWindow.Rows{first,rows},above_range,below_range,Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below)),Tensors.buffer(~Element,fill,middle_count),Tensors.buffer(~Element,fill,count))) ==
        pooled_model(~Element,~operation,pool,pad,origin,channels,_,Buffer.Content{Tree.fresh(Element,Tensors.depth_for(middle_count),fill),U32.to_nat(middle_count)},Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)}) : Buffer.Content<Element>}
    %AllocationLaws.contents(~Element,Tensors.depth_for(middle_count),fill,U32.to_nat(middle_count)) :
      {Buffer.contents(~Element,BandPool.pooled(~Element,~operation,pool,pad,origin,channels,BandWindow.assemble(~Element,fill,BandWindow.input_geometry(geometry),window,BandWindow.Rows{first,rows},above_range,below_range,Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below)),Tensors.buffer(~Element,fill,middle_count),Tensors.buffer(~Element,fill,count))) ==
        pooled_model(~Element,~operation,pool,pad,origin,channels,Buffer.contents(~Element,BandWindow.assemble(~Element,fill,BandWindow.input_geometry(geometry),window,BandWindow.Rows{first,rows},above_range,below_range,Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below))),_,Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)}) : Buffer.Content<Element>}
    %AllocationLaws.contents(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)) :
      {Buffer.contents(~Element,BandPool.pooled(~Element,~operation,pool,pad,origin,channels,BandWindow.assemble(~Element,fill,BandWindow.input_geometry(geometry),window,BandWindow.Rows{first,rows},above_range,below_range,Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below)),Tensors.buffer(~Element,fill,middle_count),Tensors.buffer(~Element,fill,count))) ==
        pooled_model(~Element,~operation,pool,pad,origin,channels,Buffer.contents(~Element,BandWindow.assemble(~Element,fill,BandWindow.input_geometry(geometry),window,BandWindow.Rows{first,rows},above_range,below_range,Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below))),Buffer.contents(~Element,Tensors.buffer(~Element,fill,middle_count)),_) : Buffer.Content<Element>}
    refinement_pooled(~Element,~operation,pool,pad,origin,channels,BandWindow.assemble(~Element,fill,BandWindow.input_geometry(geometry),window,BandWindow.Rows{first,rows},above_range,below_range,Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below)),Tensors.buffer(~Element,fill,middle_count),Tensors.buffer(~Element,fill,count)))

# The sweeps only store into their destinations: readiness of the output is
# kept whatever the planes hold.
def model_ready(~Element: Data,~operation: Element -> Element -> Element,+pool: Access.Pool,+pad: Element,+origin: U32,+channels: U32,
  input: Buffer.Content<Element>,middle: Buffer.Content<Element>,output: Buffer.Content<Element>,
  evidence: Buffer.ContentReady<Element,output>) -> Buffer.ContentReady<Element,pooled_model(~Element,~operation,pool,pad,origin,channels,input,middle,output)>:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  +pool = {Access.Pool{height,width,kernel,stride,padding,out_height,out_width} : Access.Pool}
  Sweeps.model_ready(~Element,~operation,U32.to_nat((channels * out_height * out_width : U32)),Access.SweepContext{Access.pool_rows(pool,origin),pad},
    Sweeps.model(~Element,~operation,U32.to_nat((channels * height * out_width : U32)),Access.SweepContext{Access.pool_columns(pool),pad},input,middle),output,evidence)

def pooled_ready(~Element: Data,~operation: Element -> Element -> Element,+pool: Access.Pool,+pad: Element,+origin: U32,+channels: U32,
  +input: Buffer.Model<Element>,+middle: Buffer.Model<Element>,+output: Buffer.Model<Element>,evidence: Buffer.Sized<Element,output>) ->
  Buffer.ContentReady<Element,Buffer.contents(~Element,BandPool.pooled(~Element,~operation,pool,pad,origin,channels,Buffer.pack(~Element,input),Buffer.pack(~Element,middle),Buffer.pack(~Element,output)))>:
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,BandPool.pooled(~Element,~operation,pool,pad,origin,channels,Buffer.pack(~Element,input),Buffer.pack(~Element,middle),Buffer.pack(~Element,output))),pooled_model(~Element,~operation,pool,pad,origin,channels,Buffer.content(~Element,input),Buffer.content(~Element,middle),Buffer.content(~Element,output)),packed_pooled(~Element,~operation,pool,pad,origin,channels,input,middle,output)) : Buffer.ContentReady<Element,_>
  model_ready(~Element,~operation,pool,pad,origin,channels,Buffer.content(~Element,input),Buffer.content(~Element,middle),Buffer.content(~Element,output),Buffer.content_ready(~Element,output,evidence))

def fresh_ready(~Element: Data,+fill: Element,+count: U32,evidence: Depth.Sized<count>) ->
  Buffer.ContentReady<Element,Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)}>:
  Depth.Sized{depth_ok,room} = evidence
  Buffer.content_ready(~Element,Buffer.fresh(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)),Buffer.fresh_ready(~Element,Tensors.depth_for(count),fill,U32.to_nat(count),depth_ok,room))

# Connect the actual windowed pool's fresh destination to content readiness.
# The caller must still derive this window count bound from its global guard.
def window_count(geometry: BandWindow.Geometry,window: BandWindow.RowWindow) -> U32:
  BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{first,rows},above,below,window} = window
  (channels * rows * output_width : U32)

def modeled_ready(~Element: Data,~operation: Element -> Element -> Element,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+first: U32,+rows: U32,+window: BandWindow.RowWindow,
  +core: Buffer.Content<Element>,+above: Buffer.Content<Element>,+below: Buffer.Content<Element>,
  evidence: Depth.Sized<window_count(geometry,window)>) ->
  Model.ContentReady(~Element,refinement_window_model(~Element,~operation,geometry,pad,fill,first,rows,window,core,above,below)):
  match geometry window:
    case BandWindow.Geometry{Bands.Geometry{batch,+channels,+height,+width},+kernel,+stride,+padding,+output_height,+output_width}
      BandWindow.RowWindow{BandWindow.Rows{+output_first,+output_rows},+above_range,+below_range,BandWindow.Rows{+window_first,+window_rows}}:
      +geometry = {BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,output_height,output_width} : BandWindow.Geometry}
      +window = {BandWindow.Rows{window_first,window_rows} : BandWindow.Rows}
      +middle_count = (channels * window_rows * output_width : U32)
      +count = (channels * output_rows * output_width : U32)
      model_ready(~Element,~operation,Access.Pool{window_rows,width,kernel,stride,padding,output_rows,output_width},pad,((output_first - first / height * output_height) * stride + first / height * height - window_first : U32),channels,TensorBandWindow.assembly_content(~Element,BandWindow.input_geometry(geometry),fill,window,BandWindow.Rows{first,rows},above_range,below_range,core,above,below),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(middle_count),fill),U32.to_nat(middle_count)},Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)},fresh_ready(~Element,fill,count,evidence))

def windowed_ready(~Element: Data,~operation: Element -> Element -> Element,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+first: U32,+rows: U32,+window: BandWindow.RowWindow,
  +core: Buffer.Model<Element>,+above: Buffer.Model<Element>,+below: Buffer.Model<Element>,evidence: Depth.Sized<window_count(geometry,window)>) ->
  Model.ContentReady(~Element,Model.contents(~Element,BandPool.windowed(~Element,~operation,geometry,pad,fill,first,rows,window,Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below)))):
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandPool.windowed(~Element,~operation,geometry,pad,fill,first,rows,window,Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below))),refinement_window_model(~Element,~operation,geometry,pad,fill,first,rows,window,Buffer.content(~Element,core),Buffer.content(~Element,above),Buffer.content(~Element,below)),refinement_windowed(~Element,~operation,geometry,pad,fill,first,rows,window,core,above,below)) : Model.ContentReady(~Element,_)
  modeled_ready(~Element,~operation,geometry,pad,fill,first,rows,window,Buffer.content(~Element,core),Buffer.content(~Element,above),Buffer.content(~Element,below),evidence)

# An active output leaf fits in one image; a positive batch includes that
# image. Inactive/zero-batch callers establish zero rows before using this law.
def rows_elements(+channels: Nat,+rows: Nat,+height: Nat,+width: Nat,
  within: {Nat.is_le(rows,height) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(channels,Nat.mul(rows,width)),Nat.mul(channels,Nat.mul(height,width))) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(channels,Nat.mul(rows,width)),Nat.mul(Nat.mul(rows,width),channels),RowMajor.multiply_commutative(channels,Nat.mul(rows,width))) :
    {Nat.is_le(_,Nat.mul(channels,Nat.mul(height,width))) == True{} : Bool}
  %Equal.sym(Nat,Nat.mul(channels,Nat.mul(height,width)),Nat.mul(Nat.mul(height,width),channels),RowMajor.multiply_commutative(channels,Nat.mul(height,width))) :
    {Nat.is_le(Nat.mul(Nat.mul(rows,width),channels),_) == True{} : Bool}
  RowMajor.mul_le(Nat.mul(rows,width),Nat.mul(height,width),channels,RowMajor.mul_le(rows,height,width,within))

def image_elements(+images: Nat,+elements: Nat,positive: {Nat.is_lt(0n,images) == True{} : Bool}) ->
  {Nat.is_le(elements,Nat.mul(images,elements)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(images,elements),Nat.mul(elements,images),RowMajor.multiply_commutative(images,elements)) : {Nat.is_le(elements,_) == True{} : Bool}
  RowMajor.product_covers(elements,images,positive)

def local_elements(+images: Nat,+channels: Nat,+height: Nat,+width: Nat,+rows: Nat,
  positive: {Nat.is_lt(0n,images) == True{} : Bool},within: {Nat.is_le(rows,height) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(channels,Nat.mul(rows,width)),Nat.mul(images,Nat.mul(channels,Nat.mul(height,width)))) == True{} : Bool}:
  Order.transitive(Nat.mul(channels,Nat.mul(rows,width)),Nat.mul(channels,Nat.mul(height,width)),Nat.mul(images,Nat.mul(channels,Nat.mul(height,width))),
    rows_elements(channels,rows,height,width,within),image_elements(images,Nat.mul(channels,Nat.mul(height,width)),positive))

# The accepted full pool output bounds each active window's fresh destination.
# An inactive leaf allocates count zero without requiring a positive batch.
def allocation_guards_output_geometry(geometry: BandWindow.Geometry) -> Bands.Geometry:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  Bands.Geometry{images,channels,output_height,output_width}

def count_bounded(+images: U32,+channels: U32,+height: U32,+width: U32,+rows: U32,+limits: Machine.Limits,
  positive_batch: {Nat.is_lt(0n,U32.to_nat(images)) == True{} : Bool},width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
  rows_inside: {Nat.is_le(U32.to_nat(rows),U32.to_nat(height)) == True{} : Bool},
  global_room: {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.is_le((channels * rows * width : U32),1073741824) == True{} : Bool}:
  +elements = Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))
  +room = Order.transitive(elements,Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits)),
    local_elements(U32.to_nat(images),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width),U32.to_nat(rows),positive_batch,rows_inside),global_room)
  +exact = Equal.trans(Nat,U32.to_nat((channels * rows * width : U32)),Nat.mul(Nat.mul(U32.to_nat(channels),U32.to_nat(rows)),U32.to_nat(width)),elements,
    Encoded.multiply_three(channels,rows,width,Machine.capacity(limits),Limits.capacity_fits(limits),width_positive,
      %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(channels),U32.to_nat(rows)),U32.to_nat(width)),elements,RowMajor.assoc(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(width))) :
        {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      room),RowMajor.assoc(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(width)))
  Arithmetic.count_within((channels * rows * width : U32),elements,Machine.capacity(limits),1073741824,exact,room,Limits.capacity_allocatable(limits))

def images(geometry: BandWindow.Geometry) -> U32:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  images

def from_window(+geometry: BandWindow.Geometry,+window: BandWindow.RowWindow,+limits: Machine.Limits,
  domain: BandGather.Domain<allocation_guards_output_geometry(geometry),limits,Machine.capacity(limits)>,
  positive_batch: {Nat.is_lt(0n,U32.to_nat(images(geometry))) == True{} : Bool},
  rows_inside: {Nat.is_le(U32.to_nat(TensorBandWindow.output_rows(window)),U32.to_nat(TensorBandWindow.output_height(geometry))) == True{} : Bool}) ->
  Depth.Sized<window_count(geometry,window)>:
  match geometry window:
    case BandWindow.Geometry{Bands.Geometry{+images,+channels,height,width},kernel,stride,padding,+output_height,+output_width}
      BandWindow.RowWindow{BandWindow.Rows{out_first,+out_rows},above,below,held}:
      BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
      Depth.bounded((channels * out_rows * output_width : U32),count_bounded(images,channels,output_height,output_width,out_rows,limits,positive_batch,width_positive,rows_inside,machine_room))

def active(+geometry: BandWindow.Geometry,+first: U32,+rows: U32,+limits: Machine.Limits,
  domain: BandGather.Domain<allocation_guards_output_geometry(geometry),limits,Machine.capacity(limits)>,positive_batch: {Nat.is_lt(0n,U32.to_nat(images(geometry))) == True{} : Bool}) ->
  Depth.Sized<window_count(geometry,BandWindow.window(geometry,first,rows))>:
  from_window(geometry,BandWindow.window(geometry,first,rows),limits,domain,positive_batch,TensorBandWindow.bounded(geometry,first,rows))

# The actual window apply tree for pooling. Scheduling changes neither the
# leaf's dense ordered pool nor its observed output. Structural mismatches
# retain the original core, exactly as the native ownership fallback does.
def leaf_model(~Element: Data,~operation: Element -> Element -> Element,context: BandPool.Context<Element>,first: U32,rows: U32,window: BandWindow.RowWindow,
  core: Buffer.Content<Element>,above: Buffer.Content<Element>,below: Buffer.Content<Element>) -> Model.Content<Element>:
  BandPool.Context{geometry,pad,fill,workers} = context
  refinement_window_model(~Element,~operation,geometry,pad,fill,first,rows,window,core,above,below)

def leaf_refines(~Element: Data,~operation: Element -> Element -> Element,+context: BandPool.Context<Element>,+first: U32,+rows: U32,+window: BandWindow.RowWindow,
  +core: Buffer.Model<Element>,+above: Buffer.Model<Element>,+below: Buffer.Model<Element>) ->
  {Model.contents(~Element,BandPool.leaf(~Element,~operation,context,first,rows,Buffer.pack(~Element,core),BandWindow.Halo{Buffer.pack(~Element,above),Buffer.pack(~Element,below),window})) ==
    leaf_model(~Element,~operation,context,first,rows,window,Buffer.content(~Element,core),Buffer.content(~Element,above),Buffer.content(~Element,below)) : Model.Content<Element>}:
  BandPool.Context{+geometry,+pad,+fill,workers} = context
  refinement_windowed(~Element,~operation,geometry,pad,fill,first,rows,window,core,above,below)

def pipeline_model(~Element: Data,~operation: Element -> Element -> Element,+context: BandPool.Context<Element>,layout: Bands.Plan,cores: Model.Model<Element>,halos: BandHalo.Owners<Element>) -> Model.Content<Element>:
  match layout cores:
    case Bands.Rows{+first,+rows} Model.Leaf{+core,+core_first,+core_rows}:
      match halos:
        case BandHalo.OwnedLeaf{above,below,window,halo_first,halo_rows}:
          leaf_model(~Element,~operation,context,first,rows,window,Buffer.content(~Element,core),Buffer.content(~Element,above),Buffer.content(~Element,below))
        case BandHalo.OwnedFork{left,right}: Model.content(~Element,Model.Leaf{core,core_first,core_rows})
    case Bands.Split{+left_plan,+right_plan} Model.Fork{+left,+right}:
      match halos:
        case BandHalo.OwnedFork{left_halo,right_halo}:
          Model.Joined{pipeline_model(~Element,~operation,context,left_plan,left,left_halo),pipeline_model(~Element,~operation,context,right_plan,right,right_halo)}
        case BandHalo.OwnedLeaf{above,below,window,first,rows}: Model.content(~Element,Model.Fork{left,right})
    case layout cores: Model.content(~Element,cores)

def kept(~Element: Data,+pipeline_model: Model.Model<Element>,-bands: Bands.Bands(~Element),equation: {bands == Model.pack(~Element,pipeline_model) : Bands.Bands(~Element)}) ->
  {Model.contents(~Element,bands) == Model.content(~Element,pipeline_model) : Model.Content<Element>}:
  %Equal.sym(Bands.Bands(~Element),bands,Model.pack(~Element,pipeline_model),equation) : {Model.contents(~Element,_) == Model.content(~Element,pipeline_model) : Model.Content<Element>}
  Model.packed_contents(~Element,pipeline_model)

def apply_mode(~Element: Data,~operation: Element -> Element -> Element,+context: BandPool.Context<Element>,+layout: Bands.Plan,+mode: Bool,+enabled: Bool,+geometry: Bands.Geometry,
  cores: Model.Model<Element>,halos: BandHalo.Owners<Element>) ->
  {Model.contents(~Element,BandWindow.apply(~Element,~BandPool.Context<Element>,~BandPool.leaf(~Element,~operation),layout,mode,enabled,geometry,context,Model.pack(~Element,cores),BandHalo.pack_owners(~Element,halos))) ==
    pipeline_model(~Element,~operation,context,layout,cores,halos) : Model.Content<Element>}:
  match layout cores:
    case Bands.Rows{+first,+rows} Model.Leaf{+core,+core_first,+core_rows}:
      match halos:
        case BandHalo.OwnedLeaf{+above,+below,+window,+halo_first,+halo_rows}:
          leaf_refines(~Element,~operation,context,first,rows,window,core,above,below)
        case BandHalo.OwnedFork{+left,+right}:
          kept(~Element,Model.Leaf{core,core_first,core_rows},
            BandWindow.apply(~Element,~BandPool.Context<Element>,~BandPool.leaf(~Element,~operation),Bands.Rows{first,rows},mode,enabled,geometry,context,Model.pack(~Element,Model.Leaf{core,core_first,core_rows}),BandHalo.pack_owners(~Element,BandHalo.OwnedFork{left,right})),
            Apply.leaf_forked_halos(~Element,~BandPool.Context<Element>,~BandPool.leaf(~Element,~operation),first,rows,mode,enabled,geometry,context,core,core_first,core_rows,left,right))
    case Bands.Split{+left_plan,+right_plan} Model.Fork{+left,+right}:
      match halos:
        case BandHalo.OwnedFork{+left_halo,+right_halo}:
          %Equal.sym(Bands.Bands(~Element),BandWindow.apply(~Element,~BandPool.Context<Element>,~BandPool.leaf(~Element,~operation),Bands.Split{left_plan,right_plan},mode,enabled,geometry,context,
              Model.pack(~Element,Model.Fork{left,right}),BandHalo.pack_owners(~Element,BandHalo.OwnedFork{left_halo,right_halo})),
            Partition.Fork{BandWindow.apply(~Element,~BandPool.Context<Element>,~BandPool.leaf(~Element,~operation),left_plan,enabled && Bands.same_image(geometry,left_plan),enabled,geometry,context,Model.pack(~Element,left),BandHalo.pack_owners(~Element,left_halo)),
              BandWindow.apply(~Element,~BandPool.Context<Element>,~BandPool.leaf(~Element,~operation),right_plan,enabled && Bands.same_image(geometry,right_plan),enabled,geometry,context,Model.pack(~Element,right),BandHalo.pack_owners(~Element,right_halo))},
            Apply.fork(~Element,~BandPool.Context<Element>,~BandPool.leaf(~Element,~operation),left_plan,right_plan,mode,enabled,geometry,context,left,right,left_halo,right_halo)) :
            {Model.contents(~Element,_) == Model.Joined{pipeline_model(~Element,~operation,context,left_plan,left,left_halo),pipeline_model(~Element,~operation,context,right_plan,right,right_halo)} : Model.Content<Element>}
          %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandWindow.apply(~Element,~BandPool.Context<Element>,~BandPool.leaf(~Element,~operation),left_plan,enabled && Bands.same_image(geometry,left_plan),enabled,geometry,context,Model.pack(~Element,left),BandHalo.pack_owners(~Element,left_halo))),
            pipeline_model(~Element,~operation,context,left_plan,left,left_halo),apply_mode(~Element,~operation,context,left_plan,enabled && Bands.same_image(geometry,left_plan),enabled,geometry,left,left_halo)) :
            {Model.Joined{_,Model.contents(~Element,BandWindow.apply(~Element,~BandPool.Context<Element>,~BandPool.leaf(~Element,~operation),right_plan,enabled && Bands.same_image(geometry,right_plan),enabled,geometry,context,Model.pack(~Element,right),BandHalo.pack_owners(~Element,right_halo)))} ==
              Model.Joined{pipeline_model(~Element,~operation,context,left_plan,left,left_halo),pipeline_model(~Element,~operation,context,right_plan,right,right_halo)} : Model.Content<Element>}
          %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandWindow.apply(~Element,~BandPool.Context<Element>,~BandPool.leaf(~Element,~operation),right_plan,enabled && Bands.same_image(geometry,right_plan),enabled,geometry,context,Model.pack(~Element,right),BandHalo.pack_owners(~Element,right_halo))),
            pipeline_model(~Element,~operation,context,right_plan,right,right_halo),apply_mode(~Element,~operation,context,right_plan,enabled && Bands.same_image(geometry,right_plan),enabled,geometry,right,right_halo)) :
            {Model.Joined{pipeline_model(~Element,~operation,context,left_plan,left,left_halo),_} == Model.Joined{pipeline_model(~Element,~operation,context,left_plan,left,left_halo),pipeline_model(~Element,~operation,context,right_plan,right,right_halo)} : Model.Content<Element>}
          {==}
        case BandHalo.OwnedLeaf{+above,+below,+window,+first,+rows}:
          kept(~Element,Model.Fork{left,right},
            BandWindow.apply(~Element,~BandPool.Context<Element>,~BandPool.leaf(~Element,~operation),Bands.Split{left_plan,right_plan},mode,enabled,geometry,context,Model.pack(~Element,Model.Fork{left,right}),BandHalo.pack_owners(~Element,BandHalo.OwnedLeaf{above,below,window,first,rows})),
            Apply.fork_leaf_halos(~Element,~BandPool.Context<Element>,~BandPool.leaf(~Element,~operation),left_plan,right_plan,mode,enabled,geometry,context,left,right,above,below,window,first,rows))
    case Bands.Rows{+first,+rows} Model.Fork{+left,+right}:
      +halos = halos
      kept(~Element,Model.Fork{left,right},
        BandWindow.apply(~Element,~BandPool.Context<Element>,~BandPool.leaf(~Element,~operation),Bands.Rows{first,rows},mode,enabled,geometry,context,Model.pack(~Element,Model.Fork{left,right}),BandHalo.pack_owners(~Element,halos)),
        Apply.rows_fork(~Element,~BandPool.Context<Element>,~BandPool.leaf(~Element,~operation),first,rows,mode,enabled,geometry,context,left,right,halos))
    case Bands.Split{+left_plan,+right_plan} Model.Leaf{+core,+first,+rows}:
      +halos = halos
      kept(~Element,Model.Leaf{core,first,rows},
        BandWindow.apply(~Element,~BandPool.Context<Element>,~BandPool.leaf(~Element,~operation),Bands.Split{left_plan,right_plan},mode,enabled,geometry,context,Model.pack(~Element,Model.Leaf{core,first,rows}),BandHalo.pack_owners(~Element,halos)),
        Apply.split_leaf(~Element,~BandPool.Context<Element>,~BandPool.leaf(~Element,~operation),left_plan,right_plan,mode,enabled,geometry,context,core,first,rows,halos))

def pipeline_prepared(~Element: Data,~operation: Element -> Element -> Element,+context: BandPool.Context<Element>,+layout: Bands.Plan,+cores: Model.Model<Element>,+halos: BandHalo.Owners<Element>) ->
  {Model.contents(~Element,BandPool.prepared(~Element,~operation,context,layout,(Model.pack(~Element,cores),BandHalo.pack_owners(~Element,halos)))) ==
    pipeline_model(~Element,~operation,context,layout,cores,halos) : Model.Content<Element>}:
  BandPool.Context{+window,+pad,+fill,+workers} = context
  +enabled = BandPool.parallel(window,workers,BandPool.plan_size(layout,window))
  apply_mode(~Element,~operation,BandPool.Context{window,pad,fill,workers},layout,
    enabled && Bands.same_image(BandWindow.input_geometry(window),layout),enabled,BandWindow.input_geometry(window),cores,halos)

# Actual preparation and planning refine the content model of band pooling.
# Halo contents are obtained from BandWindow.prepare, never supplied as an oracle.
def execution_model(~Element: Data,~operation: Element -> Element -> Element,+context: BandPool.Context<Element>,layout: Bands.Plan,cores: Model.Content<Element>,halos: BandHalo.Tree<Element>) -> Model.Content<Element>:
  match layout cores:
    case Bands.Rows{+first,+rows} Model.RowBand{+core,+core_first,+core_rows}:
      match halos:
        case BandHalo.Leaf{above,below,window,halo_first,halo_rows}:
          leaf_model(~Element,~operation,context,first,rows,window,core,above,below)
        case BandHalo.Fork{left,right}: Model.RowBand{core,core_first,core_rows}
    case Bands.Split{+left_plan,+right_plan} Model.Joined{+left,+right}:
      match halos:
        case BandHalo.Fork{left_halo,right_halo}:
          Model.Joined{execution_model(~Element,~operation,context,left_plan,left,left_halo),execution_model(~Element,~operation,context,right_plan,right,right_halo)}
        case BandHalo.Leaf{above,below,window,first,rows}: Model.Joined{left,right}
    case layout cores: cores

def modeled_owners(~Element: Data,~operation: Element -> Element -> Element,+context: BandPool.Context<Element>,+layout: Bands.Plan,cores: Model.Model<Element>,halos: BandHalo.Owners<Element>) ->
  {pipeline_model(~Element,~operation,context,layout,cores,halos) == execution_model(~Element,~operation,context,layout,Model.content(~Element,cores),BandHalo.owner_contents(~Element,halos)) : Model.Content<Element>}:
  match layout cores:
    case Bands.Rows{first,rows} Model.Leaf{core,core_first,core_rows}:
      match halos:
        case BandHalo.OwnedLeaf{above,below,window,halo_first,halo_rows}: {==}
        case BandHalo.OwnedFork{left,right}: {==}
    case Bands.Split{+left_plan,+right_plan} Model.Fork{+left,+right}:
      match halos:
        case BandHalo.OwnedFork{+left_halo,+right_halo}:
          %modeled_owners(~Element,~operation,context,left_plan,left,left_halo) :
            {Model.Joined{pipeline_model(~Element,~operation,context,left_plan,left,left_halo),pipeline_model(~Element,~operation,context,right_plan,right,right_halo)} ==
              Model.Joined{_,execution_model(~Element,~operation,context,right_plan,Model.content(~Element,right),BandHalo.owner_contents(~Element,right_halo))} : Model.Content<Element>}
          %modeled_owners(~Element,~operation,context,right_plan,right,right_halo) :
            {Model.Joined{pipeline_model(~Element,~operation,context,left_plan,left,left_halo),pipeline_model(~Element,~operation,context,right_plan,right,right_halo)} ==
              Model.Joined{pipeline_model(~Element,~operation,context,left_plan,left,left_halo),_} : Model.Content<Element>}
          {==}
        case BandHalo.OwnedLeaf{above,below,window,first,rows}: {==}
    case Bands.Rows{first,rows} Model.Fork{left,right}: {==}
    case Bands.Split{left,right} Model.Leaf{core,first,rows}: {==}

def prepared_model(~Element: Data,~operation: Element -> Element -> Element,+context: BandPool.Context<Element>,+layout: Bands.Plan,owners: BandHalo.Content<Element>) -> Model.Content<Element>:
  BandHalo.Content{cores,halos} = owners
  execution_model(~Element,~operation,context,layout,cores,halos)

def prepared_observed(~Element: Data,~operation: Element -> Element -> Element,+context: BandPool.Context<Element>,+layout: Bands.Plan,-cores: Bands.Bands(~Element),-halos: BandWindow.Halos(~Element),
  cores_observation: Model.Observation<Element,cores>,halos_observation: BandHalo.OwnerObservation<Element,halos>) ->
  {Model.contents(~Element,BandPool.prepared(~Element,~operation,context,layout,(cores,halos))) ==
    prepared_model(~Element,~operation,context,layout,BandHalo.refinement_contents(~Element,(cores,halos))) : Model.Content<Element>}:
  Model.Observation{+core_model,core_equal} = cores_observation
  BandHalo.OwnerObservation{+halo_model,halo_equal} = halos_observation
  %core_equal : {Model.contents(~Element,BandPool.prepared(~Element,~operation,context,layout,(_,halos))) ==
    prepared_model(~Element,~operation,context,layout,BandHalo.refinement_contents(~Element,(_,halos))) : Model.Content<Element>}
  %halo_equal : {Model.contents(~Element,BandPool.prepared(~Element,~operation,context,layout,(Model.pack(~Element,core_model),_))) ==
    prepared_model(~Element,~operation,context,layout,BandHalo.refinement_contents(~Element,(Model.pack(~Element,core_model),_))) : Model.Content<Element>}
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,core_model)),Model.content(~Element,core_model),Model.packed_contents(~Element,core_model)) :
    {Model.contents(~Element,BandPool.prepared(~Element,~operation,context,layout,(Model.pack(~Element,core_model),BandHalo.pack_owners(~Element,halo_model)))) ==
      execution_model(~Element,~operation,context,layout,_,BandHalo.reflect(~Element,BandHalo.pack_owners(~Element,halo_model))) : Model.Content<Element>}
  %Equal.sym(BandHalo.Tree<Element>,BandHalo.reflect(~Element,BandHalo.pack_owners(~Element,halo_model)),BandHalo.owner_contents(~Element,halo_model),BandHalo.packed_owners(~Element,halo_model)) :
    {Model.contents(~Element,BandPool.prepared(~Element,~operation,context,layout,(Model.pack(~Element,core_model),BandHalo.pack_owners(~Element,halo_model)))) ==
      execution_model(~Element,~operation,context,layout,Model.content(~Element,core_model),_) : Model.Content<Element>}
  Equal.trans(Model.Content<Element>,
    Model.contents(~Element,BandPool.prepared(~Element,~operation,context,layout,(Model.pack(~Element,core_model),BandHalo.pack_owners(~Element,halo_model)))),
    pipeline_model(~Element,~operation,context,layout,core_model,halo_model),
    execution_model(~Element,~operation,context,layout,Model.content(~Element,core_model),BandHalo.owner_contents(~Element,halo_model)),
    pipeline_prepared(~Element,~operation,context,layout,core_model,halo_model),modeled_owners(~Element,~operation,context,layout,core_model,halo_model))

def execution_prepared(~Element: Data,~operation: Element -> Element -> Element,+context: BandPool.Context<Element>,+layout: Bands.Plan,owners: Bands.Bands(~Element) & BandWindow.Halos(~Element)) ->
  {Model.contents(~Element,BandPool.prepared(~Element,~operation,context,layout,owners)) == prepared_model(~Element,~operation,context,layout,BandHalo.refinement_contents(~Element,owners)) : Model.Content<Element>}:
  (cores,halos) = owners
  prepared_observed(~Element,~operation,context,layout,cores,halos,Model.observe(~Element,cores),BandHalo.observe_owners(~Element,halos))

def planned_model(~Element: Data,~operation: Element -> Element -> Element,context: BandPool.Context<Element>,+layout: Bands.Plan,+cores: Model.Content<Element>) -> Model.Content<Element>:
  BandPool.Context{+geometry,+pad,+fill,+workers} = context
  execution_model(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},layout,cores,BandHalo.model(~Element,geometry,pad,layout,cores))

def planned_packed(~Element: Data,~operation: Element -> Element -> Element,+context: BandPool.Context<Element>,+layout: Bands.Plan,+cores: Model.Model<Element>) ->
  {Model.contents(~Element,BandPool.planned(~Element,~operation,context,(Model.pack(~Element,cores),layout))) == planned_model(~Element,~operation,context,layout,Model.contents(~Element,Model.pack(~Element,cores))) : Model.Content<Element>}:
  BandPool.Context{+geometry,+pad,+fill,+workers} = context
  +context = {BandPool.Context{geometry,pad,fill,workers} : BandPool.Context<Element>}
  Equal.trans(Model.Content<Element>,Model.contents(~Element,BandPool.planned(~Element,~operation,context,(Model.pack(~Element,cores),layout))),
    prepared_model(~Element,~operation,context,layout,BandHalo.refinement_contents(~Element,BandWindow.prepare(~Element,geometry,pad,layout,Model.pack(~Element,cores)))),
    planned_model(~Element,~operation,context,layout,Model.contents(~Element,Model.pack(~Element,cores))),
    execution_prepared(~Element,~operation,context,layout,BandWindow.prepare(~Element,geometry,pad,layout,Model.pack(~Element,cores))),
    Equal.cong(BandHalo.Content<Element>,Model.Content<Element>,owners => prepared_model(~Element,~operation,context,layout,owners),
      BandHalo.refinement_contents(~Element,BandWindow.prepare(~Element,geometry,pad,layout,Model.pack(~Element,cores))),
      BandHalo.Content{Model.contents(~Element,Model.pack(~Element,cores)),BandHalo.model(~Element,geometry,pad,layout,Model.contents(~Element,Model.pack(~Element,cores)))},BandHalo.prepare(~Element,geometry,pad,layout,Model.pack(~Element,cores))))

def planned_observed(~Element: Data,~operation: Element -> Element -> Element,+context: BandPool.Context<Element>,+layout: Bands.Plan,-cores: Bands.Bands(~Element),observation: Model.Observation<Element,cores>) ->
  {Model.contents(~Element,BandPool.planned(~Element,~operation,context,(cores,layout))) == planned_model(~Element,~operation,context,layout,Model.contents(~Element,cores)) : Model.Content<Element>}:
  Model.Observation{+source,equation} = observation
  %equation : {Model.contents(~Element,BandPool.planned(~Element,~operation,context,(_,layout))) == planned_model(~Element,~operation,context,layout,Model.contents(~Element,_)) : Model.Content<Element>}
  planned_packed(~Element,~operation,context,layout,source)

def planned(~Element: Data,~operation: Element -> Element -> Element,+context: BandPool.Context<Element>,+layout: Bands.Plan,cores: Bands.Bands(~Element)) ->
  {Model.contents(~Element,BandPool.planned(~Element,~operation,context,(cores,layout))) == planned_model(~Element,~operation,context,layout,Model.contents(~Element,cores)) : Model.Content<Element>}:
  planned_observed(~Element,~operation,context,layout,cores,Model.observe(~Element,cores))

def execution_execute(~Element: Data,~operation: Element -> Element -> Element,+context: BandPool.Context<Element>,+cores: Model.Model<Element>) ->
  {Model.contents(~Element,BandPool.execute(~Element,~operation,context,Model.pack(~Element,cores))) ==
    planned_model(~Element,~operation,context,Model.layout(~Element,cores),Model.content(~Element,cores)) : Model.Content<Element>}:
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,cores)),(Model.pack(~Element,cores),Model.layout(~Element,cores)),Model.plan(~Element,cores)) :
    {Model.contents(~Element,BandPool.planned(~Element,~operation,context,_)) == planned_model(~Element,~operation,context,Model.layout(~Element,cores),Model.content(~Element,cores)) : Model.Content<Element>}
  %Model.packed_contents(~Element,cores) :
    {Model.contents(~Element,BandPool.planned(~Element,~operation,context,(Model.pack(~Element,cores),Model.layout(~Element,cores)))) == planned_model(~Element,~operation,context,Model.layout(~Element,cores),_) : Model.Content<Element>}
  planned_packed(~Element,~operation,context,Model.layout(~Element,cores),cores)

# The values of band pooling. Every output cell of every band holds the pool
# of its pixel over the whole logical input: the fold, from pad, of the k
# lines of its window, each the fold, from pad, of its k samples; a line or a
# sample outside the image is pad. The leaf sweeps the columns of the rows it
# assembled and then the rows of those planes; the stored rows are a window
# of one image, and the caller states which rows of the image they are.
def global_context(~Element: Data,~Logical: Data,logical: Logical,geometry: BandWindow.Geometry,+pad: Element) -> Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>:
  BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  PoolDense.context(~Element,~Logical,logical,Access.Pool{height,width,kernel,stride,padding,output_height,output_width},pad)

def values_output_geometry(geometry: BandWindow.Geometry) -> Bands.Geometry:
  BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  Bands.Geometry{batch,channels,output_height,output_width}

# The column-swept planes of the whole input.
def planes_geometry(geometry: BandWindow.Geometry) -> Bands.Geometry:
  BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  Bands.Geometry{batch,channels,height,output_width}

# The row of the assembled window that the first tap of the first output row
# of a leaf reads, before padding.
def window_origin(geometry: BandWindow.Geometry,+first: U32,window: BandWindow.RowWindow) -> U32:
  BandWindow.Geometry{Bands.Geometry{batch,channels,+height,width},kernel,+stride,padding,+output_height,output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{+output_first,output_rows},above,below,BandWindow.Rows{+window_first,window_rows}} = window
  ((output_first - first / height * output_height) * stride + first / height * height - window_first : U32)

def assembled_input(~Element: Data,+geometry: BandWindow.Geometry,fill: Element,first: U32,rows: U32,window: BandWindow.RowWindow,
  core: Buffer.Content<Element>,above: Buffer.Content<Element>,below: Buffer.Content<Element>) -> Buffer.Content<Element>:
  BandWindow.RowWindow{output,above_range,below_range,window_range} = window
  TensorBandWindow.assembly_content(~Element,BandWindow.input_geometry(geometry),fill,window_range,BandWindow.Rows{first,rows},above_range,below_range,core,above,below)

# The planes a window sweeps along the columns: one row per assembled row.
def middle_count(geometry: BandWindow.Geometry,window: BandWindow.RowWindow) -> U32:
  BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  BandWindow.RowWindow{output,above,below,BandWindow.Rows{first,rows}} = window
  (channels * rows * output_width : U32)

def middle_cells(geometry: BandWindow.Geometry,window: BandWindow.RowWindow) -> Nat:
  BandWindow.RowWindow{output,above,below,BandWindow.Rows{first,+rows}} = window
  BandValues.cell_count(planes_geometry(geometry),rows)

# The number of cells of the output leaf of a window.
def window_cells(geometry: BandWindow.Geometry,window: BandWindow.RowWindow) -> Nat:
  BandWindow.RowWindow{BandWindow.Rows{first,+rows},above,below,input} = window
  BandValues.cell_count(values_output_geometry(geometry),rows)

def WindowRooms(geometry: BandWindow.Geometry,window: BandWindow.RowWindow,limits: Machine.Limits) -> Data:
  BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{output_first,output_rows},above,below,BandWindow.Rows{window_first,window_rows}} = window
  BandSweeps.Rooms<batch,channels,height,width,output_height,output_width,window_first,window_rows,output_first,output_rows,limits>

# For every output row of the leaf and every tap: the assembled row it reads
# is valid exactly when the row of the image is, and then it is that row.
def WindowRows(+geometry: BandWindow.Geometry,+first: U32,+window: BandWindow.RowWindow) -> Data:
  BandWindow.Geometry{Bands.Geometry{batch,channels,+height,width},+kernel,+stride,+padding,+output_height,output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{+output_first,+output_rows},above,below,BandWindow.Rows{+window_first,+window_rows}} = window
  BandSweeps.RowFacts(kernel,stride,padding,((output_first - first / height * output_height) * stride + first / height * height - window_first : U32),window_first,window_rows,height,output_height,output_first,output_rows)

def NativeWindowValues(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: BandWindow.Geometry,+fill: Element,+first: U32,+rows: U32,
  window: BandWindow.RowWindow,+core: Buffer.Model<Element>,+above: Buffer.Model<Element>,+below: Buffer.Model<Element>) -> Data:
  BandWindow.RowWindow{output,above_range,below_range,+window_range} = window
  BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),window_range,
    Buffer.contents(~Element,BandWindow.assemble(~Element,fill,BandWindow.input_geometry(geometry),window_range,BandWindow.Rows{first,rows},above_range,below_range,
      Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below))))

def assembled_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Buffer.Content<Element>,
  evidence: BandHalo.RangeValues(~Element,~Logical,~original,logical,geometry,BandWindow.Rows{first,rows},input)) ->
  BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.RowBand{input,first,rows}):
  Buffer.Content{tree,length} = input
  Shape.Both{ready,Shape.Both{metadata,values}} = evidence
  values

# What a window supplies: both fresh destinations can be allocated and hold
# their cell counts, the assembled rows hold the logical input, the bounds,
# and the rows read by every tap.
def WindowFacts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+first: U32,+rows: U32,
  +window: BandWindow.RowWindow,+core: Buffer.Content<Element>,+above: Buffer.Content<Element>,+below: Buffer.Content<Element>,+limits: Machine.Limits) -> Data:
  BandWindow.RowWindow{BandWindow.Rows{+output_first,+output_rows},above_range,below_range,BandWindow.Rows{+window_first,+window_rows}} = window
  +input = assembled_input(~Element,geometry,fill,first,rows,window,core,above,below)
  Shape.Both<Depth.Sized<middle_count(geometry,window)>,Shape.Both<Depth.Sized<window_count(geometry,window)>,
    Shape.Both<{U32.to_nat(middle_count(geometry,window)) == middle_cells(geometry,window) : Nat},
      Shape.Both<{U32.to_nat(window_count(geometry,window)) == window_cells(geometry,window) : Nat},
        Shape.Both<BandValues.Values(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),Model.RowBand{input,window_first,window_rows}),
          Shape.Both<WindowRooms(geometry,window,limits),WindowRows(geometry,first,window)>>>>>>

# TensorBandWindow.assembly_window constructs the native range relation from core/halo Values.
# This adapter transports that relation along the accepted assembly refinement
# into WindowFacts, keeping allocation and coordinate producer facts explicit.
def window_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+first: U32,+rows: U32,
  +window: BandWindow.RowWindow,+core: Buffer.Model<Element>,+above: Buffer.Model<Element>,+below: Buffer.Model<Element>,+limits: Machine.Limits,
  middle_allocation: Depth.Sized<middle_count(geometry,window)>,allocation: Depth.Sized<window_count(geometry,window)>,
  middle_exact: {U32.to_nat(middle_count(geometry,window)) == middle_cells(geometry,window) : Nat},
  count_exact: {U32.to_nat(window_count(geometry,window)) == window_cells(geometry,window) : Nat},
  range_values: NativeWindowValues(~Element,~Logical,~original,logical,geometry,fill,first,rows,window,core,above,below),
  rooms: WindowRooms(geometry,window,limits),samples: WindowRows(geometry,first,window)) ->
  WindowFacts(~Element,~Logical,~original,logical,geometry,pad,fill,first,rows,window,Buffer.content(~Element,core),Buffer.content(~Element,above),Buffer.content(~Element,below),limits):
  BandWindow.RowWindow{output,+above_range,+below_range,+window_range} = window
  BandWindow.Rows{output_first,output_rows} = output
  BandWindow.Rows{window_first,window_rows} = window_range
  Shape.Both{middle_allocation,Shape.Both{allocation,Shape.Both{middle_exact,Shape.Both{count_exact,Shape.Both{
    %TensorBandWindow.assemble_contents(~Element,BandWindow.input_geometry(geometry),fill,window_range,BandWindow.Rows{first,rows},above_range,below_range,core,above,below) :
      BandValues.Values(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),Model.RowBand{_,window_first,window_rows})
    assembled_values(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),window_first,window_rows,
      Buffer.contents(~Element,BandWindow.assemble(~Element,fill,BandWindow.input_geometry(geometry),window_range,BandWindow.Rows{first,rows},above_range,below_range,Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below))),
      range_values),Shape.Both{rooms,samples}}}}}}

# The two sweeps into fresh destinations, for any content of the stored rows.
def values_fresh(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+pad: Element,+fill: Element,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +kernel: U32,+stride: U32,+padding: U32,+out_height: U32,+out_width: U32,+window_first: U32,+window_rows: U32,+output_first: U32,+output_rows: U32,+origin: U32,
  +input: Buffer.Content<Element>,+limits: Machine.Limits,
  middle_allocation: Depth.Sized<(channels * window_rows * out_width : U32)>,allocation: Depth.Sized<(channels * output_rows * out_width : U32)>,
  +middle_exact: {U32.to_nat((channels * window_rows * out_width : U32)) == BandValues.cell_count(Bands.Geometry{batch,channels,height,out_width},window_rows) : Nat},
  +count_exact: {U32.to_nat((channels * output_rows * out_width : U32)) == BandValues.cell_count(Bands.Geometry{batch,channels,out_height,out_width},output_rows) : Nat},
  values: BandValues.Values(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},Model.RowBand{input,window_first,window_rows}),
  rooms: BandSweeps.Rooms<batch,channels,height,width,out_height,out_width,window_first,window_rows,output_first,output_rows,limits>,
  +facts: BandSweeps.RowFacts(kernel,stride,padding,origin,window_first,window_rows,height,out_height,output_first,output_rows)) ->
  BandValues.Values(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),
    Sweeps.Swept{Sweeps.Swept{logical,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad},Access.Sweep{height,out_width,kernel,stride,padding,out_height,0},pad},
    Bands.Geometry{batch,channels,out_height,out_width},
    Model.RowBand{pooled_model(~Element,~operation,Access.Pool{window_rows,width,kernel,stride,padding,output_rows,out_width},pad,origin,channels,input,
      Buffer.Content{Tree.fresh(Element,Tensors.depth_for((channels * window_rows * out_width : U32)),fill),U32.to_nat((channels * window_rows * out_width : U32))},
      Buffer.Content{Tree.fresh(Element,Tensors.depth_for((channels * output_rows * out_width : U32)),fill),U32.to_nat((channels * output_rows * out_width : U32))}),output_first,output_rows}):
  Buffer.Content{+input_tree,+input_length} = input
  Depth.Sized{+middle_depth_ok,+middle_room} = middle_allocation
  Depth.Sized{+depth_ok,+room} = allocation
  +middle_count = (channels * window_rows * out_width : U32)
  +count = (channels * output_rows * out_width : U32)
  BandSweeps.pooled_leaf(~Element,~Logical,~original,~operation,logical,pad,batch,channels,height,width,kernel,stride,padding,out_height,out_width,window_first,window_rows,output_first,output_rows,origin,
    input_tree,input_length,Buffer.fresh(~Element,Tensors.depth_for(middle_count),fill,U32.to_nat(middle_count)),Buffer.fresh(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)),limits,rooms,
    Buffer.fresh_ready(~Element,Tensors.depth_for(middle_count),fill,U32.to_nat(middle_count),middle_depth_ok,middle_room),
    Buffer.fresh_ready(~Element,Tensors.depth_for(count),fill,U32.to_nat(count),depth_ok,room),
    values,middle_exact,count_exact,facts)

def values_window_model(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+first: U32,+rows: U32,
  +window: BandWindow.RowWindow,+core: Buffer.Content<Element>,+above: Buffer.Content<Element>,+below: Buffer.Content<Element>,+limits: Machine.Limits,
  facts: WindowFacts(~Element,~Logical,~original,logical,geometry,pad,fill,first,rows,window,core,above,below,limits)) ->
  BandValues.Values(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),global_context(~Element,~Logical,logical,geometry,pad),values_output_geometry(geometry),
    refinement_window_model(~Element,~operation,geometry,pad,fill,first,rows,window,core,above,below)):
  BandWindow.Geometry{Bands.Geometry{+batch,+channels,+height,+width},+kernel,+stride,+padding,+output_height,+output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{+output_first,+output_rows},+above_range,+below_range,BandWindow.Rows{+window_first,+window_rows}} = window
  Shape.Both{middle_allocation,Shape.Both{allocation,Shape.Both{+middle_exact,Shape.Both{+count_exact,Shape.Both{range_values,Shape.Both{rooms,+samples}}}}}} = facts
  values_fresh(~Element,~Logical,~original,~operation,logical,pad,fill,batch,channels,height,width,kernel,stride,padding,output_height,output_width,window_first,window_rows,output_first,output_rows,
    ((output_first - first / height * output_height) * stride + first / height * height - window_first : U32),
    TensorBandWindow.assembly_content(~Element,Bands.Geometry{batch,channels,height,width},fill,BandWindow.Rows{window_first,window_rows},BandWindow.Rows{first,rows},above_range,below_range,core,above,below),limits,
    middle_allocation,allocation,middle_exact,count_exact,range_values,rooms,samples)

def values_windowed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+first: U32,+rows: U32,
  +window: BandWindow.RowWindow,+core: Buffer.Model<Element>,+above: Buffer.Model<Element>,+below: Buffer.Model<Element>,+limits: Machine.Limits,
  facts: WindowFacts(~Element,~Logical,~original,logical,geometry,pad,fill,first,rows,window,Buffer.content(~Element,core),Buffer.content(~Element,above),Buffer.content(~Element,below),limits)) ->
  BandValues.Values(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),global_context(~Element,~Logical,logical,geometry,pad),values_output_geometry(geometry),
    Model.contents(~Element,BandPool.windowed(~Element,~operation,geometry,pad,fill,first,rows,window,Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below)))):
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandPool.windowed(~Element,~operation,geometry,pad,fill,first,rows,window,Buffer.pack(~Element,core),Buffer.pack(~Element,above),Buffer.pack(~Element,below))),
    refinement_window_model(~Element,~operation,geometry,pad,fill,first,rows,window,Buffer.content(~Element,core),Buffer.content(~Element,above),Buffer.content(~Element,below)),
    refinement_windowed(~Element,~operation,geometry,pad,fill,first,rows,window,core,above,below)) :
    BandValues.Values(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),global_context(~Element,~Logical,logical,geometry,pad),values_output_geometry(geometry),_)
  values_window_model(~Element,~Logical,~original,~operation,logical,geometry,pad,fill,first,rows,window,Buffer.content(~Element,core),Buffer.content(~Element,above),Buffer.content(~Element,below),limits,facts)

def empty_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+first: U32,+rows: U32,+content: Buffer.Content<Element>,
  empty: {BandValues.cell_count(geometry,rows) == 0n : Nat}) -> BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.RowBand{content,first,rows}):
  Buffer.Content{+tree,+length} = content
  BandValues.empty_leaf(~Element,~Logical,~original,logical,geometry,first,rows,tree,length,empty)

# A window without output cells has no output samples: the relation holds
# independently of halo row metadata or input addresses.
def zero_window_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+first: U32,+rows: U32,
  +window: BandWindow.RowWindow,+core: Buffer.Content<Element>,+above: Buffer.Content<Element>,+below: Buffer.Content<Element>,zero: {window_cells(geometry,window) == 0n : Nat}) ->
  BandValues.Values(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),global_context(~Element,~Logical,logical,geometry,pad),values_output_geometry(geometry),
    refinement_window_model(~Element,~operation,geometry,pad,fill,first,rows,window,core,above,below)):
  BandWindow.Geometry{Bands.Geometry{+batch,+channels,+height,+width},+kernel,+stride,+padding,+output_height,+output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{+output_first,+output_rows},+above_range,+below_range,BandWindow.Rows{+window_first,+window_rows}} = window
  +geometry = {BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,output_height,output_width} : BandWindow.Geometry}
  +middle_count = (channels * window_rows * output_width : U32)
  +count = (channels * output_rows * output_width : U32)
  empty_values(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),global_context(~Element,~Logical,logical,geometry,pad),values_output_geometry(geometry),output_first,output_rows,
    pooled_model(~Element,~operation,Access.Pool{window_rows,width,kernel,stride,padding,output_rows,output_width},pad,((output_first - first / height * output_height) * stride + first / height * height - window_first : U32),channels,
      TensorBandWindow.assembly_content(~Element,Bands.Geometry{batch,channels,height,width},fill,BandWindow.Rows{window_first,window_rows},BandWindow.Rows{first,rows},above_range,below_range,core,above,below),
      Buffer.Content{Tree.fresh(Element,Tensors.depth_for(middle_count),fill),U32.to_nat(middle_count)},Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)}),zero)

# A successful semantic call must have matching plan/core/halo structures.
# Fallback branches remain in the execution model; their producer contract is
# Empty because an unchanged input is not generally a pooled output.
def TreeFacts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+context: BandPool.Context<Element>,+limits: Machine.Limits,layout: Bands.Plan,cores: Model.Content<Element>,halos: BandHalo.Tree<Element>) -> Data:
  BandPool.Context{+geometry,+pad,+fill,+workers} = context
  match layout cores halos:
    case Bands.Rows{first,rows} Model.RowBand{core,core_first,core_rows} BandHalo.Leaf{above,below,window,halo_first,halo_rows}:
      Either<&2,&2,{window_cells(geometry,window) == 0n : Nat},WindowFacts(~Element,~Logical,~original,logical,geometry,pad,fill,first,rows,window,core,above,below,limits)>
    case Bands.Split{left_plan,right_plan} Model.Joined{left,right} BandHalo.Fork{left_halo,right_halo}:
      Shape.Both<TreeFacts(~Element,~Logical,~original,logical,BandPool.Context{geometry,pad,fill,workers},limits,left_plan,left,left_halo),
        TreeFacts(~Element,~Logical,~original,logical,BandPool.Context{geometry,pad,fill,workers},limits,right_plan,right,right_halo)>
    case layout cores halos: Empty

def pool_context(~Element: Data,~Logical: Data,logical: Logical,context: BandPool.Context<Element>) -> Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>:
  BandPool.Context{geometry,+pad,fill,workers} = context
  global_context(~Element,~Logical,logical,geometry,pad)

def pool_output_geometry(~Element: Data,context: BandPool.Context<Element>) -> Bands.Geometry:
  BandPool.Context{geometry,pad,fill,workers} = context
  values_output_geometry(geometry)

def tree_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+context: BandPool.Context<Element>,+limits: Machine.Limits,
  +layout: Bands.Plan,+cores: Model.Content<Element>,+halos: BandHalo.Tree<Element>,facts: TreeFacts(~Element,~Logical,~original,logical,context,limits,layout,cores,halos)) ->
  BandValues.Values(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),pool_context(~Element,~Logical,logical,context),pool_output_geometry(~Element,context),
    execution_model(~Element,~operation,context,layout,cores,halos)):
  BandPool.Context{+geometry,+pad,+fill,+workers} = context
  match layout cores halos:
    case Bands.Rows{+first,+rows} Model.RowBand{+core,core_first,core_rows} BandHalo.Leaf{+above,+below,+window,halo_first,halo_rows}:
      match facts:
        case Inl{zero}: zero_window_values(~Element,~Logical,~original,~operation,logical,geometry,pad,fill,first,rows,window,core,above,below,zero)
        case Inr{window_facts}: values_window_model(~Element,~Logical,~original,~operation,logical,geometry,pad,fill,first,rows,window,core,above,below,limits,window_facts)
    case Bands.Split{+left_plan,+right_plan} Model.Joined{+left,+right} BandHalo.Fork{+left_halo,+right_halo}:
      Shape.Both{left_facts,right_facts} = facts
      Shape.Both{tree_values(~Element,~Logical,~original,~operation,logical,BandPool.Context{geometry,pad,fill,workers},limits,left_plan,left,left_halo,left_facts),
        tree_values(~Element,~Logical,~original,~operation,logical,BandPool.Context{geometry,pad,fill,workers},limits,right_plan,right,right_halo,right_facts)}
    case Bands.Rows{first,rows} Model.RowBand{core,core_first,core_rows} BandHalo.Fork{left_halo,right_halo}: match facts:
    case Bands.Rows{first,rows} Model.Joined{left,right} BandHalo.Leaf{above,below,window,halo_first,halo_rows}: match facts:
    case Bands.Rows{first,rows} Model.Joined{left,right} BandHalo.Fork{left_halo,right_halo}: match facts:
    case Bands.Split{left_plan,right_plan} Model.RowBand{core,core_first,core_rows} BandHalo.Leaf{above,below,window,halo_first,halo_rows}: match facts:
    case Bands.Split{left_plan,right_plan} Model.RowBand{core,core_first,core_rows} BandHalo.Fork{left_halo,right_halo}: match facts:
    case Bands.Split{left_plan,right_plan} Model.Joined{left,right} BandHalo.Leaf{above,below,window,halo_first,halo_rows}: match facts:

def PlannedFacts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+context: BandPool.Context<Element>,+limits: Machine.Limits,+layout: Bands.Plan,+cores: Model.Content<Element>) -> Data:
  BandPool.Context{+geometry,+pad,fill,workers} = context
  TreeFacts(~Element,~Logical,~original,logical,context,limits,layout,cores,BandHalo.model(~Element,geometry,pad,layout,cores))

def planned_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+context: BandPool.Context<Element>,+limits: Machine.Limits,
  +layout: Bands.Plan,+cores: Model.Content<Element>,facts: PlannedFacts(~Element,~Logical,~original,logical,context,limits,layout,cores)) ->
  BandValues.Values(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),pool_context(~Element,~Logical,logical,context),pool_output_geometry(~Element,context),
    planned_model(~Element,~operation,context,layout,cores)):
  BandPool.Context{+geometry,+pad,+fill,+workers} = context
  tree_values(~Element,~Logical,~original,~operation,logical,BandPool.Context{geometry,pad,fill,workers},limits,layout,cores,BandHalo.model(~Element,geometry,pad,layout,cores),facts)

def values_execute(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+context: BandPool.Context<Element>,+limits: Machine.Limits,+cores: Model.Model<Element>,
  facts: PlannedFacts(~Element,~Logical,~original,logical,context,limits,Model.layout(~Element,cores),Model.content(~Element,cores))) ->
  BandValues.Values(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),pool_context(~Element,~Logical,logical,context),pool_output_geometry(~Element,context),
    Model.contents(~Element,BandPool.execute(~Element,~operation,context,Model.pack(~Element,cores)))):
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandPool.execute(~Element,~operation,context,Model.pack(~Element,cores))),
    planned_model(~Element,~operation,context,Model.layout(~Element,cores),Model.content(~Element,cores)),execution_execute(~Element,~operation,context,cores)) :
    BandValues.Values(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),pool_context(~Element,~Logical,logical,context),pool_output_geometry(~Element,context),_)
  planned_values(~Element,~Logical,~original,~operation,logical,context,limits,Model.layout(~Element,cores),Model.content(~Element,cores),facts)
