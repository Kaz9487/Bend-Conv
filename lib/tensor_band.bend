# Internal H-axis ownership. Offsets and spans are logical image rows, not
# element positions: offset / height identifies the image. No leaf crosses an
# image boundary. Its buffer stores [channels][span][width]. The common
# partition tree owns all buffers; shape and element type remain parameters.
import Base
import ./storage_buffer.bend as Storage
import ./storage_copy.bend as Copy
import ./storage_clone.bend as Clone
import ./traversal_partition.bend as Partition
import ./traversal_array.bend as Traversal
import ./proofs/traversal_certificate.bend as TraversalCertificate
import ./tensor_operations.bend as Operations
import ./storage_allocation.bend as Allocation
import ./tensor_view.bend as Views
import ./tensor_indexed.bend as IndexedAccess
import ./proofs/tensor_indexed_certificate.bend as IndexedCertificate
import ./proofs/storage_certificate.bend as Certificate

type Geometry is Data:
  Geometry{batch: U32,channels: U32,height: U32,width: U32}

def Bands(~Element: Data) -> Type:
  Partition.Tree<&1,Storage.Buffer<Element>,Unit>

def elements(geometry: Geometry) -> U32:
  Geometry{batch,channels,height,width} = geometry
  (batch * channels * height * width : U32)

def local_count(geometry: Geometry,rows: U32) -> U32:
  Geometry{batch,channels,height,width} = geometry
  (channels * rows * width : U32)

# Channel-major local indices map to the same global NCHW index for every
# partition. A leaf's first row is global across images; rows are image-local.
def global_index(geometry: Geometry,+first: U32,+rows: U32,+index: U32) -> U32:
  Geometry{batch,+channels,+height,+width} = geometry
  +plane = (rows * width : U32)
  (((first / height * channels + index / plane) * height + first % height) * width + index % plane : U32)

def planes(~Element: Data,remaining: Nat,+channel: U32,+count: U32,+source_stride: U32,+destination_stride: U32,
  +source_start: U32,+destination_start: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  Storage.Buffer<Element> & Storage.Buffer<Element>:
  match remaining:
    case 0n: owners
    case 1n+rest:
      (source,destination) = owners
      planes(~Element,rest,(channel + 1 : U32),count,source_stride,destination_stride,source_start,destination_start,
        Copy.range(~Element,count,(source_start + channel * source_stride : U32),
          (destination_start + channel * destination_stride : U32),source,destination))

def destination(~Element: Data,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  (source,destination) = owners
  destination

def gather_leaf(~Element: Data,geometry: Geometry,+first: U32,+rows: U32,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  Geometry{batch,+channels,+height,+width} = geometry
  destination(~Element,planes(~Element,U32.to_nat(channels),0,(rows * width : U32),(rows * width : U32),(height * width : U32),0,
    ((first / height * channels * height + first % height) * width : U32),(input,output)))

# One allocation is provided by the caller. Each value is copied once, using
# one contiguous range per channel per band. No intermediate dense images.
def gather(~Element: Data,+geometry: Geometry,bands: Bands(~Element),output: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  match bands:
    case Partition.Leaf{input,first,rows,details}: gather_leaf(~Element,geometry,first,rows,input,output)
    case Partition.Fork{left,right}: gather(~Element,geometry,right,gather(~Element,geometry,left,output))

def copied_leaf(~Element: Data,+first: U32,+rows: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Bands(~Element) & Bands(~Element):
  (left,right) = owners
  (Partition.Leaf{left,first,rows,Unit{}},Partition.Leaf{right,first,rows,Unit{}})

def copied_fork(~Element: Data,left: Bands(~Element) & Bands(~Element),right: Bands(~Element) & Bands(~Element)) -> Bands(~Element) & Bands(~Element):
  (left_original,left_copy) = left
  (right_original,right_copy) = right
  (Partition.Fork{left_original,right_original},Partition.Fork{left_copy,right_copy})

def copy(~Element: Data,bands: Bands(~Element)) -> Bands(~Element) & Bands(~Element):
  match bands:
    case Partition.Leaf{input,+first,+rows,details}: copied_leaf(~Element,first,rows,Clone.clone(~Element,input))
    case Partition.Fork{left,right}: copied_fork(~Element,copy(~Element,left),copy(~Element,right))

# A pure ownership plan contains no Buffer. It is a concrete Data type:
# duplicating Tree<&2,...> through a computed type makes all constructors
# shared in the official compiler, including the affine Buffer tree.
type Plan is Data:
  Rows{first: U32,count: U32}
  Split{left: Plan,right: Plan}

def plan_join(~Element: Data,left: Bands(~Element) & Plan,right: Bands(~Element) & Plan) -> Bands(~Element) & Plan:
  (left,left_plan) = left
  (right,right_plan) = right
  (Partition.Fork{left,right},Split{left_plan,right_plan})

def plan(~Element: Data,bands: Bands(~Element)) -> Bands(~Element) & Plan:
  match bands:
    case Partition.Leaf{input,+first,+rows,details}:
      (Partition.Leaf{input,first,rows,Unit{}},Rows{first,rows})
    case Partition.Fork{left,right}: plan_join(~Element,plan(~Element,left),plan(~Element,right))

def same_plan(left: Plan,right: Plan) -> Bool:
  match left right:
    case Rows{first,rows} Rows{other_first,other_rows}:
      U32.is_eq(first,other_first) && U32.is_eq(rows,other_rows)
    case Split{left,right} Split{other_left,other_right}:
      same_plan(left,other_left) && same_plan(right,other_right)
    case left right: False{}

def first(layout: Plan) -> U32:
  match layout:
    case Rows{first,count}: first
    case Split{left,right}: first(left)

def last(layout: Plan) -> U32:
  match layout:
    case Rows{first,count}: (first + count : U32)
    case Split{left,right}: last(right)

def same_image(geometry: Geometry,+layout: Plan) -> Bool:
  Geometry{batch,channels,+height,width} = geometry
  U32.is_eq((first(layout) / height : U32),((last(layout) - 1) / height : U32))

# Two trees of one shape, joined leaf by leaf. A left leaf without a partner
# keeps its own row range, so the left tree can be returned unchanged.
type Paired<-Left: Type,-Right: Type> is Type:
  Both{left: Left,right: Right}
  Only{left: Left,first: U32,rows: U32}

def only(~Left: Type,~Right: Type,tree: Partition.Tree<&1,Left,Unit>) -> Partition.Tree<&1,Paired<Left,Right>,Unit>:
  match tree:
    case Partition.Leaf{left,+first,+rows,details}: Partition.Leaf{Only{left,first,rows},first,rows,Unit{}}
    case Partition.Fork{left,right}: Partition.Fork{only(~Left,~Right,left),only(~Left,~Right,right)}

def zip(~Left: Type,~Right: Type,left: Partition.Tree<&1,Left,Unit>,right: Partition.Tree<&1,Right,Unit>) -> Partition.Tree<&1,Paired<Left,Right>,Unit>:
  match left right:
    case Partition.Leaf{left,first,rows,details} Partition.Leaf{right,other_first,other_rows,other_details}:
      Partition.Leaf{Both{left,right},first,rows,Unit{}}
    case Partition.Fork{left,right} Partition.Fork{other_left,other_right}:
      Partition.Fork{zip(~Left,~Right,left,other_left),zip(~Left,~Right,right,other_right)}
    case left right: only(~Left,~Right,left)

def left_leaf(~Left: Type,~Right: Type,pair: Paired<Left,Right>,first: U32,rows: U32) -> Partition.Tree<&1,Left,Unit>:
  match pair:
    case Both{left,right}: Partition.Leaf{left,first,rows,Unit{}}
    case Only{left,own_first,own_rows}: Partition.Leaf{left,own_first,own_rows,Unit{}}

def left_of(~Left: Type,~Right: Type,tree: Partition.Tree<&1,Paired<Left,Right>,Unit>) -> Partition.Tree<&1,Left,Unit>:
  match tree:
    case Partition.Leaf{pair,first,rows,details}: left_leaf(~Left,~Right,pair,first,rows)
    case Partition.Fork{left,right}: Partition.Fork{left_of(~Left,~Right,left),left_of(~Left,~Right,right)}

def paired(~Left: Type,~Right: Type,~Resource: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Left -> Right -> Resource -> Partition.Tree<&1,Left,Unit>,
  context: Context,first: U32,rows: U32,pair: Paired<Left,Right>,resource: Resource) -> Partition.Tree<&1,Left,Unit>:
  match pair:
    case Both{left,right}: leaf(context,first,rows,left,right,resource)
    case Only{left,own_first,own_rows}: Partition.Leaf{left,own_first,own_rows,Unit{}}

def unshared(resource: Unit) -> Unit & Unit:
  (Unit{},Unit{})

def same(~Block: Type,tree: Partition.Tree<&1,Block,Unit>) -> Partition.Tree<&1,Block,Unit>:
  tree

# One leaf below a parallel split. It neither forks nor calls a closure, so
# the runtime keeps it as its own task for the work turn; a leaf inlined into
# the forking recursion would run in the grow turn on one worker per row.
def leaf_task(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Block -> Resource -> Partition.Tree<&1,Output,Unit>,
  ~keep: Partition.Tree<&1,Block,Unit> -> Partition.Tree<&1,Output,Unit>,
  first: U32,rows: U32,context: Context,tree: Partition.Tree<&1,Block,Unit>,resource: Resource) -> Partition.Tree<&1,Output,Unit>:
  match tree:
    case Partition.Leaf{block,offset,span,details}: leaf(context,first,rows,block,resource)
    case Partition.Fork{left,right}: keep(Partition.Fork{left,right})

# The one band recursion. The plan gives every leaf its rows and decides the
# structure; a tree of another shape is kept. A split inside one image may
# run its halves in parallel, and a split across images runs them in order.
# share gives each half its own owner of a read-only resource.
def each(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Block -> Resource -> Partition.Tree<&1,Output,Unit>,
  ~keep: Partition.Tree<&1,Block,Unit> -> Partition.Tree<&1,Output,Unit>,
  layout: Plan,mode: Bool,+enabled: Bool,+geometry: Geometry,+context: Context,
  tree: Partition.Tree<&1,Block,Unit>,resource: Resource) -> Partition.Tree<&1,Output,Unit>:
  match layout mode tree:
    case Rows{first,rows} mode tree: leaf_task(~Block,~Output,~Resource,~Context,~leaf,~keep,first,rows,context,tree,resource)
    case Split{+left_plan,+right_plan} False{} Partition.Fork{left,right}:
      Storage.bind_pair(~Resource,~Resource,~Partition.Tree<&1,Output,Unit>,share(resource),left_resource => right_resource =>
        Partition.Fork{each(~Block,~Output,~Resource,~Context,~share,~leaf,~keep,left_plan,enabled && same_image(geometry,left_plan),enabled,geometry,context,left,left_resource),
          each(~Block,~Output,~Resource,~Context,~share,~leaf,~keep,right_plan,enabled && same_image(geometry,right_plan),enabled,geometry,context,right,right_resource)})
    case Split{Rows{left_first,left_rows},Rows{right_first,right_rows}} True{} Partition.Fork{left,right}:
      Storage.bind_pair(~Resource,~Resource,~Partition.Tree<&1,Output,Unit>,share(resource),left_resource => right_resource =>
        left right = leaf_task(~Block,~Output,~Resource,~Context,~leaf,~keep,left_first,left_rows,context,left,left_resource) leaf_task(~Block,~Output,~Resource,~Context,~leaf,~keep,right_first,right_rows,context,right,right_resource)
        Partition.Fork{left,right})
    case Split{Rows{left_first,left_rows},+right_plan} True{} Partition.Fork{left,right}:
      Storage.bind_pair(~Resource,~Resource,~Partition.Tree<&1,Output,Unit>,share(resource),left_resource => right_resource =>
        left right = leaf_task(~Block,~Output,~Resource,~Context,~leaf,~keep,left_first,left_rows,context,left,left_resource) each(~Block,~Output,~Resource,~Context,~share,~leaf,~keep,right_plan,enabled && same_image(geometry,right_plan),enabled,geometry,context,right,right_resource)
        Partition.Fork{left,right})
    case Split{+left_plan,Rows{right_first,right_rows}} True{} Partition.Fork{left,right}:
      Storage.bind_pair(~Resource,~Resource,~Partition.Tree<&1,Output,Unit>,share(resource),left_resource => right_resource =>
        left right = each(~Block,~Output,~Resource,~Context,~share,~leaf,~keep,left_plan,enabled && same_image(geometry,left_plan),enabled,geometry,context,left,left_resource) leaf_task(~Block,~Output,~Resource,~Context,~leaf,~keep,right_first,right_rows,context,right,right_resource)
        Partition.Fork{left,right})
    case Split{+left_plan,+right_plan} True{} Partition.Fork{left,right}:
      Storage.bind_pair(~Resource,~Resource,~Partition.Tree<&1,Output,Unit>,share(resource),left_resource => right_resource =>
        left right = each(~Block,~Output,~Resource,~Context,~share,~leaf,~keep,left_plan,enabled && same_image(geometry,left_plan),enabled,geometry,context,left,left_resource) each(~Block,~Output,~Resource,~Context,~share,~leaf,~keep,right_plan,enabled && same_image(geometry,right_plan),enabled,geometry,context,right,right_resource)
        Partition.Fork{left,right})
    case layout mode tree: keep(tree)

type Indexed is Data:
  Indexed{geometry: Geometry,first: U32,rows: U32}

def map_leaf(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,+context: Context,buffer: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  Storage.Buffer{storage,+length,certificate} = buffer
  Storage.Buffer{Traversal.map_in_place(~Element,~Context,~operation,length,0,context,storage),length,
    TraversalCertificate.map_in_place(~Element,~Context,~operation,length,0,context,storage,certificate)}

# A mapper over the whole of one band.
def map_whole_context(~Element: Data,~Context: Data,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,+context: Context,buffer: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  Storage.Buffer{storage,+length,certificate} = buffer
  mapper(length,0,context,Storage.Buffer{storage,length,certificate})

def map_band(~Element: Data,~Context: Data,~leaf: @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,context: Context,first: U32,rows: U32,
  buffer: Storage.Buffer<Element>,resource: Unit) -> Bands(~Element):
  Partition.Leaf{leaf(context,buffer),first,rows,Unit{}}

# Parallelism is a planner decision. Sequential execution keeps the bands and
# their owners: a small operation need not pay either a fork or a gather.
def map_with(~Element: Data,~Context: Data,~leaf: @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,+layout: Plan,+parallel: Bool,+geometry: Geometry,
  context: Context,bands: Bands(~Element)) -> Bands(~Element):
  each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~unshared,~map_band(~Element,~Context,~leaf),~same(~Storage.Buffer<Element>),
    layout,parallel && same_image(geometry,layout),parallel,geometry,context,bands,Unit{})

def scalar_value(~Element: Data,~operation: Element -> Element -> Element,value: Element,element: Element) -> Element:
  operation(element,value)

def scalar(~Element: Data,~operation: Element -> Element -> Element,layout: Plan,parallel: Bool,geometry: Geometry,value: Element,bands: Bands(~Element)) -> Bands(~Element):
  map_with(~Element,~Element,~map_leaf(~Element,~Element,~scalar_value(~Element,~operation)),layout,parallel,geometry,value,bands)

def result(~Element: Data,owners: Storage.Buffer<Element> & Bands(~Element)) -> Bands(~Element):
  (source,bands) = owners
  bands

def index_of(context: Indexed,index: U32) -> U32:
  Indexed{geometry,first,rows} = context
  global_index(geometry,first,rows,index)

# Channel c of a leaf holds the positions from c * rows * width; their global
# indices are consecutive from this base.
def channel_base(context: Indexed,channel: U32) -> U32:
  Indexed{Geometry{batch,channels,+height,width},+first,rows} = context
  (((first / height * channels + channel) * height + first % height) * width : U32)

def channels_of(context: Indexed) -> U32:
  Indexed{Geometry{batch,channels,height,width},first,rows} = context
  channels

def plane_of(context: Indexed) -> U32:
  Indexed{Geometry{batch,channels,height,width},first,rows} = context
  (rows * width : U32)

def indexed_buffers(~Element: Data,~operation: U32 -> Element -> Element,+count: U32,+context: Indexed,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  Storage.Buffer{input,input_length,input_certificate} = input
  Storage.Buffer{output,output_length,output_certificate} = output
  destination(~Element,Operations.pair_buffers(~Element,input_length,output_length,
    IndexedAccess.run_tiled(~Element,~Indexed,~index_of,~channel_base,~channels_of,~plane_of,~operation,count,context,input,output),
    IndexedCertificate.tiled_certificates(~Element,~Indexed,~index_of,~channel_base,~channels_of,~plane_of,~operation,count,context,input,output,input_certificate,output_certificate)))

def indexed_leaf(~Element: Data,~operation: U32 -> Element -> Element,
  +geometry: Geometry,+first: U32,+rows: U32,fill: Element,input: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  +count = local_count(geometry,rows)
  indexed_buffers(~Element,~operation,count,Indexed{geometry,first,rows},input,
    Allocation.allocate(~Element,Storage.allocation_depth(U32.max(count,1)),fill,U32.to_nat(count)))

type Filled<-Element: Data> is Data:
  Filled{geometry: Geometry,fill: Element}

def indexed_band(~Element: Data,~operation: U32 -> Element -> Element,context: Filled<Element>,first: U32,rows: U32,
  input: Storage.Buffer<Element>,resource: Unit) -> Bands(~Element):
  Filled{geometry,fill} = context
  +first = first
  +rows = rows
  Partition.Leaf{indexed_leaf(~Element,~operation,geometry,first,rows,fill,input),first,rows,Unit{}}

def apply_indexed(~Element: Data,~operation: U32 -> Element -> Element,
  +layout: Plan,+parallel: Bool,+geometry: Geometry,fill: Element,bands: Bands(~Element)) -> Bands(~Element):
  each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Filled<Element>,~unshared,~indexed_band(~Element,~operation),~same(~Storage.Buffer<Element>),
    layout,parallel && same_image(geometry,layout),parallel,geometry,Filled{geometry,fill},bands,Unit{})

def range_result(~Element: Data,+first: U32,+rows: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Bands(~Element) & Storage.Buffer<Element>:
  (source,output) = owners
  (Partition.Leaf{source,first,rows,Unit{}},output)

# An empty intersection copies nothing, so it skips the per-channel loop.
def range_planes(~Element: Data,empty: Bool,+channels: U32,+count: U32,+source_stride: U32,+destination_stride: U32,
  +source_start: U32,+destination_start: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  Storage.Buffer<Element> & Storage.Buffer<Element>:
  match empty:
    case True{}: owners
    case False{}: planes(~Element,U32.to_nat(channels),0,count,source_stride,destination_stride,source_start,destination_start,owners)

# Repartition and halo extraction share this owned range reader. Empty
# intersections copy zero elements; each source owner is returned unchanged.
def range_leaf(~Element: Data,geometry: Geometry,+target_first: U32,+target_rows: U32,target_offset: U32,+first: U32,+rows: U32,
  source: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Bands(~Element) & Storage.Buffer<Element>:
  Geometry{batch,channels,height,+width} = geometry
  +begin = U32.max(first,target_first)
  +end = U32.max(begin,U32.min((first + rows : U32),(target_first + target_rows : U32)))
  range_result(~Element,first,rows,range_planes(~Element,U32.is_eq(begin,end),channels,((end - begin) * width : U32),
    (rows * width : U32),(target_rows * width : U32),((begin - first) * width : U32),(target_offset + (begin - target_first) * width : U32),(source,output)))

def range_into(~Element: Data,+geometry: Geometry,+first: U32,+rows: U32,+target_offset: U32,bands: Bands(~Element),output: Storage.Buffer<Element>) -> Bands(~Element) & Storage.Buffer<Element>:
  match bands:
    case Partition.Leaf{input,source_first,source_rows,details}: range_leaf(~Element,geometry,first,rows,target_offset,source_first,source_rows,input,output)
    case Partition.Fork{left,right}:
      Storage.bind_pair(~Bands(~Element),~Storage.Buffer<Element>,~(Bands(~Element) & Storage.Buffer<Element>),
        range_into(~Element,geometry,first,rows,target_offset,left,output),left => output =>
          Storage.bind_pair(~Bands(~Element),~Storage.Buffer<Element>,~(Bands(~Element) & Storage.Buffer<Element>),
            range_into(~Element,geometry,first,rows,target_offset,right,output),right => output => (Partition.Fork{left,right},output)))

def range(~Element: Data,geometry: Geometry,first: U32,rows: U32,bands: Bands(~Element),output: Storage.Buffer<Element>) -> Bands(~Element) & Storage.Buffer<Element>:
  range_into(~Element,geometry,first,rows,0,bands,output)

def allocate(~Element: Data,fill: Element,+count: U32) -> Storage.Buffer<Element>:
  Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count))

def repartition(~Element: Data,+geometry: Geometry,+fill: Element,layout: Plan,bands: Bands(~Element)) -> Bands(~Element) & Bands(~Element):
  match layout:
    case Rows{+first,+rows}:
      Storage.bind_pair(~Bands(~Element),~Storage.Buffer<Element>,~(Bands(~Element) & Bands(~Element)),
        range(~Element,geometry,first,rows,bands,allocate(~Element,fill,local_count(geometry,rows))),source => output =>
          (source,Partition.Leaf{output,first,rows,Unit{}}))
    case Split{left,right}:
      Storage.bind_pair(~Bands(~Element),~Bands(~Element),~(Bands(~Element) & Bands(~Element)),
        repartition(~Element,geometry,fill,left,bands),source => left =>
          Storage.bind_pair(~Bands(~Element),~Bands(~Element),~(Bands(~Element) & Bands(~Element)),
            repartition(~Element,geometry,fill,right,source),source => right => (source,Partition.Fork{left,right})))

def repartitioned(~Element: Data,owners: Bands(~Element) & Bands(~Element)) -> Bands(~Element):
  (source,result) = owners
  result

def align_when(~Element: Data,same: Bool,geometry: Geometry,fill: Element,layout: Plan,bands: Bands(~Element)) -> Bands(~Element):
  match same:
    case True{}: bands
    case False{}: repartitioned(~Element,repartition(~Element,geometry,fill,layout,bands))

def align_planned(~Element: Data,geometry: Geometry,fill: Element,+layout: Plan,planned: Bands(~Element) & Plan) -> Bands(~Element):
  (bands,other) = planned
  align_when(~Element,same_plan(layout,other),geometry,fill,layout,bands)

def align(~Element: Data,geometry: Geometry,fill: Element,layout: Plan,bands: Bands(~Element)) -> Bands(~Element):
  align_planned(~Element,geometry,fill,layout,plan(~Element,bands))

def dense_result(~Element: Data,+first: U32,+rows: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Storage.Buffer<Element> & Bands(~Element):
  (source,band) = owners
  (source,Partition.Leaf{band,first,rows,Unit{}})

def dense_leaf(~Element: Data,geometry: Geometry,fill: Element,+first: U32,+rows: U32,input: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Bands(~Element):
  Geometry{batch,+channels,+height,+width} = geometry
  dense_result(~Element,first,rows,
    planes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),
      ((first / height * channels * height + first % height) * width : U32),0,(input,allocate(~Element,fill,(channels * rows * width : U32)))))

def dense_ranges(~Element: Data,+geometry: Geometry,+fill: Element,layout: Plan,input: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Bands(~Element):
  match layout:
    case Rows{first,rows}: dense_leaf(~Element,geometry,fill,first,rows,input)
    case Split{left,right}:
      Storage.bind_pair(~Storage.Buffer<Element>,~Bands(~Element),~(Storage.Buffer<Element> & Bands(~Element)),
        dense_ranges(~Element,geometry,fill,left,input),source => left =>
          Storage.bind_pair(~Storage.Buffer<Element>,~Bands(~Element),~(Storage.Buffer<Element> & Bands(~Element)),
            dense_ranges(~Element,geometry,fill,right,source),source => right => (source,Partition.Fork{left,right})))

def from_dense(~Element: Data,geometry: Geometry,fill: Element,layout: Plan,input: Storage.Buffer<Element>) -> Bands(~Element):
  result(~Element,dense_ranges(~Element,geometry,fill,layout,input))

def row_plan(depth: Nat,split: Bool,+first: U32,+rows: U32) -> Plan:
  match depth split:
    case 0n split: Rows{first,rows}
    case 1n+rest False{}: Rows{first,rows}
    case 1n+ +rest True{}:
      +half = (rows / 2 : U32)
      Split{row_plan(rest,U32.is_lt(1,half),first,half),
        row_plan(rest,U32.is_lt(1,(rows - half : U32)),(first + half : U32),(rows - half : U32))}

def image_plan(remaining: Nat,+image: U32,+depth: Nat,+height: U32) -> Plan:
  match remaining:
    case 0n: Rows{0,0}
    case 1n: row_plan(depth,U32.is_lt(1,height),(image * height : U32),height)
    case 2n+rest:
      Split{row_plan(depth,U32.is_lt(1,height),(image * height : U32),height),image_plan(1n+rest,(image + 1 : U32),depth,height)}

# Plan metadata first, then transfer each dense channel range once. First
# splits and later changed boundaries share the same copy implementation.
def split(~Element: Data,depth: Nat,+geometry: Geometry,fill: Element,input: Storage.Buffer<Element>) -> Bands(~Element):
  Geometry{batch,channels,height,width} = geometry
  from_dense(~Element,geometry,fill,image_plan(U32.to_nat(batch),0,depth,height),input)

def combined(~Element: Data,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) -> Storage.Buffer<Element>:
  match result:
    case Operations.Accepted{(left,right)}: left
    case Operations.Rejected{(left,right)}: left

def combine_leaf_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+count: U32,left: Storage.Buffer<Element>,right: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  combined(~Element,Operations.combine_with(~Element,~kernel,left,Views.View{[Views.Axis{count,1}],0},right,Views.View{[Views.Axis{count,1}],0}))

def combine_band_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,geometry: Geometry,first: U32,rows: U32,
  left: Storage.Buffer<Element>,right: Storage.Buffer<Element>,resource: Unit) -> Bands(~Element):
  +rows = rows
  Partition.Leaf{combine_leaf_with(~Element,~kernel,local_count(geometry,rows),left,right),first,rows,Unit{}}

def combine_planned_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+geometry: Geometry,+parallel: Bool,planned: Bands(~Element) & Plan,right: Bands(~Element)) -> Bands(~Element):
  (left,+layout) = planned
  each(~Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Storage.Buffer<Element>,~Unit,~Geometry,~unshared,
    ~paired(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Geometry,~combine_band_with(~Element,~kernel)),
    ~left_of(~Storage.Buffer<Element>,~Storage.Buffer<Element>),
    layout,parallel && same_image(geometry,layout),parallel,geometry,geometry,zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,left,right),Unit{})

# The caller supplies the same plan on both sides, after shape and status
# checks; the plan is read from the left. Trees of another shape keep the left owner.
def combine_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,geometry: Geometry,parallel: Bool,left: Bands(~Element),right: Bands(~Element)) -> Bands(~Element):
  combine_planned_with(~Element,~kernel,geometry,parallel,plan(~Element,left),right)

def transform_band(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Bands(~Element),
  context: Context,first: U32,rows: U32,input: Storage.Buffer<Element>,resource: Unit) -> Bands(~Element):
  leaf(context,first,rows,input)

def transform(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Bands(~Element),
  +layout: Plan,+parallel: Bool,+geometry: Geometry,context: Context,bands: Bands(~Element)) -> Bands(~Element):
  each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~unshared,~transform_band(~Element,~Context,~leaf),~same(~Storage.Buffer<Element>),
    layout,parallel && same_image(geometry,layout),parallel,geometry,context,bands,Unit{})

type Inspection is Data:
  Inspection{valid: Bool,first: U32,end: U32,leaves: U32}

def join_inspection(left: Inspection,right: Inspection) -> Inspection:
  Inspection{lv,lf,+le,ln} = left
  Inspection{rv,+rf,re,rn} = right
  Inspection{lv && rv && U32.is_eq(le,rf),lf,re,(ln + rn : U32)}

def inspected_pair(~Element: Data,left: Bands(~Element) & Inspection,right: Bands(~Element) & Inspection) -> Bands(~Element) & Inspection:
  (left,left_inspection) = left
  (right,right_inspection) = right
  (Partition.Fork{left,right},join_inspection(left_inspection,right_inspection))

def leaf_inspection(geometry: Geometry,+length: Nat,depth: Nat,+first: U32,+rows: U32) -> Inspection:
  Geometry{batch,+channels,+height,+width} = geometry
  Inspection{U32.is_le(rows,(height - first % height : U32)) &&
      Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(height))) &&
      Nat.is_eq(length,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))) &&
      Nat.is_le(length,U32.to_nat(U32.shln(1,depth))),first,(first + rows : U32),1}

def inspect_leaf(~Element: Data,geometry: Geometry,buffer: Storage.Buffer<Element>,+first: U32,+rows: U32) -> Bands(~Element) & Inspection:
  Storage.Buffer{storage,+length,Certificate.Certificate{+depth,equation}} = buffer
  (Partition.Leaf{Storage.Buffer{storage,length,Certificate.Certificate{depth,equation}},first,rows,Unit{}},
    leaf_inspection(geometry,length,depth,first,rows))

def inspect(~Element: Data,+geometry: Geometry,bands: Bands(~Element)) -> Bands(~Element) & Inspection:
  match bands:
    case Partition.Leaf{buffer,first,rows,details}: inspect_leaf(~Element,geometry,buffer,first,rows)
    case Partition.Fork{left,right}: inspected_pair(~Element,inspect(~Element,geometry,left),inspect(~Element,geometry,right))
