# Public tensors own dense row-major storage or image-row bands, their extents,
# the value used to initialize allocations, a sticky status, and the number of
# workers their operations may use. A failed operation leaves values untouched and
# records why; later operations skip a failed tensor, and check is the only way
# to observe or clear the failure.
import Base
import ./storage_buffer.bend as Storage
import ./storage_allocation.bend as Allocation
import ./tensor_shape.bend as Shapes
import ./tensor_view.bend as Views
import ./tensor_operations.bend as Operations
import ./machine_limits.bend as Machine
import ./default_machine_limits.bend as Default
import ./tensor_band.bend as Bands
import ./tensor_band_schedule.bend as BandSchedule

type Reason is Data:
  ShapeMismatch{left: List<&2,U32>,right: List<&2,U32>}
  InvalidShape{shape: List<&2,U32>}
  InvalidArgument{shape: List<&2,U32>}
  FileFault{path: String,detail: String}
  MissingName{name: String}

type Problem is Data:
  Problem{operation: String,reason: Reason}

type Status is Data:
  Usable{}
  Failed{problem: Problem}

# Like a PyTorch tensor's device, workers is execution context: the size of
# the runtime's worker pool, read once with IO.thread_count and set with
# with_workers. Pure constructors cannot read it and use one worker. Results
# keep the largest worker count of their operands. It never changes a value.
type Tensor<-Element: Data> is Type:
  Tensor{storage: Storage.Buffer<Element>,shape: List<&2,U32>,fill: Element,status: Status,workers: U32}
  Banded{bands: Bands.Bands(~Element),geometry: Bands.Geometry,shape: List<&2,U32>,fill: Element,status: Status,workers: U32}

# A nonempty sequence of tensors. Base List is not used for tensors: the
# compiler shares every constructor of a list type that is duplicated anywhere,
# and that sharing would reach the certificate inside each tensor's storage.
type Parts<-Element: Data> is Type:
  Single{tensor: Tensor<Element>}
  Cons{tensor: Tensor<Element>,rest: Parts<Element>}

# Element count of a shape that the default machine can allocate.
def within(fits: Bool,count: U32) -> Maybe<&2,U32>:
  match fits:
    case True{}: Some{count}
    case False{}: None{}

def allocatable(count: Maybe<&2,U32>) -> Maybe<&2,U32>:
  match count:
    case None{}: None{}
    case Some{+count}: within(U32.is_le(count,Machine.capacity(Default.limits())),count)

def count_of(+shape: List<&2,U32>) -> Maybe<&2,U32>:
  allocatable(Shapes.numel(Shapes.Shape{shape}))

def known_count(count: Maybe<&2,U32>) -> U32:
  match count:
    case Some{count}: count
    case None{}: 0

# The element count as a number; a shape too large to allocate counts 0.
def count_or_zero(+shape: List<&2,U32>) -> U32:
  known_count(count_of(shape))

def view_of(+shape: List<&2,U32>) -> Maybe<&2,Views.View>:
  Views.contiguous(Shapes.Shape{shape},0)

# Bits of value, counted by tail recursion. Storage.bit_length is not tail
# recursive, and a non-tail recursive callee turns the calling chain into
# worklist dispatch: measured allocation of 2^20 floats took twice as long.
def significant_bits(remaining: Nat,+value: U32,+bits: Nat,zero: Bool) -> Nat:
  Storage.significant_bits(remaining,value,bits,zero)

# The smallest depth whose capacity holds count elements (at least one).
def depth_for(+count: U32) -> Nat:
  Storage.depth_for(count)

def buffer(~Element: Data,+fill: Element,+count: U32) -> Storage.Buffer<Element>:
  Allocation.allocate(~Element,depth_for(count),fill,U32.to_nat(count))

def with_dense(~Element: Data,~Result: Type,tensor: Tensor<Element>,
  body: Storage.Buffer<Element> -> List<&2,U32> -> Element -> Status -> U32 -> Result) -> Result:
  match tensor:
    case Tensor{storage,shape,fill,status,workers}: body(storage,shape,fill,status,workers)
    case Banded{bands,+geometry,shape,+fill,status,workers}:
      body(Bands.gather(~Element,geometry,bands,buffer(~Element,fill,Bands.elements(geometry))),shape,fill,status,workers)

# Metadata inspection retains the original layout and affine storage.
def with_metadata(~Element: Data,~Result: Type,tensor: Tensor<Element>,
  body: Tensor<Element> -> List<&2,U32> -> Element -> Status -> U32 -> Result) -> Result:
  match tensor:
    case Tensor{storage,+shape,+fill,+status,+workers}: body(Tensor{storage,shape,fill,status,workers},shape,fill,status,workers)
    case Banded{bands,geometry,+shape,+fill,+status,+workers}: body(Banded{bands,geometry,shape,fill,status,workers},shape,fill,status,workers)

def failed(~Element: Data,operation: String,reason: Reason,shape: List<&2,U32>,+fill: Element,workers: U32) -> Tensor<Element>:
  Tensor{buffer(~Element,fill,0),shape,fill,Failed{Problem{operation,reason}},workers}

# A usable tensor fails with an invalid argument of the operation; a failed one keeps its problem.
def rejected(~Element: Data,name: String,tensor: Tensor<Element>) -> Tensor<Element>:
  match tensor:
    case Tensor{storage,+shape,fill,Usable{},workers}: Tensor{storage,shape,fill,Failed{Problem{name,InvalidArgument{shape}}},workers}
    case Tensor{storage,shape,fill,Failed{problem},workers}: Tensor{storage,shape,fill,Failed{problem},workers}
    case Banded{bands,geometry,+shape,fill,Usable{},workers}: Banded{bands,geometry,shape,fill,Failed{Problem{name,InvalidArgument{shape}}},workers}
    case Banded{bands,geometry,shape,fill,Failed{problem},workers}: Banded{bands,geometry,shape,fill,Failed{problem},workers}

def mismatch(same: Bool,left: List<&2,U32>,right: List<&2,U32>) -> Reason:
  match same:
    case True{}: InvalidShape{left}
    case False{}: ShapeMismatch{left,right}

def same_or_invalid(+left: List<&2,U32>,+right: List<&2,U32>) -> Reason:
  mismatch(Shapes.same_extents(left,right),left,right)

# Construction

def full_count(~Element: Data,+shape: List<&2,U32>,count: Maybe<&2,U32>,+value: Element) -> Tensor<Element>:
  match count:
    case Some{count}: Tensor{buffer(~Element,value,count),shape,value,Usable{},1}
    case None{}: failed(~Element,"full",InvalidShape{shape},shape,value,1)

def full(~Element: Data,+shape: List<&2,U32>,+value: Element) -> Tensor<Element>:
  full_count(~Element,shape,count_of(shape),value)

def listed_count(~Element: Data,matches: Bool,+shape: List<&2,U32>,+count: U32,values: List<&2,Element>,+fill: Element) -> Tensor<Element>:
  match matches:
    case True{}:
      Tensor{Storage.from_list(~Element,values,depth_for(count),fill,U32.to_nat(count)),shape,fill,Usable{},1}
    case False{}: failed(~Element,"from_list",InvalidArgument{shape},shape,fill,1)

def listed(~Element: Data,+shape: List<&2,U32>,count: Maybe<&2,U32>,+values: List<&2,Element>,+fill: Element) -> Tensor<Element>:
  match count:
    case Some{+count}: listed_count(~Element,Nat.is_eq(List.length(&2,Element,values),U32.to_nat(count)),shape,count,values,fill)
    case None{}: failed(~Element,"from_list",InvalidShape{shape},shape,fill,1)

# fill initializes storage this tensor or its results allocate; it is never
# observable in a successful result.
def from_list(~Element: Data,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element) -> Tensor<Element>:
  listed(~Element,shape,count_of(shape),values,fill)

def stored(~Element: Data,matches: Bool,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,fill: Element) -> Tensor<Element>:
  match matches:
    case True{}: Tensor{storage,shape,fill,Usable{},1}
    case False{}: Tensor{storage,shape,fill,Failed{Problem{"from_storage",InvalidShape{shape}}},1}

def matches_count(count: Maybe<&2,U32>,length: Nat) -> Bool:
  match count:
    case Some{count}: Nat.is_eq(length,U32.to_nat(count))
    case None{}: False{}

def storage_of(~Element: Data,+shape: List<&2,U32>,count: Maybe<&2,U32>,storage: Storage.Buffer<Element>,fill: Element) -> Tensor<Element>:
  Storage.Buffer{storage,+length,certificate} = storage
  stored(~Element,matches_count(count,length),Storage.Buffer{storage,length,certificate},shape,fill)

# Storage handed over by foreign I/O keeps its documented trust boundary.
def from_storage(~Element: Data,+shape: List<&2,U32>,storage: Storage.Buffer<Element>,+fill: Element) -> Tensor<Element>:
  storage_of(~Element,shape,count_of(shape),storage,fill)

# Execution context

# Sets the worker count, at least one; values and status are unchanged.
def with_workers(~Element: Data,tensor: Tensor<Element>,+count: U32) -> Tensor<Element>:
  match tensor:
    case Tensor{storage,shape,fill,status,workers}: Tensor{storage,shape,fill,status,U32.max(1,count)}
    case Banded{bands,geometry,shape,fill,status,workers}: Banded{bands,geometry,shape,fill,status,U32.max(1,count)}

def workers(~Element: Data,tensor: Tensor<Element>) -> Tensor<Element> & U32:
  match tensor:
    case Tensor{storage,shape,fill,status,+workers}: (Tensor{storage,shape,fill,status,workers},workers)
    case Banded{bands,geometry,shape,fill,status,+workers}: (Banded{bands,geometry,shape,fill,status,workers},workers)

# Observation

def shape(~Element: Data,tensor: Tensor<Element>) -> Tensor<Element> & List<&2,U32>:
  match tensor:
    case Tensor{storage,+shape,fill,status,workers}: (Tensor{storage,shape,fill,status,workers},shape)
    case Banded{bands,geometry,+shape,fill,status,workers}: (Banded{bands,geometry,shape,fill,status,workers},shape)

def checked(~Element: Data,status: Status,storage: Storage.Buffer<Element>,shape: List<&2,U32>,fill: Element,workers: U32) ->
  Result<&1,&1,Tensor<Element> & Problem,Tensor<Element>>:
  match status:
    case Usable{}: Done{Tensor{storage,shape,fill,Usable{},workers}}
    case Failed{problem}: Fail{(Tensor{storage,shape,fill,Usable{},workers},problem)}

def checked_bands(~Element: Data,status: Status,bands: Bands.Bands(~Element),geometry: Bands.Geometry,shape: List<&2,U32>,fill: Element,workers: U32) ->
  Result<&1,&1,Tensor<Element> & Problem,Tensor<Element>>:
  match status:
    case Usable{}: Done{Banded{bands,geometry,shape,fill,Usable{},workers}}
    case Failed{problem}: Fail{(Banded{bands,geometry,shape,fill,Usable{},workers},problem)}

def check(~Element: Data,tensor: Tensor<Element>) -> Result<&1,&1,Tensor<Element> & Problem,Tensor<Element>>:
  match tensor:
    case Tensor{storage,shape,fill,status,workers}: checked(~Element,status,storage,shape,fill,workers)
    case Banded{bands,geometry,shape,fill,status,workers}: checked_bands(~Element,status,bands,geometry,shape,fill,workers)

def released(~Element: Data,status: Status,storage: Storage.Buffer<Element>) -> Result<&2,&1,Problem,Storage.Buffer<Element>>:
  match status:
    case Usable{}: Done{storage}
    case Failed{problem}: Fail{problem}

def into_storage(~Element: Data,tensor: Tensor<Element>) -> Result<&2,&1,Problem,Storage.Buffer<Element>>:
  with_dense(~Element,~Result<&2,&1,Problem,Storage.Buffer<Element>>,tensor,storage => shape => fill => status => workers => released(~Element,status,storage))

def listed_values(~Element: Data,status: Status,storage: Array<Element>,length: Nat) -> Result<&2,&2,Problem,List<&2,Element>>:
  match status:
    case Usable{}: Done{Storage.read_list(~Element,length,0,storage)}
    case Failed{problem}: Fail{problem}

def listed_buffer(~Element: Data,status: Status,buffer: Storage.Buffer<Element>) -> Result<&2,&2,Problem,List<&2,Element>>:
  Storage.Buffer{storage,length,certificate} = buffer
  listed_values(~Element,status,storage,length)

def to_list(~Element: Data,tensor: Tensor<Element>) -> Result<&2,&2,Problem,List<&2,Element>>:
  with_dense(~Element,~Result<&2,&2,Problem,List<&2,Element>>,tensor,buffer => shape => fill => status => workers =>
    listed_buffer(~Element,status,buffer))

def only_value(~Element: Data,+shape: List<&2,U32>,values: List<&2,Element>) -> Result<&2,&2,Problem,Element>:
  match values:
    case Con{value,Nil{}}: Done{value}
    case values: Fail{Problem{"item",InvalidShape{shape}}}

def single(~Element: Data,+shape: List<&2,U32>,values: Result<&2,&2,Problem,List<&2,Element>>) -> Result<&2,&2,Problem,Element>:
  match values:
    case Done{values}: only_value(~Element,shape,values)
    case Fail{problem}: Fail{problem}

def item(~Element: Data,tensor: Tensor<Element>) -> Result<&2,&2,Problem,Element>:
  with_dense(~Element,~Result<&2,&2,Problem,Element>,tensor,storage => shape => fill => status => workers =>
    +shape = shape
    single(~Element,shape,to_list(~Element,Tensor{storage,shape,fill,status,workers})))

# The value read, or fallback when the tensor carries a failure.
def value_or(~Value: Data,result: Result<&2,&2,Problem,Value>,fallback: Value) -> Value:
  match result:
    case Done{value}: value
    case Fail{problem}: fallback

# Operations consume their tensor arguments and return one result; callers
# keep a tensor by copying it first. Binary elementwise operands follow NumPy
# broadcasting; the result reuses the left storage when it has the left
# extents. The result carries the first failure of its operands.

def first_failure(left: Status,right: Status) -> Status:
  match left:
    case Usable{}: right
    case Failed{problem}: Failed{problem}

def combined(~Element: Data,operation: String,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+fill: Element,+workers: U32,
  result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) -> Tensor<Element>:
  match result:
    case Operations.Accepted{(left,right)}: Tensor{left,left_shape,fill,Usable{},workers}
    case Operations.Rejected{(left,right)}: Tensor{left,left_shape,fill,Failed{Problem{operation,same_or_invalid(left_shape,right_shape)}},workers}

def combine_views_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,name: String,left_view: Maybe<&2,Views.View>,right_view: Maybe<&2,Views.View>,
  left: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+fill: Element,right: Storage.Buffer<Element>,+right_shape: List<&2,U32>,+workers: U32) -> Tensor<Element>:
  match left_view right_view:
    case Some{left_view} Some{right_view}:
      combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,left,left_view,right,right_view))
    case left_view right_view: Tensor{left,left_shape,fill,Failed{Problem{name,same_or_invalid(left_shape,right_shape)}},workers}

# A view of an operand read with the extents target; see Views.stretch.
def stretched(view: Maybe<&2,Views.View>,+target: List<&2,U32>) -> Maybe<&2,Views.View>:
  match view:
    case Some{view}: Views.stretch(view,target)
    case None{}: None{}

# The left operand was copied through its stretched view into new storage of
# the common extents; the right operand is then combined into that storage.
# The view of that storage is built once for each of the two steps: using one
# view value twice would make the compiler share every View constructor.
def combined_fresh_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,name: String,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  right: Storage.Buffer<Element>,right_view: Views.View,target: Views.View,
  copied: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) -> Tensor<Element>:
  match copied:
    case Operations.Accepted{(left,output)}:
      combined(~Element,name,shape,shape,fill,workers,Operations.combine_with(~Element,~kernel,output,target,right,right_view))
    case Operations.Rejected{(left,output)}: Tensor{output,shape,fill,Failed{Problem{name,InvalidShape{shape}}},workers}

def combine_fresh_views_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,name: String,left_view: Maybe<&2,Views.View>,right_view: Maybe<&2,Views.View>,
  target: Maybe<&2,Views.View>,output: Maybe<&2,Views.View>,count: Maybe<&2,U32>,left: Storage.Buffer<Element>,right: Storage.Buffer<Element>,
  +shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensor<Element>:
  match left_view right_view target output count:
    case Some{left_view} Some{right_view} Some{target} Some{output} Some{count}:
      combined_fresh_with(~Element,~kernel,name,shape,fill,workers,right,right_view,output,
        Operations.copy(~Element,left,left_view,buffer(~Element,fill,count),target))
    case left_view right_view target output count: failed(~Element,name,InvalidShape{shape},shape,fill,workers)

# Operands of different extents with the common extents shape. When shape is
# the left extents, only the right operand is stretched and the result reuses
# the left storage. Otherwise the left operand is first copied through its
# stretched view into new storage, which the right operand is combined into;
# both steps run the innermost axis as one loop, which the three-buffer
# combination with two cursors does not.
def combine_stretched_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,name: String,in_place: Bool,
  left: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+fill: Element,right: Storage.Buffer<Element>,+right_shape: List<&2,U32>,+shape: List<&2,U32>,+workers: U32) -> Tensor<Element>:
  match in_place:
    case True{}:
      combine_views_with(~Element,~kernel,name,view_of(left_shape),stretched(view_of(right_shape),left_shape),left,left_shape,fill,right,right_shape,workers)
    case False{}:
      combine_fresh_views_with(~Element,~kernel,name,stretched(view_of(left_shape),shape),stretched(view_of(right_shape),shape),view_of(shape),view_of(shape),count_of(shape),
        left,right,shape,fill,workers)

def combine_broadcast_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,name: String,shape: Maybe<&2,List<&2,U32>>,
  left: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+fill: Element,right: Storage.Buffer<Element>,+right_shape: List<&2,U32>,+workers: U32) -> Tensor<Element>:
  match shape:
    case Some{+shape}:
      combine_stretched_with(~Element,~kernel,name,Shapes.same_extents(shape,left_shape),left,left_shape,fill,right,right_shape,shape,workers)
    case None{}: Tensor{left,left_shape,fill,Failed{Problem{name,ShapeMismatch{left_shape,right_shape}}},workers}

# Equal extents take the direct path. Different extents follow NumPy
# broadcasting (Shapes.broadcast) and fail only when they are incompatible.
def combine_shapes_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,name: String,same: Bool,
  left: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+fill: Element,right: Storage.Buffer<Element>,+right_shape: List<&2,U32>,+workers: U32) -> Tensor<Element>:
  match same:
    case True{}: combine_views_with(~Element,~kernel,name,view_of(left_shape),view_of(right_shape),left,left_shape,fill,right,right_shape,workers)
    case False{}: combine_broadcast_with(~Element,~kernel,name,Shapes.broadcast(left_shape,right_shape),left,left_shape,fill,right,right_shape,workers)

def combine_status_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,name: String,status: Status,
  left: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+fill: Element,right: Storage.Buffer<Element>,+right_shape: List<&2,U32>,+workers: U32) -> Tensor<Element>:
  match status:
    case Usable{}: combine_shapes_with(~Element,~kernel,name,Shapes.same_extents(left_shape,right_shape),left,left_shape,fill,right,right_shape,workers)
    case Failed{problem}: Tensor{left,left_shape,fill,Failed{problem},workers}

# Band layout is kept only for equal extents: operands that need broadcasting
# are gathered and combined on dense storage.
def combine_dense_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,name: String,left: Tensor<Element>,right: Tensor<Element>) -> Tensor<Element>:
  with_dense(~Element,~Tensor<Element>,left,left => left_shape => fill => left_status => left_workers =>
    with_dense(~Element,~Tensor<Element>,right,right => right_shape => right_fill => right_status => right_workers =>
      combine_status_with(~Element,~kernel,name,first_failure(left_status,right_status),left,left_shape,fill,right,right_shape,U32.max(left_workers,right_workers))))

def with_status(~Element: Data,status: Status,tensor: Tensor<Element>) -> Tensor<Element>:
  match tensor:
    case Tensor{storage,shape,fill,previous,workers}: Tensor{storage,shape,fill,status,workers}
    case Banded{bands,geometry,shape,fill,previous,workers}: Banded{bands,geometry,shape,fill,status,workers}

def aligned(~Element: Data,geometry: Bands.Geometry,layout: Bands.Plan,tensor: Tensor<Element>) -> Bands.Bands(~Element):
  match tensor:
    case Tensor{storage,shape,fill,status,workers}: Bands.from_dense(~Element,geometry,fill,layout,storage)
    case Banded{bands,previous,shape,fill,status,workers}: Bands.align(~Element,geometry,fill,layout,bands)

def combine_bands_checked_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,name: String,operations: F32,status: Status,same: Bool,+geometry: Bands.Geometry,
  +layout: Bands.Plan,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,fill: Element,+workers: U32,left: Tensor<Element>,right: Tensor<Element>) -> Tensor<Element>:
  match status same:
    case Usable{} True{}:
      Banded{Bands.combine_with(~Element,~kernel,geometry,BandSchedule.parallel(geometry,workers,12.0,operations,layout),aligned(~Element,geometry,layout,left),aligned(~Element,geometry,layout,right)),
        geometry,left_shape,fill,Usable{},workers}
    case Usable{} False{}: combine_dense_with(~Element,~kernel,name,left,right)
    case Failed{problem} same: with_status(~Element,Failed{problem},left)

def combine_bands_plan_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,name: String,operations: F32,status: Status,same: Bool,+geometry: Bands.Geometry,
  +left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+fill: Element,+workers: U32,right: Tensor<Element>,planned: Bands.Bands(~Element) & Bands.Plan) -> Tensor<Element>:
  (left,layout) = planned
  combine_bands_checked_with(~Element,~kernel,name,operations,status,same,geometry,layout,left_shape,right_shape,fill,workers,
    Banded{left,geometry,left_shape,fill,Usable{},workers},right)

def combine_right_plan_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,name: String,operations: F32,status: Status,same: Bool,+geometry: Bands.Geometry,
  +left_shape: List<&2,U32>,+right_shape: List<&2,U32>,left_fill: Element,+right_fill: Element,+workers: U32,left: Storage.Buffer<Element>,
  planned: Bands.Bands(~Element) & Bands.Plan) -> Tensor<Element>:
  (right,layout) = planned
  +left_fill = left_fill
  combine_bands_checked_with(~Element,~kernel,name,operations,status,same,geometry,layout,left_shape,right_shape,left_fill,workers,
    Tensor{left,left_shape,left_fill,Usable{},workers},Banded{right,geometry,right_shape,right_fill,Usable{},workers})

def combine_pair_plans_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,name: String,operations: F32,status: Status,same: Bool,+geometry: Bands.Geometry,right_geometry: Bands.Geometry,
  +left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+left_fill: Element,right_fill: Element,+workers: U32,
  left: Bands.Bands(~Element) & Bands.Plan,right: Bands.Bands(~Element) & Bands.Plan) -> Tensor<Element>:
  (left,left_plan) = left
  (right,right_plan) = right
  combine_bands_checked_with(~Element,~kernel,name,operations,status,same,geometry,BandSchedule.alignment(geometry,left_plan,right_plan),left_shape,right_shape,left_fill,workers,
    Banded{left,geometry,left_shape,left_fill,Usable{},workers},Banded{right,right_geometry,right_shape,right_fill,Usable{},workers})

# operation(left element, right element), retaining band layout when either
# operand is banded. Alignment is done once, before any parallel leaf work.
# operations counts the operation's scalar work for the band schedule.
def combine_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,name: String,operations: F32,left: Tensor<Element>,right: Tensor<Element>) -> Tensor<Element>:
  match left right:
    case Tensor{left,left_shape,left_fill,left_status,left_workers} Tensor{right,right_shape,right_fill,right_status,right_workers}:
      combine_status_with(~Element,~kernel,name,first_failure(left_status,right_status),left,left_shape,left_fill,right,right_shape,U32.max(left_workers,right_workers))
    case Banded{left,geometry,+left_shape,left_fill,left_status,left_workers} Tensor{right,+right_shape,right_fill,+right_status,+right_workers}:
      combine_bands_plan_with(~Element,~kernel,name,operations,first_failure(left_status,right_status),Shapes.same_extents(left_shape,right_shape),geometry,
        left_shape,right_shape,left_fill,U32.max(left_workers,right_workers),Tensor{right,right_shape,right_fill,right_status,right_workers},Bands.plan(~Element,left))
    case Banded{left,geometry,+left_shape,left_fill,left_status,left_workers} Banded{right,right_geometry,+right_shape,right_fill,+right_status,+right_workers}:
      combine_pair_plans_with(~Element,~kernel,name,operations,first_failure(left_status,right_status),Shapes.same_extents(left_shape,right_shape),geometry,right_geometry,
        left_shape,right_shape,left_fill,right_fill,U32.max(left_workers,right_workers),Bands.plan(~Element,left),Bands.plan(~Element,right))
    case Tensor{left,+left_shape,left_fill,left_status,left_workers} Banded{right,geometry,+right_shape,right_fill,right_status,right_workers}:
      combine_right_plan_with(~Element,~kernel,name,operations,first_failure(left_status,right_status),Shapes.same_extents(left_shape,right_shape),geometry,
        left_shape,right_shape,left_fill,right_fill,U32.max(left_workers,right_workers),left,Bands.plan(~Element,right))

# The elementwise form: the kernel applies operation to each pair in order.
def combine(~Element: Data,~operation: Element -> Element -> Element,name: String,operations: F32,left: Tensor<Element>,right: Tensor<Element>) -> Tensor<Element>:
  combine_with(~Element,~Operations.combine_kernel(~Element,~operation),name,operations,left,right)

def applied(~Element: Data,name: String,+shape: List<&2,U32>,+fill: Element,+workers: U32,result: Operations.Outcome<Storage.Buffer<Element>>) -> Tensor<Element>:
  match result:
    case Operations.Accepted{storage}: Tensor{storage,shape,fill,Usable{},workers}
    case Operations.Rejected{storage}: Tensor{storage,shape,fill,Failed{Problem{name,InvalidShape{shape}}},workers}

def apply_view_context(~Element: Data,~Context: Data,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,context: Context,name: String,view: Maybe<&2,Views.View>,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,
  +workers: U32) -> Tensor<Element>:
  match view:
    case Some{view}: applied(~Element,name,shape,fill,workers,Operations.apply_context(~Element,~Context,~mapper,context,storage,view))
    case None{}: Tensor{storage,shape,fill,Failed{Problem{name,InvalidShape{shape}}},workers}

def apply_status_context(~Element: Data,~Context: Data,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,context: Context,name: String,status: Status,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,
  +workers: U32) -> Tensor<Element>:
  match status:
    case Usable{}: apply_view_context(~Element,~Context,~mapper,context,name,view_of(shape),storage,shape,fill,workers)
    case Failed{problem}: Tensor{storage,shape,fill,Failed{problem},workers}

def apply_bands_context(~Element: Data,~Context: Data,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,context: Context,operations: F32,status: Status,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,
  shape: List<&2,U32>,fill: Element,+workers: U32) -> Tensor<Element>:
  match status:
    case Usable{}: Banded{BandSchedule.map_context(~Element,~Context,~mapper,context,geometry,workers,operations,bands),geometry,shape,fill,Usable{},workers}
    case Failed{problem}: Banded{bands,geometry,shape,fill,Failed{problem},workers}

def apply_context(~Element: Data,~Context: Data,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,context: Context,name: String,operations: F32,tensor: Tensor<Element>) -> Tensor<Element>:
  match tensor:
    case Tensor{storage,shape,fill,status,workers}: apply_status_context(~Element,~Context,~mapper,context,name,status,storage,shape,fill,workers)
    case Banded{bands,geometry,shape,fill,status,workers}: apply_bands_context(~Element,~Context,~mapper,context,operations,status,bands,geometry,shape,fill,workers)

def apply(~Element: Data,~mapper: @+count: Nat -> @+position: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element>,name: String,operations: F32,tensor: Tensor<Element>) -> Tensor<Element>:
  apply_context(~Element,~Unit,~Operations.without_context(~Element,~mapper),Unit{},name,operations,tensor)


# The function also receives each element's row-major index.
def indexed(~Element: Data,+shape: List<&2,U32>,+fill: Element,+workers: U32,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) -> Tensor<Element>:
  match result:
    case Operations.Accepted{(input,output)}: Tensor{output,shape,fill,Usable{},workers}
    case Operations.Rejected{(input,output)}: Tensor{input,shape,fill,Failed{Problem{"apply_indexed",InvalidShape{shape}}},workers}

def apply_indexed_count(~Element: Data,~operation: U32 -> Element -> Element,count: Maybe<&2,U32>,view: Maybe<&2,Views.View>,
  storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensor<Element>:
  match count view:
    case Some{count} Some{view}:
      indexed(~Element,shape,fill,workers,Operations.apply_indexed(~Element,~operation,storage,view,buffer(~Element,fill,count)))
    case count view: Tensor{storage,shape,fill,Failed{Problem{"apply_indexed",InvalidShape{shape}}},workers}

def apply_indexed_status(~Element: Data,~operation: U32 -> Element -> Element,status: Status,storage: Storage.Buffer<Element>,
  +shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensor<Element>:
  match status:
    case Usable{}: apply_indexed_count(~Element,~operation,count_of(shape),view_of(shape),storage,shape,fill,workers)
    case Failed{problem}: Tensor{storage,shape,fill,Failed{problem},workers}

def indexed_width_positive(geometry: Bands.Geometry) -> Bool:
  Bands.Geometry{batch,channels,height,width} = geometry
  Nat.is_lt(0n,U32.to_nat(width))

# Empty widths use the existing dense indexed operation after one gather.
# The choice is made once, before the band scheduler or element loop.
def apply_indexed_positive(~Element: Data,~operation: U32 -> Element -> Element,positive: Bool,operations: F32,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,
  +shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensor<Element>:
  match positive:
    case True{}: Banded{BandSchedule.indexed(~Element,~operation,geometry,workers,operations,fill,bands),geometry,shape,fill,Usable{},workers}
    case False{}:
      with_dense(~Element,~Tensor<Element>,Banded{bands,geometry,shape,fill,Usable{},workers},
        storage => shape => fill => status => workers =>
          apply_indexed_status(~Element,~operation,status,storage,shape,fill,workers))

def apply_indexed_bands(~Element: Data,~operation: U32 -> Element -> Element,operations: F32,status: Status,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,
  shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensor<Element>:
  match status:
    case Usable{}: apply_indexed_positive(~Element,~operation,indexed_width_positive(geometry),operations,bands,geometry,shape,fill,workers)
    case Failed{problem}: Banded{bands,geometry,shape,fill,Failed{problem},workers}

def apply_indexed(~Element: Data,~operation: U32 -> Element -> Element,operations: F32,tensor: Tensor<Element>) -> Tensor<Element>:
  match tensor:
    case Tensor{storage,shape,fill,status,workers}: apply_indexed_status(~Element,~operation,status,storage,shape,fill,workers)
    case Banded{bands,geometry,shape,fill,status,workers}: apply_indexed_bands(~Element,~operation,operations,status,bands,geometry,shape,fill,workers)

# Reductions visit elements in row-major order.
def reduced(~Element: Data,~Accumulator: Data,name: String,+shape: List<&2,U32>,result: Operations.Outcome<(Storage.Buffer<Element> & Accumulator)>) ->
  Result<&2,&2,Problem,Accumulator>:
  match result:
    case Operations.Accepted{(storage,value)}: Done{value}
    case Operations.Rejected{(storage,value)}: Fail{Problem{name,InvalidShape{shape}}}

def reduce_view(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,name: String,view: Maybe<&2,Views.View>,
  storage: Storage.Buffer<Element>,+shape: List<&2,U32>,initial: Accumulator) -> Result<&2,&2,Problem,Accumulator>:
  match view:
    case Some{view}: reduced(~Element,~Accumulator,name,shape,Operations.reduce(~Element,~Accumulator,~operation,storage,view,initial))
    case None{}: Fail{Problem{name,InvalidShape{shape}}}

def reduce_status(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,name: String,status: Status,
  storage: Storage.Buffer<Element>,+shape: List<&2,U32>,initial: Accumulator) -> Result<&2,&2,Problem,Accumulator>:
  match status:
    case Usable{}: reduce_view(~Element,~Accumulator,~operation,name,view_of(shape),storage,shape,initial)
    case Failed{problem}: Fail{problem}

def reduce(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,name: String,tensor: Tensor<Element>,initial: Accumulator) ->
  Result<&2,&2,Problem,Accumulator>:
  with_dense(~Element,~Result<&2,&2,Problem,Accumulator>,tensor,storage => shape => fill => status => workers =>
    reduce_status(~Element,~Accumulator,~operation,name,status,storage,shape,initial))

# A reduction as a rank-0 tensor, so a failure propagates like any result.
def scalar_result(~Element: Data,+fill: Element,+count: U32,result: Result<&2,&2,Problem,Element>) -> Tensor<Element>:
  match result:
    case Done{value}: with_workers(~Element,full(~Element,[],value),count)
    case Fail{problem}: Tensor{buffer(~Element,fill,0),[],fill,Failed{problem},count}

def reduce_counted(~Element: Data,~operation: Element -> Element -> Element,name: String,pair: Tensor<Element> & U32,+initial: Element) -> Tensor<Element>:
  (tensor,+count) = pair
  scalar_result(~Element,initial,count,reduce(~Element,~Element,~operation,name,tensor,initial))

def reduce_all(~Element: Data,~operation: Element -> Element -> Element,name: String,tensor: Tensor<Element>,+initial: Element) -> Tensor<Element>:
  reduce_counted(~Element,~operation,name,workers(~Element,tensor),initial)

# Scalar operands are read through a stride-zero view of one element.
def broadcast_axes(axes: List<&2,Views.Axis>) -> List<&2,Views.Axis>:
  match axes:
    case []: []
    case Views.Axis{extent,stride} <> rest: Views.Axis{extent,0} <> broadcast_axes(rest)

def broadcast(view: Views.View) -> Views.View:
  Views.View{axes,offset} = view
  Views.View{broadcast_axes(axes),0}

def scalar_applied(~Element: Data,name: String,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) -> Tensor<Element>:
  match result:
    case Operations.Accepted{(storage,scalar)}: Tensor{storage,shape,fill,Usable{},workers}
    case Operations.Rejected{(storage,scalar)}: Tensor{storage,shape,fill,Failed{Problem{name,InvalidShape{shape}}},workers}

def scalar_view(~Element: Data,~operation: Element -> Element -> Element,name: String,view: Maybe<&2,Views.View>,
  storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+value: Element,+workers: U32) -> Tensor<Element>:
  match view:
    case Some{+view}:
      scalar_applied(~Element,name,shape,fill,workers,Operations.combine(~Element,~operation,storage,view,buffer(~Element,value,1),broadcast(view)))
    case None{}: Tensor{storage,shape,fill,Failed{Problem{name,InvalidShape{shape}}},workers}

def scalar_status(~Element: Data,~operation: Element -> Element -> Element,name: String,status: Status,
  storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+value: Element,+workers: U32) -> Tensor<Element>:
  match status:
    case Usable{}: scalar_view(~Element,~operation,name,view_of(shape),storage,shape,fill,value,workers)
    case Failed{problem}: Tensor{storage,shape,fill,Failed{problem},workers}

# Updates every element x to operation(x, value).
def scalar_bands(~Element: Data,~operation: Element -> Element -> Element,operations: F32,status: Status,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,
  shape: List<&2,U32>,fill: Element,+workers: U32,value: Element) -> Tensor<Element>:
  match status:
    case Usable{}: Banded{BandSchedule.scalar(~Element,~operation,geometry,workers,operations,value,bands),geometry,shape,fill,Usable{},workers}
    case Failed{problem}: Banded{bands,geometry,shape,fill,Failed{problem},workers}

def combine_scalar(~Element: Data,~operation: Element -> Element -> Element,name: String,operations: F32,tensor: Tensor<Element>,+value: Element) -> Tensor<Element>:
  match tensor:
    case Tensor{storage,shape,fill,status,workers}: scalar_status(~Element,~operation,name,status,storage,shape,fill,value,workers)
    case Banded{bands,geometry,shape,fill,status,workers}:
      scalar_bands(~Element,~operation,operations,status,bands,geometry,shape,fill,workers,value)

# Messages

def show_extents(extents: List<&2,U32>) -> String:
  match extents:
    case []: ""
    case [extent]: U32.show(extent)
    case extent <> rest: String.append(U32.show(extent),String.append(", ",show_extents(rest)))

def show_shape(extents: List<&2,U32>) -> String:
  String.append("[",String.append(show_extents(extents),"]"))

def explain(reason: Reason) -> String:
  match reason:
    case ShapeMismatch{left,right}:
      String.append("shapes ",String.append(show_shape(left),String.append(" and ",String.append(show_shape(right)," do not match"))))
    case InvalidShape{shape}:
      String.append("shape ",String.append(show_shape(shape)," is invalid or exceeds the machine capacity"))
    case InvalidArgument{shape}:
      String.append("invalid argument for shape ",show_shape(shape))
    case FileFault{path,detail}:
      String.append("file ",String.append(path,String.append(" ",detail)))
    case MissingName{name}:
      String.append("no tensor named ",name)

def describe(problem: Problem) -> String:
  Problem{operation,reason} = problem
  String.append(operation,String.append(": ",explain(reason)))

def expected(~Element: Data,result: Result<&1,&1,Tensor<Element> & Problem,Tensor<Element>>) -> IO(Tensor<Element>):
  match result:
    case Done{tensor}: IO.pure(Tensor<Element>,tensor)
    case Fail{(tensor,problem)}: IO.die(Tensor<Element>,1,describe(problem))

# Stops the program with the recorded problem, for scripts and examples.
def expect(~Element: Data,tensor: Tensor<Element>) -> IO(Tensor<Element>):
  expected(~Element,check(~Element,tensor))

def expected_storage(~Element: Data,result: Result<&2,&1,Problem,Storage.Buffer<Element>>) -> IO(Storage.Buffer<Element>):
  match result:
    case Done{storage}: IO.pure(Storage.Buffer<Element>,storage)
    case Fail{problem}: IO.die(Storage.Buffer<Element>,1,describe(problem))

def expect_storage(~Element: Data,tensor: Tensor<Element>) -> IO(Storage.Buffer<Element>):
  expected_storage(~Element,into_storage(~Element,tensor))

# Read-only operations return their tensor in a pair; Bend destructures only
# parameters, so a continuation receives both halves.
def then(~First: Type,~Second: Type,~Result: Type,pair: First & Second,body: First -> Second -> Result) -> Result:
  (first,second) = pair
  body(first,second)
