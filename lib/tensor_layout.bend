# Layout operations consume their arguments. Copy and nearest upsampling
# preserve row bands. Other layout operations first obtain one dense owner;
# reshape relabels it, while the remaining operations use checked view copies.
import Base
import ./storage_buffer.bend as Storage
import ./tensor_shape.bend as Shapes
import ./tensor_view.bend as Views
import ./tensor_operations.bend as Operations
import ./tensor.bend as Tensors
import ./tensor_band.bend as Bands
import ./traversal_partition.bend as Partition
import ./tensor_band_concat.bend as BandConcat
import ./tensor_band_schedule.bend as BandSchedule

def extents(view: Maybe<&2,Views.View>) -> List<&2,U32>:
  match view:
    case Some{view}: Views.axis_extents_of(view)
    case None{}: []

def naturals(values: List<&2,U32>) -> List<&2,Nat>:
  match values:
    case []: []
    case value <> rest: U32.to_nat(value) <> naturals(rest)

def copied(~Element: Data,name: String,+source_shape: List<&2,U32>,+target_shape: List<&2,U32>,+fill: Element,+workers: U32,
  result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) -> Tensors.Tensor<Element> & Tensors.Tensor<Element>:
  match result:
    case Operations.Accepted{(input,output)}:
      (Tensors.Tensor{output,target_shape,fill,Tensors.Ready{},workers},Tensors.Tensor{input,source_shape,fill,Tensors.Ready{},workers})
    case Operations.Rejected{(input,output)}:
      (Tensors.Tensor{output,target_shape,fill,Tensors.Failed{Tensors.Problem{name,Tensors.InvalidArgument{source_shape}}},workers},
        Tensors.Tensor{input,source_shape,fill,Tensors.Ready{},workers})

def gather_views(~Element: Data,name: String,source_view: Maybe<&2,Views.View>,target_view: Maybe<&2,Views.View>,count: Maybe<&2,U32>,
  input: Storage.Buffer<Element>,+source_shape: List<&2,U32>,+target_shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  Tensors.Tensor<Element> & Tensors.Tensor<Element>:
  match source_view target_view count:
    case Some{source_view} Some{target_view} Some{count}:
      copied(~Element,name,source_shape,target_shape,fill,workers,Operations.copy(~Element,input,source_view,Tensors.buffer(~Element,fill,count),target_view))
    case source_view target_view count:
      (Tensors.failed(~Element,name,Tensors.InvalidArgument{source_shape},target_shape,fill,workers),Tensors.Tensor{input,source_shape,fill,Tensors.Ready{},workers})

# Copy the elements selected by source_view, in its row-major order, into a
# new tensor whose storage is laid out as view_shape and labelled target_shape.
def gather_status(~Element: Data,name: String,status: Tensors.Status,source_view: Maybe<&2,Views.View>,+view_shape: List<&2,U32>,
  +target_shape: List<&2,U32>,input: Storage.Buffer<Element>,+source_shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  Tensors.Tensor<Element> & Tensors.Tensor<Element>:
  match status:
    case Tensors.Ready{}:
      gather_views(~Element,name,source_view,Tensors.view_of(view_shape),Tensors.count_of(view_shape),input,source_shape,target_shape,fill,workers)
    case Tensors.Failed{+problem}:
      (Tensors.Tensor{Tensors.buffer(~Element,fill,0),target_shape,fill,Tensors.Failed{problem},workers},
        Tensors.Tensor{input,source_shape,fill,Tensors.Failed{problem},workers})

def result_of(~Element: Data,pair: Tensors.Tensor<Element> & Tensors.Tensor<Element>) -> Tensors.Tensor<Element>:
  (result,source) = pair
  result

def swapped(~Element: Data,pair: Tensors.Tensor<Element> & Tensors.Tensor<Element>) -> Tensors.Tensor<Element> & Tensors.Tensor<Element>:
  (result,source) = pair
  (source,result)

# The only layout operation that keeps its argument: (original, copy).
def copy_bands(~Element: Data,status: Tensors.Status,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  Tensors.Tensor<Element> & Tensors.Tensor<Element>:
  match status:
    case Tensors.Ready{}:
      Storage.bind_pair(~Bands.Bands(~Element),~Bands.Bands(~Element),~(Tensors.Tensor<Element> & Tensors.Tensor<Element>),
        Bands.copy(~Element,bands),left => right =>
          (Tensors.Banded{left,geometry,shape,fill,Tensors.Ready{},workers},Tensors.Banded{right,geometry,shape,fill,Tensors.Ready{},workers}))
    case Tensors.Failed{+problem}:
      (Tensors.Banded{bands,geometry,shape,fill,Tensors.Failed{problem},workers},
        Tensors.Tensor{Tensors.buffer(~Element,fill,0),shape,fill,Tensors.Failed{problem},workers})

def copy(~Element: Data,tensor: Tensors.Tensor<Element>) -> Tensors.Tensor<Element> & Tensors.Tensor<Element>:
  match tensor:
    case Tensors.Tensor{storage,+shape,fill,status,workers}:
      swapped(~Element,gather_status(~Element,"copy",status,Tensors.view_of(shape),shape,shape,storage,shape,fill,workers))
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}: copy_bands(~Element,status,bands,geometry,shape,fill,workers)

def permuted(view: Maybe<&2,Views.View>,order: List<&2,Nat>) -> Maybe<&2,Views.View>:
  match view:
    case Some{view}: Views.permute(view,order)
    case None{}: None{}

def transpose_view(~Element: Data,+source: Maybe<&2,Views.View>,status: Tensors.Status,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,fill: Element,
  +workers: U32) -> Tensors.Tensor<Element> & Tensors.Tensor<Element>:
  gather_status(~Element,"transpose",status,source,extents(source),extents(source),storage,shape,fill,workers)

# The result's axis i is the source axis order[i].
def transpose(~Element: Data,tensor: Tensors.Tensor<Element>,order: List<&2,U32>) -> Tensors.Tensor<Element>:
  Tensors.with_dense(~Element,~Tensors.Tensor<Element>,tensor,storage => shape => fill => status => workers =>
    +shape = shape
    result_of(~Element,transpose_view(~Element,permuted(Tensors.view_of(shape),naturals(order)),status,storage,shape,fill,workers)))

def sliced(view: Maybe<&2,Views.View>,axis: U32,start: U32,count: U32,step: U32) -> Maybe<&2,Views.View>:
  match view:
    case Some{view}: Views.slice(view,U32.to_nat(axis),start,count,step)
    case None{}: None{}

def slice_view(~Element: Data,+source: Maybe<&2,Views.View>,status: Tensors.Status,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,fill: Element,
  +workers: U32) -> Tensors.Tensor<Element> & Tensors.Tensor<Element>:
  gather_status(~Element,"slice",status,source,extents(source),extents(source),storage,shape,fill,workers)

# Elements start, start + step, ... (count of them) along one axis.
def slice(~Element: Data,tensor: Tensors.Tensor<Element>,axis: U32,start: U32,count: U32,step: U32) -> Tensors.Tensor<Element>:
  Tensors.with_dense(~Element,~Tensors.Tensor<Element>,tensor,storage => shape => fill => status => workers =>
    +shape = shape
    result_of(~Element,slice_view(~Element,sliced(Tensors.view_of(shape),axis,start,count,step),status,storage,shape,fill,workers)))

def same_count(left: Maybe<&2,U32>,right: Maybe<&2,U32>) -> Bool:
  match left right:
    case Some{left} Some{right}: U32.is_eq(left,right)
    case left right: False{}

def relabel_ready(~Element: Data,accepted: Bool,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+target: List<&2,U32>,fill: Element,
  workers: U32) -> Tensors.Tensor<Element>:
  match accepted:
    case True{}: Tensors.Tensor{storage,target,fill,Tensors.Ready{},workers}
    case False{}: Tensors.Tensor{storage,shape,fill,Tensors.Failed{Tensors.Problem{"reshape",Tensors.ShapeMismatch{shape,target}}},workers}

def relabel(~Element: Data,accepted: Bool,current: Tensors.Status,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+target: List<&2,U32>,fill: Element,
  workers: U32) -> Tensors.Tensor<Element>:
  match current:
    case Tensors.Ready{}: relabel_ready(~Element,accepted,storage,shape,target,fill,workers)
    case Tensors.Failed{problem}: Tensors.Tensor{storage,shape,fill,Tensors.Failed{problem},workers}

# Same elements in the same row-major order under new extents.
def reshape(~Element: Data,tensor: Tensors.Tensor<Element>,+shape: List<&2,U32>) -> Tensors.Tensor<Element>:
  Tensors.with_dense(~Element,~Tensors.Tensor<Element>,tensor,storage => source_shape => fill => status => workers =>
    +source_shape = source_shape
    relabel(~Element,same_count(Tensors.count_of(source_shape),Tensors.count_of(shape)),status,storage,source_shape,shape,fill,workers))

# Nearest-neighbour upsampling repeats each element of the last two axes
# scale times, by copying from a view whose repeated axes have stride zero.
def prepend_repeated(axis: Views.Axis,rest: Maybe<&2,List<&2,Views.Axis>>) -> Maybe<&2,List<&2,Views.Axis>>:
  match rest:
    case Some{axes}: Some{axis <> axes}
    case None{}: None{}

def repeated_axes(axes: List<&2,Views.Axis>,+scale: U32) -> Maybe<&2,List<&2,Views.Axis>>:
  match axes:
    case [Views.Axis{height,height_stride},Views.Axis{width,width_stride}]:
      Some{[Views.Axis{height,height_stride},Views.Axis{scale,0},Views.Axis{width,width_stride},Views.Axis{scale,0}]}
    case [axis] : None{}
    case []: None{}
    case axis <> rest: prepend_repeated(axis,repeated_axes(rest,scale))

def with_offset(axes: Maybe<&2,List<&2,Views.Axis>>,offset: U32) -> Maybe<&2,Views.View>:
  match axes:
    case Some{axes}: Some{Views.View{axes,offset}}
    case None{}: None{}

def repeated(view: Maybe<&2,Views.View>,scale: U32) -> Maybe<&2,Views.View>:
  match view:
    case Some{Views.View{axes,offset}}: with_offset(repeated_axes(axes,scale),offset)
    case None{}: None{}

def scaled_extents(extents: List<&2,U32>,+scale: U32) -> List<&2,U32>:
  match extents:
    case [height,width]: [(height * scale : U32),(width * scale : U32)]
    case extent <> rest: extent <> scaled_extents(rest,scale)
    case []: []

def upsample_view(~Element: Data,+source: Maybe<&2,Views.View>,status: Tensors.Status,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+scale: U32,fill: Element,
  +workers: U32) -> Tensors.Tensor<Element> & Tensors.Tensor<Element>:
  gather_status(~Element,"upsample_nearest",status,source,extents(source),scaled_extents(shape,scale),storage,shape,fill,workers)

type Upsample<-Element: Data> is Data:
  Upsample{geometry: Bands.Geometry,scale: U32,fill: Element}

def upsample_leaf(~Element: Data,context: Upsample<Element>,first: U32,rows: U32,input: Storage.Buffer<Element>) -> Bands.Bands(~Element):
  Upsample{Bands.Geometry{batch,channels,height,width},+scale,fill} = context
  +rows = rows
  +shape = {[channels,rows,width] : List<&2,U32>}
  +first = (first * scale : U32)
  +rows = (rows * scale : U32)
  Tensors.with_dense(~Element,~Bands.Bands(~Element),
    result_of(~Element,upsample_view(~Element,repeated(Tensors.view_of(shape),scale),Tensors.Ready{},input,shape,scale,fill,1)),
    output => shape => fill => status => workers => Partition.Leaf{output,first,rows,Unit{}})

def upsample_supported(~Element: Data,valid: Bool,bands: Bands.Bands(~Element),geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,
  +fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match valid geometry:
    case False{} geometry:
      Tensors.with_dense(~Element,~Tensors.Tensor<Element>,Tensors.Banded{bands,geometry,shape,fill,Tensors.Ready{},workers},
        storage => shape => fill => status => workers =>
          +shape = shape
          result_of(~Element,upsample_view(~Element,repeated(Tensors.view_of(shape),scale),status,storage,shape,scale,fill,workers)))
    case True{} Bands.Geometry{+batch,+channels,+height,+width}:
      Tensors.Banded{BandSchedule.transform(~Element,~Upsample<Element>,~upsample_leaf(~Element),Upsample{Bands.Geometry{batch,channels,height,width},scale,fill},
          Bands.Geometry{batch,channels,height,width},workers,(U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),bands),
        Bands.Geometry{batch,channels,(height * scale : U32),(width * scale : U32)},scaled_extents(shape,scale),fill,Tensors.Ready{},workers}

def upsample_bands(~Element: Data,status: Tensors.Status,bands: Bands.Bands(~Element),geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,
  +fill: Element,workers: U32) -> Tensors.Tensor<Element>:
  match status geometry:
    case Tensors.Failed{+problem} geometry:
      Tensors.with_dense(~Element,~Tensors.Tensor<Element>,Tensors.Banded{bands,geometry,shape,fill,Tensors.Failed{problem},workers},
        storage => shape => fill => status => workers =>
          +shape = shape
          result_of(~Element,upsample_view(~Element,repeated(Tensors.view_of(shape),scale),status,storage,shape,scale,fill,workers)))
    case Tensors.Ready{} Bands.Geometry{batch,channels,+height,+width}:
      upsample_supported(~Element,Bool.not(U32.is_zero(scale)) && Views.has_count(Shapes.checked_multiply(height,scale)) &&
        Views.has_count(Shapes.checked_multiply(width,scale)) && Views.has_count(Tensors.count_of(scaled_extents(shape,scale))) &&
        Bool.not(Shapes.has_zero(scaled_extents(shape,scale))),
        bands,Bands.Geometry{batch,channels,height,width},shape,scale,fill,workers)

def upsample_nearest(~Element: Data,tensor: Tensors.Tensor<Element>,+scale: U32) -> Tensors.Tensor<Element>:
  match tensor:
    case Tensors.Tensor{storage,+shape,fill,status,workers}:
      result_of(~Element,upsample_view(~Element,repeated(Tensors.view_of(shape),scale),status,storage,shape,scale,fill,workers))
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}: upsample_bands(~Element,status,bands,geometry,shape,scale,fill,workers)

def settle(~Element: Data,valid: Bool,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) ->
  (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool:
  match result:
    case Operations.Accepted{owners}: (owners,valid)
    case Operations.Rejected{owners}: (owners,False{})

def count_or_zero(count: Maybe<&2,U32>) -> U32:
  match count:
    case Some{count}: count
    case None{}: 0

def extent_at(extents: List<&2,U32>,axis: Nat) -> U32:
  count_or_zero(List.get(&2,U32,extents,axis))

def settled(~Element: Data,name: String,valid: Bool,part: Storage.Buffer<Element>,target: Storage.Buffer<Element>,+target_shape: List<&2,U32>,+part_shape: List<&2,U32>,
  target_fill: Element,part_fill: Element,+workers: U32) -> Tensors.Tensor<Element> & Tensors.Tensor<Element>:
  match valid:
    case True{}: (Tensors.Tensor{target,target_shape,target_fill,Tensors.Ready{},workers},Tensors.Tensor{part,part_shape,part_fill,Tensors.Ready{},workers})
    case False{}:
      (Tensors.Tensor{target,target_shape,target_fill,Tensors.Failed{Tensors.Problem{name,Tensors.ShapeMismatch{target_shape,part_shape}}},workers},
        Tensors.Tensor{part,part_shape,part_fill,Tensors.Ready{},workers})

def placed(~Element: Data,name: String,+target_shape: List<&2,U32>,+part_shape: List<&2,U32>,target_fill: Element,part_fill: Element,+workers: U32,
  result: (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool) -> Tensors.Tensor<Element> & Tensors.Tensor<Element>:
  (owners,valid) = result
  (part,target) = owners
  settled(~Element,name,valid,part,target,target_shape,part_shape,target_fill,part_fill,workers)

def same_extents(slice: Views.View,part_shape: List<&2,U32>) -> Bool:
  Shapes.same(Views.shape(slice),Shapes.Shape{part_shape})

# The part written through a view of the target storage: one checked copy,
# which rejects a view whose extents differ from the part. A part with no
# element writes nothing; its extents are compared here.
def place(~Element: Data,name: String,empty: Bool,slice: Maybe<&2,Views.View>,source: Maybe<&2,Views.View>,
  target: Storage.Buffer<Element>,+target_shape: List<&2,U32>,target_fill: Element,
  part: Storage.Buffer<Element>,+part_shape: List<&2,U32>,part_fill: Element,+workers: U32) -> Tensors.Tensor<Element> & Tensors.Tensor<Element>:
  match empty slice source:
    case False{} Some{slice} Some{source}:
      placed(~Element,name,target_shape,part_shape,target_fill,part_fill,workers,settle(~Element,True{},Operations.scatter(~Element,part,source,target,slice)))
    case True{} Some{slice} source:
      settled(~Element,name,same_extents(slice,part_shape),part,target,target_shape,part_shape,target_fill,part_fill,workers)
    case empty slice source: settled(~Element,name,False{},part,target,target_shape,part_shape,target_fill,part_fill,workers)

def place_status(~Element: Data,name: String,target_status: Tensors.Status,part_status: Tensors.Status,slice: Maybe<&2,Views.View>,
  target: Storage.Buffer<Element>,+target_shape: List<&2,U32>,+target_fill: Element,
  part: Storage.Buffer<Element>,+part_shape: List<&2,U32>,+part_fill: Element,+workers: U32) -> Tensors.Tensor<Element> & Tensors.Tensor<Element>:
  match target_status part_status:
    case Tensors.Ready{} Tensors.Ready{}:
      place(~Element,name,Shapes.has_zero(part_shape),slice,Tensors.view_of(part_shape),target,target_shape,target_fill,part,part_shape,part_fill,workers)
    case Tensors.Ready{} Tensors.Failed{+problem}:
      (Tensors.Tensor{target,target_shape,target_fill,Tensors.Failed{problem},workers},Tensors.Tensor{part,part_shape,part_fill,Tensors.Failed{problem},workers})
    case Tensors.Failed{problem} part_status:
      (Tensors.Tensor{target,target_shape,target_fill,Tensors.Failed{problem},workers},Tensors.Tensor{part,part_shape,part_fill,part_status,workers})

# target[..., start : start + part extent, ...] = part along axis: the part is
# written through the slice of the target that it replaces.
def set_slice(~Element: Data,target: Tensors.Tensor<Element>,+axis: U32,start: U32,part: Tensors.Tensor<Element>) -> Tensors.Tensor<Element>:
  Tensors.with_dense(~Element,~Tensors.Tensor<Element>,target,target => target_shape => target_fill => target_status => target_workers =>
    +target_shape = target_shape
    Tensors.with_dense(~Element,~Tensors.Tensor<Element>,part,part => part_shape => part_fill => part_status => part_workers =>
      +part_shape = part_shape
      result_of(~Element,place_status(~Element,"set_slice",target_status,part_status,
        sliced(Tensors.view_of(target_shape),axis,start,extent_at(part_shape,U32.to_nat(axis)),1),target,target_shape,target_fill,part,part_shape,part_fill,
        U32.max(target_workers,part_workers)))))

# Concatenation along an axis.
type Survey is Data:
  Survey{valid: Bool,total: U32,status: Tensors.Status,workers: U32}

def surveyed(survey: Survey,+first: List<&2,U32>,+shape: List<&2,U32>,+axis: Nat,status: Tensors.Status,workers: U32) -> Survey:
  Survey{valid,+total,previous,most} = survey
  +width = extent_at(shape,axis)
  Survey{valid && Nat.is_eq(List.length(&2,U32,first),List.length(&2,U32,shape)) &&
      Shapes.same_extents(List.take(&2,U32,first,axis),List.take(&2,U32,shape,axis)) &&
      Shapes.same_extents(List.drop(&2,U32,first,1n+axis),List.drop(&2,U32,shape,1n+axis)) &&
      U32.is_le(width,(4294967295 - total : U32)),
    (total + width : U32),Tensors.first_failure(previous,status),U32.max(most,workers)}

def prepend_survey(~Element: Data,tensor: Tensors.Tensor<Element>,result: Tensors.Parts<Element> & Survey) -> Tensors.Parts<Element> & Survey:
  (rest,summary) = result
  (Tensors.Cons{tensor,rest},summary)

# Summarizes the extents of every part and returns the parts in order.
def survey(~Element: Data,parts: Tensors.Parts<Element>,+first: List<&2,U32>,+axis: Nat,summary: Survey) -> Tensors.Parts<Element> & Survey:
  match parts:
    case Tensors.Single{tensor}:
      Tensors.with_metadata(~Element,~(Tensors.Parts<Element> & Survey),tensor,tensor => shape => fill => status => workers =>
        +shape = shape
        +status = status
        +workers = workers
        (Tensors.Single{tensor},surveyed(summary,first,shape,axis,status,workers)))
    case Tensors.Cons{tensor,rest}:
      Tensors.with_metadata(~Element,~(Tensors.Parts<Element> & Survey),tensor,tensor => shape => fill => status => workers =>
        +shape = shape
        +status = status
        +workers = workers
        prepend_survey(~Element,tensor,survey(~Element,rest,first,axis,surveyed(summary,first,shape,axis,status,workers))))

def with_extent(extents: List<&2,U32>,axis: Nat,+extent: U32) -> List<&2,U32>:
  List.set(&2,U32,extents,axis,extent)

def append_at(~Element: Data,+axis: U32,+start: U32,output: Tensors.Tensor<Element>,part: Tensors.Tensor<Element>) -> Tensors.Tensor<Element> & U32:
  Tensors.with_dense(~Element,~(Tensors.Tensor<Element> & U32),part,storage => shape => fill => status => workers =>
    +shape = shape
    (set_slice(~Element,output,axis,start,Tensors.Tensor{storage,shape,fill,status,workers}),(start + extent_at(shape,U32.to_nat(axis)) : U32)))

def append_part(~Element: Data,+axis: U32,state: Tensors.Tensor<Element> & U32,part: Tensors.Tensor<Element>) -> Tensors.Tensor<Element> & U32:
  (output,start) = state
  append_at(~Element,axis,start,output,part)

def appended_all(~Element: Data,state: Tensors.Tensor<Element> & U32) -> Tensors.Tensor<Element>:
  (output,start) = state
  output

def append_parts(~Element: Data,parts: Tensors.Parts<Element>,+axis: U32,state: Tensors.Tensor<Element> & U32) -> Tensors.Tensor<Element>:
  match parts:
    case Tensors.Single{part}: appended_all(~Element,append_part(~Element,axis,state,part))
    case Tensors.Cons{part,rest}: append_parts(~Element,rest,axis,append_part(~Element,axis,state,part))

def join_selected(~Element: Data,geometry: Maybe<&2,Bands.Geometry>,count: Maybe<&2,U32>,shape: List<&2,U32>,axis: U32,fill: Element,workers: U32,
  layout: Maybe<&2,Bands.Plan>,parts: Tensors.Parts<Element>) -> Tensors.Tensor<Element>:
  match geometry count layout:
    case Some{geometry} Some{count} Some{layout}: BandConcat.join(~Element,layout,geometry,parts,shape,fill,workers)
    case geometry count layout: append_parts(~Element,parts,axis,(Tensors.with_workers(~Element,Tensors.full(~Element,shape,fill),workers),0))

def join_discovered(~Element: Data,+shape: List<&2,U32>,+axis: U32,fill: Element,workers: U32,
  discovered: Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) -> Tensors.Tensor<Element>:
  (parts,layout) = discovered
  join_selected(~Element,BandConcat.geometry(shape,axis),Tensors.count_of(shape),shape,axis,fill,workers,layout,parts)

# The final branch is one Bool. Plan comparison stays in the metadata pass,
# outside every element-copy loop. Dense append gathers each band source once.
def join_compatible_selected(~Element: Data,compatible: Bool,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,
  discovered: Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) -> Tensors.Tensor<Element>:
  match compatible:
    case True{}: join_discovered(~Element,shape,axis,fill,workers,discovered)
    case False{}:
      (parts,layout) = discovered
      append_parts(~Element,parts,axis,(Tensors.with_workers(~Element,Tensors.full(~Element,shape,fill),workers),0))

def join_compatible(~Element: Data,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,
  discovered: (Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool) -> Tensors.Tensor<Element>:
  (parts_and_layout,compatible) = discovered
  join_compatible_selected(~Element,compatible && Bool.not(Shapes.has_zero(shape)),shape,axis,fill,workers,parts_and_layout)

def join_status(~Element: Data,status: Tensors.Status,valid: Bool,+first: List<&2,U32>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,
  parts: Tensors.Parts<Element>) -> Tensors.Tensor<Element>:
  match status valid:
    case Tensors.Ready{} True{}: join_compatible(~Element,shape,axis,fill,workers,BandConcat.discover_compatible(~Element,parts))
    case Tensors.Ready{} False{}: Tensors.failed(~Element,"concat",Tensors.InvalidArgument{first},first,fill,workers)
    case Tensors.Failed{problem} other: Tensors.Tensor{Tensors.buffer(~Element,fill,0),first,fill,Tensors.Failed{problem},workers}

def joined(~Element: Data,+first: List<&2,U32>,+axis: U32,+fill: Element,surveyed: Tensors.Parts<Element> & Survey) -> Tensors.Tensor<Element>:
  (parts,summary) = surveyed
  Survey{valid,total,status,workers} = summary
  join_status(~Element,status,valid,first,with_extent(first,U32.to_nat(axis),total),axis,fill,workers,parts)

def lead(~Element: Data,parts: Tensors.Parts<Element>) -> Tensors.Parts<Element> & (List<&2,U32> & Element):
  match parts:
    case Tensors.Single{tensor}:
      Tensors.with_metadata(~Element,~(Tensors.Parts<Element> & (List<&2,U32> & Element)),tensor,tensor => shape => fill => status => workers =>
        +shape = shape
        +fill = fill
        (Tensors.Single{tensor},(shape,fill)))
    case Tensors.Cons{tensor,rest}:
      Tensors.with_metadata(~Element,~(Tensors.Parts<Element> & (List<&2,U32> & Element)),tensor,tensor => shape => fill => status => workers =>
        +shape = shape
        +fill = fill
        (Tensors.Cons{tensor,rest},(shape,fill)))

def concat_first(~Element: Data,+axis: U32,parts: Tensors.Parts<Element>,+shape: List<&2,U32>,+fill: Element) -> Tensors.Tensor<Element>:
  joined(~Element,shape,axis,fill,
    survey(~Element,parts,shape,U32.to_nat(axis),Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)),0,Tensors.Ready{},1}))

def concat_led(~Element: Data,+axis: U32,led: Tensors.Parts<Element> & (List<&2,U32> & Element)) -> Tensors.Tensor<Element>:
  (parts,first) = led
  (shape,fill) = first
  concat_first(~Element,axis,parts,shape,fill)

# Parts agree on every extent except axis; the result stacks them in order.
def concat(~Element: Data,parts: Tensors.Parts<Element>,+axis: U32) -> Tensors.Tensor<Element>:
  concat_led(~Element,axis,lead(~Element,parts))

# Broadcasting to given extents: one checked copy from a stretched view.
def broadcast_status(~Element: Data,status: Tensors.Status,view: Maybe<&2,Views.View>,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,
  +target: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match status view:
    case Tensors.Ready{} None{}: Tensors.failed(~Element,"broadcast_to",Tensors.ShapeMismatch{shape,target},target,fill,workers)
    case status view: result_of(~Element,gather_status(~Element,"broadcast_to",status,view,target,target,storage,shape,fill,workers))

# The tensor repeated to the extents target under NumPy broadcasting.
def broadcast_to(~Element: Data,tensor: Tensors.Tensor<Element>,+target: List<&2,U32>) -> Tensors.Tensor<Element>:
  Tensors.with_dense(~Element,~Tensors.Tensor<Element>,tensor,storage => shape => fill => status => workers =>
    +shape = shape
    broadcast_status(~Element,status,Tensors.stretched(Tensors.view_of(shape),target),storage,shape,target,fill,workers))

# Reduction along one axis. The output starts at the initial value and takes
# the slices of the axis in increasing index, so each output element is
# operation(... operation(operation(initial, x[0]), x[1]) ..., x[extent - 1]).
def fold_slice(~Element: Data,~operation: Element -> Element -> Element,valid: Bool,slice: Maybe<&2,Views.View>,target: Maybe<&2,Views.View>,
  output: Storage.Buffer<Element>,input: Storage.Buffer<Element>) -> (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool:
  match slice target:
    case Some{slice} Some{target}: settle(~Element,valid,Operations.combine(~Element,~operation,output,target,input,slice))
    case slice target: ((output,input),False{})

# The views are built from the extents at every step: a loop that carried one
# View value through its iterations would make the compiler share every View
# constructor of the program.
def fold_step(~Element: Data,~operation: Element -> Element -> Element,index: U32,axis: U32,shape: List<&2,U32>,unit: List<&2,U32>,
  state: (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool) -> (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool:
  (owners,valid) = state
  (output,input) = owners
  fold_slice(~Element,~operation,valid,sliced(Tensors.view_of(shape),axis,index,1,1),Tensors.view_of(unit),output,input)

def fold_slices(~Element: Data,~operation: Element -> Element -> Element,remaining: Nat,+index: U32,+axis: U32,+shape: List<&2,U32>,+unit: List<&2,U32>,
  state: (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool) -> (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool:
  match remaining:
    case 0n: state
    case 1n+rest:
      fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,fold_step(~Element,~operation,index,axis,shape,unit,state))

def folded_valid(~Element: Data,name: String,valid: Bool,output: Storage.Buffer<Element>,shape: List<&2,U32>,result_shape: List<&2,U32>,fill: Element,
  workers: U32) -> Tensors.Tensor<Element>:
  match valid:
    case True{}: Tensors.Tensor{output,result_shape,fill,Tensors.Ready{},workers}
    case False{}: Tensors.Tensor{output,result_shape,fill,Tensors.Failed{Tensors.Problem{name,Tensors.InvalidArgument{shape}}},workers}

def folded(~Element: Data,name: String,shape: List<&2,U32>,result_shape: List<&2,U32>,fill: Element,workers: U32,
  result: (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool) -> Tensors.Tensor<Element>:
  (owners,valid) = result
  (output,input) = owners
  folded_valid(~Element,name,valid,output,shape,result_shape,fill,workers)

# unit is the shape with extent 1 at the axis: the layout of the output and
# the extents of every slice.
def reduce_axis_count(~Element: Data,~operation: Element -> Element -> Element,name: String,count: Maybe<&2,U32>,+axis: U32,input: Storage.Buffer<Element>,
  +shape: List<&2,U32>,unit: List<&2,U32>,+initial: Element,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match count:
    case Some{count}:
      folded(~Element,name,shape,Shapes.without_axis(shape,U32.to_nat(axis)),fill,workers,
        fold_slices(~Element,~operation,U32.to_nat(extent_at(shape,U32.to_nat(axis))),0,axis,shape,unit,((Tensors.buffer(~Element,initial,count),input),True{})))
    case None{}: Tensors.failed(~Element,name,Tensors.InvalidArgument{shape},Shapes.without_axis(shape,U32.to_nat(axis)),fill,workers)

def reduce_axis_status(~Element: Data,~operation: Element -> Element -> Element,name: String,status: Tensors.Status,valid: Bool,+axis: U32,
  input: Storage.Buffer<Element>,+shape: List<&2,U32>,+initial: Element,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match status valid:
    case Tensors.Ready{} True{}:
      reduce_axis_count(~Element,~operation,name,Tensors.count_of(with_extent(shape,U32.to_nat(axis),1)),axis,input,shape,with_extent(shape,U32.to_nat(axis),1),
        initial,fill,workers)
    case Tensors.Ready{} False{}: Tensors.failed(~Element,name,Tensors.InvalidArgument{shape},shape,fill,workers)
    case Tensors.Failed{problem} valid: Tensors.Tensor{input,shape,fill,Tensors.Failed{problem},workers}

# The result has the extents of the tensor without axis.
def reduce_axis(~Element: Data,~operation: Element -> Element -> Element,name: String,+initial: Element,tensor: Tensors.Tensor<Element>,+axis: U32) ->
  Tensors.Tensor<Element>:
  Tensors.with_dense(~Element,~Tensors.Tensor<Element>,tensor,storage => shape => fill => status => workers =>
    +shape = shape
    reduce_axis_status(~Element,~operation,name,status,Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)),axis,storage,shape,initial,fill,workers))

# Padding. Constant fills the new elements with one value and writes the
# tensor once, whatever the axes; Edge repeats the nearest element of the
# tensor one axis after another, so a corner holds the nearest corner element.
type Border<-Element: Data> is Data:
  Constant{value: Element}
  Edge{}

def border_name(~Element: Data,border: Border<Element>) -> String:
  match border:
    case Constant{value}: "pad"
    case Edge{}: "pad_edge"

def blank(~Element: Data,count: Maybe<&2,U32>,+shape: List<&2,U32>,+value: Element,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match count:
    case Some{count}: Tensors.Tensor{Tensors.buffer(~Element,value,count),shape,fill,Tensors.Ready{},workers}
    case None{}: Tensors.failed(~Element,"pad",Tensors.InvalidShape{shape},shape,fill,workers)

# The elements at one end of axis, repeated count times: (strip, tensor).
def edge_strip(~Element: Data,tensor: Tensors.Tensor<Element>,+axis: U32,from_end: Bool,+count: U32) -> Tensors.Tensor<Element> & Tensors.Tensor<Element>:
  Tensors.with_dense(~Element,~(Tensors.Tensor<Element> & Tensors.Tensor<Element>),tensor,storage => shape => fill => status => workers =>
    +shape = shape
    +strip = with_extent(shape,U32.to_nat(axis),count)
    gather_status(~Element,"pad_edge",status,
      Tensors.stretched(sliced(Tensors.view_of(shape),axis,Bool.pick(U32,from_end,(extent_at(shape,U32.to_nat(axis)) - 1 : U32),0),1,1),strip),
      strip,strip,storage,shape,fill,workers))

def pad_edge_axis(~Element: Data,unchanged: Bool,tensor: Tensors.Tensor<Element>,+axis: U32,+before: U32,+after: U32) -> Tensors.Tensor<Element>:
  match unchanged:
    case True{}: tensor
    case False{}:
      Tensors.then(~Tensors.Tensor<Element>,~Tensors.Tensor<Element>,~Tensors.Tensor<Element>,edge_strip(~Element,tensor,axis,False{},before),first => tensor =>
        Tensors.then(~Tensors.Tensor<Element>,~Tensors.Tensor<Element>,~Tensors.Tensor<Element>,edge_strip(~Element,tensor,axis,True{},after),last => tensor =>
          concat(~Element,Tensors.Cons{first,Tensors.Cons{tensor,Tensors.Single{last}}},axis)))

def pad_edge_axes(~Element: Data,before: List<&2,U32>,after: List<&2,U32>,tensor: Tensors.Tensor<Element>,+axis: U32) -> Tensors.Tensor<Element>:
  match before after:
    case Con{+front,before} Con{+back,after}:
      pad_edge_axes(~Element,before,after,pad_edge_axis(~Element,U32.is_zero(front) && U32.is_zero(back),tensor,axis,front,back),(axis + 1 : U32))
    case before after: tensor

# The view of the padded extents that holds the tensor: on every axis, the
# extent of the tensor starting after the elements added in front.
def interior(before: List<&2,U32>,shape: List<&2,U32>,view: Maybe<&2,Views.View>,+axis: U32) -> Maybe<&2,Views.View>:
  match before shape:
    case Con{front,before} Con{extent,shape}: interior(before,shape,sliced(view,axis,front,extent,1),(axis + 1 : U32))
    case before shape: view

def all_zero(values: List<&2,U32>) -> Bool:
  match values:
    case []: True{}
    case value <> rest: U32.is_zero(value) && all_zero(rest)

# Constant padding: storage of the padded extents filled with the value, then
# the tensor written through the interior view. One copy whatever the axes.
def pad_constant(~Element: Data,unchanged: Bool,+value: Element,tensor: Tensors.Tensor<Element>,+before: List<&2,U32>,+padded: List<&2,U32>) -> Tensors.Tensor<Element>:
  match unchanged:
    case True{}: tensor
    case False{}:
      Tensors.with_dense(~Element,~Tensors.Tensor<Element>,tensor,part => part_shape => part_fill => part_status => part_workers =>
        +part_shape = part_shape
        +part_fill = part_fill
        +part_workers = part_workers
        Tensors.with_dense(~Element,~Tensors.Tensor<Element>,blank(~Element,Tensors.count_of(padded),padded,value,part_fill,part_workers),
          target => target_shape => target_fill => target_status => target_workers =>
            result_of(~Element,place_status(~Element,"pad",target_status,part_status,interior(before,part_shape,Tensors.view_of(padded),0),
              target,padded,target_fill,part,part_shape,part_fill,part_workers))))

def pad_border(~Element: Data,border: Border<Element>,tensor: Tensors.Tensor<Element>,+before: List<&2,U32>,+after: List<&2,U32>,padded: Maybe<&2,List<&2,U32>>) -> Tensors.Tensor<Element>:
  match border padded:
    case Constant{value} Some{padded}: pad_constant(~Element,all_zero(before) && all_zero(after),value,tensor,before,padded)
    case border padded: pad_edge_axes(~Element,before,after,tensor,0)

def padded_count(shape: Maybe<&2,List<&2,U32>>) -> Maybe<&2,U32>:
  match shape:
    case Some{shape}: Tensors.count_of(shape)
    case None{}: None{}

def pad_valid(~Element: Data,+border: Border<Element>,status: Tensors.Status,valid: Bool,tensor: Tensors.Tensor<Element>,shape: List<&2,U32>,
  before: List<&2,U32>,after: List<&2,U32>,padded: Maybe<&2,List<&2,U32>>) -> Tensors.Tensor<Element>:
  match status valid:
    case Tensors.Ready{} True{}: pad_border(~Element,border,tensor,before,after,padded)
    case Tensors.Ready{} False{}: Tensors.with_status(~Element,Tensors.Failed{Tensors.Problem{border_name(~Element,border),Tensors.InvalidArgument{shape}}},tensor)
    case Tensors.Failed{problem} valid: tensor

# before[i] elements are added in front of axis i and after[i] behind it. Both
# lists have one entry per axis and the padded extents must be allocatable.
def pad(~Element: Data,+border: Border<Element>,tensor: Tensors.Tensor<Element>,+before: List<&2,U32>,+after: List<&2,U32>) -> Tensors.Tensor<Element>:
  Tensors.with_metadata(~Element,~Tensors.Tensor<Element>,tensor,tensor => shape => fill => status => workers =>
    +shape = shape
    +padded = Shapes.padded(shape,before,after)
    pad_valid(~Element,border,status,Views.has_count(padded_count(padded)),tensor,shape,before,after,padded))
