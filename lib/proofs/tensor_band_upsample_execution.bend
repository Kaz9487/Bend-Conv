# Nearest upsampling of bands: scaled row words, guards, local descriptors,
# the checked copy of each leaf and whole-tree execution.
import Base
import ../tensor_band.bend as Bands
import ../tensor.bend as Tensors
import ../tensor_layout.bend as Layout
import ../tensor_view.bend as Views
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor
import ../tensor_shape.bend as Shapes
import ./storage_buffer_model.bend as Buffer
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_shape_refinement.bend as ShapeLaws
import ./boolean_logic.bend as Boolean
import ./checked_word_refinement.bend as Checked
import ./storage_tree_model.bend as Tree
import ./tensor_band_count.bend as BandCount
import ./tensor_band_model.bend as Model
import ./tensor_count_refinement.bend as Count
import ./tensor_operations_refinement.bend as OperationsLaws
import ./word_encoding.bend as WordEncoding
import ../storage_buffer.bend as Storage
import ../tensor_operations.bend as Operations
import ../default_machine_limits.bend as Default
import ../machine_limits.bend as Machine
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./natural_addition.bend as Addition
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_shape.bend as Shape
import ./tensor_access_refinement.bend as AccessLaws
import ./tensor_band_plan.bend as BandPlan
import ./tensor_count_completion.bend as CountCompletion
import ./u32_comparison.bend as Comparison
import ./u32_full_range.bend as FullRange
import ./word_multiplication.bend as Multiplication
import ../traversal_partition.bend as Partition
import ./contiguous_copy.bend as Contiguous
import ./nat_to_u32_bounds.bend as Bounds
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ./storage_refinement.bend as StorageLaws
import ./tensor_allocation_depth.bend as Depth
import ./tensor_band_each.bend as Each
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_values.bend as BandValues
import ./tensor_cursor_refinement.bend as CursorLaws
import ./tensor_public_refinement.bend as Public
import ./tensor_visit_refinement.bend as TensorVisit
import ./tensor_view_metadata_refinement.bend as Metadata
import ./traversal_array_refinement.bend as ArrayLaws
import ./traversal_visit_refinement.bend as Visit
import ./write_program.bend as Writes
import ./write_program_readback.bend as Readback
import ../tensor_band_schedule.bend as BandSchedule
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./tensor_copy_values.bend as CopyValues

# Upsampling scales a leaf's rows and the whole image by the same factor.
# The existing global count_of allocation guard therefore bounds each leaf.
# This layer is unbounded arithmetic; native scaled-word equations follow it.
def bounds_rows_within(+first: Nat,+rows: Nat,+images: Nat,+height: Nat,
  contained: {Nat.is_le(Nat.add(first,rows),Nat.mul(images,height)) == True{} : Bool}) ->
  {Nat.is_le(rows,Nat.mul(images,height)) == True{} : Bool}:
  Order.transitive(rows,Nat.add(first,rows),Nat.mul(images,height),
    %Order.commutative(rows,first) : {Nat.is_le(rows,_) == True{} : Bool}
    Order.le_add(rows,first),contained)

def grouped_channels(+images: Nat,+channels: Nat,+height: Nat) ->
  {Nat.mul(Nat.mul(images,height),channels) == Nat.mul(images,Nat.mul(channels,height)) : Nat}:
  Equal.trans(Nat,Nat.mul(Nat.mul(images,height),channels),Nat.mul(images,Nat.mul(height,channels)),Nat.mul(images,Nat.mul(channels,height)),
    RowMajor.assoc(images,height,channels),
    Equal.cong(Nat,Nat,value => Nat.mul(images,value),Nat.mul(height,channels),Nat.mul(channels,height),RowMajor.multiply_commutative(height,channels)))

def scaled_channels(+images: Nat,+channels: Nat,+height: Nat,+rows: Nat,+scale: Nat,
  contained: {Nat.is_le(rows,Nat.mul(images,height)) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(channels,Nat.mul(rows,scale)),Nat.mul(images,Nat.mul(channels,Nat.mul(height,scale)))) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(channels,Nat.mul(rows,scale)),Nat.mul(Nat.mul(rows,scale),channels),RowMajor.multiply_commutative(channels,Nat.mul(rows,scale))) :
    {Nat.is_le(_,Nat.mul(images,Nat.mul(channels,Nat.mul(height,scale)))) == True{} : Bool}
  %grouped_channels(images,channels,Nat.mul(height,scale)) :
    {Nat.is_le(Nat.mul(Nat.mul(rows,scale),channels),_) == True{} : Bool}
  %RowMajor.assoc(images,height,scale) :
    {Nat.is_le(Nat.mul(Nat.mul(rows,scale),channels),Nat.mul(_,channels)) == True{} : Bool}
  RowMajor.mul_le(Nat.mul(rows,scale),Nat.mul(Nat.mul(images,height),scale),channels,RowMajor.mul_le(rows,Nat.mul(images,height),scale,contained))

def local_elements(+images: Nat,+channels: Nat,+height: Nat,+width: Nat,+rows: Nat,+scale: Nat,
  contained: {Nat.is_le(rows,Nat.mul(images,height)) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(channels,Nat.mul(Nat.mul(rows,scale),Nat.mul(width,scale))),
    Nat.mul(images,Nat.mul(channels,Nat.mul(Nat.mul(height,scale),Nat.mul(width,scale))))) == True{} : Bool}:
  %RowMajor.assoc(channels,Nat.mul(rows,scale),Nat.mul(width,scale)) :
    {Nat.is_le(_,Nat.mul(images,Nat.mul(channels,Nat.mul(Nat.mul(height,scale),Nat.mul(width,scale))))) == True{} : Bool}
  %RowMajor.assoc(channels,Nat.mul(height,scale),Nat.mul(width,scale)) :
    {Nat.is_le(Nat.mul(Nat.mul(channels,Nat.mul(rows,scale)),Nat.mul(width,scale)),Nat.mul(images,_)) == True{} : Bool}
  %RowMajor.assoc(images,Nat.mul(channels,Nat.mul(height,scale)),Nat.mul(width,scale)) :
    {Nat.is_le(Nat.mul(Nat.mul(channels,Nat.mul(rows,scale)),Nat.mul(width,scale)),_) == True{} : Bool}
  RowMajor.mul_le(Nat.mul(channels,Nat.mul(rows,scale)),Nat.mul(images,Nat.mul(channels,Nat.mul(height,scale))),Nat.mul(width,scale),scaled_channels(images,channels,height,rows,scale,contained))

# Recover scaled-word and allocation evidence from the common arithmetic
# conditions. The actual band selector also requires a nonempty output shape.
def guard(geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32) -> Bool:
  Bands.Geometry{batch,channels,+guards_height,+guards_width} = geometry
  Bool.not(U32.is_zero(scale)) && Views.has_count(Shapes.checked_multiply(guards_height,scale)) &&
    Views.has_count(Shapes.checked_multiply(guards_width,scale)) && Views.has_count(Tensors.count_of(Layout.scaled_extents(shape,scale)))

def enabled(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32) -> Bool:
  guard(geometry,shape,scale) && Bool.not(Shapes.has_zero(Layout.scaled_extents(shape,scale)))

def legal(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,truth: {enabled(geometry,shape,scale) == True{} : Bool}) ->
  {guard(geometry,shape,scale) == True{} : Bool}:
  Boolean.left(guard(geometry,shape,scale),Bool.not(Shapes.has_zero(Layout.scaled_extents(shape,scale))),truth)

def nonempty_when(empty: Bool,truth: {Bool.not(empty) == True{} : Bool}) -> {empty == False{} : Bool}:
  match empty:
    case False{}: {==}
    case True{}: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(truth))

def nonempty(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,truth: {enabled(geometry,shape,scale) == True{} : Bool}) ->
  {Shapes.has_zero(Layout.scaled_extents(shape,scale)) == False{} : Bool}:
  nonempty_when(Shapes.has_zero(Layout.scaled_extents(shape,scale)),Boolean.right(guard(geometry,shape,scale),Bool.not(Shapes.has_zero(Layout.scaled_extents(shape,scale))),truth))

def guards_height(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{batch,channels,guards_height,guards_width} = geometry
  guards_height

def guards_width(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{batch,channels,guards_height,guards_width} = geometry
  guards_width

def found(+result: Maybe<&2,U32>,truth: {Views.has_count(result) == True{} : Bool}) ->
  {result == Some{Checked.value(result)} : Maybe<&2,U32>}:
  match result:
    case None{}: Empty.absurd({None{} == Some{0} : Maybe<&2,U32>},WordEncoding.false_true(truth))
    case Some{value}: {==}

type Scaled<-extent: U32,-scale: U32> is Data:
  Scaled{value: U32,accepted: {Shapes.checked_multiply(extent,scale) == Some{value} : Maybe<&2,U32>},
    exact: {Nat.mul(U32.to_nat(extent),U32.to_nat(scale)) == U32.to_nat(value) : Nat}}

def scaled(+extent: U32,+scale: U32,truth: {Views.has_count(Shapes.checked_multiply(extent,scale)) == True{} : Bool}) -> Scaled<extent,scale>:
  +value = Checked.value(Shapes.checked_multiply(extent,scale))
  +accepted = found(Shapes.checked_multiply(extent,scale),truth)
  Scaled{value,accepted,Checked.multiply_sound(extent,scale,value,accepted)}

def scaled_value(+extent: U32,+scale: U32,evidence: Scaled<extent,scale>) -> U32:
  Scaled{value,accepted,exact} = evidence
  value

def native_scaled(+extent: U32,+scale: U32,+evidence: Scaled<extent,scale>) ->
  {U32.mul(extent,scale) == scaled_value(extent,scale,evidence) : U32}:
  Scaled{+value,accepted,+exact} = evidence
  Equal.trans(U32,U32.mul(extent,scale),U32.from_nat(Nat.mul(U32.to_nat(extent),U32.to_nat(scale))),value,
    Multiplication.native_multiply(extent,scale),
    %Equal.sym(Nat,Nat.mul(U32.to_nat(extent),U32.to_nat(scale)),U32.to_nat(value),exact) : {U32.from_nat(_) == value : U32}
    Arithmetic.native_roundtrip(value))

def native_exact(+extent: U32,+scale: U32,+evidence: Scaled<extent,scale>) ->
  {Nat.mul(U32.to_nat(extent),U32.to_nat(scale)) == U32.to_nat(U32.mul(extent,scale)) : Nat}:
  Scaled{+value,+accepted,+exact} = evidence
  +evidence = {Scaled{value,accepted,exact} : Scaled<extent,scale>}
  Equal.trans(Nat,Nat.mul(U32.to_nat(extent),U32.to_nat(scale)),U32.to_nat(value),U32.to_nat(U32.mul(extent,scale)),exact,
    Equal.cong(U32,Nat,word => U32.to_nat(word),value,U32.mul(extent,scale),Equal.sym(U32,U32.mul(extent,scale),value,native_scaled(extent,scale,evidence))))

def height_scaled(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,truth: {guard(geometry,shape,scale) == True{} : Bool}) ->
  Scaled<guards_height(geometry),scale>:
  Bands.Geometry{batch,channels,+guards_height,+guards_width} = geometry
  +first = Bool.not(U32.is_zero(scale))
  +height_ok = Views.has_count(Shapes.checked_multiply(guards_height,scale))
  +width_ok = Views.has_count(Shapes.checked_multiply(guards_width,scale))
  +count_ok = Views.has_count(Tensors.count_of(Layout.scaled_extents(shape,scale)))
  scaled(guards_height,scale,Boolean.right(first,height_ok,Boolean.left((first && height_ok),width_ok,Boolean.left(((first && height_ok) && width_ok),count_ok,truth))))

def width_scaled(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,truth: {guard(geometry,shape,scale) == True{} : Bool}) ->
  Scaled<guards_width(geometry),scale>:
  Bands.Geometry{batch,channels,+guards_height,+guards_width} = geometry
  +prefix = Bool.not(U32.is_zero(scale)) && Views.has_count(Shapes.checked_multiply(guards_height,scale))
  +width_ok = Views.has_count(Shapes.checked_multiply(guards_width,scale))
  +count_ok = Views.has_count(Tensors.count_of(Layout.scaled_extents(shape,scale)))
  scaled(guards_width,scale,Boolean.right(prefix,width_ok,Boolean.left((prefix && width_ok),count_ok,truth)))

def guards_allocation(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,truth: {guard(geometry,shape,scale) == True{} : Bool}) ->
  Count.Count<ShapeLaws.elements(Layout.scaled_extents(shape,scale)),Checked.value(Tensors.count_of(Layout.scaled_extents(shape,scale)))>:
  Bands.Geometry{batch,channels,+guards_height,+guards_width} = geometry
  +prefix = Bool.not(U32.is_zero(scale)) && Views.has_count(Shapes.checked_multiply(guards_height,scale)) && Views.has_count(Shapes.checked_multiply(guards_width,scale))
  +result = Tensors.count_of(Layout.scaled_extents(shape,scale))
  Count.count_of(Layout.scaled_extents(shape,scale),Checked.value(result),found(result,Boolean.right(prefix,Views.has_count(result),truth)))

def copied_band(~Element: Data,+first: U32,+rows: U32,+source_shape: List<&2,U32>,+target_shape: List<&2,U32>,+fill: Element,
  result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) -> Bands.Bands(~Element):
  Tensors.with_dense(~Element,~Bands.Bands(~Element),Layout.result_of(~Element,Layout.copied(~Element,"upsample_nearest",source_shape,target_shape,fill,1,result)),
    output => shape => fill => status => workers => Partition.Leaf{output,first,rows,Unit{}})

def copied(~Element: Data,+first: U32,+rows: U32,+source_shape: List<&2,U32>,+target_shape: List<&2,U32>,+fill: Element,
  result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) ->
  {Model.contents(~Element,copied_band(~Element,first,rows,source_shape,target_shape,fill,result)) ==
    Model.RowBand{CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,result)),first,rows} : Model.Content<Element>}:
  match result:
    case Operations.Accepted{(left,Storage.Buffer{storage,length,certificate})}: {==}
    case Operations.Rejected{(left,Storage.Buffer{storage,length,certificate})}: {==}

def local_shape(geometry: Bands.Geometry,rows: U32) -> List<&2,U32>:
  Bands.Geometry{batch,channels,height,width} = geometry
  [channels,rows,width]

def leaf_copy(~Element: Data,+geometry: Bands.Geometry,+scale: U32,+fill: Element,+first: U32,+rows: U32,input: Storage.Buffer<Element>,
  +source: Views.View,+target: Views.View,+count: U32,
  repeated: {Layout.repeated(Tensors.view_of(local_shape(geometry,rows)),scale) == Some{source} : Maybe<&2,Views.View>},
  target_view: {Tensors.view_of(Layout.extents(Some{source})) == Some{target} : Maybe<&2,Views.View>},
  counted: {Tensors.count_of(Layout.extents(Some{source})) == Some{count} : Maybe<&2,U32>}) ->
  {Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,input) ==
    copied_band(~Element,(first * scale : U32),(rows * scale : U32),local_shape(geometry,rows),Layout.scaled_extents(local_shape(geometry,rows),scale),fill,
      Operations.copy(~Element,input,source,Tensors.buffer(~Element,fill,count),target)) : Bands.Bands(~Element)}:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  %Equal.sym(Maybe<&2,Views.View>,Layout.repeated(Tensors.view_of(local_shape(geometry,rows)),scale),Some{source},repeated) :
    {Tensors.with_dense(~Element,~Bands.Bands(~Element),Layout.result_of(~Element,Layout.upsample_view(~Element,_,Tensors.Usable{},input,local_shape(geometry,rows),scale,fill,1)),
      output => shape => fill => status => workers => Partition.Leaf{output,(first * scale : U32),(rows * scale : U32),Unit{}}) ==
      copied_band(~Element,(first * scale : U32),(rows * scale : U32),local_shape(geometry,rows),Layout.scaled_extents(local_shape(geometry,rows),scale),fill,
        Operations.copy(~Element,input,source,Tensors.buffer(~Element,fill,count),target)) : Bands.Bands(~Element)}
  Equal.cong(Tensors.Tensor<Element> & Tensors.Tensor<Element>,Bands.Bands(~Element),pair =>
    Tensors.with_dense(~Element,~Bands.Bands(~Element),Layout.result_of(~Element,pair),
      output => shape => fill => status => workers => Partition.Leaf{output,(first * scale : U32),(rows * scale : U32),Unit{}}),
    Layout.upsample_view(~Element,Some{source},Tensors.Usable{},input,local_shape(geometry,rows),scale,fill,1),
    Layout.copied(~Element,"upsample_nearest",local_shape(geometry,rows),Layout.scaled_extents(local_shape(geometry,rows),scale),fill,1,
      Operations.copy(~Element,input,source,Tensors.buffer(~Element,fill,count),target)),
    Public.gathered(~Element,"upsample_nearest",source,Layout.extents(Some{source}),Layout.scaled_extents(local_shape(geometry,rows),scale),input,local_shape(geometry,rows),fill,target,count,1,target_view,counted))

def leaf(~Element: Data,+geometry: Bands.Geometry,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +source: Views.View,+target: Views.View,+count: U32,
  +repeated: {Layout.repeated(Tensors.view_of(local_shape(geometry,rows)),scale) == Some{source} : Maybe<&2,Views.View>},
  +target_view: {Tensors.view_of(Layout.extents(Some{source})) == Some{target} : Maybe<&2,Views.View>},
  +counted: {Tensors.count_of(Layout.extents(Some{source})) == Some{count} : Maybe<&2,U32>}) ->
  {Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input))) ==
    Model.RowBand{CopyValues.right_content(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target)),
      (first * scale : U32),(rows * scale : U32)} : Model.Content<Element>}:
  Equal.trans(Model.Content<Element>,
    Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input))),
    Model.contents(~Element,copied_band(~Element,(first * scale : U32),(rows * scale : U32),local_shape(geometry,rows),Layout.scaled_extents(local_shape(geometry,rows),scale),fill,
      Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target))),
    Model.RowBand{CopyValues.right_content(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target)),(first * scale : U32),(rows * scale : U32)},
    Equal.cong(Bands.Bands(~Element),Model.Content<Element>,bands => Model.contents(~Element,bands),
      Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input)),
      copied_band(~Element,(first * scale : U32),(rows * scale : U32),local_shape(geometry,rows),Layout.scaled_extents(local_shape(geometry,rows),scale),fill,
        Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target)),
      leaf_copy(~Element,geometry,scale,fill,first,rows,Buffer.pack(~Element,input),source,target,count,repeated,target_view,counted)),
    Equal.trans(Model.Content<Element>,
      Model.contents(~Element,copied_band(~Element,(first * scale : U32),(rows * scale : U32),local_shape(geometry,rows),Layout.scaled_extents(local_shape(geometry,rows),scale),fill,
        Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target))),
      Model.RowBand{CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target))),
        (first * scale : U32),(rows * scale : U32)},
      Model.RowBand{CopyValues.right_content(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target)),(first * scale : U32),(rows * scale : U32)},
      copied(~Element,(first * scale : U32),(rows * scale : U32),local_shape(geometry,rows),Layout.scaled_extents(local_shape(geometry,rows),scale),fill,
        Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target)),
      Equal.cong(Operations.Outcome<OperationsLaws.PairOwners(Element)>,Model.Content<Element>,result => Model.RowBand{CopyValues.right_content(~Element,result),(first * scale : U32),(rows * scale : U32)},
        OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target)),
        OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target),
        OperationsLaws.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target))))

# Successful local descriptors for the actual nearest-neighbour leaf.
# The producer consumes exact local/global counts, never successful View
# computations supplied by the caller. Zero-sized shapes remain included.
def producer_batch(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{producer_batch,producer_channels,producer_height,producer_width} = geometry
  producer_batch

def producer_channels(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{producer_batch,producer_channels,producer_height,producer_width} = geometry
  producer_channels

def producer_height(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{producer_batch,producer_channels,producer_height,producer_width} = geometry
  producer_height

def producer_width(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{producer_batch,producer_channels,producer_height,producer_width} = geometry
  producer_width

def axis_model(reversed: List<&2,U32>,+stride: U32,axes: List<&2,Views.Axis>) -> List<&2,Views.Axis>:
  match reversed:
    case []: axes
    case +head <> tail: axis_model(tail,U32.mul(stride,head),Views.Axis{head,stride} <> axes)

def zero_product(+right: U32) -> {U32.mul(0,right) == 0 : U32}:
  Multiplication.native_multiply(0,right)

def built_empty(+reversed: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+limit: U32,+maximum: {limit == Machine.word_max() : U32}) ->
  {Views.build_axes(reversed,Some{0},axes,offset) == Some{Views.View{axis_model(reversed,0,axes),offset}} : Maybe<&2,Views.View>}:
  match reversed:
    case []: {==}
    case +head <> +tail:
      %Equal.sym(Maybe<&2,U32>,Shapes.checked_multiply(0,head),Some{0},
        %zero_product(head) : {Shapes.checked_multiply(0,head) == Some{_} : Maybe<&2,U32>}
        CountCompletion.checked_product(0,head,limit,maximum,Order.zero_le(U32.to_nat(limit)))) :
        {Views.build_axes(tail,_,Views.Axis{head,0} <> axes,offset) == Some{Views.View{axis_model(tail,U32.mul(0,head),Views.Axis{head,0} <> axes),offset}} : Maybe<&2,Views.View>}
      %Equal.sym(U32,U32.mul(0,head),0,zero_product(head)) : {Views.build_axes(tail,Some{0},Views.Axis{head,0} <> axes,offset) == Some{Views.View{axis_model(tail,_,Views.Axis{head,0} <> axes),offset}} : Maybe<&2,Views.View>}
      built_empty(tail,Views.Axis{head,0} <> axes,offset,limit,maximum)

def built_nonempty(+reversed: List<&2,U32>,+stride: U32,+axes: List<&2,Views.Axis>,+offset: U32,+limit: U32,+maximum: {limit == Machine.word_max() : U32},
  +nonempty: {Shapes.has_zero(reversed) == False{} : Bool},+room: {Nat.is_le(Nat.mul(U32.to_nat(stride),ShapeLaws.elements(reversed)),U32.to_nat(limit)) == True{} : Bool}) ->
  {Views.build_axes(reversed,Some{stride},axes,offset) == Some{Views.View{axis_model(reversed,stride,axes),offset}} : Maybe<&2,Views.View>}:
  match reversed:
    case []: {==}
    case +head <> +tail:
      +tail_nonempty = Boolean.or_false_right(U32.is_zero(head),Shapes.has_zero(tail),nonempty)
      +prefix_room = Order.transitive(Nat.mul(U32.to_nat(stride),U32.to_nat(head)),Nat.mul(U32.to_nat(stride),ShapeLaws.elements(head <> tail)),U32.to_nat(limit),
        %RowMajor.assoc(U32.to_nat(stride),U32.to_nat(head),ShapeLaws.elements(tail)) : {Nat.is_le(Nat.mul(U32.to_nat(stride),U32.to_nat(head)),_) == True{} : Bool}
        RowMajor.product_covers(Nat.mul(U32.to_nat(stride),U32.to_nat(head)),ShapeLaws.elements(tail),CountCompletion.positive_elements(tail,tail_nonempty)),room)
      +product_exact = FullRange.multiply(stride,head,limit,prefix_room)
      %Equal.sym(Maybe<&2,U32>,Shapes.checked_multiply(stride,head),Some{U32.mul(stride,head)},CountCompletion.checked_product(stride,head,limit,maximum,prefix_room)) :
        {Views.build_axes(tail,_,Views.Axis{head,stride} <> axes,offset) == Some{Views.View{axis_model(tail,U32.mul(stride,head),Views.Axis{head,stride} <> axes),offset}} : Maybe<&2,Views.View>}
      built_nonempty(tail,U32.mul(stride,head),Views.Axis{head,stride} <> axes,offset,limit,maximum,tail_nonempty,
        %Equal.sym(Nat,U32.to_nat(U32.mul(stride,head)),Nat.mul(U32.to_nat(stride),U32.to_nat(head)),product_exact) : {Nat.is_le(Nat.mul(_,ShapeLaws.elements(tail)),U32.to_nat(limit)) == True{} : Bool}
        %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(stride),U32.to_nat(head)),ShapeLaws.elements(tail)),Nat.mul(U32.to_nat(stride),Nat.mul(U32.to_nat(head),ShapeLaws.elements(tail))),RowMajor.assoc(U32.to_nat(stride),U32.to_nat(head),ShapeLaws.elements(tail))) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool} room)

def reversed_elements(+extents: List<&2,U32>,+tail: List<&2,U32>) ->
  {ShapeLaws.elements(List.reverse.go(&2,U32,extents,tail)) == Nat.mul(ShapeLaws.elements(extents),ShapeLaws.elements(tail)) : Nat}:
  match extents:
    case []: Equal.sym(Nat,Nat.add(ShapeLaws.elements(tail),0n),ShapeLaws.elements(tail),Addition.zero_right(ShapeLaws.elements(tail)))
    case +head <> +rest:
      Equal.trans(Nat,ShapeLaws.elements(List.reverse.go(&2,U32,rest,head <> tail)),Nat.mul(ShapeLaws.elements(rest),Nat.mul(U32.to_nat(head),ShapeLaws.elements(tail))),
        Nat.mul(Nat.mul(U32.to_nat(head),ShapeLaws.elements(rest)),ShapeLaws.elements(tail)),reversed_elements(rest,head <> tail),
        Equal.trans(Nat,Nat.mul(ShapeLaws.elements(rest),Nat.mul(U32.to_nat(head),ShapeLaws.elements(tail))),Nat.mul(Nat.mul(ShapeLaws.elements(rest),U32.to_nat(head)),ShapeLaws.elements(tail)),Nat.mul(Nat.mul(U32.to_nat(head),ShapeLaws.elements(rest)),ShapeLaws.elements(tail)),
          Equal.sym(Nat,Nat.mul(Nat.mul(ShapeLaws.elements(rest),U32.to_nat(head)),ShapeLaws.elements(tail)),Nat.mul(ShapeLaws.elements(rest),Nat.mul(U32.to_nat(head),ShapeLaws.elements(tail))),RowMajor.assoc(ShapeLaws.elements(rest),U32.to_nat(head),ShapeLaws.elements(tail))),
          Equal.cong(Nat,Nat,value => Nat.mul(value,ShapeLaws.elements(tail)),Nat.mul(ShapeLaws.elements(rest),U32.to_nat(head)),Nat.mul(U32.to_nat(head),ShapeLaws.elements(rest)),RowMajor.multiply_commutative(ShapeLaws.elements(rest),U32.to_nat(head)))))

def reverse_elements(+extents: List<&2,U32>) -> {ShapeLaws.elements(List.reverse(&2,U32,extents)) == ShapeLaws.elements(extents) : Nat}:
  Equal.trans(Nat,ShapeLaws.elements(List.reverse(&2,U32,extents)),Nat.mul(ShapeLaws.elements(extents),1n),ShapeLaws.elements(extents),reversed_elements(extents,[]),ShapeLaws.times_one(ShapeLaws.elements(extents)))

def canonical_view(+extents: List<&2,U32>,+count: U32) -> Views.View:
  Views.View{axis_model(List.reverse(&2,U32,extents),Bool.pick(U32,U32.is_zero(count),0,1),[]),0}

def view_when_bounded(+extents: List<&2,U32>,+count: U32,+limit: U32,+maximum: {limit == Machine.word_max() : U32},
  empty: Bool,+classified: {U32.is_zero(count) == empty : Bool},+exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat},
  +within: {U32.is_le(count,limit) == True{} : Bool}) ->
  {Views.contiguous_count(Some{count},extents,0) == Some{canonical_view(extents,count)} : Maybe<&2,Views.View>}:
  match empty:
    case True{}:
      %Equal.sym(Bool,U32.is_zero(count),True{},classified) :
        {Views.build_axes(List.reverse(&2,U32,extents),Some{Bool.pick(U32,_,0,1)},[],0) == Some{Views.View{axis_model(List.reverse(&2,U32,extents),Bool.pick(U32,_,0,1),[]),0}} : Maybe<&2,Views.View>}
      built_empty(List.reverse(&2,U32,extents),[],0,limit,maximum)
    case False{}:
      +positive = CountCompletion.positive_word(count,classified)
      %Equal.sym(Bool,U32.is_zero(count),False{},classified) :
        {Views.build_axes(List.reverse(&2,U32,extents),Some{Bool.pick(U32,_,0,1)},[],0) == Some{Views.View{axis_model(List.reverse(&2,U32,extents),Bool.pick(U32,_,0,1),[]),0}} : Maybe<&2,Views.View>}
      built_nonempty(List.reverse(&2,U32,extents),1,[],0,limit,maximum,
        CountCompletion.positive_not_zero(List.reverse(&2,U32,extents),%Equal.sym(Nat,ShapeLaws.elements(List.reverse(&2,U32,extents)),ShapeLaws.elements(extents),reverse_elements(extents)) :
          {Nat.is_lt(0n,_) == True{} : Bool} %Equal.sym(Nat,ShapeLaws.elements(extents),U32.to_nat(count),exact) : {Nat.is_lt(0n,_) == True{} : Bool} positive),
        %Equal.sym(Nat,Nat.add(ShapeLaws.elements(List.reverse(&2,U32,extents)),0n),ShapeLaws.elements(List.reverse(&2,U32,extents)),Addition.zero_right(ShapeLaws.elements(List.reverse(&2,U32,extents)))) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
        %Equal.sym(Nat,ShapeLaws.elements(List.reverse(&2,U32,extents)),ShapeLaws.elements(extents),reverse_elements(extents)) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
        %Equal.sym(Nat,ShapeLaws.elements(extents),U32.to_nat(count),exact) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool} Bounds.as_nat_le(count,limit,within))

def view_when(+extents: List<&2,U32>,+count: U32,empty: Bool,+classified: {U32.is_zero(count) == empty : Bool},
  +exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat}) ->
  {Views.contiguous_count(Some{count},extents,0) == Some{canonical_view(extents,count)} : Maybe<&2,Views.View>}:
  view_when_bounded(extents,count,Machine.word_max(),{==},empty,classified,exact,FullRange.maximum(count))

def view(+extents: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat}) ->
  {Tensors.view_of(extents) == Some{canonical_view(extents,count)} : Maybe<&2,Views.View>}:
  %Equal.sym(Maybe<&2,U32>,Shapes.numel(Shapes.Shape{extents}),Some{count},CountCompletion.numel(extents,count,exact)) :
    {Views.contiguous_count(_,extents,0) == Some{canonical_view(extents,count)} : Maybe<&2,Views.View>}
  view_when(extents,count,U32.is_zero(count),{==},exact)

def repeated_shape(+geometry: Bands.Geometry,+rows: U32,+scale: U32) -> List<&2,U32>:
  Bands.Geometry{producer_batch,producer_channels,producer_height,producer_width} = geometry
  [producer_channels,rows,scale,producer_width,scale]

def repeated_view(+geometry: Bands.Geometry,+rows: U32,+scale: U32,+count: U32) -> Views.View:
  Bands.Geometry{producer_batch,producer_channels,producer_height,+producer_width} = geometry
  +base = Bool.pick(U32,U32.is_zero(count),0,1)
  Views.View{[Views.Axis{producer_channels,U32.mul(U32.mul(base,producer_width),rows)},Views.Axis{rows,U32.mul(base,producer_width)},Views.Axis{scale,0},Views.Axis{producer_width,base},Views.Axis{scale,0}],0}

def repeated(+geometry: Bands.Geometry,+rows: U32,+scale: U32,+count: U32,
  input_exact: {ShapeLaws.elements(local_shape(geometry,rows)) == U32.to_nat(count) : Nat}) ->
  {Layout.repeated(Tensors.view_of(local_shape(geometry,rows)),scale) == Some{repeated_view(geometry,rows,scale,count)} : Maybe<&2,Views.View>}:
  Bands.Geometry{producer_batch,+producer_channels,producer_height,+producer_width} = geometry
  %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(local_shape(geometry,rows)),Some{canonical_view(local_shape(geometry,rows),count)},
    view(local_shape(geometry,rows),count,input_exact)) :
    {Layout.repeated(_,scale) == Some{repeated_view(geometry,rows,scale,count)} : Maybe<&2,Views.View>}
  {==}

type Descriptor<-geometry: Bands.Geometry,-rows: U32,-scale: U32,-input_count: U32,-producer_output_count: U32> is Data:
  Descriptor{repeated: {Layout.repeated(Tensors.view_of(local_shape(geometry,rows)),scale) == Some{repeated_view(geometry,rows,scale,input_count)} : Maybe<&2,Views.View>},
    target: {Tensors.view_of(Layout.extents(Some{repeated_view(geometry,rows,scale,input_count)})) == Some{canonical_view(repeated_shape(geometry,rows,scale),producer_output_count)} : Maybe<&2,Views.View>},
    count: {Tensors.count_of(Layout.extents(Some{repeated_view(geometry,rows,scale,input_count)})) == Some{producer_output_count} : Maybe<&2,U32>}}

def descriptor(+geometry: Bands.Geometry,+rows: U32,+scale: U32,+input_count: U32,+producer_output_count: U32,
  input_exact: {ShapeLaws.elements(local_shape(geometry,rows)) == U32.to_nat(input_count) : Nat},
  +output_exact: {ShapeLaws.elements(repeated_shape(geometry,rows,scale)) == U32.to_nat(producer_output_count) : Nat},
  output_within: {U32.is_le(producer_output_count,Machine.capacity(Default.limits())) == True{} : Bool}) -> Descriptor<geometry,rows,scale,input_count,producer_output_count>:
  Bands.Geometry{producer_batch,+producer_channels,producer_height,+producer_width} = geometry
  Descriptor{
    repeated(geometry,rows,scale,input_count,input_exact),
    view(repeated_shape(geometry,rows,scale),producer_output_count,output_exact),
    %Equal.sym(Maybe<&2,U32>,Shapes.numel(Shapes.Shape{repeated_shape(geometry,rows,scale)}),Some{producer_output_count},CountCompletion.numel(repeated_shape(geometry,rows,scale),producer_output_count,output_exact)) :
      {Tensors.allocatable(_) == Some{producer_output_count} : Maybe<&2,U32>}
    %Equal.sym(Bool,U32.is_le(producer_output_count,Machine.capacity(Default.limits())),True{},output_within) : {Tensors.within(_,producer_output_count) == Some{producer_output_count} : Maybe<&2,U32>} {==}}

def repeated_elements(+geometry: Bands.Geometry,+rows: U32,+scale: U32) ->
  {ShapeLaws.elements(repeated_shape(geometry,rows,scale)) ==
    Nat.mul(U32.to_nat(producer_channels(geometry)),Nat.mul(Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),Nat.mul(U32.to_nat(producer_width(geometry)),U32.to_nat(scale)))) : Nat}:
  Bands.Geometry{producer_batch,+producer_channels,producer_height,+producer_width} = geometry
  %Equal.sym(Nat,Nat.mul(U32.to_nat(scale),1n),U32.to_nat(scale),ShapeLaws.times_one(U32.to_nat(scale))) :
    {Nat.mul(U32.to_nat(producer_channels),Nat.mul(U32.to_nat(rows),Nat.mul(U32.to_nat(scale),Nat.mul(U32.to_nat(producer_width),_)))) ==
      Nat.mul(U32.to_nat(producer_channels),Nat.mul(Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),Nat.mul(U32.to_nat(producer_width),U32.to_nat(scale)))) : Nat}
  Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(producer_channels),value),
    Nat.mul(U32.to_nat(rows),Nat.mul(U32.to_nat(scale),Nat.mul(U32.to_nat(producer_width),U32.to_nat(scale)))),
    Nat.mul(Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),Nat.mul(U32.to_nat(producer_width),U32.to_nat(scale))),
    Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),Nat.mul(U32.to_nat(producer_width),U32.to_nat(scale))),
      Nat.mul(U32.to_nat(rows),Nat.mul(U32.to_nat(scale),Nat.mul(U32.to_nat(producer_width),U32.to_nat(scale)))),RowMajor.assoc(U32.to_nat(rows),U32.to_nat(scale),Nat.mul(U32.to_nat(producer_width),U32.to_nat(scale)))))

def producer_output_count(+geometry: Bands.Geometry,+rows: U32,+scale: U32) -> U32:
  U32.from_nat(ShapeLaws.elements(repeated_shape(geometry,rows,scale)))

def local_bound(+geometry: Bands.Geometry,+first: U32,+rows: U32,+scale: U32,+global_count: U32,
  rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(producer_batch(geometry)),U32.to_nat(producer_height(geometry)))) == True{} : Bool},
  global_exact: {Nat.mul(U32.to_nat(producer_batch(geometry)),Nat.mul(U32.to_nat(producer_channels(geometry)),
    Nat.mul(Nat.mul(U32.to_nat(producer_height(geometry)),U32.to_nat(scale)),Nat.mul(U32.to_nat(producer_width(geometry)),U32.to_nat(scale))))) == U32.to_nat(global_count) : Nat}) ->
  {Nat.is_le(ShapeLaws.elements(repeated_shape(geometry,rows,scale)),U32.to_nat(global_count)) == True{} : Bool}:
  Bands.Geometry{+producer_batch,+producer_channels,+producer_height,+producer_width} = geometry
  %Equal.sym(Nat,ShapeLaws.elements(repeated_shape(geometry,rows,scale)),Nat.mul(U32.to_nat(producer_channels),Nat.mul(Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),Nat.mul(U32.to_nat(producer_width),U32.to_nat(scale)))),repeated_elements(geometry,rows,scale)) :
    {Nat.is_le(_,U32.to_nat(global_count)) == True{} : Bool}
  %global_exact : {Nat.is_le(Nat.mul(U32.to_nat(producer_channels),Nat.mul(Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),Nat.mul(U32.to_nat(producer_width),U32.to_nat(scale)))),_) == True{} : Bool}
  local_elements(U32.to_nat(producer_batch),U32.to_nat(producer_channels),U32.to_nat(producer_height),U32.to_nat(producer_width),U32.to_nat(rows),U32.to_nat(scale),
    bounds_rows_within(U32.to_nat(first),U32.to_nat(rows),U32.to_nat(producer_batch),U32.to_nat(producer_height),rows_fit))

def from_global(+geometry: Bands.Geometry,+rows: U32,+scale: U32,+input_count: U32,+global_count: U32,
  input_exact: {ShapeLaws.elements(local_shape(geometry,rows)) == U32.to_nat(input_count) : Nat},
  +local_room: {Nat.is_le(ShapeLaws.elements(repeated_shape(geometry,rows,scale)),U32.to_nat(global_count)) == True{} : Bool},
  global_within: {U32.is_le(global_count,Machine.capacity(Default.limits())) == True{} : Bool}) ->
  Descriptor<geometry,rows,scale,input_count,producer_output_count(geometry,rows,scale)>:
  +exact = FullRange.roundtrip(ShapeLaws.elements(repeated_shape(geometry,rows,scale)),global_count,local_room)
  descriptor(geometry,rows,scale,input_count,producer_output_count(geometry,rows,scale),input_exact,
    Equal.sym(Nat,U32.to_nat(producer_output_count(geometry,rows,scale)),ShapeLaws.elements(repeated_shape(geometry,rows,scale)),exact),
    Arithmetic.count_within(producer_output_count(geometry,rows,scale),ShapeLaws.elements(repeated_shape(geometry,rows,scale)),global_count,Machine.capacity(Default.limits()),exact,local_room,global_within))

def canonical_shape(+extents: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat}) ->
  {Views.axis_extents_of(canonical_view(extents,count)) == extents : List<&2,U32>}:
  %view(extents,count,exact) : {Metadata.observed_extents(_,extents) == extents : List<&2,U32>}
  Metadata.contiguous_shape(extents,0)

def counted(+geometry: Bands.Geometry,+rows: U32,+scale: U32,+input_count: U32,+producer_output_count: U32,
  evidence: Descriptor<geometry,rows,scale,input_count,producer_output_count>) ->
  Count.Count<ShapeLaws.elements(repeated_shape(geometry,rows,scale)),producer_output_count>:
  Bands.Geometry{producer_batch,producer_channels,producer_height,producer_width} = geometry
  Descriptor{repeated,target,count} = evidence
  Count.count_of(repeated_shape(geometry,rows,scale),producer_output_count,count)

def target_counted_from_count(+extents: List<&2,U32>,+count: U32,evidence: Count.Count<ShapeLaws.elements(extents),count>) ->
  {Views.numel(canonical_view(extents,count)) == Some{count} : Maybe<&2,U32>}:
  Count.Count{+exact,bounded} = evidence
  %Equal.sym(List<&2,U32>,Views.axis_extents_of(canonical_view(extents,count)),extents,
    canonical_shape(extents,count,exact)) : {Shapes.numel(Shapes.Shape{_}) == Some{count} : Maybe<&2,U32>}
  CountCompletion.numel(extents,count,exact)

def target_counted(+geometry: Bands.Geometry,+rows: U32,+scale: U32,+input_count: U32,+producer_output_count: U32,
  evidence: Descriptor<geometry,rows,scale,input_count,producer_output_count>) ->
  {Views.numel(canonical_view(repeated_shape(geometry,rows,scale),producer_output_count)) == Some{producer_output_count} : Maybe<&2,U32>}:
  target_counted_from_count(repeated_shape(geometry,rows,scale),producer_output_count,counted(geometry,rows,scale,input_count,producer_output_count,evidence))

def allocated_from_count(+elements: Nat,+count: U32,evidence: Count.Count<elements,count>) -> Depth.Sized<count>:
  Count.Count{exact,bounded} = evidence
  Depth.bounded(count,bounded)

def allocated(+geometry: Bands.Geometry,+rows: U32,+scale: U32,+input_count: U32,+producer_output_count: U32,
  evidence: Descriptor<geometry,rows,scale,input_count,producer_output_count>) -> Depth.Sized<producer_output_count>:
  allocated_from_count(ShapeLaws.elements(repeated_shape(geometry,rows,scale)),producer_output_count,counted(geometry,rows,scale,input_count,producer_output_count,evidence))

def source_exact(+geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,
  metadata: BandShape.Leaf(geometry,first,rows,length),count: BandShape.Count<geometry,rows,length>) ->
  {ShapeLaws.elements(local_shape(geometry,rows)) == U32.to_nat(Bands.local_count(geometry,rows)) : Nat}:
  Bands.Geometry{producer_batch,+producer_channels,producer_height,+producer_width} = geometry
  BandShape.Fields{local,rows_fit,+length_exact} = metadata
  BandShape.Count{+exact,allocatable} = count
  %Equal.sym(Nat,Nat.mul(U32.to_nat(producer_width),1n),U32.to_nat(producer_width),ShapeLaws.times_one(U32.to_nat(producer_width))) :
    {Nat.mul(U32.to_nat(producer_channels),Nat.mul(U32.to_nat(rows),_)) == U32.to_nat(Bands.local_count(geometry,rows)) : Nat}
  Equal.trans(Nat,Nat.mul(U32.to_nat(producer_channels),Nat.mul(U32.to_nat(rows),U32.to_nat(producer_width))),length,U32.to_nat(Bands.local_count(geometry,rows)),
    Equal.sym(Nat,length,Nat.mul(U32.to_nat(producer_channels),Nat.mul(U32.to_nat(rows),U32.to_nat(producer_width))),length_exact),
    Equal.sym(Nat,U32.to_nat(Bands.local_count(geometry,rows)),length,exact))

def producer_owner_descriptor(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+scale: U32,+input: Buffer.Model<Element>,+global_count: U32,
  width_positive: {BandShape.width_positive(geometry) == True{} : Bool},+metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),ready: Buffer.Sized<Element,input>,
  global_exact: {Nat.mul(U32.to_nat(producer_batch(geometry)),Nat.mul(U32.to_nat(producer_channels(geometry)),
    Nat.mul(Nat.mul(U32.to_nat(producer_height(geometry)),U32.to_nat(scale)),Nat.mul(U32.to_nat(producer_width(geometry)),U32.to_nat(scale))))) == U32.to_nat(global_count) : Nat},
  global_within: {U32.is_le(global_count,Machine.capacity(Default.limits())) == True{} : Bool}) ->
  Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),producer_output_count(geometry,rows,scale)>:
  Bands.Geometry{producer_batch,producer_channels,producer_height,producer_width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  from_global(geometry,rows,scale,Bands.local_count(geometry,rows),global_count,
    source_exact(geometry,first,rows,Buffer.length(~Element,input),metadata,BandShape.count(~Element,geometry,first,rows,input,width_positive,metadata,ready)),
    local_bound(geometry,first,rows,scale,global_count,rows_fit,global_exact),global_within)

# Source metadata and Ready discharge the local input count. The public
# shape/geometry relation must still establish scaled_elements. This helper
# uses the existing producer_width-positive metadata theorem; descriptor/from_global
# above also cover zero producer_width when an exact input count is supplied.
def guarded_owner_from_count(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+first: U32,+rows: U32,+scale: U32,+input: Buffer.Model<Element>,
  width_positive: {BandShape.width_positive(geometry) == True{} : Bool},metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),ready: Buffer.Sized<Element,input>,
  scaled_elements: {ShapeLaws.elements(Layout.scaled_extents(shape,scale)) == Nat.mul(U32.to_nat(producer_batch(geometry)),Nat.mul(U32.to_nat(producer_channels(geometry)),
    Nat.mul(Nat.mul(U32.to_nat(producer_height(geometry)),U32.to_nat(scale)),Nat.mul(U32.to_nat(producer_width(geometry)),U32.to_nat(scale))))) : Nat},
  allocation: Count.Count<ShapeLaws.elements(Layout.scaled_extents(shape,scale)),Checked.value(Tensors.count_of(Layout.scaled_extents(shape,scale)))>) ->
  Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),producer_output_count(geometry,rows,scale)>:
  Count.Count{exact,bounded} = allocation
  producer_owner_descriptor(~Element,geometry,first,rows,scale,input,Checked.value(Tensors.count_of(Layout.scaled_extents(shape,scale))),width_positive,metadata,ready,
    Equal.trans(Nat,Nat.mul(U32.to_nat(producer_batch(geometry)),Nat.mul(U32.to_nat(producer_channels(geometry)),
      Nat.mul(Nat.mul(U32.to_nat(producer_height(geometry)),U32.to_nat(scale)),Nat.mul(U32.to_nat(producer_width(geometry)),U32.to_nat(scale))))),
      ShapeLaws.elements(Layout.scaled_extents(shape,scale)),U32.to_nat(Checked.value(Tensors.count_of(Layout.scaled_extents(shape,scale)))),
      Equal.sym(Nat,ShapeLaws.elements(Layout.scaled_extents(shape,scale)),Nat.mul(U32.to_nat(producer_batch(geometry)),Nat.mul(U32.to_nat(producer_channels(geometry)),
        Nat.mul(Nat.mul(U32.to_nat(producer_height(geometry)),U32.to_nat(scale)),Nat.mul(U32.to_nat(producer_width(geometry)),U32.to_nat(scale))))),scaled_elements),exact),bounded)

def guarded_owner(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+first: U32,+rows: U32,+scale: U32,+input: Buffer.Model<Element>,
  width_positive: {BandShape.width_positive(geometry) == True{} : Bool},metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),ready: Buffer.Sized<Element,input>,
  truth: {guard(geometry,shape,scale) == True{} : Bool},
  scaled_elements: {ShapeLaws.elements(Layout.scaled_extents(shape,scale)) == Nat.mul(U32.to_nat(producer_batch(geometry)),Nat.mul(U32.to_nat(producer_channels(geometry)),
    Nat.mul(Nat.mul(U32.to_nat(producer_height(geometry)),U32.to_nat(scale)),Nat.mul(U32.to_nat(producer_width(geometry)),U32.to_nat(scale))))) : Nat}) ->
  Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),producer_output_count(geometry,rows,scale)>:
  guarded_owner_from_count(~Element,geometry,shape,first,rows,scale,input,width_positive,metadata,ready,scaled_elements,guards_allocation(geometry,shape,scale,truth))

def leaf_equation(~Element: Data,+geometry: Bands.Geometry,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+input_count: U32,+output_count: U32,
  descriptor: Descriptor<geometry,rows,scale,input_count,output_count>) ->
  {Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input))) ==
    Model.RowBand{CopyValues.right_content(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),repeated_view(geometry,rows,scale,input_count),
      Tensors.buffer(~Element,fill,output_count),canonical_view(repeated_shape(geometry,rows,scale),output_count))),
      (first * scale : U32),(rows * scale : U32)} : Model.Content<Element>}:
  Descriptor{repeated,target,count} = descriptor
  leaf(~Element,geometry,scale,fill,first,rows,input,repeated_view(geometry,rows,scale,input_count),
    canonical_view(repeated_shape(geometry,rows,scale),output_count),output_count,repeated,target,count)

def leaf_ready(~Element: Data,+geometry: Bands.Geometry,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+input_count: U32,+output_count: U32,
  +descriptor: Descriptor<geometry,rows,scale,input_count,output_count>) ->
  Model.ContentReady(~Element,Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input)))):
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input))),
    Model.RowBand{CopyValues.right_content(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),repeated_view(geometry,rows,scale,input_count),
      Tensors.buffer(~Element,fill,output_count),canonical_view(repeated_shape(geometry,rows,scale),output_count))),(first * scale : U32),(rows * scale : U32)},
    leaf_equation(~Element,geometry,scale,fill,first,rows,input,input_count,output_count,descriptor)) : Model.ContentReady(~Element,_)
  CopyValues.copy_observed_ready(~Element,Buffer.pack(~Element,input),repeated_view(geometry,rows,scale,input_count),Tensors.buffer(~Element,fill,output_count),
    canonical_view(repeated_shape(geometry,rows,scale),output_count),
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Tensors.buffer(~Element,fill,output_count)),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(output_count),fill),U32.to_nat(output_count)},
      AllocationLaws.contents(~Element,Tensors.depth_for(output_count),fill,U32.to_nat(output_count))) : Buffer.ContentReady<Element,_>
    Buffer.content_ready(~Element,BandCount.model(~Element,output_count,fill),BandCount.model_ready(~Element,output_count,fill,allocated(geometry,rows,scale,input_count,output_count,descriptor))))

def leaf_program(~Element: Data,+geometry: Bands.Geometry,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+input_count: U32,+output_count: U32,
  +descriptor: Descriptor<geometry,rows,scale,input_count,output_count>,
  guarded: {CopyValues.copy_guard(~Element,Buffer.pack(~Element,input),repeated_view(geometry,rows,scale,input_count),Tensors.buffer(~Element,fill,output_count),
    canonical_view(repeated_shape(geometry,rows,scale),output_count)) == True{} : Bool}) ->
  {Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input))) ==
    Model.RowBand{CopyValues.program_content(~Element,output_count,repeated_view(geometry,rows,scale,input_count),canonical_view(repeated_shape(geometry,rows,scale),output_count),
      Buffer.content(~Element,input),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(output_count),fill),U32.to_nat(output_count)}),(first * scale : U32),(rows * scale : U32)} : Model.Content<Element>}:
  +source = repeated_view(geometry,rows,scale,input_count)
  +target = canonical_view(repeated_shape(geometry,rows,scale),output_count)
  %Buffer.packed_contents(~Element,input) :
    {Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input))) ==
      Model.RowBand{CopyValues.program_content(~Element,output_count,source,target,_,Buffer.Content{Tree.fresh(Element,Tensors.depth_for(output_count),fill),U32.to_nat(output_count)}),(first * scale : U32),(rows * scale : U32)} : Model.Content<Element>}
  %AllocationLaws.contents(~Element,Tensors.depth_for(output_count),fill,U32.to_nat(output_count)) :
    {Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input))) ==
      Model.RowBand{CopyValues.program_content(~Element,output_count,source,target,Buffer.contents(~Element,Buffer.pack(~Element,input)),_),(first * scale : U32),(rows * scale : U32)} : Model.Content<Element>}
  Equal.trans(Model.Content<Element>,Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input))),
    Model.RowBand{CopyValues.right_content(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,output_count),target)),(first * scale : U32),(rows * scale : U32)},
    Model.RowBand{CopyValues.program_content(~Element,output_count,source,target,Buffer.contents(~Element,Buffer.pack(~Element,input)),Buffer.contents(~Element,Tensors.buffer(~Element,fill,output_count))),(first * scale : U32),(rows * scale : U32)},
    leaf_equation(~Element,geometry,scale,fill,first,rows,input,input_count,output_count,descriptor),
    Equal.cong(Buffer.Content<Element>,Model.Content<Element>,content => Model.RowBand{content,(first * scale : U32),(rows * scale : U32)},
      CopyValues.right_content(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,output_count),target)),
      CopyValues.program_content(~Element,output_count,source,target,Buffer.contents(~Element,Buffer.pack(~Element,input)),Buffer.contents(~Element,Tensors.buffer(~Element,fill,output_count))),
      CopyValues.copy_observed_content(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,output_count),target,output_count,guarded,target_counted(geometry,rows,scale,input_count,output_count,descriptor))))

def scaled_geometry(geometry: Bands.Geometry,+scale: U32) -> Bands.Geometry:
  Bands.Geometry{batch,channels,height,width} = geometry
  Bands.Geometry{batch,channels,(height * scale : U32),(width * scale : U32)}

type Context<-Logical: Data> is Data:
  Context{logical: Logical,geometry: Bands.Geometry,scale: U32}

def nearest_index(geometry: Bands.Geometry,+scale: U32,+index: U32) -> U32:
  Bands.Geometry{batch,channels,+height,+width} = geometry
  +output_height = (height * scale : U32)
  +output_width = (width * scale : U32)
  +plane = (output_height * output_width : U32)
  +row = (index % plane / output_width / scale : U32)
  +column = (index % output_width / scale : U32)
  (index / plane * height * width + row * width + column : U32)

def value(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,context: Context<Logical>,index: U32) -> Element:
  Context{logical,geometry,scale} = context
  original(logical,nearest_index(geometry,scale,index))

def leaf_writes(~Element: Data,+geometry: Bands.Geometry,+scale: U32,+rows: U32,+input_count: U32,+output_count: U32,input: Buffer.Model<Element>) -> List<&2,Writes.Assignment<Element>>:
  +source = repeated_view(geometry,rows,scale,input_count)
  CopyValues.copy_writes(~Element,Views.is_contiguous(source),U32.to_nat(output_count),source,0,Buffer.tree(~Element,input))

def fresh_tree(~Element: Data,+fill: Element,count: U32) -> Tree.Tree<Element>:
  Tree.fresh(Element,Tensors.depth_for(count),fill)

# The write program stores the wanted value of a cell at its position.
def program_cell(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+output: Tree.Tree<Element>,+length: Nat,+writes: List<&2,Writes.Assignment<Element>>,
  +channel: Nat,+row: Nat,+column: Nat,ready: Buffer.ContentReady<Element,Buffer.Content{output,length}>,stores_inside: Readback.Inside(~Element,Tree.capacity(Element,output),writes),
  inside: {Nat.is_lt(BandValues.position(geometry,rows,channel,row,column),length) == True{} : Bool},
  point: Shape.Both<{U32.is_lt(U32.from_nat(BandValues.position(geometry,rows,channel,row,column)),Tree.capacity(Element,output)) == True{} : Bool},
    CopyValues.Point<Element,value(context,U32.from_nat(BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column))),U32.from_nat(BandValues.position(geometry,rows,channel,row,column)),writes>>) ->
  BandValues.At(~Element,~Context,~value,context,geometry,first,rows,Writes.pure(Element,writes,output),length,channel,row,column):
  Shape.Both{query_inside,evidence} = point
  Shape.Both{inside,
    CopyValues.program_point(~Element,writes,output,length,U32.from_nat(BandValues.position(geometry,rows,channel,row,column)),value(context,U32.from_nat(BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column))),ready,query_inside,stores_inside,evidence)}

def leaf_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+input_count: U32,+output_count: U32,
  +descriptor: Descriptor<geometry,rows,scale,input_count,output_count>,
  guarded: {CopyValues.copy_guard(~Element,Buffer.pack(~Element,input),repeated_view(geometry,rows,scale,input_count),Tensors.buffer(~Element,fill,output_count),
    canonical_view(repeated_shape(geometry,rows,scale),output_count)) == True{} : Bool},
  values: BandValues.Leaf(~Element,~Context<Logical>,~value(~Element,~Logical,~original),Context{logical,geometry,scale},scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),
    Writes.pure(Element,leaf_writes(~Element,geometry,scale,rows,input_count,output_count,input),fresh_tree(~Element,fill,output_count)),U32.to_nat(output_count))) ->
  BandValues.Values(~Element,~Context<Logical>,~value(~Element,~Logical,~original),Context{logical,geometry,scale},scaled_geometry(geometry,scale),
    Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input)))):
  match input:
    case Buffer.Model{+tree,+length,+certificate}:
      +input = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
      %Equal.sym(Model.Content<Element>,Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input))),
        Model.RowBand{CopyValues.program_content(~Element,output_count,repeated_view(geometry,rows,scale,input_count),canonical_view(repeated_shape(geometry,rows,scale),output_count),
          Buffer.content(~Element,input),Buffer.Content{fresh_tree(~Element,fill,output_count),U32.to_nat(output_count)}),(first * scale : U32),(rows * scale : U32)},
        leaf_program(~Element,geometry,scale,fill,first,rows,input,input_count,output_count,descriptor,guarded)) :
        BandValues.Values(~Element,~Context<Logical>,~value(~Element,~Logical,~original),Context{logical,geometry,scale},scaled_geometry(geometry,scale),_)
      values

def values_leaf_metadata(~Element: Data,+geometry: Bands.Geometry,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+input_count: U32,+output_count: U32,
  descriptor: Descriptor<geometry,rows,scale,input_count,output_count>,
  guarded: {CopyValues.copy_guard(~Element,Buffer.pack(~Element,input),repeated_view(geometry,rows,scale,input_count),Tensors.buffer(~Element,fill,output_count),
    canonical_view(repeated_shape(geometry,rows,scale),output_count)) == True{} : Bool},
  placement: BandShape.Leaf(scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),U32.to_nat(output_count))) ->
  BandShape.Contents(~Element,scaled_geometry(geometry,scale),Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input)))):
  match input:
    case Buffer.Model{+tree,+length,+certificate}:
      +input = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
      %Equal.sym(Model.Content<Element>,Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input))),
        Model.RowBand{CopyValues.program_content(~Element,output_count,repeated_view(geometry,rows,scale,input_count),canonical_view(repeated_shape(geometry,rows,scale),output_count),
          Buffer.content(~Element,input),Buffer.Content{fresh_tree(~Element,fill,output_count),U32.to_nat(output_count)}),(first * scale : U32),(rows * scale : U32)},
        leaf_program(~Element,geometry,scale,fill,first,rows,input,input_count,output_count,descriptor,guarded)) : BandShape.Contents(~Element,scaled_geometry(geometry,scale),_)
      placement

def leaf_layout(~Element: Data,+geometry: Bands.Geometry,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+input_count: U32,+output_count: U32,
  descriptor: Descriptor<geometry,rows,scale,input_count,output_count>) ->
  {Model.content_layout(~Element,Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input)))) ==
    Bands.Rows{(first * scale : U32),(rows * scale : U32)} : Bands.Plan}:
  Equal.cong(Model.Content<Element>,Bands.Plan,content => Model.content_layout(~Element,content),
    Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input))),
    Model.RowBand{CopyValues.right_content(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),repeated_view(geometry,rows,scale,input_count),
      Tensors.buffer(~Element,fill,output_count),canonical_view(repeated_shape(geometry,rows,scale),output_count))),(first * scale : U32),(rows * scale : U32)},
    leaf_equation(~Element,geometry,scale,fill,first,rows,input,input_count,output_count,descriptor))

def leaf_well_formed(~Element: Data,+geometry: Bands.Geometry,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+input_count: U32,+output_count: U32,
  +descriptor: Descriptor<geometry,rows,scale,input_count,output_count>,
  guarded: {CopyValues.copy_guard(~Element,Buffer.pack(~Element,input),repeated_view(geometry,rows,scale,input_count),Tensors.buffer(~Element,fill,output_count),
    canonical_view(repeated_shape(geometry,rows,scale),output_count)) == True{} : Bool},
  placement: BandShape.Leaf(scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),U32.to_nat(output_count)),
  coverage: BandPlan.Coverage(Bands.Rows{(first * scale : U32),(rows * scale : U32)},0n,Inspection.total_rows(scaled_geometry(geometry,scale)))) ->
  BandShape.WellFormed<Element,scaled_geometry(geometry,scale),Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input)))>:
  BandShape.WellFormed{
    values_leaf_metadata(~Element,geometry,scale,fill,first,rows,input,input_count,output_count,descriptor,guarded,placement),
    leaf_ready(~Element,geometry,scale,fill,first,rows,input,input_count,output_count,descriptor),
    %Equal.sym(Bands.Plan,Model.content_layout(~Element,Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input)))),
      Bands.Rows{(first * scale : U32),(rows * scale : U32)},leaf_layout(~Element,geometry,scale,fill,first,rows,input,input_count,output_count,descriptor)) :
      BandPlan.Coverage(_,0n,Inspection.total_rows(scaled_geometry(geometry,scale))) coverage}

# Scaled row-word arithmetic uses a formal native capacity throughout.
# Callers supply the actual metadata, scaled-height equation and count room;
# this module introduces no Default machine numeral or acceptance predicate.
type Words<-scale: U32,-first: U32,-rows: U32> is Data:
  Words{first_exact: {U32.to_nat((first * scale : U32)) == Nat.mul(U32.to_nat(first),U32.to_nat(scale)) : Nat},
    rows_exact: {U32.to_nat((rows * scale : U32)) == Nat.mul(U32.to_nat(rows),U32.to_nat(scale)) : Nat}}

def words_first_exact(+scale: U32,+first: U32,+rows: U32,words: Words<scale,first,rows>) ->
  {U32.to_nat((first * scale : U32)) == Nat.mul(U32.to_nat(first),U32.to_nat(scale)) : Nat}:
  Words{first_exact,rows_exact} = words
  first_exact

def words_rows_exact(+scale: U32,+first: U32,+rows: U32,words: Words<scale,first,rows>) ->
  {U32.to_nat((rows * scale : U32)) == Nat.mul(U32.to_nat(rows),U32.to_nat(scale)) : Nat}:
  Words{first_exact,rows_exact} = words
  rows_exact

def extent_scaled_room(+geometry: Bands.Geometry,+scale: U32,+extent: U32,+capacity: U32,
  +within: {Nat.is_le(U32.to_nat(extent),Inspection.total_rows(geometry)) == True{} : Bool},
  +total_room: {Nat.is_le(Inspection.total_rows(scaled_geometry(geometry,scale)),U32.to_nat(capacity)) == True{} : Bool},
  height_scaled: {Nat.mul(U32.to_nat(producer_height(geometry)),U32.to_nat(scale)) == U32.to_nat((producer_height(geometry) * scale : U32)) : Nat}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(extent),U32.to_nat(scale)),U32.to_nat(capacity)) == True{} : Bool}:
  Bands.Geometry{+batch,channels,+height,width} = geometry
  Order.transitive(Nat.mul(U32.to_nat(extent),U32.to_nat(scale)),Nat.mul(Inspection.total_rows(geometry),U32.to_nat(scale)),U32.to_nat(capacity),
    RowMajor.mul_le(U32.to_nat(extent),Inspection.total_rows(geometry),U32.to_nat(scale),within),
    %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(batch),U32.to_nat(height)),U32.to_nat(scale)),Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(height),U32.to_nat(scale))),RowMajor.assoc(U32.to_nat(batch),U32.to_nat(height),U32.to_nat(scale))) :
      {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(U32.to_nat(height),U32.to_nat(scale)),U32.to_nat((height * scale : U32)),height_scaled) :
      {Nat.is_le(Nat.mul(U32.to_nat(batch),_),U32.to_nat(capacity)) == True{} : Bool} total_room)

def scaled_words(+geometry: Bands.Geometry,+scale: U32,+first: U32,+rows: U32,+length: Nat,+capacity: U32,
  metadata: BandShape.Leaf(geometry,first,rows,length),
  +height_scaled: {Nat.mul(U32.to_nat(producer_height(geometry)),U32.to_nat(scale)) == U32.to_nat((producer_height(geometry) * scale : U32)) : Nat},
  +total_room: {Nat.is_le(Inspection.total_rows(scaled_geometry(geometry,scale)),U32.to_nat(capacity)) == True{} : Bool}) -> Words<scale,first,rows>:
  Bands.Geometry{+batch,channels,+height,width} = geometry
  BandShape.Fields{local,+rows_fit,length_exact} = metadata
  +first_within = Order.transitive(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(geometry),Order.le_add(U32.to_nat(first),U32.to_nat(rows)),rows_fit)
  +rows_within = bounds_rows_within(U32.to_nat(first),U32.to_nat(rows),U32.to_nat(batch),U32.to_nat(height),rows_fit)
  Words{FullRange.multiply(first,scale,capacity,extent_scaled_room(geometry,scale,first,capacity,first_within,total_room,height_scaled)),
    FullRange.multiply(rows,scale,capacity,extent_scaled_room(geometry,scale,rows,capacity,rows_within,total_room,height_scaled))}

def scaled_rows_fit(+geometry: Bands.Geometry,+scale: U32,+first: U32,+rows: U32,+length: Nat,
  metadata: BandShape.Leaf(geometry,first,rows,length),words: Words<scale,first,rows>,
  height_scaled: {Nat.mul(U32.to_nat(producer_height(geometry)),U32.to_nat(scale)) == U32.to_nat((producer_height(geometry) * scale : U32)) : Nat}) ->
  {Nat.is_le(Nat.add(U32.to_nat((first * scale : U32)),U32.to_nat((rows * scale : U32))),Inspection.total_rows(scaled_geometry(geometry,scale))) == True{} : Bool}:
  Bands.Geometry{+batch,channels,+height,width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  Words{first_exact,rows_exact} = words
  %Equal.sym(Nat,U32.to_nat((first * scale : U32)),Nat.mul(U32.to_nat(first),U32.to_nat(scale)),first_exact) :
    {Nat.is_le(Nat.add(_,U32.to_nat((rows * scale : U32))),Inspection.total_rows(scaled_geometry(geometry,scale))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((rows * scale : U32)),Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),rows_exact) :
    {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(scale)),_),Inspection.total_rows(scaled_geometry(geometry,scale))) == True{} : Bool}
  %height_scaled :
    {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(scale)),Nat.mul(U32.to_nat(rows),U32.to_nat(scale))),Nat.mul(U32.to_nat(batch),_)) == True{} : Bool}
  %RowMajor.assoc(U32.to_nat(batch),U32.to_nat(height),U32.to_nat(scale)) :
    {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(scale)),Nat.mul(U32.to_nat(rows),U32.to_nat(scale))),_) == True{} : Bool}
  %RowMajor.distrib(U32.to_nat(first),U32.to_nat(rows),U32.to_nat(scale)) :
    {Nat.is_le(_,Nat.mul(Inspection.total_rows(geometry),U32.to_nat(scale))) == True{} : Bool}
  RowMajor.mul_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(geometry),U32.to_nat(scale),rows_fit)

# Actual upsample guards give scaled-shape counts and the row envelope.
# Natural comparisons use a formal capacity tied to the actual native limit.
def not_true(+condition: Bool,truth: {Bool.not(condition) == True{} : Bool}) -> {condition == False{} : Bool}:
  match condition:
    case False{}: {==}
    case True{}: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(truth))

def scale_positive(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,truth: {guard(geometry,shape,scale) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(scale)) == True{} : Bool}:
  Bands.Geometry{batch,channels,+height,+width} = geometry
  +first = Bool.not(U32.is_zero(scale))
  +height_ok = Views.has_count(Shapes.checked_multiply(height,scale))
  +width_ok = Views.has_count(Shapes.checked_multiply(width,scale))
  +count_ok = Views.has_count(Tensors.count_of(Layout.scaled_extents(shape,scale)))
  CountCompletion.positive_word(scale,not_true(U32.is_zero(scale),Boolean.left(first,height_ok,
    Boolean.left((first && height_ok),width_ok,Boolean.left(((first && height_ok) && width_ok),count_ok,truth)))))

def scaled_shape(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,
  original: {shape == BandCount.extents(geometry) : List<&2,U32>}) ->
  {Layout.scaled_extents(shape,scale) == BandCount.extents(scaled_geometry(geometry,scale)) : List<&2,U32>}:
  Bands.Geometry{batch,channels,height,width} = geometry
  Equal.cong(List<&2,U32>,List<&2,U32>,shape => Layout.scaled_extents(shape,scale),shape,BandCount.extents(geometry),original)

def scaled_elements(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,
  +original: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {guard(geometry,shape,scale) == True{} : Bool}) ->
  {ShapeLaws.elements(Layout.scaled_extents(shape,scale)) == Nat.mul(U32.to_nat(producer_batch(geometry)),Nat.mul(U32.to_nat(producer_channels(geometry)),
    Nat.mul(Nat.mul(U32.to_nat(producer_height(geometry)),U32.to_nat(scale)),Nat.mul(U32.to_nat(producer_width(geometry)),U32.to_nat(scale))))) : Nat}:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  %Equal.sym(Nat,Nat.mul(U32.to_nat(height),U32.to_nat(scale)),U32.to_nat((height * scale : U32)),native_exact(height,scale,height_scaled(geometry,shape,scale,truth))) :
    {ShapeLaws.elements(Layout.scaled_extents(shape,scale)) == Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(_,Nat.mul(U32.to_nat(width),U32.to_nat(scale))))) : Nat}
  %Equal.sym(Nat,Nat.mul(U32.to_nat(width),U32.to_nat(scale)),U32.to_nat((width * scale : U32)),native_exact(width,scale,width_scaled(geometry,shape,scale,truth))) :
    {ShapeLaws.elements(Layout.scaled_extents(shape,scale)) == Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat((height * scale : U32)),_))) : Nat}
  Equal.trans(Nat,ShapeLaws.elements(Layout.scaled_extents(shape,scale)),ShapeLaws.elements(BandCount.extents(scaled_geometry(geometry,scale))),BandGather.elements(scaled_geometry(geometry,scale)),
    Equal.cong(List<&2,U32>,Nat,ShapeLaws.elements,Layout.scaled_extents(shape,scale),BandCount.extents(scaled_geometry(geometry,scale)),scaled_shape(geometry,shape,scale,original)),
    BandCount.domain_elements(scaled_geometry(geometry,scale)))

def grouped_elements(+images: Nat,+channels: Nat,+height: Nat,+width: Nat) ->
  {Nat.mul(images,Nat.mul(channels,Nat.mul(height,width))) == Nat.mul(Nat.mul(Nat.mul(images,height),channels),width) : Nat}:
  Equal.trans(Nat,Nat.mul(images,Nat.mul(channels,Nat.mul(height,width))),Nat.mul(images,Nat.mul(Nat.mul(channels,height),width)),Nat.mul(Nat.mul(Nat.mul(images,height),channels),width),
    Equal.cong(Nat,Nat,value => Nat.mul(images,value),Nat.mul(channels,Nat.mul(height,width)),Nat.mul(Nat.mul(channels,height),width),
      Equal.sym(Nat,Nat.mul(Nat.mul(channels,height),width),Nat.mul(channels,Nat.mul(height,width)),RowMajor.assoc(channels,height,width))),
    Equal.trans(Nat,Nat.mul(images,Nat.mul(Nat.mul(channels,height),width)),Nat.mul(Nat.mul(images,Nat.mul(channels,height)),width),Nat.mul(Nat.mul(Nat.mul(images,height),channels),width),
      Equal.sym(Nat,Nat.mul(Nat.mul(images,Nat.mul(channels,height)),width),Nat.mul(images,Nat.mul(Nat.mul(channels,height),width)),RowMajor.assoc(images,Nat.mul(channels,height),width)),
      Equal.cong(Nat,Nat,value => Nat.mul(value,width),Nat.mul(images,Nat.mul(channels,height)),Nat.mul(Nat.mul(images,height),channels),
        Equal.sym(Nat,Nat.mul(Nat.mul(images,height),channels),Nat.mul(images,Nat.mul(channels,height)),grouped_channels(images,channels,height)))))

def count_exact(+elements: Nat,+count: U32,evidence: Count.Count<elements,count>) -> {elements == U32.to_nat(count) : Nat}:
  Count.Count{exact,bounded} = evidence
  exact

def count_bounded(+elements: Nat,+count: U32,evidence: Count.Count<elements,count>) ->
  {U32.is_le(count,Machine.capacity(Default.limits())) == True{} : Bool}:
  Count.Count{exact,bounded} = evidence
  bounded

def total_rows_bound(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+capacity: U32,
  capacity_actual: {capacity == Machine.capacity(Default.limits()) : U32},
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},+width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  +original: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {guard(geometry,shape,scale) == True{} : Bool}) ->
  {Nat.is_le(Inspection.total_rows(scaled_geometry(geometry,scale)),U32.to_nat(capacity)) == True{} : Bool}:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  +allocation = {guards_allocation(geometry,shape,scale,truth) : Count.Count<ShapeLaws.elements(Layout.scaled_extents(shape,scale)),Checked.value(Tensors.count_of(Layout.scaled_extents(shape,scale)))>}
  +exact = count_exact(ShapeLaws.elements(Layout.scaled_extents(shape,scale)),Checked.value(Tensors.count_of(Layout.scaled_extents(shape,scale))),allocation)
  +bounded = {
    %Equal.sym(U32,capacity,Machine.capacity(Default.limits()),capacity_actual) : {U32.is_le(Checked.value(Tensors.count_of(Layout.scaled_extents(shape,scale))),_) == True{} : Bool}
    count_bounded(ShapeLaws.elements(Layout.scaled_extents(shape,scale)),Checked.value(Tensors.count_of(Layout.scaled_extents(shape,scale))),allocation) :
      {U32.is_le(Checked.value(Tensors.count_of(Layout.scaled_extents(shape,scale))),capacity) == True{} : Bool}}
  +output_height = U32.to_nat((height * scale : U32))
  +output_width = U32.to_nat((width * scale : U32))
  +rows = Nat.mul(U32.to_nat(batch),output_height)
  +wide_positive = {
    %native_exact(width,scale,width_scaled(geometry,shape,scale,truth)) : {Nat.is_lt(0n,_) == True{} : Bool}
    CountCompletion.positive_product(U32.to_nat(width),U32.to_nat(scale),width_positive,scale_positive(geometry,shape,scale,truth)) : {Nat.is_lt(0n,output_width) == True{} : Bool}}
  +global_room = {
    %Equal.trans(Nat,ShapeLaws.elements(Layout.scaled_extents(shape,scale)),ShapeLaws.elements(BandCount.extents(scaled_geometry(geometry,scale))),BandGather.elements(scaled_geometry(geometry,scale)),
      Equal.cong(List<&2,U32>,Nat,ShapeLaws.elements,Layout.scaled_extents(shape,scale),BandCount.extents(scaled_geometry(geometry,scale)),scaled_shape(geometry,shape,scale,original)),
      BandCount.domain_elements(scaled_geometry(geometry,scale))) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    BandCount.natural_room(ShapeLaws.elements(Layout.scaled_extents(shape,scale)),Checked.value(Tensors.count_of(Layout.scaled_extents(shape,scale))),capacity,exact,bounded) :
      {Nat.is_le(BandGather.elements(scaled_geometry(geometry,scale)),U32.to_nat(capacity)) == True{} : Bool}}
  Order.transitive(rows,BandGather.elements(scaled_geometry(geometry,scale)),U32.to_nat(capacity),
    %Equal.sym(Nat,Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(output_height,output_width))),Nat.mul(Nat.mul(rows,U32.to_nat(channels)),output_width),
      grouped_elements(U32.to_nat(batch),U32.to_nat(channels),output_height,output_width)) : {Nat.is_le(rows,_) == True{} : Bool}
    Order.transitive(rows,Nat.mul(rows,U32.to_nat(channels)),Nat.mul(Nat.mul(rows,U32.to_nat(channels)),output_width),
      RowMajor.product_covers(rows,U32.to_nat(channels),channels_positive),RowMajor.product_covers(Nat.mul(rows,U32.to_nat(channels)),output_width,wide_positive)),global_room)

def output_length_exact(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+first: U32,+rows: U32,
  descriptor: Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),producer_output_count(geometry,rows,scale)>,words: Words<scale,first,rows>,
  truth: {guard(geometry,shape,scale) == True{} : Bool}) ->
  {U32.to_nat(producer_output_count(geometry,rows,scale)) == Nat.mul(U32.to_nat(producer_channels(geometry)),
    Nat.mul(U32.to_nat((rows * scale : U32)),U32.to_nat((producer_width(geometry) * scale : U32)))) : Nat}:
  Bands.Geometry{batch,+channels,height,+width} = geometry
  Descriptor{repeated,target,count} = descriptor
  Words{first_exact,rows_exact} = words
  +exact = count_exact(ShapeLaws.elements(repeated_shape(geometry,rows,scale)),producer_output_count(geometry,rows,scale),Count.count_of(repeated_shape(geometry,rows,scale),producer_output_count(geometry,rows,scale),count))
  %Equal.sym(Nat,U32.to_nat((rows * scale : U32)),Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),rows_exact) :
    {U32.to_nat(producer_output_count(geometry,rows,scale)) == Nat.mul(U32.to_nat(channels),Nat.mul(_,U32.to_nat((width * scale : U32)))) : Nat}
  %native_exact(width,scale,width_scaled(geometry,shape,scale,truth)) :
    {U32.to_nat(producer_output_count(geometry,rows,scale)) == Nat.mul(U32.to_nat(channels),Nat.mul(Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),_)) : Nat}
  Equal.trans(Nat,U32.to_nat(producer_output_count(geometry,rows,scale)),ShapeLaws.elements(repeated_shape(geometry,rows,scale)),
    Nat.mul(U32.to_nat(channels),Nat.mul(Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),Nat.mul(U32.to_nat(width),U32.to_nat(scale)))),
    Equal.sym(Nat,ShapeLaws.elements(repeated_shape(geometry,rows,scale)),U32.to_nat(producer_output_count(geometry,rows,scale)),exact),repeated_elements(geometry,rows,scale))

def scaled_total(+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,truth: {guard(geometry,shape,scale) == True{} : Bool}) ->
  {Inspection.total_rows(scaled_geometry(geometry,scale)) == Nat.mul(Inspection.total_rows(geometry),U32.to_nat(scale)) : Nat}:
  Bands.Geometry{+batch,channels,+height,width} = geometry
  Equal.trans(Nat,Nat.mul(U32.to_nat(batch),U32.to_nat((height * scale : U32))),Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(height),U32.to_nat(scale))),
    Nat.mul(Nat.mul(U32.to_nat(batch),U32.to_nat(height)),U32.to_nat(scale)),
    Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(batch),value),U32.to_nat((height * scale : U32)),Nat.mul(U32.to_nat(height),U32.to_nat(scale)),
      Equal.sym(Nat,Nat.mul(U32.to_nat(height),U32.to_nat(scale)),U32.to_nat((height * scale : U32)),native_exact(height,scale,height_scaled(geometry,shape,scale,truth)))),
    Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(batch),U32.to_nat(height)),U32.to_nat(scale)),Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(height),U32.to_nat(scale))),RowMajor.assoc(U32.to_nat(batch),U32.to_nat(height),U32.to_nat(scale))))

# Whole-tree execution of the actual nearest-neighbour band transform.
# Each leaf retains its real checked-copy outcomes. A scaled placement is
# local to that leaf; only the full tree consumes whole-tensor coverage.
def execution_owner_descriptor(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  width_positive: {BandShape.width_positive(geometry) == True{} : Bool},metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),ready: Buffer.Sized<Element,input>,
  +original: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {guard(geometry,shape,scale) == True{} : Bool}) ->
  Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),producer_output_count(geometry,rows,scale)>:
  guarded_owner(~Element,geometry,shape,first,rows,scale,input,width_positive,metadata,ready,truth,scaled_elements(geometry,shape,scale,original,truth))

def Descriptors(~Element: Data,+geometry: Bands.Geometry,+scale: U32,bands: Model.Model<Element>) -> Data:
  match bands:
    case Model.Leaf{input,first,rows}:
      Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),producer_output_count(geometry,rows,scale)>
    case Model.Fork{left,right}: Shape.Both<Descriptors(~Element,geometry,scale,left),Descriptors(~Element,geometry,scale,right)>

def descriptors(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,bands: Model.Model<Element>,
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands)),ready: Model.Ready(~Element,bands),
  +original: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {guard(geometry,shape,scale) == True{} : Bool}) -> Descriptors(~Element,geometry,scale,bands):
  match bands:
    case Model.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+rows}:
      +input = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
      execution_owner_descriptor(~Element,geometry,shape,scale,first,rows,input,width_positive,metadata,ready,original,truth)
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{descriptors(~Element,geometry,shape,scale,left,width_positive,left_metadata,left_ready,original,truth),
        descriptors(~Element,geometry,shape,scale,right,width_positive,right_metadata,right_ready,original,truth)}

def same_extents(+extents: List<&2,U32>) -> {Shapes.same_extents(extents,extents) == True{} : Bool}:
  match extents:
    case []: {==}
    case +head <> +tail:
      %Equal.sym(Bool,U32.is_eq(head,head),True{},
        %Equal.sym(Bool,U32.is_eq(head,head),Nat.is_eq(U32.to_nat(head),U32.to_nat(head)),Comparison.eq(head,head)) : {_ == True{} : Bool}
        Order.equal_reflexive(U32.to_nat(head))) : {_ && Shapes.same_extents(tail,tail) == True{} : Bool}
      same_extents(tail)

# The first condition in Operations.copy follows from the produced Views.
# This is an actual descriptor calculation, not an external shape equality.
def same_shape(+geometry: Bands.Geometry,+scale: U32,+rows: U32,+input_count: U32,+output_count: U32,
  descriptor: Descriptor<geometry,rows,scale,input_count,output_count>) ->
  {Shapes.same(Views.shape(repeated_view(geometry,rows,scale,input_count)),
    Views.shape(canonical_view(repeated_shape(geometry,rows,scale),output_count))) == True{} : Bool}:
  Bands.Geometry{batch,channels,height,width} = geometry
  +exact = count_exact(ShapeLaws.elements(repeated_shape(geometry,rows,scale)),output_count,counted(geometry,rows,scale,input_count,output_count,descriptor))
  %Equal.sym(List<&2,U32>,Views.axis_extents_of(canonical_view(repeated_shape(geometry,rows,scale),output_count)),repeated_shape(geometry,rows,scale),
    canonical_shape(repeated_shape(geometry,rows,scale),output_count,exact)) :
    {Shapes.same_extents(repeated_shape(geometry,rows,scale),_) == True{} : Bool}
  same_extents(repeated_shape(geometry,rows,scale))

def band_model(~Element: Data,context: Layout.Upsample<Element>,first: U32,rows: U32,input: Buffer.Model<Element>) -> Model.Content<Element>:
  Model.contents(~Element,Layout.upsample_leaf(~Element,context,first,rows,Buffer.pack(~Element,input)))

def model(~Element: Data,geometry: Bands.Geometry,scale: U32,fill: Element,bands: Model.Model<Element>) -> Model.Content<Element>:
  Each.content_map(~Element,~Layout.Upsample<Element>,~band_model(~Element),Layout.Upsample{geometry,scale,fill},bands)

def band(~Element: Data,+context: Layout.Upsample<Element>,+first: U32,+rows: U32,+input: Buffer.Model<Element>) ->
  {Model.contents(~Element,Bands.transform_band(~Element,~Layout.Upsample<Element>,~Layout.upsample_leaf(~Element),context,first,rows,Buffer.pack(~Element,input),Unit{})) ==
    band_model(~Element,context,first,rows,input) : Model.Content<Element>}:
  {==}

# The band upsample on the plan of its own bands, in either mode.
def transform(~Element: Data,+geometry: Bands.Geometry,+scale: U32,+fill: Element,+parallel: Bool,+bands: Model.Model<Element>) ->
  {Model.contents(~Element,Bands.transform(~Element,~Layout.Upsample<Element>,~Layout.upsample_leaf(~Element),Model.layout(~Element,bands),parallel,geometry,
      Layout.Upsample{geometry,scale,fill},Model.pack(~Element,bands))) ==
    model(~Element,geometry,scale,fill,bands) : Model.Content<Element>}:
  Each.mapped(~Element,~Layout.Upsample<Element>,~Bands.transform_band(~Element,~Layout.Upsample<Element>,~Layout.upsample_leaf(~Element)),~band_model(~Element),~band(~Element),
    bands,parallel && Bands.same_image(geometry,Model.layout(~Element,bands)),parallel,geometry,Layout.Upsample{geometry,scale,fill})

def scheduled(~Element: Data,+geometry: Bands.Geometry,+scale: U32,+fill: Element,+workers: U32,+bands: Model.Model<Element>) ->
  {Model.contents(~Element,BandSchedule.transform(~Element,~Layout.Upsample<Element>,~Layout.upsample_leaf(~Element),Layout.Upsample{geometry,scale,fill},geometry,workers,
      (U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),Model.pack(~Element,bands))) == model(~Element,geometry,scale,fill,bands) : Model.Content<Element>}:
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,bands)),(Model.pack(~Element,bands),Model.layout(~Element,bands)),Model.plan(~Element,bands)) :
    {Model.contents(~Element,BandSchedule.transformed(~Element,~Layout.Upsample<Element>,~Layout.upsample_leaf(~Element),Layout.Upsample{geometry,scale,fill},geometry,workers,
      (U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),_)) == model(~Element,geometry,scale,fill,bands) : Model.Content<Element>}
  transform(~Element,geometry,scale,fill,BandSchedule.parallel(geometry,workers,(U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),0.0,Model.layout(~Element,bands)),bands)

def model_ready(~Element: Data,+geometry: Bands.Geometry,+scale: U32,+fill: Element,bands: Model.Model<Element>,evidence: Descriptors(~Element,geometry,scale,bands)) ->
  Model.ContentReady(~Element,model(~Element,geometry,scale,fill,bands)):
  match bands:
    case Model.Leaf{+input,+first,+rows}:
      leaf_ready(~Element,geometry,scale,fill,first,rows,input,Bands.local_count(geometry,rows),producer_output_count(geometry,rows,scale),evidence)
    case Model.Fork{+left,+right}:
      Shape.Both{left_descriptors,right_descriptors} = evidence
      Shape.Both{model_ready(~Element,geometry,scale,fill,left,left_descriptors),model_ready(~Element,geometry,scale,fill,right,right_descriptors)}

def scaled_plan(plan: Bands.Plan,+scale: U32) -> Bands.Plan:
  match plan:
    case Bands.Rows{first,rows}: Bands.Rows{(first * scale : U32),(rows * scale : U32)}
    case Bands.Split{left,right}: Bands.Split{scaled_plan(left,scale),scaled_plan(right,scale)}

def model_layout(~Element: Data,+geometry: Bands.Geometry,+scale: U32,+fill: Element,bands: Model.Model<Element>,evidence: Descriptors(~Element,geometry,scale,bands)) ->
  {Model.content_layout(~Element,model(~Element,geometry,scale,fill,bands)) == scaled_plan(Model.layout(~Element,bands),scale) : Bands.Plan}:
  match bands:
    case Model.Leaf{+input,+first,+rows}:
      leaf_layout(~Element,geometry,scale,fill,first,rows,input,Bands.local_count(geometry,rows),producer_output_count(geometry,rows,scale),evidence)
    case Model.Fork{+left,+right}:
      Shape.Both{left_descriptors,right_descriptors} = evidence
      %model_layout(~Element,geometry,scale,fill,left,left_descriptors) :
        {Bands.Split{Model.content_layout(~Element,model(~Element,geometry,scale,fill,left)),Model.content_layout(~Element,model(~Element,geometry,scale,fill,right))} ==
          Bands.Split{_,scaled_plan(Model.layout(~Element,right),scale)} : Bands.Plan}
      %model_layout(~Element,geometry,scale,fill,right,right_descriptors) :
        {Bands.Split{Model.content_layout(~Element,model(~Element,geometry,scale,fill,left)),Model.content_layout(~Element,model(~Element,geometry,scale,fill,right))} ==
          Bands.Split{Model.content_layout(~Element,model(~Element,geometry,scale,fill,left)),_} : Bands.Plan}
      {==}

type Placement<-geometry: Bands.Geometry,-scale: U32,-first: U32,-rows: U32> is Data:
  Placement{first_exact: {U32.to_nat((first * scale : U32)) == Nat.mul(U32.to_nat(first),U32.to_nat(scale)) : Nat},
    rows_exact: {U32.to_nat((rows * scale : U32)) == Nat.mul(U32.to_nat(rows),U32.to_nat(scale)) : Nat},
    metadata: BandShape.Leaf(scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),U32.to_nat(producer_output_count(geometry,rows,scale)))}

def placement_metadata(+geometry: Bands.Geometry,+scale: U32,+first: U32,+rows: U32,placement: Placement<geometry,scale,first,rows>) ->
  BandShape.Leaf(scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),U32.to_nat(producer_output_count(geometry,rows,scale))):
  Placement{first_exact,rows_exact,metadata} = placement
  metadata

def Placements(~Element: Data,+geometry: Bands.Geometry,+scale: U32,bands: Model.Model<Element>) -> Data:
  match bands:
    case Model.Leaf{input,first,rows}: Placement<geometry,scale,first,rows>
    case Model.Fork{left,right}: Shape.Both<Placements(~Element,geometry,scale,left),Placements(~Element,geometry,scale,right)>

def row_metadata(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,content: Buffer.Content<Element>,+length: Nat,
  exact: {Buffer.content_length(~Element,content) == length : Nat},fields: BandShape.Leaf(geometry,first,rows,length)) ->
  BandShape.Contents(~Element,geometry,Model.RowBand{content,first,rows}):
  match content:
    case Buffer.Content{tree,native_length}:
      %Equal.sym(Nat,native_length,length,exact) : BandShape.Leaf(geometry,first,rows,_)
      fields

def execution_leaf_metadata(~Element: Data,+geometry: Bands.Geometry,+scale: U32,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +descriptor: Descriptor<geometry,rows,scale,Bands.local_count(geometry,rows),producer_output_count(geometry,rows,scale)>,placement: Placement<geometry,scale,first,rows>) ->
  BandShape.Contents(~Element,scaled_geometry(geometry,scale),Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input)))):
  Placement{first_exact,rows_exact,metadata} = placement
  +count = producer_output_count(geometry,rows,scale)
  +source = {repeated_view(geometry,rows,scale,Bands.local_count(geometry,rows)) : Views.View}
  +target = {canonical_view(repeated_shape(geometry,rows,scale),count) : Views.View}
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Layout.upsample_leaf(~Element,Layout.Upsample{geometry,scale,fill},first,rows,Buffer.pack(~Element,input))),
    Model.RowBand{CopyValues.right_content(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target)),(first * scale : U32),(rows * scale : U32)},
    leaf_equation(~Element,geometry,scale,fill,first,rows,input,Bands.local_count(geometry,rows),count,descriptor)) : BandShape.Contents(~Element,scaled_geometry(geometry,scale),_)
  row_metadata(~Element,scaled_geometry(geometry,scale),(first * scale : U32),(rows * scale : U32),
    CopyValues.right_content(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target)),U32.to_nat(count),
    Equal.trans(Nat,Buffer.content_length(~Element,CopyValues.right_content(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target))),
      Buffer.content_length(~Element,Buffer.contents(~Element,Tensors.buffer(~Element,fill,count))),U32.to_nat(count),
      CopyValues.copy_length(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target),
      Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.contents(~Element,Tensors.buffer(~Element,fill,count)),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)},
        AllocationLaws.contents(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)))),metadata)

def model_metadata(~Element: Data,+geometry: Bands.Geometry,+scale: U32,+fill: Element,bands: Model.Model<Element>,descriptors: Descriptors(~Element,geometry,scale,bands),placements: Placements(~Element,geometry,scale,bands)) ->
  BandShape.Contents(~Element,scaled_geometry(geometry,scale),model(~Element,geometry,scale,fill,bands)):
  match bands:
    case Model.Leaf{+input,+first,+rows}: execution_leaf_metadata(~Element,geometry,scale,fill,first,rows,input,descriptors,placements)
    case Model.Fork{+left,+right}:
      Shape.Both{left_descriptors,right_descriptors} = descriptors
      Shape.Both{left_placements,right_placements} = placements
      Shape.Both{model_metadata(~Element,geometry,scale,fill,left,left_descriptors,left_placements),model_metadata(~Element,geometry,scale,fill,right,right_descriptors,right_placements)}

def scaled_coverage(~Element: Data,+geometry: Bands.Geometry,+scale: U32,bands: Model.Model<Element>,+first: Nat,+rows: Nat,
  placements: Placements(~Element,geometry,scale,bands),coverage: BandPlan.Coverage(Model.layout(~Element,bands),first,rows)) ->
  BandPlan.Coverage(scaled_plan(Model.layout(~Element,bands),scale),Nat.mul(first,U32.to_nat(scale)),Nat.mul(rows,U32.to_nat(scale))):
  match bands:
    case Model.Leaf{input,+native_first,+native_rows}:
      Placement{first_exact,rows_exact,metadata} = placements
      BandPlan.LeafCoverage{first_equal,rows_equal} = coverage
      BandPlan.LeafCoverage{Equal.trans(Nat,U32.to_nat((native_first * scale : U32)),Nat.mul(U32.to_nat(native_first),U32.to_nat(scale)),Nat.mul(first,U32.to_nat(scale)),first_exact,
          Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(scale)),U32.to_nat(native_first),first,first_equal)),
        Equal.trans(Nat,U32.to_nat((native_rows * scale : U32)),Nat.mul(U32.to_nat(native_rows),U32.to_nat(scale)),Nat.mul(rows,U32.to_nat(scale)),rows_exact,
          Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(scale)),U32.to_nat(native_rows),rows,rows_equal))}
    case Model.Fork{+left,+right}:
      Shape.Both{left_placements,right_placements} = placements
      BandPlan.ForkCoverage{+left_rows,+right_rows,+rows_equal,left_coverage,right_coverage} = coverage
      BandPlan.ForkCoverage{Nat.mul(left_rows,U32.to_nat(scale)),Nat.mul(right_rows,U32.to_nat(scale)),
        Equal.trans(Nat,Nat.mul(rows,U32.to_nat(scale)),Nat.mul(Nat.add(left_rows,right_rows),U32.to_nat(scale)),Nat.add(Nat.mul(left_rows,U32.to_nat(scale)),Nat.mul(right_rows,U32.to_nat(scale))),
          Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(scale)),rows,Nat.add(left_rows,right_rows),rows_equal),RowMajor.distrib(left_rows,right_rows,U32.to_nat(scale))),
        scaled_coverage(~Element,geometry,scale,left,first,left_rows,left_placements,left_coverage),
        %RowMajor.distrib(first,left_rows,U32.to_nat(scale)) : BandPlan.Coverage(scaled_plan(Model.layout(~Element,right),scale),_,Nat.mul(right_rows,U32.to_nat(scale)))
        scaled_coverage(~Element,geometry,scale,right,Nat.add(first,left_rows),right_rows,right_placements,right_coverage)}

def scheduled_well_formed(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,+bands: Model.Model<Element>,
  width_positive: {BandShape.width_positive(geometry) == True{} : Bool},source: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,
  +original: {shape == BandCount.extents(geometry) : List<&2,U32>},+truth: {guard(geometry,shape,scale) == True{} : Bool},+placements: Placements(~Element,geometry,scale,bands)) ->
  BandShape.WellFormed<Element,scaled_geometry(geometry,scale),Model.contents(~Element,BandSchedule.transform(~Element,~Layout.Upsample<Element>,~Layout.upsample_leaf(~Element),
    Layout.Upsample{geometry,scale,fill},geometry,workers,(U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),Model.pack(~Element,bands)))>:
  BandShape.WellFormed{metadata,ready,coverage} = source
  +produced_descriptors = {descriptors(~Element,geometry,shape,scale,bands,width_positive,metadata,Model.ready(~Element,bands,ready),original,truth) : Descriptors(~Element,geometry,scale,bands)}
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandSchedule.transform(~Element,~Layout.Upsample<Element>,~Layout.upsample_leaf(~Element),Layout.Upsample{geometry,scale,fill},geometry,workers,
      (U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),Model.pack(~Element,bands))),model(~Element,geometry,scale,fill,bands),scheduled(~Element,geometry,scale,fill,workers,bands)) :
    BandShape.WellFormed<Element,scaled_geometry(geometry,scale),_>
  BandShape.WellFormed{model_metadata(~Element,geometry,scale,fill,bands,produced_descriptors,placements),model_ready(~Element,geometry,scale,fill,bands,produced_descriptors),
    %Equal.sym(Bands.Plan,Model.content_layout(~Element,model(~Element,geometry,scale,fill,bands)),scaled_plan(Model.layout(~Element,bands),scale),model_layout(~Element,geometry,scale,fill,bands,produced_descriptors)) :
      BandPlan.Coverage(_,0n,Inspection.total_rows(scaled_geometry(geometry,scale)))
    %Equal.sym(Nat,Inspection.total_rows(scaled_geometry(geometry,scale)),Nat.mul(Inspection.total_rows(geometry),U32.to_nat(scale)),scaled_total(geometry,shape,scale,truth)) :
      BandPlan.Coverage(scaled_plan(Model.layout(~Element,bands),scale),0n,_)
    scaled_coverage(~Element,geometry,scale,bands,0n,Inspection.total_rows(geometry),placements,
      %Equal.sym(Bands.Plan,Model.layout(~Element,bands),Model.content_layout(~Element,Model.content(~Element,bands)),Model.layout_matches_content(~Element,bands)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry)) coverage)}
