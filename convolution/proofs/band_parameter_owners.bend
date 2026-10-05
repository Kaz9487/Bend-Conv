# Parameter flattening uses the actual with_dense callback and whole gather.
# Finite reads reuse the proved gather assignments and coverage. Storage
# readiness follows the same write program; parameter padding is irrelevant.
import Base
import ../specification.bend as ConvSpec
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_band.bend as Bands
import ../../lib/storage_buffer.bend as Storage
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/math_activation.bend as Activations
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/proofs/tensor_band_count.bend as BandCount
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/proofs/tensor_dense_contract.bend as Dense
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/proofs/tensor_band_dense.bend as BandDense
import ../../lib/proofs/storage_allocation_refinement.bend as AllocationLaws
import ../../lib/default_machine_limits.bend as Default
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/row_major_coordinates.bend as RowCoordinates
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ./input_storage.bend as Inputs
import ./band_native_caller.bend as Native
import ./band_normal_caller.bend as NormalCaller
import ./band_public_frame.bend as PublicFrame
import ./list_array_roundtrip.bend as Roundtrip

def flattened(tensor: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  Tensors.with_dense(~F32,~Tensors.Tensor<F32>,tensor,storage => shape => fill => status => workers =>
    Tensors.Tensor{storage,shape,fill,status,workers})

def with_dense_band(~Result: Type,+geometry: Bands.Geometry,+bands: Model.Model<F32>,+shape: List<&2,U32>,+fill: F32,+status: Tensors.Status,+workers: U32,
  body: Storage.Buffer<F32> -> List<&2,U32> -> F32 -> Tensors.Status -> U32 -> Result) ->
  {Tensors.with_dense(~F32,~Result,Tensors.Banded{Model.pack(~F32,bands),geometry,shape,fill,status,workers},body) ==
    body(Bands.gather(~F32,geometry,Model.pack(~F32,bands),Tensors.buffer(~F32,fill,Bands.elements(geometry))),shape,fill,status,workers) : Result}:
  {==}

type LinearPoint<-geometry: Bands.Geometry,-position: Nat> is Data:
  LinearPoint{image: Nat,channel: Nat,row: Nat,column: Nat,
    inside: BandGather.Inside(geometry,image,channel,row,column),
    exact: {position == BandGather.point(geometry,image,channel,row,column) : Nat}}

def split_coordinates(+images: U32,+channels: U32,+height: U32,+width: U32,+position: Nat,
  +channels_positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool},
  coordinates: RowCoordinates.Coordinates<Nat.mul(U32.to_nat(images),U32.to_nat(channels)),U32.to_nat(height),U32.to_nat(width),position>) ->
  LinearPoint<Bands.Geometry{images,channels,height,width},position>:
  RowCoordinates.Coordinates{+combined,+row,+column,+combined_inside,row_inside,column_inside,+exact} = coordinates
  +image = Nat.div(combined,U32.to_nat(channels))
  +channel = Nat.mod(combined,U32.to_nat(channels))
  +split = DivisionOrder.quotient_exact(combined,U32.to_nat(channels),channels_positive)
  LinearPoint{image,channel,row,column,
    BandGather.PointInside{RowMajor.row_inside(image,U32.to_nat(images),U32.to_nat(channels),channel,
        %split : {Nat.is_lt(_,Nat.mul(U32.to_nat(images),U32.to_nat(channels))) == True{} : Bool} combined_inside),
      DivisionOrder.remainder_small(combined,U32.to_nat(channels),channels_positive),row_inside,column_inside},
    Equal.trans(Nat,position,RowMajor.index(combined,U32.to_nat(height),U32.to_nat(width),row,column),BandGather.point(Bands.Geometry{images,channels,height,width},image,channel,row,column),
      exact,Equal.cong(Nat,Nat,combined => RowMajor.index(combined,U32.to_nat(height),U32.to_nat(width),row,column),combined,Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),split))}

def linear_point(+geometry: Bands.Geometry,+position: Nat,
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  inside: {Nat.is_lt(position,BandGather.elements(geometry)) == True{} : Bool}) -> LinearPoint<geometry,position>:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  split_coordinates(images,channels,height,width,position,channels_positive,
    RowCoordinates.locate(Nat.mul(U32.to_nat(images),U32.to_nat(channels)),U32.to_nat(height),U32.to_nat(width),position,width_positive,
      %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(images),U32.to_nat(channels)),Nat.mul(U32.to_nat(height),U32.to_nat(width))),BandGather.elements(geometry),
        RowMajor.assoc(U32.to_nat(images),U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))) : {Nat.is_lt(position,_) == True{} : Bool}
      inside))

def channels_positive(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>) ->
  {BandGather.channels_positive(geometry) == True{} : Bool}:
  BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
  channels_positive

def width_positive(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>) ->
  {BandGather.width_positive(geometry) == True{} : Bool}:
  BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
  width_positive

def depth_ready(+result: Buffer.Model<F32>,ready: Buffer.Ready<F32,result>) ->
  {Nat.is_le(Buffer.depth(~F32,result),30n) == True{} : Bool}:
  Buffer.Ready{depth_ok,room} = ready
  depth_ok

def point_read(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,+bands: Model.Model<F32>,+result: Buffer.Model<F32>,
  +domain: BandGather.Domain<geometry,limits,Tree.capacity(F32,Buffer.tree(~F32,result))>,source_shape: BandShape.WellFormed<F32,geometry,Model.content(~F32,bands)>,
  values: BandValues.Values(~F32,~Logical,~original,logical,geometry,Model.content(~F32,bands)),ready: Buffer.Ready<F32,result>,+position: Nat,
  point: LinearPoint<geometry,position>) ->
  {Buffer.value(~F32,U32.from_nat(position),Buffer.contents(~F32,Bands.gather(~F32,geometry,Model.pack(~F32,bands),Buffer.pack(~F32,result)))) == original(logical,U32.from_nat(position)) : F32}:
  LinearPoint{+image,+channel,+row,+column,+inside,+exact} = point
  %Equal.sym(Nat,position,BandGather.point(geometry,image,channel,row,column),exact) :
    {Buffer.value(~F32,U32.from_nat(_),Buffer.contents(~F32,Bands.gather(~F32,geometry,Model.pack(~F32,bands),Buffer.pack(~F32,result)))) == original(logical,U32.from_nat(_)) : F32}
  BandGather.gathered_value(~F32,~Logical,~original,logical,geometry,limits,bands,result,domain,
    BandShape.well_formed_checked(~F32,geometry,bands,Machine.capacity(limits),Limits.capacity_fits(limits),
      BandGather.rows_room(geometry,limits,Tree.capacity(F32,Buffer.tree(~F32,result)),domain),source_shape),values,image,channel,row,column,inside,depth_ready(result,ready))

def finite_read(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,+bands: Model.Model<F32>,+result: Buffer.Model<F32>,
  +domain: BandGather.Domain<geometry,limits,Tree.capacity(F32,Buffer.tree(~F32,result))>,source_shape: BandShape.WellFormed<F32,geometry,Model.content(~F32,bands)>,
  values: BandValues.Values(~F32,~Logical,~original,logical,geometry,Model.content(~F32,bands)),ready: Buffer.Ready<F32,result>,+position: Nat,
  inside: {Nat.is_lt(position,BandGather.elements(geometry)) == True{} : Bool}) ->
  Dense.At(~F32,~Logical,~original,logical,Buffer.contents(~F32,Bands.gather(~F32,geometry,Model.pack(~F32,bands),Buffer.pack(~F32,result))),position):
  point_read(~Logical,~original,logical,geometry,limits,bands,result,domain,source_shape,values,ready,position,
    linear_point(geometry,position,channels_positive(geometry,limits,Tree.capacity(F32,Buffer.tree(~F32,result)),domain),
      width_positive(geometry,limits,Tree.capacity(F32,Buffer.tree(~F32,result)),domain),inside))

def finite_values(~Logical: Data,~original: Logical -> U32 -> F32,count: Nat,+start: Nat,+logical: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,+bands: Model.Model<F32>,+result: Buffer.Model<F32>,
  +domain: BandGather.Domain<geometry,limits,Tree.capacity(F32,Buffer.tree(~F32,result))>,+source_shape: BandShape.WellFormed<F32,geometry,Model.content(~F32,bands)>,
  +values: BandValues.Values(~F32,~Logical,~original,logical,geometry,Model.content(~F32,bands)),+ready: Buffer.Ready<F32,result>,
  +room: {Nat.is_le(Nat.add(start,count),BandGather.elements(geometry)) == True{} : Bool}) ->
  Members.All(Nat,{position => Dense.At(~F32,~Logical,~original,logical,Buffer.contents(~F32,Bands.gather(~F32,geometry,Model.pack(~F32,bands),Buffer.pack(~F32,result))),position) : Nat -> Data},Ranges.range(count,start)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{finite_read(~Logical,~original,logical,geometry,limits,bands,result,domain,source_shape,values,ready,start,Order.room_index(start,rest,BandGather.elements(geometry),room)),
        finite_values(~Logical,~original,rest,1n+start,logical,geometry,limits,bands,result,domain,source_shape,values,ready,Order.advance_room(start,rest,BandGather.elements(geometry),room))}

def gathered_length(+geometry: Bands.Geometry,+bands: Model.Model<F32>,+result: Buffer.Model<F32>) ->
  {Buffer.content_length(~F32,Buffer.contents(~F32,Bands.gather(~F32,geometry,Model.pack(~F32,bands),Buffer.pack(~F32,result)))) == Buffer.length(~F32,result) : Nat}:
  Equal.cong(Buffer.Content<F32>,Nat,Buffer.content_length(~F32),
    Buffer.contents(~F32,Bands.gather(~F32,geometry,Model.pack(~F32,bands),Buffer.pack(~F32,result))),
    Buffer.Content{Writes.pure(F32,BandGather.refinement_writes(~F32,geometry,bands),Buffer.tree(~F32,result)),Buffer.length(~F32,result)},BandGather.gathered_writes(~F32,geometry,bands,result))

def gathered_ready(+geometry: Bands.Geometry,+bands: Model.Model<F32>,+result: Buffer.Model<F32>,ready: Buffer.Ready<F32,result>) ->
  Buffer.ContentReady<F32,Buffer.contents(~F32,Bands.gather(~F32,geometry,Model.pack(~F32,bands),Buffer.pack(~F32,result)))>:
  Buffer.Model{+tree,+length,certificate} = result
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Bands.gather(~F32,geometry,Model.pack(~F32,bands),Buffer.pack(~F32,result))),
    Buffer.Content{Writes.pure(F32,BandGather.refinement_writes(~F32,geometry,bands),tree),length},BandGather.gathered_writes(~F32,geometry,bands,result)) : Buffer.ContentReady<F32,_>
  Writes.content_ready(~F32,BandGather.refinement_writes(~F32,geometry,bands),tree,length,Buffer.content_ready(~F32,result,ready))

def gathered_values(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,+bands: Model.Model<F32>,+result: Buffer.Model<F32>,
  +domain: BandGather.Domain<geometry,limits,Tree.capacity(F32,Buffer.tree(~F32,result))>,+source_shape: BandShape.WellFormed<F32,geometry,Model.content(~F32,bands)>,
  +values: BandValues.Values(~F32,~Logical,~original,logical,geometry,Model.content(~F32,bands)),+ready: Buffer.Ready<F32,result>,
  +length_exact: {Buffer.length(~F32,result) == BandGather.elements(geometry) : Nat}) ->
  Dense.Values(~F32,~Logical,~original,logical,Buffer.contents(~F32,Bands.gather(~F32,geometry,Model.pack(~F32,bands),Buffer.pack(~F32,result)))):
  %Equal.sym(Nat,Buffer.content_length(~F32,Buffer.contents(~F32,Bands.gather(~F32,geometry,Model.pack(~F32,bands),Buffer.pack(~F32,result)))),Buffer.length(~F32,result),gathered_length(geometry,bands,result)) :
    Members.All(Nat,{position => Dense.At(~F32,~Logical,~original,logical,Buffer.contents(~F32,Bands.gather(~F32,geometry,Model.pack(~F32,bands),Buffer.pack(~F32,result))),position) : Nat -> Data},Ranges.range(_,0n))
  finite_values(~Logical,~original,Buffer.length(~F32,result),0n,logical,geometry,limits,bands,result,domain,source_shape,values,ready,
    %Equal.sym(Nat,Buffer.length(~F32,result),BandGather.elements(geometry),length_exact) : {Nat.is_le(_,BandGather.elements(geometry)) == True{} : Bool}
    Order.reflexive(BandGather.elements(geometry)))

def gathered_contract(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,+limits: Machine.Limits,
  +bands: Model.Model<F32>,+result: Buffer.Model<F32>,+domain: BandGather.Domain<geometry,limits,Tree.capacity(F32,Buffer.tree(~F32,result))>,
  +source_shape: BandShape.WellFormed<F32,geometry,Model.content(~F32,bands)>,values: BandValues.Values(~F32,~Logical,~original,logical,geometry,Model.content(~F32,bands)),+ready: Buffer.Ready<F32,result>,
  +length_exact: {Buffer.length(~F32,result) == BandGather.elements(geometry) : Nat},coherent: {ShapeLaws.elements(shape) == BandGather.elements(geometry) : Nat}) ->
  Dense.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{Bands.gather(~F32,geometry,Model.pack(~F32,bands),Buffer.pack(~F32,result)),shape,fill,Tensors.Ready{},workers}):
  Shape.Both{Shape.Both{Dense.WellFormed{
      Equal.trans(Nat,Buffer.content_length(~F32,Buffer.contents(~F32,Bands.gather(~F32,geometry,Model.pack(~F32,bands),Buffer.pack(~F32,result)))),Buffer.length(~F32,result),ShapeLaws.elements(shape),
        gathered_length(geometry,bands,result),Equal.trans(Nat,Buffer.length(~F32,result),BandGather.elements(geometry),ShapeLaws.elements(shape),length_exact,Equal.sym(Nat,ShapeLaws.elements(shape),BandGather.elements(geometry),coherent))),
      gathered_ready(geometry,bands,result,ready)},{==}},
    gathered_values(~Logical,~original,logical,geometry,limits,bands,result,domain,source_shape,values,ready,length_exact)}

# Every parameter sequence is an observation of the supplied physical owner.
# No constructor identity or equality between unused padding cells is needed.
def sequence(content: Buffer.Content<F32>) -> List<&2,F32>:
  Buffer.Content{tree,length} = content
  Tree.read_sequence(~F32,length,0,tree)

def sequence_evidence(+content: Buffer.Content<F32>,ready: Buffer.ContentReady<F32,content>) ->
  StorageLaws.SequenceEvidence<F32,ConvSpec.length(sequence(content)),sequence(content)>:
  Buffer.Content{+tree,+length} = content
  Buffer.ContentReady{depth,balanced,depth_ok,room} = ready
  Inputs.prefix_evidence(length,tree,room)

def sequence_tree(+content: Buffer.Content<F32>,+ready: Buffer.ContentReady<F32,content>) ->
  {StorageLaws.sequence_storage(F32,ConvSpec.length(sequence(content)),sequence(content),sequence_evidence(content,ready)) == Buffer.content_tree(~F32,content) : Tree.Tree<F32>}:
  Buffer.Content{+tree,+length} = content
  Buffer.ContentReady{depth,balanced,depth_ok,room} = ready
  {==}

def sequence_length(+content: Buffer.Content<F32>) ->
  {ConvSpec.length(sequence(content)) == Buffer.content_length(~F32,content) : Nat}:
  Buffer.Content{tree,length} = content
  Inputs.observed_length(length,tree)

def sequence_value(values: List<&2,F32>,index: U32) -> F32:
  ConvSpec.at(U32.to_nat(index),values)

def content_value(+content: Buffer.Content<F32>,+index: U32) ->
  {Tree.read(F32,Buffer.content_tree(~F32,content),index) == Buffer.value(~F32,index,content) : F32}:
  Buffer.Content{tree,length} = content
  {==}

def list_read(+values: List<&2,F32>,+content: Buffer.Content<F32>,+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},
  +length_exact: {Buffer.content_length(~F32,content) == ConvSpec.length(values) : Nat},+within: {Nat.is_le(ConvSpec.length(values),U32.to_nat(upper)) == True{} : Bool},
  contents: Dense.Values(~F32,~(List<&2,F32>),~sequence_value,values,content),+index: Nat,+inside: {Nat.is_lt(index,ConvSpec.length(values)) == True{} : Bool}) ->
  {Buffer.value(~F32,U32.from_nat(index),content) == ConvSpec.at(index,values) : F32}:
  Equal.trans(F32,Buffer.value(~F32,U32.from_nat(index),content),sequence_value(values,U32.from_nat(index)),ConvSpec.at(index,values),
    Dense.read(~F32,~(List<&2,F32>),~sequence_value,values,content,index,contents,
      %Equal.sym(Nat,Buffer.content_length(~F32,content),ConvSpec.length(values),length_exact) : {Nat.is_lt(index,_) == True{} : Bool} inside),
    Equal.cong(Nat,F32,index => ConvSpec.at(index,values),U32.to_nat(U32.from_nat(index)),index,
      Bounds.roundtrip(index,upper,upper_ok,Bounds.strict_le(index,U32.to_nat(upper),Bounds.nat_trans(index,ConvSpec.length(values),U32.to_nat(upper),inside,within)))))

# This induction only collects the finite read equalities already established
# by gather; it does not introduce another executable gather or copy schedule.
def list_readback(count: Nat,+index: Nat,+values: List<&2,F32>,+content: Buffer.Content<F32>,+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},
  +length_exact: {Buffer.content_length(~F32,content) == ConvSpec.length(values) : Nat},+within: {Nat.is_le(ConvSpec.length(values),U32.to_nat(upper)) == True{} : Bool},
  +contents: Dense.Values(~F32,~(List<&2,F32>),~sequence_value,values,content),+room: {Nat.is_le(Nat.add(index,count),ConvSpec.length(values)) == True{} : Bool}) ->
  {Tree.read_sequence(~F32,count,U32.from_nat(index),Buffer.content_tree(~F32,content)) == Roundtrip.enumerate(count,index,values) : List<&2,F32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(F32,Tree.read(F32,Buffer.content_tree(~F32,content),U32.from_nat(index)),ConvSpec.at(index,values),
        Equal.trans(F32,Tree.read(F32,Buffer.content_tree(~F32,content),U32.from_nat(index)),Buffer.value(~F32,U32.from_nat(index),content),ConvSpec.at(index,values),content_value(content,U32.from_nat(index)),
          list_read(values,content,upper,upper_ok,length_exact,within,contents,index,Order.room_index(index,rest,ConvSpec.length(values),room)))) :
        {_ <> Tree.read_sequence(~F32,rest,(U32.from_nat(index) + 1 : U32),Buffer.content_tree(~F32,content)) == Roundtrip.enumerate(1n+rest,index,values) : List<&2,F32>}
      %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.add_one(U32.from_nat(index))) :
        {ConvSpec.at(index,values) <> Tree.read_sequence(~F32,rest,_,Buffer.content_tree(~F32,content)) == Roundtrip.enumerate(1n+rest,index,values) : List<&2,F32>}
      Equal.cong(List<&2,F32>,List<&2,F32>,tail => ConvSpec.at(index,values) <> tail,Tree.read_sequence(~F32,rest,U32.from_nat(1n+index),Buffer.content_tree(~F32,content)),Roundtrip.enumerate(rest,1n+index,values),
        list_readback(rest,1n+index,values,content,upper,upper_ok,length_exact,within,contents,Order.advance_room(index,rest,ConvSpec.length(values),room)))

def sequence_equals(+content: Buffer.Content<F32>,+values: List<&2,F32>,+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},
  +length_exact: {Buffer.content_length(~F32,content) == ConvSpec.length(values) : Nat},within: {Nat.is_le(ConvSpec.length(values),U32.to_nat(upper)) == True{} : Bool},
  contents: Dense.Values(~F32,~(List<&2,F32>),~sequence_value,values,content)) -> {sequence(content) == values : List<&2,F32>}:
  Buffer.Content{+tree,+length} = content
  %Equal.sym(Nat,length,ConvSpec.length(values),length_exact) : {Tree.read_sequence(~F32,_,0,tree) == values : List<&2,F32>}
  Equal.trans(List<&2,F32>,Tree.read_sequence(~F32,ConvSpec.length(values),0,tree),Roundtrip.enumerate(ConvSpec.length(values),0n,values),values,
    list_readback(ConvSpec.length(values),0n,values,content,upper,upper_ok,length_exact,within,contents,Order.reflexive(ConvSpec.length(values))),Roundtrip.enumerated(values))

def list_evidence(+content: Buffer.Content<F32>,+values: List<&2,F32>,+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},
  +length_exact: {Buffer.content_length(~F32,content) == ConvSpec.length(values) : Nat},within: {Nat.is_le(ConvSpec.length(values),U32.to_nat(upper)) == True{} : Bool},
  contents: Dense.Values(~F32,~(List<&2,F32>),~sequence_value,values,content),ready: Buffer.ContentReady<F32,content>) -> StorageLaws.SequenceEvidence<F32,ConvSpec.length(values),values>:
  Buffer.Content{+tree,+length} = content
  Buffer.ContentReady{depth,balanced,depth_ok,room} = ready
  StorageLaws.SequenceEvidence{tree,
    %length_exact : {Tree.read_sequence(~F32,_,0,tree) == values : List<&2,F32>}
    sequence_equals(content,values,upper,upper_ok,length_exact,within,contents),
    %length_exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool} room}

def model_length(+model: Buffer.Model<F32>) -> {Buffer.content_length(~F32,Buffer.content(~F32,model)) == Buffer.length(~F32,model) : Nat}:
  Buffer.Model{tree,length,certificate} = model
  {==}

def allocation_length(+count: U32,+fill: F32,+result: Buffer.Model<F32>,
  equation: {Buffer.pack(~F32,result) == Tensors.buffer(~F32,fill,count) : Storage.Buffer<F32>}) -> {Buffer.length(~F32,result) == U32.to_nat(count) : Nat}:
  Equal.trans(Nat,Buffer.length(~F32,result),Buffer.content_length(~F32,Buffer.contents(~F32,Buffer.pack(~F32,result))),U32.to_nat(count),
    Equal.sym(Nat,Buffer.content_length(~F32,Buffer.contents(~F32,Buffer.pack(~F32,result))),Buffer.length(~F32,result),
      Equal.trans(Nat,Buffer.content_length(~F32,Buffer.contents(~F32,Buffer.pack(~F32,result))),Buffer.content_length(~F32,Buffer.content(~F32,result)),Buffer.length(~F32,result),
        Equal.cong(Buffer.Content<F32>,Nat,Buffer.content_length(~F32),Buffer.contents(~F32,Buffer.pack(~F32,result)),Buffer.content(~F32,result),Buffer.packed_contents(~F32,result)),model_length(result))),
    Equal.trans(Nat,Buffer.content_length(~F32,Buffer.contents(~F32,Buffer.pack(~F32,result))),Buffer.content_length(~F32,Buffer.contents(~F32,Tensors.buffer(~F32,fill,count))),U32.to_nat(count),
      Equal.cong(Storage.Buffer<F32>,Nat,buffer => Buffer.content_length(~F32,Buffer.contents(~F32,buffer)),Buffer.pack(~F32,result),Tensors.buffer(~F32,fill,count),equation),
      Equal.cong(Buffer.Content<F32>,Nat,Buffer.content_length(~F32),Buffer.contents(~F32,Tensors.buffer(~F32,fill,count)),
        Buffer.Content{Tree.fresh(F32,Tensors.depth_for(count),fill),U32.to_nat(count)},AllocationLaws.contents(~F32,Tensors.depth_for(count),fill,U32.to_nat(count)))))

def allocated_observed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,+limits: Machine.Limits,+capacity: U32,
  +bands: Model.Model<F32>,+whole: BandGather.Domain<geometry,limits,capacity>,source_shape: BandShape.WellFormed<F32,geometry,Model.content(~F32,bands)>,
  values: BandValues.Values(~F32,~Logical,~original,logical,geometry,Model.content(~F32,bands)),coherent: {ShapeLaws.elements(shape) == BandGather.elements(geometry) : Nat},
  observation: Buffer.Observation<F32,Tensors.buffer(~F32,fill,Bands.elements(geometry))>) ->
  Dense.Contract(~F32,~Logical,~original,logical,flattened(Tensors.Banded{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Ready{},workers})):
  Buffer.Observation{+result,+equation} = observation
  +ready = Buffer.ready_observed(~F32,Tensors.buffer(~F32,fill,Bands.elements(geometry)),result,equation,
    BandCount.allocated(~F32,Bands.elements(geometry),fill,BandCount.allocatable(geometry,limits,capacity,whole)))
  +length_exact = Equal.trans(Nat,Buffer.length(~F32,result),U32.to_nat(Bands.elements(geometry)),BandGather.elements(geometry),
    allocation_length(Bands.elements(geometry),fill,result,equation),BandCount.exact(geometry,limits,capacity,whole))
  %equation : Dense.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{Bands.gather(~F32,geometry,Model.pack(~F32,bands),_),shape,fill,Tensors.Ready{},workers})
  gathered_contract(~Logical,~original,logical,geometry,shape,fill,workers,limits,bands,result,
    BandDense.owned(~F32,geometry,limits,capacity,result,Equal.sym(Nat,Buffer.length(~F32,result),BandGather.elements(geometry),length_exact),ready,whole),
    source_shape,values,ready,length_exact,coherent)

def banded(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,+limits: Machine.Limits,+capacity: U32,
  +bands: Model.Model<F32>,whole: BandGather.Domain<geometry,limits,capacity>,source_shape: BandShape.WellFormed<F32,geometry,Model.content(~F32,bands)>,
  values: BandValues.Values(~F32,~Logical,~original,logical,geometry,Model.content(~F32,bands)),coherent: {ShapeLaws.elements(shape) == BandGather.elements(geometry) : Nat}) ->
  Dense.Contract(~F32,~Logical,~original,logical,flattened(Tensors.Banded{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Ready{},workers})):
  allocated_observed(~Logical,~original,logical,geometry,shape,fill,workers,limits,capacity,bands,whole,source_shape,values,coherent,
    Buffer.observe(~F32,Tensors.buffer(~F32,fill,Bands.elements(geometry))))

def normal_banded(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,+bands: Model.Model<F32>,
  normal: Global.Normal(~F32,Tensors.Banded{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Ready{},workers}),
  values: BandValues.Values(~F32,~Logical,~original,logical,geometry,Model.contents(~F32,Model.pack(~F32,bands))),
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool}) ->
  Dense.Contract(~F32,~Logical,~original,logical,flattened(Tensors.Banded{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Ready{},workers})):
  Shape.Both{source_shape,Shape.Both{+counted,+coherent}} = normal
  banded(~Logical,~original,logical,geometry,shape,fill,workers,Default.limits(),Machine.capacity(Default.limits()),bands,
    Global.domain(shape,geometry,counted,coherent,channels_positive,height_positive,width_positive),
    %Model.packed_contents(~F32,bands) : BandShape.WellFormed<F32,geometry,_> source_shape,
    %Model.packed_contents(~F32,bands) : BandValues.Values(~F32,~Logical,~original,logical,geometry,_) values,
    BandCount.shape_coherence_elements(shape,geometry,coherent))

type Parameter<-buffer: Storage.Buffer<F32>> is Data:
  Parameter{model: Buffer.Model<F32>,equation: {Buffer.pack(~F32,model) == buffer : Storage.Buffer<F32>},ready: Buffer.Ready<F32,model>,
    values: List<&2,F32>,length_exact: {ConvSpec.length(values) == Buffer.length(~F32,model) : Nat},
    evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(values),values>,
    tree_exact: {Buffer.tree(~F32,model) == StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence) : Tree.Tree<F32>}}

def model_tree(+model: Buffer.Model<F32>) -> {Buffer.content_tree(~F32,Buffer.content(~F32,model)) == Buffer.tree(~F32,model) : Tree.Tree<F32>}:
  Buffer.Model{tree,length,certificate} = model
  {==}

def observed_parameter(-buffer: Storage.Buffer<F32>,+ready: Buffer.ContentReady<F32,Buffer.contents(~F32,buffer)>,observation: Buffer.Observation<F32,buffer>) -> Parameter<buffer>:
  Buffer.Observation{+model,+equation} = observation
  +model_ready = Buffer.ready_content(~F32,model,
    %Buffer.packed_contents(~F32,model) : Buffer.ContentReady<F32,_>
    %Equal.sym(Storage.Buffer<F32>,Buffer.pack(~F32,model),buffer,equation) : Buffer.ContentReady<F32,Buffer.contents(~F32,_)>
    ready)
  +content_ready = Buffer.content_ready(~F32,model,model_ready)
  Parameter{model,equation,model_ready,sequence(Buffer.content(~F32,model)),
    Equal.trans(Nat,ConvSpec.length(sequence(Buffer.content(~F32,model))),Buffer.content_length(~F32,Buffer.content(~F32,model)),Buffer.length(~F32,model),sequence_length(Buffer.content(~F32,model)),model_length(model)),
    sequence_evidence(Buffer.content(~F32,model),content_ready),
    Equal.sym(Tree.Tree<F32>,StorageLaws.sequence_storage(F32,ConvSpec.length(sequence(Buffer.content(~F32,model))),sequence(Buffer.content(~F32,model)),sequence_evidence(Buffer.content(~F32,model),content_ready)),Buffer.tree(~F32,model),
      Equal.trans(Tree.Tree<F32>,StorageLaws.sequence_storage(F32,ConvSpec.length(sequence(Buffer.content(~F32,model))),sequence(Buffer.content(~F32,model)),sequence_evidence(Buffer.content(~F32,model),content_ready)),
        Buffer.content_tree(~F32,Buffer.content(~F32,model)),Buffer.tree(~F32,model),sequence_tree(Buffer.content(~F32,model),content_ready),model_tree(model)))}

def parameter(buffer: Storage.Buffer<F32>,ready: Buffer.ContentReady<F32,Buffer.contents(~F32,buffer)>) -> Parameter<buffer>:
  observed_parameter(buffer,ready,Buffer.observe(~F32,buffer))

def contract_ready(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,-buffer: Storage.Buffer<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  contract: Dense.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{buffer,shape,fill,Tensors.Ready{},workers})) -> Buffer.ContentReady<F32,Buffer.contents(~F32,buffer)>:
  Shape.Both{Shape.Both{Dense.WellFormed{length_exact,ready},status_ready},values} = contract
  ready

def band_parameter(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,+limits: Machine.Limits,+capacity: U32,
  +bands: Model.Model<F32>,whole: BandGather.Domain<geometry,limits,capacity>,source_shape: BandShape.WellFormed<F32,geometry,Model.content(~F32,bands)>,
  values: BandValues.Values(~F32,~Logical,~original,logical,geometry,Model.content(~F32,bands)),coherent: {ShapeLaws.elements(shape) == BandGather.elements(geometry) : Nat}) ->
  Parameter<Bands.gather(~F32,geometry,Model.pack(~F32,bands),Tensors.buffer(~F32,fill,Bands.elements(geometry)))>:
  parameter(Bands.gather(~F32,geometry,Model.pack(~F32,bands),Tensors.buffer(~F32,fill,Bands.elements(geometry))),
    contract_ready(~Logical,~original,logical,Bands.gather(~F32,geometry,Model.pack(~F32,bands),Tensors.buffer(~F32,fill,Bands.elements(geometry))),shape,fill,workers,
      banded(~Logical,~original,logical,geometry,shape,fill,workers,limits,capacity,bands,whole,source_shape,values,coherent)))

def parameter_values(-buffer: Storage.Buffer<F32>,parameter: Parameter<buffer>) -> List<&2,F32>:
  Parameter{model,equation,ready,values,length_exact,evidence,tree_exact} = parameter
  values

def parameter_model(-buffer: Storage.Buffer<F32>,parameter: Parameter<buffer>) -> Buffer.Model<F32>:
  Parameter{model,equation,ready,values,length_exact,evidence,tree_exact} = parameter
  model

def parameter_evidence(-buffer: Storage.Buffer<F32>,+parameter: Parameter<buffer>) ->
  StorageLaws.SequenceEvidence<F32,ConvSpec.length(parameter_values(buffer,parameter)),parameter_values(buffer,parameter)>:
  Parameter{model,equation,ready,values,length_exact,evidence,tree_exact} = parameter
  evidence

def parameter_equation(-buffer: Storage.Buffer<F32>,+parameter: Parameter<buffer>) ->
  {Buffer.pack(~F32,parameter_model(buffer,parameter)) == buffer : Storage.Buffer<F32>}:
  Parameter{model,equation,ready,values,length_exact,evidence,tree_exact} = parameter
  equation

def model_contents(+model: Buffer.Model<F32>) ->
  {Buffer.content(~F32,model) == Buffer.Content{Buffer.tree(~F32,model),Buffer.length(~F32,model)} : Buffer.Content<F32>}:
  Buffer.Model{tree,length,certificate} = model
  {==}

def parameter_content(-buffer: Storage.Buffer<F32>,+parameter: Parameter<buffer>) ->
  {Buffer.content(~F32,parameter_model(buffer,parameter)) ==
    Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(parameter_values(buffer,parameter)),parameter_values(buffer,parameter),parameter_evidence(buffer,parameter)),ConvSpec.length(parameter_values(buffer,parameter))} : Buffer.Content<F32>}:
  Parameter{+model,equation,ready,+values,+length_exact,+evidence,+tree_exact} = parameter
  Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,model),Buffer.Content{Buffer.tree(~F32,model),Buffer.length(~F32,model)},
    Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence),ConvSpec.length(values)},model_contents(model),
    %Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,model),StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence),tree_exact) :
      {Buffer.Content{_,Buffer.length(~F32,model)} == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence),ConvSpec.length(values)} : Buffer.Content<F32>}
    %length_exact : {Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence),_} == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence),ConvSpec.length(values)} : Buffer.Content<F32>}
    {==})

def prefix_sequence(+model: Buffer.Model<F32>,+values: List<&2,F32>,+evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(values),values>,
  length_exact: {ConvSpec.length(values) == Buffer.length(~F32,model) : Nat},
  tree_exact: {Buffer.tree(~F32,model) == StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence) : Tree.Tree<F32>}) ->
  {sequence(Buffer.content(~F32,model)) == values : List<&2,F32>}:
  StorageLaws.SequenceEvidence{storage,contents,room} = evidence
  %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,model),Buffer.Content{Buffer.tree(~F32,model),Buffer.length(~F32,model)},model_contents(model)) : {sequence(_) == values : List<&2,F32>}
  %length_exact : {Tree.read_sequence(~F32,_,0,Buffer.tree(~F32,model)) == values : List<&2,F32>}
  %Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,model),storage,tree_exact) : {Tree.read_sequence(~F32,ConvSpec.length(values),0,_) == values : List<&2,F32>}
  contents

def parameter_contents(-buffer: Storage.Buffer<F32>,+parameter: Parameter<buffer>) ->
  {Buffer.content(~F32,parameter_model(buffer,parameter)) == Buffer.contents(~F32,buffer) : Buffer.Content<F32>}:
  Parameter{+model,+equation,ready,values,length_exact,evidence,tree_exact} = parameter
  Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,model),Buffer.contents(~F32,Buffer.pack(~F32,model)),Buffer.contents(~F32,buffer),
    Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,model)),Buffer.content(~F32,model),Buffer.packed_contents(~F32,model)),
    Equal.cong(Storage.Buffer<F32>,Buffer.Content<F32>,buffer => Buffer.contents(~F32,buffer),Buffer.pack(~F32,model),buffer,equation))

# The incoming logical parameter list is recovered from the finite values
# produced by actual gather, rather than supplied as a readback assumption.
def original_values(-buffer: Storage.Buffer<F32>,+parameter: Parameter<buffer>,+values: List<&2,F32>,+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},
  length_exact: {Buffer.content_length(~F32,Buffer.contents(~F32,buffer)) == ConvSpec.length(values) : Nat},within: {Nat.is_le(ConvSpec.length(values),U32.to_nat(upper)) == True{} : Bool},
  contents: Dense.Values(~F32,~(List<&2,F32>),~sequence_value,values,Buffer.contents(~F32,buffer))) -> {parameter_values(buffer,parameter) == values : List<&2,F32>}:
  Parameter{+model,equation,ready,+observed_values,observed_length,+evidence,tree_exact} = parameter
  +content_exact = parameter_contents(buffer,parameter)
  Equal.trans(List<&2,F32>,observed_values,sequence(Buffer.content(~F32,model)),values,
    Equal.sym(List<&2,F32>,sequence(Buffer.content(~F32,model)),observed_values,prefix_sequence(model,observed_values,evidence,observed_length,tree_exact)),
    sequence_equals(Buffer.content(~F32,model),values,upper,upper_ok,
      %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,model),Buffer.contents(~F32,buffer),content_exact) : {Buffer.content_length(~F32,_) == ConvSpec.length(values) : Nat} length_exact,
      within,
      %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,model),Buffer.contents(~F32,buffer),content_exact) : Dense.Values(~F32,~(List<&2,F32>),~sequence_value,values,_) contents))
