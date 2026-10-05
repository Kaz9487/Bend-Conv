# Output band metadata: plan mapping, allocation bounds, window guards and
# readiness.
import Base
import ./band_pipeline_execution.bend as PipelineExecution
import ../../lib/proofs/tensor_band_window.bend as TensorBandWindow
import ./band_leaf_values.bend as LeafValues
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/tensor_band_window.bend as BandWindow
import ../specification.bend as ConvSpec
import ../../lib/proofs/tensor_band_plan.bend as BandPlan
import ../../lib/tensor_band.bend as Bands
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ./band_caller_guards.bend as Caller
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/kernel_band_buffer_shape.bend as KernelBandBufferShape
import ../../lib/proofs/tensor_band_inspection.bend as Inspection
import ./concrete_support_guard.bend as Support
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./supported_padded_coordinates.bend as Padded
import ../../lib/kernel_band_pipeline.bend as BandPipeline
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/kernel_output_count_bounds.bend as Counts
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_band_allocated_metadata.bend as Allocated

# Actual shape/support guards supply the window mapping's machine envelope.
# The source Domain and accepted batch output bound are retained in the frame.
def window_guards_height(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> U32:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},window_guards_images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} = frame
  U32.from_nat(h)

def output_height(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> U32:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},window_guards_images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} = frame
  U32.from_nat(ConvSpec.axis_size(h,k,s,p))

def window_guards_stride(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> U32:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},window_guards_images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} = frame
  U32.from_nat(s)

def window_guards_images(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> U32:
  PipelineExecution.Frame{logical,activation,dimensions,window_guards_images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} = frame
  window_guards_images

type Envelope<-Logical: Data,-frame: PipelineExecution.Frame<Logical>> is Data:
  Envelope{height_positive: {Nat.is_lt(0n,U32.to_nat(window_guards_height(~Logical,frame))) == True{} : Bool},
    stride_positive: {Nat.is_lt(0n,U32.to_nat(window_guards_stride(~Logical,frame))) == True{} : Bool},
    height_room: {Nat.is_le(U32.to_nat(window_guards_height(~Logical,frame)),U32.to_nat(Machine.capacity(PipelineExecution.convolution_limits(~Logical,frame)))) == True{} : Bool},
    output_height_room: {Nat.is_le(U32.to_nat(output_height(~Logical,frame)),U32.to_nat(Machine.capacity(PipelineExecution.convolution_limits(~Logical,frame)))) == True{} : Bool},
    height_divisor_room: {Nat.is_le(Nat.double(U32.to_nat(window_guards_height(~Logical,frame))),U32.to_nat(Machine.half_word(PipelineExecution.convolution_limits(~Logical,frame)))) == True{} : Bool},
    stride_divisor_room: {Nat.is_le(Nat.double(U32.to_nat(window_guards_stride(~Logical,frame))),U32.to_nat(Machine.half_word(PipelineExecution.convolution_limits(~Logical,frame)))) == True{} : Bool},
    output_rows_room: {Nat.is_le(Nat.mul(U32.to_nat(window_guards_images(~Logical,frame)),U32.to_nat(output_height(~Logical,frame))),U32.to_nat(Machine.capacity(PipelineExecution.convolution_limits(~Logical,frame)))) == True{} : Bool},
    output_height_positive: {Nat.is_lt(0n,U32.to_nat(output_height(~Logical,frame))) == True{} : Bool}}

def rows_room(+window_guards_images: Nat,+channels: Nat,+window_guards_height: Nat,+width: Nat,+capacity: Nat,
  channels_positive: {Nat.is_lt(0n,channels) == True{} : Bool},width_positive: {Nat.is_lt(0n,width) == True{} : Bool},
  room: {Nat.is_le(Nat.mul(window_guards_images,Nat.mul(channels,Nat.mul(window_guards_height,width))),capacity) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(window_guards_images,window_guards_height),capacity) == True{} : Bool}:
  +positions = Nat.mul(window_guards_height,width)
  +density = Nat.mul(channels,positions)
  +inside = Order.transitive(window_guards_height,positions,density,RowMajor.product_covers(window_guards_height,width,width_positive),
    %RowMajor.multiply_commutative(positions,channels) : {Nat.is_le(positions,_) == True{} : Bool}
    RowMajor.product_covers(positions,channels,channels_positive))
  Order.transitive(Nat.mul(window_guards_images,window_guards_height),Nat.mul(window_guards_images,density),capacity,
    %RowMajor.multiply_commutative(window_guards_height,window_guards_images) : {Nat.is_le(_,Nat.mul(window_guards_images,density)) == True{} : Bool}
    %RowMajor.multiply_commutative(density,window_guards_images) : {Nat.is_le(Nat.mul(window_guards_height,window_guards_images),_) == True{} : Bool}
    RowMajor.mul_le(window_guards_height,density,window_guards_images,inside),room)

def Words(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> Data:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},window_guards_images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} = frame
  LeafValues.Words<ci,co,h,w,k,s,p>

def stride_word(+ci: Nat,+co: Nat,+h: Nat,+w: Nat,+k: Nat,+s: Nat,+p: Nat,words: LeafValues.Words<ci,co,h,w,k,s,p>) -> {U32.to_nat(U32.from_nat(s)) == s : Nat}:
  LeafValues.Words{ci,co,h,w,k,s,p,oh,ow} = words
  s

def output_height_word(+ci: Nat,+co: Nat,+h: Nat,+w: Nat,+k: Nat,+s: Nat,+p: Nat,words: LeafValues.Words<ci,co,h,w,k,s,p>) ->
  {U32.to_nat(U32.from_nat(ConvSpec.axis_size(h,k,s,p))) == ConvSpec.axis_size(h,k,s,p) : Nat}:
  LeafValues.Words{ci,co,h,w,k,s,p,oh,ow} = words
  oh

def stride_value(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> Nat:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},window_guards_images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} = frame
  s

def output_channels(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> Nat:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},window_guards_images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} = frame
  co

def window_guards_from_words(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+words: Words(~Logical,frame),
  +height_positive: {Nat.is_lt(0n,U32.to_nat(window_guards_height(~Logical,frame))) == True{} : Bool},
  +height_room: {Nat.is_le(U32.to_nat(window_guards_height(~Logical,frame)),U32.to_nat(Machine.capacity(PipelineExecution.convolution_limits(~Logical,frame)))) == True{} : Bool},
  +stride_positive: {Nat.is_lt(0n,stride_value(~Logical,frame)) == True{} : Bool},
  +stride_room: {Nat.is_le(stride_value(~Logical,frame),U32.to_nat(Machine.capacity(PipelineExecution.convolution_limits(~Logical,frame)))) == True{} : Bool},
  +outputs_positive: {Nat.is_lt(0n,output_channels(~Logical,frame)) == True{} : Bool}) -> Envelope<Logical,frame>:
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+window_guards_images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  +frame = {PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},window_guards_images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  +s_exact = stride_word(ci,co,h,w,k,s,p,words)
  +oh_exact = output_height_word(ci,co,h,w,k,s,p,words)
  +oh = ConvSpec.axis_size(h,k,s,p)
  +ow = ConvSpec.axis_size(w,k,s,p)
  +oh_room = LeafValues.output_height_room(limits,window_guards_images,co,h,w,k,s,p,Caller.images_positive(window_guards_images,True{},ci,co,h,w,k,s,p,eligible),outputs_positive,room)
  +native_stride_room = {
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(s)),s,s_exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    stride_room : {Nat.is_le(U32.to_nat(U32.from_nat(s)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  Envelope{height_positive,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(s)),s,s_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    stride_positive,
    height_room,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(oh)),oh,oh_exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    oh_room,
    Limits.double_bound(limits,U32.to_nat(U32.from_nat(h)),height_room),Limits.double_bound(limits,U32.to_nat(U32.from_nat(s)),native_stride_room),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(oh)),oh,oh_exact) : {Nat.is_le(Nat.mul(U32.to_nat(window_guards_images),_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    rows_room(U32.to_nat(window_guards_images),co,oh,ow,U32.to_nat(Machine.capacity(limits)),outputs_positive,{==},room),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(oh)),oh,oh_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    {==}}

def window_guards_envelope(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>) -> Envelope<Logical,frame>:
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+window_guards_images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  +dimensions = {KernelShape.Dimensions{ci,co,h,w,k,s,p} : KernelShape.Dimensions}
  +frame = {PipelineExecution.Frame{logical,activation,dimensions,window_guards_images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  +problem = Caller.problem(~Logical,~original,logical,dimensions,sw,sb,0n)
  +problem_valid = Caller.valid(~Logical,~original,logical,dimensions,sw,sb,0n,valid)
  +problem_supported = Caller.supported(~Logical,~original,logical,limits,dimensions,sw,sb,0n,supported)
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,problem_supported))
  window_guards_from_words(~Logical,frame,Caller.words(limits,problem,problem_valid,problem_supported),
    TensorBandWindow.height_positive(PipelineExecution.input_geometry(~Logical,frame),limits,capacity,domain),PipelineExecution.input_height_room(~Logical,frame),Caller.stride_positive(problem,problem_valid),
    Order.transitive(s,U32.to_nat(Machine.dimension(limits)),U32.to_nat(Machine.capacity(limits)),SupportedCoordinateBounds.stride_bound(limits,problem,support),Bounds.as_nat_le(Machine.dimension(limits),Machine.capacity(limits),Limits.dimension_within_capacity(limits))),
    Caller.outputs_positive(problem,problem_valid))

# The real native window rows refine the monotone global output row map.
# Source metadata and the accepted Frame guards discharge the word premises.
def mapping(~Logical: Data,+frame: PipelineExecution.Frame<Logical>) -> BandPlan.Mapping:
  BandPlan.Mapping{U32.to_nat(window_guards_height(~Logical,frame)),U32.to_nat(output_height(~Logical,frame)),U32.to_nat(window_guards_stride(~Logical,frame))}

def plan_mapping_begin(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,first: U32) -> Nat:
  BandPlan.mapping(mapping(~Logical,frame),U32.to_nat(first))

def plan_mapping_end(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,first: U32,rows: U32) -> Nat:
  BandPlan.mapping(mapping(~Logical,frame),Nat.add(U32.to_nat(first),U32.to_nat(rows)))

def native_first(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+first: U32) -> U32:
  +height = window_guards_height(~Logical,frame)
  +oh = output_height(~Logical,frame)
  (first / height * oh + TensorBandWindow.output_first(height,oh,window_guards_stride(~Logical,frame),first) : U32)

def native_rows(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+first: U32,rows: U32) -> U32:
  +height = window_guards_height(~Logical,frame)
  +oh = output_height(~Logical,frame)
  +stride = window_guards_stride(~Logical,frame)
  BandWindow.subtract_clipped(TensorBandWindow.output_end(height,oh,stride,first,rows),TensorBandWindow.output_first(height,oh,stride,first))

type RowMapping<-Logical: Data,-frame: PipelineExecution.Frame<Logical>,-first: U32,-rows: U32> is Data:
  RowMapping{first_exact: {U32.to_nat(native_first(~Logical,frame,first)) == plan_mapping_begin(~Logical,frame,first) : Nat},
    rows_exact: {U32.to_nat(native_rows(~Logical,frame,first,rows)) == Nat.sub(plan_mapping_end(~Logical,frame,first,rows),plan_mapping_begin(~Logical,frame,first)) : Nat}}

def from_envelope(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,+plan_mapping_coordinates: TensorBandWindow.CoreEnd<window_guards_height(~Logical,frame),first,rows>,
  first_inside: {Nat.is_le(U32.to_nat(first),Nat.mul(U32.to_nat(window_guards_images(~Logical,frame)),U32.to_nat(window_guards_height(~Logical,frame)))) == True{} : Bool},
  envelope: Envelope<Logical,frame>) -> RowMapping<Logical,frame,first,rows>:
  Envelope{+hp,+sp,+hr,+ohr,+hd,+sd,+output_room,ohp} = envelope
  +height = window_guards_height(~Logical,frame)
  +oh = output_height(~Logical,frame)
  +stride = window_guards_stride(~Logical,frame)
  +limits = PipelineExecution.convolution_limits(~Logical,frame)
  +capacity = Machine.capacity(limits)
  +division_capacity = Machine.half_word(limits)
  RowMapping{TensorBandWindow.first_global(height,oh,stride,first,capacity,division_capacity,Limits.capacity_fits(limits),Limits.half_word_fits(limits),hp,sp,hd,sd,hr,
      BandPlan.output_endpoint_room(U32.to_nat(window_guards_images(~Logical,frame)),U32.to_nat(height),U32.to_nat(oh),U32.to_nat(stride),U32.to_nat(first),capacity,hp,sp,output_room,first_inside)),
    TensorBandWindow.rows(height,oh,stride,first,rows,capacity,division_capacity,Limits.capacity_fits(limits),Limits.half_word_fits(limits),hp,sp,hd,sd,hr,ohr,plan_mapping_coordinates)}

def plan_mapping_coordinates(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,
  local: {U32.is_le(rows,(window_guards_height(~Logical,frame) - first % window_guards_height(~Logical,frame) : U32)) == True{} : Bool},
  envelope: Envelope<Logical,frame>) -> TensorBandWindow.CoreEnd<window_guards_height(~Logical,frame),first,rows>:
  Envelope{hp,sp,hr,ohr,hd,sd,room,ohp} = envelope
  +limits = PipelineExecution.convolution_limits(~Logical,frame)
  TensorBandWindow.core_end(window_guards_height(~Logical,frame),first,rows,Machine.capacity(limits),Machine.half_word(limits),Limits.capacity_fits(limits),Limits.half_word_fits(limits),hp,hd,hr,local)

def plan_mapping_leaf(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,+length: Nat,
  metadata: BandShape.Leaf(PipelineExecution.input_geometry(~Logical,frame),first,rows,length)) -> RowMapping<Logical,frame,first,rows>:
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  BandShape.Fields{+local,+rows_fit,length_exact} = metadata
  +frame = {PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  +envelope = window_guards_envelope(~Logical,~original,frame)
  from_envelope(~Logical,frame,first,rows,plan_mapping_coordinates(~Logical,frame,first,rows,local,envelope),
    Order.transitive(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(U32.from_nat(h))),Order.le_add(U32.to_nat(first),U32.to_nat(rows)),rows_fit),envelope)

def native_plan(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,plan: Bands.Plan) -> Bands.Plan:
  match plan:
    case Bands.Rows{+first,rows}: Bands.Rows{native_first(~Logical,frame,first),native_rows(~Logical,frame,first,rows)}
    case Bands.Split{left,right}: Bands.Split{native_plan(~Logical,frame,left),native_plan(~Logical,frame,right)}

def row_plan(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,evidence: RowMapping<Logical,frame,first,rows>) ->
  {native_plan(~Logical,frame,Bands.Rows{first,rows}) == BandPlan.mapped(~BandPlan.Mapping,~BandPlan.mapping,mapping(~Logical,frame),Bands.Rows{first,rows}) : Bands.Plan}:
  RowMapping{first_exact,rows_exact} = evidence
  %Equal.sym(U32,native_first(~Logical,frame,first),U32.from_nat(plan_mapping_begin(~Logical,frame,first)),Encoded.encoding(native_first(~Logical,frame,first),plan_mapping_begin(~Logical,frame,first),first_exact)) :
    {Bands.Rows{_,native_rows(~Logical,frame,first,rows)} == Bands.Rows{U32.from_nat(plan_mapping_begin(~Logical,frame,first)),U32.from_nat(Nat.sub(plan_mapping_end(~Logical,frame,first,rows),plan_mapping_begin(~Logical,frame,first)))} : Bands.Plan}
  %Equal.sym(U32,native_rows(~Logical,frame,first,rows),U32.from_nat(Nat.sub(plan_mapping_end(~Logical,frame,first,rows),plan_mapping_begin(~Logical,frame,first))),Encoded.encoding(native_rows(~Logical,frame,first,rows),Nat.sub(plan_mapping_end(~Logical,frame,first,rows),plan_mapping_begin(~Logical,frame,first)),rows_exact)) :
    {Bands.Rows{U32.from_nat(plan_mapping_begin(~Logical,frame,first)),_} == Bands.Rows{U32.from_nat(plan_mapping_begin(~Logical,frame,first)),U32.from_nat(Nat.sub(plan_mapping_end(~Logical,frame,first,rows),plan_mapping_begin(~Logical,frame,first)))} : Bands.Plan}
  {==}

def source_plan(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,source: Model.Model<F32>,
  metadata: BandShape.Contents(~F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source))) ->
  {native_plan(~Logical,frame,Model.layout(~F32,source)) == BandPlan.mapped(~BandPlan.Mapping,~BandPlan.mapping,mapping(~Logical,frame),Model.layout(~F32,source)) : Bands.Plan}:
  match source:
    case Model.Leaf{Buffer.Model{tree,length,certificate},+first,+rows}: row_plan(~Logical,frame,first,rows,plan_mapping_leaf(~Logical,~original,frame,first,rows,length,metadata))
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      %source_plan(~Logical,~original,frame,left,left_metadata) :
        {Bands.Split{native_plan(~Logical,frame,Model.layout(~F32,left)),native_plan(~Logical,frame,Model.layout(~F32,right))} == Bands.Split{_,BandPlan.mapped(~BandPlan.Mapping,~BandPlan.mapping,mapping(~Logical,frame),Model.layout(~F32,right))} : Bands.Plan}
      %source_plan(~Logical,~original,frame,right,right_metadata) :
        {Bands.Split{native_plan(~Logical,frame,Model.layout(~F32,left)),native_plan(~Logical,frame,Model.layout(~F32,right))} == Bands.Split{native_plan(~Logical,frame,Model.layout(~F32,left)),_} : Bands.Plan}
      {==}

def covered(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+source: Model.Model<F32>,
  metadata: BandShape.Contents(~F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)),
  coverage: BandPlan.Coverage(Model.layout(~F32,source),0n,Nat.mul(U32.to_nat(window_guards_images(~Logical,frame)),U32.to_nat(window_guards_height(~Logical,frame)))),
  envelope: Envelope<Logical,frame>) ->
  BandPlan.Coverage(native_plan(~Logical,frame,Model.layout(~F32,source)),0n,Nat.mul(U32.to_nat(window_guards_images(~Logical,frame)),U32.to_nat(output_height(~Logical,frame)))):
  Envelope{hp,sp,hr,ohr,hd,sd,output_room,ohp} = envelope
  %Equal.sym(Bands.Plan,native_plan(~Logical,frame,Model.layout(~F32,source)),BandPlan.mapped(~BandPlan.Mapping,~BandPlan.mapping,mapping(~Logical,frame),Model.layout(~F32,source)),source_plan(~Logical,~original,frame,source,metadata)) :
    BandPlan.Coverage(_,0n,Nat.mul(U32.to_nat(window_guards_images(~Logical,frame)),U32.to_nat(output_height(~Logical,frame))))
  BandPlan.output_coverage(U32.to_nat(window_guards_images(~Logical,frame)),U32.to_nat(window_guards_height(~Logical,frame)),U32.to_nat(output_height(~Logical,frame)),U32.to_nat(window_guards_stride(~Logical,frame)),
    Model.layout(~F32,source),Machine.capacity(PipelineExecution.convolution_limits(~Logical,frame)),Limits.capacity_fits(PipelineExecution.convolution_limits(~Logical,frame)),hp,sp,output_room,coverage)

def coverage(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+source: Model.Model<F32>,
  evidence: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>) ->
  BandPlan.Coverage(native_plan(~Logical,frame,Model.layout(~F32,source)),0n,Nat.mul(U32.to_nat(window_guards_images(~Logical,frame)),U32.to_nat(output_height(~Logical,frame)))):
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  BandShape.WellFormed{metadata,ready,coverage} = evidence
  +frame = {PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  covered(~Logical,~original,frame,source,metadata,
    %Equal.sym(Bands.Plan,Model.layout(~F32,source),Model.content_layout(~F32,Model.content(~F32,source)),Model.layout_matches_content(~F32,source)) : BandPlan.Coverage(_,0n,Inspection.total_rows(PipelineExecution.input_geometry(~Logical,frame)))
    coverage,window_guards_envelope(~Logical,~original,frame))

# Actual mapped row descriptors retain image locality and batch containment.
# Allocation length is supplied separately by the convolution leaf owner.
def bounds_geometry(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> Bands.Geometry:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,+k,+s,+p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} = frame
  Bands.Geometry{images,U32.from_nat(co),U32.from_nat(ConvSpec.axis_size(h,k,s,p)),U32.from_nat(ConvSpec.axis_size(w,k,s,p))}

def local(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,
  mapped: RowMapping<Logical,frame,first,rows>,
  endpoints: TensorBandWindow.LocalMapping<window_guards_height(~Logical,frame),output_height(~Logical,frame),window_guards_stride(~Logical,frame),first,rows>,
  envelope: Envelope<Logical,frame>) ->
  {U32.is_le(native_rows(~Logical,frame,first,rows),(output_height(~Logical,frame) - native_first(~Logical,frame,first) % output_height(~Logical,frame) : U32)) == True{} : Bool}:
  RowMapping{+first_exact,rows_exact} = mapped
  TensorBandWindow.LocalMapping{+begin_exact,end_exact,+ordered} = endpoints
  Envelope{hp,sp,hr,+ohr,hd,sd,room,+ohp} = envelope
  +height = window_guards_height(~Logical,frame)
  +oh = output_height(~Logical,frame)
  +stride = window_guards_stride(~Logical,frame)
  +begin = TensorBandWindow.output_first(height,oh,stride,first)
  +finish = TensorBandWindow.output_end(height,oh,stride,first,rows)
  +core_end = (first % height + rows : U32)
  +limits = PipelineExecution.convolution_limits(~Logical,frame)
  +capacity = Machine.capacity(limits)
  +begin_inside = TensorBandWindow.below(oh,(first % height : U32),stride)
  +end_inside = TensorBandWindow.end_below(oh,core_end,stride,U32.is_eq(core_end,height))
  Inspection.local_guard(oh,native_first(~Logical,frame,first),native_rows(~Logical,frame,first,rows),Machine.half_word(limits),Limits.half_word_fits(limits),ohp,Limits.double_bound(limits,U32.to_nat(oh),ohr),
    BandPlan.span(U32.to_nat(oh),Nat.div(U32.to_nat(first),U32.to_nat(height)),U32.to_nat(begin),U32.to_nat(finish),
      U32.to_nat(native_first(~Logical,frame,first)),U32.to_nat(native_rows(~Logical,frame,first,rows)),ohp,begin_inside,ordered,end_inside,
      Equal.trans(Nat,U32.to_nat(native_first(~Logical,frame,first)),plan_mapping_begin(~Logical,frame,first),
        Nat.add(Nat.mul(Nat.div(U32.to_nat(first),U32.to_nat(height)),U32.to_nat(oh)),U32.to_nat(begin)),first_exact,begin_exact),
      TensorBandWindow.subtract_clipped(finish,begin,capacity,Limits.capacity_fits(limits),
        Order.transitive(U32.to_nat(finish),U32.to_nat(oh),U32.to_nat(capacity),end_inside,ohr),
        Order.transitive(U32.to_nat(begin),U32.to_nat(oh),U32.to_nat(capacity),begin_inside,ohr))))

def rows_fit(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,
  mapped: RowMapping<Logical,frame,first,rows>,
  source_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(window_guards_images(~Logical,frame)),U32.to_nat(window_guards_height(~Logical,frame)))) == True{} : Bool},
  envelope: Envelope<Logical,frame>) ->
  {Nat.is_le(Nat.add(U32.to_nat(native_first(~Logical,frame,first)),U32.to_nat(native_rows(~Logical,frame,first,rows))),Nat.mul(U32.to_nat(window_guards_images(~Logical,frame)),U32.to_nat(output_height(~Logical,frame)))) == True{} : Bool}:
  RowMapping{+first_exact,+rows_exact} = mapped
  Envelope{+hp,+sp,hr,ohr,hd,sd,room,ohp} = envelope
  +height = U32.to_nat(window_guards_height(~Logical,frame))
  +oh = U32.to_nat(output_height(~Logical,frame))
  +stride = U32.to_nat(window_guards_stride(~Logical,frame))
  +images = U32.to_nat(window_guards_images(~Logical,frame))
  +begin = plan_mapping_begin(~Logical,frame,first)
  +end = plan_mapping_end(~Logical,frame,first,rows)
  +sum_exact = Equal.trans(Nat,
    Nat.add(U32.to_nat(native_first(~Logical,frame,first)),U32.to_nat(native_rows(~Logical,frame,first,rows))),
    Nat.add(begin,U32.to_nat(native_rows(~Logical,frame,first,rows))),end,
    Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(native_rows(~Logical,frame,first,rows))),U32.to_nat(native_first(~Logical,frame,first)),begin,first_exact),
    Equal.trans(Nat,Nat.add(begin,U32.to_nat(native_rows(~Logical,frame,first,rows))),Nat.add(begin,Nat.sub(end,begin)),end,
      Equal.cong(Nat,Nat,value => Nat.add(begin,value),U32.to_nat(native_rows(~Logical,frame,first,rows)),Nat.sub(end,begin),rows_exact),
      Order.restore_shift(begin,end,BandPlan.monotone(height,oh,stride,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(rows)),hp,sp,Order.le_add(U32.to_nat(first),U32.to_nat(rows))))))
  %Equal.sym(Nat,Nat.add(U32.to_nat(native_first(~Logical,frame,first)),U32.to_nat(native_rows(~Logical,frame,first,rows))),end,sum_exact) :
    {Nat.is_le(_,Nat.mul(images,oh)) == True{} : Bool}
  %BandPlan.image_end(images,height,oh,stride,hp,sp) : {Nat.is_le(end,_) == True{} : Bool}
  BandPlan.monotone(height,oh,stride,Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(images,height),hp,sp,source_fit)

def local_endpoints(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,
  coordinates: TensorBandWindow.CoreEnd<window_guards_height(~Logical,frame),first,rows>,envelope: Envelope<Logical,frame>) ->
  TensorBandWindow.LocalMapping<window_guards_height(~Logical,frame),output_height(~Logical,frame),window_guards_stride(~Logical,frame),first,rows>:
  Envelope{hp,sp,hr,ohr,hd,sd,room,ohp} = envelope
  +limits = PipelineExecution.convolution_limits(~Logical,frame)
  TensorBandWindow.local_mapping(window_guards_height(~Logical,frame),output_height(~Logical,frame),window_guards_stride(~Logical,frame),first,rows,
    Machine.capacity(limits),Machine.half_word(limits),Limits.capacity_fits(limits),Limits.half_word_fits(limits),hp,sp,hd,sd,hr,coordinates)

def geometry_elements(bounds_geometry: Bands.Geometry,rows: U32) -> Nat:
  Bands.Geometry{images,channels,height,width} = bounds_geometry
  Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))

def bounds_elements(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32) -> Nat:
  geometry_elements(bounds_geometry(~Logical,frame),native_rows(~Logical,frame,first,rows))

def fields(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,+length: Nat,
  +mapped: RowMapping<Logical,frame,first,rows>,
  endpoints: TensorBandWindow.LocalMapping<window_guards_height(~Logical,frame),output_height(~Logical,frame),window_guards_stride(~Logical,frame),first,rows>,
  source_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(window_guards_images(~Logical,frame)),U32.to_nat(window_guards_height(~Logical,frame)))) == True{} : Bool},
  +envelope: Envelope<Logical,frame>,length_exact: {length == bounds_elements(~Logical,frame,first,rows) : Nat}) ->
  BandShape.Leaf(bounds_geometry(~Logical,frame),native_first(~Logical,frame,first),native_rows(~Logical,frame,first,rows),length):
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  +frame = {PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  BandShape.Fields{local(~Logical,frame,first,rows,mapped,endpoints,envelope),rows_fit(~Logical,frame,first,rows,mapped,source_fit,envelope),length_exact}

def from_source(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,+source_length: Nat,+output_length: Nat,
  source: BandShape.Leaf(PipelineExecution.input_geometry(~Logical,frame),first,rows,source_length),
  length_exact: {output_length == bounds_elements(~Logical,frame,first,rows) : Nat}) ->
  BandShape.Leaf(bounds_geometry(~Logical,frame),native_first(~Logical,frame,first),native_rows(~Logical,frame,first,rows),output_length):
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  BandShape.Fields{+local,+fit,+source_exact} = source
  +frame = {PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  +source = {BandShape.Fields{local,fit,source_exact} : BandShape.Leaf(PipelineExecution.input_geometry(~Logical,frame),first,rows,source_length)}
  +envelope = window_guards_envelope(~Logical,~original,frame)
  fields(~Logical,frame,first,rows,output_length,plan_mapping_leaf(~Logical,~original,frame,first,rows,source_length,source),
    local_endpoints(~Logical,frame,first,rows,plan_mapping_coordinates(~Logical,frame,first,rows,local,envelope),envelope),fit,envelope,length_exact)

# The real per-band output allocation is bounded by the accepted batch shape.
# These caller facts use the existing convolution guard and leaf contract.
def allocation_bounds_count(~Logical: Data,frame: PipelineExecution.Frame<Logical>,first: U32,rows: U32) -> U32:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} = frame
  (U32.from_nat(co) * (TensorBandWindow.output_rows(BandWindow.window(LeafValues.geometry(images,KernelShape.Dimensions{ci,co,h,w,k,s,p}),first,rows)) * U32.from_nat(ConvSpec.axis_size(w,k,s,p)) : U32) : U32)

def allocation_bounds_from_words(+images: U32,+ci: Nat,+co: Nat,+h: Nat,+w: Nat,+k: Nat,+s: Nat,+p: Nat,+limits: Machine.Limits,+first: U32,+rows: U32,
  words: LeafValues.Words<ci,co,h,w,k,s,p>,
  co_positive: {Nat.is_lt(0n,co) == True{} : Bool},
  single_room: {Nat.is_le(Nat.mul(co,Nat.mul(ConvSpec.axis_size(h,k,s,p),ConvSpec.axis_size(w,k,s,p))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Counts.CountEvidence<U32.from_nat(co),TensorBandWindow.output_rows(BandWindow.window(LeafValues.geometry(images,KernelShape.Dimensions{ci,co,h,w,k,s,p}),first,rows)),U32.from_nat(ConvSpec.axis_size(w,k,s,p))>:
  LeafValues.Words{ci_word,+co_word,h_word,w_word,k_word,s_word,p_word,+oh_word,+ow_word} = words
  +dimensions = {KernelShape.Dimensions{ci,co,h,w,k,s,p} : KernelShape.Dimensions}
  +oh = ConvSpec.axis_size(h,k,s,p)
  +ow = ConvSpec.axis_size(w,k,s,p)
  +native_rows = TensorBandWindow.output_rows(BandWindow.window(LeafValues.geometry(images,dimensions),first,rows))
  +rows_room = {
    %oh_word : {Nat.is_le(U32.to_nat(native_rows),_) == True{} : Bool}
    TensorBandWindow.bounded(LeafValues.geometry(images,dimensions),first,rows) : {Nat.is_le(U32.to_nat(native_rows),oh) == True{} : Bool}}
  Counts.proved(U32.from_nat(co),native_rows,U32.from_nat(ow),Machine.capacity(limits),Limits.capacity_fits(limits),Limits.capacity_allocatable(limits),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(co)),co,co_word) : {Nat.is_lt(0n,_) == True{} : Bool}
    co_positive,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(co)),co,co_word) : {Nat.is_le(Nat.mul(_,Nat.mul(U32.to_nat(native_rows),U32.to_nat(U32.from_nat(ow)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(ow)),ow,ow_word) : {Nat.is_le(Nat.mul(co,Nat.mul(U32.to_nat(native_rows),_)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    Counts.row_product_room(co,U32.to_nat(native_rows),oh,ow,U32.to_nat(Machine.capacity(limits)),rows_room,single_room))

def bounded(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32) ->
  {U32.is_le(allocation_bounds_count(~Logical,frame,first,rows),1073741824) == True{} : Bool}:
  PipelineExecution.Frame{+logical,activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,parallel,+limits,capacity,+weights,+bias,we,be,domain,valid,supported,room,eligible} = frame
  +dimensions = {KernelShape.Dimensions{ci,co,h,w,k,s,p} : KernelShape.Dimensions}
  +problem = Caller.problem(~Logical,~original,logical,dimensions,weights,bias,0n)
  +problem_valid = Caller.valid(~Logical,~original,logical,dimensions,weights,bias,0n,valid)
  +problem_supported = Caller.supported(~Logical,~original,logical,limits,dimensions,weights,bias,0n,supported)
  Counts.allocatable_of(U32.from_nat(co),TensorBandWindow.output_rows(BandWindow.window(LeafValues.geometry(images,dimensions),first,rows)),U32.from_nat(ConvSpec.axis_size(w,k,s,p)),
    allocation_bounds_from_words(images,ci,co,h,w,k,s,p,limits,first,rows,Caller.words(limits,problem,problem_valid,problem_supported),Caller.outputs_positive(problem,problem_valid),
      LeafValues.right_factor_room(U32.to_nat(images),Nat.mul(co,Nat.mul(ConvSpec.axis_size(h,k,s,p),ConvSpec.axis_size(w,k,s,p))),Machine.capacity(limits),Caller.images_positive(images,True{},ci,co,h,w,k,s,p,eligible),room)))

def allocation_bounds_elements(~Logical: Data,frame: PipelineExecution.Frame<Logical>,first: U32,rows: U32) -> Nat:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room,eligible} = frame
  Nat.mul(U32.to_nat(U32.from_nat(co)),Nat.mul(U32.to_nat(TensorBandWindow.output_rows(BandWindow.window(LeafValues.geometry(images,KernelShape.Dimensions{ci,co,h,w,k,s,p}),first,rows))),U32.to_nat(U32.from_nat(ConvSpec.axis_size(w,k,s,p)))))

def exact(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32) ->
  {U32.to_nat(allocation_bounds_count(~Logical,frame,first,rows)) == allocation_bounds_elements(~Logical,frame,first,rows) : Nat}:
  PipelineExecution.Frame{+logical,activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,parallel,+limits,capacity,+weights,+bias,we,be,domain,valid,supported,room,eligible} = frame
  +dimensions = {KernelShape.Dimensions{ci,co,h,w,k,s,p} : KernelShape.Dimensions}
  +problem = Caller.problem(~Logical,~original,logical,dimensions,weights,bias,0n)
  +problem_valid = Caller.valid(~Logical,~original,logical,dimensions,weights,bias,0n,valid)
  +problem_supported = Caller.supported(~Logical,~original,logical,limits,dimensions,weights,bias,0n,supported)
  Counts.exact_of(U32.from_nat(co),TensorBandWindow.output_rows(BandWindow.window(LeafValues.geometry(images,dimensions),first,rows)),U32.from_nat(ConvSpec.axis_size(w,k,s,p)),
    allocation_bounds_from_words(images,ci,co,h,w,k,s,p,limits,first,rows,Caller.words(limits,problem,problem_valid,problem_supported),Caller.outputs_positive(problem,problem_valid),
      LeafValues.right_factor_room(U32.to_nat(images),Nat.mul(co,Nat.mul(ConvSpec.axis_size(h,k,s,p),ConvSpec.axis_size(w,k,s,p))),Machine.capacity(limits),Caller.images_positive(images,True{},ci,co,h,w,k,s,p,eligible),room)))

# The actual convolution leaf returns an allocator-ready output owner. Its
# size premise is discharged by the public shape/support and batch guards.
def bands(~custom: F32 -> F32,~Logical: Data,frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,
  core: Buffer.Model<F32>,above: Buffer.Model<F32>,below: Buffer.Model<F32>,weights: Buffer.Model<Tile.Tile8>,bias: Buffer.Model<F32>) -> Bands.Bands(~F32):
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},images,parallel,limits,capacity,source_weights,source_bias,we,be,domain,valid,supported,room,eligible} = frame
  +dimensions = {KernelShape.Dimensions{ci,co,h,w,k,s,p} : KernelShape.Dimensions}
  +geometry = LeafValues.geometry(images,dimensions)
  BandPipeline.windowed(~custom,activation,geometry,U32.from_nat(co),U32.from_nat(Nat.mul(ci,Nat.mul(k,k))),first,rows,BandWindow.window(geometry,first,rows),
    Buffer.pack(~F32,core),Buffer.pack(~F32,above),Buffer.pack(~F32,below),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))

def ready(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  Model.ContentReady(~F32,Model.contents(~F32,bands(~custom,~Logical,frame,first,rows,core,above,below,weights,bias))):
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,+parallel,+limits,+capacity,+source_weights,+source_bias,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  +dimensions = {KernelShape.Dimensions{ci,co,h,w,k,s,p} : KernelShape.Dimensions}
  +frame = {PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,source_weights,source_bias,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  +geometry = LeafValues.geometry(images,dimensions)
  KernelBandBufferShape.ready_leaf(allocation_bounds_count(~Logical,frame,first,rows),bands(~custom,~Logical,frame,first,rows,core,above,below,weights,bias),
    KernelBandBufferShape.windowed(~custom,activation,geometry,U32.from_nat(co),U32.from_nat(Nat.mul(ci,Nat.mul(k,k))),first,rows,BandWindow.window(geometry,first,rows),
      Buffer.pack(~F32,core),Buffer.pack(~F32,above),Buffer.pack(~F32,below),weights,bias),
    bounded(~Logical,~original,frame,first,rows))

# The output rows of the actual window survive its inactive-halo branch.
# This connects window metadata to the native plan whose coverage is proved.
def when(~Logical: Data,enabled: Bool,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32) ->
  {PipelineExecution.output_window(LeafValues.window_when(enabled,PipelineExecution.dimensions(~Logical,frame),first,rows)) ==
    Bands.Rows{native_first(~Logical,frame,first),native_rows(~Logical,frame,first,rows)} : Bands.Plan}:
  match enabled frame:
    case True{} PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible}: {==}
    case False{} PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible}: {==}

def window_layout_window(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32) ->
  {PipelineExecution.output_window(BandWindow.window(PipelineExecution.window_geometry(~Logical,frame),first,rows)) == Bands.Rows{native_first(~Logical,frame,first),native_rows(~Logical,frame,first,rows)} : Bands.Plan}:
  PipelineExecution.Frame{+logical,+activation,+dimensions,+images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  +frame = {PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  %Equal.sym(BandWindow.RowWindow,BandWindow.window(LeafValues.geometry(images,dimensions),first,rows),LeafValues.window_when(LeafValues.active(dimensions,first,rows),dimensions,first,rows),LeafValues.window_matches(images,dimensions,first,rows)) :
    {PipelineExecution.output_window(_) == Bands.Rows{native_first(~Logical,frame,first),native_rows(~Logical,frame,first,rows)} : Bands.Plan}
  when(~Logical,LeafValues.active(dimensions,first,rows),frame,first,rows)

# Output metadata follows actual allocation and actual window descriptors.
# The proof boundary is the existing convolution leaf, independently of GEMM.
def window_rows(+window: BandWindow.RowWindow) ->
  {TensorBandWindow.output_rows(window) == Allocated.rows(PipelineExecution.output_window(window)) : U32}:
  BandWindow.RowWindow{BandWindow.Rows{first,rows},above,below,read} = window
  {==}

def metadata_elements(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32) ->
  {allocation_bounds_elements(~Logical,frame,first,rows) == bounds_elements(~Logical,frame,first,rows) : Nat}:
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  +dimensions = {KernelShape.Dimensions{ci,co,h,w,k,s,p} : KernelShape.Dimensions}
  +frame = {PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  +window = BandWindow.window(LeafValues.geometry(images,dimensions),first,rows)
  Equal.cong(U32,Nat,row => Nat.mul(U32.to_nat(U32.from_nat(co)),Nat.mul(U32.to_nat(row),U32.to_nat(U32.from_nat(ConvSpec.axis_size(w,k,s,p))))),
    TensorBandWindow.output_rows(window),native_rows(~Logical,frame,first,rows),
    Equal.trans(U32,TensorBandWindow.output_rows(window),Allocated.rows(PipelineExecution.output_window(window)),native_rows(~Logical,frame,first,rows),window_rows(window),
      Equal.cong(Bands.Plan,U32,Allocated.rows,PipelineExecution.output_window(window),Bands.Rows{native_first(~Logical,frame,first),native_rows(~Logical,frame,first,rows)},window_layout_window(~Logical,frame,first,rows))))

def metadata_leaf(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+first: U32,+rows: U32,
  +core: Buffer.Model<F32>,+above: Buffer.Model<F32>,+below: Buffer.Model<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,+source_length: Nat,
  source: BandShape.Leaf(PipelineExecution.input_geometry(~Logical,frame),first,rows,source_length)) ->
  BandShape.Contents(~F32,bounds_geometry(~Logical,frame),Model.contents(~F32,bands(~custom,~Logical,frame,first,rows,core,above,below,weights,bias))):
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,+parallel,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room,+eligible} = frame
  +dimensions = {KernelShape.Dimensions{ci,co,h,w,k,s,p} : KernelShape.Dimensions}
  +frame = {PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room,eligible} : PipelineExecution.Frame<Logical>}
  +geometry = LeafValues.geometry(images,dimensions)
  +window = BandWindow.window(geometry,first,rows)
  +count = allocation_bounds_count(~Logical,frame,first,rows)
  Allocated.leaf(bounds_geometry(~Logical,frame),native_first(~Logical,frame,first),native_rows(~Logical,frame,first,rows),count,
    bands(~custom,~Logical,frame,first,rows,core,above,below,weights,bias),
    KernelBandBufferShape.windowed(~custom,activation,geometry,U32.from_nat(co),U32.from_nat(Nat.mul(ci,Nat.mul(k,k))),first,rows,window,
      Buffer.pack(~F32,core),Buffer.pack(~F32,above),Buffer.pack(~F32,below),weights,bias),
    Equal.trans(Bands.Plan,PipelineExecution.layout(bands(~custom,~Logical,frame,first,rows,core,above,below,weights,bias)),PipelineExecution.output_window(window),
      Bands.Rows{native_first(~Logical,frame,first),native_rows(~Logical,frame,first,rows)},
      PipelineExecution.windowed_layout(~custom,activation,geometry,U32.from_nat(co),U32.from_nat(Nat.mul(ci,Nat.mul(k,k))),first,rows,window,
        Buffer.pack(~F32,core),Buffer.pack(~F32,above),Buffer.pack(~F32,below),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)),window_layout_window(~Logical,frame,first,rows)),
    from_source(~Logical,~original,frame,first,rows,source_length,U32.to_nat(count),source,
      Equal.trans(Nat,U32.to_nat(count),allocation_bounds_elements(~Logical,frame,first,rows),bounds_elements(~Logical,frame,first,rows),exact(~Logical,~original,frame,first,rows),metadata_elements(~Logical,frame,first,rows))))
