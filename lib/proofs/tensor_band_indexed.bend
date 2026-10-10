# Indexed band maps: the leaf uses the global NCHW index of each element.
import Base
import ./tensor_band_gather.bend as BandGather
import ../machine_limits.bend as Machine
import ./nat_order.bend as Order
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_band_count.bend as BandCount
import ./tensor_band_model.bend as Model
import ./tensor_indexed_refinement.bend as IndexedTensorLaws
import ./write_program.bend as Writes
import ../tensor_band.bend as Bands
import ../tensor_band_schedule.bend as BandSchedule
import ../tensor_indexed.bend as Indexed
import ../default_machine_limits.bend as Default
import ./encoded_coordinates.bend as Encoded
import ./machine_limit_bounds.bend as Limits
import ./nat_division.bend as Division
import ./nat_to_u32_bounds.bend as Bounds
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_certificate.bend as Certificate
import ./storage_copy_refinement.bend as StorageCopyLaws
import ./storage_shape.bend as Shape
import ./tensor_allocation_depth.bend as Depth
import ./tensor_band_each.bend as Each
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_plan.bend as BandPlan
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ./tensor_indexed_certificate.bend as IndexedCertificate
import ./traversal_certificate.bend as TraversalCertificate
import ./traversal_indexed_writes.bend as TraversalIndexedWrites
import ./u32_division.bend as U32Division
import ./word_subtraction.bend as Subtraction
import ../storage_allocation.bend as Allocation
import ../storage_buffer.bend as Storage
import ../tensor_operations.bend as Operations
import ../traversal_partition.bend as Partition

# The actual band indexed leaf uses the global NCHW index and the shared
# ordered writer. Allocation and certificates are erased only in observation.
def observed_pair(~Element: Data,+input_length: Nat,+output_length: Nat,result: Array<Element> & Array<Element>,
  certificates: TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),result>) ->
  {Buffer.contents(~Element,Bands.destination(~Element,Operations.pair_buffers(~Element,input_length,output_length,result,certificates))) ==
    BandGather.output(~Element,StorageCopyLaws.values(~Element,input_length,output_length,result)) : Buffer.Content<Element>}:
  (input,output) = result
  TraversalCertificate.Certificates{input_certificate,output_certificate} = certificates
  {==}

def buffer_model(~Element: Data,~operation: U32 -> Element -> Element,call: Indexed.Call<Bands.Indexed>,input: Buffer.Content<Element>,output: Buffer.Content<Element>) -> Buffer.Content<Element>:
  Indexed.Call{count,context} = call
  Buffer.Content{input,input_length} = input
  Buffer.Content{output,output_length} = output
  Buffer.Content{Writes.pure(Element,IndexedTensorLaws.writes(~Element,~Bands.Indexed,~Bands.index_of,~operation,count,context,input),output),output_length}

# A position of channel block at offset below the plane size decodes to that
# channel and offset, when the count is within the machine capacity.
type Decoded<-plane: U32,-block: Nat,-offset: Nat> is Data:
  Decoded{quotient: {(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset)) / plane : U32) == U32.from_nat(block) : U32},
    remainder: {(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset)) % plane : U32) == U32.from_nat(offset) : U32},
    local: {(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset)) - U32.from_nat(Nat.mul(block,U32.to_nat(plane))) : U32) == U32.from_nat(offset) : U32}}

def local_offset(+first: Nat,+offset: Nat) -> {(U32.from_nat(Nat.add(first,offset)) - U32.from_nat(first) : U32) == U32.from_nat(offset) : U32}:
  %Order.commutative(offset,first) : {(U32.from_nat(_) - U32.from_nat(first) : U32) == U32.from_nat(offset) : U32}
  Subtraction.encoded_subtract_add(offset,first)

def plane_within(+first: Nat,+plane: Nat,+limit: Nat,room: {Nat.is_le(Nat.add(first,plane),limit) == True{} : Bool}) -> {Nat.is_le(plane,limit) == True{} : Bool}:
  Order.transitive(plane,Nat.add(first,plane),limit,
    %Order.commutative(plane,first) : {Nat.is_le(plane,_) == True{} : Bool} Order.le_add(plane,first),room)

def decoded(+limits: Machine.Limits,+total: U32,+plane: U32,+block: Nat,+offset: Nat,
  +inside: {Nat.is_lt(offset,U32.to_nat(plane)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(plane)),U32.to_nat(plane)),U32.to_nat(total)) == True{} : Bool},
  bounded: {U32.is_le(total,Machine.capacity(limits)) == True{} : Bool}) -> Decoded<plane,block,offset>:
  +positive = Order.le_lt_trans(0n,offset,U32.to_nat(plane),Order.zero_le(offset),inside)
  +within = Bounds.as_nat_le(total,Machine.capacity(limits),bounded)
  +below = Bounds.nat_trans(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset),U32.to_nat(total),U32.to_nat(Machine.capacity(limits)),Bounds.nat_trans(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset),Nat.add(Nat.mul(block,U32.to_nat(plane)),U32.to_nat(plane)),U32.to_nat(total),Order.offset_lt(Nat.mul(block,U32.to_nat(plane)),offset,U32.to_nat(plane),inside),room),within)
  +exact = Bounds.roundtrip(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset),Machine.half_word(limits),Limits.half_word_fits(limits),
    Bounds.strict_le(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset),U32.to_nat(Machine.half_word(limits)),Bounds.nat_trans(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset),U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.half_word(limits)),below,
      Bounds.as_nat_le(Machine.capacity(limits),Machine.half_word(limits),Limits.capacity_within_half(limits)))))
  +divisor = Limits.double_bound(limits,U32.to_nat(plane),Order.transitive(U32.to_nat(plane),U32.to_nat(total),U32.to_nat(Machine.capacity(limits)),plane_within(Nat.mul(block,U32.to_nat(plane)),U32.to_nat(plane),U32.to_nat(total),room),within))
  +pair = Division.unique(U32.to_nat(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset))),U32.to_nat(plane),block,offset,exact,positive,inside)
  Decoded{
    Encoded.encoding((U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset)) / plane : U32),block,
      Equal.trans(Nat,U32.to_nat((U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset)) / plane : U32)),Nat.div(U32.to_nat(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset))),U32.to_nat(plane)),block,
        U32Division.divide(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset)),plane,Machine.half_word(limits),Limits.half_word_fits(limits),divisor,positive),
        Equal.cong(Nat & Nat,Nat,result => Pair.fst(Nat,Nat,result),Nat.divmod(U32.to_nat(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset))),U32.to_nat(plane)),(block,offset),pair))),
    Encoded.encoding((U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset)) % plane : U32),offset,
      Equal.trans(Nat,U32.to_nat((U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset)) % plane : U32)),Nat.mod(U32.to_nat(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset))),U32.to_nat(plane)),offset,
        U32Division.modulo(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset)),plane,Machine.half_word(limits),Limits.half_word_fits(limits),divisor,positive),
        Equal.cong(Nat & Nat,Nat,result => Pair.snd(Nat,Nat,result),Nat.divmod(U32.to_nat(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(plane)),offset))),U32.to_nat(plane)),(block,offset),pair))),
    local_offset(Nat.mul(block,U32.to_nat(plane)),offset)}

def decoded_agrees(+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+block: Nat,+offset: Nat,facts: Decoded<(rows * width : U32),block,offset>) ->
  {(((first / height * channels + U32.from_nat(block)) * height + first % height) * width + (U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat((rows * width : U32))),offset)) - U32.from_nat(Nat.mul(block,U32.to_nat((rows * width : U32)))) : U32) : U32) == (((first / height * channels + U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat((rows * width : U32))),offset)) / (rows * width : U32)) * height + first % height) * width + U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat((rows * width : U32))),offset)) % (rows * width : U32) : U32) : U32}:
  Decoded{quotient,remainder,local} = facts
  %Equal.sym(U32,(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat((rows * width : U32))),offset)) - U32.from_nat(Nat.mul(block,U32.to_nat((rows * width : U32)))) : U32),U32.from_nat(offset),local) :
    {(((first / height * channels + U32.from_nat(block)) * height + first % height) * width + _ : U32) == (((first / height * channels + U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat((rows * width : U32))),offset)) / (rows * width : U32)) * height + first % height) * width + U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat((rows * width : U32))),offset)) % (rows * width : U32) : U32) : U32}
  %Equal.sym(U32,(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat((rows * width : U32))),offset)) / (rows * width : U32) : U32),U32.from_nat(block),quotient) :
    {(((first / height * channels + U32.from_nat(block)) * height + first % height) * width + U32.from_nat(offset) : U32) == (((first / height * channels + _) * height + first % height) * width + U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat((rows * width : U32))),offset)) % (rows * width : U32) : U32) : U32}
  %Equal.sym(U32,(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat((rows * width : U32))),offset)) % (rows * width : U32) : U32),U32.from_nat(offset),remainder) :
    {(((first / height * channels + U32.from_nat(block)) * height + first % height) * width + U32.from_nat(offset) : U32) == (((first / height * channels + U32.from_nat(block)) * height + first % height) * width + _ : U32) : U32}
  {==}

def channel_agrees_at(+limits: Machine.Limits,+context: Bands.Indexed,+total: U32,+block: Nat,+offset: Nat,
  inside: {Nat.is_lt(offset,U32.to_nat(Bands.plane_of(context))) == True{} : Bool},
  room: {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(Bands.plane_of(context))),U32.to_nat(Bands.plane_of(context))),U32.to_nat(total)) == True{} : Bool},
  bounded: {U32.is_le(total,Machine.capacity(limits)) == True{} : Bool}) ->
  {Indexed.block_index(Indexed.Block{Bands.channel_base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(Bands.plane_of(context))))},U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(Bands.plane_of(context))),offset))) ==
    Bands.index_of(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(Bands.plane_of(context))),offset))) : U32}:
  Bands.Indexed{Bands.Geometry{batch,+channels,+height,+width},+first,+rows} = context
  decoded_agrees(channels,height,width,first,rows,block,offset,decoded(limits,total,(rows * width : U32),block,offset,inside,room,bounded))

# The limit the run-time guard uses is the capacity of the default machine.
def channel_agrees(+context: Bands.Indexed,+total: U32,+block: Nat,+offset: Nat,
  inside: {Nat.is_lt(offset,U32.to_nat(Bands.plane_of(context))) == True{} : Bool},
  room: {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(Bands.plane_of(context))),U32.to_nat(Bands.plane_of(context))),U32.to_nat(total)) == True{} : Bool},
  bounded: {U32.is_le(total,Indexed.block_limit()) == True{} : Bool}) ->
  {Indexed.block_index(Indexed.Block{Bands.channel_base(context,U32.from_nat(block)),U32.from_nat(Nat.mul(block,U32.to_nat(Bands.plane_of(context))))},U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(Bands.plane_of(context))),offset))) ==
    Bands.index_of(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(Bands.plane_of(context))),offset))) : U32}:
  channel_agrees_at(Default.limits(),context,total,block,offset,inside,room,bounded)

def buffers(~Element: Data,~operation: U32 -> Element -> Element,+count: U32,+context: Bands.Indexed,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,Bands.indexed_buffers(~Element,~operation,count,context,Buffer.pack(~Element,input),Buffer.pack(~Element,output))) ==
    buffer_model(~Element,~operation,Indexed.Call{U32.to_nat(count),context},Buffer.content(~Element,input),Buffer.content(~Element,output)) : Buffer.Content<Element>}:
  Buffer.Model{+source,+input_length,+input_certificate} = input
  Buffer.Model{+target,+output_length,+output_certificate} = output
  +program = IndexedTensorLaws.writes(~Element,~Bands.Indexed,~Bands.index_of,~operation,U32.to_nat(count),context,source)
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Bands.indexed_buffers(~Element,~operation,count,context,Buffer.pack(~Element,input),Buffer.pack(~Element,output))),
    BandGather.output(~Element,StorageCopyLaws.values(~Element,input_length,output_length,Indexed.run_tiled(~Element,~Bands.Indexed,~Bands.index_of,~Bands.channel_base,~Bands.channels_of,~Bands.plane_of,~operation,count,context,Tree.pack(Element,source),Tree.pack(Element,target)))),
    observed_pair(~Element,input_length,output_length,Indexed.run_tiled(~Element,~Bands.Indexed,~Bands.index_of,~Bands.channel_base,~Bands.channels_of,~Bands.plane_of,~operation,count,context,Tree.pack(Element,source),Tree.pack(Element,target)),
      IndexedCertificate.tiled_certificates(~Element,~Bands.Indexed,~Bands.index_of,~Bands.channel_base,~Bands.channels_of,~Bands.plane_of,~operation,count,context,
        Tree.pack(Element,source),Tree.pack(Element,target),input_certificate,output_certificate))) :
    {_ == buffer_model(~Element,~operation,Indexed.Call{U32.to_nat(count),context},Buffer.content(~Element,input),Buffer.content(~Element,output)) : Buffer.Content<Element>}
  %Equal.sym(Array<Element> & Array<Element>,Indexed.run_tiled(~Element,~Bands.Indexed,~Bands.index_of,~Bands.channel_base,~Bands.channels_of,~Bands.plane_of,~operation,count,context,Tree.pack(Element,source),Tree.pack(Element,target)),Indexed.run(~Element,~Bands.Indexed,~Bands.index_of,~operation,Indexed.Call{U32.to_nat(count),context},Tree.pack(Element,source),Tree.pack(Element,target)),
    IndexedTensorLaws.tiled(~Element,~Bands.Indexed,~Bands.index_of,~Bands.channel_base,~Bands.channels_of,~Bands.plane_of,~operation,~channel_agrees,count,context,source,target)) :
    {BandGather.output(~Element,StorageCopyLaws.values(~Element,input_length,output_length,_)) == buffer_model(~Element,~operation,Indexed.Call{U32.to_nat(count),context},Buffer.content(~Element,input),Buffer.content(~Element,output)) : Buffer.Content<Element>}
  %Equal.sym(Array<Element> & Array<Element>,Indexed.run(~Element,~Bands.Indexed,~Bands.index_of,~operation,Indexed.Call{U32.to_nat(count),context},Tree.pack(Element,source),Tree.pack(Element,target)),
    (Tree.pack(Element,source),Tree.pack(Element,Writes.pure(Element,program,target))),IndexedTensorLaws.run_writes(~Element,~Bands.Indexed,~Bands.index_of,~operation,U32.to_nat(count),context,source,target)) :
    {BandGather.output(~Element,StorageCopyLaws.values(~Element,input_length,output_length,_)) == buffer_model(~Element,~operation,Indexed.Call{U32.to_nat(count),context},Buffer.content(~Element,input),Buffer.content(~Element,output)) : Buffer.Content<Element>}
  Equal.cong(StorageCopyLaws.Owners<Element>,Buffer.Content<Element>,owners => BandGather.output(~Element,owners),
    StorageCopyLaws.values(~Element,input_length,output_length,(Tree.pack(Element,source),Tree.pack(Element,Writes.pure(Element,program,target)))),StorageCopyLaws.Owners{source,input_length,Writes.pure(Element,program,target),output_length},
    StorageCopyLaws.represented_values(~Element,source,input_length,Writes.pure(Element,program,target),output_length))

def leaf_model(~Element: Data,~operation: U32 -> Element -> Element,+geometry: Bands.Geometry,+first: U32,+rows: U32,+fill: Element,input: Buffer.Content<Element>) -> Buffer.Content<Element>:
  buffer_model(~Element,~operation,Indexed.Call{U32.to_nat(Bands.local_count(geometry,rows)),Bands.Indexed{geometry,first,rows}},input,Buffer.content(~Element,BandCount.model(~Element,Bands.local_count(geometry,rows),fill)))

type Context<-Element: Data> is Data:
  Context{count: U32,indexed: Bands.Indexed,input: Buffer.Model<Element>}

def proposition(~Element: Data,~operation: U32 -> Element -> Element,context: Context<Element>,-output: Storage.Buffer<Element>) -> Type:
  Context{+count,+indexed,+input} = context
  {Buffer.contents(~Element,Bands.indexed_buffers(~Element,~operation,count,indexed,Buffer.pack(~Element,input),output)) ==
    buffer_model(~Element,~operation,Indexed.Call{U32.to_nat(count),indexed},Buffer.content(~Element,input),Buffer.contents(~Element,output)) : Buffer.Content<Element>}

def buffers_native(~Element: Data,~operation: U32 -> Element -> Element,+count: U32,+context: Bands.Indexed,+input: Buffer.Model<Element>,output: Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,Bands.indexed_buffers(~Element,~operation,count,context,Buffer.pack(~Element,input),output)) ==
    buffer_model(~Element,~operation,Indexed.Call{U32.to_nat(count),context},Buffer.content(~Element,input),Buffer.contents(~Element,output)) : Buffer.Content<Element>}:
  Buffer.lift(~Element,~Context<Element>,~proposition(~Element,~operation),Context{count,context,input},output,Buffer.observe(~Element,output),output =>
    +output = output
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,output)),Buffer.content(~Element,output),Buffer.packed_contents(~Element,output)) :
      {Buffer.contents(~Element,Bands.indexed_buffers(~Element,~operation,count,context,Buffer.pack(~Element,input),Buffer.pack(~Element,output))) == buffer_model(~Element,~operation,Indexed.Call{U32.to_nat(count),context},Buffer.content(~Element,input),_) : Buffer.Content<Element>}
    buffers(~Element,~operation,count,context,input,output))

def leaf(~Element: Data,~operation: U32 -> Element -> Element,+geometry: Bands.Geometry,+first: U32,+rows: U32,+fill: Element,+input: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,Bands.indexed_leaf(~Element,~operation,geometry,first,rows,fill,Buffer.pack(~Element,input))) == leaf_model(~Element,~operation,geometry,first,rows,fill,Buffer.content(~Element,input)) : Buffer.Content<Element>}:
  +count = Bands.local_count(geometry,rows)
  +context = {Bands.Indexed{geometry,first,rows} : Bands.Indexed}
  +call = {Indexed.Call{U32.to_nat(count),context} : Indexed.Call<Bands.Indexed>}
  %Depth.depth(count) : {Buffer.contents(~Element,Bands.indexed_buffers(~Element,~operation,count,context,Buffer.pack(~Element,input),Allocation.allocate(~Element,_,fill,U32.to_nat(count)))) ==
    leaf_model(~Element,~operation,geometry,first,rows,fill,Buffer.content(~Element,input)) : Buffer.Content<Element>}
  Equal.trans(Buffer.Content<Element>,Buffer.contents(~Element,Bands.indexed_buffers(~Element,~operation,count,context,Buffer.pack(~Element,input),Bands.allocate(~Element,fill,count))),
    buffer_model(~Element,~operation,call,Buffer.content(~Element,input),Buffer.contents(~Element,Bands.allocate(~Element,fill,count))),leaf_model(~Element,~operation,geometry,first,rows,fill,Buffer.content(~Element,input)),
    buffers_native(~Element,~operation,count,context,input,Bands.allocate(~Element,fill,count)),
    Equal.cong(Buffer.Content<Element>,Buffer.Content<Element>,output => buffer_model(~Element,~operation,call,Buffer.content(~Element,input),output),
      Buffer.contents(~Element,Bands.allocate(~Element,fill,count)),Buffer.content(~Element,BandCount.model(~Element,count,fill)),AllocationLaws.contents(~Element,Storage.depth_for(count),fill,U32.to_nat(count))))

def band_model(~Element: Data,~operation: U32 -> Element -> Element,context: Bands.Filled<Element>,first: U32,rows: U32,input: Buffer.Model<Element>) -> Model.Content<Element>:
  Bands.Filled{geometry,fill} = context
  +first = first
  +rows = rows
  Model.RowBand{leaf_model(~Element,~operation,geometry,first,rows,fill,Buffer.content(~Element,input)),first,rows}

def model(~Element: Data,~operation: U32 -> Element -> Element,geometry: Bands.Geometry,fill: Element,bands: Model.Model<Element>) -> Model.Content<Element>:
  Each.content_map(~Element,~Bands.Filled<Element>,~band_model(~Element,~operation),Bands.Filled{geometry,fill},bands)

def band(~Element: Data,~operation: U32 -> Element -> Element,+context: Bands.Filled<Element>,+first: U32,+rows: U32,+input: Buffer.Model<Element>) ->
  {Model.contents(~Element,Bands.indexed_band(~Element,~operation,context,first,rows,Buffer.pack(~Element,input),Unit{})) ==
    band_model(~Element,~operation,context,first,rows,input) : Model.Content<Element>}:
  Bands.Filled{+geometry,+fill} = context
  Equal.cong(Buffer.Content<Element>,Model.Content<Element>,buffer => Model.RowBand{buffer,first,rows},
    Buffer.contents(~Element,Bands.indexed_leaf(~Element,~operation,geometry,first,rows,fill,Buffer.pack(~Element,input))),leaf_model(~Element,~operation,geometry,first,rows,fill,Buffer.content(~Element,input)),leaf(~Element,~operation,geometry,first,rows,fill,input))

# The indexed operation on the plan of its own bands, in either mode.
def apply(~Element: Data,~operation: U32 -> Element -> Element,+parallel: Bool,+geometry: Bands.Geometry,+fill: Element,+bands: Model.Model<Element>) ->
  {Model.contents(~Element,Bands.apply_indexed(~Element,~operation,Model.layout(~Element,bands),parallel,geometry,fill,Model.pack(~Element,bands))) == model(~Element,~operation,geometry,fill,bands) : Model.Content<Element>}:
  Each.mapped(~Element,~Bands.Filled<Element>,~Bands.indexed_band(~Element,~operation),~band_model(~Element,~operation),~band(~Element,~operation),
    bands,parallel && Bands.same_image(geometry,Model.layout(~Element,bands)),parallel,geometry,Bands.Filled{geometry,fill})

def scheduled(~Element: Data,~operation: U32 -> Element -> Element,+operations: F32,+geometry: Bands.Geometry,+workers: U32,+fill: Element,+bands: Model.Model<Element>) ->
  {Model.contents(~Element,BandSchedule.indexed(~Element,~operation,geometry,workers,operations,fill,Model.pack(~Element,bands))) == model(~Element,~operation,geometry,fill,bands) : Model.Content<Element>}:
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,bands)),(Model.pack(~Element,bands),Model.layout(~Element,bands)),Model.plan(~Element,bands)) :
    {Model.contents(~Element,BandSchedule.indexed_planned(~Element,~operation,geometry,workers,operations,fill,_)) == model(~Element,~operation,geometry,fill,bands) : Model.Content<Element>}
  apply(~Element,~operation,BandSchedule.parallel(geometry,workers,12.0,operations,Model.layout(~Element,bands)),geometry,fill,bands)

# Indexed band maps read the original value of a cell and apply the operation
# at the index of its pixel. The actual allocation and cursor discharge
# readback; the gather domain keeps the native index within the machine word.
def value(~Element: Data,~Logical: Data,~operation: U32 -> Element -> Element,~original: Logical -> U32 -> Element,logical: Logical,index: U32) -> Element:
  +index = index
  operation(index,original(logical,index))

# A written cell reads the source cell and applies the operation at the native
# global index, which the leaf facts identify with the index of the pixel.
def written_cell(~Element: Data,~Logical: Data,~operation: U32 -> Element -> Element,~original: Logical -> U32 -> Element,
  +logical: Logical,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+limits: Machine.Limits,+capacity: U32,
  +channel: Nat,+row: Nat,+column: Nat,facts: BandGather.Facts<BandGather.Call{geometry,first,rows,Buffer.length(~Element,input),capacity,Machine.half_word(limits)}>,
  ready: Buffer.Sized<Element,output>,count_equal: {U32.to_nat(Bands.local_count(geometry,rows)) == Buffer.length(~Element,input) : Nat},
  length_equal: {U32.to_nat(Bands.local_count(geometry,rows)) == Buffer.length(~Element,output) : Nat},
  room: {Nat.is_le(U32.to_nat(Bands.local_count(geometry,rows)),U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,output)))) == True{} : Bool},
  channel_inside: {Nat.is_lt(channel,BandValues.channels(geometry)) == True{} : Bool},row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(geometry)) == True{} : Bool},
  cell: BandValues.At(~Element,~Logical,~original,logical,geometry,first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input),channel,row,column)) ->
  BandValues.At(~Element,~Logical,~value(~Element,~Logical,~operation,~original),logical,geometry,first,rows,
    Buffer.content_tree(~Element,buffer_model(~Element,~operation,Indexed.Call{U32.to_nat(Bands.local_count(geometry,rows)),Bands.Indexed{geometry,first,rows}},Buffer.content(~Element,input),Buffer.content(~Element,output))),
    Buffer.content_length(~Element,buffer_model(~Element,~operation,Indexed.Call{U32.to_nat(Bands.local_count(geometry,rows)),Bands.Indexed{geometry,first,rows}},Buffer.content(~Element,input),Buffer.content(~Element,output))),channel,row,column):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  Buffer.Model{+source,+input_length,+input_certificate} = input
  Buffer.Model{+target,+output_length,+output_certificate} = output
  Buffer.Sized{+depth_ok,output_room} = ready
  Shape.Both{+inside,equal} = cell
  +count = U32.to_nat(Bands.local_count(Bands.Geometry{images,channels,height,width},rows))
  +context = {Bands.Indexed{Bands.Geometry{images,channels,height,width},first,rows} : Bands.Indexed}
  +position = BandValues.position(Bands.Geometry{images,channels,height,width},rows,channel,row,column)
  +global = Bands.global_index(Bands.Geometry{images,channels,height,width},first,rows,U32.from_nat(position))
  +pixel = U32.from_nat(BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),row),column))
  +selected = {%Equal.sym(Nat,count,input_length,count_equal) : {Nat.is_lt(position,_) == True{} : Bool} inside : {Nat.is_lt(position,count) == True{} : Bool}}
  +bound = {%length_equal : {Nat.is_lt(position,_) == True{} : Bool} selected : {Nat.is_lt(position,output_length) == True{} : Bool}}
  Shape.Both{bound,
    Equal.trans(Element,Tree.read(Element,Writes.pure(Element,IndexedTensorLaws.writes(~Element,~Bands.Indexed,~Bands.index_of,~operation,count,context,source),target),U32.from_nat(position)),
      operation(global,Tree.read(Element,source,U32.from_nat(position))),value(~Element,~Logical,~operation,~original,logical,pixel),
      TraversalIndexedWrites.read(~Element,~TraversalIndexedWrites.Input<Tree.Tree<Element>,Bands.Indexed>,~TraversalIndexedWrites.value(~Tree.Tree<Element>,~Element,~Bands.Indexed,~IndexedTensorLaws.sample(~Element,~Bands.Indexed,~Bands.index_of,~operation)),
        count,TraversalIndexedWrites.Input{source,context},target,Buffer.depth(~Element,output),position,Buffer.balanced(~Element,output),depth_ok,Buffer.capacity_bound(~Element,output,depth_ok),room,selected),
      %Equal.sym(U32,global,pixel,BandGather.indexed_pixel(BandGather.Call{Bands.Geometry{images,channels,height,width},first,rows,input_length,capacity,Machine.half_word(limits)},channel,row,column,facts,channel_inside,row_inside,column_inside,inside)) :
        {operation(_,Tree.read(Element,source,U32.from_nat(position))) == value(~Element,~Logical,~operation,~original,logical,pixel) : Element}
      Equal.cong(Element,Element,element => operation(pixel,element),Tree.read(Element,source,U32.from_nat(position)),original(logical,pixel),equal))}

def allocated_cell(~Element: Data,~Logical: Data,~operation: U32 -> Element -> Element,~original: Logical -> U32 -> Element,
  +logical: Logical,+geometry: Bands.Geometry,+first: U32,+rows: U32,+fill: Element,+input: Buffer.Model<Element>,+limits: Machine.Limits,+capacity: U32,
  +channel: Nat,+row: Nat,+column: Nat,facts: BandGather.Facts<BandGather.Call{geometry,first,rows,Buffer.length(~Element,input),capacity,Machine.half_word(limits)}>,
  +ready: Buffer.Sized<Element,BandCount.model(~Element,Bands.local_count(geometry,rows),fill)>,count_equal: {U32.to_nat(Bands.local_count(geometry,rows)) == Buffer.length(~Element,input) : Nat},
  channel_inside: {Nat.is_lt(channel,BandValues.channels(geometry)) == True{} : Bool},row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(geometry)) == True{} : Bool},
  cell: BandValues.At(~Element,~Logical,~original,logical,geometry,first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input),channel,row,column)) ->
  BandValues.At(~Element,~Logical,~value(~Element,~Logical,~operation,~original),logical,geometry,first,rows,
    Buffer.content_tree(~Element,leaf_model(~Element,~operation,geometry,first,rows,fill,Buffer.content(~Element,input))),Buffer.content_length(~Element,leaf_model(~Element,~operation,geometry,first,rows,fill,Buffer.content(~Element,input))),channel,row,column):
  Buffer.Sized{depth_ok,room} = ready
  written_cell(~Element,~Logical,~operation,~original,logical,geometry,first,rows,input,BandCount.model(~Element,Bands.local_count(geometry,rows),fill),limits,capacity,channel,row,column,
    facts,ready,count_equal,{==},room,channel_inside,row_inside,column_inside,cell)

# What one leaf supplies to every cell of its output.
type LeafCells<-Element: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  LeafCells{logical: Logical,geometry: Bands.Geometry,first: U32,rows: U32,fill: Element,input: Buffer.Model<Element>,limits: Machine.Limits,capacity: U32,
    domain: BandGather.Domain<geometry,limits,capacity>,metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),ready: Buffer.Sized<Element,input>,
    count: BandShape.Count<geometry,rows,Buffer.length(~Element,input)>,
    values: BandValues.Leaf(~Element,~Logical,~original,logical,geometry,first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input))}

def cells_geometry(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: LeafCells<Element,Logical,original>) -> Bands.Geometry:
  LeafCells{logical,geometry,first,rows,fill,input,limits,capacity,domain,metadata,ready,count,values} = evidence
  geometry

def cells_rows(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: LeafCells<Element,Logical,original>) -> U32:
  LeafCells{logical,geometry,first,rows,fill,input,limits,capacity,domain,metadata,ready,count,values} = evidence
  rows

def Cell(~Element: Data,~Logical: Data,~operation: U32 -> Element -> Element,~original: Logical -> U32 -> Element,+evidence: LeafCells<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  LeafCells{logical,+geometry,+first,+rows,+fill,+input,limits,capacity,domain,metadata,ready,count,values} = evidence
  BandValues.At(~Element,~Logical,~value(~Element,~Logical,~operation,~original),logical,geometry,first,rows,
    Buffer.content_tree(~Element,leaf_model(~Element,~operation,geometry,first,rows,fill,Buffer.content(~Element,input))),Buffer.content_length(~Element,leaf_model(~Element,~operation,geometry,first,rows,fill,Buffer.content(~Element,input))),channel,row,column)

def leaf_cell(~Element: Data,~Logical: Data,~operation: U32 -> Element -> Element,~original: Logical -> U32 -> Element,+evidence: LeafCells<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(cells_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(cells_rows(~Element,~Logical,~original,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(cells_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool}) ->
  Cell(~Element,~Logical,~operation,~original,evidence,channel,row,column):
  LeafCells{+logical,+geometry,+first,+rows,+fill,+input,+limits,+capacity,domain,metadata,ready,count,values} = evidence
  BandShape.Count{exact,allocatable} = count
  +channel_inside = channel_inside
  +row_inside = row_inside
  +column_inside = column_inside
  allocated_cell(~Element,~Logical,~operation,~original,logical,geometry,first,rows,fill,input,limits,capacity,channel,row,column,
    BandGather.from_metadata(~Element,geometry,limits,first,rows,input,capacity,domain,metadata,Order.le_lt_trans(0n,row,U32.to_nat(rows),Order.zero_le(row),row_inside),ready),
    BandCount.bounded_model(~Element,Bands.local_count(geometry,rows),fill,allocatable),exact,channel_inside,row_inside,column_inside,
    BandValues.leaf_cell(~Element,~Logical,~original,logical,geometry,first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input),channel,row,column,values,channel_inside,row_inside,column_inside))

def leaf_values(~Element: Data,~Logical: Data,~operation: U32 -> Element -> Element,~original: Logical -> U32 -> Element,
  +logical: Logical,+geometry: Bands.Geometry,+first: U32,+rows: U32,+fill: Element,+input: Buffer.Model<Element>,+limits: Machine.Limits,+capacity: U32,
  domain: BandGather.Domain<geometry,limits,capacity>,metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),ready: Buffer.Sized<Element,input>,
  facts: BandShape.Count<geometry,rows,Buffer.length(~Element,input)>,
  values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.RowBand{Buffer.content(~Element,input),first,rows})) ->
  BandValues.Values(~Element,~Logical,~value(~Element,~Logical,~operation,~original),logical,geometry,Model.RowBand{leaf_model(~Element,~operation,geometry,first,rows,fill,Buffer.content(~Element,input)),first,rows}):
  Buffer.Model{+tree,+length,+certificate} = input
  BandValues.cells(~LeafCells<Element,Logical,original>,~Cell(~Element,~Logical,~operation,~original),~cells_geometry(~Element,~Logical,~original),~cells_rows(~Element,~Logical,~original),~leaf_cell(~Element,~Logical,~operation,~original),
    LeafCells{logical,geometry,first,rows,fill,Buffer.Model{tree,length,certificate},limits,capacity,domain,metadata,ready,facts,values})

def leaf_ready(~Element: Data,~operation: U32 -> Element -> Element,+geometry: Bands.Geometry,+first: U32,+rows: U32,+fill: Element,input: Buffer.Model<Element>,
  facts: BandShape.Count<geometry,rows,Buffer.length(~Element,input)>) -> Buffer.ContentReady<Element,leaf_model(~Element,~operation,geometry,first,rows,fill,Buffer.content(~Element,input))>:
  Buffer.Model{tree,length,certificate} = input
  BandShape.Count{exact,allocatable} = facts
  +count = Bands.local_count(geometry,rows)
  Writes.content_ready(~Element,IndexedTensorLaws.writes(~Element,~Bands.Indexed,~Bands.index_of,~operation,U32.to_nat(count),Bands.Indexed{geometry,first,rows},tree),
    Buffer.tree(~Element,BandCount.model(~Element,count,fill)),U32.to_nat(count),Buffer.content_ready(~Element,BandCount.model(~Element,count,fill),BandCount.bounded_model(~Element,count,fill,allocatable)))

def leaf_shape(~Element: Data,~operation: U32 -> Element -> Element,+geometry: Bands.Geometry,+first: U32,+rows: U32,+fill: Element,input: Buffer.Model<Element>,
  facts: BandShape.Count<geometry,rows,Buffer.length(~Element,input)>,metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input))) ->
  BandShape.Contents(~Element,geometry,Model.RowBand{leaf_model(~Element,~operation,geometry,first,rows,fill,Buffer.content(~Element,input)),first,rows}):
  Buffer.Model{tree,+length,certificate} = input
  BandShape.Count{exact,allocatable} = facts
  %Equal.sym(Nat,U32.to_nat(Bands.local_count(geometry,rows)),length,exact) : BandShape.Leaf(geometry,first,rows,_) metadata

def modeled_values(~Element: Data,~Logical: Data,~operation: U32 -> Element -> Element,~original: Logical -> U32 -> Element,
  +logical: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+fill: Element,bands: Model.Model<Element>,+domain: BandGather.Domain<geometry,limits,capacity>,
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands)),ready: Model.Ready(~Element,bands),values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands))) ->
  BandValues.Values(~Element,~Logical,~value(~Element,~Logical,~operation,~original),logical,geometry,model(~Element,~operation,geometry,fill,bands)):
  match bands:
    case Model.Leaf{+input,+first,+rows}:
      Buffer.Model{+tree,+length,+certificate} = input
      +metadata = metadata
      +ready = ready
      leaf_values(~Element,~Logical,~operation,~original,logical,geometry,first,rows,fill,input,limits,capacity,domain,metadata,ready,
        BandShape.count(~Element,geometry,first,rows,input,width_positive,metadata,ready),values)
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{left_values,right_values} = values
      Shape.Both{modeled_values(~Element,~Logical,~operation,~original,logical,geometry,limits,capacity,fill,left,domain,width_positive,left_metadata,left_ready,left_values),
        modeled_values(~Element,~Logical,~operation,~original,logical,geometry,limits,capacity,fill,right,domain,width_positive,right_metadata,right_ready,right_values)}

def modeled_fields(~Element: Data,~operation: U32 -> Element -> Element,+geometry: Bands.Geometry,+fill: Element,bands: Model.Model<Element>,+width_positive: {BandShape.width_positive(geometry) == True{} : Bool},
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands)),ready: Model.Ready(~Element,bands)) ->
  Shape.Both<BandShape.Contents(~Element,geometry,model(~Element,~operation,geometry,fill,bands)),Model.ContentReady(~Element,model(~Element,~operation,geometry,fill,bands))>:
  match bands:
    case Model.Leaf{+input,+first,+rows}:
      Buffer.Model{+tree,+length,+certificate} = input
      +metadata = metadata
      +count = BandShape.count(~Element,geometry,first,rows,input,width_positive,metadata,ready)
      Shape.Both{leaf_shape(~Element,~operation,geometry,first,rows,fill,input,count,metadata),leaf_ready(~Element,~operation,geometry,first,rows,fill,input,count)}
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_ready,right_ready} = ready
      +left_fields = modeled_fields(~Element,~operation,geometry,fill,left,width_positive,left_metadata,left_ready)
      +right_fields = modeled_fields(~Element,~operation,geometry,fill,right,width_positive,right_metadata,right_ready)
      Shape.Both{
        Shape.Both{Shape.first(BandShape.Contents(~Element,geometry,model(~Element,~operation,geometry,fill,left)),Model.ContentReady(~Element,model(~Element,~operation,geometry,fill,left)),left_fields),
          Shape.first(BandShape.Contents(~Element,geometry,model(~Element,~operation,geometry,fill,right)),Model.ContentReady(~Element,model(~Element,~operation,geometry,fill,right)),right_fields)},
        Shape.Both{Shape.second(BandShape.Contents(~Element,geometry,model(~Element,~operation,geometry,fill,left)),Model.ContentReady(~Element,model(~Element,~operation,geometry,fill,left)),left_fields),
          Shape.second(BandShape.Contents(~Element,geometry,model(~Element,~operation,geometry,fill,right)),Model.ContentReady(~Element,model(~Element,~operation,geometry,fill,right)),right_fields)}}

def modeled_layout(~Element: Data,~operation: U32 -> Element -> Element,+geometry: Bands.Geometry,+fill: Element,bands: Model.Model<Element>) ->
  {Model.content_layout(~Element,model(~Element,~operation,geometry,fill,bands)) == Model.layout(~Element,bands) : Bands.Plan}:
  match bands:
    case Model.Leaf{Buffer.Model{tree,length,certificate},first,rows}: {==}
    case Model.Fork{+left,+right}:
      %modeled_layout(~Element,~operation,geometry,fill,left) :
        {Bands.Split{Model.content_layout(~Element,model(~Element,~operation,geometry,fill,left)),Model.content_layout(~Element,model(~Element,~operation,geometry,fill,right))} == Bands.Split{_,Model.layout(~Element,right)} : Bands.Plan}
      %modeled_layout(~Element,~operation,geometry,fill,right) :
        {Bands.Split{Model.content_layout(~Element,model(~Element,~operation,geometry,fill,left)),Model.content_layout(~Element,model(~Element,~operation,geometry,fill,right))} == Bands.Split{Model.content_layout(~Element,model(~Element,~operation,geometry,fill,left)),_} : Bands.Plan}
      {==}

def modeled_well_formed(~Element: Data,~operation: U32 -> Element -> Element,+geometry: Bands.Geometry,+fill: Element,+bands: Model.Model<Element>,width_positive: {BandShape.width_positive(geometry) == True{} : Bool},
  evidence: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>) -> BandShape.WellFormed<Element,geometry,model(~Element,~operation,geometry,fill,bands)>:
  BandShape.WellFormed{metadata,ready,coverage} = evidence
  +fields = modeled_fields(~Element,~operation,geometry,fill,bands,width_positive,metadata,Model.ready(~Element,bands,ready))
  BandShape.WellFormed{
    Shape.first(BandShape.Contents(~Element,geometry,model(~Element,~operation,geometry,fill,bands)),Model.ContentReady(~Element,model(~Element,~operation,geometry,fill,bands)),fields),
    Shape.second(BandShape.Contents(~Element,geometry,model(~Element,~operation,geometry,fill,bands)),Model.ContentReady(~Element,model(~Element,~operation,geometry,fill,bands)),fields),
    %Equal.sym(Bands.Plan,Model.content_layout(~Element,model(~Element,~operation,geometry,fill,bands)),Model.layout(~Element,bands),modeled_layout(~Element,~operation,geometry,fill,bands)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
    %Equal.sym(Bands.Plan,Model.layout(~Element,bands),Model.content_layout(~Element,Model.content(~Element,bands)),Model.layout_matches_content(~Element,bands)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry)) coverage}

def scheduled_values(~Element: Data,~Logical: Data,~operation: U32 -> Element -> Element,~original: Logical -> U32 -> Element,+operations: F32,
  +logical: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+workers: U32,+fill: Element,+bands: Model.Model<Element>,domain: BandGather.Domain<geometry,limits,capacity>,
  width_positive: {BandShape.width_positive(geometry) == True{} : Bool},
  evidence: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands))) ->
  BandValues.Values(~Element,~Logical,~value(~Element,~Logical,~operation,~original),logical,geometry,Model.contents(~Element,BandSchedule.indexed(~Element,~operation,geometry,workers,operations,fill,Model.pack(~Element,bands)))):
  BandShape.WellFormed{metadata,ready,coverage} = evidence
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandSchedule.indexed(~Element,~operation,geometry,workers,operations,fill,Model.pack(~Element,bands))),model(~Element,~operation,geometry,fill,bands),scheduled(~Element,~operation,operations,geometry,workers,fill,bands)) :
    BandValues.Values(~Element,~Logical,~value(~Element,~Logical,~operation,~original),logical,geometry,_)
  modeled_values(~Element,~Logical,~operation,~original,logical,geometry,limits,capacity,fill,bands,domain,width_positive,metadata,Model.ready(~Element,bands,ready),values)

def scheduled_well_formed(~Element: Data,~operation: U32 -> Element -> Element,+operations: F32,+geometry: Bands.Geometry,+workers: U32,+fill: Element,+bands: Model.Model<Element>,width_positive: {BandShape.width_positive(geometry) == True{} : Bool},
  evidence: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>) ->
  BandShape.WellFormed<Element,geometry,Model.contents(~Element,BandSchedule.indexed(~Element,~operation,geometry,workers,operations,fill,Model.pack(~Element,bands)))>:
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandSchedule.indexed(~Element,~operation,geometry,workers,operations,fill,Model.pack(~Element,bands))),model(~Element,~operation,geometry,fill,bands),scheduled(~Element,~operation,operations,geometry,workers,fill,bands)) : BandShape.WellFormed<Element,geometry,_>
  modeled_well_formed(~Element,~operation,geometry,fill,bands,width_positive,evidence)

def gathered(~Element: Data,~Logical: Data,~operation: U32 -> Element -> Element,~original: Logical -> U32 -> Element,+operations: F32,
  +logical: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,+workers: U32,+fill: Element,+bands: Model.Model<Element>,+result: Buffer.Model<Element>,
  +domain: BandGather.Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result))>,
  +well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands)),
  +image: Nat,+channel: Nat,+row: Nat,+column: Nat,inside: BandGather.Inside(geometry,image,channel,row,column),
  depth_ok: {Nat.is_le(Buffer.depth(~Element,result),30n) == True{} : Bool}) ->
  {Buffer.value(~Element,U32.from_nat(BandGather.point(geometry,image,channel,row,column)),
    Buffer.contents(~Element,Bands.gather(~Element,geometry,BandSchedule.indexed(~Element,~operation,geometry,workers,operations,fill,Model.pack(~Element,bands)),Buffer.pack(~Element,result)))) ==
    operation(U32.from_nat(BandGather.point(geometry,image,channel,row,column)),original(logical,U32.from_nat(BandGather.point(geometry,image,channel,row,column)))) : Element}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandGather.Domain{capacity_ok,channels_positive,height_positive,+width_positive,machine_room,allocation_room} = domain
  BandGather.observed_value(~Element,~Logical,~value(~Element,~Logical,~operation,~original),logical,geometry,limits,
    BandSchedule.indexed(~Element,~operation,geometry,workers,operations,fill,Model.pack(~Element,bands)),Model.observe(~Element,BandSchedule.indexed(~Element,~operation,geometry,workers,operations,fill,Model.pack(~Element,bands))),result,domain,
    scheduled_well_formed(~Element,~operation,operations,geometry,workers,fill,bands,width_positive,well_formed),
    scheduled_values(~Element,~Logical,~operation,~original,operations,logical,geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result)),workers,fill,bands,domain,width_positive,well_formed,values),image,channel,row,column,inside,depth_ok)
