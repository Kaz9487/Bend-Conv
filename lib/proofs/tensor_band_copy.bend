# Per-channel band copies: region bounds, the write program they refine, and
# readback of the copied region.
import Base
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./storage_buffer_model.bend as Buffer
import ./storage_copy_refinement.bend as StorageCopyLaws
import ./storage_tree_model.bend as Tree
import ./strided_range_bounds.bend as Strided
import ./write_program.bend as Writes
import ../tensor_band.bend as Bands
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./contiguous_copy.bend as Contiguous
import ./contiguous_copy_readback.bend as CopyReadback
import ./nat_division.bend as Division
import ./nat_interval.bend as Interval
import ./natural_addition.bend as Addition
import ./storage_certificate.bend as Certificate
import ./storage_refinement.bend as StorageLaws
import ./storage_shape.bend as Shape
import ./tensor_band_coordinates.bend as BandCoordinates
import ./word_encoding.bend as WordEncoding
import ./write_program_readback.bend as Readback
import ../storage_buffer.bend as Storage

# The actual per-channel Buffer loop refines an independent sequence of range
# writes. Native owners are observed once; the induction works on their trees.
def source(~Element: Data,owners: StorageCopyLaws.Owners<Element>) -> Buffer.Content<Element>:
  StorageCopyLaws.Owners{input,input_length,output,output_length} = owners
  Buffer.Content{input,input_length}

type Request is Data:
  Request{remaining: Nat,channel: U32,count: U32,source_stride: U32,destination_stride: U32,source_start: U32,destination_start: U32,
    source_length: Nat,output_length: Nat}

def source_length(request: Request) -> Nat:
  Request{remaining,channel,count,ss,ds,so,target_start,source_length,output_length} = request
  source_length

def output_length(request: Request) -> Nat:
  Request{remaining,channel,count,ss,ds,so,target_start,source_length,output_length} = request
  output_length

def model(~Element: Data,remaining: Nat,+channel: U32,+count: U32,+source_stride: U32,+destination_stride: U32,+source_start: U32,+destination_start: U32,
  owners: StorageCopyLaws.Owners<Element>) -> StorageCopyLaws.Owners<Element>:
  match remaining:
    case 0n: owners
    case 1n+rest:
      model(~Element,rest,(channel + 1 : U32),count,source_stride,destination_stride,source_start,destination_start,
        StorageCopyLaws.model(~Element,count,(source_start + channel * source_stride : U32),(destination_start + channel * destination_stride : U32),owners))

def plane_writes(~Element: Data,channel: U32,count: U32,source_stride: U32,destination_stride: U32,source_start: U32,destination_start: U32,input: Tree.Tree<Element>) -> List<&2,Writes.Assignment<Element>>:
  +channel = channel
  Contiguous.writes(~Element,U32.to_nat(count),input,0,(source_start + channel * source_stride : U32),(destination_start + channel * destination_stride : U32))

def writes(~Element: Data,remaining: Nat,+channel: U32,+count: U32,+source_stride: U32,+destination_stride: U32,+source_start: U32,+destination_start: U32,+input: Tree.Tree<Element>) ->
  List<&2,Writes.Assignment<Element>>:
  match remaining:
    case 0n: []
    case 1n+rest:
      List.append(&2,Writes.Assignment<Element>,plane_writes(~Element,channel,count,source_stride,destination_stride,source_start,destination_start,input),
        writes(~Element,rest,(channel + 1 : U32),count,source_stride,destination_stride,source_start,destination_start,input))

def model_writes(~Element: Data,remaining: Nat,+channel: U32,+count: U32,+source_stride: U32,+destination_stride: U32,+source_start: U32,+destination_start: U32,
  +input: Tree.Tree<Element>,+input_length: Nat,+output: Tree.Tree<Element>,+output_length: Nat) ->
  {model(~Element,remaining,channel,count,source_stride,destination_stride,source_start,destination_start,StorageCopyLaws.Owners{input,input_length,output,output_length}) ==
    StorageCopyLaws.Owners{input,input_length,Writes.pure(Element,writes(~Element,remaining,channel,count,source_stride,destination_stride,source_start,destination_start,input),output),output_length} : StorageCopyLaws.Owners<Element>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +here = plane_writes(~Element,channel,count,source_stride,destination_stride,source_start,destination_start,input)
      +later = writes(~Element,rest,(channel + 1 : U32),count,source_stride,destination_stride,source_start,destination_start,input)
      %Equal.sym(Tree.Tree<Element>,Writes.pure(Element,List.append(&2,Writes.Assignment<Element>,here,later),output),
        Writes.pure(Element,later,Writes.pure(Element,here,output)),Writes.concatenated_writes(Element,here,later,output)) :
        {model(~Element,1n+rest,channel,count,source_stride,destination_stride,source_start,destination_start,StorageCopyLaws.Owners{input,input_length,output,output_length}) == StorageCopyLaws.Owners{input,input_length,_,output_length} : StorageCopyLaws.Owners<Element>}
      model_writes(~Element,rest,(channel + 1 : U32),count,source_stride,destination_stride,source_start,destination_start,input,input_length,Writes.pure(Element,here,output),output_length)

def modeled(~Element: Data,request: Request,owners: StorageCopyLaws.Owners<Element>) -> StorageCopyLaws.Owners<Element>:
  Request{remaining,channel,count,ss,ds,so,target_start,source_length,output_length} = request
  model(~Element,remaining,channel,count,ss,ds,so,target_start,owners)

def actual(~Element: Data,request: Request,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  Request{remaining,channel,count,ss,ds,so,target_start,source_length,output_length} = request
  Bands.planes(~Element,remaining,channel,count,ss,ds,so,target_start,owners)

def proposition(~Element: Data,request: Request,-input: Array<Element>,-output: Array<Element>,-input_tree: Tree.Tree<Element>,-output_tree: Tree.Tree<Element>) -> Type:
  +request = request
  @input_certificate: Certificate.Certificate<Element,input> -> @output_certificate: Certificate.Certificate<Element,output> ->
  {StorageCopyLaws.observe(~Element,actual(~Element,request,(Storage.Buffer{input,source_length(request),input_certificate},Storage.Buffer{output,output_length(request),output_certificate}))) ==
    modeled(~Element,request,StorageCopyLaws.Owners{input_tree,source_length(request),output_tree,output_length(request)}) : StorageCopyLaws.Owners<Element>}

def copied(~Element: Data,call: StorageCopyLaws.Call,input: Tree.Tree<Element>,output: Tree.Tree<Element>,
  input_certificate: Certificate.Certificate<Element,Tree.pack(Element,input)>,output_certificate: Certificate.Certificate<Element,Tree.pack(Element,output)>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  StorageCopyLaws.Call{count,source,destination,input_length,output_length} = call
  StorageCopyLaws.range_pair(~Element,count,source,destination,(Storage.Buffer{Tree.pack(Element,input),input_length,input_certificate},Storage.Buffer{Tree.pack(Element,output),output_length,output_certificate}))

def planes(~Element: Data,remaining: Nat,+channel: U32,+count: U32,+source_stride: U32,+destination_stride: U32,+source_start: U32,+destination_start: U32,
  owners: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  {StorageCopyLaws.observe(~Element,Bands.planes(~Element,remaining,channel,count,source_stride,destination_stride,source_start,destination_start,owners)) ==
    model(~Element,remaining,channel,count,source_stride,destination_stride,source_start,destination_start,StorageCopyLaws.observe(~Element,owners)) : StorageCopyLaws.Owners<Element>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      (Storage.Buffer{input,+input_length,input_certificate},Storage.Buffer{output,+output_length,output_certificate}) = owners
      +request = {Request{1n+rest,channel,count,source_stride,destination_stride,source_start,destination_start,input_length,output_length} : Request}
      +call = {StorageCopyLaws.Call{count,(source_start + channel * source_stride : U32),(destination_start + channel * destination_stride : U32),input_length,output_length} : StorageCopyLaws.Call}
      StorageLaws.lift_pair_observation(~Element,~Request,~proposition(~Element),request,input,output,StorageLaws.observe(Element,input),StorageLaws.observe(Element,output),
        input_tree => output_tree => input_certificate => output_certificate =>
          +input_tree = input_tree
          +output_tree = output_tree
          +input_certificate = input_certificate
          +output_certificate = output_certificate
          Equal.trans(StorageCopyLaws.Owners<Element>,
            StorageCopyLaws.observe(~Element,Bands.planes(~Element,rest,(channel + 1 : U32),count,source_stride,destination_stride,source_start,destination_start,
              copied(~Element,call,input_tree,output_tree,input_certificate,output_certificate))),
            model(~Element,rest,(channel + 1 : U32),count,source_stride,destination_stride,source_start,destination_start,
              StorageCopyLaws.observe(~Element,copied(~Element,call,input_tree,output_tree,input_certificate,output_certificate))),
            model(~Element,1n+rest,channel,count,source_stride,destination_stride,source_start,destination_start,StorageCopyLaws.Owners{input_tree,input_length,output_tree,output_length}),
            planes(~Element,rest,(channel + 1 : U32),count,source_stride,destination_stride,source_start,destination_start,
              copied(~Element,call,input_tree,output_tree,input_certificate,output_certificate)),
            Equal.cong(StorageCopyLaws.Owners<Element>,StorageCopyLaws.Owners<Element>,owners => model(~Element,rest,(channel + 1 : U32),count,source_stride,destination_stride,source_start,destination_start,owners),
              StorageCopyLaws.observe(~Element,copied(~Element,call,input_tree,output_tree,input_certificate,output_certificate)),
              StorageCopyLaws.model(~Element,count,(source_start + channel * source_stride : U32),(destination_start + channel * destination_stride : U32),StorageCopyLaws.Owners{input_tree,input_length,output_tree,output_length}),
              StorageCopyLaws.packed(~Element,call,input_tree,output_tree,input_certificate,output_certificate))))(input_certificate)(output_certificate)

# Combine the affine execution refinement with the complete store program.
# The original Buffer certificates supply the trees; no readback is assumed.
def packed_writes(~Element: Data,+remaining: Nat,+channel: U32,+count: U32,+source_stride: U32,+destination_stride: U32,+source_start: U32,+destination_start: U32,
  +input: Buffer.Model<Element>,+output: Buffer.Model<Element>) ->
  {StorageCopyLaws.observe(~Element,Bands.planes(~Element,remaining,channel,count,source_stride,destination_stride,source_start,destination_start,
      (Buffer.pack(~Element,input),Buffer.pack(~Element,output)))) ==
    StorageCopyLaws.Owners{Buffer.tree(~Element,input),Buffer.length(~Element,input),
      Writes.pure(Element,writes(~Element,remaining,channel,count,source_stride,destination_stride,source_start,destination_start,Buffer.tree(~Element,input)),Buffer.tree(~Element,output)),
      Buffer.length(~Element,output)} : StorageCopyLaws.Owners<Element>}:
  Buffer.Model{+input,+input_length,+input_certificate} = input
  Buffer.Model{+output,+output_length,+output_certificate} = output
  %Equal.sym(StorageCopyLaws.Owners<Element>,StorageCopyLaws.observe(~Element,Bands.planes(~Element,remaining,channel,count,source_stride,destination_stride,source_start,destination_start,
      (Storage.Buffer{Tree.pack(Element,input),input_length,input_certificate},Storage.Buffer{Tree.pack(Element,output),output_length,output_certificate}))),
    model(~Element,remaining,channel,count,source_stride,destination_stride,source_start,destination_start,StorageCopyLaws.observe(~Element,
      (Storage.Buffer{Tree.pack(Element,input),input_length,input_certificate},Storage.Buffer{Tree.pack(Element,output),output_length,output_certificate}))),
    planes(~Element,remaining,channel,count,source_stride,destination_stride,source_start,destination_start,
      (Storage.Buffer{Tree.pack(Element,input),input_length,input_certificate},Storage.Buffer{Tree.pack(Element,output),output_length,output_certificate}))) :
    {_ == StorageCopyLaws.Owners{input,input_length,Writes.pure(Element,writes(~Element,remaining,channel,count,source_stride,destination_stride,source_start,destination_start,input),output),output_length} : StorageCopyLaws.Owners<Element>}
  %Equal.sym(StorageCopyLaws.Owners<Element>,StorageCopyLaws.observe(~Element,
      (Storage.Buffer{Tree.pack(Element,input),input_length,input_certificate},Storage.Buffer{Tree.pack(Element,output),output_length,output_certificate})),
    StorageCopyLaws.Owners{input,input_length,output,output_length},StorageCopyLaws.represented_values(~Element,input,input_length,output,output_length)) :
    {model(~Element,remaining,channel,count,source_stride,destination_stride,source_start,destination_start,_) ==
      StorageCopyLaws.Owners{input,input_length,Writes.pure(Element,writes(~Element,remaining,channel,count,source_stride,destination_stride,source_start,destination_start,input),output),output_length} : StorageCopyLaws.Owners<Element>}
  model_writes(~Element,remaining,channel,count,source_stride,destination_stride,source_start,destination_start,input,input_length,output,output_length)

def empty_writes(~Element: Data,remaining: Nat,+channel: U32,+source_stride: U32,+destination_stride: U32,+source_start: U32,+destination_start: U32,+input: Tree.Tree<Element>) ->
  {writes(~Element,remaining,channel,0,source_stride,destination_stride,source_start,destination_start,input) == [] : List<&2,Writes.Assignment<Element>>}:
  match remaining:
    case 0n: {==}
    case 1n+rest: empty_writes(~Element,rest,(channel + 1 : U32),source_stride,destination_stride,source_start,destination_start,input)

# Readback bounds for the concrete per-channel copy schedule. A region names
# an allocation base and a subrange repeated in each complete channel plane.
type Region is Data:
  Region{base: Nat,readback_offset: Nat,channels: Nat,count: U32,stride: U32,destination: U32,capacity: U32}

def base(region: Region) -> Nat:
  Region{base,readback_offset,channels,count,stride,destination,capacity} = region
  base

def readback_offset(region: Region) -> Nat:
  Region{base,readback_offset,channels,count,stride,destination,capacity} = region
  readback_offset

def channels(region: Region) -> Nat:
  Region{base,readback_offset,channels,count,stride,destination,capacity} = region
  channels

def count(region: Region) -> U32:
  Region{base,readback_offset,channels,count,stride,destination,capacity} = region
  count

def stride(region: Region) -> U32:
  Region{base,readback_offset,channels,count,stride,destination,capacity} = region
  stride

def destination(region: Region) -> U32:
  Region{base,readback_offset,channels,count,stride,destination,capacity} = region
  destination

def capacity(region: Region) -> U32:
  Region{base,readback_offset,channels,count,stride,destination,capacity} = region
  capacity

type Valid<-region: Region> is Data:
  Valid{readback_capacity_ok: {U32.is_le(capacity(region),2147483648) == True{} : Bool},
    channels_fit: {Nat.is_le(channels(region),U32.to_nat(capacity(region))) == True{} : Bool},
    start_exact: {U32.to_nat(destination(region)) == Nat.add(base(region),readback_offset(region)) : Nat},
    span: {Nat.is_le(Nat.add(readback_offset(region),U32.to_nat(count(region))),U32.to_nat(stride(region))) == True{} : Bool},
    room: {Nat.is_le(Nat.add(base(region),Nat.mul(channels(region),U32.to_nat(stride(region)))),U32.to_nat(capacity(region))) == True{} : Bool}}

def whole_region(channels: Nat,+plane: U32,capacity: U32) -> Region:
  Region{0n,0n,channels,plane,plane,0,capacity}

def whole_valid(+channels: Nat,+plane: U32,+capacity: U32,readback_capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  positive: {Nat.is_lt(0n,U32.to_nat(plane)) == True{} : Bool},+room: {Nat.is_le(Nat.mul(channels,U32.to_nat(plane)),U32.to_nat(capacity)) == True{} : Bool}) ->
  Valid<whole_region(channels,plane,capacity)>:
  Valid{readback_capacity_ok,Order.transitive(channels,Nat.mul(channels,U32.to_nat(plane)),U32.to_nat(capacity),RowMajor.product_covers(channels,U32.to_nat(plane),positive),room),
    {==},Order.reflexive(U32.to_nat(plane)),room}

def start(+region: Region,channel: Nat) -> Nat:
  Strided.start(base(region),readback_offset(region),U32.to_nat(stride(region)),channel)

def native_start(+region: Region,channel: Nat) -> U32:
  (destination(region) + U32.from_nat(channel) * stride(region) : U32)

def end_inside(+region: Region,+channel: Nat,valid: Valid<region>,inside: {Nat.is_lt(channel,channels(region)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(start(region,channel),U32.to_nat(count(region))),U32.to_nat(capacity(region))) == True{} : Bool}:
  Valid{readback_capacity_ok,channels_fit,start_exact,span,room} = valid
  Strided.end_inside(base(region),readback_offset(region),U32.to_nat(stride(region)),U32.to_nat(count(region)),channels(region),channel,U32.to_nat(capacity(region)),inside,span,room)

def start_exact(+region: Region,+channel: Nat,+valid: Valid<region>,+inside: {Nat.is_lt(channel,channels(region)) == True{} : Bool}) ->
  {U32.to_nat(native_start(region,channel)) == start(region,channel) : Nat}:
  Valid{readback_capacity_ok,channels_fit,+destination_exact,span,room} = valid
  %destination_exact : {U32.to_nat(native_start(region,channel)) == Nat.add(_,Nat.mul(channel,U32.to_nat(stride(region)))) : Nat}
  Strided.native_start(destination(region),stride(region),channel,capacity(region),readback_capacity_ok,
    Order.transitive(channel,channels(region),U32.to_nat(capacity(region)),Bounds.strict_le(channel,channels(region),inside),channels_fit),
    %Equal.sym(Nat,U32.to_nat(destination(region)),Nat.add(base(region),readback_offset(region)),destination_exact) :
      {Nat.is_le(Nat.add(_,Nat.mul(channel,U32.to_nat(stride(region)))),U32.to_nat(capacity(region))) == True{} : Bool}
    Order.transitive(start(region,channel),Nat.add(start(region,channel),U32.to_nat(count(region))),U32.to_nat(capacity(region)),Order.le_add(start(region,channel),U32.to_nat(count(region))),end_inside(region,channel,valid,inside)))

def copy_room(+region: Region,+channel: Nat,+valid: Valid<region>,+inside: {Nat.is_lt(channel,channels(region)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.add(U32.to_nat(native_start(region,channel)),0n),U32.to_nat(count(region))),U32.to_nat(capacity(region))) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(U32.to_nat(native_start(region,channel)),0n),U32.to_nat(native_start(region,channel)),Addition.zero_right(U32.to_nat(native_start(region,channel)))) :
    {Nat.is_le(Nat.add(_,U32.to_nat(count(region))),U32.to_nat(capacity(region))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(native_start(region,channel)),start(region,channel),start_exact(region,channel,valid,inside)) :
    {Nat.is_le(Nat.add(_,U32.to_nat(count(region))),U32.to_nat(capacity(region))) == True{} : Bool}
  end_inside(region,channel,valid,inside)

def readback_capacity_ok(+region: Region,valid: Valid<region>) -> {U32.is_le(capacity(region),2147483648) == True{} : Bool}:
  Valid{readback_capacity_ok,channels_fit,start_exact,span,room} = valid
  readback_capacity_ok

def span(+region: Region,valid: Valid<region>) -> {Nat.is_le(Nat.add(readback_offset(region),U32.to_nat(count(region))),U32.to_nat(stride(region))) == True{} : Bool}:
  Valid{readback_capacity_ok,channels_fit,start_exact,span,room} = valid
  span

def readback_query(+region: Region,channel: Nat,index: Nat) -> U32:
  (native_start(region,channel) + U32.from_nat(index) : U32)

def query_inside(+region: Region,+channel: Nat,+index: Nat,+valid: Valid<region>,channel_inside: {Nat.is_lt(channel,channels(region)) == True{} : Bool},
  index_inside: {Nat.is_lt(index,U32.to_nat(count(region))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(start(region,channel),index),U32.to_nat(capacity(region))) == True{} : Bool}:
  Valid{readback_capacity_ok,channels_fit,start_exact,span,room} = valid
  Strided.query_inside(base(region),readback_offset(region),U32.to_nat(stride(region)),U32.to_nat(count(region)),channels(region),channel,index,U32.to_nat(capacity(region)),channel_inside,index_inside,span,room)

def readback_query_exact(+region: Region,+channel: Nat,+index: Nat,+valid: Valid<region>,+channel_inside: {Nat.is_lt(channel,channels(region)) == True{} : Bool},
  index_inside: {Nat.is_lt(index,U32.to_nat(count(region))) == True{} : Bool}) ->
  {U32.to_nat(readback_query(region,channel,index)) == Nat.add(start(region,channel),index) : Nat}:
  %start_exact(region,channel,valid,channel_inside) : {U32.to_nat(readback_query(region,channel,index)) == Nat.add(_,index) : Nat}
  CopyReadback.offset_exact(native_start(region,channel),index,capacity(region),readback_capacity_ok(region,valid),
    %Equal.sym(Nat,U32.to_nat(native_start(region,channel)),start(region,channel),start_exact(region,channel,valid,channel_inside)) :
      {Nat.is_le(Nat.add(_,index),U32.to_nat(capacity(region))) == True{} : Bool}
    Bounds.strict_le(Nat.add(start(region,channel),index),U32.to_nat(capacity(region)),query_inside(region,channel,index,valid,channel_inside,index_inside)))

def query_word_inside(+region: Region,+channel: Nat,+index: Nat,+valid: Valid<region>,+channel_inside: {Nat.is_lt(channel,channels(region)) == True{} : Bool},
  index_inside: {Nat.is_lt(index,U32.to_nat(count(region))) == True{} : Bool}) -> {U32.is_lt(readback_query(region,channel,index),capacity(region)) == True{} : Bool}:
  CopyReadback.offset_inside(native_start(region,channel),index,capacity(region),readback_capacity_ok(region,valid),
    %Equal.sym(Nat,U32.to_nat(native_start(region,channel)),start(region,channel),start_exact(region,channel,valid,channel_inside)) :
      {Nat.is_lt(Nat.add(_,index),U32.to_nat(capacity(region))) == True{} : Bool}
    query_inside(region,channel,index,valid,channel_inside,index_inside))

def query_before(+region: Region,+target: Nat,+later: Nat,+index: Nat,+valid: Valid<region>,+target_inside: {Nat.is_lt(target,channels(region)) == True{} : Bool},
  +later_inside: {Nat.is_lt(later,channels(region)) == True{} : Bool},+index_inside: {Nat.is_lt(index,U32.to_nat(count(region))) == True{} : Bool},
  ordered: {Nat.is_lt(target,later) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(readback_query(region,target,index)),Nat.add(U32.to_nat(native_start(region,later)),0n)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(U32.to_nat(native_start(region,later)),0n),U32.to_nat(native_start(region,later)),Addition.zero_right(U32.to_nat(native_start(region,later)))) :
    {Nat.is_lt(U32.to_nat(readback_query(region,target,index)),_) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(native_start(region,later)),start(region,later),start_exact(region,later,valid,later_inside)) :
    {Nat.is_lt(U32.to_nat(readback_query(region,target,index)),_) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(readback_query(region,target,index)),Nat.add(start(region,target),index),readback_query_exact(region,target,index,valid,target_inside,index_inside)) :
    {Nat.is_lt(_,start(region,later)) == True{} : Bool}
  Strided.query_before(base(region),readback_offset(region),U32.to_nat(stride(region)),U32.to_nat(count(region)),target,later,index,ordered,index_inside,span(region,valid))

def writes_inside(~Element: Data,remaining: Nat,+channel: Nat,+region: Region,+source_stride: U32,+source_start: U32,+input: Tree.Tree<Element>,+valid: Valid<region>,
  +room: {Nat.is_le(Nat.add(channel,remaining),channels(region)) == True{} : Bool}) ->
  Readback.Inside(~Element,capacity(region),writes(~Element,remaining,U32.from_nat(channel),count(region),source_stride,stride(region),source_start,destination(region),input)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      +inside = Order.room_index(channel,rest,channels(region),room)
      Readback.inside_append(~Element,capacity(region),plane_writes(~Element,U32.from_nat(channel),count(region),source_stride,stride(region),source_start,destination(region),input),
        writes(~Element,rest,(U32.from_nat(channel) + 1 : U32),count(region),source_stride,stride(region),source_start,destination(region),input),
        CopyReadback.writes_inside(~Element,U32.to_nat(count(region)),0n,input,(source_start + U32.from_nat(channel) * source_stride : U32),native_start(region,channel),capacity(region),readback_capacity_ok(region,valid),copy_room(region,channel,valid,inside)),
        %Equal.sym(U32,(U32.from_nat(channel) + 1 : U32),U32.from_nat(1n+channel),Bounds.add_one(U32.from_nat(channel))) :
          Readback.Inside(~Element,capacity(region),writes(~Element,rest,_,count(region),source_stride,stride(region),source_start,destination(region),input))
        writes_inside(~Element,rest,1n+channel,region,source_stride,source_start,input,valid,Order.advance_room(channel,rest,channels(region),room)))

def selected_later(~Element: Data,remaining: Nat,+channel: Nat,+target: Nat,+index: Nat,+region: Region,+source_stride: U32,+source_start: U32,+input: Tree.Tree<Element>,+valid: Valid<region>,
  +room: {Nat.is_le(Nat.add(channel,remaining),channels(region)) == True{} : Bool},+target_inside: {Nat.is_lt(target,channels(region)) == True{} : Bool},
  +index_inside: {Nat.is_lt(index,U32.to_nat(count(region))) == True{} : Bool},+before: {Nat.is_lt(target,channel) == True{} : Bool},+previous: Element) ->
  {Readback.selected(~Element,writes(~Element,remaining,U32.from_nat(channel),count(region),source_stride,stride(region),source_start,destination(region),input),readback_query(region,target,index),previous) == previous : Element}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +here = plane_writes(~Element,U32.from_nat(channel),count(region),source_stride,stride(region),source_start,destination(region),input)
      +later = writes(~Element,rest,(U32.from_nat(channel) + 1 : U32),count(region),source_stride,stride(region),source_start,destination(region),input)
      +inside = Order.room_index(channel,rest,channels(region),room)
      %Equal.sym(Element,Readback.selected(~Element,List.append(&2,Writes.Assignment<Element>,here,later),readback_query(region,target,index),previous),
        Readback.selected(~Element,later,readback_query(region,target,index),Readback.selected(~Element,here,readback_query(region,target,index),previous)),Readback.selected_append(~Element,here,later,readback_query(region,target,index),previous)) : {_ == previous : Element}
      %Equal.sym(Element,Readback.selected(~Element,here,readback_query(region,target,index),previous),previous,
        CopyReadback.selected_outside(~Element,U32.to_nat(count(region)),0n,readback_query(region,target,index),input,(source_start + U32.from_nat(channel) * source_stride : U32),native_start(region,channel),capacity(region),
          readback_capacity_ok(region,valid),copy_room(region,channel,valid,inside),Interval.Before{query_before(region,target,channel,index,valid,target_inside,inside,index_inside,before)},previous)) :
        {Readback.selected(~Element,later,readback_query(region,target,index),_) == previous : Element}
      %Equal.sym(U32,(U32.from_nat(channel) + 1 : U32),U32.from_nat(1n+channel),Bounds.add_one(U32.from_nat(channel))) :
        {Readback.selected(~Element,writes(~Element,rest,_,count(region),source_stride,stride(region),source_start,destination(region),input),readback_query(region,target,index),previous) == previous : Element}
      selected_later(~Element,rest,1n+channel,target,index,region,source_stride,source_start,input,valid,Order.advance_room(channel,rest,channels(region),room),target_inside,index_inside,
        Order.lt_succ(target,channel,Bounds.strict_le(target,channel,before)),previous)

def sample(~Element: Data,input: Tree.Tree<Element>,source_stride: U32,source_start: U32,target: Nat,index: Nat) -> Element:
  Tree.read(Element,input,((source_start + U32.from_nat(target) * source_stride : U32) + U32.from_nat(index) : U32))

def selected_inside(~Element: Data,remaining: Nat,+channel: Nat,target: Nat,+index: Nat,+region: Region,+source_stride: U32,+source_start: U32,+input: Tree.Tree<Element>,+valid: Valid<region>,
  +room: {Nat.is_le(Nat.add(channel,remaining),channels(region)) == True{} : Bool},target_inside: {Nat.is_lt(target,remaining) == True{} : Bool},
  +index_inside: {Nat.is_lt(index,U32.to_nat(count(region))) == True{} : Bool},+previous: Element) ->
  {Readback.selected(~Element,writes(~Element,remaining,U32.from_nat(channel),count(region),source_stride,stride(region),source_start,destination(region),input),readback_query(region,Nat.add(channel,target),index),previous) ==
    sample(~Element,input,source_stride,source_start,Nat.add(channel,target),index) : Element}:
  match remaining target:
    case 0n +target:
      Empty.absurd({previous == sample(~Element,input,source_stride,source_start,Nat.add(channel,target),index) : Element},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(target,0n),True{},Equal.sym(Bool,Nat.is_lt(target,0n),False{},Division.lt_zero(target)),target_inside)))
    case 1n+ +rest 0n:
      +here = plane_writes(~Element,U32.from_nat(channel),count(region),source_stride,stride(region),source_start,destination(region),input)
      +later = writes(~Element,rest,(U32.from_nat(channel) + 1 : U32),count(region),source_stride,stride(region),source_start,destination(region),input)
      +inside = Order.room_index(channel,rest,channels(region),room)
      %Equal.sym(Nat,Nat.add(channel,0n),channel,Addition.zero_right(channel)) :
        {Readback.selected(~Element,List.append(&2,Writes.Assignment<Element>,here,later),readback_query(region,_,index),previous) == sample(~Element,input,source_stride,source_start,_,index) : Element}
      %Equal.sym(Element,Readback.selected(~Element,List.append(&2,Writes.Assignment<Element>,here,later),readback_query(region,channel,index),previous),
        Readback.selected(~Element,later,readback_query(region,channel,index),Readback.selected(~Element,here,readback_query(region,channel,index),previous)),Readback.selected_append(~Element,here,later,readback_query(region,channel,index),previous)) :
        {_ == sample(~Element,input,source_stride,source_start,channel,index) : Element}
      %Equal.sym(Element,Readback.selected(~Element,here,readback_query(region,channel,index),previous),sample(~Element,input,source_stride,source_start,channel,index),
        CopyReadback.selected_inside(~Element,U32.to_nat(count(region)),0n,index,input,(source_start + U32.from_nat(channel) * source_stride : U32),native_start(region,channel),capacity(region),
          readback_capacity_ok(region,valid),copy_room(region,channel,valid,inside),index_inside,previous)) :
        {Readback.selected(~Element,later,readback_query(region,channel,index),_) == sample(~Element,input,source_stride,source_start,channel,index) : Element}
      %Equal.sym(U32,(U32.from_nat(channel) + 1 : U32),U32.from_nat(1n+channel),Bounds.add_one(U32.from_nat(channel))) :
        {Readback.selected(~Element,writes(~Element,rest,_,count(region),source_stride,stride(region),source_start,destination(region),input),readback_query(region,channel,index),sample(~Element,input,source_stride,source_start,channel,index)) ==
          sample(~Element,input,source_stride,source_start,channel,index) : Element}
      selected_later(~Element,rest,1n+channel,channel,index,region,source_stride,source_start,input,valid,Order.advance_room(channel,rest,channels(region),room),inside,index_inside,
        Order.lt_succ(channel,channel,Order.reflexive(channel)),sample(~Element,input,source_stride,source_start,channel,index))
    case 1n+ +rest 1n+ +target:
      +here = plane_writes(~Element,U32.from_nat(channel),count(region),source_stride,stride(region),source_start,destination(region),input)
      +later = writes(~Element,rest,(U32.from_nat(channel) + 1 : U32),count(region),source_stride,stride(region),source_start,destination(region),input)
      %Equal.sym(Element,Readback.selected(~Element,List.append(&2,Writes.Assignment<Element>,here,later),readback_query(region,Nat.add(channel,1n+target),index),previous),
        Readback.selected(~Element,later,readback_query(region,Nat.add(channel,1n+target),index),Readback.selected(~Element,here,readback_query(region,Nat.add(channel,1n+target),index),previous)),
        Readback.selected_append(~Element,here,later,readback_query(region,Nat.add(channel,1n+target),index),previous)) :
        {_ == sample(~Element,input,source_stride,source_start,Nat.add(channel,1n+target),index) : Element}
      %Equal.sym(U32,(U32.from_nat(channel) + 1 : U32),U32.from_nat(1n+channel),Bounds.add_one(U32.from_nat(channel))) :
        {Readback.selected(~Element,writes(~Element,rest,_,count(region),source_stride,stride(region),source_start,destination(region),input),readback_query(region,Nat.add(channel,1n+target),index),
          Readback.selected(~Element,here,readback_query(region,Nat.add(channel,1n+target),index),previous)) == sample(~Element,input,source_stride,source_start,Nat.add(channel,1n+target),index) : Element}
      %Equal.sym(Nat,Nat.add(channel,1n+target),Nat.add(1n+channel,target),Addition.successor_right(channel,target)) :
        {Readback.selected(~Element,writes(~Element,rest,U32.from_nat(1n+channel),count(region),source_stride,stride(region),source_start,destination(region),input),readback_query(region,_,index),
          Readback.selected(~Element,here,readback_query(region,_,index),previous)) == sample(~Element,input,source_stride,source_start,_,index) : Element}
      selected_inside(~Element,rest,1n+channel,target,index,region,source_stride,source_start,input,valid,Order.advance_room(channel,rest,channels(region),room),target_inside,index_inside,
        Readback.selected(~Element,here,readback_query(region,Nat.add(1n+channel,target),index),previous))

def read_inside(~Element: Data,+remaining: Nat,+channel: Nat,+target: Nat,+index: Nat,+region: Region,+source_stride: U32,+source_start: U32,
  +input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+depth: Nat,+balanced: Shape.Balanced(Element,depth,output),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_equal: {Tree.capacity(Element,output) == capacity(region) : U32},+valid: Valid<region>,
  +room: {Nat.is_le(Nat.add(channel,remaining),channels(region)) == True{} : Bool},+target_inside: {Nat.is_lt(target,remaining) == True{} : Bool},
  +index_inside: {Nat.is_lt(index,U32.to_nat(count(region))) == True{} : Bool}) ->
  {Tree.read(Element,Writes.pure(Element,writes(~Element,remaining,U32.from_nat(channel),count(region),source_stride,stride(region),source_start,destination(region),input),output),readback_query(region,Nat.add(channel,target),index)) ==
    sample(~Element,input,source_stride,source_start,Nat.add(channel,target),index) : Element}:
  +program = writes(~Element,remaining,U32.from_nat(channel),count(region),source_stride,stride(region),source_start,destination(region),input)
  +address = readback_query(region,Nat.add(channel,target),index)
  +channel_inside = Bounds.nat_trans(Nat.add(channel,target),Nat.add(channel,remaining),channels(region),Order.offset_lt(channel,target,remaining,target_inside),room)
  Equal.trans(Element,Tree.read(Element,Writes.pure(Element,program,output),address),Readback.selected(~Element,program,address,Tree.read(Element,output,address)),
    sample(~Element,input,source_stride,source_start,Nat.add(channel,target),index),
    Readback.program(~Element,program,depth,output,address,balanced,depth_ok,
      %Equal.sym(U32,Tree.capacity(Element,output),capacity(region),capacity_equal) : {U32.is_lt(address,_) == True{} : Bool}
      query_word_inside(region,Nat.add(channel,target),index,valid,channel_inside,index_inside),
      %Equal.sym(U32,Tree.capacity(Element,output),capacity(region),capacity_equal) : Readback.Inside(~Element,_,program)
      writes_inside(~Element,remaining,channel,region,source_stride,source_start,input,valid,room)),
    selected_inside(~Element,remaining,channel,target,index,region,source_stride,source_start,input,valid,room,target_inside,index_inside,Tree.read(Element,output,address)))

# Separation is stated in logical intervals, not as an assumed readback law.
# Gather/range callers obtain these intervals from their row partition.
law Outside:
  for remaining: Nat
  for channel: Nat
  for region: Region
  for readback_query: U32
  Data

def Outside(remaining,channel,region,readback_query):
  match remaining:
    case 0n: Unit
    case 1n+rest:
      Shape.Both<Interval.Outside<start(region,channel),U32.to_nat(count(region)),U32.to_nat(readback_query)>,Outside(rest,1n+channel,region,readback_query)>

def native_outside(+region: Region,+channel: Nat,+readback_query: U32,+valid: Valid<region>,+inside: {Nat.is_lt(channel,channels(region)) == True{} : Bool},
  outside: Interval.Outside<start(region,channel),U32.to_nat(count(region)),U32.to_nat(readback_query)>) ->
  Interval.Outside<Nat.add(U32.to_nat(native_start(region,channel)),0n),U32.to_nat(count(region)),U32.to_nat(readback_query)>:
  %Equal.sym(Nat,Nat.add(U32.to_nat(native_start(region,channel)),0n),U32.to_nat(native_start(region,channel)),Addition.zero_right(U32.to_nat(native_start(region,channel)))) :
    Interval.Outside<_,U32.to_nat(count(region)),U32.to_nat(readback_query)>
  %Equal.sym(Nat,U32.to_nat(native_start(region,channel)),start(region,channel),start_exact(region,channel,valid,inside)) :
    Interval.Outside<_,U32.to_nat(count(region)),U32.to_nat(readback_query)>
  outside

def selected_outside(~Element: Data,remaining: Nat,+channel: Nat,+region: Region,+source_stride: U32,+source_start: U32,+input: Tree.Tree<Element>,+readback_query: U32,+previous: Element,+valid: Valid<region>,
  +room: {Nat.is_le(Nat.add(channel,remaining),channels(region)) == True{} : Bool},outside: Outside(remaining,channel,region,readback_query)) ->
  {Readback.selected(~Element,writes(~Element,remaining,U32.from_nat(channel),count(region),source_stride,stride(region),source_start,destination(region),input),readback_query,previous) == previous : Element}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      Shape.Both{here_outside,later_outside} = outside
      +here = plane_writes(~Element,U32.from_nat(channel),count(region),source_stride,stride(region),source_start,destination(region),input)
      +later = writes(~Element,rest,(U32.from_nat(channel) + 1 : U32),count(region),source_stride,stride(region),source_start,destination(region),input)
      +inside = Order.room_index(channel,rest,channels(region),room)
      %Equal.sym(Element,Readback.selected(~Element,List.append(&2,Writes.Assignment<Element>,here,later),readback_query,previous),
        Readback.selected(~Element,later,readback_query,Readback.selected(~Element,here,readback_query,previous)),Readback.selected_append(~Element,here,later,readback_query,previous)) : {_ == previous : Element}
      %Equal.sym(Element,Readback.selected(~Element,here,readback_query,previous),previous,
        CopyReadback.selected_outside(~Element,U32.to_nat(count(region)),0n,readback_query,input,(source_start + U32.from_nat(channel) * source_stride : U32),native_start(region,channel),capacity(region),
          readback_capacity_ok(region,valid),copy_room(region,channel,valid,inside),native_outside(region,channel,readback_query,valid,inside,here_outside),previous)) :
        {Readback.selected(~Element,later,readback_query,_) == previous : Element}
      %Equal.sym(U32,(U32.from_nat(channel) + 1 : U32),U32.from_nat(1n+channel),Bounds.add_one(U32.from_nat(channel))) :
        {Readback.selected(~Element,writes(~Element,rest,_,count(region),source_stride,stride(region),source_start,destination(region),input),readback_query,previous) == previous : Element}
      selected_outside(~Element,rest,1n+channel,region,source_stride,source_start,input,readback_query,previous,valid,Order.advance_room(channel,rest,channels(region),room),later_outside)

def read_outside(~Element: Data,+remaining: Nat,+channel: Nat,+region: Region,+source_stride: U32,+source_start: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+readback_query: U32,
  +depth: Nat,+balanced: Shape.Balanced(Element,depth,output),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_equal: {Tree.capacity(Element,output) == capacity(region) : U32},+valid: Valid<region>,+query_inside: {U32.is_lt(readback_query,capacity(region)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(channel,remaining),channels(region)) == True{} : Bool},+outside: Outside(remaining,channel,region,readback_query)) ->
  {Tree.read(Element,Writes.pure(Element,writes(~Element,remaining,U32.from_nat(channel),count(region),source_stride,stride(region),source_start,destination(region),input),output),readback_query) == Tree.read(Element,output,readback_query) : Element}:
  +program = writes(~Element,remaining,U32.from_nat(channel),count(region),source_stride,stride(region),source_start,destination(region),input)
  Equal.trans(Element,Tree.read(Element,Writes.pure(Element,program,output),readback_query),Readback.selected(~Element,program,readback_query,Tree.read(Element,output,readback_query)),Tree.read(Element,output,readback_query),
    Readback.program(~Element,program,depth,output,readback_query,balanced,depth_ok,
      %Equal.sym(U32,Tree.capacity(Element,output),capacity(region),capacity_equal) : {U32.is_lt(readback_query,_) == True{} : Bool} query_inside,
      %Equal.sym(U32,Tree.capacity(Element,output),capacity(region),capacity_equal) : Readback.Inside(~Element,_,program)
      writes_inside(~Element,remaining,channel,region,source_stride,source_start,input,valid,room)),
    selected_outside(~Element,remaining,channel,region,source_stride,source_start,input,readback_query,Tree.read(Element,output,readback_query),valid,room,outside))

def covered(~Element: Data,remaining: Nat,+channel: Nat,target: Nat,+index: Nat,+region: Region,+source_stride: U32,+source_start: U32,+input: Tree.Tree<Element>,
  target_inside: {Nat.is_lt(target,remaining) == True{} : Bool},+index_inside: {Nat.is_lt(index,U32.to_nat(count(region))) == True{} : Bool}) ->
  {Readback.contains(~Element,readback_query(region,Nat.add(channel,target),index),writes(~Element,remaining,U32.from_nat(channel),count(region),source_stride,stride(region),source_start,destination(region),input)) == True{} : Bool}:
  match remaining target:
    case 0n +target:
      Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(target,0n),True{},Equal.sym(Bool,Nat.is_lt(target,0n),False{},Division.lt_zero(target)),target_inside)))
    case 1n+ +rest 0n:
      %Equal.sym(Nat,Nat.add(channel,0n),channel,Addition.zero_right(channel)) :
        {Readback.contains(~Element,readback_query(region,_,index),writes(~Element,1n+rest,U32.from_nat(channel),count(region),source_stride,stride(region),source_start,destination(region),input)) == True{} : Bool}
      Readback.covered_left(~Element,readback_query(region,channel,index),plane_writes(~Element,U32.from_nat(channel),count(region),source_stride,stride(region),source_start,destination(region),input),
        writes(~Element,rest,(U32.from_nat(channel) + 1 : U32),count(region),source_stride,stride(region),source_start,destination(region),input),
        CopyReadback.covered(~Element,U32.to_nat(count(region)),0n,index,input,(source_start + U32.from_nat(channel) * source_stride : U32),native_start(region,channel),index_inside))
    case 1n+ +rest 1n+ +target:
      Readback.covered_right(~Element,readback_query(region,Nat.add(channel,1n+target),index),plane_writes(~Element,U32.from_nat(channel),count(region),source_stride,stride(region),source_start,destination(region),input),
        writes(~Element,rest,(U32.from_nat(channel) + 1 : U32),count(region),source_stride,stride(region),source_start,destination(region),input),
        %Equal.sym(U32,(U32.from_nat(channel) + 1 : U32),U32.from_nat(1n+channel),Bounds.add_one(U32.from_nat(channel))) :
          {Readback.contains(~Element,readback_query(region,Nat.add(channel,1n+target),index),writes(~Element,rest,_,count(region),source_stride,stride(region),source_start,destination(region),input)) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(channel,1n+target),Nat.add(1n+channel,target),Addition.successor_right(channel,target)) :
          {Readback.contains(~Element,readback_query(region,_,index),writes(~Element,rest,U32.from_nat(1n+channel),count(region),source_stride,stride(region),source_start,destination(region),input)) == True{} : Bool}
        covered(~Element,rest,1n+channel,target,index,region,source_stride,source_start,input,target_inside,index_inside))

# Scale a bounded row slice into the exact per-channel copy region. The base
# also covers concat's destination offset without changing field meanings.
def region(+base: U32,+first: U32,+rows: U32,channels: U32,+height: U32,+width: U32,capacity: U32) -> Region:
  Region{U32.to_nat(base),Nat.mul(U32.to_nat(first),U32.to_nat(width)),U32.to_nat(channels),(rows * width : U32),(height * width : U32),(base + first * width : U32),capacity}

type Facts<-base: U32,-first: U32,-rows: U32,-channels: U32,-height: U32,-width: U32,-capacity: U32> is Data:
  Facts{valid: Valid<region(base,first,rows,channels,height,width,capacity)>,
    stride_exact: {U32.to_nat((height * width : U32)) == Nat.mul(U32.to_nat(height),U32.to_nat(width)) : Nat},
    count_exact: {U32.to_nat((rows * width : U32)) == Nat.mul(U32.to_nat(rows),U32.to_nat(width)) : Nat},
    offset_exact: {U32.to_nat((first * width : U32)) == Nat.mul(U32.to_nat(first),U32.to_nat(width)) : Nat}}

def bounded(+base: U32,+first: U32,+rows: U32,+channels: U32,+height: U32,+width: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},channels_positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool},
  height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
  row_span: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(height)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(base),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(capacity)) == True{} : Bool}) ->
  Facts<base,first,rows,channels,height,width,capacity>:
  +plane = Nat.mul(U32.to_nat(height),U32.to_nat(width))
  +frame = Nat.mul(U32.to_nat(channels),plane)
  +offset = Nat.mul(U32.to_nat(first),U32.to_nat(width))
  +count = Nat.mul(U32.to_nat(rows),U32.to_nat(width))
  +frame_room = Strided.count_in_stride(U32.to_nat(base),frame,U32.to_nat(capacity),room)
  +plane_frame = {
    %RowMajor.multiply_commutative(plane,U32.to_nat(channels)) : {Nat.is_le(plane,_) == True{} : Bool}
    RowMajor.product_covers(plane,U32.to_nat(channels),channels_positive) : {Nat.is_le(plane,frame) == True{} : Bool}}
  +plane_room = Order.transitive(plane,frame,U32.to_nat(capacity),plane_frame,frame_room)
  +span = Strided.scaled_span(U32.to_nat(first),U32.to_nat(rows),U32.to_nat(height),U32.to_nat(width),row_span)
  +offset_plane = Order.transitive(offset,Nat.add(offset,count),plane,Order.le_add(offset,count),span)
  +count_exact = Arithmetic.multiply(rows,width,capacity,capacity_ok,Order.transitive(count,plane,U32.to_nat(capacity),Strided.count_in_stride(offset,count,plane,span),plane_room))
  +plane_exact = Arithmetic.multiply(height,width,capacity,capacity_ok,plane_room)
  +offset_exact = Arithmetic.multiply(first,width,capacity,capacity_ok,Order.transitive(offset,plane,U32.to_nat(capacity),offset_plane,plane_room))
  +start_exact = {
    %offset_exact : {U32.to_nat((base + first * width : U32)) == Nat.add(U32.to_nat(base),_) : Nat}
    Arithmetic.add(base,(first * width : U32),capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat((first * width : U32)),offset,offset_exact) : {Nat.is_le(Nat.add(U32.to_nat(base),_),U32.to_nat(capacity)) == True{} : Bool}
      Order.transitive(Nat.add(U32.to_nat(base),offset),Nat.add(U32.to_nat(base),frame),U32.to_nat(capacity),
        Order.offset_le(U32.to_nat(base),offset,frame,Order.transitive(offset,plane,frame,offset_plane,plane_frame)),room)) : {U32.to_nat((base + first * width : U32)) == Nat.add(U32.to_nat(base),offset) : Nat}}
  Facts{Valid{capacity_ok,
      Order.transitive(U32.to_nat(channels),frame,U32.to_nat(capacity),RowMajor.product_covers(U32.to_nat(channels),plane,RowMajor.multiply_lt(0n,U32.to_nat(height),U32.to_nat(width),height_positive,width_positive)),frame_room),start_exact,
      %Equal.sym(Nat,U32.to_nat((rows * width : U32)),count,count_exact) : {Nat.is_le(Nat.add(offset,_),U32.to_nat((height * width : U32))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat((height * width : U32)),plane,plane_exact) : {Nat.is_le(Nat.add(offset,count),_) == True{} : Bool} span,
      %Equal.sym(Nat,U32.to_nat((height * width : U32)),plane,plane_exact) : {Nat.is_le(Nat.add(U32.to_nat(base),Nat.mul(U32.to_nat(channels),_)),U32.to_nat(capacity)) == True{} : Bool} room},plane_exact,count_exact,offset_exact}

def pixel_query(+base: U32,+first: U32,+rows: U32,+channels: U32,+height: U32,+width: U32,+capacity: U32,+channel: Nat,+row: Nat,+column: Nat,
  facts: Facts<base,first,rows,channels,height,width,capacity>,+channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},column_inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool}) ->
  {U32.to_nat(readback_query(region(base,first,rows,channels,height,width,capacity),channel,Nat.add(Nat.mul(row,U32.to_nat(width)),column))) ==
    Nat.add(U32.to_nat(base),RowMajor.index(channel,U32.to_nat(height),U32.to_nat(width),Nat.add(U32.to_nat(first),row),column)) : Nat}:
  Facts{+valid,+stride_exact,count_exact,offset_exact} = facts
  +region = region(base,first,rows,channels,height,width,capacity)
  +index = Nat.add(Nat.mul(row,U32.to_nat(width)),column)
  +inside = {
    %Equal.sym(Nat,U32.to_nat((rows * width : U32)),Nat.mul(U32.to_nat(rows),U32.to_nat(width)),count_exact) : {Nat.is_lt(index,_) == True{} : Bool}
    RowMajor.radix(row,U32.to_nat(rows),column,U32.to_nat(width),row_inside,column_inside) : {Nat.is_lt(index,U32.to_nat((rows * width : U32))) == True{} : Bool}}
  Equal.trans(Nat,U32.to_nat(readback_query(region,channel,index)),Nat.add(start(region,channel),index),
    Nat.add(U32.to_nat(base),RowMajor.index(channel,U32.to_nat(height),U32.to_nat(width),Nat.add(U32.to_nat(first),row),column)),readback_query_exact(region,channel,index,valid,channel_inside,inside),
    %Equal.sym(Nat,U32.to_nat((height * width : U32)),Nat.mul(U32.to_nat(height),U32.to_nat(width)),stride_exact) :
      {Nat.add(Strided.start(U32.to_nat(base),Nat.mul(U32.to_nat(first),U32.to_nat(width)),_,channel),index) == Nat.add(U32.to_nat(base),RowMajor.index(channel,U32.to_nat(height),U32.to_nat(width),Nat.add(U32.to_nat(first),row),column)) : Nat}
    Equal.trans(Nat,Nat.add(Strided.start(U32.to_nat(base),Nat.mul(U32.to_nat(first),U32.to_nat(width)),Nat.mul(U32.to_nat(height),U32.to_nat(width)),channel),index),
      Nat.add(U32.to_nat(base),RowMajor.index(channel,U32.to_nat(height),U32.to_nat(width),U32.to_nat(first),index)),
      Nat.add(U32.to_nat(base),RowMajor.index(channel,U32.to_nat(height),U32.to_nat(width),Nat.add(U32.to_nat(first),row),column)),
      Strided.end_form(U32.to_nat(base),Nat.mul(U32.to_nat(first),U32.to_nat(width)),Nat.mul(U32.to_nat(height),U32.to_nat(width)),channel,index),
      Equal.cong(Nat,Nat,value => Nat.add(U32.to_nat(base),value),RowMajor.index(channel,U32.to_nat(height),U32.to_nat(width),U32.to_nat(first),index),RowMajor.index(channel,U32.to_nat(height),U32.to_nat(width),Nat.add(U32.to_nat(first),row),column),
        BandCoordinates.row_join(channel,U32.to_nat(height),U32.to_nat(width),U32.to_nat(first),row,column))))

def relative_pixel_query(+base: U32,+first: U32,+rows: U32,+channels: U32,+height: U32,+width: U32,+capacity: U32,+channel: Nat,+row: Nat,+column: Nat,
  +facts: Facts<base,first,rows,channels,height,width,capacity>,+channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool}) ->
  {U32.to_nat((readback_query(region(base,first,rows,channels,height,width,capacity),channel,Nat.add(Nat.mul(row,U32.to_nat(width)),column)) - base : U32)) ==
    RowMajor.index(channel,U32.to_nat(height),U32.to_nat(width),Nat.add(U32.to_nat(first),row),column) : Nat}:
  Facts{+valid,stride_exact,count_exact,offset_exact} = facts
  +region = region(base,first,rows,channels,height,width,capacity)
  +index = Nat.add(Nat.mul(row,U32.to_nat(width)),column)
  +query = readback_query(region,channel,index)
  +inside = {
    %Equal.sym(Nat,U32.to_nat((rows * width : U32)),Nat.mul(U32.to_nat(rows),U32.to_nat(width)),count_exact) : {Nat.is_lt(index,_) == True{} : Bool}
    RowMajor.radix(row,U32.to_nat(rows),column,U32.to_nat(width),row_inside,column_inside) : {Nat.is_lt(index,U32.to_nat((rows * width : U32))) == True{} : Bool}}
  Arithmetic.remove_prefix(query,base,RowMajor.index(channel,U32.to_nat(height),U32.to_nat(width),Nat.add(U32.to_nat(first),row),column),capacity,readback_capacity_ok(region,valid),
    pixel_query(base,first,rows,channels,height,width,capacity,channel,row,column,facts,channel_inside,row_inside,column_inside),
    Bounds.strict_le(U32.to_nat(query),U32.to_nat(capacity),Bounds.as_nat_lt(query,capacity,query_word_inside(region,channel,index,valid,channel_inside,inside))))
