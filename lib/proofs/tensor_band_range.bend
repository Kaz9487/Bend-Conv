# Range copies shared by halo preparation and repartition: the write program,
# its bounds, coverage, coordinates and values.
import Base
import ./tensor_band_copy.bend as BandCopy
import ../tensor_band.bend as Bands
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_model.bend as Model
import ./nat_order.bend as Order
import ./write_program.bend as Writes
import ./word_interval.bend as WordInterval
import ./tensor_band_plan.bend as BandPlan
import ./tensor_band_shape_refinement.bend as BandShape
import ./write_program_readback.bend as Readback
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./storage_shape.bend as Shape
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_values.bend as BandValues
import ../machine_limits.bend as Machine
import ./encoded_coordinates.bend as Encoded
import ./storage_copy_refinement.bend as StorageCopyLaws
import ./strided_range_bounds.bend as Strided
import ../storage_buffer.bend as Storage
import ./contiguous_copy.bend as Contiguous
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./ordered_ranges.bend as Ranges
import ./row_major_coordinates.bend as RowCoordinates
import ./sequence_membership.bend as Members
import ./storage_certificate.bend as Certificate
import ./u32_comparison.bend as Comparison
import ./word_encoding.bend as WordEncoding
import ./word_subtraction.bend as Subtraction
import ../traversal_partition.bend as Partition

# Range extraction and halo preparation share these exact channel writes.
type Content<-Element: Data> is Data:
  Content{source: Model.Content<Element>,result: Buffer.Content<Element>}

def result(~Element: Data,content: Content<Element>) -> Buffer.Content<Element>:
  Content{source,result} = content
  result

def contents(~Element: Data,pair: Bands.Bands(~Element) & Storage.Buffer<Element>) -> Content<Element>:
  (source,result) = pair
  Content{Model.contents(~Element,source),Buffer.contents(~Element,result)}

def leaf_result(~Element: Data,first: U32,rows: U32,+owners: StorageCopyLaws.Owners<Element>) -> Content<Element>:
  Content{Model.RowBand{BandCopy.source(~Element,owners),first,rows},BandGather.output(~Element,owners)}

def finished(~Element: Data,+first: U32,+rows: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  {contents(~Element,Bands.range_result(~Element,first,rows,owners)) == leaf_result(~Element,first,rows,StorageCopyLaws.observe(~Element,owners)) : Content<Element>}:
  (Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate}) = owners
  {==}

type Request is Data:
  Request{geometry: Bands.Geometry,first: U32,rows: U32,offset: U32}

def request_geometry(request: Request) -> Bands.Geometry:
  Request{geometry,first,rows,offset} = request
  geometry

def request_first(request: Request) -> U32:
  Request{geometry,first,rows,offset} = request
  first

def request_rows(request: Request) -> U32:
  Request{geometry,first,rows,offset} = request
  rows

def request_offset(request: Request) -> U32:
  Request{geometry,first,rows,offset} = request
  offset

def parameters(request: Request,+first: U32,+rows: U32,input_length: Nat,output_length: Nat) -> BandCopy.Request:
  Request{Bands.Geometry{batch,channels,height,+width},+target_first,+target_rows,target_offset} = request
  +begin = U32.max(first,target_first)
  +end = U32.max(begin,U32.min((first + rows : U32),(target_first + target_rows : U32)))
  BandCopy.Request{U32.to_nat(channels),0,((end - begin) * width : U32),(rows * width : U32),(target_rows * width : U32),
    ((begin - first) * width : U32),(target_offset + (begin - target_first) * width : U32),input_length,output_length}

def leaf_model(~Element: Data,request: Request,+first: U32,+rows: U32,source: Buffer.Content<Element>,result: Buffer.Content<Element>) -> Content<Element>:
  Buffer.Content{input,+input_length} = source
  Buffer.Content{output,+output_length} = result
  leaf_result(~Element,first,rows,BandCopy.modeled(~Element,parameters(request,first,rows,input_length,output_length),StorageCopyLaws.Owners{input,input_length,output,output_length}))

def plane_result(~Element: Data,+first: U32,+rows: U32,+request: BandCopy.Request,+input: Buffer.Model<Element>,+result: Buffer.Model<Element>) ->
  {contents(~Element,Bands.range_result(~Element,first,rows,BandCopy.actual(~Element,request,(Buffer.pack(~Element,input),Buffer.pack(~Element,result))))) ==
    leaf_result(~Element,first,rows,BandCopy.modeled(~Element,request,BandGather.owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,result)))) : Content<Element>}:
  BandCopy.Request{+remaining,+channel,+count,+source_stride,+destination_stride,+source_start,+destination_start,input_length,output_length} = request
  %Equal.sym(Content<Element>,contents(~Element,Bands.range_result(~Element,first,rows,BandCopy.actual(~Element,request,(Buffer.pack(~Element,input),Buffer.pack(~Element,result))))),
    leaf_result(~Element,first,rows,StorageCopyLaws.observe(~Element,BandCopy.actual(~Element,request,(Buffer.pack(~Element,input),Buffer.pack(~Element,result))))),
    finished(~Element,first,rows,BandCopy.actual(~Element,request,(Buffer.pack(~Element,input),Buffer.pack(~Element,result))))) :
    {_ == leaf_result(~Element,first,rows,BandCopy.modeled(~Element,request,BandGather.owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,result)))) : Content<Element>}
  Equal.cong(StorageCopyLaws.Owners<Element>,Content<Element>,owners => leaf_result(~Element,first,rows,owners),
    StorageCopyLaws.observe(~Element,BandCopy.actual(~Element,request,(Buffer.pack(~Element,input),Buffer.pack(~Element,result)))),
    BandCopy.modeled(~Element,request,BandGather.owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,result))),
    Equal.trans(StorageCopyLaws.Owners<Element>,
      StorageCopyLaws.observe(~Element,BandCopy.actual(~Element,request,(Buffer.pack(~Element,input),Buffer.pack(~Element,result)))),
      BandCopy.modeled(~Element,request,StorageCopyLaws.observe(~Element,(Buffer.pack(~Element,input),Buffer.pack(~Element,result)))),
      BandCopy.modeled(~Element,request,BandGather.owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,result))),
      BandCopy.planes(~Element,remaining,channel,count,source_stride,destination_stride,source_start,destination_start,(Buffer.pack(~Element,input),Buffer.pack(~Element,result))),
      Equal.cong(StorageCopyLaws.Owners<Element>,StorageCopyLaws.Owners<Element>,owners => BandCopy.modeled(~Element,request,owners),
        StorageCopyLaws.observe(~Element,(Buffer.pack(~Element,input),Buffer.pack(~Element,result))),BandGather.owners(~Element,Buffer.content(~Element,input),Buffer.content(~Element,result)),BandGather.pair_contents(~Element,input,result))))

def same_word(+left: U32,+right: U32,equal: {U32.is_eq(left,right) == True{} : Bool}) -> {left == right : U32}:
  +natural = Order.equal_of_is_eq(U32.to_nat(left),U32.to_nat(right),
    %Comparison.eq(left,right) : {_ == True{} : Bool}
    equal)
  %Arithmetic.native_roundtrip(left) : {_ == right : U32}
  %Arithmetic.native_roundtrip(right) : {U32.from_nat(U32.to_nat(left)) == _ : U32}
  Equal.cong(Nat,U32,value => U32.from_nat(value),U32.to_nat(left),U32.to_nat(right),natural)

# An empty row intersection copies zero elements per channel.
def empty_count(+begin: U32,+end: U32,+width: U32,equal: {U32.is_eq(begin,end) == True{} : Bool}) -> {((end - begin) * width : U32) == 0 : U32}:
  %same_word(begin,end,equal) : {((_ - begin) * width : U32) == 0 : U32}
  %Equal.sym(U32,(begin - begin : U32),0,WordInterval.subtract_self(begin)) : {(_ * width : U32) == 0 : U32}
  Arithmetic.multiply_zero_left(width)

# The skipped copy loop leaves both owners as the zero-length plane refinement_model does.
def empty_result(~Element: Data,+first: U32,+rows: U32,+remaining: Nat,+count: U32,+source_stride: U32,+destination_stride: U32,
  +source_start: U32,+destination_start: U32,+input: Tree.Tree<Element>,+input_length: Nat,+input_certificate: Certificate.Certificate<Element,Tree.pack(Element,input)>,
  +output: Tree.Tree<Element>,+output_length: Nat,+output_certificate: Certificate.Certificate<Element,Tree.pack(Element,output)>,
  zero: {count == 0 : U32}) ->
  {contents(~Element,Bands.range_result(~Element,first,rows,
      (Buffer.pack(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.pack(~Element,Buffer.Model{output,output_length,output_certificate})))) ==
    leaf_result(~Element,first,rows,BandCopy.modeled(~Element,BandCopy.Request{remaining,0,count,source_stride,destination_stride,source_start,destination_start,input_length,output_length},
      BandGather.owners(~Element,Buffer.content(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.content(~Element,Buffer.Model{output,output_length,output_certificate})))) : Content<Element>}:
  +original = {StorageCopyLaws.Owners{input,input_length,output,output_length} : StorageCopyLaws.Owners<Element>}
  %Equal.sym(U32,count,0,zero) :
    {contents(~Element,Bands.range_result(~Element,first,rows,(Buffer.pack(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.pack(~Element,Buffer.Model{output,output_length,output_certificate})))) ==
      leaf_result(~Element,first,rows,BandCopy.modeled(~Element,BandCopy.Request{remaining,0,_,source_stride,destination_stride,source_start,destination_start,input_length,output_length},original)) : Content<Element>}
  %Equal.sym(StorageCopyLaws.Owners<Element>,BandCopy.model(~Element,remaining,0,0,source_stride,destination_stride,source_start,destination_start,original),
    StorageCopyLaws.Owners{input,input_length,Writes.pure(Element,BandCopy.writes(~Element,remaining,0,0,source_stride,destination_stride,source_start,destination_start,input),output),output_length},
    BandCopy.model_writes(~Element,remaining,0,0,source_stride,destination_stride,source_start,destination_start,input,input_length,output,output_length)) :
    {contents(~Element,Bands.range_result(~Element,first,rows,(Buffer.pack(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.pack(~Element,Buffer.Model{output,output_length,output_certificate})))) == leaf_result(~Element,first,rows,_) : Content<Element>}
  %Equal.sym(List<&2,Writes.Assignment<Element>>,BandCopy.writes(~Element,remaining,0,0,source_stride,destination_stride,source_start,destination_start,input),[],
    BandCopy.empty_writes(~Element,remaining,0,source_stride,destination_stride,source_start,destination_start,input)) :
    {contents(~Element,Bands.range_result(~Element,first,rows,(Buffer.pack(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.pack(~Element,Buffer.Model{output,output_length,output_certificate})))) ==
      leaf_result(~Element,first,rows,StorageCopyLaws.Owners{input,input_length,Writes.pure(Element,_,output),output_length}) : Content<Element>}
  Equal.trans(Content<Element>,contents(~Element,Bands.range_result(~Element,first,rows,(Buffer.pack(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.pack(~Element,Buffer.Model{output,output_length,output_certificate})))),leaf_result(~Element,first,rows,StorageCopyLaws.observe(~Element,(Buffer.pack(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.pack(~Element,Buffer.Model{output,output_length,output_certificate})))),
    leaf_result(~Element,first,rows,original),finished(~Element,first,rows,(Buffer.pack(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.pack(~Element,Buffer.Model{output,output_length,output_certificate}))),
    Equal.cong(StorageCopyLaws.Owners<Element>,Content<Element>,result => leaf_result(~Element,first,rows,result),StorageCopyLaws.observe(~Element,(Buffer.pack(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.pack(~Element,Buffer.Model{output,output_length,output_certificate}))),original,
      BandGather.pair_contents(~Element,Buffer.Model{input,input_length,input_certificate},Buffer.Model{output,output_length,output_certificate})))

def range_result_matches(~Element: Data,empty: Bool,+first: U32,+rows: U32,+channels: U32,+begin: U32,+end: U32,+width: U32,
  +source_stride: U32,+destination_stride: U32,+source_start: U32,+destination_start: U32,
  +input: Tree.Tree<Element>,+input_length: Nat,+input_certificate: Certificate.Certificate<Element,Tree.pack(Element,input)>,
  +output: Tree.Tree<Element>,+output_length: Nat,+output_certificate: Certificate.Certificate<Element,Tree.pack(Element,output)>,
  exact: {U32.is_eq(begin,end) == empty : Bool}) ->
  {contents(~Element,Bands.range_result(~Element,first,rows,Bands.range_planes(~Element,empty,channels,((end - begin) * width : U32),
      source_stride,destination_stride,source_start,destination_start,
      (Buffer.pack(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.pack(~Element,Buffer.Model{output,output_length,output_certificate}))))) ==
    leaf_result(~Element,first,rows,BandCopy.modeled(~Element,BandCopy.Request{U32.to_nat(channels),0,((end - begin) * width : U32),
        source_stride,destination_stride,source_start,destination_start,input_length,output_length},
      BandGather.owners(~Element,Buffer.content(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.content(~Element,Buffer.Model{output,output_length,output_certificate})))) : Content<Element>}:
  match empty:
    case False{}:
      plane_result(~Element,first,rows,BandCopy.Request{U32.to_nat(channels),0,((end - begin) * width : U32),source_stride,destination_stride,source_start,destination_start,
        input_length,output_length},Buffer.Model{input,input_length,input_certificate},Buffer.Model{output,output_length,output_certificate})
    case True{}:
      empty_result(~Element,first,rows,U32.to_nat(channels),((end - begin) * width : U32),source_stride,destination_stride,source_start,destination_start,
        input,input_length,input_certificate,output,output_length,output_certificate,empty_count(begin,end,width,exact))

def refinement_leaf(~Element: Data,request: Request,+first: U32,+rows: U32,input: Buffer.Model<Element>,result: Buffer.Model<Element>) ->
  {contents(~Element,Bands.range_leaf(~Element,request_geometry(request),request_first(request),request_rows(request),request_offset(request),first,rows,Buffer.pack(~Element,input),Buffer.pack(~Element,result))) ==
    leaf_model(~Element,request,first,rows,Buffer.content(~Element,input),Buffer.content(~Element,result)) : Content<Element>}:
  match request input result:
    case Request{Bands.Geometry{batch,+channels,height,+width},+target_first,+target_rows,+target_offset} Buffer.Model{+input,+input_length,+input_certificate} Buffer.Model{+output,+output_length,+output_certificate}:
      +begin = U32.max(first,target_first)
      +end = U32.max(begin,U32.min((first + rows : U32),(target_first + target_rows : U32)))
      range_result_matches(~Element,U32.is_eq(begin,end),first,rows,channels,begin,end,width,(rows * width : U32),(target_rows * width : U32),
        ((begin - first) * width : U32),(target_offset + (begin - target_first) * width : U32),
        input,input_length,input_certificate,output,output_length,output_certificate,{==})

def joined(~Element: Data,left: Model.Content<Element>,right: Content<Element>) -> Content<Element>:
  Content{source,result} = right
  Content{Model.Joined{left,source},result}

def bind_content(~Element: Data,content: Content<Element>,continuation: Model.Content<Element> -> Buffer.Content<Element> -> Content<Element>) -> Content<Element>:
  Content{source,result} = content
  continuation(source)(result)

def refinement_model(~Element: Data,+request: Request,bands: Model.Model<Element>,result: Buffer.Content<Element>) -> Content<Element>:
  match bands:
    case Model.Leaf{input,first,rows}: leaf_model(~Element,request,first,rows,Buffer.content(~Element,input),result)
    case Model.Fork{left,right}:
      bind_content(~Element,refinement_model(~Element,request,left,result),left => result => joined(~Element,left,refinement_model(~Element,request,right,result)))

def continued_model(~Element: Data,request: Request,right: Model.Model<Element>,left: Content<Element>) -> Content<Element>:
  bind_content(~Element,left,source => result => joined(~Element,source,refinement_model(~Element,request,right,result)))

def run(~Element: Data,+request: Request,bands: Bands.Bands(~Element),result: Storage.Buffer<Element>) -> Bands.Bands(~Element) & Storage.Buffer<Element>:
  Bands.range_into(~Element,request_geometry(request),request_first(request),request_rows(request),request_offset(request),bands,result)

def joined_native(~Element: Data,left: Bands.Bands(~Element),right: Bands.Bands(~Element) & Storage.Buffer<Element>) -> Bands.Bands(~Element) & Storage.Buffer<Element>:
  Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~(Bands.Bands(~Element) & Storage.Buffer<Element>),right,
    right => result => (Partition.Fork{left,right},result))

def continued(~Element: Data,request: Request,right: Model.Model<Element>,left: Bands.Bands(~Element) & Storage.Buffer<Element>) -> Bands.Bands(~Element) & Storage.Buffer<Element>:
  Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~(Bands.Bands(~Element) & Storage.Buffer<Element>),left,
    left => result => joined_native(~Element,left,run(~Element,request,Model.pack(~Element,right),result)))

def joined_contents(~Element: Data,left: Bands.Bands(~Element),right: Bands.Bands(~Element) & Storage.Buffer<Element>) ->
  {contents(~Element,joined_native(~Element,left,right)) == joined(~Element,Model.contents(~Element,left),contents(~Element,right)) : Content<Element>}:
  (right,result) = right
  {==}

def continue_packed(~Element: Data,+request: Request,+right: Model.Model<Element>,+left: Model.Model<Element>,+result: Buffer.Model<Element>,
  rule: @result: Storage.Buffer<Element> -> {contents(~Element,run(~Element,request,Model.pack(~Element,right),result)) == refinement_model(~Element,request,right,Buffer.contents(~Element,result)) : Content<Element>}) ->
  {contents(~Element,continued(~Element,request,right,(Model.pack(~Element,left),Buffer.pack(~Element,result)))) ==
    continued_model(~Element,request,right,contents(~Element,(Model.pack(~Element,left),Buffer.pack(~Element,result)))) : Content<Element>}:
  Equal.trans(Content<Element>,
    contents(~Element,continued(~Element,request,right,(Model.pack(~Element,left),Buffer.pack(~Element,result)))),
    joined(~Element,Model.contents(~Element,Model.pack(~Element,left)),contents(~Element,run(~Element,request,Model.pack(~Element,right),Buffer.pack(~Element,result)))),
    continued_model(~Element,request,right,contents(~Element,(Model.pack(~Element,left),Buffer.pack(~Element,result)))),
    joined_contents(~Element,Model.pack(~Element,left),run(~Element,request,Model.pack(~Element,right),Buffer.pack(~Element,result))),
    Equal.cong(Content<Element>,Content<Element>,right => joined(~Element,Model.contents(~Element,Model.pack(~Element,left)),right),
      contents(~Element,run(~Element,request,Model.pack(~Element,right),Buffer.pack(~Element,result))),
      refinement_model(~Element,request,right,Buffer.contents(~Element,Buffer.pack(~Element,result))),rule(Buffer.pack(~Element,result))))

def continue_observed(~Element: Data,+request: Request,+right: Model.Model<Element>,-left: Bands.Bands(~Element),-result: Storage.Buffer<Element>,
  left_observation: Model.Observation<Element,left>,result_observation: Buffer.Observation<Element,result>,
  rule: @result: Storage.Buffer<Element> -> {contents(~Element,run(~Element,request,Model.pack(~Element,right),result)) == refinement_model(~Element,request,right,Buffer.contents(~Element,result)) : Content<Element>}) ->
  {contents(~Element,continued(~Element,request,right,(left,result))) == continued_model(~Element,request,right,contents(~Element,(left,result))) : Content<Element>}:
  Model.Observation{+left_model,left_equal} = left_observation
  Buffer.Observation{+result_model,result_equal} = result_observation
  %left_equal : {contents(~Element,continued(~Element,request,right,(_,result))) == continued_model(~Element,request,right,contents(~Element,(_,result))) : Content<Element>}
  %result_equal : {contents(~Element,continued(~Element,request,right,(Model.pack(~Element,left_model),_))) ==
    continued_model(~Element,request,right,contents(~Element,(Model.pack(~Element,left_model),_))) : Content<Element>}
  continue_packed(~Element,request,right,left_model,result_model,rule)

def continuation(~Element: Data,+request: Request,+right: Model.Model<Element>,left: Bands.Bands(~Element) & Storage.Buffer<Element>,
  rule: @result: Storage.Buffer<Element> -> {contents(~Element,run(~Element,request,Model.pack(~Element,right),result)) == refinement_model(~Element,request,right,Buffer.contents(~Element,result)) : Content<Element>}) ->
  {contents(~Element,continued(~Element,request,right,left)) == continued_model(~Element,request,right,contents(~Element,left)) : Content<Element>}:
  (left,result) = left
  continue_observed(~Element,request,right,left,result,Model.observe(~Element,left),Buffer.observe(~Element,result),rule)

type RefinementContext<-Element: Data> is Data:
  RefinementContext{request: Request,bands: Model.Model<Element>}

def refinement_proposition(~Element: Data,context: RefinementContext<Element>,-result: Storage.Buffer<Element>) -> Type:
  RefinementContext{+request,+bands} = context
  {contents(~Element,run(~Element,request,Model.pack(~Element,bands),result)) == refinement_model(~Element,request,bands,Buffer.contents(~Element,result)) : Content<Element>}

def refinement_range(~Element: Data,+request: Request,bands: Model.Model<Element>,result: Storage.Buffer<Element>) ->
  {contents(~Element,run(~Element,request,Model.pack(~Element,bands),result)) == refinement_model(~Element,request,bands,Buffer.contents(~Element,result)) : Content<Element>}:
  match bands:
    case Model.Leaf{+input,+first,+rows}:
      Buffer.lift(~Element,~RefinementContext<Element>,~refinement_proposition(~Element),RefinementContext{request,Model.Leaf{input,first,rows}},result,Buffer.observe(~Element,result),
        result =>
          +result = result
          Equal.trans(Content<Element>,contents(~Element,run(~Element,request,Model.pack(~Element,Model.Leaf{input,first,rows}),Buffer.pack(~Element,result))),
            leaf_model(~Element,request,first,rows,Buffer.content(~Element,input),Buffer.content(~Element,result)),
            leaf_model(~Element,request,first,rows,Buffer.content(~Element,input),Buffer.contents(~Element,Buffer.pack(~Element,result))),
            refinement_leaf(~Element,request,first,rows,input,result),
            Equal.cong(Buffer.Content<Element>,Content<Element>,output => leaf_model(~Element,request,first,rows,Buffer.content(~Element,input),output),
              Buffer.content(~Element,result),Buffer.contents(~Element,Buffer.pack(~Element,result)),
              Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,result)),Buffer.content(~Element,result),Buffer.packed_contents(~Element,result)))))
    case Model.Fork{+left,+right}:
      Buffer.lift(~Element,~RefinementContext<Element>,~refinement_proposition(~Element),RefinementContext{request,Model.Fork{left,right}},result,Buffer.observe(~Element,result),
        result =>
          +result = result
          Equal.trans(Content<Element>,contents(~Element,run(~Element,request,Model.pack(~Element,Model.Fork{left,right}),Buffer.pack(~Element,result))),
            continued_model(~Element,request,right,contents(~Element,run(~Element,request,Model.pack(~Element,left),Buffer.pack(~Element,result)))),
            continued_model(~Element,request,right,refinement_model(~Element,request,left,Buffer.contents(~Element,Buffer.pack(~Element,result)))),
            continuation(~Element,request,right,run(~Element,request,Model.pack(~Element,left),Buffer.pack(~Element,result)),output => refinement_range(~Element,request,right,output)),
            Equal.cong(Content<Element>,Content<Element>,left => continued_model(~Element,request,right,left),
              contents(~Element,run(~Element,request,Model.pack(~Element,left),Buffer.pack(~Element,result))),
              refinement_model(~Element,request,left,Buffer.contents(~Element,Buffer.pack(~Element,result))),refinement_range(~Element,request,left,Buffer.pack(~Element,result)))))

type Refinement<-Element: Data,-request: Request,-bands: Bands.Bands(~Element),-output: Storage.Buffer<Element>> is Data:
  Refinement{representation: Model.Model<Element>,
    owners: {Model.pack(~Element,representation) == bands : Bands.Bands(~Element)},
    execution: {contents(~Element,run(~Element,request,bands,output)) == refinement_model(~Element,request,representation,Buffer.contents(~Element,output)) : Content<Element>}}

def observed(~Element: Data,+request: Request,-bands: Bands.Bands(~Element),output: Storage.Buffer<Element>,observation: Model.Observation<Element,bands>) ->
  Refinement<Element,request,bands,output>:
  Model.Observation{+representation,+owners} = observation
  Refinement{representation,owners,
    %owners : {contents(~Element,run(~Element,request,_,output)) == refinement_model(~Element,request,representation,Buffer.contents(~Element,output)) : Content<Element>}
    refinement_range(~Element,request,representation,output)}

def actual(~Element: Data,request: Request,bands: Bands.Bands(~Element),output: Storage.Buffer<Element>) -> Refinement<Element,request,bands,output>:
  observed(~Element,request,bands,output,Model.observe(~Element,bands))

# Halo and repartition copies share one exact ordered write program. The
# source partition is returned unchanged, including every ownership leaf.
def leaf_writes(~Element: Data,request: Request,+first: U32,+rows: U32,input: Tree.Tree<Element>) -> List<&2,Writes.Assignment<Element>>:
  Request{Bands.Geometry{images,channels,height,+width},+target_first,+target_rows,target_offset} = request
  +begin = U32.max(first,target_first)
  +end = U32.max(begin,U32.min((first + rows : U32),(target_first + target_rows : U32)))
  BandCopy.writes(~Element,U32.to_nat(channels),0,((end - begin) * width : U32),(rows * width : U32),(target_rows * width : U32),
    ((begin - first) * width : U32),(target_offset + (begin - target_first) * width : U32),input)

def writes(~Element: Data,+request: Request,bands: Model.Model<Element>) -> List<&2,Writes.Assignment<Element>>:
  match bands:
    case Model.Leaf{Buffer.Model{input,length,certificate},first,rows}: leaf_writes(~Element,request,first,rows,input)
    case Model.Fork{left,right}: List.append(&2,Writes.Assignment<Element>,writes(~Element,request,left),writes(~Element,request,right))

def content_writes(~Element: Data,+request: Request,bands: Model.Content<Element>) -> List<&2,Writes.Assignment<Element>>:
  match bands:
    case Model.RowBand{Buffer.Content{input,length},first,rows}: leaf_writes(~Element,request,first,rows,input)
    case Model.Joined{left,right}: List.append(&2,Writes.Assignment<Element>,content_writes(~Element,request,left),content_writes(~Element,request,right))

def content_program(~Element: Data,+request: Request,bands: Model.Model<Element>) ->
  {writes(~Element,request,bands) == content_writes(~Element,request,Model.content(~Element,bands)) : List<&2,Writes.Assignment<Element>>}:
  match bands:
    case Model.Leaf{Buffer.Model{input,length,certificate},first,rows}: {==}
    case Model.Fork{+left,+right}:
      %content_program(~Element,request,left) : {List.append(&2,Writes.Assignment<Element>,writes(~Element,request,left),writes(~Element,request,right)) == List.append(&2,Writes.Assignment<Element>,_,content_writes(~Element,request,Model.content(~Element,right))) : List<&2,Writes.Assignment<Element>>}
      %content_program(~Element,request,right) : {List.append(&2,Writes.Assignment<Element>,writes(~Element,request,left),writes(~Element,request,right)) == List.append(&2,Writes.Assignment<Element>,writes(~Element,request,left),_) : List<&2,Writes.Assignment<Element>>}
      {==}

def written(~Element: Data,request: Request,+bands: Model.Content<Element>,output: Buffer.Content<Element>) -> Content<Element>:
  Buffer.Content{tree,length} = output
  Content{bands,Buffer.Content{Writes.pure(Element,content_writes(~Element,request,bands),tree),length}}

def program_leaf(~Element: Data,+request: Request,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+input_length: Nat,+output: Tree.Tree<Element>,+output_length: Nat) ->
  {leaf_model(~Element,request,first,rows,Buffer.Content{input,input_length},Buffer.Content{output,output_length}) ==
    Content{Model.RowBand{Buffer.Content{input,input_length},first,rows},Buffer.Content{Writes.pure(Element,leaf_writes(~Element,request,first,rows,input),output),output_length}} : Content<Element>}:
  Request{Bands.Geometry{+images,+channels,+height,+width},+target_first,+target_rows,+target_offset} = request
  +begin = U32.max(first,target_first)
  +end = U32.max(begin,U32.min((first + rows : U32),(target_first + target_rows : U32)))
  +count = ((end - begin) * width : U32)
  Equal.cong(StorageCopyLaws.Owners<Element>,Content<Element>,owners => leaf_result(~Element,first,rows,owners),
    BandCopy.modeled(~Element,parameters(request,first,rows,input_length,output_length),StorageCopyLaws.Owners{input,input_length,output,output_length}),
    StorageCopyLaws.Owners{input,input_length,Writes.pure(Element,leaf_writes(~Element,request,first,rows,input),output),output_length},
    BandCopy.model_writes(~Element,U32.to_nat(channels),0,count,(rows * width : U32),(target_rows * width : U32),((begin - first) * width : U32),(target_offset + (begin - target_first) * width : U32),input,input_length,output,output_length))

def program_model(~Element: Data,+request: Request,bands: Model.Model<Element>,+output: Tree.Tree<Element>,+length: Nat) ->
  {refinement_model(~Element,request,bands,Buffer.Content{output,length}) ==
    Content{Model.content(~Element,bands),Buffer.Content{Writes.pure(Element,writes(~Element,request,bands),output),length}} : Content<Element>}:
  match bands:
    case Model.Leaf{Buffer.Model{input,input_length,certificate},first,rows}: program_leaf(~Element,request,first,rows,input,input_length,output,length)
    case Model.Fork{+left,+right}:
      +left_writes = writes(~Element,request,left)
      +right_writes = writes(~Element,request,right)
      +middle = Writes.pure(Element,left_writes,output)
      %Equal.sym(Content<Element>,refinement_model(~Element,request,left,Buffer.Content{output,length}),
        Content{Model.content(~Element,left),Buffer.Content{middle,length}},program_model(~Element,request,left,output,length)) :
        {continued_model(~Element,request,right,_) == Content{Model.content(~Element,bands),Buffer.Content{Writes.pure(Element,writes(~Element,request,bands),output),length}} : Content<Element>}
      %Equal.sym(Content<Element>,refinement_model(~Element,request,right,Buffer.Content{middle,length}),
        Content{Model.content(~Element,right),Buffer.Content{Writes.pure(Element,right_writes,middle),length}},program_model(~Element,request,right,middle,length)) :
        {joined(~Element,Model.content(~Element,left),_) == Content{Model.content(~Element,bands),Buffer.Content{Writes.pure(Element,writes(~Element,request,bands),output),length}} : Content<Element>}
      Equal.cong(Tree.Tree<Element>,Content<Element>,tree => Content{Model.content(~Element,bands),Buffer.Content{tree,length}},
        Writes.pure(Element,right_writes,middle),Writes.pure(Element,List.append(&2,Writes.Assignment<Element>,left_writes,right_writes),output),
        Equal.sym(Tree.Tree<Element>,Writes.pure(Element,List.append(&2,Writes.Assignment<Element>,left_writes,right_writes),output),Writes.pure(Element,right_writes,middle),Writes.concatenated_writes(Element,left_writes,right_writes,output)))

def program_range(~Element: Data,+request: Request,+bands: Model.Model<Element>,+output: Buffer.Model<Element>) ->
  {contents(~Element,run(~Element,request,Model.pack(~Element,bands),Buffer.pack(~Element,output))) ==
    Content{Model.content(~Element,bands),Buffer.Content{Writes.pure(Element,writes(~Element,request,bands),Buffer.tree(~Element,output)),Buffer.length(~Element,output)}} : Content<Element>}:
  Buffer.Model{+tree,+length,+certificate} = output
  Equal.trans(Content<Element>,contents(~Element,run(~Element,request,Model.pack(~Element,bands),Buffer.pack(~Element,output))),
    refinement_model(~Element,request,bands,Buffer.content(~Element,output)),Content{Model.content(~Element,bands),Buffer.Content{Writes.pure(Element,writes(~Element,request,bands),tree),length}},
    %Buffer.packed_contents(~Element,output) : {contents(~Element,run(~Element,request,Model.pack(~Element,bands),Buffer.pack(~Element,output))) == refinement_model(~Element,request,bands,_) : Content<Element>}
    refinement_range(~Element,request,bands,Buffer.pack(~Element,output)),program_model(~Element,request,bands,tree,length))

def output_ready(~Element: Data,+request: Request,+bands: Model.Model<Element>,+output: Buffer.Model<Element>,ready: Buffer.Sized<Element,output>) ->
  Buffer.ContentReady<Element,result(~Element,contents(~Element,run(~Element,request,Model.pack(~Element,bands),Buffer.pack(~Element,output))))>:
  Buffer.Model{+tree,+length,+certificate} = output
  %Equal.sym(Content<Element>,contents(~Element,run(~Element,request,Model.pack(~Element,bands),Buffer.pack(~Element,output))),
    Content{Model.content(~Element,bands),Buffer.Content{Writes.pure(Element,writes(~Element,request,bands),Buffer.tree(~Element,output)),Buffer.length(~Element,output)}},program_range(~Element,request,bands,output)) : Buffer.ContentReady<Element,result(~Element,_)>
  Writes.content_ready(~Element,writes(~Element,request,bands),Buffer.tree(~Element,output),Buffer.length(~Element,output),Buffer.content_ready(~Element,output,ready))

def modeled_content(~Element: Data,+request: Request,+bands: Model.Model<Element>,output: Buffer.Content<Element>) ->
  {refinement_model(~Element,request,bands,output) == written(~Element,request,Model.content(~Element,bands),output) : Content<Element>}:
  Buffer.Content{+tree,+length} = output
  %content_program(~Element,request,bands) :
    {refinement_model(~Element,request,bands,output) == Content{Model.content(~Element,bands),Buffer.Content{Writes.pure(Element,_,tree),length}} : Content<Element>}
  program_model(~Element,request,bands,tree,length)

type ProgramContext<-Element: Data> is Data:
  ProgramContext{request: Request,bands: Model.Model<Element>}

def program_proposition(~Element: Data,context: ProgramContext<Element>,-output: Storage.Buffer<Element>) -> Type:
  ProgramContext{+request,+bands} = context
  {contents(~Element,run(~Element,request,Model.pack(~Element,bands),output)) == written(~Element,request,Model.content(~Element,bands),Buffer.contents(~Element,output)) : Content<Element>}

def native_range(~Element: Data,+request: Request,+bands: Model.Model<Element>,output: Storage.Buffer<Element>) ->
  {contents(~Element,run(~Element,request,Model.pack(~Element,bands),output)) == written(~Element,request,Model.content(~Element,bands),Buffer.contents(~Element,output)) : Content<Element>}:
  Buffer.lift(~Element,~ProgramContext<Element>,~program_proposition(~Element),ProgramContext{request,bands},output,Buffer.observe(~Element,output),output =>
    +output = output
    Equal.trans(Content<Element>,contents(~Element,run(~Element,request,Model.pack(~Element,bands),Buffer.pack(~Element,output))),
      refinement_model(~Element,request,bands,Buffer.contents(~Element,Buffer.pack(~Element,output))),written(~Element,request,Model.content(~Element,bands),Buffer.contents(~Element,Buffer.pack(~Element,output))),
      refinement_range(~Element,request,bands,Buffer.pack(~Element,output)),modeled_content(~Element,request,bands,Buffer.contents(~Element,Buffer.pack(~Element,output)))))

def canonical_range(~Element: Data,+request: Request,+bands: Model.Model<Element>,+output: Buffer.Model<Element>) ->
  {contents(~Element,run(~Element,request,Model.pack(~Element,bands),Buffer.pack(~Element,output))) == written(~Element,request,Model.content(~Element,bands),Buffer.content(~Element,output)) : Content<Element>}:
  %Buffer.packed_contents(~Element,output) : {contents(~Element,run(~Element,request,Model.pack(~Element,bands),Buffer.pack(~Element,output))) == written(~Element,request,Model.content(~Element,bands),_) : Content<Element>}
  native_range(~Element,request,bands,Buffer.pack(~Element,output))

# Concrete range copies discharge their per-channel bounds from the source
# invariant and the requested destination allocation. Empty overlaps do not
# demand a valid source address or a positive source band size.
def bounds_begin(request: Request,first: U32) -> U32:
  Request{geometry,target_first,target_rows,offset} = request
  U32.max(first,target_first)

def bounds_end(request: Request,+first: U32,rows: U32) -> U32:
  Request{geometry,+target_first,target_rows,offset} = request
  U32.max(U32.max(first,target_first),U32.min((first + rows : U32),(target_first + target_rows : U32)))

def copied_rows(+request: Request,+first: U32,rows: U32) -> U32:
  (bounds_end(request,first,rows) - bounds_begin(request,first) : U32)

def copied_elements(+request: Request,+first: U32,rows: U32) -> U32:
  Request{Bands.Geometry{images,channels,height,width},target_first,target_rows,offset} = request
  (copied_rows(request,first,rows) * width : U32)

def Source(request: Request,+first: U32,+rows: U32,capacity: U32) -> Data:
  Request{Bands.Geometry{images,channels,height,width},+target_first,+target_rows,offset} = request
  +bounds_begin = U32.max(first,target_first)
  +bounds_end = U32.max(bounds_begin,U32.min((first + rows : U32),(target_first + target_rows : U32)))
  BandCopy.Facts<0,(bounds_begin - first : U32),(bounds_end - bounds_begin : U32),channels,rows,width,capacity>

def Destination(request: Request,+first: U32,rows: U32,capacity: U32) -> Data:
  Request{Bands.Geometry{images,channels,height,width},+target_first,+target_rows,offset} = request
  +bounds_begin = U32.max(first,target_first)
  +bounds_end = U32.max(bounds_begin,U32.min((first + rows : U32),(target_first + target_rows : U32)))
  BandCopy.Facts<offset,(bounds_begin - target_first : U32),(bounds_end - bounds_begin : U32),channels,target_rows,width,capacity>

type Facts<-request: Request,-first: U32,-rows: U32,-source_capacity: U32,-output_capacity: U32> is Data:
  BoundsEmpty{empty: {copied_elements(request,first,rows) == 0 : U32}}
  Copied{source: Source(request,first,rows,source_capacity),destination: Destination(request,first,rows,output_capacity),
    source_rows: WordInterval.Span<first,rows,bounds_begin(request,first),bounds_end(request,first,rows)>,
    target_rows: WordInterval.Span<request_first(request),request_rows(request),bounds_begin(request,first),bounds_end(request,first,rows)>,
    positive: {Nat.is_lt(U32.to_nat(bounds_begin(request,first)),U32.to_nat(bounds_end(request,first,rows))) == True{} : Bool}}

def Target(request: Request,row_capacity: U32,output_capacity: U32) -> Data:
  Request{Bands.Geometry{images,channels,height,width},first,rows,offset} = request
  TargetFields<channels,width,first,rows,offset,row_capacity,output_capacity>

type TargetFields<-channels: U32,-width: U32,-first: U32,-rows: U32,-offset: U32,-row_capacity: U32,-output_capacity: U32> is Data:
  TargetFields{row_capacity_ok: {U32.is_le(row_capacity,2147483648) == True{} : Bool},output_capacity_ok: {U32.is_le(output_capacity,2147483648) == True{} : Bool},
    channels_positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool},width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
    rows_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(row_capacity)) == True{} : Bool},
    storage_room: {Nat.is_le(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))),U32.to_nat(output_capacity)) == True{} : Bool}}

def selected(~Element: Data,+request: Request,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+row_capacity: U32,+output_capacity: U32,
  target: Target(request,row_capacity,output_capacity),metadata: BandShape.Leaf(request_geometry(request),first,rows,Buffer.length(~Element,input)),ready: Buffer.Sized<Element,input>,
  intersection: WordInterval.Intersection<first,rows,request_first(request),request_rows(request),bounds_begin(request,first),bounds_end(request,first,rows)>) ->
  Facts<request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input)),output_capacity>:
  Request{Bands.Geometry{+images,+channels,+height,+width},+target_first,+target_rows,+offset} = request
  TargetFields{row_capacity_ok,+output_capacity_ok,+channels_positive,+width_positive,rows_room,+storage_room} = target
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  Buffer.Sized{+depth_ok,room} = ready
  match intersection:
    case WordInterval.EmptyIntersection{empty}:
      BoundsEmpty{
        %Equal.sym(U32,(bounds_end(request,first,rows) - bounds_begin(request,first) : U32),0,empty) : {(_ * width : U32) == 0 : U32}
        Arithmetic.multiply_zero_left(width)}
    case WordInterval.NonemptyIntersection{+positive,+source_span,+target_span}:
      WordInterval.Span{source_lower,source_shift,source_count,source_fits} = source_span
      WordInterval.Span{target_lower,target_shift,target_count,target_fits} = target_span
      +bounds_begin = bounds_begin(request,first)
      +bounds_end = bounds_end(request,first,rows)
      Copied{
        BandCopy.bounded(0,(bounds_begin - first : U32),(bounds_end - bounds_begin : U32),channels,rows,width,Tree.capacity(Element,Buffer.tree(~Element,input)),Buffer.capacity_bound(~Element,input,depth_ok),channels_positive,
          WordInterval.rows_positive(first,rows,bounds_begin,bounds_end,positive,source_span),width_positive,source_fits,
          %length_exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,input)))) == True{} : Bool} room),
        BandCopy.bounded(offset,(bounds_begin - target_first : U32),(bounds_end - bounds_begin : U32),channels,target_rows,width,output_capacity,output_capacity_ok,channels_positive,
          WordInterval.rows_positive(target_first,target_rows,bounds_begin,bounds_end,positive,target_span),width_positive,target_fits,storage_room),source_span,target_span,positive}

def bounds_leaf(~Element: Data,+request: Request,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+row_capacity: U32,+output_capacity: U32,
  +target: Target(request,row_capacity,output_capacity),+metadata: BandShape.Leaf(request_geometry(request),first,rows,Buffer.length(~Element,input)),ready: Buffer.Sized<Element,input>,
  total_room: {Nat.is_le(Inspection.total_rows(request_geometry(request)),U32.to_nat(row_capacity)) == True{} : Bool}) ->
  Facts<request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input)),output_capacity>:
  Request{Bands.Geometry{+images,+channels,+height,+width},+target_first,+target_rows,+offset} = request
  TargetFields{row_capacity_ok,output_capacity_ok,channels_positive,width_positive,rows_room,storage_room} = target
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  selected(~Element,request,first,rows,input,row_capacity,output_capacity,target,metadata,ready,
    WordInterval.intersection(first,rows,target_first,target_rows,row_capacity,row_capacity_ok,
      Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(height)),U32.to_nat(row_capacity),rows_fit,total_room),rows_room))

def BoundsTreeFacts(~Element: Data,+request: Request,+output_capacity: U32,bands: Model.Model<Element>) -> Data:
  match bands:
    case Model.Leaf{input,first,rows}: Facts<request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input)),output_capacity>
    case Model.Fork{left,right}: Shape.Both<BoundsTreeFacts(~Element,request,output_capacity,left),BoundsTreeFacts(~Element,request,output_capacity,right)>

def tree(~Element: Data,+request: Request,bands: Model.Model<Element>,+row_capacity: U32,+output_capacity: U32,
  +target: Target(request,row_capacity,output_capacity),metadata: BandShape.Contents(~Element,request_geometry(request),Model.content(~Element,bands)),ready: Model.Ready(~Element,bands),
  +total_room: {Nat.is_le(Inspection.total_rows(request_geometry(request)),U32.to_nat(row_capacity)) == True{} : Bool}) -> BoundsTreeFacts(~Element,request,output_capacity,bands):
  match bands:
    case Model.Leaf{+input,first,rows}:
      Buffer.Model{+source,+length,+certificate} = input
      bounds_leaf(~Element,request,first,rows,input,row_capacity,output_capacity,target,metadata,ready,total_room)
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{tree(~Element,request,left,row_capacity,output_capacity,target,left_metadata,left_ready,total_room),tree(~Element,request,right,row_capacity,output_capacity,target,right_metadata,right_ready,total_room)}

def empty_writes(~Element: Data,+request: Request,+first: U32,+rows: U32,+input: Tree.Tree<Element>,empty: {copied_elements(request,first,rows) == 0 : U32}) ->
  {leaf_writes(~Element,request,first,rows,input) == [] : List<&2,Writes.Assignment<Element>>}:
  Request{Bands.Geometry{images,+channels,height,+width},+target_first,+target_rows,+offset} = request
  +bounds_begin = bounds_begin(request,first)
  %Equal.sym(U32,copied_elements(request,first,rows),0,empty) : {BandCopy.writes(~Element,U32.to_nat(channels),0,_,(rows * width : U32),(target_rows * width : U32),((bounds_begin - first) * width : U32),(offset + (bounds_begin - target_first) * width : U32),input) == [] : List<&2,Writes.Assignment<Element>>}
  BandCopy.empty_writes(~Element,U32.to_nat(channels),0,(rows * width : U32),(target_rows * width : U32),((bounds_begin - first) * width : U32),(offset + (bounds_begin - target_first) * width : U32),input)

def leaf_inside(~Element: Data,+request: Request,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+source_capacity: U32,+output_capacity: U32,
  facts: Facts<request,first,rows,source_capacity,output_capacity>) -> Readback.Inside(~Element,output_capacity,leaf_writes(~Element,request,first,rows,input)):
  Request{Bands.Geometry{+images,+channels,+height,+width},+target_first,+target_rows,+offset} = request
  match facts:
    case BoundsEmpty{empty}:
      %Equal.sym(List<&2,Writes.Assignment<Element>>,leaf_writes(~Element,request,first,rows,input),[],empty_writes(~Element,request,first,rows,input,empty)) : Readback.Inside(~Element,output_capacity,_) Unit{}
    case Copied{source,target,source_span,target_span,positive}:
      BandCopy.Facts{valid,stride_exact,count_exact,offset_exact} = target
      BandCopy.writes_inside(~Element,U32.to_nat(channels),0n,BandCopy.region(offset,(bounds_begin(request,first) - target_first : U32),copied_rows(request,first,rows),channels,target_rows,width,output_capacity),
        (rows * width : U32),((bounds_begin(request,first) - first) * width : U32),input,valid,Order.reflexive(U32.to_nat(channels)))

def writes_inside(~Element: Data,+request: Request,bands: Model.Model<Element>,+output_capacity: U32,facts: BoundsTreeFacts(~Element,request,output_capacity,bands)) ->
  Readback.Inside(~Element,output_capacity,writes(~Element,request,bands)):
  match bands:
    case Model.Leaf{Buffer.Model{+input,length,certificate},first,rows}: leaf_inside(~Element,request,first,rows,input,Tree.capacity(Element,input),output_capacity,facts)
    case Model.Fork{+left,+right}:
      Shape.Both{left_facts,right_facts} = facts
      Readback.inside_append(~Element,output_capacity,writes(~Element,request,left),writes(~Element,request,right),writes_inside(~Element,request,left,output_capacity,left_facts),writes_inside(~Element,request,right,output_capacity,right_facts))

# The cells of the target band and the destination query of each.
def channel_count(request: Request) -> U32:
  Request{Bands.Geometry{images,channels,height,width},first,rows,offset} = request
  channels

def row_width(request: Request) -> U32:
  Request{Bands.Geometry{images,channels,height,width},first,rows,offset} = request
  width

def global_row(request: Request,row: Nat) -> Nat:
  Nat.add(U32.to_nat(request_first(request)),row)

def coverage_query(request: Request,channel: Nat,row: Nat,column: Nat) -> U32:
  Request{Bands.Geometry{images,channels,height,width},first,rows,offset} = request
  U32.from_nat(Nat.add(U32.to_nat(offset),RowMajor.index(channel,U32.to_nat(rows),U32.to_nat(width),row,column)))

# The source and destination regions of one leaf, and the source cell that
# each copied element reads from the shared Values relation.
def source_region(+request: Request,+first: U32,+rows: U32,capacity: U32) -> BandCopy.Region:
  Request{Bands.Geometry{images,channels,height,width},target_first,target_rows,offset} = request
  BandCopy.region(0,(bounds_begin(request,first) - first : U32),copied_rows(request,first,rows),channels,rows,width,capacity)

def destination_region(+request: Request,+first: U32,+rows: U32,capacity: U32) -> BandCopy.Region:
  Request{Bands.Geometry{images,channels,height,width},target_first,target_rows,offset} = request
  BandCopy.region(offset,(bounds_begin(request,first) - target_first : U32),copied_rows(request,first,rows),channels,target_rows,width,capacity)

def source_start(+request: Request,+first: U32) -> U32:
  Request{Bands.Geometry{images,channels,height,width},target_first,target_rows,offset} = request
  ((bounds_begin(request,first) - first) * width : U32)

# A copied source element is the cell at its channel, its row shifted into the
# overlap and its column.
def source_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+request: Request,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+channel: Nat,+local: Nat,+column: Nat,
  facts: Source(request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input))),span: WordInterval.Span<first,rows,bounds_begin(request,first),bounds_end(request,first,rows)>,
  values: BandValues.Leaf(~Element,~Context,~value,context,request_geometry(request),first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input)),
  +channel_inside: {Nat.is_lt(channel,BandCopy.channels(source_region(request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input))))) == True{} : Bool},
  +local_inside: {Nat.is_lt(local,U32.to_nat(copied_rows(request,first,rows))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(row_width(request))) == True{} : Bool}) ->
  {BandCopy.sample(~Element,Buffer.tree(~Element,input),BandCopy.stride(source_region(request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input)))),
      source_start(request,first),channel,Nat.add(Nat.mul(local,U32.to_nat(row_width(request))),column)) ==
    value(context,U32.from_nat(BandValues.pixel(request_geometry(request),channel,
      Nat.add(U32.to_nat(first),Nat.add(U32.to_nat((bounds_begin(request,first) - first : U32)),local)),column))) : Element}:
  Request{Bands.Geometry{+images,+channels,+height,+width},+target_first,+target_rows,+offset} = request
  +begin = bounds_begin(request,first)
  +end = bounds_end(request,first,rows)
  +shift = (begin - first : U32)
  +region = source_region(request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input)))
  +index = Nat.add(Nat.mul(local,U32.to_nat(width)),column)
  +query = BandCopy.readback_query(region,channel,index)
  +position = BandValues.position(Bands.Geometry{images,channels,height,width},rows,channel,Nat.add(U32.to_nat(shift),local),column)
  +pixel = U32.from_nat(BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),Nat.add(U32.to_nat(shift),local)),column))
  +start = (shift * width : U32)
  +stride = (rows * width : U32)
  +query_address = Equal.cong(U32,U32,start => (start + U32.from_nat(channel) * stride + U32.from_nat(index) : U32),(0 + start : U32),start,Arithmetic.add_zero_left(start))
  %query_address : {Tree.read(Element,Buffer.tree(~Element,input),_) == value(context,pixel) : Element}
  %Equal.sym(U32,query,U32.from_nat(position),Encoded.encoding(query,position,
    BandCopy.pixel_query(0,shift,(end - begin : U32),channels,rows,width,Tree.capacity(Element,Buffer.tree(~Element,input)),channel,local,column,facts,channel_inside,local_inside,column_inside))) :
    {Tree.read(Element,Buffer.tree(~Element,input),_) == value(context,pixel) : Element}
  BandValues.leaf_read(~Element,~Context,~value,context,Bands.Geometry{images,channels,height,width},first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input),channel,Nat.add(U32.to_nat(shift),local),column,values,
    channel_inside,WordInterval.row_inside(first,rows,begin,end,local,span,local_inside),column_inside)

# A target row located in the source partition names an actual range write.
# Coverage follows the planner tree and the concrete unsigned intersection.
def covered_pixel(~Element: Data,+request: Request,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+output_capacity: U32,+channel: Nat,+row: Nat,+column: Nat,+copy_row: Nat,
  +target: Destination(request,first,rows,output_capacity),local_exact: {Nat.add(U32.to_nat((bounds_begin(request,first) - request_first(request) : U32)),copy_row) == row : Nat},
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channel_count(request))) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(row_width(request))) == True{} : Bool},
  +copy_inside: {Nat.is_lt(copy_row,U32.to_nat(copied_rows(request,first,rows))) == True{} : Bool}) ->
  {Readback.contains(~Element,coverage_query(request,channel,row,column),leaf_writes(~Element,request,first,rows,input)) == True{} : Bool}:
  Request{Bands.Geometry{+images,+channels,+height,+width},+target_first,+target_rows,+offset} = request
  BandCopy.Facts{valid,stride_exact,count_exact,offset_exact} = target
  +region = destination_region(request,first,rows,output_capacity)
  +index = Nat.add(Nat.mul(copy_row,U32.to_nat(width)),column)
  +index_inside = {
    %Equal.sym(Nat,U32.to_nat(copied_elements(request,first,rows)),Nat.mul(U32.to_nat(copied_rows(request,first,rows)),U32.to_nat(width)),count_exact) : {Nat.is_lt(index,_) == True{} : Bool}
    RowMajor.radix(copy_row,U32.to_nat(copied_rows(request,first,rows)),column,U32.to_nat(width),copy_inside,column_inside) : {Nat.is_lt(index,U32.to_nat(copied_elements(request,first,rows))) == True{} : Bool}}
  +query_exact = {
    %local_exact : {U32.to_nat(BandCopy.readback_query(region,channel,index)) == Nat.add(U32.to_nat(offset),RowMajor.index(channel,U32.to_nat(target_rows),U32.to_nat(width),_,column)) : Nat}
    BandCopy.pixel_query(offset,(bounds_begin(request,first) - target_first : U32),copied_rows(request,first,rows),channels,target_rows,width,output_capacity,channel,copy_row,column,target,channel_inside,copy_inside,column_inside) :
      {U32.to_nat(BandCopy.readback_query(region,channel,index)) == Nat.add(U32.to_nat(offset),RowMajor.index(channel,U32.to_nat(target_rows),U32.to_nat(width),row,column)) : Nat}}
  %Encoded.encoding(BandCopy.readback_query(region,channel,index),Nat.add(U32.to_nat(offset),RowMajor.index(channel,U32.to_nat(target_rows),U32.to_nat(width),row,column)),query_exact) :
    {Readback.contains(~Element,_,leaf_writes(~Element,request,first,rows,input)) == True{} : Bool}
  BandCopy.covered(~Element,U32.to_nat(channels),0n,channel,index,region,(rows * width : U32),source_start(request,first),input,channel_inside,index_inside)

def coverage_point(~Element: Data,+request: Request,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+row_capacity: U32,+output_capacity: U32,+channel: Nat,+row: Nat,+column: Nat,
  target: Target(request,row_capacity,output_capacity),located: WordInterval.Point<request_first(request),request_rows(request),bounds_begin(request,first),bounds_end(request,first,rows),row>,
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channel_count(request))) == True{} : Bool},row_inside: {Nat.is_lt(row,U32.to_nat(request_rows(request))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(row_width(request))) == True{} : Bool}) ->
  {Readback.contains(~Element,coverage_query(request,channel,row,column),leaf_writes(~Element,request,first,rows,input)) == True{} : Bool}:
  Request{Bands.Geometry{+images,+channels,+height,+width},+target_first,+target_rows,+offset} = request
  TargetFields{row_capacity_ok,output_capacity_ok,channels_positive,width_positive,rows_room,storage_room} = target
  WordInterval.Point{+copy_row,+copy_inside,local_exact,span} = located
  WordInterval.Span{lower,shift_exact,count_exact,fits} = span
  covered_pixel(~Element,request,first,rows,input,output_capacity,channel,row,column,copy_row,
    BandCopy.bounded(offset,(bounds_begin(request,first) - target_first : U32),copied_rows(request,first,rows),channels,target_rows,width,output_capacity,output_capacity_ok,channels_positive,
      Order.le_lt_trans(0n,row,U32.to_nat(target_rows),Order.zero_le(row),row_inside),width_positive,fits,storage_room),local_exact,channel_inside,column_inside,copy_inside)

def coverage_leaf(~Element: Data,+request: Request,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+row_capacity: U32,+output_capacity: U32,+channel: Nat,+row: Nat,+column: Nat,
  +target: Target(request,row_capacity,output_capacity),source_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(row_capacity)) == True{} : Bool},
  located: BandPlan.LeafLocation<first,rows,global_row(request,row)>,channel_inside: {Nat.is_lt(channel,U32.to_nat(channel_count(request))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(request_rows(request))) == True{} : Bool},column_inside: {Nat.is_lt(column,U32.to_nat(row_width(request))) == True{} : Bool}) ->
  {Readback.contains(~Element,coverage_query(request,channel,row,column),leaf_writes(~Element,request,first,rows,input)) == True{} : Bool}:
  Request{Bands.Geometry{+images,+channels,+height,+width},+target_first,+target_rows,+offset} = request
  TargetFields{row_capacity_ok,output_capacity_ok,channels_positive,width_positive,rows_room,storage_room} = target
  BandPlan.LeafLocation{+local,inside,address} = located
  coverage_point(~Element,request,first,rows,input,row_capacity,output_capacity,channel,row,column,target,
    WordInterval.overlap_point(first,rows,target_first,target_rows,local,row,row_capacity,row_capacity_ok,source_room,rows_room,inside,row_inside,
      Equal.sym(Nat,global_row(request,row),Nat.add(U32.to_nat(first),local),address)),channel_inside,row_inside,column_inside)

def located(~Element: Data,+request: Request,bands: Model.Model<Element>,+row_capacity: U32,+output_capacity: U32,+channel: Nat,+row: Nat,+column: Nat,
  +target: Target(request,row_capacity,output_capacity),within: BandPlan.Within(Model.layout(~Element,bands),0n,U32.to_nat(row_capacity)),
  location: BandPlan.Located(Model.layout(~Element,bands),global_row(request,row)),+channel_inside: {Nat.is_lt(channel,U32.to_nat(channel_count(request))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(request_rows(request))) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(row_width(request))) == True{} : Bool}) ->
  {Readback.contains(~Element,coverage_query(request,channel,row,column),writes(~Element,request,bands)) == True{} : Bool}:
  match bands:
    case Model.Leaf{Buffer.Model{input,length,certificate},first,rows}:
      BandPlan.LeafWithin{lower,upper} = within
      coverage_leaf(~Element,request,first,rows,input,row_capacity,output_capacity,channel,row,column,target,upper,location,channel_inside,row_inside,column_inside)
    case Model.Fork{+left,+right}:
      Shape.Both{+left_within,+right_within} = within
      match location:
        case Inl{here}: Readback.covered_left(~Element,coverage_query(request,channel,row,column),writes(~Element,request,left),writes(~Element,request,right),
          located(~Element,request,left,row_capacity,output_capacity,channel,row,column,target,left_within,here,channel_inside,row_inside,column_inside))
        case Inr{here}: Readback.covered_right(~Element,coverage_query(request,channel,row,column),writes(~Element,request,left),writes(~Element,request,right),
          located(~Element,request,right,row_capacity,output_capacity,channel,row,column,target,right_within,here,channel_inside,row_inside,column_inside))

def complete(~Element: Data,+request: Request,+bands: Model.Model<Element>,+row_capacity: U32,+output_capacity: U32,+channel: Nat,+row: Nat,+column: Nat,
  target: Target(request,row_capacity,output_capacity),+coverage: BandPlan.Coverage(Model.layout(~Element,bands),0n,Inspection.total_rows(request_geometry(request))),
  total_room: {Nat.is_le(Inspection.total_rows(request_geometry(request)),U32.to_nat(row_capacity)) == True{} : Bool},
  target_fits: {Nat.is_le(Nat.add(U32.to_nat(request_first(request)),U32.to_nat(request_rows(request))),Inspection.total_rows(request_geometry(request))) == True{} : Bool},
  channel_inside: {Nat.is_lt(channel,U32.to_nat(channel_count(request))) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(request_rows(request))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,U32.to_nat(row_width(request))) == True{} : Bool}) ->
  {Readback.contains(~Element,coverage_query(request,channel,row,column),writes(~Element,request,bands)) == True{} : Bool}:
  +total = Inspection.total_rows(request_geometry(request))
  +global = global_row(request,row)
  located(~Element,request,bands,row_capacity,output_capacity,channel,row,column,target,
    BandPlan.covered(Model.layout(~Element,bands),0n,total,0n,U32.to_nat(row_capacity),coverage,{==},total_room),
    BandPlan.locate(Model.layout(~Element,bands),0n,total,global,coverage,
      Bounds.nat_trans(global,Nat.add(U32.to_nat(request_first(request)),U32.to_nat(request_rows(request))),total,
        Order.offset_lt(U32.to_nat(request_first(request)),row,U32.to_nat(request_rows(request)),row_inside),target_fits)),channel_inside,row_inside,column_inside)

def query_inside(+request: Request,+row_capacity: U32,+output_capacity: U32,+channel: Nat,+row: Nat,+column: Nat,
  target: Target(request,row_capacity,output_capacity),channel_inside: {Nat.is_lt(channel,U32.to_nat(channel_count(request))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(request_rows(request))) == True{} : Bool},column_inside: {Nat.is_lt(column,U32.to_nat(row_width(request))) == True{} : Bool}) ->
  {U32.is_lt(coverage_query(request,channel,row,column),output_capacity) == True{} : Bool}:
  Request{Bands.Geometry{images,+channels,height,+width},target_first,+target_rows,+offset} = request
  TargetFields{row_capacity_ok,output_capacity_ok,channels_positive,width_positive,rows_room,storage_room} = target
  +index = RowMajor.index(channel,U32.to_nat(target_rows),U32.to_nat(width),row,column)
  +full = Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(target_rows),U32.to_nat(width)))
  +inside = Bounds.nat_trans(Nat.add(U32.to_nat(offset),index),Nat.add(U32.to_nat(offset),full),U32.to_nat(output_capacity),
    Order.offset_lt(U32.to_nat(offset),index,full,RowMajor.in_bounds(channel,U32.to_nat(channels),U32.to_nat(target_rows),U32.to_nat(width),row,column,channel_inside,row_inside,column_inside)),storage_room)
  Bounds.native_bound(Nat.add(U32.to_nat(offset),index),output_capacity,
    Bounds.roundtrip(Nat.add(U32.to_nat(offset),index),output_capacity,output_capacity_ok,Bounds.strict_le(Nat.add(U32.to_nat(offset),index),U32.to_nat(output_capacity),inside)),inside)

# The destination query of an overlap row decodes to the pixel of the target
# band. The destination base is removed before decoding the local index.
def target_call(request: Request,length: Nat,capacity: U32,division_capacity: U32) -> BandGather.Call:
  Request{geometry,first,rows,offset} = request
  BandGather.Call{geometry,first,rows,length,capacity,division_capacity}

def target_address(+request: Request,+first: U32,+rows: U32,+output_capacity: U32,+length: Nat,+capacity: U32,+division_capacity: U32,
  +image: Nat,+channel: Nat,+row: Nat,+column: Nat,+local_row: Nat,
  target: Destination(request,first,rows,output_capacity),+span: WordInterval.Span<request_first(request),request_rows(request),bounds_begin(request,first),bounds_end(request,first,rows)>,
  facts: BandGather.Facts<target_call(request,length,capacity,division_capacity)>,
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(BandGather.values_channels(target_call(request,length,capacity,division_capacity)))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(BandGather.values_height(target_call(request,length,capacity,division_capacity)))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(BandGather.values_width(target_call(request,length,capacity,division_capacity)))) == True{} : Bool},
  +local_inside: {Nat.is_lt(local_row,U32.to_nat(copied_rows(request,first,rows))) == True{} : Bool},
  located: {BandGather.global_row(request_geometry(request),image,row) == Nat.add(U32.to_nat(bounds_begin(request,first)),local_row) : Nat}) ->
  {Bands.global_index(request_geometry(request),request_first(request),request_rows(request),
      (BandCopy.readback_query(destination_region(request,first,rows,output_capacity),channel,Nat.add(Nat.mul(local_row,U32.to_nat(BandGather.values_width(target_call(request,length,capacity,division_capacity)))),column)) - request_offset(request) : U32)) ==
    U32.from_nat(BandGather.point(request_geometry(request),image,channel,row,column)) : U32}:
  Request{Bands.Geometry{+images,+channels,+height,+width},+target_first,+target_rows,+offset} = request
  +begin = bounds_begin(request,first)
  +end = bounds_end(request,first,rows)
  +shift = (begin - target_first : U32)
  +index = Nat.add(Nat.mul(local_row,U32.to_nat(width)),column)
  +position = (BandCopy.readback_query(destination_region(request,first,rows,output_capacity),channel,index) - offset : U32)
  BandGather.indexed_leaf_address(target_call(request,length,capacity,division_capacity),position,image,channel,row,column,Nat.add(U32.to_nat(shift),local_row),facts,
    BandCopy.relative_pixel_query(offset,shift,(end - begin : U32),channels,target_rows,width,output_capacity,channel,local_row,column,target,channel_inside,local_inside,column_inside),
    channel_inside,row_inside,column_inside,WordInterval.row_inside(target_first,target_rows,begin,end,local_row,span,local_inside),
    WordInterval.located_row(target_first,target_rows,begin,end,BandGather.global_row(request_geometry(request),image,row),local_row,span,located))

# Every range assignment agrees with the logical value at its destination.
# This is the premise used by the shared whole-program readback theorem.
type Evidence<-Element: Data,-request: Request,-first: U32,-rows: U32,-input: Buffer.Model<Element>,-output_capacity: U32,-target_length: Nat,-capacity: U32,-division_capacity: U32> is Data:
  Evidence{source: Source(request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input))),destination: Destination(request,first,rows,output_capacity),
    source_span: WordInterval.Span<first,rows,bounds_begin(request,first),bounds_end(request,first,rows)>,
    target_span: WordInterval.Span<request_first(request),request_rows(request),bounds_begin(request,first),bounds_end(request,first,rows)>,
    target_geometry: BandGather.Facts<target_call(request,target_length,capacity,division_capacity)>}

type MatchingContext<-Logical: Data> is Data:
  MatchingContext{logical: Logical,request: Request}

def value(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,context: MatchingContext<Logical>,query: U32) -> Element:
  MatchingContext{logical,Request{geometry,first,rows,offset}} = context
  original(logical,Bands.global_index(geometry,first,rows,(query - offset : U32)))

# The source row and the destination row of one copied element are the same
# row of the batch, so both name one pixel. Only the destination is decoded.
def copied_value(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+request: Request,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+output_capacity: U32,
  +target_length: Nat,+capacity: U32,+division_capacity: U32,+channel: Nat,+index: Nat,
  +source: Source(request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input))),target: Destination(request,first,rows,output_capacity),
  +source_span: WordInterval.Span<first,rows,bounds_begin(request,first),bounds_end(request,first,rows)>,
  +target_span: WordInterval.Span<request_first(request),request_rows(request),bounds_begin(request,first),bounds_end(request,first,rows)>,
  +target_facts: BandGather.Facts<target_call(request,target_length,capacity,division_capacity)>,
  values: BandValues.Leaf(~Element,~Logical,~original,logical,request_geometry(request),first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input)),
  +channel_inside: {Nat.is_lt(channel,BandCopy.channels(source_region(request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input))))) == True{} : Bool},
  +index_inside: {Nat.is_lt(index,U32.to_nat(copied_elements(request,first,rows))) == True{} : Bool}) ->
  {BandCopy.sample(~Element,Buffer.tree(~Element,input),BandCopy.stride(source_region(request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input)))),source_start(request,first),channel,index) ==
    value(~Element,~Logical,~original,MatchingContext{logical,request},BandCopy.readback_query(destination_region(request,first,rows,output_capacity),channel,index)) : Element}:
  Request{Bands.Geometry{+images,+channels,+height,+width},+target_first,+target_rows,+offset} = request
  BandCopy.Facts{valid,stride_exact,count_exact,offset_exact} = source
  BandGather.Facts{target_valid,target_stride,length_exact,length_bound,division_ok,plane_bound,+height_positive,+width_positive,plane_exact,height_bound,row_span} = target_facts
  +begin = bounds_begin(request,first)
  +end = bounds_end(request,first,rows)
  +shift = (begin - first : U32)
  +local = Nat.div(index,U32.to_nat(width))
  +column = Nat.mod(index,U32.to_nat(width))
  +index_exact = DivisionOrder.quotient_exact(index,U32.to_nat(width),width_positive)
  +local_inside = RowMajor.row_inside(local,U32.to_nat(copied_rows(request,first,rows)),U32.to_nat(width),column,
    %index_exact : {Nat.is_lt(_,Nat.mul(U32.to_nat(copied_rows(request,first,rows)),U32.to_nat(width))) == True{} : Bool}
    %count_exact : {Nat.is_lt(index,_) == True{} : Bool} index_inside)
  +column_inside = DivisionOrder.remainder_small(index,U32.to_nat(width),width_positive)
  +absolute = Nat.add(U32.to_nat(begin),local)
  +image = Nat.div(absolute,U32.to_nat(height))
  +row = Nat.mod(absolute,U32.to_nat(height))
  +query = BandCopy.readback_query(destination_region(request,first,rows,output_capacity),channel,Nat.add(Nat.mul(local,U32.to_nat(width)),column))
  %Equal.sym(Nat,index,Nat.add(Nat.mul(local,U32.to_nat(width)),column),index_exact) :
    {BandCopy.sample(~Element,Buffer.tree(~Element,input),(rows * width : U32),source_start(request,first),channel,_) ==
      value(~Element,~Logical,~original,MatchingContext{logical,request},BandCopy.readback_query(destination_region(request,first,rows,output_capacity),channel,_)) : Element}
  Equal.trans(Element,BandCopy.sample(~Element,Buffer.tree(~Element,input),(rows * width : U32),source_start(request,first),channel,Nat.add(Nat.mul(local,U32.to_nat(width)),column)),
    original(logical,U32.from_nat(BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),Nat.add(U32.to_nat(shift),local)),column))),
    value(~Element,~Logical,~original,MatchingContext{logical,request},query),
    source_value(~Element,~Logical,~original,logical,request,first,rows,input,channel,local,column,source,source_span,values,channel_inside,local_inside,column_inside),
    Equal.cong(U32,Element,address => original(logical,address),U32.from_nat(BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),Nat.add(U32.to_nat(shift),local)),column)),Bands.global_index(Bands.Geometry{images,channels,height,width},target_first,target_rows,(query - offset : U32)),
      Equal.trans(U32,U32.from_nat(BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),Nat.add(U32.to_nat(shift),local)),column)),U32.from_nat(BandGather.point(Bands.Geometry{images,channels,height,width},image,channel,row,column)),Bands.global_index(Bands.Geometry{images,channels,height,width},target_first,target_rows,(query - offset : U32)),
        Equal.cong(Nat,U32,global => U32.from_nat(BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,global,column)),Nat.add(U32.to_nat(first),Nat.add(U32.to_nat(shift),local)),absolute,
          Equal.sym(Nat,absolute,Nat.add(U32.to_nat(first),Nat.add(U32.to_nat(shift),local)),WordInterval.located_row(first,rows,begin,end,absolute,local,source_span,{==}))),
        Equal.sym(U32,Bands.global_index(Bands.Geometry{images,channels,height,width},target_first,target_rows,(query - offset : U32)),U32.from_nat(BandGather.point(Bands.Geometry{images,channels,height,width},image,channel,row,column)),
          target_address(request,first,rows,output_capacity,target_length,capacity,division_capacity,image,channel,row,column,local,target,target_span,target_facts,
            channel_inside,DivisionOrder.remainder_small(absolute,U32.to_nat(height),height_positive),column_inside,local_inside,
            Equal.sym(Nat,absolute,Nat.add(Nat.mul(image,U32.to_nat(height)),row),DivisionOrder.quotient_exact(absolute,U32.to_nat(height),height_positive)))))))

def span_writes(~Element: Data,+request: Request,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+output_capacity: U32,remaining: Nat,+channel: Nat,position: Nat) -> List<&2,Writes.Assignment<Element>>:
  +source = source_region(request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input)))
  Contiguous.writes(~Element,remaining,Buffer.tree(~Element,input),U32.from_nat(position),
    (source_start(request,first) + U32.from_nat(channel) * BandCopy.stride(source) : U32),BandCopy.native_start(destination_region(request,first,rows,output_capacity),channel))

def span(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,remaining: Nat,+position: Nat,+channel: Nat,+query: U32,+logical: Logical,
  +request: Request,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+output_capacity: U32,+target_length: Nat,+capacity: U32,+division_capacity: U32,
  +evidence: Evidence<Element,request,first,rows,input,output_capacity,target_length,capacity,division_capacity>,
  +values: BandValues.Leaf(~Element,~Logical,~original,logical,request_geometry(request),first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input)),
  +channel_inside: {Nat.is_lt(channel,BandCopy.channels(source_region(request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input))))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(position,remaining),U32.to_nat(copied_elements(request,first,rows))) == True{} : Bool}) ->
  Readback.Matching(~Element,query,value(~Element,~Logical,~original,MatchingContext{logical,request},query),span_writes(~Element,request,first,rows,input,output_capacity,remaining,channel,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Evidence{source,target,source_span,target_span,target_geometry} = evidence
      +destination = BandCopy.readback_query(destination_region(request,first,rows,output_capacity),channel,position)
      +actual = BandCopy.sample(~Element,Buffer.tree(~Element,input),BandCopy.stride(source_region(request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input)))),source_start(request,first),channel,position)
      Shape.Both{
        Readback.value_matches(~Element,~MatchingContext<Logical>,~value(~Element,~Logical,~original),MatchingContext{logical,request},destination,query,actual,U32.is_eq(destination,query),{==},
          copied_value(~Element,~Logical,~original,logical,request,first,rows,input,output_capacity,target_length,capacity,division_capacity,channel,position,source,target,source_span,target_span,target_geometry,values,channel_inside,
            Order.room_index(position,rest,U32.to_nat(copied_elements(request,first,rows)),room))),
        %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
          Readback.Matching(~Element,query,value(~Element,~Logical,~original,MatchingContext{logical,request},query),
            Contiguous.writes(~Element,rest,Buffer.tree(~Element,input),_,(source_start(request,first) + U32.from_nat(channel) * BandCopy.stride(source_region(request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input)))) : U32),BandCopy.native_start(destination_region(request,first,rows,output_capacity),channel)))
        span(~Element,~Logical,~original,rest,1n+position,channel,query,logical,request,first,rows,input,output_capacity,target_length,capacity,division_capacity,evidence,values,channel_inside,Order.advance_room(position,rest,U32.to_nat(copied_elements(request,first,rows)),room))}

def channel_writes(~Element: Data,+request: Request,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+output_capacity: U32,remaining: Nat,channel: U32) -> List<&2,Writes.Assignment<Element>>:
  +source = source_region(request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input)))
  +target = destination_region(request,first,rows,output_capacity)
  BandCopy.writes(~Element,remaining,channel,copied_elements(request,first,rows),BandCopy.stride(source),BandCopy.stride(target),source_start(request,first),BandCopy.destination(target),Buffer.tree(~Element,input))

def matching_channels(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,remaining: Nat,+channel: Nat,+query: U32,+logical: Logical,
  +request: Request,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+output_capacity: U32,+target_length: Nat,+capacity: U32,+division_capacity: U32,
  +evidence: Evidence<Element,request,first,rows,input,output_capacity,target_length,capacity,division_capacity>,
  +values: BandValues.Leaf(~Element,~Logical,~original,logical,request_geometry(request),first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input)),
  +room: {Nat.is_le(Nat.add(channel,remaining),BandCopy.channels(source_region(request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input))))) == True{} : Bool}) ->
  Readback.Matching(~Element,query,value(~Element,~Logical,~original,MatchingContext{logical,request},query),channel_writes(~Element,request,first,rows,input,output_capacity,remaining,U32.from_nat(channel))):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      +source = source_region(request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input)))
      +target = destination_region(request,first,rows,output_capacity)
      Readback.matching_append(~Element,query,value(~Element,~Logical,~original,MatchingContext{logical,request},query),
        BandCopy.plane_writes(~Element,U32.from_nat(channel),copied_elements(request,first,rows),BandCopy.stride(source),BandCopy.stride(target),source_start(request,first),BandCopy.destination(target),Buffer.tree(~Element,input)),
        channel_writes(~Element,request,first,rows,input,output_capacity,rest,(U32.from_nat(channel) + 1 : U32)),
        span(~Element,~Logical,~original,U32.to_nat(copied_elements(request,first,rows)),0n,channel,query,logical,request,first,rows,input,output_capacity,target_length,capacity,division_capacity,evidence,values,
          Order.room_index(channel,rest,BandCopy.channels(source),room),Order.reflexive(U32.to_nat(copied_elements(request,first,rows)))),
        %Equal.sym(U32,(U32.from_nat(channel) + 1 : U32),U32.from_nat(1n+channel),Bounds.add_one(U32.from_nat(channel))) :
          Readback.Matching(~Element,query,value(~Element,~Logical,~original,MatchingContext{logical,request},query),channel_writes(~Element,request,first,rows,input,output_capacity,rest,_))
        matching_channels(~Element,~Logical,~original,rest,1n+channel,query,logical,request,first,rows,input,output_capacity,target_length,capacity,division_capacity,evidence,values,Order.advance_room(channel,rest,BandCopy.channels(source),room)))

def copied_leaf(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+query: U32,+logical: Logical,
  +request: Request,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+output_capacity: U32,+target_length: Nat,+capacity: U32,+division_capacity: U32,
  evidence: Evidence<Element,request,first,rows,input,output_capacity,target_length,capacity,division_capacity>,
  values: BandValues.Leaf(~Element,~Logical,~original,logical,request_geometry(request),first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input))) ->
  Readback.Matching(~Element,query,value(~Element,~Logical,~original,MatchingContext{logical,request},query),leaf_writes(~Element,request,first,rows,Buffer.tree(~Element,input))):
  Request{Bands.Geometry{+images,+channels,+height,+width},+target_first,+target_rows,+offset} = request
  matching_channels(~Element,~Logical,~original,U32.to_nat(channels),0n,query,logical,request,first,rows,input,output_capacity,target_length,capacity,division_capacity,evidence,values,Order.reflexive(U32.to_nat(channels)))

def LeafFacts(~Element: Data,+request: Request,+first: U32,+rows: U32,input: Buffer.Model<Element>,output_capacity: U32,target_length: Nat,capacity: U32,division_capacity: U32) -> Data:
  Either<&2,&2,{copied_elements(request,first,rows) == 0 : U32},Evidence<Element,request,first,rows,input,output_capacity,target_length,capacity,division_capacity>>

def MatchingTreeFacts(~Element: Data,+request: Request,bands: Model.Model<Element>,+output_capacity: U32,+target_length: Nat,+capacity: U32,+division_capacity: U32) -> Data:
  match bands:
    case Model.Leaf{input,first,rows}: LeafFacts(~Element,request,first,rows,input,output_capacity,target_length,capacity,division_capacity)
    case Model.Fork{left,right}: Shape.Both<MatchingTreeFacts(~Element,request,left,output_capacity,target_length,capacity,division_capacity),MatchingTreeFacts(~Element,request,right,output_capacity,target_length,capacity,division_capacity)>

def target_geometry(~Element: Data,+request: Request,+output: Buffer.Model<Element>,+limits: Machine.Limits,+row_capacity: U32,+target_length: Nat,+capacity: U32,
  target: Target(request,row_capacity,Tree.capacity(Element,Buffer.tree(~Element,output))),
  +metadata: BandShape.Leaf(request_geometry(request),request_first(request),request_rows(request),target_length),
  positive: {Nat.is_lt(0n,U32.to_nat(request_rows(request))) == True{} : Bool},domain: BandGather.Domain<request_geometry(request),limits,capacity>) ->
  BandGather.Facts<target_call(request,target_length,capacity,Machine.half_word(limits))>:
  Request{Bands.Geometry{+images,+channels,+height,+width},+first,+rows,+offset} = request
  TargetFields{row_capacity_ok,output_capacity_ok,channels_positive,width_positive,rows_room,storage_room} = target
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  BandGather.from_slice(~Element,request_geometry(request),limits,first,rows,target_length,output,capacity,domain,metadata,positive,
    %Equal.sym(Nat,target_length,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),length_exact) : {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,output)))) == True{} : Bool}
    Strided.count_in_stride(U32.to_nat(offset),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,output))),storage_room))

def produced_leaf(~Element: Data,+request: Request,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+limits: Machine.Limits,+output_capacity: U32,+target_length: Nat,+capacity: U32,
  raw: Facts<request,first,rows,Tree.capacity(Element,Buffer.tree(~Element,input)),output_capacity>,target_geometry: BandGather.Facts<target_call(request,target_length,capacity,Machine.half_word(limits))>) ->
  LeafFacts(~Element,request,first,rows,input,output_capacity,target_length,capacity,Machine.half_word(limits)):
  match raw:
    case BoundsEmpty{empty}: Inl{empty}
    case Copied{source,target,source_span,target_span,positive}: Inr{Evidence{source,target,source_span,target_span,target_geometry}}

def produced_tree(~Element: Data,+request: Request,bands: Model.Model<Element>,+limits: Machine.Limits,+output_capacity: U32,+target_length: Nat,+capacity: U32,
  raw: BoundsTreeFacts(~Element,request,output_capacity,bands),+target_geometry: BandGather.Facts<target_call(request,target_length,capacity,Machine.half_word(limits))>) ->
  MatchingTreeFacts(~Element,request,bands,output_capacity,target_length,capacity,Machine.half_word(limits)):
  match bands:
    case Model.Leaf{+input,first,rows}:
      Buffer.Model{+tree,+length,+certificate} = input
      produced_leaf(~Element,request,first,rows,input,limits,output_capacity,target_length,capacity,raw,target_geometry)
    case Model.Fork{+left,+right}:
      Shape.Both{left_raw,right_raw} = raw
      Shape.Both{produced_tree(~Element,request,left,limits,output_capacity,target_length,capacity,left_raw,target_geometry),
        produced_tree(~Element,request,right,limits,output_capacity,target_length,capacity,right_raw,target_geometry)}

def matching_leaf(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+query: U32,+logical: Logical,
  +request: Request,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+output_capacity: U32,+target_length: Nat,+capacity: U32,+division_capacity: U32,
  facts: LeafFacts(~Element,request,first,rows,input,output_capacity,target_length,capacity,division_capacity),
  values: BandValues.Leaf(~Element,~Logical,~original,logical,request_geometry(request),first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input))) ->
  Readback.Matching(~Element,query,value(~Element,~Logical,~original,MatchingContext{logical,request},query),leaf_writes(~Element,request,first,rows,Buffer.tree(~Element,input))):
  match facts:
    case Inl{empty}:
      %Equal.sym(List<&2,Writes.Assignment<Element>>,leaf_writes(~Element,request,first,rows,Buffer.tree(~Element,input)),[],empty_writes(~Element,request,first,rows,Buffer.tree(~Element,input),empty)) : Readback.Matching(~Element,query,value(~Element,~Logical,~original,MatchingContext{logical,request},query),_) Unit{}
    case Inr{evidence}: copied_leaf(~Element,~Logical,~original,query,logical,request,first,rows,input,output_capacity,target_length,capacity,division_capacity,evidence,values)

def whole(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+query: U32,+logical: Logical,
  +request: Request,bands: Model.Model<Element>,+output_capacity: U32,+target_length: Nat,+capacity: U32,+division_capacity: U32,
  facts: MatchingTreeFacts(~Element,request,bands,output_capacity,target_length,capacity,division_capacity),values: BandValues.Values(~Element,~Logical,~original,logical,request_geometry(request),Model.content(~Element,bands))) ->
  Readback.Matching(~Element,query,value(~Element,~Logical,~original,MatchingContext{logical,request},query),writes(~Element,request,bands)):
  match bands:
    case Model.Leaf{+input,first,rows}:
      Buffer.Model{+tree,+length,+certificate} = input
      matching_leaf(~Element,~Logical,~original,query,logical,request,first,rows,input,output_capacity,target_length,capacity,division_capacity,facts,values)
    case Model.Fork{+left,+right}:
      Shape.Both{left_facts,right_facts} = facts
      Shape.Both{left_values,right_values} = values
      Readback.matching_append(~Element,query,value(~Element,~Logical,~original,MatchingContext{logical,request},query),writes(~Element,request,left),writes(~Element,request,right),
        whole(~Element,~Logical,~original,query,logical,request,left,output_capacity,target_length,capacity,division_capacity,left_facts,left_values),
        whole(~Element,~Logical,~original,query,logical,request,right,output_capacity,target_length,capacity,division_capacity,right_facts,right_values))

# The query of a target cell, relative to the destination base, is its position.
def relative_query(+offset: U32,+index: Nat) -> {(U32.from_nat(Nat.add(U32.to_nat(offset),index)) - offset : U32) == U32.from_nat(index) : U32}:
  %Arithmetic.native_roundtrip(offset) : {(U32.from_nat(Nat.add(U32.to_nat(offset),index)) - _ : U32) == U32.from_nat(index) : U32}
  %Order.commutative(index,U32.to_nat(offset)) : {(U32.from_nat(_) - U32.from_nat(U32.to_nat(offset)) : U32) == U32.from_nat(index) : U32}
  Subtraction.encoded_subtract_add(index,U32.to_nat(offset))

# The value matched at the query of a target cell is the value of its pixel.
def cell_value(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+request: Request,+target_length: Nat,+capacity: U32,+division_capacity: U32,
  +channel: Nat,+row: Nat,+column: Nat,facts: BandGather.Facts<target_call(request,target_length,capacity,division_capacity)>,
  metadata: BandShape.Leaf(request_geometry(request),request_first(request),request_rows(request),target_length),
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channel_count(request))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(request_rows(request))) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(row_width(request))) == True{} : Bool}) ->
  {value(~Element,~Logical,~original,MatchingContext{logical,request},coverage_query(request,channel,row,column)) == original(logical,U32.from_nat(BandValues.pixel(request_geometry(request),channel,global_row(request,row),column))) : Element}:
  Request{Bands.Geometry{+images,+channels,+height,+width},+first,+rows,+offset} = request
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  +position = BandValues.position(Bands.Geometry{images,channels,height,width},rows,channel,row,column)
  +pixel = U32.from_nat(BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),row),column))
  Equal.cong(U32,Element,address => original(logical,address),
    Bands.global_index(Bands.Geometry{images,channels,height,width},first,rows,(U32.from_nat(Nat.add(U32.to_nat(offset),position)) - offset : U32)),pixel,
    %Equal.sym(U32,(U32.from_nat(Nat.add(U32.to_nat(offset),position)) - offset : U32),U32.from_nat(position),relative_query(offset,position)) :
      {Bands.global_index(Bands.Geometry{images,channels,height,width},first,rows,_) == pixel : U32}
    BandGather.indexed_pixel(BandGather.Call{Bands.Geometry{images,channels,height,width},first,rows,target_length,capacity,division_capacity},channel,row,column,facts,channel_inside,row_inside,column_inside,
      BandValues.position_within(Bands.Geometry{images,channels,height,width},rows,target_length,channel,row,column,length_exact,channel_inside,row_inside,column_inside)))

# Whole range readback, with actual intersection bounds, planner coverage and
# matching values constructed internally from the two layout invariants.
# Readback of one target cell whose global row lies in some source leaf. The
# sources need not cover the tensor; complete coverage is one such caller.
def located_point(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+request: Request,+bands: Model.Model<Element>,+output: Buffer.Model<Element>,
  +limits: Machine.Limits,+target_length: Nat,+capacity: U32,+domain: BandGather.Domain<request_geometry(request),limits,capacity>,
  +target: Target(request,Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))),
  metadata: BandShape.Contents(~Element,request_geometry(request),Model.content(~Element,bands)),source_ready: Model.Ready(~Element,bands),
  +target_metadata: BandShape.Leaf(request_geometry(request),request_first(request),request_rows(request),target_length),
  ready: Buffer.Sized<Element,output>,values: BandValues.Values(~Element,~Logical,~original,logical,request_geometry(request),Model.content(~Element,bands)),
  +channel: Nat,+row: Nat,+column: Nat,+channel_inside: {Nat.is_lt(channel,U32.to_nat(channel_count(request))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(request_rows(request))) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(row_width(request))) == True{} : Bool},
  within: BandPlan.Within(Model.layout(~Element,bands),0n,U32.to_nat(Machine.capacity(limits))),
  location: BandPlan.Located(Model.layout(~Element,bands),global_row(request,row))) ->
  {Buffer.value(~Element,coverage_query(request,channel,row,column),result(~Element,contents(~Element,run(~Element,request,Model.pack(~Element,bands),Buffer.pack(~Element,output))))) == original(logical,U32.from_nat(BandValues.pixel(request_geometry(request),channel,global_row(request,row),column))) : Element}:
  Request{Bands.Geometry{+images,+channels,+height,+width},+target_first,+target_rows,+offset} = request
  Buffer.Sized{depth_ok,output_room} = ready
  +row_capacity = Machine.capacity(limits)
  +output_capacity = Tree.capacity(Element,Buffer.tree(~Element,output))
  +total_room = BandGather.rows_room(request_geometry(request),limits,capacity,domain)
  +raw = tree(~Element,request,bands,row_capacity,output_capacity,target,metadata,source_ready,total_room)
  +geometry = target_geometry(~Element,request,output,limits,row_capacity,target_length,capacity,target,target_metadata,
    Order.le_lt_trans(0n,row,U32.to_nat(target_rows),Order.zero_le(row),row_inside),domain)
  +facts = produced_tree(~Element,request,bands,limits,output_capacity,target_length,capacity,raw,geometry)
  +query = coverage_query(request,channel,row,column)
  +read = {
    %Equal.sym(Content<Element>,contents(~Element,run(~Element,request,Model.pack(~Element,bands),Buffer.pack(~Element,output))),
      Content{Model.content(~Element,bands),Buffer.Content{Writes.pure(Element,writes(~Element,request,bands),Buffer.tree(~Element,output)),Buffer.length(~Element,output)}},program_range(~Element,request,bands,output)) :
      {Buffer.value(~Element,query,result(~Element,_)) == value(~Element,~Logical,~original,MatchingContext{logical,request},query) : Element}
    Readback.covered(~Element,writes(~Element,request,bands),Buffer.depth(~Element,output),Buffer.tree(~Element,output),query,value(~Element,~Logical,~original,MatchingContext{logical,request},query),
      Buffer.balanced(~Element,output),depth_ok,query_inside(request,row_capacity,output_capacity,channel,row,column,target,channel_inside,row_inside,column_inside),writes_inside(~Element,request,bands,output_capacity,raw),
      whole(~Element,~Logical,~original,query,logical,request,bands,output_capacity,target_length,capacity,Machine.half_word(limits),facts,values),
      located(~Element,request,bands,row_capacity,output_capacity,channel,row,column,target,within,location,channel_inside,row_inside,column_inside)) :
      {Buffer.value(~Element,query,result(~Element,contents(~Element,run(~Element,request,Model.pack(~Element,bands),Buffer.pack(~Element,output))))) == value(~Element,~Logical,~original,MatchingContext{logical,request},query) : Element}}
  Equal.trans(Element,Buffer.value(~Element,query,result(~Element,contents(~Element,run(~Element,request,Model.pack(~Element,bands),Buffer.pack(~Element,output))))),value(~Element,~Logical,~original,MatchingContext{logical,request},query),original(logical,U32.from_nat(BandValues.pixel(request_geometry(request),channel,global_row(request,row),column))),read,
    cell_value(~Element,~Logical,~original,logical,request,target_length,capacity,Machine.half_word(limits),channel,row,column,geometry,target_metadata,channel_inside,row_inside,column_inside))

def semantics_point(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+request: Request,+bands: Model.Model<Element>,+output: Buffer.Model<Element>,
  +limits: Machine.Limits,+target_length: Nat,+capacity: U32,+domain: BandGather.Domain<request_geometry(request),limits,capacity>,
  +target: Target(request,Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))),
  source: BandShape.WellFormed<Element,request_geometry(request),Model.content(~Element,bands)>,
  +target_metadata: BandShape.Leaf(request_geometry(request),request_first(request),request_rows(request),target_length),
  ready: Buffer.Sized<Element,output>,values: BandValues.Values(~Element,~Logical,~original,logical,request_geometry(request),Model.content(~Element,bands)),
  +channel: Nat,+row: Nat,+column: Nat,+channel_inside: {Nat.is_lt(channel,U32.to_nat(channel_count(request))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(request_rows(request))) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(row_width(request))) == True{} : Bool}) ->
  {Buffer.value(~Element,coverage_query(request,channel,row,column),result(~Element,contents(~Element,run(~Element,request,Model.pack(~Element,bands),Buffer.pack(~Element,output))))) == original(logical,U32.from_nat(BandValues.pixel(request_geometry(request),channel,global_row(request,row),column))) : Element}:
  Request{Bands.Geometry{+images,+channels,+height,+width},+target_first,+target_rows,+offset} = request
  BandShape.WellFormed{metadata,source_ready,+coverage} = source
  BandShape.Fields{local,target_fits,length_exact} = target_metadata
  +total = Inspection.total_rows(request_geometry(request))
  +total_room = BandGather.rows_room(request_geometry(request),limits,capacity,domain)
  +global = global_row(request,row)
  +layout_coverage = {%Equal.sym(Bands.Plan,Model.layout(~Element,bands),Model.content_layout(~Element,Model.content(~Element,bands)),Model.layout_matches_content(~Element,bands)) :
    BandPlan.Coverage(_,0n,total) coverage : BandPlan.Coverage(Model.layout(~Element,bands),0n,total)}
  located_point(~Element,~Logical,~original,logical,request,bands,output,limits,target_length,capacity,domain,target,metadata,Model.ready(~Element,bands,source_ready),
    target_metadata,ready,values,channel,row,column,channel_inside,row_inside,column_inside,
    BandPlan.covered(Model.layout(~Element,bands),0n,total,0n,U32.to_nat(Machine.capacity(limits)),layout_coverage,{==},total_room),
    BandPlan.locate(Model.layout(~Element,bands),0n,total,global,layout_coverage,
      Bounds.nat_trans(global,Nat.add(U32.to_nat(target_first),U32.to_nat(target_rows)),total,
        Order.offset_lt(U32.to_nat(target_first),row,U32.to_nat(target_rows),row_inside),target_fits)))

# The cell of the canonical written content when the sources cover the tensor.
def written_point(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+request: Request,+bands: Model.Model<Element>,+output: Buffer.Model<Element>,
  +limits: Machine.Limits,+target_length: Nat,+capacity: U32,domain: BandGather.Domain<request_geometry(request),limits,capacity>,
  target: Target(request,Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))),
  source: BandShape.WellFormed<Element,request_geometry(request),Model.content(~Element,bands)>,
  target_metadata: BandShape.Leaf(request_geometry(request),request_first(request),request_rows(request),target_length),
  ready: Buffer.Sized<Element,output>,values: BandValues.Values(~Element,~Logical,~original,logical,request_geometry(request),Model.content(~Element,bands)),
  +channel: Nat,+row: Nat,+column: Nat,channel_inside: {Nat.is_lt(channel,U32.to_nat(channel_count(request))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(request_rows(request))) == True{} : Bool},column_inside: {Nat.is_lt(column,U32.to_nat(row_width(request))) == True{} : Bool}) ->
  {Buffer.value(~Element,coverage_query(request,channel,row,column),result(~Element,written(~Element,request,Model.content(~Element,bands),Buffer.content(~Element,output)))) == original(logical,U32.from_nat(BandValues.pixel(request_geometry(request),channel,global_row(request,row),column))) : Element}:
  %canonical_range(~Element,request,bands,output) :
    {Buffer.value(~Element,coverage_query(request,channel,row,column),result(~Element,_)) == original(logical,U32.from_nat(BandValues.pixel(request_geometry(request),channel,global_row(request,row),column))) : Element}
  semantics_point(~Element,~Logical,~original,logical,request,bands,output,limits,target_length,capacity,domain,target,source,target_metadata,ready,values,channel,row,column,channel_inside,row_inside,column_inside)

# A source leaf for every target row.
def RowLocations(~Element: Data,+request: Request,+bands: Model.Model<Element>) -> Data:
  Members.All(Nat,{row => BandPlan.Located(Model.layout(~Element,bands),global_row(request,row)) : Nat -> Data},Ranges.range(U32.to_nat(request_rows(request)),0n))

# The same cell when each target row has a source leaf, as for a window
# assembled from its core and halos.
def located_written_point(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+request: Request,+bands: Model.Model<Element>,+output: Buffer.Model<Element>,
  +limits: Machine.Limits,+target_length: Nat,+capacity: U32,domain: BandGather.Domain<request_geometry(request),limits,capacity>,
  target: Target(request,Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))),
  metadata: BandShape.Contents(~Element,request_geometry(request),Model.content(~Element,bands)),source_ready: Model.Ready(~Element,bands),
  target_metadata: BandShape.Leaf(request_geometry(request),request_first(request),request_rows(request),target_length),
  ready: Buffer.Sized<Element,output>,values: BandValues.Values(~Element,~Logical,~original,logical,request_geometry(request),Model.content(~Element,bands)),
  within: BandPlan.Within(Model.layout(~Element,bands),0n,U32.to_nat(Machine.capacity(limits))),locations: RowLocations(~Element,request,bands),
  +channel: Nat,+row: Nat,+column: Nat,+channel_inside: {Nat.is_lt(channel,U32.to_nat(channel_count(request))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(request_rows(request))) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(row_width(request))) == True{} : Bool}) ->
  {Buffer.value(~Element,coverage_query(request,channel,row,column),result(~Element,written(~Element,request,Model.content(~Element,bands),Buffer.content(~Element,output)))) == original(logical,U32.from_nat(BandValues.pixel(request_geometry(request),channel,global_row(request,row),column))) : Element}:
  %canonical_range(~Element,request,bands,output) :
    {Buffer.value(~Element,coverage_query(request,channel,row,column),result(~Element,_)) == original(logical,U32.from_nat(BandValues.pixel(request_geometry(request),channel,global_row(request,row),column))) : Element}
  located_point(~Element,~Logical,~original,logical,request,bands,output,limits,target_length,capacity,domain,target,metadata,source_ready,target_metadata,ready,values,
    channel,row,column,channel_inside,row_inside,column_inside,within,
    Members.all_member(Nat,{row => BandPlan.Located(Model.layout(~Element,bands),global_row(request,row)) : Nat -> Data},row,Ranges.range(U32.to_nat(request_rows(request)),0n),locations,
      Members.range_member(U32.to_nat(request_rows(request)),0n,row,row_inside)))
