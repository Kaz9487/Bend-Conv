# The values of the gather every layout operation shares: one checked copy
# through a source view into fresh storage. When the call returns Ready, the
# result is well formed and holds, at every index, the logical input at the
# address the coordinate model gives to that index of the source view. The
# Ready result is the only premise about the call; its guards are read from
# the call itself. An operation proves only the address map of its own view.
import Base
import ../tensor.bend as Tensors
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ../tensor_layout.bend as Layout
import ../tensor_operations.bend as Operations
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./storage_shape.bend as Shape
import ./storage_allocation_refinement.bend as AllocationLaws
import ./tensor_dense_contract.bend as Dense
import ./tensor_copy_values.bend as CopyValues
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_view_metadata_refinement.bend as Metadata
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_refinement.bend as Count
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_allocation_depth.bend as Depth
import ./tensor_band_count.bend as BandCount
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./write_program.bend as Writes

# The logical input read through a view.
type Gather<-Logical: Data> is Data:
  Gather{logical: Logical,source: Views.View}

def value(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,context: Gather<Logical>,index: U32) -> Element:
  Gather{logical,source} = context
  original(logical,CopyValues.address(source,U32.to_nat(index)))

# One index of the copy program into a target at offset zero.
def cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: Views.View,+target_axes: List<&2,Views.Axis>,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+count: U32,+index: Nat,
  input_ready: Buffer.Ready<Element,input>,ready: Buffer.Ready<Element,output>,values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +written: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == U32.to_nat(count) : Nat},
  fits: {Views.fits(source,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool},
  +inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  Dense.At(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},CopyValues.program_content(~Element,count,source,Views.View{target_axes,0},Buffer.content(~Element,input),Buffer.content(~Element,output)),index):
  Buffer.Model{+input_tree,+input_length,+input_certificate} = input
  Buffer.Model{+output_tree,+output_length,+output_certificate} = output
  Buffer.Ready{+input_depth_ok,+input_room} = input_ready
  Buffer.Ready{+depth_ok,+room} = ready
  +input = {Buffer.Model{input_tree,input_length,input_certificate} : Buffer.Model<Element>}
  +output = {Buffer.Model{output_tree,output_length,output_certificate} : Buffer.Model<Element>}
  +capacity = Tree.capacity(Element,output_tree)
  +capacity_ok = Buffer.capacity_bound(~Element,output,depth_ok)
  +count_room = {
    %Equal.sym(Nat,U32.to_nat(count),output_length,written) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    room : {Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)) == True{} : Bool}}
  +encoded = Bounds.roundtrip(index,capacity,capacity_ok,
    Bounds.strict_le(index,U32.to_nat(capacity),Bounds.nat_trans(index,U32.to_nat(count),U32.to_nat(capacity),inside,count_room)))
  +nonempty = CountCompletion.positive_not_zero(Views.axis_extents_of(source),
    %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents_of(source)),U32.to_nat(count),exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.le_lt_trans(0n,index,U32.to_nat(count),Order.zero_le(index),inside))
  +address = CopyValues.address(source,index)
  +address_inside = {
    %Bounds.roundtrip(input_length,Tree.capacity(Element,input_tree),Buffer.capacity_bound(~Element,input,input_depth_ok),input_room) :
      {Nat.is_lt(U32.to_nat(address),_) == True{} : Bool}
    CopyValues.address_inside(source,index,U32.from_nat(input_length),
      %nonempty : {Views.fits_nonempty(_,source,U32.from_nat(input_length)) == True{} : Bool}
      fits) : {Nat.is_lt(U32.to_nat(address),input_length) == True{} : Bool}}
  Equal.trans(Element,
    Tree.read(Element,Writes.pure(Element,CopyValues.copy_writes(~Element,Views.is_contiguous(source),U32.to_nat(count),source,0,input_tree),output_tree),U32.from_nat(index)),
    Tree.read(Element,input_tree,address),
    original(logical,CopyValues.address(source,U32.to_nat(U32.from_nat(index)))),
    CopyValues.copied_value(~Element,source,input_tree,output_tree,output_length,U32.to_nat(count),index,
      Buffer.content_ready(~Element,output,Buffer.Ready{depth_ok,room}),capacity_ok,count_room,inside,exact),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,encoded) : {Tree.read(Element,input_tree,address) == original(logical,CopyValues.address(source,_)) : Element}
    %Arithmetic.native_roundtrip(address) : {Tree.read(Element,input_tree,_) == original(logical,_) : Element}
    Dense.read(~Element,~Logical,~original,logical,Buffer.Content{input_tree,input_length},U32.to_nat(address),values,address_inside))

def cells_range(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,remaining: Nat,+position: Nat,+logical: Logical,+source: Views.View,+target_axes: List<&2,Views.Axis>,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+count: U32,
  +input_ready: Buffer.Ready<Element,input>,+ready: Buffer.Ready<Element,output>,+values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +written: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == U32.to_nat(count) : Nat},
  +fits: {Views.fits(source,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(position,remaining),U32.to_nat(count)) == True{} : Bool}) ->
  Members.All(Nat,{index => Dense.At(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},CopyValues.program_content(~Element,count,source,Views.View{target_axes,0},Buffer.content(~Element,input),Buffer.content(~Element,output)),index) : Nat -> Data},Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{cell(~Element,~Logical,~original,logical,source,target_axes,input,output,count,position,input_ready,ready,values,written,exact,fits,Order.room_index(position,rest,U32.to_nat(count),room)),
        cells_range(~Element,~Logical,~original,rest,1n+position,logical,source,target_axes,input,output,count,input_ready,ready,values,written,exact,fits,Order.advance_room(position,rest,U32.to_nat(count),room))}

def program_length(~Element: Data,+count: U32,+source: Views.View,+target: Views.View,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>) ->
  {Buffer.content_length(~Element,CopyValues.program_content(~Element,count,source,target,Buffer.content(~Element,input),Buffer.content(~Element,output))) ==
    Buffer.length(~Element,output) : Nat}:
  Buffer.Model{input_tree,input_length,input_certificate} = input
  Buffer.Model{output_tree,output_length,output_certificate} = output
  {==}

def program_ready(~Element: Data,+count: U32,+source: Views.View,+target: Views.View,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,ready: Buffer.Ready<Element,output>) ->
  Buffer.ContentReady<Element,CopyValues.program_content(~Element,count,source,target,Buffer.content(~Element,input),Buffer.content(~Element,output))>:
  Buffer.Model{+input_tree,input_length,input_certificate} = input
  Buffer.Model{+output_tree,+output_length,+output_certificate} = output
  Writes.content_ready(~Element,CopyValues.copy_writes(~Element,Views.is_contiguous(source),U32.to_nat(count),source,CopyValues.view_offset(target),input_tree),output_tree,output_length,
    Buffer.content_ready(~Element,Buffer.Model{output_tree,output_length,output_certificate},ready))

def program_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: Views.View,+target_axes: List<&2,Views.Axis>,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+count: U32,
  +input_ready: Buffer.Ready<Element,input>,+ready: Buffer.Ready<Element,output>,+values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +written: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == U32.to_nat(count) : Nat},
  +fits: {Views.fits(source,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool}) ->
  Dense.Values(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},CopyValues.program_content(~Element,count,source,Views.View{target_axes,0},Buffer.content(~Element,input),Buffer.content(~Element,output))):
  %Equal.sym(Nat,Buffer.content_length(~Element,CopyValues.program_content(~Element,count,source,Views.View{target_axes,0},Buffer.content(~Element,input),Buffer.content(~Element,output))),U32.to_nat(count),
    Equal.trans(Nat,Buffer.content_length(~Element,CopyValues.program_content(~Element,count,source,Views.View{target_axes,0},Buffer.content(~Element,input),Buffer.content(~Element,output))),Buffer.length(~Element,output),U32.to_nat(count),
      program_length(~Element,count,source,Views.View{target_axes,0},input,output),Equal.sym(Nat,U32.to_nat(count),Buffer.length(~Element,output),written))) :
    Members.All(Nat,{index => Dense.At(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},CopyValues.program_content(~Element,count,source,Views.View{target_axes,0},Buffer.content(~Element,input),Buffer.content(~Element,output)),index) : Nat -> Data},Ranges.range(_,0n))
  cells_range(~Element,~Logical,~original,U32.to_nat(count),0n,logical,source,target_axes,input,output,count,input_ready,ready,values,written,exact,fits,Order.reflexive(U32.to_nat(count)))

# What a copied tensor needs from its storage content.
def Facts(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,-content: Buffer.Content<Element>) -> Data:
  Shape.Both<Dense.WellFormed<Element,shape,content>,Dense.Values(~Element,~Context,~reader,context,content)>

def program_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: Views.View,+target_axes: List<&2,Views.Axis>,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+target_shape: List<&2,U32>,+count: U32,
  +input_ready: Buffer.Ready<Element,input>,+ready: Buffer.Ready<Element,output>,+values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +written: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == U32.to_nat(count) : Nat},
  +fits: {Views.fits(source,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool},
  length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(target_shape) : Nat}) ->
  Facts(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},target_shape,CopyValues.program_content(~Element,count,source,Views.View{target_axes,0},Buffer.content(~Element,input),Buffer.content(~Element,output))):
  Shape.Both{
    Dense.WellFormed{
      Equal.trans(Nat,Buffer.content_length(~Element,CopyValues.program_content(~Element,count,source,Views.View{target_axes,0},Buffer.content(~Element,input),Buffer.content(~Element,output))),Buffer.length(~Element,output),ShapeLaws.elements(target_shape),
        program_length(~Element,count,source,Views.View{target_axes,0},input,output),length_exact),
      program_ready(~Element,count,source,Views.View{target_axes,0},input,output,ready)},
    program_values(~Element,~Logical,~original,logical,source,target_axes,input,output,count,input_ready,ready,values,written,exact,fits)}

# The observed outcome of the checked copy decides the status of the result.
def accepted_owners(~Element: Data,outcome: Operations.Outcome<OperationsLaws.PairOwners(Element)>) -> Bool:
  match outcome:
    case Operations.Accepted{value}: True{}
    case Operations.Rejected{value}: False{}

def ready_accepted(~Element: Data,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,+name: String,+source_shape: List<&2,U32>,+target_shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.copied(~Element,name,source_shape,target_shape,fill,workers,outcome))) == accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome)) : Bool}:
  match outcome:
    case Operations.Accepted{(input,output)}: {==}
    case Operations.Rejected{(input,output)}: {==}

def copied_contract(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,+context: Context,+name: String,+source_shape: List<&2,U32>,
  +target_shape: List<&2,U32>,+fill: Element,+workers: U32,
  facts: Facts(~Element,~Context,~reader,context,target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome))),
  accepted: {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.copied(~Element,name,source_shape,target_shape,fill,workers,outcome))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Context,~reader,context,Layout.result_of(~Element,Layout.copied(~Element,name,source_shape,target_shape,fill,workers,outcome))):
  match outcome:
    case Operations.Accepted{(input,Storage.Buffer{storage,length,certificate})}:
      Shape.Both{well_formed,values} = facts
      Shape.Both{Shape.Both{well_formed,{==}},values}
    case Operations.Rejected{(input,output)}:
      Empty.absurd(Dense.Contract(~Element,~Context,~reader,context,Layout.result_of(~Element,Layout.copied(~Element,name,source_shape,target_shape,fill,workers,Operations.Rejected{(input,output)}))),WordEncoding.false_true(accepted))

# An accepted copy model passed its guard.
def model_guard(~Element: Data,valid: Bool,count: Maybe<&2,U32>,+source: Views.View,+target: Views.View,+input: Tree.Tree<Element>,+input_length: Nat,
  +output: Tree.Tree<Element>,+output_length: Nat,
  accepted: {accepted_owners(~Element,OperationsLaws.copy_model(~Element,valid,count,source,target,input,input_length,output,output_length)) == True{} : Bool}) ->
  {valid == True{} : Bool}:
  match valid count:
    case True{} Some{count}: {==}
    case True{} None{}: {==}
    case False{} Some{count}: accepted
    case False{} None{}: accepted

type Guarded<-Element: Data,-source: Views.View,-target: Views.View,-input: Buffer.Model<Element>,-output: Buffer.Model<Element>> is Data:
  Guarded{guard: {CopyValues.copy_guard(~Element,Buffer.pack(~Element,input),source,Buffer.pack(~Element,output),target) == True{} : Bool},
    same: {Shapes.same(Views.shape(source),Views.shape(target)) == True{} : Bool},
    fits: {Views.fits(source,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool}}

def guarded(~Element: Data,+source: Views.View,+target: Views.View,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,
  accepted: {accepted_owners(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,Buffer.pack(~Element,output),target)) == True{} : Bool}) ->
  Guarded<Element,source,target,input,output>:
  Buffer.Model{+input_tree,+input_length,Certificate.Certificate{+input_depth,+input_equation}} = input
  Buffer.Model{+output_tree,+output_length,Certificate.Certificate{+output_depth,+output_equation}} = output
  +same = Shapes.same(Views.shape(source),Views.shape(target))
  +contiguous = Views.is_contiguous(target)
  +input_valid = Operations.valid_buffer(input_length,U32.shln(1,input_depth),source)
  +output_valid = Operations.valid_buffer(output_length,U32.shln(1,output_depth),target)
  +truth = model_guard(~Element,(same && contiguous && input_valid && output_valid : Bool),Views.numel(target),source,target,
    Tree.reflect(Element,Tree.pack(Element,input_tree)),input_length,Tree.reflect(Element,Tree.pack(Element,output_tree)),output_length,accepted)
  +leading = Boolean.left((same && contiguous && input_valid : Bool),output_valid,truth)
  Guarded{truth,
    Boolean.left(same,contiguous,Boolean.left((same && contiguous : Bool),input_valid,leading)),
    Boolean.right(Nat.is_le(input_length,U32.to_nat(U32.shln(1,input_depth))),Views.fits(source,U32.from_nat(input_length)),
      Boolean.right((same && contiguous : Bool),input_valid,leading))}

def same_extents(+source: Views.View,+target: Views.View,same: {Shapes.same(Views.shape(source),Views.shape(target)) == True{} : Bool}) ->
  {Views.axis_extents_of(source) == Views.axis_extents_of(target) : List<&2,U32>}:
  Views.View{+source_axes,source_offset} = source
  Views.View{+target_axes,target_offset} = target
  ShapeLaws.same_extents(Views.axis_extents(source_axes),Views.axis_extents(target_axes),same)

def guarded_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: Views.View,+target_axes: List<&2,Views.Axis>,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+target_shape: List<&2,U32>,+count: U32,
  +input_ready: Buffer.Ready<Element,input>,+ready: Buffer.Ready<Element,output>,+values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +written: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +target_exact: {ShapeLaws.elements(Views.axis_extents(target_axes)) == U32.to_nat(count) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(target_shape) : Nat},
  facts: Guarded<Element,source,Views.View{target_axes,0},input,output>) ->
  Facts(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Buffer.pack(~Element,output),Views.View{target_axes,0})))):
  Guarded{+guard,+same,+fits} = facts
  +exact = {
    %Equal.sym(List<&2,U32>,Views.axis_extents_of(source),Views.axis_extents(target_axes),same_extents(source,Views.View{target_axes,0},same)) : {ShapeLaws.elements(_) == U32.to_nat(count) : Nat}
    target_exact : {ShapeLaws.elements(Views.axis_extents_of(source)) == U32.to_nat(count) : Nat}}
  %Equal.sym(Operations.Outcome<OperationsLaws.PairOwners(Element)>,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Buffer.pack(~Element,output),Views.View{target_axes,0})),OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,Buffer.pack(~Element,output),Views.View{target_axes,0}),OperationsLaws.copy(~Element,Buffer.pack(~Element,input),source,Buffer.pack(~Element,output),Views.View{target_axes,0})) :
    Facts(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},target_shape,CopyValues.right_content(~Element,_))
  %Equal.sym(Buffer.Content<Element>,CopyValues.right_content(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,Buffer.pack(~Element,output),Views.View{target_axes,0})),
    CopyValues.program_content(~Element,count,source,Views.View{target_axes,0},Buffer.contents(~Element,Buffer.pack(~Element,input)),Buffer.contents(~Element,Buffer.pack(~Element,output))),
    CopyValues.copy_observed_content(~Element,Buffer.pack(~Element,input),source,Buffer.pack(~Element,output),Views.View{target_axes,0},count,guard,CountCompletion.numel(Views.axis_extents(target_axes),count,target_exact))) :
    Facts(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},target_shape,_)
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,input)),Buffer.content(~Element,input),Buffer.packed_contents(~Element,input)) :
    Facts(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},target_shape,CopyValues.program_content(~Element,count,source,Views.View{target_axes,0},_,Buffer.contents(~Element,Buffer.pack(~Element,output))))
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,output)),Buffer.content(~Element,output),Buffer.packed_contents(~Element,output)) :
    Facts(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},target_shape,CopyValues.program_content(~Element,count,source,Views.View{target_axes,0},Buffer.content(~Element,input),_))
  program_facts(~Element,~Logical,~original,logical,source,target_axes,input,output,target_shape,count,input_ready,ready,values,written,exact,fits,length_exact)

def modeled_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: Views.View,+target_axes: List<&2,Views.Axis>,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+target_shape: List<&2,U32>,+count: U32,
  +input_ready: Buffer.Ready<Element,input>,+ready: Buffer.Ready<Element,output>,+values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +written: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +target_exact: {ShapeLaws.elements(Views.axis_extents(target_axes)) == U32.to_nat(count) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(target_shape) : Nat},
  accepted: {accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Buffer.pack(~Element,output),Views.View{target_axes,0}))) == True{} : Bool}) ->
  Facts(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Buffer.pack(~Element,output),Views.View{target_axes,0})))):
  guarded_facts(~Element,~Logical,~original,logical,source,target_axes,input,output,target_shape,count,input_ready,ready,values,written,target_exact,length_exact,
    guarded(~Element,source,Views.View{target_axes,0},input,output,
      %OperationsLaws.copy(~Element,Buffer.pack(~Element,input),source,Buffer.pack(~Element,output),Views.View{target_axes,0}) : {accepted_owners(~Element,_) == True{} : Bool}
      accepted))

# The fresh output of the public call, observed as a model.
def allocated_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: Views.View,+target_axes: List<&2,Views.Axis>,+input: Buffer.Model<Element>,+view_shape: List<&2,U32>,
  +target_shape: List<&2,U32>,+fill: Element,+count: U32,
  observation: Buffer.Observation<Element,Tensors.buffer(~Element,fill,count)>,evidence: Count.Count<ShapeLaws.elements(view_shape),count>,
  +input_ready: Buffer.Ready<Element,input>,+values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +target_extents: {Views.axis_extents(target_axes) == view_shape : List<&2,U32>},
  +same_elements: {ShapeLaws.elements(view_shape) == ShapeLaws.elements(target_shape) : Nat},
  accepted: {accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),Views.View{target_axes,0}))) == True{} : Bool}) ->
  Facts(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),Views.View{target_axes,0})))):
  Buffer.Observation{+output,+equation} = observation
  Count.Count{+exact,+bounded} = evidence
  +fresh = {Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)} : Buffer.Content<Element>}
  +allocated = Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,output),Buffer.contents(~Element,Buffer.pack(~Element,output)),fresh,
    Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,output)),Buffer.content(~Element,output),Buffer.packed_contents(~Element,output)),
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,output),Tensors.buffer(~Element,fill,count),equation) : {Buffer.contents(~Element,_) == fresh : Buffer.Content<Element>}
    AllocationLaws.contents(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)))
  +length_count = Equal.trans(Nat,Buffer.length(~Element,output),Buffer.content_length(~Element,Buffer.content(~Element,output)),U32.to_nat(count),
    Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,output)),Buffer.length(~Element,output),Buffer.content_length_of(~Element,output)),
    Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.content(~Element,output),fresh,allocated))
  %equation : Facts(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,_,Views.View{target_axes,0}))))
  modeled_facts(~Element,~Logical,~original,logical,source,target_axes,input,output,target_shape,count,input_ready,
    Buffer.ready_observed(~Element,Tensors.buffer(~Element,fill,count),output,equation,BandCount.allocated_ready(~Element,count,fill,Depth.bounded(count,bounded))),
    values,Equal.sym(Nat,Buffer.length(~Element,output),U32.to_nat(count),length_count),
    %Equal.sym(List<&2,U32>,Views.axis_extents(target_axes),view_shape,target_extents) : {ShapeLaws.elements(_) == U32.to_nat(count) : Nat}
    exact,
    Equal.trans(Nat,Buffer.length(~Element,output),U32.to_nat(count),ShapeLaws.elements(target_shape),length_count,
      Equal.trans(Nat,U32.to_nat(count),ShapeLaws.elements(view_shape),ShapeLaws.elements(target_shape),
        Equal.sym(Nat,ShapeLaws.elements(view_shape),U32.to_nat(count),exact),same_elements)),
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,output),Tensors.buffer(~Element,fill,count),equation) : {accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,_,Views.View{target_axes,0}))) == True{} : Bool}
    accepted)

# The canonical view of the fresh storage starts at offset zero and has the
# extents it was built from.
def targeted_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,target: Views.View,+logical: Logical,+source: Views.View,+input: Buffer.Model<Element>,+view_shape: List<&2,U32>,
  +target_shape: List<&2,U32>,+fill: Element,+count: U32,
  +input_ready: Buffer.Ready<Element,input>,+values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +targets: {Tensors.view_of(view_shape) == Some{target} : Maybe<&2,Views.View>},
  +counts: {Tensors.count_of(view_shape) == Some{count} : Maybe<&2,U32>},
  same_elements: @+count: U32 -> {Tensors.count_of(view_shape) == Some{count} : Maybe<&2,U32>} -> {ShapeLaws.elements(view_shape) == ShapeLaws.elements(target_shape) : Nat},
  accepted: {accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target))) == True{} : Bool}) ->
  Facts(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target)))):
  Views.View{+target_axes,+target_offset} = target
  +offset_zero = {
    %targets : {Metadata.observed_offset(_,0) == 0 : U32}
    Metadata.contiguous_count_offset(Shapes.numel(Shapes.Shape{view_shape}),view_shape,0) : {target_offset == 0 : U32}}
  +target_extents = {
    %targets : {Metadata.observed_extents(_,view_shape) == view_shape : List<&2,U32>}
    Metadata.contiguous_shape(view_shape,0) : {Views.axis_extents(target_axes) == view_shape : List<&2,U32>}}
  %Equal.sym(U32,target_offset,0,offset_zero) : Facts(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),Views.View{target_axes,_}))))
  allocated_facts(~Element,~Logical,~original,logical,source,target_axes,input,view_shape,target_shape,fill,count,
    Buffer.observe(~Element,Tensors.buffer(~Element,fill,count)),Count.count_of(view_shape,count,counts),input_ready,values,target_extents,same_elements(count,counts),
    %offset_zero : {accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),Views.View{target_axes,_}))) == True{} : Bool}
    accepted)

def observed_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: Views.View,+target: Views.View,+view_shape: List<&2,U32>,-storage: Storage.Buffer<Element>,
  +target_shape: List<&2,U32>,+fill: Element,+count: U32,
  observation: Buffer.Observation<Element,storage>,
  +targets: {Tensors.view_of(view_shape) == Some{target} : Maybe<&2,Views.View>},
  +counts: {Tensors.count_of(view_shape) == Some{count} : Maybe<&2,U32>},
  ready: Buffer.ContentReady<Element,Buffer.contents(~Element,storage)>,
  values: Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,storage)),
  same_elements: @+count: U32 -> {Tensors.count_of(view_shape) == Some{count} : Maybe<&2,U32>} -> {ShapeLaws.elements(view_shape) == ShapeLaws.elements(target_shape) : Nat},
  accepted: {accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,storage,source,Tensors.buffer(~Element,fill,count),target))) == True{} : Bool}) ->
  Facts(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,storage,source,Tensors.buffer(~Element,fill,count),target)))):
  Buffer.Observation{+input,+equation} = observation
  %equation : Facts(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,_,source,Tensors.buffer(~Element,fill,count),target))))
  targeted_facts(~Element,~Logical,~original,target,logical,source,input,view_shape,target_shape,fill,count,
    Buffer.ready_content(~Element,input,
      %Buffer.packed_contents(~Element,input) : Buffer.ContentReady<Element,_>
      %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,input),storage,equation) : Buffer.ContentReady<Element,Buffer.contents(~Element,_)>
      ready),
    %Buffer.packed_contents(~Element,input) : Dense.Values(~Element,~Logical,~original,logical,_)
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,input),storage,equation) : Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,_))
    values,
    targets,counts,same_elements,
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,input),storage,equation) : {accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,_,source,Tensors.buffer(~Element,fill,count),target))) == True{} : Bool}
    accepted)

# The copy step of any operation: an accepted checked copy of a storage with
# complete values, through a source view into fresh storage laid out as
# view_shape, has a well-formed content that holds the gathered values.
def copy_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: Views.View,+target: Views.View,+view_shape: List<&2,U32>,storage: Storage.Buffer<Element>,
  +target_shape: List<&2,U32>,+fill: Element,+count: U32,
  +targets: {Tensors.view_of(view_shape) == Some{target} : Maybe<&2,Views.View>},
  +counts: {Tensors.count_of(view_shape) == Some{count} : Maybe<&2,U32>},
  ready: Buffer.ContentReady<Element,Buffer.contents(~Element,storage)>,
  values: Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,storage)),
  same_elements: @+count: U32 -> {Tensors.count_of(view_shape) == Some{count} : Maybe<&2,U32>} -> {ShapeLaws.elements(view_shape) == ShapeLaws.elements(target_shape) : Nat},
  accepted: {accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,storage,source,Tensors.buffer(~Element,fill,count),target))) == True{} : Bool}) ->
  Facts(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,storage,source,Tensors.buffer(~Element,fill,count),target)))):
  observed_facts(~Element,~Logical,~original,logical,source,target,view_shape,storage,target_shape,fill,count,Buffer.observe(~Element,storage),targets,counts,ready,values,same_elements,accepted)

# The public selection: a Ready result has its target view and its count.
def selected(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,target: Maybe<&2,Views.View>,count: Maybe<&2,U32>,+logical: Logical,+name: String,+source: Views.View,+input: Buffer.Model<Element>,
  +view_shape: List<&2,U32>,+source_shape: List<&2,U32>,+target_shape: List<&2,U32>,+fill: Element,+workers: U32,
  +input_ready: Buffer.Ready<Element,input>,+values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +targets: {Tensors.view_of(view_shape) == target : Maybe<&2,Views.View>},
  +counts: {Tensors.count_of(view_shape) == count : Maybe<&2,U32>},
  same_elements: @+count: U32 -> {Tensors.count_of(view_shape) == Some{count} : Maybe<&2,U32>} -> {ShapeLaws.elements(view_shape) == ShapeLaws.elements(target_shape) : Nat},
  +accepted: {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.gather_views(~Element,name,Some{source},target,count,Buffer.pack(~Element,input),source_shape,target_shape,fill,workers))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},Layout.result_of(~Element,Layout.gather_views(~Element,name,Some{source},target,count,Buffer.pack(~Element,input),source_shape,target_shape,fill,workers))):
  match target count:
    case Some{+target} Some{+count}:
      copied_contract(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target),Gather{logical,source},name,source_shape,target_shape,fill,workers,
        targeted_facts(~Element,~Logical,~original,target,logical,source,input,view_shape,target_shape,fill,count,input_ready,values,targets,counts,same_elements,
          Equal.trans(Bool,accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target))),Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.copied(~Element,name,source_shape,target_shape,fill,workers,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target)))),True{},
            Equal.sym(Bool,Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.copied(~Element,name,source_shape,target_shape,fill,workers,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target)))),accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target))),
              ready_accepted(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,Tensors.buffer(~Element,fill,count),target),name,source_shape,target_shape,fill,workers)),accepted)),
        accepted)
    case Some{target} None{}: Empty.absurd(Dense.Contract(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},Layout.result_of(~Element,Layout.gather_views(~Element,name,Some{source},Some{target},None{},Buffer.pack(~Element,input),source_shape,target_shape,fill,workers))),WordEncoding.false_true(accepted))
    case None{} Some{count}: Empty.absurd(Dense.Contract(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},Layout.result_of(~Element,Layout.gather_views(~Element,name,Some{source},None{},Some{count},Buffer.pack(~Element,input),source_shape,target_shape,fill,workers))),WordEncoding.false_true(accepted))
    case None{} None{}: Empty.absurd(Dense.Contract(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},Layout.result_of(~Element,Layout.gather_views(~Element,name,Some{source},None{},None{},Buffer.pack(~Element,input),source_shape,target_shape,fill,workers))),WordEncoding.false_true(accepted))

def observed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+name: String,+source: Views.View,+view_shape: List<&2,U32>,-storage: Storage.Buffer<Element>,
  +source_shape: List<&2,U32>,+target_shape: List<&2,U32>,+fill: Element,+workers: U32,
  observation: Buffer.Observation<Element,storage>,
  contract: Dense.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{storage,source_shape,fill,Tensors.Ready{},workers}),
  same_elements: @+count: U32 -> {Tensors.count_of(view_shape) == Some{count} : Maybe<&2,U32>} -> {ShapeLaws.elements(view_shape) == ShapeLaws.elements(target_shape) : Nat},
  accepted: {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.gather_views(~Element,name,Some{source},Tensors.view_of(view_shape),Tensors.count_of(view_shape),storage,source_shape,target_shape,fill,workers))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},Layout.result_of(~Element,Layout.gather_views(~Element,name,Some{source},Tensors.view_of(view_shape),Tensors.count_of(view_shape),storage,source_shape,target_shape,fill,workers))):
  Buffer.Observation{+input,+equation} = observation
  Shape.Both{Shape.Both{Dense.WellFormed{length_exact,content_ready},source_ready},source_values} = contract
  %equation : Dense.Contract(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},Layout.result_of(~Element,Layout.gather_views(~Element,name,Some{source},Tensors.view_of(view_shape),Tensors.count_of(view_shape),_,source_shape,target_shape,fill,workers)))
  selected(~Element,~Logical,~original,Tensors.view_of(view_shape),Tensors.count_of(view_shape),logical,name,source,input,view_shape,source_shape,target_shape,fill,workers,
    Buffer.ready_content(~Element,input,
      %Buffer.packed_contents(~Element,input) : Buffer.ContentReady<Element,_>
      %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,input),storage,equation) : Buffer.ContentReady<Element,Buffer.contents(~Element,_)>
      content_ready),
    %Buffer.packed_contents(~Element,input) : Dense.Values(~Element,~Logical,~original,logical,_)
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,input),storage,equation) : Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,_))
    source_values,
    {==},{==},same_elements,
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,input),storage,equation) : {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.gather_views(~Element,name,Some{source},Tensors.view_of(view_shape),Tensors.count_of(view_shape),_,source_shape,target_shape,fill,workers))) == True{} : Bool}
    accepted)

# A Ready dense tensor with complete values, gathered through a source view
# into storage laid out as view_shape and labelled target_shape: when the
# call returns Ready, the result is well formed, Ready, and holds the logical
# input at the address of every index of the source view. An operation whose
# label differs from the layout shows that both have the same elements, from
# the count the call itself accepted.
def gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+name: String,+source: Views.View,+view_shape: List<&2,U32>,storage: Storage.Buffer<Element>,
  +source_shape: List<&2,U32>,+target_shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Dense.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{storage,source_shape,fill,Tensors.Ready{},workers}),
  same_elements: @+count: U32 -> {Tensors.count_of(view_shape) == Some{count} : Maybe<&2,U32>} -> {ShapeLaws.elements(view_shape) == ShapeLaws.elements(target_shape) : Nat},
  accepted: {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.gather_views(~Element,name,Some{source},Tensors.view_of(view_shape),Tensors.count_of(view_shape),storage,source_shape,target_shape,fill,workers))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},Layout.result_of(~Element,Layout.gather_views(~Element,name,Some{source},Tensors.view_of(view_shape),Tensors.count_of(view_shape),storage,source_shape,target_shape,fill,workers))):
  observed(~Element,~Logical,~original,logical,name,source,view_shape,storage,source_shape,target_shape,fill,workers,Buffer.observe(~Element,storage),contract,same_elements,accepted)

# The same statement for a source view an operation computes: a Ready result
# has a source view, and its values are read through it.
def Viewed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,source: Maybe<&2,Views.View>,-tensor: Tensors.Tensor<Element>) -> Data:
  match source:
    case Some{source}: Dense.Contract(~Element,~Gather<Logical>,~value(~Element,~Logical,~original),Gather{logical,source},tensor)
    case None{}: Empty

def status(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,source: Maybe<&2,Views.View>,+logical: Logical,+name: String,+view_shape: List<&2,U32>,storage: Storage.Buffer<Element>,
  +source_shape: List<&2,U32>,+target_shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Dense.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{storage,source_shape,fill,Tensors.Ready{},workers}),
  same_elements: @+count: U32 -> {Tensors.count_of(view_shape) == Some{count} : Maybe<&2,U32>} -> {ShapeLaws.elements(view_shape) == ShapeLaws.elements(target_shape) : Nat},
  accepted: {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.gather_status(~Element,name,Tensors.Ready{},source,view_shape,target_shape,storage,source_shape,fill,workers))) == True{} : Bool}) ->
  Viewed(~Element,~Logical,~original,logical,source,Layout.result_of(~Element,Layout.gather_status(~Element,name,Tensors.Ready{},source,view_shape,target_shape,storage,source_shape,fill,workers))):
  match source:
    case Some{+source}: gather(~Element,~Logical,~original,logical,name,source,view_shape,storage,source_shape,target_shape,fill,workers,contract,same_elements,accepted)
    case None{}: WordEncoding.false_true(accepted)

# The public layout operations that are one gather of a view of the input.
# Each statement names the view the operation itself computes.
def transpose(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+order: List<&2,U32>,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Dense.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers}),
  accepted: {Dense.tensor_ready(~Element,Layout.transpose(~Element,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers},order)) == True{} : Bool}) ->
  Viewed(~Element,~Logical,~original,logical,Layout.permuted(Tensors.view_of(shape),Layout.naturals(order)),Layout.transpose(~Element,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers},order)):
  status(~Element,~Logical,~original,Layout.permuted(Tensors.view_of(shape),Layout.naturals(order)),logical,"transpose",Layout.extents(Layout.permuted(Tensors.view_of(shape),Layout.naturals(order))),storage,shape,Layout.extents(Layout.permuted(Tensors.view_of(shape),Layout.naturals(order))),fill,workers,contract,count => counted => {==},accepted)

def slice(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+axis: U32,+start: U32,+count: U32,+step: U32,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Dense.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers}),
  accepted: {Dense.tensor_ready(~Element,Layout.slice(~Element,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers},axis,start,count,step)) == True{} : Bool}) ->
  Viewed(~Element,~Logical,~original,logical,Layout.sliced(Tensors.view_of(shape),axis,start,count,step),Layout.slice(~Element,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers},axis,start,count,step)):
  status(~Element,~Logical,~original,Layout.sliced(Tensors.view_of(shape),axis,start,count,step),logical,"slice",Layout.extents(Layout.sliced(Tensors.view_of(shape),axis,start,count,step)),storage,shape,Layout.extents(Layout.sliced(Tensors.view_of(shape),axis,start,count,step)),fill,workers,contract,count => counted => {==},accepted)

def broadcast_selected(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,source: Maybe<&2,Views.View>,+logical: Logical,+target: List<&2,U32>,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Dense.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers}),
  accepted: {Dense.tensor_ready(~Element,Layout.broadcast_status(~Element,Tensors.Ready{},source,storage,shape,target,fill,workers)) == True{} : Bool}) ->
  Viewed(~Element,~Logical,~original,logical,source,Layout.broadcast_status(~Element,Tensors.Ready{},source,storage,shape,target,fill,workers)):
  match source:
    case Some{+source}: gather(~Element,~Logical,~original,logical,"broadcast_to",source,target,storage,shape,target,fill,workers,contract,count => counted => {==},accepted)
    case None{}: WordEncoding.false_true(accepted)

def broadcast_to(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+target: List<&2,U32>,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Dense.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers}),
  accepted: {Dense.tensor_ready(~Element,Layout.broadcast_to(~Element,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers},target)) == True{} : Bool}) ->
  Viewed(~Element,~Logical,~original,logical,Tensors.stretched(Tensors.view_of(shape),target),Layout.broadcast_to(~Element,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers},target)):
  broadcast_selected(~Element,~Logical,~original,Tensors.stretched(Tensors.view_of(shape),target),logical,target,storage,shape,fill,workers,contract,accepted)

# A Ready dense tensor of a known shape, observed as a model whose content
# satisfies the facts of a reader. The equation replaces the tensor by its
# model wherever it occurs, so a proof never holds the native storage twice.
type Shaped<-Element: Data,-Context: Data,-reader: Context -> U32 -> Element,-context: Context,-shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>> is Data:
  Shaped{model: Buffer.Model<Element>,fill: Element,workers: U32,
    equation: {tensor == Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Ready{},workers} : Tensors.Tensor<Element>},
    facts: Facts(~Element,~Context,~reader,context,shape,Buffer.content(~Element,model))}

def shaped_observed(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,-storage: Storage.Buffer<Element>,+context: Context,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  observation: Buffer.Observation<Element,storage>,
  facts: Facts(~Element,~Context,~reader,context,shape,Buffer.contents(~Element,storage))) ->
  Shaped<Element,Context,reader,context,shape,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers}>:
  Buffer.Observation{+model,+equation} = observation
  Shaped{model,fill,workers,
    Equal.cong(Storage.Buffer<Element>,Tensors.Tensor<Element>,buffer => Tensors.Tensor{buffer,shape,fill,Tensors.Ready{},workers},storage,Buffer.pack(~Element,model),
      Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),storage,equation)),
    %Buffer.packed_contents(~Element,model) : Facts(~Element,~Context,~reader,context,shape,_)
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),storage,equation) : Facts(~Element,~Context,~reader,context,shape,Buffer.contents(~Element,_))
    facts}

# A shaped tensor satisfies the dense contract of its reader.
def shaped_contract(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>,
  evidence: Shaped<Element,Context,reader,context,shape,tensor>) ->
  Dense.Contract(~Element,~Context,~reader,context,tensor):
  Shaped{+model,+fill,+workers,+equation,Shape.Both{well_formed,values}} = evidence
  %Equal.sym(Tensors.Tensor<Element>,tensor,Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Ready{},workers},equation) : Dense.Contract(~Element,~Context,~reader,context,_)
  Shape.Both{Shape.Both{
      %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model)) : Dense.WellFormed<Element,shape,_>
      well_formed,{==}},
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model)) : Dense.Values(~Element,~Context,~reader,context,_)
    values}
