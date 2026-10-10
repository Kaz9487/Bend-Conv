# Values of the checked copy. The run of a copy is a write program; a
# position inside the count holds the sample of its index, and the sample
# of a strided source is the storage at the address of the index in the
# coordinate model. Every operation that copies through a view proves
# only its own address map.
import Base
import ./boolean_logic.bend as Boolean
import ../tensor_access.bend as Access
import ./tensor_cursor_bounds.bend as CursorBounds
import ./tensor_access_refinement.bend as AccessLaws
import ./storage_allocation_refinement.bend as AllocationLaws
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_band_combine_leaf.bend as BandCombineLeaf
import ./nat_to_u32_bounds.bend as Bounds
import ./storage_buffer_model.bend as Buffer
import ./storage_certificate.bend as Certificate
import ./contiguous_copy.bend as Contiguous
import ./tensor_cursor_refinement.bend as CursorLaws
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_cursor_model.bend as CursorModel
import ./tensor_cursor_order.bend as CursorOrder
import ./tensor_allocation_depth.bend as Depth
import ./traversal_array_refinement.bend as ArrayLaws
import ../traversal_loop.bend as Loop
import ./traversal_map_readback.bend as MapReadback
import ./sequence_membership.bend as Members
import ../storage_allocation.bend as Allocation
import ../storage_buffer.bend as Storage
import ../tensor_operations.bend as Operations
import ./nat_order.bend as Order
import ./tensor_operations_refinement.bend as OperationsLaws
import ./ordered_ranges.bend as Ranges
import ./write_program_readback.bend as Readback
import ./tensor_visit_refinement.bend as TensorVisit
import ./traversal_visit_refinement.bend as Visit
import ./tensor_shape_refinement.bend as ShapeLaws
import ../tensor_shape.bend as Shapes
import ./storage_refinement.bend as StorageLaws
import ./storage_shape.bend as Shape
import ../tensor.bend as Tensors
import ../traversal_array.bend as Traversal
import ./traversal_indexed_writes.bend as TraversalIndexedWrites
import ./traversal_visit_writes.bend as TraversalVisitWrites
import ./storage_tree_model.bend as Tree
import ../tensor_view.bend as Views
import ./write_program.bend as Writes

# Nearest upsampling keeps the band representation and calls the same checked
# repeated-axis copy as dense upsampling. View/count premises below are the
# actual runtime computations; their successful producer guards remain to prove.
def right_content(~Element: Data,result: Operations.Outcome<OperationsLaws.PairOwners(Element)>) -> Buffer.Content<Element>:
  match result:
    case Operations.Accepted{(left,OperationsLaws.Owner{storage,length})}: Buffer.Content{Tree.reflect(Element,storage),length}
    case Operations.Rejected{(left,OperationsLaws.Owner{storage,length})}: Buffer.Content{Tree.reflect(Element,storage),length}

# Actual checked-copy output contracts for the nearest-neighbour band leaf.
# Ready includes both accepted and rejected outcomes. Values below require
# the copy guard, store bounds, sample matching, and coverage explicitly;
# those premises are not claimed to follow from the public caller yet.
def output_tree(~Element: Data,result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) -> Tree.Tree<Element>:
  AccessLaws.ModelResult{reader,output} = result
  output

def view_offset(view: Views.View) -> U32:
  Views.View{axes,offset} = view
  offset

def visit_writes(~Element: Data,count: Nat,+position: U32,+coordinates: CursorLaws.Coordinates,+input: Tree.Tree<Element>) -> List<&2,Writes.Assignment<Element>>:
  match count:
    case 0n: []
    case 1n+rest:
      Writes.Put{position,TensorVisit.sample(~Element,input,coordinates)} <>
        visit_writes(~Element,rest,(position + 1 : U32),CursorLaws.next(coordinates),input)

def fast_program(~Element: Data,count: Nat,+position: U32,+source: U32,+destination: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {output_tree(~Element,AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Traversal.Offsets,
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Traversal.Offsets>,
        ~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Traversal.Offsets,~ArrayLaws.sample_offset(~Element),~Traversal.destination_offset),
        count,Traversal.State{input,output,position,Traversal.Offsets{source,destination}}))) ==
    Writes.pure(Element,Contiguous.writes(~Element,count,input,position,source,destination),output) : Tree.Tree<Element>}:
  match count:
    case 0n: {==}
    case 1n+rest:
      fast_program(~Element,rest,(position + 1 : U32),source,destination,input,
        Tree.write(Element,output,(destination + position : U32),Tree.read(Element,input,(source + position : U32))))

def visit_program(~Element: Data,count: Nat,+position: U32,+coordinates: CursorLaws.Coordinates,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {output_tree(~Element,AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~CursorLaws.Coordinates,
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,CursorLaws.Coordinates>,
        ~Visit.model_step(~Tree.Tree<Element>,~Element,~CursorLaws.Coordinates,~TensorVisit.sample(~Element),~CursorLaws.next),
        count,Traversal.State{input,output,position,coordinates}))) ==
    Writes.pure(Element,visit_writes(~Element,count,position,coordinates,input),output) : Tree.Tree<Element>}:
  match count:
    case 0n: {==}
    case 1n+rest:
      visit_program(~Element,rest,(position + 1 : U32),CursorLaws.next(coordinates),input,
        Tree.write(Element,output,position,TensorVisit.sample(~Element,input,coordinates)))

def copy_writes(~Element: Data,fast: Bool,count: Nat,+source: Views.View,destination: U32,input: Tree.Tree<Element>) -> List<&2,Writes.Assignment<Element>>:
  match fast:
    case True{}:
      Views.View{axes,offset} = source
      Contiguous.writes(~Element,count,input,0,offset,destination)
    case False{}: visit_writes(~Element,count,destination,CursorLaws.initial(source),input)

def copy_program(~Element: Data,+fast: Bool,+count: Nat,+source: Views.View,+destination: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {output_tree(~Element,AccessLaws.copy_model(~Element,fast,count,source,destination,input,output)) ==
    Writes.pure(Element,copy_writes(~Element,fast,count,source,destination,input),output) : Tree.Tree<Element>}:
  match fast:
    case True{}:
      Views.View{+axes,+offset} = source
      fast_program(~Element,count,0,offset,destination,input,output)
    case False{}: visit_program(~Element,count,destination,CursorLaws.initial(source),input,output)

def modeled_content(~Element: Data,+input_length: Nat,+output_length: Nat,+result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) ->
  {right_content(~Element,Operations.Accepted{OperationsLaws.modeled_pair(~Element,input_length,output_length,result)}) ==
    Buffer.Content{output_tree(~Element,result),output_length} : Buffer.Content<Element>}:
  AccessLaws.ModelResult{reader,+output} = result
  Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,output_length},
    Tree.reflect(Element,Tree.pack(Element,output)),output,StorageLaws.reflect_packed(Element,output))

def accepted_content(~Element: Data,+count: U32,+source: Views.View,+target: Views.View,+input: Tree.Tree<Element>,+input_length: Nat,+output: Tree.Tree<Element>,+output_length: Nat) ->
  {right_content(~Element,OperationsLaws.copy_model(~Element,True{},Some{count},source,target,input,input_length,output,output_length)) ==
    Buffer.Content{Writes.pure(Element,copy_writes(~Element,Views.is_contiguous(source),U32.to_nat(count),source,view_offset(target),input),output),output_length} : Buffer.Content<Element>}:
  Views.View{+axes,+destination} = target
  Equal.trans(Buffer.Content<Element>,
    right_content(~Element,OperationsLaws.copy_model(~Element,True{},Some{count},source,target,input,input_length,output,output_length)),
    Buffer.Content{output_tree(~Element,AccessLaws.copy_model(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input,output)),output_length},
    Buffer.Content{Writes.pure(Element,copy_writes(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input),output),output_length},
    modeled_content(~Element,input_length,output_length,AccessLaws.copy_model(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input,output)),
    Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,output_length},
      output_tree(~Element,AccessLaws.copy_model(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input,output)),
      Writes.pure(Element,copy_writes(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input),output),
      copy_program(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input,output)))

def copy_model_ready(~Element: Data,+valid: Bool,+count: Maybe<&2,U32>,+source: Views.View,+target: Views.View,+input: Tree.Tree<Element>,+input_length: Nat,+output: Tree.Tree<Element>,+output_length: Nat,
  ready: Buffer.ContentReady<Element,Buffer.Content{output,output_length}>) ->
  Buffer.ContentReady<Element,right_content(~Element,OperationsLaws.copy_model(~Element,valid,count,source,target,input,input_length,output,output_length))>:
  match valid count:
    case True{} Some{+count}:
      Views.View{+axes,+destination} = target
      %Equal.sym(Buffer.Content<Element>,right_content(~Element,OperationsLaws.copy_model(~Element,True{},Some{count},source,target,input,input_length,output,output_length)),
        Buffer.Content{Writes.pure(Element,copy_writes(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input),output),output_length},
        accepted_content(~Element,count,source,target,input,input_length,output,output_length)) : Buffer.ContentReady<Element,_>
      Writes.content_ready(~Element,copy_writes(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input),output,output_length,ready)
    case False{} None{}:
      %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,output)),output,StorageLaws.reflect_packed(Element,output)) : Buffer.ContentReady<Element,Buffer.Content{_,output_length}> ready
    case False{} Some{count}:
      %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,output)),output,StorageLaws.reflect_packed(Element,output)) : Buffer.ContentReady<Element,Buffer.Content{_,output_length}> ready
    case True{} None{}:
      %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,output)),output,StorageLaws.reflect_packed(Element,output)) : Buffer.ContentReady<Element,Buffer.Content{_,output_length}> ready

def copy_observed_ready(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,output: Storage.Buffer<Element>,+target: Views.View,
  ready: Buffer.ContentReady<Element,Buffer.contents(~Element,output)>) ->
  Buffer.ContentReady<Element,right_content(~Element,OperationsLaws.copy_denotation(~Element,input,source,output,target))>:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Storage.Buffer{output,+output_length,output_certificate} = output
  copy_model_ready(~Element,Shapes.same(Views.shape(source),Views.shape(target)) && Views.is_contiguous(target) &&
    Operations.valid_buffer(input_length,Storage.capacity(~Element,input,input_certificate),source) &&
    Operations.valid_buffer(output_length,Storage.capacity(~Element,output,output_certificate),target),Views.numel(target),source,target,
    Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length,ready)

def copy_guard(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,output: Storage.Buffer<Element>,+target: Views.View) -> Bool:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Storage.Buffer{output,+output_length,output_certificate} = output
  Shapes.same(Views.shape(source),Views.shape(target)) && Views.is_contiguous(target) &&
    Operations.valid_buffer(input_length,Storage.capacity(~Element,input,input_certificate),source) &&
    Operations.valid_buffer(output_length,Storage.capacity(~Element,output,output_certificate),target)

def program_content(~Element: Data,+count: U32,+source: Views.View,+target: Views.View,input: Buffer.Content<Element>,output: Buffer.Content<Element>) -> Buffer.Content<Element>:
  Buffer.Content{input,input_length} = input
  Buffer.Content{output,output_length} = output
  Buffer.Content{Writes.pure(Element,copy_writes(~Element,Views.is_contiguous(source),U32.to_nat(count),source,view_offset(target),input),output),output_length}

def copy_observed_content(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,output: Storage.Buffer<Element>,+target: Views.View,+count: U32,
  guarded: {copy_guard(~Element,input,source,output,target) == True{} : Bool},counted: {Views.numel(target) == Some{count} : Maybe<&2,U32>}) ->
  {right_content(~Element,OperationsLaws.copy_denotation(~Element,input,source,output,target)) ==
    program_content(~Element,count,source,target,Buffer.contents(~Element,input),Buffer.contents(~Element,output)) : Buffer.Content<Element>}:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Storage.Buffer{output,+output_length,output_certificate} = output
  %Equal.sym(Bool,Shapes.same(Views.shape(source),Views.shape(target)) && Views.is_contiguous(target) &&
      Operations.valid_buffer(input_length,Storage.capacity(~Element,input,input_certificate),source) &&
      Operations.valid_buffer(output_length,Storage.capacity(~Element,output,output_certificate),target),True{},guarded) :
    {right_content(~Element,OperationsLaws.copy_model(~Element,_,Views.numel(target),source,target,Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length)) ==
      program_content(~Element,count,source,target,Buffer.Content{Tree.reflect(Element,input),input_length},Buffer.Content{Tree.reflect(Element,output),output_length}) : Buffer.Content<Element>}
  %Equal.sym(Maybe<&2,U32>,Views.numel(target),Some{count},counted) :
    {right_content(~Element,OperationsLaws.copy_model(~Element,True{},_,source,target,Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length)) ==
      program_content(~Element,count,source,target,Buffer.Content{Tree.reflect(Element,input),input_length},Buffer.Content{Tree.reflect(Element,output),output_length}) : Buffer.Content<Element>}
  accepted_content(~Element,count,source,target,Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length)

type Point<-Element: Data,-wanted: Element,-query: U32,-writes: List<&2,Writes.Assignment<Element>>> is Data:
  Point{matching: Readback.Matching(~Element,query,wanted,writes),covered: {Readback.contains(~Element,query,writes) == True{} : Bool}}

def program_point(~Element: Data,+writes: List<&2,Writes.Assignment<Element>>,+output: Tree.Tree<Element>,+length: Nat,+query: U32,+wanted: Element,
  ready: Buffer.ContentReady<Element,Buffer.Content{output,length}>,query_inside: {U32.is_lt(query,Tree.capacity(Element,output)) == True{} : Bool},
  stores_inside: Readback.Inside(~Element,Tree.capacity(Element,output),writes),evidence: Point<Element,wanted,query,writes>) ->
  {Tree.read(Element,Writes.pure(Element,writes,output),query) == wanted : Element}:
  Buffer.ContentReady{depth,balanced,depth_ok,room} = ready
  Point{matching,covered} = evidence
  Readback.covered(~Element,writes,depth,output,query,wanted,balanced,depth_ok,query_inside,stores_inside,matching,covered)

def modeled_length(~Element: Data,+input_length: Nat,+output_length: Nat,result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) ->
  {Buffer.content_length(~Element,right_content(~Element,Operations.Accepted{OperationsLaws.modeled_pair(~Element,input_length,output_length,result)})) == output_length : Nat}:
  match result:
    case AccessLaws.ModelResult{reader,output}: {==}

def copy_model_length(~Element: Data,+valid: Bool,+count: Maybe<&2,U32>,+source: Views.View,+target: Views.View,+input: Tree.Tree<Element>,+input_length: Nat,+output: Tree.Tree<Element>,+output_length: Nat) ->
  {Buffer.content_length(~Element,right_content(~Element,OperationsLaws.copy_model(~Element,valid,count,source,target,input,input_length,output,output_length))) == output_length : Nat}:
  match valid count:
    case True{} Some{count}:
      Views.View{axes,destination} = target
      modeled_length(~Element,input_length,output_length,AccessLaws.copy_model(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input,output))
    case False{} None{}: {==}
    case False{} Some{count}: {==}
    case True{} None{}: {==}

def copy_length(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,output: Storage.Buffer<Element>,+target: Views.View) ->
  {Buffer.content_length(~Element,right_content(~Element,OperationsLaws.copy_denotation(~Element,input,source,output,target))) == Buffer.content_length(~Element,Buffer.contents(~Element,output)) : Nat}:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Storage.Buffer{output,+output_length,output_certificate} = output
  copy_model_length(~Element,Shapes.same(Views.shape(source),Views.shape(target)) && Views.is_contiguous(target) &&
    Operations.valid_buffer(input_length,Storage.capacity(~Element,input,input_certificate),source) && Operations.valid_buffer(output_length,Storage.capacity(~Element,output,output_certificate),target),
    Views.numel(target),source,target,Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length)

def valid_owned(~Element: Data,buffer: Storage.Buffer<Element>,view: Views.View) -> Bool:
  Storage.Buffer{storage,length,certificate} = buffer
  Operations.valid_buffer(length,Storage.capacity(~Element,storage,certificate),view)

def valid_model(~Element: Data,+input: Buffer.Model<Element>,+view: Views.View,+count: U32,exact: {U32.to_nat(count) == Buffer.length(~Element,input) : Nat},
  ready: Buffer.Sized<Element,input>,fits: {Views.fits(view,count) == True{} : Bool}) -> {valid_owned(~Element,Buffer.pack(~Element,input),view) == True{} : Bool}:
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = input
  Buffer.Sized{depth_ok,+room} = ready
  +model = {Buffer.Model{tree,length,Certificate.Certificate{depth,equation}} : Buffer.Model<Element>}
  %Shape.capacity(Element,depth,tree,Buffer.balanced(~Element,model)) : {Operations.valid_buffer(length,_,view) == True{} : Bool}
  %Equal.sym(U32,U32.from_nat(length),count,BandCombineLeaf.native_length(count,length,exact)) :
    {Nat.is_le(length,U32.to_nat(Tree.capacity(Element,tree))) && Views.fits(view,_) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_le(length,U32.to_nat(Tree.capacity(Element,tree))),True{},room) : {_ && Views.fits(view,count) == True{} : Bool}
  fits

def valid_allocated(~Element: Data,+count: U32,+fill: Element,+view: Views.View,ready: Depth.Sized<count>,fits: {Views.fits(view,count) == True{} : Bool}) ->
  {valid_owned(~Element,Tensors.buffer(~Element,fill,count),view) == True{} : Bool}:
  Depth.Sized{depth_ok,+room} = ready
  %Equal.sym(Storage.Buffer<Element>,Allocation.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)),Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)),
    AllocationLaws.dispatch_same(Element,Allocation.allocated(~Element),Tensors.depth_for(count),fill,U32.to_nat(count))) : {valid_owned(~Element,_,view) == True{} : Bool}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(count)),count,Arithmetic.native_roundtrip(count)) :
    {Nat.is_le(U32.to_nat(count),U32.to_nat(U32.shln(1,Tensors.depth_for(count)))) && Views.fits(view,_) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_le(U32.to_nat(count),U32.to_nat(U32.shln(1,Tensors.depth_for(count)))),True{},room) : {_ && Views.fits(view,count) == True{} : Bool}
  fits

def joined_guard(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,output: Storage.Buffer<Element>,+target: Views.View,
  same: {Shapes.same(Views.shape(source),Views.shape(target)) == True{} : Bool},contiguous: {Views.is_contiguous(target) == True{} : Bool},
  input_valid: {valid_owned(~Element,input,source) == True{} : Bool},output_valid: {valid_owned(~Element,output,target) == True{} : Bool}) ->
  {copy_guard(~Element,input,source,output,target) == True{} : Bool}:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Storage.Buffer{output,+output_length,output_certificate} = output
  %Equal.sym(Bool,Shapes.same(Views.shape(source),Views.shape(target)),True{},same) :
    {_ && Views.is_contiguous(target) && Operations.valid_buffer(input_length,Storage.capacity(~Element,input,input_certificate),source) && Operations.valid_buffer(output_length,Storage.capacity(~Element,output,output_certificate),target) == True{} : Bool}
  %Equal.sym(Bool,Views.is_contiguous(target),True{},contiguous) :
    {True{} && _ && Operations.valid_buffer(input_length,Storage.capacity(~Element,input,input_certificate),source) && Operations.valid_buffer(output_length,Storage.capacity(~Element,output,output_certificate),target) == True{} : Bool}
  %Equal.sym(Bool,Operations.valid_buffer(input_length,Storage.capacity(~Element,input,input_certificate),source),True{},input_valid) :
    {True{} && True{} && _ && Operations.valid_buffer(output_length,Storage.capacity(~Element,output,output_certificate),target) == True{} : Bool}
  output_valid

# Finite actual samples and canonical destination stores of checked upsampling.
# The cursor branch follows CursorLaws.next; the contiguous branch follows
# its actual offset. Neither branch assumes a nearest-neighbour value equation.
type CopyContext<-Element: Data> is Data:
  CopyContext{fast: Bool,source: Views.View,input: Tree.Tree<Element>}

def sample_value(~Element: Data,context: CopyContext<Element>,position: U32) -> Element:
  CopyContext{fast,source,input} = context
  match fast:
    case True{}: Tree.read(Element,input,(view_offset(source) + position : U32))
    case False{}:
      TensorVisit.sample(~Element,input,Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,U32.to_nat(position),CursorLaws.initial(source)))

def fast_indexed(~Element: Data,count: Nat,+position: U32,+source: Views.View,+input: Tree.Tree<Element>) ->
  {Contiguous.writes(~Element,count,input,position,view_offset(source),0) ==
    TraversalIndexedWrites.writes(~Element,~CopyContext<Element>,~sample_value(~Element),count,position,CopyContext{True{},source,input}) : List<&2,Writes.Assignment<Element>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(U32,(0 + position : U32),position,Arithmetic.add_zero_left(position)) :
        {Writes.Put{_,Tree.read(Element,input,(view_offset(source) + position : U32))} <>
          Contiguous.writes(~Element,rest,input,(position + 1 : U32),view_offset(source),0) ==
          TraversalIndexedWrites.writes(~Element,~CopyContext<Element>,~sample_value(~Element),1n+rest,position,CopyContext{True{},source,input}) : List<&2,Writes.Assignment<Element>>}
      Equal.cong(List<&2,Writes.Assignment<Element>>,List<&2,Writes.Assignment<Element>>,
        tail => Writes.Put{position,Tree.read(Element,input,(view_offset(source) + position : U32))} <> tail,
        Contiguous.writes(~Element,rest,input,(position + 1 : U32),view_offset(source),0),
        TraversalIndexedWrites.writes(~Element,~CopyContext<Element>,~sample_value(~Element),rest,(position + 1 : U32),CopyContext{True{},source,input}),
        fast_indexed(~Element,rest,(position + 1 : U32),source,input))

def visit_indexed(~Element: Data,count: Nat,+position: Nat,+source: Views.View,+input: Tree.Tree<Element>,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(position,count),U32.to_nat(capacity)) == True{} : Bool}) ->
  {visit_writes(~Element,count,U32.from_nat(position),Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,position,CursorLaws.initial(source)),input) ==
    TraversalIndexedWrites.writes(~Element,~CopyContext<Element>,~sample_value(~Element),count,U32.from_nat(position),CopyContext{False{},source,input}) : List<&2,Writes.Assignment<Element>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +initial = CursorLaws.initial(source)
      +cursor = Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,position,initial)
      +encoded = Equal.cong(Nat,Element,index => TensorVisit.sample(~Element,input,Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,initial)),
        U32.to_nat(U32.from_nat(position)),position,Bounds.roundtrip(position,capacity,capacity_ok,
          Order.transitive(position,Nat.add(position,1n+rest),U32.to_nat(capacity),Order.le_add(position,1n+rest),room)))
      %Equal.sym(Element,sample_value(~Element,CopyContext{False{},source,input},U32.from_nat(position)),TensorVisit.sample(~Element,input,cursor),encoded) :
        {Writes.Put{U32.from_nat(position),TensorVisit.sample(~Element,input,cursor)} <>
          visit_writes(~Element,rest,(U32.from_nat(position) + 1 : U32),CursorLaws.next(cursor),input) ==
          Writes.Put{U32.from_nat(position),_} <>
            TraversalIndexedWrites.writes(~Element,~CopyContext<Element>,~sample_value(~Element),rest,(U32.from_nat(position) + 1 : U32),CopyContext{False{},source,input}) : List<&2,Writes.Assignment<Element>>}
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {Writes.Put{U32.from_nat(position),TensorVisit.sample(~Element,input,cursor)} <>
          visit_writes(~Element,rest,_,CursorLaws.next(cursor),input) ==
          Writes.Put{U32.from_nat(position),TensorVisit.sample(~Element,input,cursor)} <>
            TraversalIndexedWrites.writes(~Element,~CopyContext<Element>,~sample_value(~Element),rest,_,CopyContext{False{},source,input}) : List<&2,Writes.Assignment<Element>>}
      %TraversalVisitWrites.next_after(~CursorLaws.Coordinates,~CursorLaws.next,position,initial) :
        {Writes.Put{U32.from_nat(position),TensorVisit.sample(~Element,input,cursor)} <>
          visit_writes(~Element,rest,U32.from_nat(1n+position),_,input) ==
          Writes.Put{U32.from_nat(position),TensorVisit.sample(~Element,input,cursor)} <>
            TraversalIndexedWrites.writes(~Element,~CopyContext<Element>,~sample_value(~Element),rest,U32.from_nat(1n+position),CopyContext{False{},source,input}) : List<&2,Writes.Assignment<Element>>}
      Equal.cong(List<&2,Writes.Assignment<Element>>,List<&2,Writes.Assignment<Element>>,
        tail => Writes.Put{U32.from_nat(position),TensorVisit.sample(~Element,input,cursor)} <> tail,
        visit_writes(~Element,rest,U32.from_nat(1n+position),Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,1n+position,initial),input),
        TraversalIndexedWrites.writes(~Element,~CopyContext<Element>,~sample_value(~Element),rest,U32.from_nat(1n+position),CopyContext{False{},source,input}),
        visit_indexed(~Element,rest,1n+position,source,input,capacity,capacity_ok,Order.advance_room(position,rest,U32.to_nat(capacity),room)))

def copy_indexed(~Element: Data,fast: Bool,+count: Nat,+source: Views.View,+input: Tree.Tree<Element>,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(count,U32.to_nat(capacity)) == True{} : Bool}) ->
  {copy_writes(~Element,fast,count,source,0,input) ==
    TraversalIndexedWrites.writes(~Element,~CopyContext<Element>,~sample_value(~Element),count,0,CopyContext{fast,source,input}) : List<&2,Writes.Assignment<Element>>}:
  match fast:
    case True{}:
      Views.View{axes,offset} = source
      fast_indexed(~Element,count,0,Views.View{axes,offset},input)
    case False{}: visit_indexed(~Element,count,0n,source,input,capacity,capacity_ok,room)

def copy_inside(~Element: Data,+fast: Bool,+count: Nat,+source: Views.View,+input: Tree.Tree<Element>,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(count,U32.to_nat(capacity)) == True{} : Bool}) ->
  Readback.Inside(~Element,capacity,copy_writes(~Element,fast,count,source,0,input)):
  %Equal.sym(List<&2,Writes.Assignment<Element>>,copy_writes(~Element,fast,count,source,0,input),
    TraversalIndexedWrites.writes(~Element,~CopyContext<Element>,~sample_value(~Element),count,0,CopyContext{fast,source,input}),copy_indexed(~Element,fast,count,source,input,capacity,capacity_ok,room)) :
    Readback.Inside(~Element,capacity,_)
  TraversalIndexedWrites.inside(~Element,~CopyContext<Element>,~sample_value(~Element),count,0n,CopyContext{fast,source,input},capacity,capacity_ok,room)

def copy_point(~Element: Data,+fast: Bool,+count: Nat,+source: Views.View,+input: Tree.Tree<Element>,+capacity: U32,+index: Nat,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(count,U32.to_nat(capacity)) == True{} : Bool},inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  Point<Element,sample_value(~Element,CopyContext{fast,source,input},U32.from_nat(index)),U32.from_nat(index),copy_writes(~Element,fast,count,source,0,input)>:
  %Equal.sym(List<&2,Writes.Assignment<Element>>,copy_writes(~Element,fast,count,source,0,input),
    TraversalIndexedWrites.writes(~Element,~CopyContext<Element>,~sample_value(~Element),count,0,CopyContext{fast,source,input}),copy_indexed(~Element,fast,count,source,input,capacity,capacity_ok,room)) :
    Point<Element,sample_value(~Element,CopyContext{fast,source,input},U32.from_nat(index)),U32.from_nat(index),_>
  Point{
    TraversalIndexedWrites.matching(~Element,~CopyContext<Element>,~sample_value(~Element),count,0,CopyContext{fast,source,input},U32.from_nat(index)),
    TraversalIndexedWrites.contains(~Element,~CopyContext<Element>,~sample_value(~Element),count,0n,index,CopyContext{fast,source,input},inside)}

def copy_read(~Element: Data,+fast: Bool,+count: Nat,+source: Views.View,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+length: Nat,+index: Nat,
  +ready: Buffer.ContentReady<Element,Buffer.Content{output,length}>,+capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},
  +room: {Nat.is_le(count,U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},+inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Tree.read(Element,Writes.pure(Element,copy_writes(~Element,fast,count,source,0,input),output),U32.from_nat(index)) ==
    sample_value(~Element,CopyContext{fast,source,input},U32.from_nat(index)) : Element}:
  program_point(~Element,copy_writes(~Element,fast,count,source,0,input),output,length,U32.from_nat(index),sample_value(~Element,CopyContext{fast,source,input},U32.from_nat(index)),ready,
    MapReadback.address_inside(index,Tree.capacity(Element,output),capacity_ok,Bounds.nat_trans(index,count,U32.to_nat(Tree.capacity(Element,output)),inside,room)),
    copy_inside(~Element,fast,count,source,input,Tree.capacity(Element,output),capacity_ok,room),copy_point(~Element,fast,count,source,input,Tree.capacity(Element,output),index,capacity_ok,room,inside))

def StorePoint(~Element: Data,+fast: Bool,+count: Nat,+source: Views.View,+input: Tree.Tree<Element>,+capacity: U32,position: Nat) -> Data:
  Shape.Both<{U32.is_lt(U32.from_nat(position),capacity) == True{} : Bool},
    Point<Element,sample_value(~Element,CopyContext{fast,source,input},U32.from_nat(position)),U32.from_nat(position),copy_writes(~Element,fast,count,source,0,input)>>

def store_points(~Element: Data,+fast: Bool,+whole: Nat,+source: Views.View,+input: Tree.Tree<Element>,+capacity: U32,count: Nat,+position: Nat,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(whole,U32.to_nat(capacity)) == True{} : Bool},+remaining: {Nat.is_le(Nat.add(position,count),whole) == True{} : Bool}) ->
  Members.All(Nat,{index => StorePoint(~Element,fast,whole,source,input,capacity,index) : Nat -> Data},Ranges.range(count,position)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +inside = Order.room_index(position,rest,whole,remaining)
      Shape.Both{Shape.Both{
        MapReadback.address_inside(position,capacity,capacity_ok,Bounds.nat_trans(position,whole,U32.to_nat(capacity),inside,room)),
        copy_point(~Element,fast,whole,source,input,capacity,position,capacity_ok,room,inside)},
        store_points(~Element,fast,whole,source,input,capacity,rest,1n+position,capacity_ok,room,Order.advance_room(position,rest,whole,remaining))}

type Finite<-Element: Data,-fast: Bool,-count: Nat,-source: Views.View,-input: Tree.Tree<Element>,-capacity: U32> is Data:
  Finite{inside: Readback.Inside(~Element,capacity,copy_writes(~Element,fast,count,source,0,input)),
    points: Members.All(Nat,{position => StorePoint(~Element,fast,count,source,input,capacity,position) : Nat -> Data},Ranges.range(count,0n))}

def finite(~Element: Data,+fast: Bool,+count: Nat,+source: Views.View,+input: Tree.Tree<Element>,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(count,U32.to_nat(capacity)) == True{} : Bool}) -> Finite<Element,fast,count,source,input,capacity>:
  Finite{copy_inside(~Element,fast,count,source,input,capacity,capacity_ok,room),
    store_points(~Element,fast,count,source,input,capacity,count,0n,capacity_ok,room,Order.reflexive(count))}

def finite_point(~Element: Data,+fast: Bool,+count: Nat,+source: Views.View,+input: Tree.Tree<Element>,+capacity: U32,+index: Nat,
  evidence: Finite<Element,fast,count,source,input,capacity>,inside: {Nat.is_lt(index,count) == True{} : Bool}) -> StorePoint(~Element,fast,count,source,input,capacity,index):
  Finite{stores,points} = evidence
  Members.all_member(Nat,{position => StorePoint(~Element,fast,count,source,input,capacity,position) : Nat -> Data},index,Ranges.range(count,0n),points,Members.range_member(count,0n,index,inside))

def owned_finite(~Element: Data,+fast: Bool,+count: Nat,+source: Views.View,+input: Tree.Tree<Element>,output: Buffer.Model<Element>,
  +exact: {Buffer.length(~Element,output) == count : Nat},ready: Buffer.Sized<Element,output>) -> Finite<Element,fast,count,source,input,Tree.capacity(Element,Buffer.tree(~Element,output))>:
  Buffer.Model{+tree,+length,+certificate} = output
  Buffer.Sized{depth_ok,room} = ready
  +output = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
  finite(~Element,fast,count,source,input,Tree.capacity(Element,tree),Buffer.capacity_bound(~Element,output,depth_ok),
    %exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool} room)

def cursor_address(coordinates: CursorLaws.Coordinates) -> U32:
  CursorLaws.Coordinates{axes,offset,valid} = coordinates
  U32.from_nat(CursorModel.address(axes,offset))

def cursor_sample_projection(~Element: Data,+input: Tree.Tree<Element>,+coordinates: CursorLaws.Coordinates) ->
  {Tree.read(Element,input,cursor_address(coordinates)) == TensorVisit.sample(~Element,input,coordinates) : Element}:
  CursorLaws.Coordinates{axes,offset,valid} = coordinates
  {==}

def cursor_axes(coordinates: CursorLaws.Coordinates) -> List<&2,CursorModel.Coordinate>:
  CursorLaws.Coordinates{axes,offset,valid} = coordinates
  axes

def cursor_offset(coordinates: CursorLaws.Coordinates) -> Nat:
  CursorLaws.Coordinates{axes,offset,valid} = coordinates
  offset

def iterated_address(count: Nat,+axes: List<&2,CursorModel.Coordinate>,+offset: Nat,+valid: CursorLaws.Valid(axes)) ->
  {cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,count,CursorLaws.Coordinates{axes,offset,valid})) == U32.from_nat(CursorModel.address(CursorOrder.iterate(count,axes),offset)) : U32}:
  match count:
    case 0n: {==}
    case 1n+remaining: iterated_address(remaining,CursorModel.advance(axes),offset,CursorLaws.advance_valid(axes,valid))

def cursor_sample(~Element: Data,fast: Bool,source: Views.View,+input: Tree.Tree<Element>,+count: Nat,+index: Nat,+capacity: U32,
  +classified: {Views.is_contiguous(source) == fast : Bool},+exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == count : Nat},
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(count,U32.to_nat(capacity)) == True{} : Bool},+inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {sample_value(~Element,CopyContext{fast,source,input},U32.from_nat(index)) == TensorVisit.sample(~Element,input,Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(source))) : Element}:
  match fast:
    case False{}:
      Equal.cong(Nat,Element,position => TensorVisit.sample(~Element,input,Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,position,CursorLaws.initial(source))),
        U32.to_nat(U32.from_nat(index)),index,Bounds.roundtrip(index,capacity,capacity_ok,Bounds.strict_le(index,U32.to_nat(capacity),Bounds.nat_trans(index,count,U32.to_nat(capacity),inside,room))))
    case True{}:
      Views.View{+axes,+offset} = source
      +view = {Views.View{axes,offset} : Views.View}
      +index_room = Bounds.strict_le(index,U32.to_nat(capacity),Bounds.nat_trans(index,count,U32.to_nat(capacity),inside,room))
      +nonempty = CountCompletion.positive_not_zero(Views.axis_extents(axes),
        %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(axes)),count,exact) : {Nat.is_lt(0n,_) == True{} : Bool}
        Order.le_lt_trans(0n,index,count,Order.zero_le(index),inside))
      +natural_address = CursorOrder.contiguous_addresses(axes,offset,index,nonempty,classified,
        %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(axes)),count,exact) : {Nat.is_lt(index,_) == True{} : Bool} inside)
      +native_address = Equal.trans(U32,cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(view))),
        U32.from_nat(CursorModel.address(CursorOrder.iterate(index,CursorModel.prepare_axes(axes,[])),U32.to_nat(offset))),U32.from_nat(Nat.add(U32.to_nat(offset),index)),
        iterated_address(index,CursorModel.prepare_axes(axes,[]),U32.to_nat(offset),CursorLaws.prepared_valid(axes,[],Unit{})),Equal.cong(Nat,U32,U32.from_nat,
          CursorModel.address(CursorOrder.iterate(index,CursorModel.prepare_axes(axes,[])),U32.to_nat(offset)),Nat.add(U32.to_nat(offset),index),natural_address))
      +native_offset = Equal.trans(U32,(offset + U32.from_nat(index) : U32),U32.from_nat(Nat.add(U32.to_nat(offset),U32.to_nat(U32.from_nat(index)))),U32.from_nat(Nat.add(U32.to_nat(offset),index)),
        Arithmetic.native_add(offset,U32.from_nat(index)),Equal.cong(Nat,U32,position => U32.from_nat(Nat.add(U32.to_nat(offset),position)),U32.to_nat(U32.from_nat(index)),index,Bounds.roundtrip(index,capacity,capacity_ok,index_room)))
      Equal.trans(Element,Tree.read(Element,input,(offset + U32.from_nat(index) : U32)),
        Tree.read(Element,input,cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(view)))),
        TensorVisit.sample(~Element,input,Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(view))),
        Equal.cong(U32,Element,address => Tree.read(Element,input,address),(offset + U32.from_nat(index) : U32),cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(view))),
          Equal.trans(U32,(offset + U32.from_nat(index) : U32),U32.from_nat(Nat.add(U32.to_nat(offset),index)),cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(view))),native_offset,
            Equal.sym(U32,cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(view))),U32.from_nat(Nat.add(U32.to_nat(offset),index)),native_address))),
        cursor_sample_projection(~Element,input,Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(view))))

# The address the coordinate model gives to the index of a view.
def address(source: Views.View,index: Nat) -> U32:
  Views.View{axes,offset} = source
  U32.from_nat(CursorModel.address(CursorOrder.iterate(index,CursorModel.prepare_axes(axes,[])),U32.to_nat(offset)))

# The coordinates behind that address are the row-major digits of the index
# over the extents of the view (axes of extent one carry no digit); the
# address is their dot product with the strides, plus the offset.
def coordinates(source: Views.View,index: Nat) -> List<&2,CursorModel.Coordinate>:
  Views.View{axes,offset} = source
  CursorOrder.iterate(index,CursorModel.prepare_axes(axes,[]))

def coordinates_position(+source: Views.View,+index: Nat,+inside: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents_of(source))) == True{} : Bool}) ->
  {CursorModel.position(coordinates(source,index)) == index : Nat}:
  Views.View{+axes,offset} = source
  +start = CursorModel.prepare_axes(axes,[])
  +nonempty = CountCompletion.positive_not_zero(Views.axis_extents(axes),Order.le_lt_trans(0n,index,ShapeLaws.elements(Views.axis_extents(axes)),Order.zero_le(index),inside))
  %CursorOrder.prepared_zero(axes,[],{==}) : {CursorModel.position(CursorOrder.iterate(index,start)) == Nat.add(_,index) : Nat}
  CursorOrder.iterate_position(index,start,CursorLaws.prepared_valid(axes,[],Unit{}),{
    %Equal.sym(Nat,CursorModel.position(start),0n,CursorOrder.prepared_zero(axes,[],{==})) : {Nat.is_lt(Nat.add(_,index),CursorModel.extent_product(start)) == True{} : Bool}
    %Equal.sym(Nat,CursorModel.extent_product(start),ShapeLaws.elements(Views.axis_extents(axes)),CursorOrder.initial_size(axes,nonempty)) : {Nat.is_lt(index,_) == True{} : Bool}
    inside : {Nat.is_lt(Nat.add(CursorModel.position(start),index),CursorModel.extent_product(start)) == True{} : Bool}})

# The cursor of the strided traversal reads, at step index, the address of
# that index.
def sample_address(~Element: Data,+source: Views.View,+input: Tree.Tree<Element>,+index: Nat) ->
  {TensorVisit.sample(~Element,input,Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(source))) == Tree.read(Element,input,address(source,index)) : Element}:
  Views.View{+axes,+offset} = source
  +view = {Views.View{axes,offset} : Views.View}
  +coordinates = Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(view))
  Equal.trans(Element,TensorVisit.sample(~Element,input,coordinates),Tree.read(Element,input,cursor_address(coordinates)),Tree.read(Element,input,address(view,index)),
    Equal.sym(Element,Tree.read(Element,input,cursor_address(coordinates)),TensorVisit.sample(~Element,input,coordinates),cursor_sample_projection(~Element,input,coordinates)),
    Equal.cong(U32,Element,word => Tree.read(Element,input,word),cursor_address(coordinates),address(view,index),
      iterated_address(index,CursorModel.prepare_axes(axes,[]),U32.to_nat(offset),CursorLaws.prepared_valid(axes,[],Unit{}))))

# A contiguous view at offset zero addresses its indices by themselves.
def zero_contiguous_address(+source: Views.View,+index: Nat,+zero: {Access.zero_contiguous(source) == True{} : Bool},
  +inside: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents_of(source))) == True{} : Bool}) -> {address(source,index) == U32.from_nat(index) : U32}:
  Views.View{+axes,+offset} = source
  +nonempty = CountCompletion.positive_not_zero(Views.axis_extents(axes),Order.le_lt_trans(0n,index,ShapeLaws.elements(Views.axis_extents(axes)),Order.zero_le(index),inside))
  %Equal.sym(U32,offset,0,ShapeLaws.native_equal(offset,0,Boolean.left(U32.is_zero(offset),Views.is_contiguous(Views.View{axes,offset}),zero))) :
    {U32.from_nat(CursorModel.address(CursorOrder.iterate(index,CursorModel.prepare_axes(axes,[])),U32.to_nat(_))) == U32.from_nat(index) : U32}
  Equal.cong(Nat,U32,U32.from_nat,CursorModel.address(CursorOrder.iterate(index,CursorModel.prepare_axes(axes,[])),U32.to_nat(0)),Nat.add(U32.to_nat(0),index),
    CursorOrder.contiguous_addresses(axes,0,index,nonempty,Boolean.right(U32.is_zero(offset),Views.is_contiguous(Views.View{axes,offset}),zero),inside))

# A copy into a target at offset zero stores, at every index inside its
# count, the source storage at the address of that index. A contiguous and a
# strided source give the same statement.
def copied_value(~Element: Data,+source: Views.View,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+length: Nat,+count: Nat,+index: Nat,
  +ready: Buffer.ContentReady<Element,Buffer.Content{output,length}>,+capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},
  +room: {Nat.is_le(count,U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},+inside: {Nat.is_lt(index,count) == True{} : Bool},
  +exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == count : Nat}) ->
  {Tree.read(Element,Writes.pure(Element,copy_writes(~Element,Views.is_contiguous(source),count,source,0,input),output),U32.from_nat(index)) ==
    Tree.read(Element,input,address(source,index)) : Element}:
  +fast = Views.is_contiguous(source)
  +coordinates = Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(source))
  Equal.trans(Element,Tree.read(Element,Writes.pure(Element,copy_writes(~Element,fast,count,source,0,input),output),U32.from_nat(index)),TensorVisit.sample(~Element,input,coordinates),Tree.read(Element,input,address(source,index)),
    Equal.trans(Element,Tree.read(Element,Writes.pure(Element,copy_writes(~Element,fast,count,source,0,input),output),U32.from_nat(index)),sample_value(~Element,CopyContext{fast,source,input},U32.from_nat(index)),TensorVisit.sample(~Element,input,coordinates),
      copy_read(~Element,fast,count,source,input,output,length,index,ready,capacity_ok,room,inside),
      cursor_sample(~Element,fast,source,input,count,index,Tree.capacity(Element,output),{==},exact,capacity_ok,room,inside)),
    sample_address(~Element,source,input,index))

# The span of the coordinates does not change along the iteration.
def iterate_span(count: Nat,+axes: List<&2,CursorModel.Coordinate>) -> {CursorModel.span(CursorOrder.iterate(count,axes)) == CursorModel.span(axes) : Nat}:
  match count:
    case 0n: {==}
    case 1n+rest:
      Equal.trans(Nat,CursorModel.span(CursorOrder.iterate(rest,CursorModel.advance(axes))),CursorModel.span(CursorModel.advance(axes)),CursorModel.span(axes),
        iterate_span(rest,CursorModel.advance(axes)),CursorBounds.advance_span(axes))

# The address of every index of a view that fits a length is inside it.
def address_inside(+source: Views.View,+index: Nat,+length: U32,fits: {Views.fits_nonempty(False{},source,length) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(address(source,index)),U32.to_nat(length)) == True{} : Bool}:
  Views.View{+axes,+offset} = source
  +start = CursorModel.prepare_axes(axes,[])
  CursorBounds.read_inside(CursorOrder.iterate(index,start),U32.to_nat(offset),length,CursorOrder.iterate_valid(index,start,CursorLaws.prepared_valid(axes,[],Unit{})),
    %Equal.sym(Nat,CursorModel.span(CursorOrder.iterate(index,start)),CursorModel.span(start),iterate_span(index,start)) :
      {Nat.is_lt(Nat.add(_,U32.to_nat(offset)),U32.to_nat(length)) == True{} : Bool}
    CursorBounds.fits_bound(axes,offset,length,fits))
