# The values of the dense elementwise combination through views. The left
# operand is combined in place; the right operand is read through its view
# (canonical for equal extents, stretched under broadcasting). When the call
# returns Ready, the result is well formed and holds, at every index, the
# operation applied to the left value at that index and the right value at
# the address of that index in the right view. The Ready result is the only
# premise about the call; its guards are read from the call itself.
import Base
import ../tensor.bend as Tensors
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ../tensor_access.bend as Access
import ../tensor_operations.bend as Operations
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./storage_shape.bend as Shape
import ./storage_refinement.bend as StorageLaws
import ./tensor_dense_contract.bend as Dense
import ./tensor_copy_values.bend as CopyValues
import ./tensor_combine_values.bend as CombineValues
import ./tensor_gather_values.bend as GatherValues
import ./tensor_band_combine_leaf.bend as BandCombineLeaf
import ./tensor_access_refinement.bend as AccessLaws
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_view_metadata_refinement.bend as Metadata
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_completion.bend as CountCompletion
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges

# The two logical operands and the view the right one is read through.
type Combine<-Left: Data,-Right: Data> is Data:
  Combine{left: Left,right: Right,source: Views.View}

def value(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,context: Combine<Left,Right>,index: U32) -> Element:
  Combine{left,right,source} = context
  +index = index
  operation(left_value(left,index),right_value(right,CopyValues.address(source,U32.to_nat(index))))

def combined_content(~Element: Data,~operation: Element -> Element -> Element,+count: U32,+source: Views.View,output: Buffer.Content<Element>,input: Buffer.Content<Element>) ->
  Buffer.Content<Element>:
  Buffer.Content{output,length} = output
  Buffer.Content{input,input_length} = input
  Buffer.Content{BandCombineLeaf.readback_output(~Element,AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(source) && U32.is_zero(0),U32.to_nat(count),source,0,input,output)),length}

def and_true(+value: Bool) -> {value == (value && True{}) : Bool}:
  match value:
    case True{}: {==}
    case False{}: {==}

# One index of the combination into a left operand at offset zero.
def cell(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,+left: Left,+right: Right,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,+count: U32,+index: Nat,
  ready: Buffer.Sized<Element,output>,input_ready: Buffer.Sized<Element,input>,left_values: Dense.Values(~Element,~Left,~left_value,left,Buffer.content(~Element,output)),right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.content(~Element,input)),
  +written: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == U32.to_nat(count) : Nat},
  fits: {Views.fits(source,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool},
  +inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  Dense.At(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input)),index):
  Buffer.Model{+output_tree,+output_length,+output_certificate} = output
  Buffer.Model{+input_tree,+input_length,+input_certificate} = input
  Buffer.Sized{+depth_ok,+room} = ready
  Buffer.Sized{+input_depth_ok,+input_room} = input_ready
  +output = {Buffer.Model{output_tree,output_length,output_certificate} : Buffer.Model<Element>}
  +input = {Buffer.Model{input_tree,input_length,input_certificate} : Buffer.Model<Element>}
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
  +position = U32.from_nat(index)
  Equal.trans(Element,
    Tree.read(Element,BandCombineLeaf.readback_output(~Element,AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(source) && U32.is_zero(0),U32.to_nat(count),source,0,input_tree,output_tree)),position),
    operation(Tree.read(Element,output_tree,position),Tree.read(Element,input_tree,address)),
    operation(left_value(left,position),right_value(right,CopyValues.address(source,U32.to_nat(position)))),
    CombineValues.combined_value(~Element,~operation,Access.zero_contiguous(source) && U32.is_zero(0),source,input_tree,output_tree,output_length,U32.to_nat(count),index,and_true(Access.zero_contiguous(source)),
      Buffer.content_ready(~Element,output,Buffer.Sized{depth_ok,room}),capacity_ok,count_room,inside,exact),
    %Equal.sym(Nat,U32.to_nat(position),index,encoded) :
      {operation(Tree.read(Element,output_tree,position),Tree.read(Element,input_tree,address)) == operation(left_value(left,position),right_value(right,CopyValues.address(source,_))) : Element}
    Equal.trans(Element,operation(Tree.read(Element,output_tree,position),Tree.read(Element,input_tree,address)),
      operation(left_value(left,position),Tree.read(Element,input_tree,address)),operation(left_value(left,position),right_value(right,address)),
      Equal.cong(Element,Element,element => operation(element,Tree.read(Element,input_tree,address)),Tree.read(Element,output_tree,position),left_value(left,position),
        Dense.read(~Element,~Left,~left_value,left,Buffer.Content{output_tree,output_length},index,left_values,
          %written : {Nat.is_lt(index,_) == True{} : Bool} inside)),
      Equal.cong(Element,Element,element => operation(left_value(left,position),element),Tree.read(Element,input_tree,address),right_value(right,address),
        %Arithmetic.native_roundtrip(address) : {Tree.read(Element,input_tree,_) == right_value(right,_) : Element}
        Dense.read(~Element,~Right,~right_value,right,Buffer.Content{input_tree,input_length},U32.to_nat(address),right_values,address_inside))))

def cells_range(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,remaining: Nat,+position: Nat,+left: Left,+right: Right,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,+count: U32,
  +ready: Buffer.Sized<Element,output>,+input_ready: Buffer.Sized<Element,input>,+left_values: Dense.Values(~Element,~Left,~left_value,left,Buffer.content(~Element,output)),+right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.content(~Element,input)),
  +written: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == U32.to_nat(count) : Nat},
  +fits: {Views.fits(source,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(position,remaining),U32.to_nat(count)) == True{} : Bool}) ->
  Members.All(Nat,{index => Dense.At(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input)),index) : Nat -> Data},Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{cell(~Element,~Left,~Right,~left_value,~right_value,~operation,left,right,source,output,input,count,position,ready,input_ready,left_values,right_values,written,exact,fits,Order.room_index(position,rest,U32.to_nat(count),room)),
        cells_range(~Element,~Left,~Right,~left_value,~right_value,~operation,rest,1n+position,left,right,source,output,input,count,ready,input_ready,left_values,right_values,written,exact,fits,Order.advance_room(position,rest,U32.to_nat(count),room))}

def program_length(~Element: Data,~operation: Element -> Element -> Element,+count: U32,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>) ->
  {Buffer.content_length(~Element,combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input))) == Buffer.length(~Element,output) : Nat}:
  Buffer.Model{output_tree,output_length,output_certificate} = output
  Buffer.Model{input_tree,input_length,input_certificate} = input
  {==}

def program_ready(~Element: Data,~operation: Element -> Element -> Element,+count: U32,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,ready: Buffer.Sized<Element,output>) ->
  Buffer.ContentReady<Element,combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input))>:
  Buffer.Model{+output_tree,+output_length,+output_certificate} = output
  Buffer.Model{+input_tree,input_length,input_certificate} = input
  CombineValues.combined_ready(~Element,~operation,Access.zero_contiguous(source) && U32.is_zero(0),U32.to_nat(count),source,input_tree,output_tree,output_length,
    Buffer.content_ready(~Element,Buffer.Model{output_tree,output_length,output_certificate},ready))

def program_facts(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,+left: Left,+right: Right,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,+count: U32,+left_shape: List<&2,U32>,
  +ready: Buffer.Sized<Element,output>,+input_ready: Buffer.Sized<Element,input>,+left_values: Dense.Values(~Element,~Left,~left_value,left,Buffer.content(~Element,output)),+right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.content(~Element,input)),
  +written: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == U32.to_nat(count) : Nat},
  +fits: {Views.fits(source,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool},
  length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(left_shape) : Nat}) ->
  GatherValues.Facts(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},left_shape,combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input))):
  Shape.Both{
    Dense.WellFormed{
      Equal.trans(Nat,Buffer.content_length(~Element,combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input))),Buffer.length(~Element,output),ShapeLaws.elements(left_shape),
        program_length(~Element,~operation,count,source,output,input),length_exact),
      program_ready(~Element,~operation,count,source,output,input,ready)},
    %Equal.sym(Nat,Buffer.content_length(~Element,combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input))),U32.to_nat(count),
      Equal.trans(Nat,Buffer.content_length(~Element,combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input))),Buffer.length(~Element,output),U32.to_nat(count),
        program_length(~Element,~operation,count,source,output,input),Equal.sym(Nat,U32.to_nat(count),Buffer.length(~Element,output),written))) :
      Members.All(Nat,{index => Dense.At(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input)),index) : Nat -> Data},Ranges.range(_,0n))
    cells_range(~Element,~Left,~Right,~left_value,~right_value,~operation,U32.to_nat(count),0n,left,right,source,output,input,count,ready,input_ready,left_values,right_values,written,exact,fits,Order.reflexive(U32.to_nat(count)))}

# The guard of the checked combination, over the models of its two owners.
def guard(~Element: Data,+target: Views.View,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>) -> Bool:
  Shapes.same(Views.shape(target),Views.shape(source)) && Views.is_contiguous(target) &&
    Operations.valid_buffer(Buffer.length(~Element,output),U32.shln(1,Buffer.depth(~Element,output)),target) &&
    Operations.valid_buffer(Buffer.length(~Element,input),U32.shln(1,Buffer.depth(~Element,input)),source)

type Counted<-target: Views.View> is Data:
  Counted{count: U32,counted: {Views.numel(target) == Some{count} : Maybe<&2,U32>}}

# An accepted combination passed its guard and has the count of its target.
def model_guard(~Element: Data,~operation: Element -> Element -> Element,valid: Bool,count: Maybe<&2,U32>,+source: Views.View,+target: Views.View,+input_length: Nat,+output_length: Nat,
  +input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+counts: {Views.numel(target) == count : Maybe<&2,U32>},
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{valid,count,source,target,input_length,output_length},input,output)) == True{} : Bool}) ->
  Shape.Both<{valid == True{} : Bool},Counted<target>>:
  match valid count:
    case True{} Some{+count}: Shape.Both{{==},Counted{count,counts}}
    case True{} None{}: Empty.absurd(Shape.Both<{True{} == True{} : Bool},Counted<target>>,WordEncoding.false_true(accepted))
    case False{} Some{count}: Empty.absurd(Shape.Both<{False{} == True{} : Bool},Counted<target>>,WordEncoding.false_true(accepted))
    case False{} None{}: Empty.absurd(Shape.Both<{False{} == True{} : Bool},Counted<target>>,WordEncoding.false_true(accepted))

type Guarded<-Element: Data,-target: Views.View,-source: Views.View,-output: Buffer.Model<Element>,-input: Buffer.Model<Element>> is Data:
  Guarded{count: U32,counted: {Views.numel(target) == Some{count} : Maybe<&2,U32>},
    guard: {guard(~Element,target,source,output,input) == True{} : Bool},
    same: {Shapes.same(Views.shape(target),Views.shape(source)) == True{} : Bool},
    fits: {Views.fits(source,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool}}

def guarded_selected(~Element: Data,+target: Views.View,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,
  facts: Shape.Both<{guard(~Element,target,source,output,input) == True{} : Bool},Counted<target>>) -> Guarded<Element,target,source,output,input>:
  Buffer.Model{+output_tree,+output_length,Certificate.Certificate{+output_depth,+output_equation}} = output
  Buffer.Model{+input_tree,+input_length,Certificate.Certificate{+input_depth,+input_equation}} = input
  Shape.Both{+truth,Counted{+count,+counted}} = facts
  +same = Shapes.same(Views.shape(target),Views.shape(source))
  +contiguous = Views.is_contiguous(target)
  +output_valid = Operations.valid_buffer(output_length,U32.shln(1,output_depth),target)
  +input_valid = Operations.valid_buffer(input_length,U32.shln(1,input_depth),source)
  Guarded{count,counted,truth,
    Boolean.left(same,contiguous,Boolean.left((same && contiguous : Bool),output_valid,Boolean.left((same && contiguous && output_valid : Bool),input_valid,truth))),
    Boolean.right(Nat.is_le(input_length,U32.to_nat(U32.shln(1,input_depth))),Views.fits(source,U32.from_nat(input_length)),
      Boolean.right((same && contiguous && output_valid : Bool),input_valid,truth))}

def guarded(~Element: Data,~operation: Element -> Element -> Element,+target: Views.View,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,output),target,Buffer.pack(~Element,input),source)) == True{} : Bool}) ->
  Guarded<Element,target,source,output,input>:
  Buffer.Model{+output_tree,+output_length,Certificate.Certificate{+output_depth,+output_equation}} = output
  Buffer.Model{+input_tree,+input_length,Certificate.Certificate{+input_depth,+input_equation}} = input
  guarded_selected(~Element,target,source,Buffer.Model{output_tree,output_length,Certificate.Certificate{output_depth,output_equation}},
    Buffer.Model{input_tree,input_length,Certificate.Certificate{input_depth,input_equation}},
    model_guard(~Element,~operation,(Shapes.same(Views.shape(target),Views.shape(source)) && Views.is_contiguous(target) && Operations.valid_buffer(output_length,U32.shln(1,output_depth),target) && Operations.valid_buffer(input_length,U32.shln(1,input_depth),source) : Bool),Views.numel(target),source,target,input_length,output_length,
      Tree.reflect(Element,Tree.pack(Element,input_tree)),Tree.reflect(Element,Tree.pack(Element,output_tree)),{==},accepted))

# The observed left content of an accepted combination is the model above.
def denoted_content(~Element: Data,~operation: Element -> Element -> Element,+count: U32,+target_axes: List<&2,Views.Axis>,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,
  guarded: {guard(~Element,Views.View{target_axes,0},source,output,input) == True{} : Bool},counted: {Views.numel(Views.View{target_axes,0}) == Some{count} : Maybe<&2,U32>}) ->
  {BandCombineLeaf.left_content(~Element,OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)) == combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input)) : Buffer.Content<Element>}:
  Buffer.Model{+output_tree,+output_length,Certificate.Certificate{+output_depth,+output_equation}} = output
  Buffer.Model{+input_tree,+input_length,Certificate.Certificate{+input_depth,+input_equation}} = input
  +target = {Views.View{target_axes,0} : Views.View}
  +model = {Buffer.Content{BandCombineLeaf.readback_output(~Element,AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(source) && U32.is_zero(0),U32.to_nat(count),source,0,input_tree,output_tree)),output_length} : Buffer.Content<Element>}
  %Equal.sym(Bool,(Shapes.same(Views.shape(target),Views.shape(source)) && Views.is_contiguous(target) && Operations.valid_buffer(output_length,U32.shln(1,output_depth),target) && Operations.valid_buffer(input_length,U32.shln(1,input_depth),source) : Bool),True{},guarded) :
    {BandCombineLeaf.left_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{_,Views.numel(target),source,target,input_length,output_length},
      Tree.reflect(Element,Tree.pack(Element,input_tree)),Tree.reflect(Element,Tree.pack(Element,output_tree)))) == model : Buffer.Content<Element>}
  %Equal.sym(Maybe<&2,U32>,Views.numel(target),Some{count},counted) :
    {BandCombineLeaf.left_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{True{},_,source,target,input_length,output_length},
      Tree.reflect(Element,Tree.pack(Element,input_tree)),Tree.reflect(Element,Tree.pack(Element,output_tree)))) == model : Buffer.Content<Element>}
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,input_tree)),input_tree,StorageLaws.reflect_packed(Element,input_tree)) :
    {BandCombineLeaf.left_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{True{},Some{count},source,target,input_length,output_length},
      _,Tree.reflect(Element,Tree.pack(Element,output_tree)))) == model : Buffer.Content<Element>}
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,output_tree)),output_tree,StorageLaws.reflect_packed(Element,output_tree)) :
    {BandCombineLeaf.left_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{True{},Some{count},source,target,input_length,output_length},
      input_tree,_)) == model : Buffer.Content<Element>}
  BandCombineLeaf.model_contents(~Element,output_length,input_length,AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(source) && U32.is_zero(0),U32.to_nat(count),source,0,input_tree,output_tree))

# The observed outcome of the checked combination decides the status of the result.
def ready_accepted(~Element: Data,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,+name: String,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  {Dense.tensor_ready(~Element,Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,outcome)) == GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome)) : Bool}:
  match outcome:
    case Operations.Accepted{(left,right)}: {==}
    case Operations.Rejected{(left,right)}: {==}

def combined_contract(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,+context: Context,+name: String,+left_shape: List<&2,U32>,
  +right_shape: List<&2,U32>,+fill: Element,+workers: U32,
  facts: GatherValues.Facts(~Element,~Context,~reader,context,left_shape,BandCombineLeaf.left_content(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome))),
  accepted: {Dense.tensor_ready(~Element,Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,outcome)) == True{} : Bool}) ->
  Dense.Contract(~Element,~Context,~reader,context,Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,outcome)):
  match outcome:
    case Operations.Accepted{(Storage.Buffer{storage,length,certificate},right)}:
      Shape.Both{well_formed,values} = facts
      Shape.Both{Shape.Both{well_formed,{==}},values}
    case Operations.Rejected{(left,right)}:
      Empty.absurd(Dense.Contract(~Element,~Context,~reader,context,Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.Rejected{(left,right)})),WordEncoding.false_true(accepted))

# The right operand comes back unchanged.
def model_right(~Element: Data,+left_length: Nat,+right_length: Nat,+result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) ->
  {CopyValues.right_content(~Element,Operations.Accepted{OperationsLaws.combined_pair(~Element,left_length,right_length,result)}) ==
    Buffer.Content{CombineValues.reader(~Element,result),right_length} : Buffer.Content<Element>}:
  AccessLaws.ModelResult{+reader,output} = result
  Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,right_length},
    Tree.reflect(Element,Tree.pack(Element,reader)),reader,StorageLaws.reflect_packed(Element,reader))

def right_unchanged(~Element: Data,~operation: Element -> Element -> Element,+count: U32,+target_axes: List<&2,Views.Axis>,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,
  guarded: {guard(~Element,Views.View{target_axes,0},source,output,input) == True{} : Bool},counted: {Views.numel(Views.View{target_axes,0}) == Some{count} : Maybe<&2,U32>}) ->
  {CopyValues.right_content(~Element,OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)) == Buffer.content(~Element,input) : Buffer.Content<Element>}:
  Buffer.Model{+output_tree,+output_length,Certificate.Certificate{+output_depth,+output_equation}} = output
  Buffer.Model{+input_tree,+input_length,Certificate.Certificate{+input_depth,+input_equation}} = input
  +target = {Views.View{target_axes,0} : Views.View}
  +model = {Buffer.Content{input_tree,input_length} : Buffer.Content<Element>}
  +result = AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(source) && U32.is_zero(0),U32.to_nat(count),source,0,input_tree,output_tree)
  %Equal.sym(Bool,(Shapes.same(Views.shape(target),Views.shape(source)) && Views.is_contiguous(target) && Operations.valid_buffer(output_length,U32.shln(1,output_depth),target) && Operations.valid_buffer(input_length,U32.shln(1,input_depth),source) : Bool),True{},guarded) :
    {CopyValues.right_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{_,Views.numel(target),source,target,input_length,output_length},
      Tree.reflect(Element,Tree.pack(Element,input_tree)),Tree.reflect(Element,Tree.pack(Element,output_tree)))) == model : Buffer.Content<Element>}
  %Equal.sym(Maybe<&2,U32>,Views.numel(target),Some{count},counted) :
    {CopyValues.right_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{True{},_,source,target,input_length,output_length},
      Tree.reflect(Element,Tree.pack(Element,input_tree)),Tree.reflect(Element,Tree.pack(Element,output_tree)))) == model : Buffer.Content<Element>}
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,input_tree)),input_tree,StorageLaws.reflect_packed(Element,input_tree)) :
    {CopyValues.right_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{True{},Some{count},source,target,input_length,output_length},
      _,Tree.reflect(Element,Tree.pack(Element,output_tree)))) == model : Buffer.Content<Element>}
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,output_tree)),output_tree,StorageLaws.reflect_packed(Element,output_tree)) :
    {CopyValues.right_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{True{},Some{count},source,target,input_length,output_length},
      input_tree,_)) == model : Buffer.Content<Element>}
  Equal.trans(Buffer.Content<Element>,CopyValues.right_content(~Element,Operations.Accepted{OperationsLaws.combined_pair(~Element,output_length,input_length,result)}),
    Buffer.Content{CombineValues.reader(~Element,result),input_length},model,model_right(~Element,output_length,input_length,result),
    Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,input_length},CombineValues.reader(~Element,result),input_tree,
      CombineValues.combined_reader(~Element,~operation,Access.zero_contiguous(source) && U32.is_zero(0),U32.to_nat(count),source,input_tree,output_tree)))

# What an accepted combination leaves in its two owners: the left one holds
# the combined values, the right one is unchanged.
def Step(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,+left: Left,+right: Right,+source: Views.View,+left_shape: List<&2,U32>,-owners: Operations.Outcome<OperationsLaws.PairOwners(Element)>,-right_content: Buffer.Content<Element>) -> Data:
  Shape.Both<GatherValues.Facts(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},left_shape,BandCombineLeaf.left_content(~Element,owners)),{CopyValues.right_content(~Element,owners) == right_content : Buffer.Content<Element>}>

def guarded_facts(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+left: Left,+right: Right,+target_axes: List<&2,Views.Axis>,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,+left_shape: List<&2,U32>,
  +ready: Buffer.Sized<Element,output>,+input_ready: Buffer.Sized<Element,input>,+left_values: Dense.Values(~Element,~Left,~left_value,left,Buffer.content(~Element,output)),+right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.content(~Element,input)),
  +target_exact: {ShapeLaws.elements(Views.axis_extents(target_axes)) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(left_shape) : Nat},
  facts: Guarded<Element,Views.View{target_axes,0},source,output,input>) ->
  Step(~Element,~Left,~Right,~left_value,~right_value,~operation,left,right,source,left_shape,OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)),Buffer.content(~Element,input)):
  Guarded{+count,+counted,+guard,+same,+fits} = facts
  +count_exact = {
    %counted : ShapeLaws.Count(_,ShapeLaws.elements(Views.axis_extents(target_axes)))
    ShapeLaws.numel(Views.axis_extents(target_axes)) : {ShapeLaws.elements(Views.axis_extents(target_axes)) == U32.to_nat(count) : Nat}}
  +written = Equal.trans(Nat,U32.to_nat(count),ShapeLaws.elements(Views.axis_extents(target_axes)),Buffer.length(~Element,output),
    Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(target_axes)),U32.to_nat(count),count_exact),target_exact)
  +exact = {
    %GatherValues.same_extents(Views.View{target_axes,0},source,same) : {ShapeLaws.elements(_) == U32.to_nat(count) : Nat}
    count_exact : {ShapeLaws.elements(Views.axis_extents_of(source)) == U32.to_nat(count) : Nat}}
  Shape.Both{
    %Equal.sym(Operations.Outcome<OperationsLaws.PairOwners(Element)>,OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)),OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source),OperationsLaws.combine(~Element,~operation,~kernel,~kernel_law,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)) :
      GatherValues.Facts(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},left_shape,BandCombineLeaf.left_content(~Element,_))
    %Equal.sym(Buffer.Content<Element>,BandCombineLeaf.left_content(~Element,OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)),combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input)),
      denoted_content(~Element,~operation,count,target_axes,source,output,input,guard,counted)) : GatherValues.Facts(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},left_shape,_)
    program_facts(~Element,~Left,~Right,~left_value,~right_value,~operation,left,right,source,output,input,count,left_shape,ready,input_ready,left_values,right_values,written,exact,fits,length_exact),
    %Equal.sym(Operations.Outcome<OperationsLaws.PairOwners(Element)>,OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)),OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source),OperationsLaws.combine(~Element,~operation,~kernel,~kernel_law,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)) :
      {CopyValues.right_content(~Element,_) == Buffer.content(~Element,input) : Buffer.Content<Element>}
    right_unchanged(~Element,~operation,count,target_axes,source,output,input,guard,counted)}

def modeled_facts(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+left: Left,+right: Right,+target_axes: List<&2,Views.Axis>,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,+left_shape: List<&2,U32>,
  +ready: Buffer.Sized<Element,output>,+input_ready: Buffer.Sized<Element,input>,+left_values: Dense.Values(~Element,~Left,~left_value,left,Buffer.content(~Element,output)),+right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.content(~Element,input)),
  +target_exact: {ShapeLaws.elements(Views.axis_extents(target_axes)) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(left_shape) : Nat},
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source))) == True{} : Bool}) ->
  Step(~Element,~Left,~Right,~left_value,~right_value,~operation,left,right,source,left_shape,OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)),Buffer.content(~Element,input)):
  guarded_facts(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left,right,target_axes,source,output,input,left_shape,ready,input_ready,left_values,right_values,target_exact,length_exact,
    guarded(~Element,~operation,Views.View{target_axes,0},source,output,input,
      %OperationsLaws.combine(~Element,~operation,~kernel,~kernel_law,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source) : {GatherValues.accepted_owners(~Element,_) == True{} : Bool}
      accepted))

def stepped_contract(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+left: Left,+right: Right,+name: String,+target_axes: List<&2,Views.Axis>,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+fill: Element,+workers: U32,
  step: Step(~Element,~Left,~Right,~left_value,~right_value,~operation,left,right,source,left_shape,OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)),Buffer.content(~Element,input)),
  accepted: {Dense.tensor_ready(~Element,Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source))):
  Shape.Both{facts,unchanged} = step
  combined_contract(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source),Combine{left,right,source},name,left_shape,right_shape,fill,workers,facts,accepted)

def modeled_call(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+left: Left,+right: Right,+name: String,+target_axes: List<&2,Views.Axis>,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+fill: Element,+workers: U32,
  +ready: Buffer.Sized<Element,output>,+input_ready: Buffer.Sized<Element,input>,+left_values: Dense.Values(~Element,~Left,~left_value,left,Buffer.content(~Element,output)),+right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.content(~Element,input)),
  +target_exact: {ShapeLaws.elements(Views.axis_extents(target_axes)) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(left_shape) : Nat},
  +accepted: {Dense.tensor_ready(~Element,Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source))):
  stepped_contract(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left,right,name,target_axes,source,output,input,left_shape,right_shape,fill,workers,
    modeled_facts(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left,right,target_axes,source,output,input,left_shape,ready,input_ready,left_values,right_values,target_exact,length_exact,
      Equal.trans(Bool,GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source))),Dense.tensor_ready(~Element,Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source))),True{},
        Equal.sym(Bool,Dense.tensor_ready(~Element,Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source))),GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source))),
          ready_accepted(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source),name,left_shape,right_shape,fill,workers)),accepted)),
    accepted)

def observed(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+left: Left,+right: Right,+name: String,+target_axes: List<&2,Views.Axis>,+source: Views.View,-left_storage: Storage.Buffer<Element>,-right_storage: Storage.Buffer<Element>,
  +left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+fill: Element,+workers: U32,
  left_observation: Buffer.Observation<Element,left_storage>,right_observation: Buffer.Observation<Element,right_storage>,
  left_facts: GatherValues.Facts(~Element,~Left,~left_value,left,left_shape,Buffer.contents(~Element,left_storage)),
  right_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,right_storage)>,
  right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.contents(~Element,right_storage)),
  target_exact: {ShapeLaws.elements(Views.axis_extents(target_axes)) == ShapeLaws.elements(left_shape) : Nat},
  accepted: {Dense.tensor_ready(~Element,Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,left_storage,Views.View{target_axes,0},right_storage,source))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,left_storage,Views.View{target_axes,0},right_storage,source))):
  Buffer.Observation{+output,+output_equation} = left_observation
  Buffer.Observation{+input,+input_equation} = right_observation
  Shape.Both{Dense.WellFormed{length_exact,content_ready},left_values} = left_facts
  +length_model = {
    %Buffer.content_length_of(~Element,output) : {_ == ShapeLaws.elements(left_shape) : Nat}
    %Buffer.packed_contents(~Element,output) : {Buffer.content_length(~Element,_) == ShapeLaws.elements(left_shape) : Nat}
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,output),left_storage,output_equation) : {Buffer.content_length(~Element,Buffer.contents(~Element,_)) == ShapeLaws.elements(left_shape) : Nat}
    length_exact : {Buffer.length(~Element,output) == ShapeLaws.elements(left_shape) : Nat}}
  %output_equation : Dense.Contract(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,_,Views.View{target_axes,0},right_storage,source)))
  %input_equation : Dense.Contract(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},_,source)))
  modeled_call(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left,right,name,target_axes,source,output,input,left_shape,right_shape,fill,workers,
    Buffer.ready_content(~Element,output,
      %Buffer.packed_contents(~Element,output) : Buffer.ContentReady<Element,_>
      %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,output),left_storage,output_equation) : Buffer.ContentReady<Element,Buffer.contents(~Element,_)>
      content_ready),
    Buffer.ready_content(~Element,input,
      %Buffer.packed_contents(~Element,input) : Buffer.ContentReady<Element,_>
      %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,input),right_storage,input_equation) : Buffer.ContentReady<Element,Buffer.contents(~Element,_)>
      right_ready),
    %Buffer.packed_contents(~Element,output) : Dense.Values(~Element,~Left,~left_value,left,_)
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,output),left_storage,output_equation) : Dense.Values(~Element,~Left,~left_value,left,Buffer.contents(~Element,_))
    left_values,
    %Buffer.packed_contents(~Element,input) : Dense.Values(~Element,~Right,~right_value,right,_)
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,input),right_storage,input_equation) : Dense.Values(~Element,~Right,~right_value,right,Buffer.contents(~Element,_))
    right_values,
    Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(target_axes)),ShapeLaws.elements(left_shape),Buffer.length(~Element,output),target_exact,
      Equal.sym(Nat,Buffer.length(~Element,output),ShapeLaws.elements(left_shape),length_model)),
    length_model,
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,input),right_storage,input_equation) : {Dense.tensor_ready(~Element,Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},_,source))) == True{} : Bool}
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,output),left_storage,output_equation) : {Dense.tensor_ready(~Element,Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,_,Views.View{target_axes,0},right_storage,source))) == True{} : Bool}
    accepted)

# The checked combination of two dense storages through a left view at offset
# zero and any right view. The left operand has a well-formed content with
# complete values under its own reader; so has the right operand.
def views(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+left: Left,+right: Right,+name: String,+target_axes: List<&2,Views.Axis>,+source: Views.View,left_storage: Storage.Buffer<Element>,right_storage: Storage.Buffer<Element>,
  +left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+fill: Element,+workers: U32,
  left_facts: GatherValues.Facts(~Element,~Left,~left_value,left,left_shape,Buffer.contents(~Element,left_storage)),
  right_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,right_storage)>,
  right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.contents(~Element,right_storage)),
  target_exact: {ShapeLaws.elements(Views.axis_extents(target_axes)) == ShapeLaws.elements(left_shape) : Nat},
  accepted: {Dense.tensor_ready(~Element,Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,left_storage,Views.View{target_axes,0},right_storage,source))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,left_storage,Views.View{target_axes,0},right_storage,source))):
  observed(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left,right,name,target_axes,source,left_storage,right_storage,left_shape,right_shape,fill,workers,
    Buffer.observe(~Element,left_storage),Buffer.observe(~Element,right_storage),left_facts,right_ready,right_values,target_exact,accepted)

# The canonical view of the left operand starts at offset zero and has its extents.
def targeted(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),target: Views.View,+left: Left,+right: Right,+name: String,+source: Views.View,left_storage: Storage.Buffer<Element>,right_storage: Storage.Buffer<Element>,
  +left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+fill: Element,+workers: U32,
  +targets: {Tensors.view_of(left_shape) == Some{target} : Maybe<&2,Views.View>},
  left_facts: GatherValues.Facts(~Element,~Left,~left_value,left,left_shape,Buffer.contents(~Element,left_storage)),
  right_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,right_storage)>,
  right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.contents(~Element,right_storage)),
  accepted: {Dense.tensor_ready(~Element,Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,left_storage,target,right_storage,source))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,left_storage,target,right_storage,source))):
  Views.View{+target_axes,+target_offset} = target
  +offset_zero = {
    %targets : {Metadata.observed_offset(_,0) == 0 : U32}
    Metadata.contiguous_count_offset(Shapes.numel(Shapes.Shape{left_shape}),left_shape,0) : {target_offset == 0 : U32}}
  +target_extents = {
    %targets : {Metadata.observed_extents(_,left_shape) == left_shape : List<&2,U32>}
    Metadata.contiguous_shape(left_shape,0) : {Views.axis_extents(target_axes) == left_shape : List<&2,U32>}}
  %Equal.sym(U32,target_offset,0,offset_zero) : Dense.Contract(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,left_storage,Views.View{target_axes,_},right_storage,source)))
  views(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left,right,name,target_axes,source,left_storage,right_storage,left_shape,right_shape,fill,workers,left_facts,right_ready,right_values,
    %Equal.sym(List<&2,U32>,Views.axis_extents(target_axes),left_shape,target_extents) : {ShapeLaws.elements(_) == ShapeLaws.elements(left_shape) : Nat}
    {==},
    %offset_zero : {Dense.tensor_ready(~Element,Tensors.combined(~Element,name,left_shape,right_shape,fill,workers,Operations.combine_with(~Element,~kernel,left_storage,Views.View{target_axes,_},right_storage,source))) == True{} : Bool}
    accepted)

# The statement for a right view an operation computes: a Ready result has
# that view, and the right values are read through it.
def Combined(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,+left: Left,+right: Right,source: Maybe<&2,Views.View>,-tensor: Tensors.Tensor<Element>) -> Data:
  match source:
    case Some{source}: Dense.Contract(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},tensor)
    case None{}: Empty

# The public in-place combination: the left view is the canonical view of the
# left extents; the right view is canonical for equal extents and stretched
# to the left extents under broadcasting.
def combine_views(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),target: Maybe<&2,Views.View>,source: Maybe<&2,Views.View>,+left: Left,+right: Right,+name: String,
  left_storage: Storage.Buffer<Element>,right_storage: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+fill: Element,+workers: U32,
  +targets: {Tensors.view_of(left_shape) == target : Maybe<&2,Views.View>},
  left_facts: GatherValues.Facts(~Element,~Left,~left_value,left,left_shape,Buffer.contents(~Element,left_storage)),
  right_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,right_storage)>,
  right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.contents(~Element,right_storage)),
  accepted: {Dense.tensor_ready(~Element,Tensors.combine_views_with(~Element,~kernel,name,target,source,left_storage,left_shape,fill,right_storage,right_shape,workers)) == True{} : Bool}) ->
  Combined(~Element,~Left,~Right,~left_value,~right_value,~operation,left,right,source,Tensors.combine_views_with(~Element,~kernel,name,target,source,left_storage,left_shape,fill,right_storage,right_shape,workers)):
  match target source:
    case Some{+target} Some{+source}:
      targeted(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,target,left,right,name,source,left_storage,right_storage,left_shape,right_shape,fill,workers,targets,left_facts,right_ready,right_values,accepted)
    case Some{target} None{}: WordEncoding.false_true(accepted)
    case None{} Some{+source}: Empty.absurd(Dense.Contract(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},Tensors.combine_views_with(~Element,~kernel,name,None{},Some{source},left_storage,left_shape,fill,right_storage,right_shape,workers)),WordEncoding.false_true(accepted))
    case None{} None{}: WordEncoding.false_true(accepted)

# Broadcasting to extents that are not the left extents: the left operand is
# first gathered through its stretched view into fresh storage, and the right
# operand is combined into that storage through its own stretched view.
def fresh_copied(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),copied: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,+left: Left,+right: Right,+name: String,+target_axes: List<&2,Views.Axis>,+source: Views.View,right_storage: Storage.Buffer<Element>,
  +shape: List<&2,U32>,+fill: Element,+workers: U32,
  copy_facts: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,copied)) == True{} : Bool} ->
    GatherValues.Facts(~Element,~Left,~left_value,left,shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,copied))),
  right_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,right_storage)>,
  right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.contents(~Element,right_storage)),
  target_exact: {ShapeLaws.elements(Views.axis_extents(target_axes)) == ShapeLaws.elements(shape) : Nat},
  accepted: {Dense.tensor_ready(~Element,Tensors.combined_fresh_with(~Element,~kernel,name,shape,fill,workers,right_storage,source,Views.View{target_axes,0},copied)) == True{} : Bool}) ->
  Dense.Contract(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},Tensors.combined_fresh_with(~Element,~kernel,name,shape,fill,workers,right_storage,source,Views.View{target_axes,0},copied)):
  match copied:
    case Operations.Accepted{(input,Storage.Buffer{storage,length,certificate})}:
      views(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left,right,name,target_axes,source,Storage.Buffer{storage,length,certificate},right_storage,shape,shape,fill,workers,copy_facts({==}),right_ready,right_values,target_exact,accepted)
    case Operations.Rejected{(input,output)}:
      Empty.absurd(Dense.Contract(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},Tensors.combined_fresh_with(~Element,~kernel,name,shape,fill,workers,right_storage,source,Views.View{target_axes,0},Operations.Rejected{(input,output)})),WordEncoding.false_true(accepted))

def Fresh(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,+left: Left,+right: Right,left_view: Maybe<&2,Views.View>,right_view: Maybe<&2,Views.View>,-tensor: Tensors.Tensor<Element>) -> Data:
  match left_view right_view:
    case Some{left_view} Some{source}: Dense.Contract(~Element,~Combine<GatherValues.Gather<Left>,Right>,~value(~Element,~GatherValues.Gather<Left>,~Right,~GatherValues.value(~Element,~Left,~left_value),~right_value,~operation),Combine{GatherValues.Gather{left,left_view},right,source},tensor)
    case left_view right_view: Empty

def fresh_targeted(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),target: Views.View,+left: Left,+right: Right,+name: String,+left_view: Views.View,+source: Views.View,+left_model: Buffer.Model<Element>,
  right_storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,+count: U32,
  +targets: {Tensors.view_of(shape) == Some{target} : Maybe<&2,Views.View>},
  +counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  +left_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,Buffer.pack(~Element,left_model))>,
  +left_values: Dense.Values(~Element,~Left,~left_value,left,Buffer.contents(~Element,Buffer.pack(~Element,left_model))),
  right_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,right_storage)>,
  right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.contents(~Element,right_storage)),
  accepted: {Dense.tensor_ready(~Element,Tensors.combined_fresh_with(~Element,~kernel,name,shape,fill,workers,right_storage,source,target,Operations.copy(~Element,Buffer.pack(~Element,left_model),left_view,Tensors.buffer(~Element,fill,count),target))) == True{} : Bool}) ->
  Dense.Contract(~Element,~Combine<GatherValues.Gather<Left>,Right>,~value(~Element,~GatherValues.Gather<Left>,~Right,~GatherValues.value(~Element,~Left,~left_value),~right_value,~operation),Combine{GatherValues.Gather{left,left_view},right,source},Tensors.combined_fresh_with(~Element,~kernel,name,shape,fill,workers,right_storage,source,target,Operations.copy(~Element,Buffer.pack(~Element,left_model),left_view,Tensors.buffer(~Element,fill,count),target))):
  Views.View{+target_axes,+target_offset} = target
  +offset_zero = {
    %targets : {Metadata.observed_offset(_,0) == 0 : U32}
    Metadata.contiguous_count_offset(Shapes.numel(Shapes.Shape{shape}),shape,0) : {target_offset == 0 : U32}}
  +target_extents = {
    %targets : {Metadata.observed_extents(_,shape) == shape : List<&2,U32>}
    Metadata.contiguous_shape(shape,0) : {Views.axis_extents(target_axes) == shape : List<&2,U32>}}
  %Equal.sym(U32,target_offset,0,offset_zero) : Dense.Contract(~Element,~Combine<GatherValues.Gather<Left>,Right>,~value(~Element,~GatherValues.Gather<Left>,~Right,~GatherValues.value(~Element,~Left,~left_value),~right_value,~operation),Combine{GatherValues.Gather{left,left_view},right,source},Tensors.combined_fresh_with(~Element,~kernel,name,shape,fill,workers,right_storage,source,Views.View{target_axes,_},Operations.copy(~Element,Buffer.pack(~Element,left_model),left_view,Tensors.buffer(~Element,fill,count),Views.View{target_axes,_})))
  fresh_copied(~Element,~GatherValues.Gather<Left>,~Right,~GatherValues.value(~Element,~Left,~left_value),~right_value,~operation,~kernel,~kernel_law,Operations.copy(~Element,Buffer.pack(~Element,left_model),left_view,Tensors.buffer(~Element,fill,count),Views.View{target_axes,0}),GatherValues.Gather{left,left_view},right,name,target_axes,source,right_storage,shape,fill,workers,
    copied => GatherValues.copy_facts(~Element,~Left,~left_value,left,left_view,Views.View{target_axes,0},shape,Buffer.pack(~Element,left_model),shape,fill,count,
      %offset_zero : {Tensors.view_of(shape) == Some{Views.View{target_axes,_}} : Maybe<&2,Views.View>} targets,
      counts,left_ready,left_values,size => sized => {==},copied),
    right_ready,right_values,
    %Equal.sym(List<&2,U32>,Views.axis_extents(target_axes),shape,target_extents) : {ShapeLaws.elements(_) == ShapeLaws.elements(shape) : Nat}
    {==},
    %offset_zero : {Dense.tensor_ready(~Element,Tensors.combined_fresh_with(~Element,~kernel,name,shape,fill,workers,right_storage,source,Views.View{target_axes,_},Operations.copy(~Element,Buffer.pack(~Element,left_model),left_view,Tensors.buffer(~Element,fill,count),Views.View{target_axes,_}))) == True{} : Bool}
    accepted)

def fresh_counted(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),target: Maybe<&2,Views.View>,count: Maybe<&2,U32>,+left: Left,+right: Right,+name: String,+left_view: Views.View,+source: Views.View,
  +left_model: Buffer.Model<Element>,right_storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +targets: {Tensors.view_of(shape) == target : Maybe<&2,Views.View>},
  +counts: {Tensors.count_of(shape) == count : Maybe<&2,U32>},
  +left_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,Buffer.pack(~Element,left_model))>,
  +left_values: Dense.Values(~Element,~Left,~left_value,left,Buffer.contents(~Element,Buffer.pack(~Element,left_model))),
  right_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,right_storage)>,
  right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.contents(~Element,right_storage)),
  accepted: {Dense.tensor_ready(~Element,Tensors.combine_fresh_views_with(~Element,~kernel,name,Some{left_view},Some{source},target,target,count,Buffer.pack(~Element,left_model),right_storage,shape,fill,workers)) == True{} : Bool}) ->
  Dense.Contract(~Element,~Combine<GatherValues.Gather<Left>,Right>,~value(~Element,~GatherValues.Gather<Left>,~Right,~GatherValues.value(~Element,~Left,~left_value),~right_value,~operation),Combine{GatherValues.Gather{left,left_view},right,source},Tensors.combine_fresh_views_with(~Element,~kernel,name,Some{left_view},Some{source},target,target,count,Buffer.pack(~Element,left_model),right_storage,shape,fill,workers)):
  match target count:
    case Some{+target} Some{+count}:
      fresh_targeted(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,target,left,right,name,left_view,source,left_model,right_storage,shape,fill,workers,count,targets,counts,left_ready,left_values,right_ready,right_values,accepted)
    case Some{target} None{}:
      Empty.absurd(Dense.Contract(~Element,~Combine<GatherValues.Gather<Left>,Right>,~value(~Element,~GatherValues.Gather<Left>,~Right,~GatherValues.value(~Element,~Left,~left_value),~right_value,~operation),Combine{GatherValues.Gather{left,left_view},right,source},Tensors.combine_fresh_views_with(~Element,~kernel,name,Some{left_view},Some{source},Some{target},Some{target},None{},Buffer.pack(~Element,left_model),right_storage,shape,fill,workers)),WordEncoding.false_true(accepted))
    case None{} Some{count}:
      Empty.absurd(Dense.Contract(~Element,~Combine<GatherValues.Gather<Left>,Right>,~value(~Element,~GatherValues.Gather<Left>,~Right,~GatherValues.value(~Element,~Left,~left_value),~right_value,~operation),Combine{GatherValues.Gather{left,left_view},right,source},Tensors.combine_fresh_views_with(~Element,~kernel,name,Some{left_view},Some{source},None{},None{},Some{count},Buffer.pack(~Element,left_model),right_storage,shape,fill,workers)),WordEncoding.false_true(accepted))
    case None{} None{}:
      Empty.absurd(Dense.Contract(~Element,~Combine<GatherValues.Gather<Left>,Right>,~value(~Element,~GatherValues.Gather<Left>,~Right,~GatherValues.value(~Element,~Left,~left_value),~right_value,~operation),Combine{GatherValues.Gather{left,left_view},right,source},Tensors.combine_fresh_views_with(~Element,~kernel,name,Some{left_view},Some{source},None{},None{},None{},Buffer.pack(~Element,left_model),right_storage,shape,fill,workers)),WordEncoding.false_true(accepted))

def fresh_selected(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),left_view: Maybe<&2,Views.View>,right_view: Maybe<&2,Views.View>,+left: Left,+right: Right,+name: String,
  +left_model: Buffer.Model<Element>,right_storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +left_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,Buffer.pack(~Element,left_model))>,
  +left_values: Dense.Values(~Element,~Left,~left_value,left,Buffer.contents(~Element,Buffer.pack(~Element,left_model))),
  right_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,right_storage)>,
  right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.contents(~Element,right_storage)),
  accepted: {Dense.tensor_ready(~Element,Tensors.combine_fresh_views_with(~Element,~kernel,name,left_view,right_view,Tensors.view_of(shape),Tensors.view_of(shape),Tensors.count_of(shape),Buffer.pack(~Element,left_model),right_storage,shape,fill,workers)) == True{} : Bool}) ->
  Fresh(~Element,~Left,~Right,~left_value,~right_value,~operation,left,right,left_view,right_view,Tensors.combine_fresh_views_with(~Element,~kernel,name,left_view,right_view,Tensors.view_of(shape),Tensors.view_of(shape),Tensors.count_of(shape),Buffer.pack(~Element,left_model),right_storage,shape,fill,workers)):
  match left_view right_view:
    case Some{+left_view} Some{+source}:
      fresh_counted(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,Tensors.view_of(shape),Tensors.count_of(shape),left,right,name,left_view,source,left_model,right_storage,shape,fill,workers,{==},{==},left_ready,left_values,right_ready,right_values,accepted)
    case Some{left_view} None{}: WordEncoding.false_true(accepted)
    case None{} Some{source}: WordEncoding.false_true(accepted)
    case None{} None{}: WordEncoding.false_true(accepted)

def fresh_observed(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+left: Left,+right: Right,+name: String,-left_storage: Storage.Buffer<Element>,right_storage: Storage.Buffer<Element>,
  left_view: Maybe<&2,Views.View>,right_view: Maybe<&2,Views.View>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  observation: Buffer.Observation<Element,left_storage>,
  left_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,left_storage)>,
  left_values: Dense.Values(~Element,~Left,~left_value,left,Buffer.contents(~Element,left_storage)),
  right_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,right_storage)>,
  right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.contents(~Element,right_storage)),
  accepted: {Dense.tensor_ready(~Element,Tensors.combine_fresh_views_with(~Element,~kernel,name,left_view,right_view,Tensors.view_of(shape),Tensors.view_of(shape),Tensors.count_of(shape),left_storage,right_storage,shape,fill,workers)) == True{} : Bool}) ->
  Fresh(~Element,~Left,~Right,~left_value,~right_value,~operation,left,right,left_view,right_view,Tensors.combine_fresh_views_with(~Element,~kernel,name,left_view,right_view,Tensors.view_of(shape),Tensors.view_of(shape),Tensors.count_of(shape),left_storage,right_storage,shape,fill,workers)):
  Buffer.Observation{+left_model,+equation} = observation
  %equation : Fresh(~Element,~Left,~Right,~left_value,~right_value,~operation,left,right,left_view,right_view,Tensors.combine_fresh_views_with(~Element,~kernel,name,left_view,right_view,Tensors.view_of(shape),Tensors.view_of(shape),Tensors.count_of(shape),_,right_storage,shape,fill,workers))
  fresh_selected(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left_view,right_view,left,right,name,left_model,right_storage,shape,fill,workers,
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,left_model),left_storage,equation) : Buffer.ContentReady<Element,Buffer.contents(~Element,_)>
    left_ready,
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,left_model),left_storage,equation) : Dense.Values(~Element,~Left,~left_value,left,Buffer.contents(~Element,_))
    left_values,
    right_ready,right_values,
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,left_model),left_storage,equation) : {Dense.tensor_ready(~Element,Tensors.combine_fresh_views_with(~Element,~kernel,name,left_view,right_view,Tensors.view_of(shape),Tensors.view_of(shape),Tensors.count_of(shape),_,right_storage,shape,fill,workers)) == True{} : Bool}
    accepted)

def fresh(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+left: Left,+right: Right,+name: String,left_storage: Storage.Buffer<Element>,right_storage: Storage.Buffer<Element>,
  left_view: Maybe<&2,Views.View>,right_view: Maybe<&2,Views.View>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  left_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,left_storage)>,
  left_values: Dense.Values(~Element,~Left,~left_value,left,Buffer.contents(~Element,left_storage)),
  right_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,right_storage)>,
  right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.contents(~Element,right_storage)),
  accepted: {Dense.tensor_ready(~Element,Tensors.combine_fresh_views_with(~Element,~kernel,name,left_view,right_view,Tensors.view_of(shape),Tensors.view_of(shape),Tensors.count_of(shape),left_storage,right_storage,shape,fill,workers)) == True{} : Bool}) ->
  Fresh(~Element,~Left,~Right,~left_value,~right_value,~operation,left,right,left_view,right_view,Tensors.combine_fresh_views_with(~Element,~kernel,name,left_view,right_view,Tensors.view_of(shape),Tensors.view_of(shape),Tensors.count_of(shape),left_storage,right_storage,shape,fill,workers)):
  fresh_observed(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left,right,name,left_storage,right_storage,left_view,right_view,shape,fill,workers,Buffer.observe(~Element,left_storage),
    left_ready,left_values,right_ready,right_values,accepted)

# The public combination of two Ready dense tensors. The statement follows
# the dispatch of the call: equal extents and broadcasting to the left
# extents combine in place; any other common extents first gather the left
# operand. Incompatible extents never return Ready.
def Stretched(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,in_place: Bool,+left: Left,+right: Right,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>) -> Data:
  match in_place:
    case True{}: Combined(~Element,~Left,~Right,~left_value,~right_value,~operation,left,right,Tensors.stretched(Tensors.view_of(right_shape),left_shape),tensor)
    case False{}: Fresh(~Element,~Left,~Right,~left_value,~right_value,~operation,left,right,Tensors.stretched(Tensors.view_of(left_shape),shape),Tensors.stretched(Tensors.view_of(right_shape),shape),tensor)

def Broadcast(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,shape: Maybe<&2,List<&2,U32>>,+left: Left,+right: Right,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>) -> Data:
  match shape:
    case Some{+shape}: Stretched(~Element,~Left,~Right,~left_value,~right_value,~operation,Shapes.same_extents(shape,left_shape),left,right,left_shape,right_shape,shape,tensor)
    case None{}: Empty

def Stated(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,same: Bool,+left: Left,+right: Right,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>) -> Data:
  match same:
    case True{}: Combined(~Element,~Left,~Right,~left_value,~right_value,~operation,left,right,Tensors.view_of(right_shape),tensor)
    case False{}: Broadcast(~Element,~Left,~Right,~left_value,~right_value,~operation,Shapes.broadcast(left_shape,right_shape),left,right,left_shape,right_shape,tensor)

def stretched(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),in_place: Bool,+left: Left,+right: Right,+name: String,left_storage: Storage.Buffer<Element>,right_storage: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  left_facts: GatherValues.Facts(~Element,~Left,~left_value,left,left_shape,Buffer.contents(~Element,left_storage)),
  right_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,right_storage)>,
  right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.contents(~Element,right_storage)),
  accepted: {Dense.tensor_ready(~Element,Tensors.combine_stretched_with(~Element,~kernel,name,in_place,left_storage,left_shape,fill,right_storage,right_shape,shape,workers)) == True{} : Bool}) ->
  Stretched(~Element,~Left,~Right,~left_value,~right_value,~operation,in_place,left,right,left_shape,right_shape,shape,Tensors.combine_stretched_with(~Element,~kernel,name,in_place,left_storage,left_shape,fill,right_storage,right_shape,shape,workers)):
  match in_place:
    case True{}:
      combine_views(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,Tensors.view_of(left_shape),Tensors.stretched(Tensors.view_of(right_shape),left_shape),left,right,name,left_storage,right_storage,left_shape,right_shape,fill,workers,{==},left_facts,right_ready,right_values,accepted)
    case False{}:
      Shape.Both{Dense.WellFormed{length_exact,left_ready},left_values} = left_facts
      fresh(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left,right,name,left_storage,right_storage,Tensors.stretched(Tensors.view_of(left_shape),shape),Tensors.stretched(Tensors.view_of(right_shape),shape),shape,fill,workers,left_ready,left_values,right_ready,right_values,accepted)

def broadcast(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),shape: Maybe<&2,List<&2,U32>>,+left: Left,+right: Right,+name: String,left_storage: Storage.Buffer<Element>,right_storage: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+fill: Element,+workers: U32,
  left_facts: GatherValues.Facts(~Element,~Left,~left_value,left,left_shape,Buffer.contents(~Element,left_storage)),
  right_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,right_storage)>,
  right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.contents(~Element,right_storage)),
  accepted: {Dense.tensor_ready(~Element,Tensors.combine_broadcast_with(~Element,~kernel,name,shape,left_storage,left_shape,fill,right_storage,right_shape,workers)) == True{} : Bool}) ->
  Broadcast(~Element,~Left,~Right,~left_value,~right_value,~operation,shape,left,right,left_shape,right_shape,Tensors.combine_broadcast_with(~Element,~kernel,name,shape,left_storage,left_shape,fill,right_storage,right_shape,workers)):
  match shape:
    case Some{+shape}:
      stretched(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,Shapes.same_extents(shape,left_shape),left,right,name,left_storage,right_storage,left_shape,right_shape,shape,fill,workers,left_facts,right_ready,right_values,accepted)
    case None{}: WordEncoding.false_true(accepted)

def shapes(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),same: Bool,+left: Left,+right: Right,+name: String,left_storage: Storage.Buffer<Element>,right_storage: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+fill: Element,+workers: U32,
  left_facts: GatherValues.Facts(~Element,~Left,~left_value,left,left_shape,Buffer.contents(~Element,left_storage)),
  right_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,right_storage)>,
  right_values: Dense.Values(~Element,~Right,~right_value,right,Buffer.contents(~Element,right_storage)),
  accepted: {Dense.tensor_ready(~Element,Tensors.combine_shapes_with(~Element,~kernel,name,same,left_storage,left_shape,fill,right_storage,right_shape,workers)) == True{} : Bool}) ->
  Stated(~Element,~Left,~Right,~left_value,~right_value,~operation,same,left,right,left_shape,right_shape,Tensors.combine_shapes_with(~Element,~kernel,name,same,left_storage,left_shape,fill,right_storage,right_shape,workers)):
  match same:
    case True{}:
      combine_views(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,Tensors.view_of(left_shape),Tensors.view_of(right_shape),left,right,name,left_storage,right_storage,left_shape,right_shape,fill,workers,{==},left_facts,right_ready,right_values,accepted)
    case False{}:
      broadcast(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,Shapes.broadcast(left_shape,right_shape),left,right,name,left_storage,right_storage,left_shape,right_shape,fill,workers,left_facts,right_ready,right_values,accepted)

# Two Ready dense tensors with complete values, combined by the public call:
# when the call returns Ready, the result is well formed and holds, at every
# index, the operation applied to the two operands read through the views of
# the dispatch. The Ready result is the only premise about the call.
def combine(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+left: Left,+right: Right,+name: String,+operations: F32,left_storage: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+left_fill: Element,+left_workers: U32,
  right_storage: Storage.Buffer<Element>,+right_shape: List<&2,U32>,+right_fill: Element,+right_workers: U32,
  left_contract: Dense.Contract(~Element,~Left,~left_value,left,Tensors.Tensor{left_storage,left_shape,left_fill,Tensors.Usable{},left_workers}),
  right_contract: Dense.Contract(~Element,~Right,~right_value,right,Tensors.Tensor{right_storage,right_shape,right_fill,Tensors.Usable{},right_workers}),
  accepted: {Dense.tensor_ready(~Element,Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Tensor{left_storage,left_shape,left_fill,Tensors.Usable{},left_workers},Tensors.Tensor{right_storage,right_shape,right_fill,Tensors.Usable{},right_workers})) == True{} : Bool}) ->
  Stated(~Element,~Left,~Right,~left_value,~right_value,~operation,Shapes.same_extents(left_shape,right_shape),left,right,left_shape,right_shape,Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Tensor{left_storage,left_shape,left_fill,Tensors.Usable{},left_workers},Tensors.Tensor{right_storage,right_shape,right_fill,Tensors.Usable{},right_workers})):
  Shape.Both{Shape.Both{left_well_formed,left_status},left_values} = left_contract
  Shape.Both{Shape.Both{Dense.WellFormed{right_length,right_ready},right_status},right_values} = right_contract
  shapes(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,Shapes.same_extents(left_shape,right_shape),left,right,name,left_storage,right_storage,left_shape,right_shape,left_fill,U32.max(left_workers,right_workers),
    Shape.Both{left_well_formed,left_values},right_ready,right_values,accepted)
