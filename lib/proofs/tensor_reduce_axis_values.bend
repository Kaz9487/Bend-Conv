# The values of a reduction along one axis. The output starts at the initial
# value and takes the slices of the axis in increasing index, each by one
# checked in-place combination; reduced is that fold written in the order of
# the loop, so every output element is
# operation(... operation(operation(initial, x[0]), x[1]) ..., x[extent - 1]).
import Base
import ../tensor.bend as Tensors
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ../tensor_layout.bend as Layout
import ../tensor_operations.bend as Operations
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./tensor_dense_contract.bend as Dense
import ./tensor_copy_values.bend as CopyValues
import ./tensor_gather_values.bend as GatherValues
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_combine_dense_values.bend as CombineDense
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_view_metadata_refinement.bend as Metadata
import ./tensor_band_combine_leaf.bend as BandCombineLeaf
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./storage_tree_model.bend as Tree
import ./storage_readback.bend as StorageReadback
import ./storage_allocation_refinement.bend as AllocationLaws
import ./tensor_count_refinement.bend as Count
import ./tensor_allocation_depth.bend as Depth
import ./tensor_band_count.bend as BandCount
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding

# The logical input read through one slice of the axis.
def slice_value(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,slice: Maybe<&2,Views.View>,logical: Logical,position: U32) -> Element:
  match slice:
    case Some{slice}: original(logical,CopyValues.address(slice,U32.to_nat(position)))
    case None{}: original(logical,position)

def reduced(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,remaining: Nat,+index: U32,+axis: U32,+shape: List<&2,U32>,+logical: Logical,accumulator: Element,+position: U32) -> Element:
  match remaining:
    case 0n: accumulator
    case 1n+rest:
      reduced(~Element,~Logical,~original,~operation,rest,(index + 1 : U32),axis,shape,logical,
        operation(accumulator,slice_value(~Element,~Logical,~original,Layout.sliced(Tensors.view_of(shape),axis,index,1,1),logical,position)),position)

# The state of the fold: (output, input) and whether every step was accepted.
def content_of(~Element: Data,state: (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool) -> Buffer.Content<Element>:
  (owners,valid) = state
  (output,input) = owners
  Buffer.contents(~Element,output)

def valid_of(~Element: Data,state: (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool) -> Bool:
  (owners,valid) = state
  valid

def folded_ready(~Element: Data,state: (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool,+name: String,+shape: List<&2,U32>,+result_shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  {Dense.tensor_ready(~Element,Layout.folded(~Element,name,shape,result_shape,fill,workers,state)) == valid_of(~Element,state) : Bool}:
  (owners,valid) = state
  (output,input) = owners
  match valid:
    case True{}: {==}
    case False{}: {==}

def folded_shaped(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,state: (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool,+context: Context,+name: String,+shape: List<&2,U32>,+unit: List<&2,U32>,
  +result_shape: List<&2,U32>,+fill: Element,+workers: U32,
  facts: GatherValues.Facts(~Element,~Context,~reader,context,unit,content_of(~Element,state)),
  same_elements: {ShapeLaws.elements(unit) == ShapeLaws.elements(result_shape) : Nat},
  accepted: {valid_of(~Element,state) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Context,reader,context,result_shape,Layout.folded(~Element,name,shape,result_shape,fill,workers,state)>:
  (owners,valid) = state
  (output,input) = owners
  match valid:
    case True{}:
      Shape.Both{Dense.WellFormed{length_exact,ready},values} = facts
      GatherValues.shaped_observed(~Element,~Context,~reader,output,context,result_shape,fill,workers,Buffer.observe(~Element,output),
        Shape.Both{Dense.WellFormed{
          Equal.trans(Nat,Buffer.content_length(~Element,Buffer.contents(~Element,output)),ShapeLaws.elements(unit),ShapeLaws.elements(result_shape),length_exact,same_elements),ready},values})
    case False{}:
      Empty.absurd(GatherValues.Shaped<Element,Context,reader,context,result_shape,Layout.folded(~Element,name,shape,result_shape,fill,workers,((output,input),False{}))>,WordEncoding.false_true(accepted))

# A rejected step is never repaired: the fold stays invalid.
def sticky_outcome(~Element: Data,~operation: Element -> Element -> Element,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,+rest: Nat,+index: U32,+axis: U32,+shape: List<&2,U32>,+unit: List<&2,U32>,
  next: @left: Storage.Buffer<Element> -> @right: Storage.Buffer<Element> ->
    {valid_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,((left,right),False{}))) == False{} : Bool}) ->
  {valid_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.settle(~Element,False{},outcome))) == False{} : Bool}:
  match outcome:
    case Operations.Accepted{(left,right)}: next(left,right)
    case Operations.Rejected{(left,right)}: next(left,right)

def sticky_slice(~Element: Data,~operation: Element -> Element -> Element,slice: Maybe<&2,Views.View>,target: Maybe<&2,Views.View>,+rest: Nat,+index: U32,+axis: U32,+shape: List<&2,U32>,+unit: List<&2,U32>,
  output: Storage.Buffer<Element>,input: Storage.Buffer<Element>,
  next: @left: Storage.Buffer<Element> -> @right: Storage.Buffer<Element> ->
    {valid_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,((left,right),False{}))) == False{} : Bool}) ->
  {valid_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.fold_slice(~Element,~operation,False{},slice,target,output,input))) == False{} : Bool}:
  match slice target:
    case Some{slice} Some{target}:
      sticky_outcome(~Element,~operation,Operations.combine(~Element,~operation,output,target,input,slice),rest,index,axis,shape,unit,next)
    case Some{slice} None{}: next(output,input)
    case None{} Some{target}: next(output,input)
    case None{} None{}: next(output,input)

def sticky(~Element: Data,~operation: Element -> Element -> Element,remaining: Nat,+index: U32,+axis: U32,+shape: List<&2,U32>,+unit: List<&2,U32>,output: Storage.Buffer<Element>,input: Storage.Buffer<Element>) ->
  {valid_of(~Element,Layout.fold_slices(~Element,~operation,remaining,index,axis,shape,unit,((output,input),False{}))) == False{} : Bool}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      sticky_slice(~Element,~operation,Layout.sliced(Tensors.view_of(shape),axis,index,1,1),Tensors.view_of(unit),rest,index,axis,shape,unit,output,input,
        left => right => sticky(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,left,right))

# The output of the fold under any reader of its starting content.
type Fold<-Before: Data,-Logical: Data> is Data:
  Fold{before: Before,logical: Logical,remaining: Nat,index: U32,axis: U32,shape: List<&2,U32>}

def value(~Element: Data,~Before: Data,~Logical: Data,~before_value: Before -> U32 -> Element,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,context: Fold<Before,Logical>,position: U32) -> Element:
  Fold{before,logical,remaining,index,axis,shape} = context
  +position = position
  reduced(~Element,~Logical,~original,~operation,remaining,index,axis,shape,logical,before_value(before,position),position)

# One accepted step leaves the combined values in the output and the input
# unchanged; the remaining steps continue from that reader.
def advanced(~Element: Data,~Before: Data,~Logical: Data,~before_value: Before -> U32 -> Element,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,left: Storage.Buffer<Element>,right: Storage.Buffer<Element>,+slice: Views.View,+rest: Nat,+index: U32,+axis: U32,+shape: List<&2,U32>,+unit: List<&2,U32>,+before: Before,+logical: Logical,
  -right_content: Buffer.Content<Element>,
  step: CombineDense.Step(~Element,~Before,~Logical,~before_value,~original,~operation,before,logical,slice,unit,OperationsLaws.observe_pair_outcome(~Element,Operations.Accepted{(left,right)}),right_content),
  right_ready: Buffer.ContentReady<Element,right_content>,right_values: Dense.Values(~Element,~Logical,~original,logical,right_content),
  next: @left: Storage.Buffer<Element> -> @right: Storage.Buffer<Element> -> @+slice: Views.View ->
    GatherValues.Facts(~Element,~CombineDense.Combine<Before,Logical>,~CombineDense.value(~Element,~Before,~Logical,~before_value,~original,~operation),CombineDense.Combine{before,logical,slice},unit,Buffer.contents(~Element,left)) ->
    Buffer.ContentReady<Element,Buffer.contents(~Element,right)> -> Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,right)) ->
    {valid_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,((left,right),True{}))) == True{} : Bool} ->
    GatherValues.Facts(~Element,~Fold<CombineDense.Combine<Before,Logical>,Logical>,~value(~Element,~CombineDense.Combine<Before,Logical>,~Logical,~CombineDense.value(~Element,~Before,~Logical,~before_value,~original,~operation),~original,~operation),Fold{CombineDense.Combine{before,logical,slice},logical,rest,(index + 1 : U32),axis,shape},unit,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,((left,right),True{})))),
  slices: {Layout.sliced(Tensors.view_of(shape),axis,index,1,1) == Some{slice} : Maybe<&2,Views.View>},
  accepted: {valid_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,((left,right),True{}))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Fold<Before,Logical>,~value(~Element,~Before,~Logical,~before_value,~original,~operation),Fold{before,logical,1n+rest,index,axis,shape},unit,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,((left,right),True{})))):
  Storage.Buffer{left_storage,left_length,left_certificate} = left
  Storage.Buffer{right_storage,right_length,right_certificate} = right
  Shape.Both{combined,+unchanged} = step
  %Equal.sym(Maybe<&2,Views.View>,Layout.sliced(Tensors.view_of(shape),axis,index,1,1),Some{slice},slices) :
    Shape.Both<Dense.WellFormed<Element,unit,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,((Storage.Buffer{left_storage,left_length,left_certificate},Storage.Buffer{right_storage,right_length,right_certificate}),True{})))>,
      Members.All(Nat,{position => {Buffer.value(~Element,U32.from_nat(position),content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,((Storage.Buffer{left_storage,left_length,left_certificate},Storage.Buffer{right_storage,right_length,right_certificate}),True{})))) ==
        reduced(~Element,~Logical,~original,~operation,rest,(index + 1 : U32),axis,shape,logical,
          operation(before_value(before,U32.from_nat(position)),slice_value(~Element,~Logical,~original,_,logical,U32.from_nat(position))),U32.from_nat(position)) : Element} : Nat -> Data},
        Ranges.range(Buffer.content_length(~Element,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,((Storage.Buffer{left_storage,left_length,left_certificate},Storage.Buffer{right_storage,right_length,right_certificate}),True{})))),0n))>
  next(Storage.Buffer{left_storage,left_length,left_certificate},Storage.Buffer{right_storage,right_length,right_certificate},slice,combined,
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Storage.Buffer{right_storage,right_length,right_certificate}),right_content,unchanged) : Buffer.ContentReady<Element,_> right_ready,
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Storage.Buffer{right_storage,right_length,right_certificate}),right_content,unchanged) : Dense.Values(~Element,~Logical,~original,logical,_) right_values,
    accepted)

def step_outcome(~Element: Data,~Before: Data,~Logical: Data,~before_value: Before -> U32 -> Element,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,+slice: Views.View,+rest: Nat,+index: U32,+axis: U32,+shape: List<&2,U32>,+unit: List<&2,U32>,+before: Before,+logical: Logical,-right_content: Buffer.Content<Element>,
  step: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome)) == True{} : Bool} -> CombineDense.Step(~Element,~Before,~Logical,~before_value,~original,~operation,before,logical,slice,unit,OperationsLaws.observe_pair_outcome(~Element,outcome),right_content),
  right_ready: Buffer.ContentReady<Element,right_content>,right_values: Dense.Values(~Element,~Logical,~original,logical,right_content),
  next: @left: Storage.Buffer<Element> -> @right: Storage.Buffer<Element> -> @+slice: Views.View ->
    GatherValues.Facts(~Element,~CombineDense.Combine<Before,Logical>,~CombineDense.value(~Element,~Before,~Logical,~before_value,~original,~operation),CombineDense.Combine{before,logical,slice},unit,Buffer.contents(~Element,left)) ->
    Buffer.ContentReady<Element,Buffer.contents(~Element,right)> -> Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,right)) ->
    {valid_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,((left,right),True{}))) == True{} : Bool} ->
    GatherValues.Facts(~Element,~Fold<CombineDense.Combine<Before,Logical>,Logical>,~value(~Element,~CombineDense.Combine<Before,Logical>,~Logical,~CombineDense.value(~Element,~Before,~Logical,~before_value,~original,~operation),~original,~operation),Fold{CombineDense.Combine{before,logical,slice},logical,rest,(index + 1 : U32),axis,shape},unit,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,((left,right),True{})))),
  slices: {Layout.sliced(Tensors.view_of(shape),axis,index,1,1) == Some{slice} : Maybe<&2,Views.View>},
  accepted: {valid_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.settle(~Element,True{},outcome))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Fold<Before,Logical>,~value(~Element,~Before,~Logical,~before_value,~original,~operation),Fold{before,logical,1n+rest,index,axis,shape},unit,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.settle(~Element,True{},outcome)))):
  match outcome:
    case Operations.Accepted{(left,right)}:
      advanced(~Element,~Before,~Logical,~before_value,~original,~operation,left,right,slice,rest,index,axis,shape,unit,before,logical,right_content,step({==}),right_ready,right_values,next,slices,accepted)
    case Operations.Rejected{(left,right)}:
      Empty.absurd(GatherValues.Facts(~Element,~Fold<Before,Logical>,~value(~Element,~Before,~Logical,~before_value,~original,~operation),Fold{before,logical,1n+rest,index,axis,shape},unit,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.settle(~Element,True{},Operations.Rejected{(left,right)})))),WordEncoding.false_true(
        %sticky(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,left,right) : {_ == True{} : Bool}
        accepted))

# The two owners of a step are observed as models; the combination of the
# models supplies the step facts.
def step_observed(~Element: Data,~Before: Data,~Logical: Data,~before_value: Before -> U32 -> Element,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+target_axes: List<&2,Views.Axis>,+slice: Views.View,+rest: Nat,+index: U32,+axis: U32,+shape: List<&2,U32>,+unit: List<&2,U32>,+before: Before,+logical: Logical,-output: Storage.Buffer<Element>,-input: Storage.Buffer<Element>,
  output_observation: Buffer.Observation<Element,output>,input_observation: Buffer.Observation<Element,input>,
  slices: {Layout.sliced(Tensors.view_of(shape),axis,index,1,1) == Some{slice} : Maybe<&2,Views.View>},
  target_extents: {Views.axis_extents(target_axes) == unit : List<&2,U32>},
  output_facts: GatherValues.Facts(~Element,~Before,~before_value,before,unit,Buffer.contents(~Element,output)),
  input_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,input)>,
  input_values: Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,input)),
  next: @left: Storage.Buffer<Element> -> @right: Storage.Buffer<Element> -> @+slice: Views.View ->
    GatherValues.Facts(~Element,~CombineDense.Combine<Before,Logical>,~CombineDense.value(~Element,~Before,~Logical,~before_value,~original,~operation),CombineDense.Combine{before,logical,slice},unit,Buffer.contents(~Element,left)) ->
    Buffer.ContentReady<Element,Buffer.contents(~Element,right)> -> Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,right)) ->
    {valid_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,((left,right),True{}))) == True{} : Bool} ->
    GatherValues.Facts(~Element,~Fold<CombineDense.Combine<Before,Logical>,Logical>,~value(~Element,~CombineDense.Combine<Before,Logical>,~Logical,~CombineDense.value(~Element,~Before,~Logical,~before_value,~original,~operation),~original,~operation),Fold{CombineDense.Combine{before,logical,slice},logical,rest,(index + 1 : U32),axis,shape},unit,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,((left,right),True{})))),
  accepted: {valid_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.settle(~Element,True{},Operations.combine(~Element,~operation,output,Views.View{target_axes,0},input,slice)))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Fold<Before,Logical>,~value(~Element,~Before,~Logical,~before_value,~original,~operation),Fold{before,logical,1n+rest,index,axis,shape},unit,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.settle(~Element,True{},Operations.combine(~Element,~operation,output,Views.View{target_axes,0},input,slice))))):
  Buffer.Observation{+left_model,+left_equation} = output_observation
  Buffer.Observation{+right_model,+right_equation} = input_observation
  Shape.Both{Dense.WellFormed{length_exact,content_ready},left_values} = output_facts
  +length_model = {
    %Buffer.content_length_of(~Element,left_model) : {_ == ShapeLaws.elements(unit) : Nat}
    %Buffer.packed_contents(~Element,left_model) : {Buffer.content_length(~Element,_) == ShapeLaws.elements(unit) : Nat}
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,left_model),output,left_equation) : {Buffer.content_length(~Element,Buffer.contents(~Element,_)) == ShapeLaws.elements(unit) : Nat}
    length_exact : {Buffer.length(~Element,left_model) == ShapeLaws.elements(unit) : Nat}}
  +left_ready = Buffer.ready_content(~Element,left_model,
    %Buffer.packed_contents(~Element,left_model) : Buffer.ContentReady<Element,_>
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,left_model),output,left_equation) : Buffer.ContentReady<Element,Buffer.contents(~Element,_)>
    content_ready)
  +right_ready = Buffer.ready_content(~Element,right_model,
    %Buffer.packed_contents(~Element,right_model) : Buffer.ContentReady<Element,_>
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,right_model),input,right_equation) : Buffer.ContentReady<Element,Buffer.contents(~Element,_)>
    input_ready)
  +left_model_values = {
    %Buffer.packed_contents(~Element,left_model) : Dense.Values(~Element,~Before,~before_value,before,_)
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,left_model),output,left_equation) : Dense.Values(~Element,~Before,~before_value,before,Buffer.contents(~Element,_))
    left_values : Dense.Values(~Element,~Before,~before_value,before,Buffer.content(~Element,left_model))}
  +right_model_values = {
    %Buffer.packed_contents(~Element,right_model) : Dense.Values(~Element,~Logical,~original,logical,_)
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,right_model),input,right_equation) : Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,_))
    input_values : Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,right_model))}
  +target_exact = Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(target_axes)),ShapeLaws.elements(unit),Buffer.length(~Element,left_model),
    Equal.cong(List<&2,U32>,Nat,ShapeLaws.elements,Views.axis_extents(target_axes),unit,target_extents),
    Equal.sym(Nat,Buffer.length(~Element,left_model),ShapeLaws.elements(unit),length_model))
  %left_equation : GatherValues.Facts(~Element,~Fold<Before,Logical>,~value(~Element,~Before,~Logical,~before_value,~original,~operation),Fold{before,logical,1n+rest,index,axis,shape},unit,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.settle(~Element,True{},Operations.combine(~Element,~operation,_,Views.View{target_axes,0},input,slice)))))
  %right_equation : GatherValues.Facts(~Element,~Fold<Before,Logical>,~value(~Element,~Before,~Logical,~before_value,~original,~operation),Fold{before,logical,1n+rest,index,axis,shape},unit,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.settle(~Element,True{},Operations.combine(~Element,~operation,Buffer.pack(~Element,left_model),Views.View{target_axes,0},_,slice)))))
  step_outcome(~Element,~Before,~Logical,~before_value,~original,~operation,Operations.combine(~Element,~operation,Buffer.pack(~Element,left_model),Views.View{target_axes,0},Buffer.pack(~Element,right_model),slice),slice,rest,index,axis,shape,unit,before,logical,Buffer.content(~Element,right_model),
    combined => CombineDense.modeled_facts(~Element,~Before,~Logical,~before_value,~original,~operation,~Operations.combine_kernel(~Element,~operation),~OperationsLaws.scalar_kernel(~Element,~operation),before,logical,target_axes,slice,left_model,right_model,unit,
      left_ready,right_ready,left_model_values,right_model_values,target_exact,length_model,combined),
    Buffer.content_ready(~Element,right_model,right_ready),right_model_values,next,slices,
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,right_model),input,right_equation) : {valid_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.settle(~Element,True{},Operations.combine(~Element,~operation,Buffer.pack(~Element,left_model),Views.View{target_axes,0},_,slice)))) == True{} : Bool}
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,left_model),output,left_equation) : {valid_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.settle(~Element,True{},Operations.combine(~Element,~operation,_,Views.View{target_axes,0},input,slice)))) == True{} : Bool}
    accepted)

# The canonical view of the output starts at offset zero and has the extents unit.
def step_target(~Element: Data,~Before: Data,~Logical: Data,~before_value: Before -> U32 -> Element,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,target: Views.View,+slice: Views.View,+rest: Nat,+index: U32,+axis: U32,+shape: List<&2,U32>,+unit: List<&2,U32>,+before: Before,+logical: Logical,output: Storage.Buffer<Element>,input: Storage.Buffer<Element>,
  slices: {Layout.sliced(Tensors.view_of(shape),axis,index,1,1) == Some{slice} : Maybe<&2,Views.View>},
  +targets: {Tensors.view_of(unit) == Some{target} : Maybe<&2,Views.View>},
  output_facts: GatherValues.Facts(~Element,~Before,~before_value,before,unit,Buffer.contents(~Element,output)),
  input_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,input)>,
  input_values: Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,input)),
  next: @left: Storage.Buffer<Element> -> @right: Storage.Buffer<Element> -> @+slice: Views.View ->
    GatherValues.Facts(~Element,~CombineDense.Combine<Before,Logical>,~CombineDense.value(~Element,~Before,~Logical,~before_value,~original,~operation),CombineDense.Combine{before,logical,slice},unit,Buffer.contents(~Element,left)) ->
    Buffer.ContentReady<Element,Buffer.contents(~Element,right)> -> Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,right)) ->
    {valid_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,((left,right),True{}))) == True{} : Bool} ->
    GatherValues.Facts(~Element,~Fold<CombineDense.Combine<Before,Logical>,Logical>,~value(~Element,~CombineDense.Combine<Before,Logical>,~Logical,~CombineDense.value(~Element,~Before,~Logical,~before_value,~original,~operation),~original,~operation),Fold{CombineDense.Combine{before,logical,slice},logical,rest,(index + 1 : U32),axis,shape},unit,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,((left,right),True{})))),
  accepted: {valid_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.settle(~Element,True{},Operations.combine(~Element,~operation,output,target,input,slice)))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Fold<Before,Logical>,~value(~Element,~Before,~Logical,~before_value,~original,~operation),Fold{before,logical,1n+rest,index,axis,shape},unit,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.settle(~Element,True{},Operations.combine(~Element,~operation,output,target,input,slice))))):
  Views.View{+target_axes,+target_offset} = target
  +offset_zero = {
    %targets : {Metadata.observed_offset(_,0) == 0 : U32}
    Metadata.contiguous_count_offset(Shapes.numel(Shapes.Shape{unit}),unit,0) : {target_offset == 0 : U32}}
  +target_extents = {
    %targets : {Metadata.observed_extents(_,unit) == unit : List<&2,U32>}
    Metadata.contiguous_shape(unit,0) : {Views.axis_extents(target_axes) == unit : List<&2,U32>}}
  %Equal.sym(U32,target_offset,0,offset_zero) : GatherValues.Facts(~Element,~Fold<Before,Logical>,~value(~Element,~Before,~Logical,~before_value,~original,~operation),Fold{before,logical,1n+rest,index,axis,shape},unit,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.settle(~Element,True{},Operations.combine(~Element,~operation,output,Views.View{target_axes,_},input,slice)))))
  step_observed(~Element,~Before,~Logical,~before_value,~original,~operation,target_axes,slice,rest,index,axis,shape,unit,before,logical,output,input,Buffer.observe(~Element,output),Buffer.observe(~Element,input),slices,target_extents,
    output_facts,input_ready,input_values,next,
    %offset_zero : {valid_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.settle(~Element,True{},Operations.combine(~Element,~operation,output,Views.View{target_axes,_},input,slice)))) == True{} : Bool}
    accepted)

# A valid fold found the slice and the output view of every step.
def step_slice(~Element: Data,~Before: Data,~Logical: Data,~before_value: Before -> U32 -> Element,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,slice: Maybe<&2,Views.View>,target: Maybe<&2,Views.View>,+rest: Nat,+index: U32,+axis: U32,+shape: List<&2,U32>,+unit: List<&2,U32>,+before: Before,+logical: Logical,output: Storage.Buffer<Element>,input: Storage.Buffer<Element>,
  +slices: {Layout.sliced(Tensors.view_of(shape),axis,index,1,1) == slice : Maybe<&2,Views.View>},
  +targets: {Tensors.view_of(unit) == target : Maybe<&2,Views.View>},
  output_facts: GatherValues.Facts(~Element,~Before,~before_value,before,unit,Buffer.contents(~Element,output)),
  input_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,input)>,
  input_values: Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,input)),
  next: @left: Storage.Buffer<Element> -> @right: Storage.Buffer<Element> -> @+slice: Views.View ->
    GatherValues.Facts(~Element,~CombineDense.Combine<Before,Logical>,~CombineDense.value(~Element,~Before,~Logical,~before_value,~original,~operation),CombineDense.Combine{before,logical,slice},unit,Buffer.contents(~Element,left)) ->
    Buffer.ContentReady<Element,Buffer.contents(~Element,right)> -> Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,right)) ->
    {valid_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,((left,right),True{}))) == True{} : Bool} ->
    GatherValues.Facts(~Element,~Fold<CombineDense.Combine<Before,Logical>,Logical>,~value(~Element,~CombineDense.Combine<Before,Logical>,~Logical,~CombineDense.value(~Element,~Before,~Logical,~before_value,~original,~operation),~original,~operation),Fold{CombineDense.Combine{before,logical,slice},logical,rest,(index + 1 : U32),axis,shape},unit,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,((left,right),True{})))),
  accepted: {valid_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.fold_slice(~Element,~operation,True{},slice,target,output,input))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Fold<Before,Logical>,~value(~Element,~Before,~Logical,~before_value,~original,~operation),Fold{before,logical,1n+rest,index,axis,shape},unit,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.fold_slice(~Element,~operation,True{},slice,target,output,input)))):
  match slice target:
    case Some{+slice} Some{+target}:
      step_target(~Element,~Before,~Logical,~before_value,~original,~operation,target,slice,rest,index,axis,shape,unit,before,logical,output,input,slices,targets,output_facts,input_ready,input_values,next,accepted)
    case Some{slice} None{}:
      Empty.absurd(GatherValues.Facts(~Element,~Fold<Before,Logical>,~value(~Element,~Before,~Logical,~before_value,~original,~operation),Fold{before,logical,1n+rest,index,axis,shape},unit,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.fold_slice(~Element,~operation,True{},Some{slice},None{},output,input)))),WordEncoding.false_true(
        %sticky(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,output,input) : {_ == True{} : Bool}
        accepted))
    case None{} Some{target}:
      Empty.absurd(GatherValues.Facts(~Element,~Fold<Before,Logical>,~value(~Element,~Before,~Logical,~before_value,~original,~operation),Fold{before,logical,1n+rest,index,axis,shape},unit,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.fold_slice(~Element,~operation,True{},None{},Some{target},output,input)))),WordEncoding.false_true(
        %sticky(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,output,input) : {_ == True{} : Bool}
        accepted))
    case None{} None{}:
      Empty.absurd(GatherValues.Facts(~Element,~Fold<Before,Logical>,~value(~Element,~Before,~Logical,~before_value,~original,~operation),Fold{before,logical,1n+rest,index,axis,shape},unit,content_of(~Element,Layout.fold_slices(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,Layout.fold_slice(~Element,~operation,True{},None{},None{},output,input)))),WordEncoding.false_true(
        %sticky(~Element,~operation,rest,(index + 1 : U32),axis,shape,unit,output,input) : {_ == True{} : Bool}
        accepted))

# The fold of the remaining slices from index: when every step was accepted,
# the output holds reduced over its starting values.
def folds(~Element: Data,~Before: Data,~Logical: Data,~before_value: Before -> U32 -> Element,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,remaining: Nat,+index: U32,+axis: U32,+shape: List<&2,U32>,+unit: List<&2,U32>,+before: Before,+logical: Logical,output: Storage.Buffer<Element>,input: Storage.Buffer<Element>,
  output_facts: GatherValues.Facts(~Element,~Before,~before_value,before,unit,Buffer.contents(~Element,output)),
  input_ready: Buffer.ContentReady<Element,Buffer.contents(~Element,input)>,
  input_values: Dense.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,input)),
  accepted: {valid_of(~Element,Layout.fold_slices(~Element,~operation,remaining,index,axis,shape,unit,((output,input),True{}))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Fold<Before,Logical>,~value(~Element,~Before,~Logical,~before_value,~original,~operation),Fold{before,logical,remaining,index,axis,shape},unit,content_of(~Element,Layout.fold_slices(~Element,~operation,remaining,index,axis,shape,unit,((output,input),True{})))):
  match remaining:
    case 0n: output_facts
    case 1n+ +rest:
      step_slice(~Element,~Before,~Logical,~before_value,~original,~operation,Layout.sliced(Tensors.view_of(shape),axis,index,1,1),Tensors.view_of(unit),rest,index,axis,shape,unit,before,logical,output,input,{==},{==},output_facts,input_ready,input_values,
        left => right => slice => stepped => ready => values => continued =>
          folds(~Element,~CombineDense.Combine<Before,Logical>,~Logical,~CombineDense.value(~Element,~Before,~Logical,~before_value,~original,~operation),~original,~operation,rest,(index + 1 : U32),axis,shape,unit,CombineDense.Combine{before,logical,slice},logical,left,right,stepped,ready,values,continued),
        accepted)

# The starting output: every element is the initial value.
def constant(~Element: Data,initial: Element,position: U32) -> Element:
  initial

def fresh_cells(~Element: Data,remaining: Nat,+position: Nat,+depth: Nat,+initial: Element,+length: Nat) ->
  Members.All(Nat,{index => Dense.At(~Element,~Element,~constant(~Element),initial,Buffer.Content{Tree.fresh(Element,depth,initial),length},index) : Nat -> Data},Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{StorageReadback.read_fresh(~Element,depth,initial,U32.from_nat(position)),fresh_cells(~Element,rest,1n+position,depth,initial,length)}

def initial_counted(~Element: Data,+unit: List<&2,U32>,+initial: Element,+count: U32,evidence: Count.Count<ShapeLaws.elements(unit),count>) ->
  GatherValues.Facts(~Element,~Element,~constant(~Element),initial,unit,Buffer.contents(~Element,Tensors.buffer(~Element,initial,count))):
  Count.Count{+exact,+bounded} = evidence
  +fresh = {Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),initial),U32.to_nat(count)} : Buffer.Content<Element>}
  +allocated = AllocationLaws.contents(~Element,Tensors.depth_for(count),initial,U32.to_nat(count))
  Shape.Both{
    Dense.WellFormed{
      Equal.trans(Nat,Buffer.content_length(~Element,Buffer.contents(~Element,Tensors.buffer(~Element,initial,count))),U32.to_nat(count),ShapeLaws.elements(unit),
        Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.contents(~Element,Tensors.buffer(~Element,initial,count)),fresh,allocated),
        Equal.sym(Nat,ShapeLaws.elements(unit),U32.to_nat(count),exact)),
      Buffer.native_content_ready(~Element,Tensors.buffer(~Element,initial,count),BandCount.allocated_ready(~Element,count,initial,Depth.bounded(count,bounded)))},
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Tensors.buffer(~Element,initial,count)),fresh,allocated) : Dense.Values(~Element,~Element,~constant(~Element),initial,_)
    fresh_cells(~Element,U32.to_nat(count),0n,Tensors.depth_for(count),initial,U32.to_nat(count))}

# The output layout has extent 1 at the axis, so it holds as many elements as
# the result shape, which omits the axis.
def drop_zero(extents: List<&2,U32>) -> {List.drop(&2,U32,extents,0n) == extents : List<&2,U32>}:
  match extents:
    case []: {==}
    case extent <> tail: {==}

def unit_elements(shape: List<&2,U32>,axis: Nat,inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {ShapeLaws.elements(Layout.with_extent(shape,axis,1)) == ShapeLaws.elements(Shapes.without_axis(shape,axis)) : Nat}:
  match shape axis:
    case [] 0n: Empty.absurd({ShapeLaws.elements(Layout.with_extent([],0n,1)) == ShapeLaws.elements(Shapes.without_axis([],0n)) : Nat},WordEncoding.false_true(inside))
    case [] 1n+rest: Empty.absurd({ShapeLaws.elements(Layout.with_extent([],1n+rest,1)) == ShapeLaws.elements(Shapes.without_axis([],1n+rest)) : Nat},WordEncoding.false_true(inside))
    case +extent <> +tail 0n:
      %Equal.sym(List<&2,U32>,List.drop(&2,U32,tail,0n),tail,drop_zero(tail)) : {Nat.add(ShapeLaws.elements(tail),0n) == ShapeLaws.elements(_) : Nat}
      Addition.zero_right(ShapeLaws.elements(tail))
    case +extent <> +tail 1n+ +rest:
      Equal.cong(Nat,Nat,elements => Nat.mul(U32.to_nat(extent),elements),ShapeLaws.elements(Layout.with_extent(tail,rest,1)),ShapeLaws.elements(Shapes.without_axis(tail,rest)),
        unit_elements(tail,rest,inside))

# The fold over a modeled input, from the starting output to the result tensor.
def modeled(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+name: String,+initial: Element,+axis: U32,+model: Buffer.Model<Element>,+shape: List<&2,U32>,+unit: List<&2,U32>,+fill: Element,+workers: U32,+count: U32,
  +counts: {Tensors.count_of(unit) == Some{count} : Maybe<&2,U32>},
  facts: GatherValues.Facts(~Element,~Logical,~original,logical,shape,Buffer.content(~Element,model)),
  same_elements: {ShapeLaws.elements(unit) == ShapeLaws.elements(Shapes.without_axis(shape,U32.to_nat(axis))) : Nat},
  accepted: {Dense.tensor_ready(~Element,Layout.folded(~Element,name,shape,Shapes.without_axis(shape,U32.to_nat(axis)),fill,workers,Layout.fold_slices(~Element,~operation,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape,unit,((Tensors.buffer(~Element,initial,count),Buffer.pack(~Element,model)),True{})))) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Fold<Element,Logical>,value(~Element,~Element,~Logical,~constant(~Element),~original,~operation),Fold{initial,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},Shapes.without_axis(shape,U32.to_nat(axis)),Layout.folded(~Element,name,shape,Shapes.without_axis(shape,U32.to_nat(axis)),fill,workers,Layout.fold_slices(~Element,~operation,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape,unit,((Tensors.buffer(~Element,initial,count),Buffer.pack(~Element,model)),True{})))>:
  Shape.Both{Dense.WellFormed{length_exact,content_ready},source_values} = facts
  +valid = Equal.trans(Bool,valid_of(~Element,Layout.fold_slices(~Element,~operation,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape,unit,((Tensors.buffer(~Element,initial,count),Buffer.pack(~Element,model)),True{}))),Dense.tensor_ready(~Element,Layout.folded(~Element,name,shape,Shapes.without_axis(shape,U32.to_nat(axis)),fill,workers,Layout.fold_slices(~Element,~operation,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape,unit,((Tensors.buffer(~Element,initial,count),Buffer.pack(~Element,model)),True{})))),True{},
    Equal.sym(Bool,Dense.tensor_ready(~Element,Layout.folded(~Element,name,shape,Shapes.without_axis(shape,U32.to_nat(axis)),fill,workers,Layout.fold_slices(~Element,~operation,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape,unit,((Tensors.buffer(~Element,initial,count),Buffer.pack(~Element,model)),True{})))),valid_of(~Element,Layout.fold_slices(~Element,~operation,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape,unit,((Tensors.buffer(~Element,initial,count),Buffer.pack(~Element,model)),True{}))),
      folded_ready(~Element,Layout.fold_slices(~Element,~operation,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape,unit,((Tensors.buffer(~Element,initial,count),Buffer.pack(~Element,model)),True{})),name,shape,Shapes.without_axis(shape,U32.to_nat(axis)),fill,workers)),accepted)
  folded_shaped(~Element,~Fold<Element,Logical>,~value(~Element,~Element,~Logical,~constant(~Element),~original,~operation),Layout.fold_slices(~Element,~operation,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape,unit,((Tensors.buffer(~Element,initial,count),Buffer.pack(~Element,model)),True{})),Fold{initial,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},name,shape,unit,Shapes.without_axis(shape,U32.to_nat(axis)),fill,workers,
    folds(~Element,~Element,~Logical,~constant(~Element),~original,~operation,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape,unit,initial,logical,Tensors.buffer(~Element,initial,count),Buffer.pack(~Element,model),
      initial_counted(~Element,unit,initial,count,Count.count_of(unit,count,counts)),
      %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model)) : Buffer.ContentReady<Element,_>
      content_ready,
      %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model)) : Dense.Values(~Element,~Logical,~original,logical,_)
      source_values,
      valid),
    same_elements,valid)

def counted(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,count: Maybe<&2,U32>,+logical: Logical,+name: String,+initial: Element,+axis: U32,+model: Buffer.Model<Element>,+shape: List<&2,U32>,+unit: List<&2,U32>,+fill: Element,+workers: U32,
  +counts: {Tensors.count_of(unit) == count : Maybe<&2,U32>},
  facts: GatherValues.Facts(~Element,~Logical,~original,logical,shape,Buffer.content(~Element,model)),
  same_elements: {ShapeLaws.elements(unit) == ShapeLaws.elements(Shapes.without_axis(shape,U32.to_nat(axis))) : Nat},
  accepted: {Dense.tensor_ready(~Element,Layout.reduce_axis_count(~Element,~operation,name,count,axis,Buffer.pack(~Element,model),shape,unit,initial,fill,workers)) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Fold<Element,Logical>,value(~Element,~Element,~Logical,~constant(~Element),~original,~operation),Fold{initial,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},Shapes.without_axis(shape,U32.to_nat(axis)),Layout.reduce_axis_count(~Element,~operation,name,count,axis,Buffer.pack(~Element,model),shape,unit,initial,fill,workers)>:
  match count:
    case Some{+count}:
      modeled(~Element,~Logical,~original,~operation,logical,name,initial,axis,model,shape,unit,fill,workers,count,counts,facts,same_elements,accepted)
    case None{}: Empty.absurd(GatherValues.Shaped<Element,Fold<Element,Logical>,value(~Element,~Element,~Logical,~constant(~Element),~original,~operation),Fold{initial,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},Shapes.without_axis(shape,U32.to_nat(axis)),Layout.reduce_axis_count(~Element,~operation,name,None{},axis,Buffer.pack(~Element,model),shape,unit,initial,fill,workers)>,WordEncoding.false_true(accepted))

def status(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,valid: Bool,+logical: Logical,+name: String,+initial: Element,+axis: U32,+model: Buffer.Model<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +valids: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)) == valid : Bool},
  facts: GatherValues.Facts(~Element,~Logical,~original,logical,shape,Buffer.content(~Element,model)),
  accepted: {Dense.tensor_ready(~Element,Layout.reduce_axis_status(~Element,~operation,name,Tensors.Usable{},valid,axis,Buffer.pack(~Element,model),shape,initial,fill,workers)) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Fold<Element,Logical>,value(~Element,~Element,~Logical,~constant(~Element),~original,~operation),Fold{initial,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},Shapes.without_axis(shape,U32.to_nat(axis)),Layout.reduce_axis_status(~Element,~operation,name,Tensors.Usable{},valid,axis,Buffer.pack(~Element,model),shape,initial,fill,workers)>:
  match valid:
    case True{}:
      counted(~Element,~Logical,~original,~operation,Tensors.count_of(Layout.with_extent(shape,U32.to_nat(axis),1)),logical,name,initial,axis,model,shape,Layout.with_extent(shape,U32.to_nat(axis),1),fill,workers,{==},facts,
        unit_elements(shape,U32.to_nat(axis),valids),accepted)
    case False{}: Empty.absurd(GatherValues.Shaped<Element,Fold<Element,Logical>,value(~Element,~Element,~Logical,~constant(~Element),~original,~operation),Fold{initial,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},Shapes.without_axis(shape,U32.to_nat(axis)),Layout.reduce_axis_status(~Element,~operation,name,Tensors.Usable{},False{},axis,Buffer.pack(~Element,model),shape,initial,fill,workers)>,WordEncoding.false_true(accepted))

# The reduction of a shaped tensor: a Ready result is a shaped tensor in the
# shape without the axis, and its element at every position is reduced over
# the slices of the axis, from the initial value, in increasing index.
def shaped(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,-tensor: Tensors.Tensor<Element>,+logical: Logical,+name: String,+initial: Element,+axis: U32,+shape: List<&2,U32>,
  evidence: GatherValues.Shaped<Element,Logical,original,logical,shape,tensor>,
  accepted: {Dense.tensor_ready(~Element,Layout.reduce_axis(~Element,~operation,name,initial,tensor,axis)) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Fold<Element,Logical>,value(~Element,~Element,~Logical,~constant(~Element),~original,~operation),Fold{initial,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},Shapes.without_axis(shape,U32.to_nat(axis)),Layout.reduce_axis(~Element,~operation,name,initial,tensor,axis)>:
  GatherValues.Shaped{+model,+fill,+workers,+equation,facts} = evidence
  %Equal.sym(Tensors.Tensor<Element>,tensor,Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers},equation) : GatherValues.Shaped<Element,Fold<Element,Logical>,value(~Element,~Element,~Logical,~constant(~Element),~original,~operation),Fold{initial,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},Shapes.without_axis(shape,U32.to_nat(axis)),Layout.reduce_axis(~Element,~operation,name,initial,_,axis)>
  status(~Element,~Logical,~original,~operation,Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)),logical,name,initial,axis,model,shape,fill,workers,{==},facts,
    %equation : {Dense.tensor_ready(~Element,Layout.reduce_axis(~Element,~operation,name,initial,_,axis)) == True{} : Bool}
    accepted)

# A Ready dense tensor with complete values, reduced along one axis: a Ready
# result satisfies the dense contract of the same reader.
def reduce_axis(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+name: String,+initial: Element,+axis: U32,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Dense.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers}),
  accepted: {Dense.tensor_ready(~Element,Layout.reduce_axis(~Element,~operation,name,initial,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},axis)) == True{} : Bool}) ->
  Dense.Contract(~Element,~Fold<Element,Logical>,~value(~Element,~Element,~Logical,~constant(~Element),~original,~operation),Fold{initial,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},Layout.reduce_axis(~Element,~operation,name,initial,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},axis)):
  Shape.Both{Shape.Both{well_formed,ready},values} = contract
  GatherValues.shaped_contract(~Element,~Fold<Element,Logical>,~value(~Element,~Element,~Logical,~constant(~Element),~original,~operation),Fold{initial,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},Shapes.without_axis(shape,U32.to_nat(axis)),Layout.reduce_axis(~Element,~operation,name,initial,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},axis),
    shaped(~Element,~Logical,~original,~operation,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},logical,name,initial,axis,shape,
      GatherValues.shaped_observed(~Element,~Logical,~original,storage,logical,shape,fill,workers,Buffer.observe(~Element,storage),Shape.Both{well_formed,values}),accepted))
