# The values of set_slice and of every write through a view of dense storage.
# The part is stored by one checked scatter through a view of the target: a
# storage index holds the part element of the last coordinates of the view at
# that index, or the target element when the view does not reach it.
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
import ./tensor_cursor_refinement.bend as CursorLaws
import ./tensor_gather_values.bend as GatherValues
import ./tensor_scatter_values.bend as Scatter
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_view_metadata_refinement.bend as Metadata
import ./tensor_reduce_axis_values.bend as ReduceAxisValues
import ./tensor_count_refinement.bend as Count
import ./word_encoding.bend as WordEncoding

# The reader of a write through a view: the scatter of the part through the
# view over the target, with the count of the view; the target itself when
# there is no view.
type Placed<-Before: Data,-Part: Data> is Data:
  Placed{before: Before,part: Part,slice: Maybe<&2,Views.View>}

def count_of(view: Views.View) -> Nat:
  U32.to_nat(Layout.count_or_zero(Views.numel(view)))

# The element of the part that the view stores at a storage index, or
# previous.
def slice_element(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,slice: Maybe<&2,Views.View>,part: Part,previous: Element,position: U32) -> Element:
  match slice:
    case Some{+view}: Scatter.scattered(~Element,~Part,~part_value,count_of(view),0,CursorLaws.initial(view),part,previous,position)
    case None{}: previous

def placed_value(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,context: Placed<Before,Part>,position: U32) -> Element:
  Placed{before,part,slice} = context
  +position = position
  slice_element(~Element,~Part,~part_value,slice,part,before_value(before,position),position)

# The state of the write: (part, target) and whether the copy was accepted.
def target_content(~Element: Data,state: (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool) -> Buffer.Content<Element>:
  (owners,valid) = state
  (part,target) = owners
  Buffer.contents(~Element,target)

def settled_valid(~Element: Data,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) ->
  {ReduceAxisValues.valid_of(~Element,Layout.settle(~Element,True{},outcome)) == GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome)) : Bool}:
  match outcome:
    case Operations.Accepted{(input,output)}: {==}
    case Operations.Rejected{(input,output)}: {==}

def settled_content(~Element: Data,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) ->
  {target_content(~Element,Layout.settle(~Element,True{},outcome)) == CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome)) : Buffer.Content<Element>}:
  match outcome:
    case Operations.Accepted{(input,Storage.Buffer{storage,length,certificate})}: {==}
    case Operations.Rejected{(input,Storage.Buffer{storage,length,certificate})}: {==}

# The result tensor of the write.
def placed_ready(~Element: Data,state: (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool,+name: String,+target_shape: List<&2,U32>,+target_fill: Element,+part_shape: List<&2,U32>,+part_fill: Element,+workers: U32) ->
  {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.placed(~Element,name,target_shape,part_shape,target_fill,part_fill,workers,state))) == ReduceAxisValues.valid_of(~Element,state) : Bool}:
  (owners,valid) = state
  (part,target) = owners
  match valid:
    case True{}: {==}
    case False{}: {==}

def placed_shaped(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,state: (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool,+context: Context,+name: String,+target_shape: List<&2,U32>,+target_fill: Element,+part_shape: List<&2,U32>,+part_fill: Element,+workers: U32,
  facts: GatherValues.Facts(~Element,~Context,~reader,context,target_shape,target_content(~Element,state)),
  accepted: {ReduceAxisValues.valid_of(~Element,state) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Context,reader,context,target_shape,Layout.result_of(~Element,Layout.placed(~Element,name,target_shape,part_shape,target_fill,part_fill,workers,state))>:
  (owners,valid) = state
  (part,target) = owners
  match valid:
    case True{}:
      GatherValues.shaped_observed(~Element,~Context,~reader,target,context,target_shape,target_fill,workers,Buffer.observe(~Element,target),facts)
    case False{}:
      Empty.absurd(GatherValues.Shaped<Element,Context,reader,context,target_shape,Layout.result_of(~Element,Layout.placed(~Element,name,target_shape,part_shape,target_fill,part_fill,workers,((part,target),False{})))>,WordEncoding.false_true(accepted))

# An accepted scatter of a part with elements, through a view with a count.
def counted(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,count: Maybe<&2,U32>,+before: Before,+part: Part,+target_axes: List<&2,Views.Axis>,+target_offset: U32,+source_axes: List<&2,Views.Axis>,
  +target_model: Buffer.Model<Element>,+target_shape: List<&2,U32>,+part_model: Buffer.Model<Element>,+part_shape: List<&2,U32>,
  target_facts: GatherValues.Facts(~Element,~Before,~before_value,before,target_shape,Buffer.content(~Element,target_model)),
  part_facts: GatherValues.Facts(~Element,~Part,~part_value,part,part_shape,Buffer.content(~Element,part_model)),
  +source_extents: {Views.axis_extents(source_axes) == part_shape : List<&2,U32>},
  +counts: {Views.numel(Views.View{target_axes,target_offset}) == count : Maybe<&2,U32>},
  +accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset}))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Placed<Before,Part>,~placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,Some{Views.View{target_axes,target_offset}}},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset})))):
  match count:
    case Some{+count}:
      Buffer.Model{+target_tree,+target_length,+target_certificate} = target_model
      Buffer.Model{+part_tree,+part_length,+part_certificate} = part_model
      Shape.Both{Dense.WellFormed{+target_length_exact,target_ready},target_values} = target_facts
      Shape.Both{Dense.WellFormed{+part_length_exact,part_ready},part_values} = part_facts
      +target_model = {Buffer.Model{target_tree,target_length,target_certificate} : Buffer.Model<Element>}
      +part_model = {Buffer.Model{part_tree,part_length,part_certificate} : Buffer.Model<Element>}
      %Equal.sym(Maybe<&2,U32>,Views.numel(Views.View{target_axes,target_offset}),Some{count},counts) :
        GatherValues.Facts(~Element,~Scatter.Scatter<Before,Part>,~Scatter.value(~Element,~Before,~Part,~before_value,~part_value),Scatter.Scatter{before,part,Views.View{target_axes,target_offset},U32.to_nat(Layout.count_or_zero(_))},target_shape,
          CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset}))))
      Scatter.modeled_facts(~Element,~Before,~Part,~before_value,~part_value,before,part,target_axes,target_offset,part_model,target_model,count,source_axes,target_shape,
        Buffer.ready_content(~Element,target_model,target_ready),part_values,target_values,
        %Equal.sym(List<&2,U32>,Views.axis_extents(source_axes),part_shape,source_extents) : {ShapeLaws.elements(_) == part_length : Nat}
        Equal.sym(Nat,part_length,ShapeLaws.elements(part_shape),part_length_exact),
        %counts : ShapeLaws.Count(_,ShapeLaws.elements(Views.axis_extents(target_axes)))
        ShapeLaws.numel(Views.axis_extents(target_axes)),
        target_length_exact,accepted)
    case None{}:
      Empty.absurd(GatherValues.Facts(~Element,~Placed<Before,Part>,~placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,Some{Views.View{target_axes,target_offset}}},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset})))),
        WordEncoding.false_true(Equal.trans(Bool,False{},
          GatherValues.accepted_owners(~Element,OperationsLaws.scatter_denotation(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset})),True{},
          Equal.sym(Bool,GatherValues.accepted_owners(~Element,OperationsLaws.scatter_denotation(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset})),False{},
            Scatter.uncounted(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset},counts)),
          %OperationsLaws.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset}) : {GatherValues.accepted_owners(~Element,_) == True{} : Bool}
          accepted)))

# The write of a part with elements: one accepted scatter.
def written(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,slice: Views.View,source: Views.View,+before: Before,+part: Part,+name: String,
  +target_model: Buffer.Model<Element>,+target_shape: List<&2,U32>,+target_fill: Element,+part_model: Buffer.Model<Element>,+part_shape: List<&2,U32>,+part_fill: Element,+workers: U32,
  target_facts: GatherValues.Facts(~Element,~Before,~before_value,before,target_shape,Buffer.content(~Element,target_model)),
  part_facts: GatherValues.Facts(~Element,~Part,~part_value,part,part_shape,Buffer.content(~Element,part_model)),
  +sources: {Tensors.view_of(part_shape) == Some{source} : Maybe<&2,Views.View>},
  +accepted: {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.placed(~Element,name,target_shape,part_shape,target_fill,part_fill,workers,Layout.settle(~Element,True{},Operations.scatter(~Element,Buffer.pack(~Element,part_model),source,Buffer.pack(~Element,target_model),slice))))) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Placed<Before,Part>,placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,Some{slice}},target_shape,Layout.result_of(~Element,Layout.placed(~Element,name,target_shape,part_shape,target_fill,part_fill,workers,Layout.settle(~Element,True{},Operations.scatter(~Element,Buffer.pack(~Element,part_model),source,Buffer.pack(~Element,target_model),slice))))>:
  Views.View{+target_axes,+target_offset} = slice
  Views.View{+source_axes,+source_offset} = source
  +offset_zero = {
    %sources : {Metadata.observed_offset(_,0) == 0 : U32}
    Metadata.contiguous_count_offset(Shapes.numel(Shapes.Shape{part_shape}),part_shape,0) : {source_offset == 0 : U32}}
  +source_extents = {
    %sources : {Metadata.observed_extents(_,part_shape) == part_shape : List<&2,U32>}
    Metadata.contiguous_shape(part_shape,0) : {Views.axis_extents(source_axes) == part_shape : List<&2,U32>}}
  +settled = {
    %offset_zero : {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.placed(~Element,name,target_shape,part_shape,target_fill,part_fill,workers,Layout.settle(~Element,True{},Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,_},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset}))))) == True{} : Bool}
    accepted : {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.placed(~Element,name,target_shape,part_shape,target_fill,part_fill,workers,Layout.settle(~Element,True{},Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset}))))) == True{} : Bool}}
  +valid = Equal.trans(Bool,ReduceAxisValues.valid_of(~Element,Layout.settle(~Element,True{},Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset}))),Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.placed(~Element,name,target_shape,part_shape,target_fill,part_fill,workers,Layout.settle(~Element,True{},Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset}))))),True{},
    Equal.sym(Bool,Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.placed(~Element,name,target_shape,part_shape,target_fill,part_fill,workers,Layout.settle(~Element,True{},Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset}))))),ReduceAxisValues.valid_of(~Element,Layout.settle(~Element,True{},Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset}))),
      placed_ready(~Element,Layout.settle(~Element,True{},Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset})),name,target_shape,target_fill,part_shape,part_fill,workers)),settled)
  %Equal.sym(U32,source_offset,0,offset_zero) :
    GatherValues.Shaped<Element,Placed<Before,Part>,placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,Some{Views.View{target_axes,target_offset}}},target_shape,Layout.result_of(~Element,Layout.placed(~Element,name,target_shape,part_shape,target_fill,part_fill,workers,Layout.settle(~Element,True{},Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,_},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset}))))>
  placed_shaped(~Element,~Placed<Before,Part>,~placed_value(~Element,~Before,~Part,~before_value,~part_value),Layout.settle(~Element,True{},Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset})),Placed{before,part,Some{Views.View{target_axes,target_offset}}},name,target_shape,target_fill,part_shape,part_fill,workers,
    %Equal.sym(Buffer.Content<Element>,target_content(~Element,Layout.settle(~Element,True{},Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset}))),CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset}))),
      settled_content(~Element,Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset}))) : GatherValues.Facts(~Element,~Placed<Before,Part>,~placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,Some{Views.View{target_axes,target_offset}}},target_shape,_)
    counted(~Element,~Before,~Part,~before_value,~part_value,Views.numel(Views.View{target_axes,target_offset}),before,part,target_axes,target_offset,source_axes,target_model,target_shape,part_model,part_shape,
      target_facts,part_facts,source_extents,{==},
      Equal.trans(Bool,GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset}))),ReduceAxisValues.valid_of(~Element,Layout.settle(~Element,True{},Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset}))),True{},
        Equal.sym(Bool,ReduceAxisValues.valid_of(~Element,Layout.settle(~Element,True{},Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset}))),GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset}))),settled_valid(~Element,Operations.scatter(~Element,Buffer.pack(~Element,part_model),Views.View{source_axes,0},Buffer.pack(~Element,target_model),Views.View{target_axes,target_offset}))),valid)),
    valid)

# A part without elements writes nothing: its extents are those of the view,
# whose count is then zero.
def kept(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,same: Bool,+slice: Views.View,+before: Before,+part: Part,+name: String,
  +target_model: Buffer.Model<Element>,+target_shape: List<&2,U32>,+target_fill: Element,+part_model: Buffer.Model<Element>,+part_shape: List<&2,U32>,+part_fill: Element,+workers: U32,
  target_facts: GatherValues.Facts(~Element,~Before,~before_value,before,target_shape,Buffer.content(~Element,target_model)),
  +extents: {Layout.same_extents(slice,part_shape) == same : Bool},
  +emptiness: {Shapes.has_zero(part_shape) == True{} : Bool},
  accepted: {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.settled(~Element,name,same,Buffer.pack(~Element,part_model),Buffer.pack(~Element,target_model),target_shape,part_shape,target_fill,part_fill,workers))) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Placed<Before,Part>,placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,Some{slice}},target_shape,Layout.result_of(~Element,Layout.settled(~Element,name,same,Buffer.pack(~Element,part_model),Buffer.pack(~Element,target_model),target_shape,part_shape,target_fill,part_fill,workers))>:
  match same:
    case True{}:
      Views.View{+target_axes,+target_offset} = slice
      GatherValues.Shaped{target_model,target_fill,workers,{==},
        %Equal.sym(Maybe<&2,U32>,Views.numel(Views.View{target_axes,target_offset}),Some{0},
          %Equal.sym(List<&2,U32>,Views.axis_extents(target_axes),part_shape,ShapeLaws.same_extents(Views.axis_extents(target_axes),part_shape,extents)) :
            {Shapes.count_nonempty(Shapes.has_zero(_),_) == Some{0} : Maybe<&2,U32>}
          %Equal.sym(Bool,Shapes.has_zero(part_shape),True{},emptiness) : {Shapes.count_nonempty(_,part_shape) == Some{0} : Maybe<&2,U32>}
          {==}) :
          GatherValues.Facts(~Element,~Scatter.Scatter<Before,Part>,~Scatter.value(~Element,~Before,~Part,~before_value,~part_value),Scatter.Scatter{before,part,Views.View{target_axes,target_offset},U32.to_nat(Layout.count_or_zero(_))},target_shape,Buffer.content(~Element,target_model))
        target_facts}
    case False{}:
      Empty.absurd(GatherValues.Shaped<Element,Placed<Before,Part>,placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,Some{slice}},target_shape,Layout.result_of(~Element,Layout.settled(~Element,name,False{},Buffer.pack(~Element,part_model),Buffer.pack(~Element,target_model),target_shape,part_shape,target_fill,part_fill,workers))>,WordEncoding.false_true(accepted))

# A write through a view that returned Ready: the result has the target shape,
# and its reader is the scatter of the part through the view.
def place(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,empty: Bool,slice: Maybe<&2,Views.View>,source: Maybe<&2,Views.View>,+before: Before,+part: Part,+name: String,
  +target_model: Buffer.Model<Element>,+target_shape: List<&2,U32>,+target_fill: Element,+part_model: Buffer.Model<Element>,+part_shape: List<&2,U32>,+part_fill: Element,+workers: U32,
  target_facts: GatherValues.Facts(~Element,~Before,~before_value,before,target_shape,Buffer.content(~Element,target_model)),
  part_facts: GatherValues.Facts(~Element,~Part,~part_value,part,part_shape,Buffer.content(~Element,part_model)),
  +emptiness: {Shapes.has_zero(part_shape) == empty : Bool},
  +sources: {Tensors.view_of(part_shape) == source : Maybe<&2,Views.View>},
  +accepted: {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.place(~Element,name,empty,slice,source,Buffer.pack(~Element,target_model),target_shape,target_fill,Buffer.pack(~Element,part_model),part_shape,part_fill,workers))) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Placed<Before,Part>,placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,slice},target_shape,Layout.result_of(~Element,Layout.place(~Element,name,empty,slice,source,Buffer.pack(~Element,target_model),target_shape,target_fill,Buffer.pack(~Element,part_model),part_shape,part_fill,workers))>:
  match empty slice source:
    case False{} Some{+slice} Some{+source}:
      written(~Element,~Before,~Part,~before_value,~part_value,slice,source,before,part,name,target_model,target_shape,target_fill,part_model,part_shape,part_fill,workers,target_facts,part_facts,sources,accepted)
    case True{} Some{+slice} Some{source}:
      kept(~Element,~Before,~Part,~before_value,~part_value,Layout.same_extents(slice,part_shape),slice,before,part,name,target_model,target_shape,target_fill,part_model,part_shape,part_fill,workers,target_facts,{==},emptiness,accepted)
    case True{} Some{+slice} None{}:
      kept(~Element,~Before,~Part,~before_value,~part_value,Layout.same_extents(slice,part_shape),slice,before,part,name,target_model,target_shape,target_fill,part_model,part_shape,part_fill,workers,target_facts,{==},emptiness,accepted)
    case False{} Some{slice} None{}:
      Empty.absurd(GatherValues.Shaped<Element,Placed<Before,Part>,placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,Some{slice}},target_shape,Layout.result_of(~Element,Layout.place(~Element,name,False{},Some{slice},None{},Buffer.pack(~Element,target_model),target_shape,target_fill,Buffer.pack(~Element,part_model),part_shape,part_fill,workers))>,WordEncoding.false_true(accepted))
    case False{} None{} Some{source}:
      Empty.absurd(GatherValues.Shaped<Element,Placed<Before,Part>,placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,None{}},target_shape,Layout.result_of(~Element,Layout.place(~Element,name,False{},None{},Some{source},Buffer.pack(~Element,target_model),target_shape,target_fill,Buffer.pack(~Element,part_model),part_shape,part_fill,workers))>,WordEncoding.false_true(accepted))
    case False{} None{} None{}:
      Empty.absurd(GatherValues.Shaped<Element,Placed<Before,Part>,placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,None{}},target_shape,Layout.result_of(~Element,Layout.place(~Element,name,False{},None{},None{},Buffer.pack(~Element,target_model),target_shape,target_fill,Buffer.pack(~Element,part_model),part_shape,part_fill,workers))>,WordEncoding.false_true(accepted))
    case True{} None{} Some{source}:
      Empty.absurd(GatherValues.Shaped<Element,Placed<Before,Part>,placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,None{}},target_shape,Layout.result_of(~Element,Layout.place(~Element,name,True{},None{},Some{source},Buffer.pack(~Element,target_model),target_shape,target_fill,Buffer.pack(~Element,part_model),part_shape,part_fill,workers))>,WordEncoding.false_true(accepted))
    case True{} None{} None{}:
      Empty.absurd(GatherValues.Shaped<Element,Placed<Before,Part>,placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,None{}},target_shape,Layout.result_of(~Element,Layout.place(~Element,name,True{},None{},None{},Buffer.pack(~Element,target_model),target_shape,target_fill,Buffer.pack(~Element,part_model),part_shape,part_fill,workers))>,WordEncoding.false_true(accepted))

# The reader of a set_slice result: the part through the slice of the target
# extents that it replaces.
def slice_of(+axis: U32,start: U32,+target_shape: List<&2,U32>,+part_shape: List<&2,U32>) -> Maybe<&2,Views.View>:
  Layout.sliced(Tensors.view_of(target_shape),axis,start,Layout.extent_at(part_shape,U32.to_nat(axis)),1)

def placement(~Before: Data,~Part: Data,before: Before,part: Part,+axis: U32,start: U32,+target_shape: List<&2,U32>,+part_shape: List<&2,U32>) -> Placed<Before,Part>:
  Placed{before,part,slice_of(axis,start,target_shape,part_shape)}

# set_slice of two shaped tensors: the result is a shaped tensor of the
# target shape.
def shaped_slice(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,-target: Tensors.Tensor<Element>,-part_tensor: Tensors.Tensor<Element>,+before: Before,+part: Part,+axis: U32,+start: U32,+target_shape: List<&2,U32>,+part_shape: List<&2,U32>,
  target_shaped: GatherValues.Shaped<Element,Before,before_value,before,target_shape,target>,
  part_shaped: GatherValues.Shaped<Element,Part,part_value,part,part_shape,part_tensor>,
  accepted: {Dense.tensor_ready(~Element,Layout.set_slice(~Element,target,axis,start,part_tensor)) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Placed<Before,Part>,placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,Layout.sliced(Tensors.view_of(target_shape),axis,start,Layout.extent_at(part_shape,U32.to_nat(axis)),1)},target_shape,Layout.set_slice(~Element,target,axis,start,part_tensor)>:
  GatherValues.Shaped{+target_model,+target_fill,+target_workers,+target_equation,target_facts} = target_shaped
  GatherValues.Shaped{+part_model,+part_fill,+part_workers,+part_equation,part_facts} = part_shaped
  %Equal.sym(Tensors.Tensor<Element>,target,Tensors.Tensor{Buffer.pack(~Element,target_model),target_shape,target_fill,Tensors.Ready{},target_workers},target_equation) : GatherValues.Shaped<Element,Placed<Before,Part>,placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,Layout.sliced(Tensors.view_of(target_shape),axis,start,Layout.extent_at(part_shape,U32.to_nat(axis)),1)},target_shape,Layout.set_slice(~Element,_,axis,start,part_tensor)>
  %Equal.sym(Tensors.Tensor<Element>,part_tensor,Tensors.Tensor{Buffer.pack(~Element,part_model),part_shape,part_fill,Tensors.Ready{},part_workers},part_equation) : GatherValues.Shaped<Element,Placed<Before,Part>,placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,Layout.sliced(Tensors.view_of(target_shape),axis,start,Layout.extent_at(part_shape,U32.to_nat(axis)),1)},target_shape,Layout.set_slice(~Element,Tensors.Tensor{Buffer.pack(~Element,target_model),target_shape,target_fill,Tensors.Ready{},target_workers},axis,start,_)>
  place(~Element,~Before,~Part,~before_value,~part_value,Shapes.has_zero(part_shape),Layout.sliced(Tensors.view_of(target_shape),axis,start,Layout.extent_at(part_shape,U32.to_nat(axis)),1),Tensors.view_of(part_shape),before,part,"set_slice",
    target_model,target_shape,target_fill,part_model,part_shape,part_fill,U32.max(target_workers,part_workers),target_facts,part_facts,{==},{==},
    %part_equation : {Dense.tensor_ready(~Element,Layout.set_slice(~Element,Tensors.Tensor{Buffer.pack(~Element,target_model),target_shape,target_fill,Tensors.Ready{},target_workers},axis,start,_)) == True{} : Bool}
    %target_equation : {Dense.tensor_ready(~Element,Layout.set_slice(~Element,_,axis,start,part_tensor)) == True{} : Bool}
    accepted)

# Two Ready dense tensors with complete values: a Ready set_slice result is
# well formed in the target shape, and every index holds the part element
# that the slice stores there, or the target element when the slice does not
# reach it.
def set_slice(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+before: Before,+part: Part,+axis: U32,+start: U32,target: Storage.Buffer<Element>,+target_shape: List<&2,U32>,+target_fill: Element,+target_workers: U32,
  part_storage: Storage.Buffer<Element>,+part_shape: List<&2,U32>,+part_fill: Element,+part_workers: U32,
  target_contract: Dense.Contract(~Element,~Before,~before_value,before,Tensors.Tensor{target,target_shape,target_fill,Tensors.Ready{},target_workers}),
  part_contract: Dense.Contract(~Element,~Part,~part_value,part,Tensors.Tensor{part_storage,part_shape,part_fill,Tensors.Ready{},part_workers}),
  accepted: {Dense.tensor_ready(~Element,Layout.set_slice(~Element,Tensors.Tensor{target,target_shape,target_fill,Tensors.Ready{},target_workers},axis,start,Tensors.Tensor{part_storage,part_shape,part_fill,Tensors.Ready{},part_workers})) == True{} : Bool}) ->
  Dense.Contract(~Element,~Placed<Before,Part>,~placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,Layout.sliced(Tensors.view_of(target_shape),axis,start,Layout.extent_at(part_shape,U32.to_nat(axis)),1)},Layout.set_slice(~Element,Tensors.Tensor{target,target_shape,target_fill,Tensors.Ready{},target_workers},axis,start,Tensors.Tensor{part_storage,part_shape,part_fill,Tensors.Ready{},part_workers})):
  Shape.Both{Shape.Both{target_well_formed,target_status},target_values} = target_contract
  Shape.Both{Shape.Both{part_well_formed,part_status},part_values} = part_contract
  GatherValues.shaped_contract(~Element,~Placed<Before,Part>,~placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,Layout.sliced(Tensors.view_of(target_shape),axis,start,Layout.extent_at(part_shape,U32.to_nat(axis)),1)},target_shape,Layout.set_slice(~Element,Tensors.Tensor{target,target_shape,target_fill,Tensors.Ready{},target_workers},axis,start,Tensors.Tensor{part_storage,part_shape,part_fill,Tensors.Ready{},part_workers}),
    shaped_slice(~Element,~Before,~Part,~before_value,~part_value,Tensors.Tensor{target,target_shape,target_fill,Tensors.Ready{},target_workers},Tensors.Tensor{part_storage,part_shape,part_fill,Tensors.Ready{},part_workers},before,part,axis,start,target_shape,part_shape,
      GatherValues.shaped_observed(~Element,~Before,~before_value,target,before,target_shape,target_fill,target_workers,Buffer.observe(~Element,target),
        Shape.Both{target_well_formed,target_values}),
      GatherValues.shaped_observed(~Element,~Part,~part_value,part_storage,part,part_shape,part_fill,part_workers,Buffer.observe(~Element,part_storage),
        Shape.Both{part_well_formed,part_values}),
      accepted))

# Constant padding: the tensor written through the interior view of storage
# of the padded extents that holds the value everywhere.
def padded_into(~Element: Data,target: Tensors.Tensor<Element>,+before: List<&2,U32>,+padded: List<&2,U32>,part: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  Tensors.with_dense(~Element,~Tensors.Tensor<Element>,target,target => target_shape => target_fill => target_status => target_workers =>
    Layout.result_of(~Element,Layout.place_status(~Element,"pad",target_status,Tensors.Ready{},Layout.interior(before,shape,Tensors.view_of(padded),0),target,padded,target_fill,part,shape,fill,workers)))

def pad_target(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,-target: Storage.Buffer<Element>,observation: Buffer.Observation<Element,target>,+part: Part,+value: Element,+before: List<&2,U32>,+padded: List<&2,U32>,+part_model: Buffer.Model<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  target_facts: GatherValues.Facts(~Element,~Element,~ReduceAxisValues.constant(~Element),value,padded,Buffer.contents(~Element,target)),
  part_facts: GatherValues.Facts(~Element,~Part,~part_value,part,shape,Buffer.content(~Element,part_model)),
  accepted: {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.place(~Element,"pad",Shapes.has_zero(shape),Layout.interior(before,shape,Tensors.view_of(padded),0),Tensors.view_of(shape),target,padded,fill,Buffer.pack(~Element,part_model),shape,fill,workers))) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Placed<Element,Part>,placed_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Placed{value,part,Layout.interior(before,shape,Tensors.view_of(padded),0)},padded,Layout.result_of(~Element,Layout.place(~Element,"pad",Shapes.has_zero(shape),Layout.interior(before,shape,Tensors.view_of(padded),0),Tensors.view_of(shape),target,padded,fill,Buffer.pack(~Element,part_model),shape,fill,workers))>:
  Buffer.Observation{+model,+equation} = observation
  %equation : GatherValues.Shaped<Element,Placed<Element,Part>,placed_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Placed{value,part,Layout.interior(before,shape,Tensors.view_of(padded),0)},padded,Layout.result_of(~Element,Layout.place(~Element,"pad",Shapes.has_zero(shape),Layout.interior(before,shape,Tensors.view_of(padded),0),Tensors.view_of(shape),_,padded,fill,Buffer.pack(~Element,part_model),shape,fill,workers))>
  place(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value,Shapes.has_zero(shape),Layout.interior(before,shape,Tensors.view_of(padded),0),Tensors.view_of(shape),value,part,"pad",
    model,padded,fill,part_model,shape,fill,workers,
    %Buffer.packed_contents(~Element,model) : GatherValues.Facts(~Element,~Element,~ReduceAxisValues.constant(~Element),value,padded,_)
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),target,equation) : GatherValues.Facts(~Element,~Element,~ReduceAxisValues.constant(~Element),value,padded,Buffer.contents(~Element,_))
    target_facts,
    part_facts,{==},{==},
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),target,equation) : {Dense.tensor_ready(~Element,Layout.result_of(~Element,Layout.place(~Element,"pad",Shapes.has_zero(shape),Layout.interior(before,shape,Tensors.view_of(padded),0),Tensors.view_of(shape),_,padded,fill,Buffer.pack(~Element,part_model),shape,fill,workers))) == True{} : Bool}
    accepted)

def pad_counted(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,count: Maybe<&2,U32>,+part: Part,+value: Element,+before: List<&2,U32>,+padded: List<&2,U32>,+part_model: Buffer.Model<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  part_facts: GatherValues.Facts(~Element,~Part,~part_value,part,shape,Buffer.content(~Element,part_model)),
  +counts: {Tensors.count_of(padded) == count : Maybe<&2,U32>},
  accepted: {Dense.tensor_ready(~Element,padded_into(~Element,Layout.blank(~Element,count,padded,value,fill,workers),before,padded,Buffer.pack(~Element,part_model),shape,fill,workers)) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Placed<Element,Part>,placed_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Placed{value,part,Layout.interior(before,shape,Tensors.view_of(padded),0)},padded,padded_into(~Element,Layout.blank(~Element,count,padded,value,fill,workers),before,padded,Buffer.pack(~Element,part_model),shape,fill,workers)>:
  match count:
    case Some{+count}:
      pad_target(~Element,~Part,~part_value,Tensors.buffer(~Element,value,count),Buffer.observe(~Element,Tensors.buffer(~Element,value,count)),part,value,before,padded,part_model,shape,fill,workers,
        ReduceAxisValues.initial_counted(~Element,padded,value,count,Count.count_of(padded,count,counts)),part_facts,accepted)
    case None{}:
      Empty.absurd(GatherValues.Shaped<Element,Placed<Element,Part>,placed_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Placed{value,part,Layout.interior(before,shape,Tensors.view_of(padded),0)},padded,padded_into(~Element,Layout.blank(~Element,None{},padded,value,fill,workers),before,padded,Buffer.pack(~Element,part_model),shape,fill,workers)>,WordEncoding.false_true(accepted))

def pad_shaped(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,-tensor: Tensors.Tensor<Element>,+part: Part,+value: Element,+before: List<&2,U32>,+padded: List<&2,U32>,+shape: List<&2,U32>,
  part_shaped: GatherValues.Shaped<Element,Part,part_value,part,shape,tensor>,
  accepted: {Dense.tensor_ready(~Element,Layout.pad_constant(~Element,False{},value,tensor,before,padded)) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Placed<Element,Part>,placed_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Placed{value,part,Layout.interior(before,shape,Tensors.view_of(padded),0)},padded,Layout.pad_constant(~Element,False{},value,tensor,before,padded)>:
  GatherValues.Shaped{+part_model,+fill,+workers,+equation,part_facts} = part_shaped
  %Equal.sym(Tensors.Tensor<Element>,tensor,Tensors.Tensor{Buffer.pack(~Element,part_model),shape,fill,Tensors.Ready{},workers},equation) : GatherValues.Shaped<Element,Placed<Element,Part>,placed_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Placed{value,part,Layout.interior(before,shape,Tensors.view_of(padded),0)},padded,Layout.pad_constant(~Element,False{},value,_,before,padded)>
  pad_counted(~Element,~Part,~part_value,Tensors.count_of(padded),part,value,before,padded,part_model,shape,fill,workers,part_facts,{==},
    %equation : {Dense.tensor_ready(~Element,Layout.pad_constant(~Element,False{},value,_,before,padded)) == True{} : Bool}
    accepted)

# A Ready dense tensor with complete values, padded with a constant: a Ready
# result is well formed in the padded extents, and every index holds the
# tensor element that the interior view stores there, or the value.
def pad_constant(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+part: Part,+value: Element,+before: List<&2,U32>,+padded: List<&2,U32>,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Dense.Contract(~Element,~Part,~part_value,part,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers}),
  accepted: {Dense.tensor_ready(~Element,Layout.pad_constant(~Element,False{},value,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers},before,padded)) == True{} : Bool}) ->
  Dense.Contract(~Element,~Placed<Element,Part>,~placed_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Placed{value,part,Layout.interior(before,shape,Tensors.view_of(padded),0)},Layout.pad_constant(~Element,False{},value,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers},before,padded)):
  Shape.Both{Shape.Both{well_formed,status},values} = contract
  GatherValues.shaped_contract(~Element,~Placed<Element,Part>,~placed_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Placed{value,part,Layout.interior(before,shape,Tensors.view_of(padded),0)},padded,Layout.pad_constant(~Element,False{},value,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers},before,padded),
    pad_shaped(~Element,~Part,~part_value,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers},part,value,before,padded,shape,
      GatherValues.shaped_observed(~Element,~Part,~part_value,storage,part,shape,fill,workers,Buffer.observe(~Element,storage),Shape.Both{well_formed,values}),
      accepted))

# The same for the public selection: when nothing is added the tensor is
# returned as it is and keeps its own reader.
def Padded(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,unchanged: Bool,+part: Part,+value: Element,+before: List<&2,U32>,+padded: List<&2,U32>,+shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>) -> Data:
  match unchanged:
    case True{}: Dense.Contract(~Element,~Part,~part_value,part,tensor)
    case False{}: Dense.Contract(~Element,~Placed<Element,Part>,~placed_value(~Element,~Element,~Part,~ReduceAxisValues.constant(~Element),~part_value),Placed{value,part,Layout.interior(before,shape,Tensors.view_of(padded),0)},tensor)

def pad_selected(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,unchanged: Bool,+part: Part,+value: Element,+before: List<&2,U32>,+padded: List<&2,U32>,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Dense.Contract(~Element,~Part,~part_value,part,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers}),
  accepted: {Dense.tensor_ready(~Element,Layout.pad_constant(~Element,unchanged,value,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers},before,padded)) == True{} : Bool}) ->
  Padded(~Element,~Part,~part_value,unchanged,part,value,before,padded,shape,Layout.pad_constant(~Element,unchanged,value,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers},before,padded)):
  match unchanged:
    case True{}: contract
    case False{}: pad_constant(~Element,~Part,~part_value,part,value,before,padded,storage,shape,fill,workers,contract,accepted)
