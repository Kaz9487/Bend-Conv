# The values of the dense combination with a scalar. The tensor is combined
# in place with a one-element buffer read through a stride-zero view, so a
# Ready result holds operation(x, scalar) at every index. mean_axis is the
# reduction along one axis followed by that combination with the extent of
# the axis: every element is the fold of its slices divided by the extent.
import Base
import ./tensor_operations_refinement.bend as OperationsLaws
import ../tensor.bend as Tensors
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ../tensor_layout.bend as Layout
import ../tensor_operations.bend as Operations
import ../tensor_f32.bend as F
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./tensor_dense_contract.bend as Dense
import ./tensor_gather_values.bend as GatherValues
import ./tensor_combine_dense_values.bend as CombineDense
import ./tensor_reduce_axis_values.bend as ReduceAxisValues
import ./tensor_count_refinement.bend as Count
import ./word_encoding.bend as WordEncoding

# The logical tensor and the scalar every element is combined with.
type Scaled<-Logical: Data,-Element: Data> is Data:
  Scaled{logical: Logical,scalar: Element}

def value(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,context: Scaled<Logical,Element>,index: U32) -> Element:
  Scaled{logical,scalar} = context
  operation(original(logical,index),scalar)

# The scalar route and the two-tensor route accept the same combinations.
def applied_ready(~Element: Data,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,+name: String,+shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  {Dense.tensor_ready(~Element,Tensors.scalar_applied(~Element,name,shape,fill,workers,outcome)) == Dense.tensor_ready(~Element,Tensors.combined(~Element,name,shape,shape,fill,workers,outcome)) : Bool}:
  match outcome:
    case Operations.Accepted{(storage,right)}: {==}
    case Operations.Rejected{(storage,right)}: {==}

# The right operand holds the scalar at every address, so the combined reader
# does not depend on the view it is read through.
def applied_shaped(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,+logical: Logical,+scalar: Element,+source: Views.View,+name: String,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Dense.Contract(~Element,~CombineDense.Combine<Logical,Element>,~CombineDense.value(~Element,~Logical,~Element,~original,~ReduceAxisValues.constant(~Element),~operation),CombineDense.Combine{logical,scalar,source},Tensors.combined(~Element,name,shape,shape,fill,workers,outcome))) ->
  GatherValues.Shaped<Element,Scaled<Logical,Element>,value(~Element,~Logical,~original,~operation),Scaled{logical,scalar},shape,Tensors.scalar_applied(~Element,name,shape,fill,workers,outcome)>:
  match outcome:
    case Operations.Accepted{(storage,right)}:
      Shape.Both{Shape.Both{well_formed,status},values} = contract
      GatherValues.shaped_observed(~Element,~Scaled<Logical,Element>,~value(~Element,~Logical,~original,~operation),storage,Scaled{logical,scalar},shape,fill,workers,Buffer.observe(~Element,storage),Shape.Both{well_formed,values})
    case Operations.Rejected{(storage,right)}:
      Shape.Both{Shape.Both{well_formed,status},values} = contract
      Empty.absurd(GatherValues.Shaped<Element,Scaled<Logical,Element>,value(~Element,~Logical,~original,~operation),Scaled{logical,scalar},shape,Tensors.scalar_applied(~Element,name,shape,fill,workers,Operations.Rejected{(storage,right)})>,WordEncoding.false_true(status))

def viewed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,view: Maybe<&2,Views.View>,+logical: Logical,+name: String,+model: Buffer.Model<Element>,+shape: List<&2,U32>,+fill: Element,+scalar: Element,+workers: U32,
  +views: {Tensors.view_of(shape) == view : Maybe<&2,Views.View>},
  facts: GatherValues.Facts(~Element,~Logical,~original,logical,shape,Buffer.content(~Element,model)),
  operand: GatherValues.Facts(~Element,~Element,~ReduceAxisValues.constant(~Element),scalar,[],Buffer.contents(~Element,Tensors.buffer(~Element,scalar,1))),
  accepted: {Dense.tensor_ready(~Element,Tensors.scalar_view(~Element,~operation,name,view,Buffer.pack(~Element,model),shape,fill,scalar,workers)) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Scaled<Logical,Element>,value(~Element,~Logical,~original,~operation),Scaled{logical,scalar},shape,Tensors.scalar_view(~Element,~operation,name,view,Buffer.pack(~Element,model),shape,fill,scalar,workers)>:
  match view:
    case Some{+view}:
      Shape.Both{Dense.WellFormed{operand_length,operand_ready},operand_values} = operand
      applied_shaped(~Element,~Logical,~original,~operation,Operations.combine(~Element,~operation,Buffer.pack(~Element,model),view,Tensors.buffer(~Element,scalar,1),Tensors.broadcast(view)),logical,scalar,Tensors.broadcast(view),name,shape,fill,workers,
        CombineDense.targeted(~Element,~Logical,~Element,~original,~ReduceAxisValues.constant(~Element),~operation,~Operations.combine_kernel(~Element,~operation),~OperationsLaws.scalar_kernel(~Element,~operation),view,logical,scalar,name,Tensors.broadcast(view),Buffer.pack(~Element,model),Tensors.buffer(~Element,scalar,1),shape,shape,fill,workers,views,
          %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model)) : GatherValues.Facts(~Element,~Logical,~original,logical,shape,_)
          facts,operand_ready,operand_values,
          Equal.trans(Bool,Dense.tensor_ready(~Element,Tensors.combined(~Element,name,shape,shape,fill,workers,Operations.combine(~Element,~operation,Buffer.pack(~Element,model),view,Tensors.buffer(~Element,scalar,1),Tensors.broadcast(view)))),Dense.tensor_ready(~Element,Tensors.scalar_applied(~Element,name,shape,fill,workers,Operations.combine(~Element,~operation,Buffer.pack(~Element,model),view,Tensors.buffer(~Element,scalar,1),Tensors.broadcast(view)))),True{},
            Equal.sym(Bool,Dense.tensor_ready(~Element,Tensors.scalar_applied(~Element,name,shape,fill,workers,Operations.combine(~Element,~operation,Buffer.pack(~Element,model),view,Tensors.buffer(~Element,scalar,1),Tensors.broadcast(view)))),Dense.tensor_ready(~Element,Tensors.combined(~Element,name,shape,shape,fill,workers,Operations.combine(~Element,~operation,Buffer.pack(~Element,model),view,Tensors.buffer(~Element,scalar,1),Tensors.broadcast(view)))),applied_ready(~Element,Operations.combine(~Element,~operation,Buffer.pack(~Element,model),view,Tensors.buffer(~Element,scalar,1),Tensors.broadcast(view)),name,shape,fill,workers)),accepted)))
    case None{}: Empty.absurd(GatherValues.Shaped<Element,Scaled<Logical,Element>,value(~Element,~Logical,~original,~operation),Scaled{logical,scalar},shape,Tensors.scalar_view(~Element,~operation,name,None{},Buffer.pack(~Element,model),shape,fill,scalar,workers)>,WordEncoding.false_true(accepted))

# The combination of a shaped tensor with a scalar: a Ready result is a
# shaped tensor of the same shape holding operation(x, scalar).
def shaped(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,-tensor: Tensors.Tensor<Element>,+logical: Logical,+name: String,+operations: F32,+shape: List<&2,U32>,+scalar: Element,
  evidence: GatherValues.Shaped<Element,Logical,original,logical,shape,tensor>,
  accepted: {Dense.tensor_ready(~Element,Tensors.combine_scalar(~Element,~operation,name,operations,tensor,scalar)) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Scaled<Logical,Element>,value(~Element,~Logical,~original,~operation),Scaled{logical,scalar},shape,Tensors.combine_scalar(~Element,~operation,name,operations,tensor,scalar)>:
  GatherValues.Shaped{+model,+fill,+workers,+equation,facts} = evidence
  %Equal.sym(Tensors.Tensor<Element>,tensor,Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers},equation) : GatherValues.Shaped<Element,Scaled<Logical,Element>,value(~Element,~Logical,~original,~operation),Scaled{logical,scalar},shape,Tensors.combine_scalar(~Element,~operation,name,operations,_,scalar)>
  viewed(~Element,~Logical,~original,~operation,Tensors.view_of(shape),logical,name,model,shape,fill,scalar,workers,{==},facts,
    ReduceAxisValues.initial_counted(~Element,[],scalar,1,Count.count_of([],1,{==})),
    %equation : {Dense.tensor_ready(~Element,Tensors.combine_scalar(~Element,~operation,name,operations,_,scalar)) == True{} : Bool}
    accepted)

# A Ready dense tensor with complete values, combined with a scalar: a Ready
# result satisfies the dense contract of operation(x, scalar).
def combine_scalar(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+name: String,+operations: F32,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,+scalar: Element,
  contract: Dense.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers}),
  accepted: {Dense.tensor_ready(~Element,Tensors.combine_scalar(~Element,~operation,name,operations,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},scalar)) == True{} : Bool}) ->
  Dense.Contract(~Element,~Scaled<Logical,Element>,~value(~Element,~Logical,~original,~operation),Scaled{logical,scalar},Tensors.combine_scalar(~Element,~operation,name,operations,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},scalar)):
  Shape.Both{Shape.Both{well_formed,status},values} = contract
  GatherValues.shaped_contract(~Element,~Scaled<Logical,Element>,~value(~Element,~Logical,~original,~operation),Scaled{logical,scalar},shape,Tensors.combine_scalar(~Element,~operation,name,operations,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},scalar),
    shaped(~Element,~Logical,~original,~operation,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},logical,name,operations,shape,scalar,
      GatherValues.shaped_observed(~Element,~Logical,~original,storage,logical,shape,fill,workers,Buffer.observe(~Element,storage),Shape.Both{well_formed,values}),accepted))

# A tensor that is not Ready stays so: a Ready result had a Ready operand.
def operand_ready(~Element: Data,~operation: Element -> Element -> Element,tensor: Tensors.Tensor<Element>,+name: String,+operations: F32,+scalar: Element,
  accepted: {Dense.tensor_ready(~Element,Tensors.combine_scalar(~Element,~operation,name,operations,tensor,scalar)) == True{} : Bool}) -> {Dense.tensor_ready(~Element,tensor) == True{} : Bool}:
  match tensor:
    case Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers}: {==}
    case Tensors.Tensor{storage,shape,fill,Tensors.Failed{problem},workers}: accepted
    case Tensors.Banded{bands,geometry,shape,fill,Tensors.Usable{},workers}: {==}
    case Tensors.Banded{bands,geometry,shape,fill,Tensors.Failed{problem},workers}: accepted

# mean_axis over a modeled input: the sum along the axis, from 0.0 in
# increasing index, divided by the extent of the axis.
def mean_modeled(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+axis: U32,+model: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  facts: GatherValues.Facts(~F32,~Logical,~original,logical,shape,Buffer.content(~F32,model)),
  +accepted: {Dense.tensor_ready(~F32,F.mean_axis(Tensors.Tensor{Buffer.pack(~F32,model),shape,fill,Tensors.Usable{},workers},axis)) == True{} : Bool}) ->
  GatherValues.Shaped<F32,Scaled<ReduceAxisValues.Fold<F32,Logical>,F32>,value(~F32,~ReduceAxisValues.Fold<F32,Logical>,~ReduceAxisValues.value(~F32,~F32,~Logical,~ReduceAxisValues.constant(~F32),~original,~F32.add),~F32.div),Scaled{ReduceAxisValues.Fold{0.0,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},U32.to_f32(Layout.extent_at(shape,U32.to_nat(axis)))},Shapes.without_axis(shape,U32.to_nat(axis)),F.mean_axis(Tensors.Tensor{Buffer.pack(~F32,model),shape,fill,Tensors.Usable{},workers},axis)>:
  shaped(~F32,~ReduceAxisValues.Fold<F32,Logical>,~ReduceAxisValues.value(~F32,~F32,~Logical,~ReduceAxisValues.constant(~F32),~original,~F32.add),~F32.div,Layout.reduce_axis(~F32,~F32.add,"mean_axis",0.0,Tensors.Tensor{Buffer.pack(~F32,model),shape,fill,Tensors.Usable{},workers},axis),ReduceAxisValues.Fold{0.0,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},"mean_axis",1.0,Shapes.without_axis(shape,U32.to_nat(axis)),U32.to_f32(Layout.extent_at(shape,U32.to_nat(axis))),
    ReduceAxisValues.shaped(~F32,~Logical,~original,~F32.add,Tensors.Tensor{Buffer.pack(~F32,model),shape,fill,Tensors.Usable{},workers},logical,"mean_axis",0.0,axis,shape,GatherValues.Shaped{model,fill,workers,{==},facts},
      operand_ready(~F32,~F32.div,Layout.reduce_axis(~F32,~F32.add,"mean_axis",0.0,Tensors.Tensor{Buffer.pack(~F32,model),shape,fill,Tensors.Usable{},workers},axis),"mean_axis",1.0,U32.to_f32(Layout.extent_at(shape,U32.to_nat(axis))),accepted)),
    accepted)

def mean_shaped(~Logical: Data,~original: Logical -> U32 -> F32,-tensor: Tensors.Tensor<F32>,+logical: Logical,+axis: U32,+shape: List<&2,U32>,
  evidence: GatherValues.Shaped<F32,Logical,original,logical,shape,tensor>,
  accepted: {Dense.tensor_ready(~F32,F.mean_axis(tensor,axis)) == True{} : Bool}) ->
  GatherValues.Shaped<F32,Scaled<ReduceAxisValues.Fold<F32,Logical>,F32>,value(~F32,~ReduceAxisValues.Fold<F32,Logical>,~ReduceAxisValues.value(~F32,~F32,~Logical,~ReduceAxisValues.constant(~F32),~original,~F32.add),~F32.div),Scaled{ReduceAxisValues.Fold{0.0,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},U32.to_f32(Layout.extent_at(shape,U32.to_nat(axis)))},Shapes.without_axis(shape,U32.to_nat(axis)),F.mean_axis(tensor,axis)>:
  GatherValues.Shaped{+model,+fill,+workers,+equation,facts} = evidence
  %Equal.sym(Tensors.Tensor<F32>,tensor,Tensors.Tensor{Buffer.pack(~F32,model),shape,fill,Tensors.Usable{},workers},equation) : GatherValues.Shaped<F32,Scaled<ReduceAxisValues.Fold<F32,Logical>,F32>,value(~F32,~ReduceAxisValues.Fold<F32,Logical>,~ReduceAxisValues.value(~F32,~F32,~Logical,~ReduceAxisValues.constant(~F32),~original,~F32.add),~F32.div),Scaled{ReduceAxisValues.Fold{0.0,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},U32.to_f32(Layout.extent_at(shape,U32.to_nat(axis)))},Shapes.without_axis(shape,U32.to_nat(axis)),F.mean_axis(_,axis)>
  mean_modeled(~Logical,~original,logical,axis,model,shape,fill,workers,facts,
    %equation : {Dense.tensor_ready(~F32,F.mean_axis(_,axis)) == True{} : Bool}
    accepted)

# A Ready dense tensor with complete values: a Ready mean_axis result is well
# formed in the shape without the axis, and every element is
# (... ((0.0 + x[0]) + x[1]) ... + x[extent - 1]) / extent.
def mean_axis(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+axis: U32,storage: Storage.Buffer<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  contract: Dense.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers}),
  accepted: {Dense.tensor_ready(~F32,F.mean_axis(Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},axis)) == True{} : Bool}) ->
  Dense.Contract(~F32,~Scaled<ReduceAxisValues.Fold<F32,Logical>,F32>,~value(~F32,~ReduceAxisValues.Fold<F32,Logical>,~ReduceAxisValues.value(~F32,~F32,~Logical,~ReduceAxisValues.constant(~F32),~original,~F32.add),~F32.div),Scaled{ReduceAxisValues.Fold{0.0,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},U32.to_f32(Layout.extent_at(shape,U32.to_nat(axis)))},F.mean_axis(Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},axis)):
  Shape.Both{Shape.Both{well_formed,status},values} = contract
  GatherValues.shaped_contract(~F32,~Scaled<ReduceAxisValues.Fold<F32,Logical>,F32>,~value(~F32,~ReduceAxisValues.Fold<F32,Logical>,~ReduceAxisValues.value(~F32,~F32,~Logical,~ReduceAxisValues.constant(~F32),~original,~F32.add),~F32.div),Scaled{ReduceAxisValues.Fold{0.0,logical,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape},U32.to_f32(Layout.extent_at(shape,U32.to_nat(axis)))},Shapes.without_axis(shape,U32.to_nat(axis)),F.mean_axis(Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},axis),
    mean_shaped(~Logical,~original,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},logical,axis,shape,
      GatherValues.shaped_observed(~F32,~Logical,~original,storage,logical,shape,fill,workers,Buffer.observe(~F32,storage),Shape.Both{well_formed,values}),accepted))
