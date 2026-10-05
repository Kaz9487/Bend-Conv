# Public full construction retains the actual count branch and allocation
# invariant. Foreign storage still supplies its documented physical readiness.
import Base
import ../tensor.bend as Tensors
import ./tensor_global_shape_count.bend as Global
import ./tensor_dense_contract.bend as Dense
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_refinement.bend as Count
import ./tensor_allocation_depth.bend as Depth
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape

def buffer_ready(~Element: Data,+count: U32,+fill: Element,allocation: Depth.Ready<count>) -> Buffer.ContentReady<Element,Buffer.contents(~Element,Tensors.buffer(~Element,fill,count))>:
  Depth.Ready{depth_ok,room} = allocation
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Tensors.buffer(~Element,fill,count)),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)},
    AllocationLaws.contents(~Element,Tensors.depth_for(count),fill,U32.to_nat(count))) : Buffer.ContentReady<Element,_>
  Buffer.content_ready(~Element,Buffer.fresh(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)),Buffer.fresh_ready(~Element,Tensors.depth_for(count),fill,U32.to_nat(count),depth_ok,room))

def buffer_length(~Element: Data,+count: U32,+fill: Element) -> {Buffer.content_length(~Element,Buffer.contents(~Element,Tensors.buffer(~Element,fill,count))) == U32.to_nat(count) : Nat}:
  Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.contents(~Element,Tensors.buffer(~Element,fill,count)),
    Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)},AllocationLaws.contents(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)))

def full_accepted(~Element: Data,+shape: List<&2,U32>,+value: Element,+count: U32,
  accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},counted: Count.Count<ShapeLaws.elements(shape),count>) ->
  Global.Normal(~Element,Tensors.full_count(~Element,shape,Some{count},value)):
  Count.Count{exact,within} = counted
  Shape.Both{buffer_ready(~Element,count,value,Depth.bounded(count,within)),
    Shape.Both{Global.Counted{count,accepted},Equal.trans(Nat,Buffer.content_length(~Element,Buffer.contents(~Element,Tensors.buffer(~Element,value,count))),U32.to_nat(count),ShapeLaws.elements(shape),
      buffer_length(~Element,count,value),Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(count),exact))}}

def full_selected(~Element: Data,+shape: List<&2,U32>,+value: Element,result: Maybe<&2,U32>,+accepted: {Tensors.count_of(shape) == result : Maybe<&2,U32>}) ->
  Global.Normal(~Element,Tensors.full_count(~Element,shape,result,value)):
  match result:
    case None{}: Shape.Both{Dense.empty_ready(~Element,value),Unit{}}
    case Some{+count}: full_accepted(~Element,shape,value,count,accepted,Count.count_of(shape,count,accepted))

def full(~Element: Data,+shape: List<&2,U32>,+value: Element) -> Global.Normal(~Element,Tensors.full(~Element,shape,value)):
  full_selected(~Element,shape,value,Tensors.count_of(shape),{==})
