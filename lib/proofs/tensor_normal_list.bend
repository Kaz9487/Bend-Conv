# from_list validates the actual public count and list length before writing.
# Its certificate and allocation bound construct physical readiness directly.
import Base
import ../tensor.bend as Tensors
import ../storage_buffer.bend as Storage
import ../traversal_array.bend as Traversal
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./storage_witness.bend as Witness
import ./storage_shape.bend as Shape
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_refinement.bend as Count
import ./tensor_allocation_depth.bend as Depth
import ./tensor_dense_contract.bend as Dense
import ./tensor_global_shape_count.bend as Global
import ./nat_order.bend as Order

def written_ready(~Element: Data,+values: List<&2,Element>,+count: U32,+fill: Element,allocation: Depth.Sized<count>) ->
  Buffer.NativeReady(~Element,Storage.from_list(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count))):
  Depth.Sized{depth_ok,room} = allocation
  Shape.Both{depth_ok,
    %Equal.sym(U32,Tree.capacity(Element,Tree.reflect(Element,Traversal.write_list(~Element,values,0,Array.new(Element,Tensors.depth_for(count),fill)))),U32.shln(1,Tensors.depth_for(count)),
      Shape.capacity(Element,Tensors.depth_for(count),Tree.reflect(Element,Traversal.write_list(~Element,values,0,Array.new(Element,Tensors.depth_for(count),fill))),
        Witness.balanced(Element,Tensors.depth_for(count),Traversal.write_list(~Element,values,0,Array.new(Element,Tensors.depth_for(count),fill)),
          Certificate.written(~Element,Tensors.depth_for(count),values,0,Array.new(Element,Tensors.depth_for(count),fill),Witness.fresh(Element,Tensors.depth_for(count),fill))))) :
      {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    room}

def listed_ready(~Element: Data,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element,+count: U32,
  accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},counted: Count.Count<ShapeLaws.elements(shape),count>) ->
  Global.Normal(~Element,Tensors.listed_count(~Element,True{},shape,count,values,fill)):
  Count.Count{+exact,within} = counted
  Shape.Both{Buffer.native_content_ready(~Element,Storage.from_list(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count)),
      written_ready(~Element,values,count,fill,Depth.bounded(count,within))),
    Shape.Both{Global.Counted{count,accepted},Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(count),exact)}}

def listed_matches(~Element: Data,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element,+count: U32,matches: Bool,
  accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},counted: Count.Count<ShapeLaws.elements(shape),count>) ->
  Global.Normal(~Element,Tensors.listed_count(~Element,matches,shape,count,values,fill)):
  match matches:
    case False{}: Shape.Both{Dense.empty_ready(~Element,fill),Unit{}}
    case True{}: listed_ready(~Element,shape,values,fill,count,accepted,counted)

def selected(~Element: Data,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element,result: Maybe<&2,U32>,+accepted: {Tensors.count_of(shape) == result : Maybe<&2,U32>}) ->
  Global.Normal(~Element,Tensors.listed(~Element,shape,result,values,fill)):
  match result:
    case None{}: Shape.Both{Dense.empty_ready(~Element,fill),Unit{}}
    case Some{+count}: listed_matches(~Element,shape,values,fill,count,Nat.is_eq(List.length(&2,Element,values),U32.to_nat(count)),accepted,Count.count_of(shape,count,accepted))

def from_list(~Element: Data,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element) -> Global.Normal(~Element,Tensors.from_list(~Element,shape,values,fill)):
  selected(~Element,shape,values,fill,Tensors.count_of(shape),{==})
