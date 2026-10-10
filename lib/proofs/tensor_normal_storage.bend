# Foreign storage supplies physical readiness at its documented boundary.
# The actual from_storage count/length checks establish the public shape part.
import Base
import ../tensor.bend as Tensors
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_refinement.bend as Count
import ./tensor_global_shape_count.bend as Global
import ./nat_order.bend as Order

def stored(~Element: Data,+model: Buffer.Model<Element>,+shape: List<&2,U32>,+fill: Element,+count: U32,
  ready: Buffer.Sized<Element,model>,+length_exact: {Buffer.length(~Element,model) == ShapeLaws.elements(shape) : Nat},
  accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},counted: Count.Count<ShapeLaws.elements(shape),count>) ->
  Global.Normal(~Element,Tensors.storage_of(~Element,shape,Some{count},Buffer.pack(~Element,model),fill)):
  Buffer.Model{+tree,+length,+certificate} = model
  Count.Count{+exact,within} = counted
  +length_count = Equal.trans(Nat,length,ShapeLaws.elements(shape),U32.to_nat(count),length_exact,exact)
  +checked = {
    %length_count : {Nat.is_eq(length,_) == True{} : Bool}
    Order.equal_reflexive(length) : {Nat.is_eq(length,U32.to_nat(count)) == True{} : Bool}}
  %Equal.sym(Bool,Nat.is_eq(length,U32.to_nat(count)),True{},checked) :
    Global.Normal(~Element,Tensors.stored(~Element,_,Buffer.pack(~Element,model),shape,fill))
  Shape.Both{
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model)) : Buffer.ContentReady<Element,_>
    Buffer.content_ready(~Element,model,ready),
    Shape.Both{Global.Counted{count,accepted},
      %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model)) : {Buffer.content_length(~Element,_) == ShapeLaws.elements(shape) : Nat}
      length_exact}}

def from_storage(~Element: Data,+model: Buffer.Model<Element>,+shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>,
  ready: Buffer.Sized<Element,model>,length_exact: {Buffer.length(~Element,model) == ShapeLaws.elements(shape) : Nat}) ->
  Global.Normal(~Element,Tensors.from_storage(~Element,shape,Buffer.pack(~Element,model),fill)):
  Global.Counted{+count,+accepted} = counted
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(shape),Some{count},accepted) : Global.Normal(~Element,Tensors.storage_of(~Element,shape,_,Buffer.pack(~Element,model),fill))
  stored(~Element,model,shape,fill,count,ready,length_exact,accepted,Count.count_of(shape,count,accepted))
