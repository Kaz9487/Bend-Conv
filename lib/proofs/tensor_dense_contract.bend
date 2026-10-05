# Dense and band tensors share finite content and allocation contracts. Dense
# values use contiguous native indices; the band branch retains its existing
# index map and complete representation invariant.
import Base
import ../tensor.bend as Tensors
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./storage_allocation_refinement.bend as AllocationLaws
import ./tensor_shape_refinement.bend as ShapeLaws
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./tensor_band_map.bend as BandMap

type WellFormed<-Element: Data,-shape: List<&2,U32>,-content: Buffer.Content<Element>> is Data:
  WellFormed{length_exact: {Buffer.content_length(~Element,content) == ShapeLaws.elements(shape) : Nat},ready: Buffer.ContentReady<Element,content>}

def At(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+content: Buffer.Content<Element>,+index: Nat) -> Data:
  {Buffer.value(~Element,U32.from_nat(index),content) == original(logical,U32.from_nat(index)) : Element}

def Values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+content: Buffer.Content<Element>) -> Data:
  Members.All(Nat,{index => At(~Element,~Logical,~original,logical,content,index) : Nat -> Data},Ranges.range(Buffer.content_length(~Element,content),0n))

def read(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+content: Buffer.Content<Element>,+index: Nat,
  values: Values(~Element,~Logical,~original,logical,content),inside: {Nat.is_lt(index,Buffer.content_length(~Element,content)) == True{} : Bool}) ->
  {Buffer.value(~Element,U32.from_nat(index),content) == original(logical,U32.from_nat(index)) : Element}:
  Members.all_member(Nat,{index => At(~Element,~Logical,~original,logical,content,index) : Nat -> Data},index,Ranges.range(Buffer.content_length(~Element,content),0n),values,
    Members.range_member(Buffer.content_length(~Element,content),0n,index,inside))

def TensorWellFormed(~Element: Data,tensor: Tensors.Tensor<Element>) -> Data:
  match tensor:
    case Tensors.Tensor{storage,shape,fill,status,workers}: WellFormed<Element,shape,Buffer.contents(~Element,storage)>
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}: BandMap.WellFormed(~Element,Tensors.Banded{bands,geometry,shape,fill,status,workers})

def TensorValues(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,tensor: Tensors.Tensor<Element>) -> Data:
  match tensor:
    case Tensors.Tensor{storage,shape,fill,status,workers}: Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,storage))
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}: BandMap.LogicalValues(~Element,~Logical,~original,logical,Tensors.Banded{bands,geometry,shape,fill,status,workers})

def status_ready(status: Tensors.Status) -> Bool:
  match status:
    case Tensors.Ready{}: True{}
    case Tensors.Failed{problem}: False{}

def tensor_ready(~Element: Data,tensor: Tensors.Tensor<Element>) -> Bool:
  match tensor:
    case Tensors.Tensor{storage,shape,fill,status,workers}: status_ready(status)
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}: status_ready(status)

def Contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,-tensor: Tensors.Tensor<Element>) -> Data:
  Shape.Both<Shape.Both<TensorWellFormed(~Element,tensor),{tensor_ready(~Element,tensor) == True{} : Bool}>,TensorValues(~Element,~Logical,~original,logical,tensor)>

def empty_contents(~Element: Data,+fill: Element) ->
  {Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)) == Buffer.Content{Tree.fresh(Element,Tensors.depth_for(0),fill),0n} : Buffer.Content<Element>}:
  AllocationLaws.contents(~Element,Tensors.depth_for(0),fill,0n)

def empty_ready(~Element: Data,+fill: Element) -> Buffer.ContentReady<Element,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0))>:
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(0),fill),0n},empty_contents(~Element,fill)) : Buffer.ContentReady<Element,_>
  Buffer.content_ready(~Element,Buffer.fresh(~Element,Tensors.depth_for(0),fill,0n),Buffer.fresh_ready(~Element,Tensors.depth_for(0),fill,0n,{==},{==}))

def empty_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+fill: Element) ->
  Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0))):
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(0),fill),0n},empty_contents(~Element,fill)) : Values(~Element,~Logical,~original,logical,_)
  Unit{}

def empty_well_formed(~Element: Data,+shape: List<&2,U32>,+fill: Element,empty: {ShapeLaws.elements(shape) == 0n : Nat}) ->
  WellFormed<Element,shape,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0))>:
  WellFormed{
    Equal.trans(Nat,Buffer.content_length(~Element,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0))),0n,ShapeLaws.elements(shape),
      Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(0),fill),0n},empty_contents(~Element,fill)),
      Equal.sym(Nat,ShapeLaws.elements(shape),0n,empty)),empty_ready(~Element,fill)}

def empty_tensor(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  empty: {ShapeLaws.elements(shape) == 0n : Nat}) ->
  Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Tensors.buffer(~Element,fill,0),shape,fill,Tensors.Ready{},workers}):
  Shape.Both{Shape.Both{empty_well_formed(~Element,shape,fill,empty),{==}},empty_values(~Element,~Logical,~original,logical,fill)}
