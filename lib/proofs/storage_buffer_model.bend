# Proof-only canonical owners retain the actual certificate. They support
# induction through affine Buffer callers without duplicating native arrays.
import Base
import ../storage_buffer.bend as Storage
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_certificate.bend as Certificate
import ./storage_shape.bend as Shape
import ./storage_witness.bend as Witness
import ./word_power_of_two.bend as Power
import ./nat_order.bend as Order

type Content<-Element: Data> is Data:
  Content{tree: Tree.Tree<Element>,length: Nat}

def contents(~Element: Data,buffer: Storage.Buffer<Element>) -> Content<Element>:
  Storage.Buffer{storage,length,certificate} = buffer
  Content{Tree.reflect(Element,storage),length}

def value(~Element: Data,index: U32,content: Content<Element>) -> Element:
  Content{tree,length} = content
  Tree.read(Element,tree,index)

def content_tree(~Element: Data,content: Content<Element>) -> Tree.Tree<Element>:
  Content{tree,length} = content
  tree

def content_length(~Element: Data,content: Content<Element>) -> Nat:
  Content{tree,length} = content
  length

type ContentReady<-Element: Data,-content: Content<Element>> is Data:
  ContentReady{depth: Nat,balanced: Shape.Balanced(Element,depth,content_tree(~Element,content)),
    depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
    room: {Nat.is_le(content_length(~Element,content),U32.to_nat(Tree.capacity(Element,content_tree(~Element,content)))) == True{} : Bool}}

type Model<-Element: Data> is Data:
  Model{tree: Tree.Tree<Element>,length: Nat,certificate: Certificate.Certificate<Element,Tree.pack(Element,tree)>}

def content(~Element: Data,model: Model<Element>) -> Content<Element>:
  Model{tree,length,certificate} = model
  Content{tree,length}

def pack(~Element: Data,model: Model<Element>) -> Storage.Buffer<Element>:
  Model{tree,length,certificate} = model
  Storage.Buffer{Tree.pack(Element,tree),length,certificate}

def tree(~Element: Data,model: Model<Element>) -> Tree.Tree<Element>:
  Model{tree,length,certificate} = model
  tree

def length(~Element: Data,model: Model<Element>) -> Nat:
  Model{tree,length,certificate} = model
  length

def content_length_of(~Element: Data,+model: Model<Element>) -> {content_length(~Element,content(~Element,model)) == length(~Element,model) : Nat}:
  Model{tree,length,certificate} = model
  {==}

def depth(~Element: Data,model: Model<Element>) -> Nat:
  Model{tree,length,Certificate.Certificate{depth,equation}} = model
  depth

def balanced(~Element: Data,+model: Model<Element>) -> Shape.Balanced(Element,depth(~Element,model),tree(~Element,model)):
  Model{+tree,length,Certificate.Certificate{+depth,equation}} = model
  %StorageLaws.reflect_packed(Element,tree) : Shape.Balanced(Element,depth,_)
  Witness.balanced(Element,depth,Tree.pack(Element,tree),Certificate.witness_at(Element,depth,Tree.pack(Element,tree),equation))

def capacity_bound(~Element: Data,+model: Model<Element>,+depth_ok: {Nat.is_le(depth(~Element,model),30n) == True{} : Bool}) ->
  {U32.is_le(Tree.capacity(Element,tree(~Element,model)),2147483648) == True{} : Bool}:
  %Equal.sym(U32,Tree.capacity(Element,tree(~Element,model)),U32.shln(1,depth(~Element,model)),
    Shape.capacity(Element,depth(~Element,model),tree(~Element,model),balanced(~Element,model))) : {U32.is_le(_,2147483648) == True{} : Bool}
  Power.native_power_bound(depth(~Element,model),Order.successor(depth(~Element,model),30n,depth_ok))

def capacity_allocatable(~Element: Data,+model: Model<Element>,depth_ok: {Nat.is_le(depth(~Element,model),30n) == True{} : Bool}) ->
  {U32.is_le(Tree.capacity(Element,tree(~Element,model)),1073741824) == True{} : Bool}:
  %Equal.sym(U32,Tree.capacity(Element,tree(~Element,model)),U32.shln(1,depth(~Element,model)),
    Shape.capacity(Element,depth(~Element,model),tree(~Element,model),balanced(~Element,model))) : {U32.is_le(_,1073741824) == True{} : Bool}
  Power.native_allocatable(depth(~Element,model),depth_ok)

# Logical length and the native machine depth are independent of the stored
# element values. Allocation establishes them; content-preserving operations
# carry them with the same owner.
type Ready<-Element: Data,-model: Model<Element>> is Data:
  Ready{depth_ok: {Nat.is_le(depth(~Element,model),30n) == True{} : Bool},
    room: {Nat.is_le(length(~Element,model),U32.to_nat(Tree.capacity(Element,tree(~Element,model)))) == True{} : Bool}}

def ready_content(~Element: Data,+model: Model<Element>,evidence: ContentReady<Element,content(~Element,model)>) -> Ready<Element,model>:
  Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = model
  ContentReady{+proved_depth,proved_balanced,depth_ok,room} = evidence
  Ready{
    %Equal.sym(Nat,depth,proved_depth,Shape.unique_depth(Element,tree,depth,proved_depth,balanced(~Element,model),proved_balanced)) : {Nat.is_le(_,30n) == True{} : Bool} depth_ok,
    room}

def content_ready(~Element: Data,+model: Model<Element>,evidence: Ready<Element,model>) -> ContentReady<Element,content(~Element,model)>:
  Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = model
  Ready{depth_ok,room} = evidence
  ContentReady{depth,balanced(~Element,model),depth_ok,room}

def NativeReady(~Element: Data,buffer: Storage.Buffer<Element>) -> Data:
  Storage.Buffer{storage,length,Certificate.Certificate{depth,equation}} = buffer
  Shape.Both<{Nat.is_le(depth,30n) == True{} : Bool},{Nat.is_le(length,U32.to_nat(Tree.capacity(Element,Tree.reflect(Element,storage)))) == True{} : Bool}>

def native_content_ready(~Element: Data,buffer: Storage.Buffer<Element>,ready: NativeReady(~Element,buffer)) -> ContentReady<Element,contents(~Element,buffer)>:
  Storage.Buffer{storage,length,Certificate.Certificate{+depth,equation}} = buffer
  Shape.Both{depth_ok,room} = ready
  ContentReady{depth,Witness.balanced(Element,depth,storage,Certificate.witness_at(Element,depth,storage,equation)),depth_ok,room}

def ready_packed(~Element: Data,+model: Model<Element>,ready: NativeReady(~Element,pack(~Element,model))) -> Ready<Element,model>:
  Model{+tree,+length,Certificate.Certificate{+depth,equation}} = model
  Shape.Both{depth_ok,room} = ready
  Ready{depth_ok,
    %StorageLaws.reflect_packed(Element,tree) : {Nat.is_le(length,U32.to_nat(Tree.capacity(Element,_))) == True{} : Bool} room}

def ready_observed(~Element: Data,-buffer: Storage.Buffer<Element>,+model: Model<Element>,equation: {pack(~Element,model) == buffer : Storage.Buffer<Element>},
  ready: NativeReady(~Element,buffer)) -> Ready<Element,model>:
  ready_packed(~Element,model,
    %Equal.sym(Storage.Buffer<Element>,pack(~Element,model),buffer,equation) : NativeReady(~Element,_) ready)

def fresh_certificate(~Element: Data,+depth: Nat,+fill: Element) -> Certificate.Certificate<Element,Tree.pack(Element,Tree.fresh(Element,depth,fill))>:
  Certificate.Certificate{depth,
    %StorageLaws.fresh(Element,depth,fill) : {_ == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,_))) : Array<Element>}
    Certificate.fresh_equation(Element,depth,fill)}

def fresh(~Element: Data,+depth: Nat,+fill: Element,length: Nat) -> Model<Element>:
  Model{Tree.fresh(Element,depth,fill),length,fresh_certificate(~Element,depth,fill)}

def fresh_ready(~Element: Data,+depth: Nat,+fill: Element,+length: Nat,depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  room: {Nat.is_le(length,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}) -> Ready<Element,fresh(~Element,depth,fill,length)>:
  Ready{depth_ok,
    %Equal.sym(U32,Tree.capacity(Element,Tree.fresh(Element,depth,fill)),U32.shln(1,depth),Shape.capacity(Element,depth,Tree.fresh(Element,depth,fill),Shape.fresh(Element,depth,fill))) :
      {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool} room}

def packed_contents(~Element: Data,+model: Model<Element>) -> {contents(~Element,pack(~Element,model)) == content(~Element,model) : Content<Element>}:
  Model{+tree,+length,certificate} = model
  Equal.cong(Tree.Tree<Element>,Content<Element>,tree => Content{tree,length},Tree.reflect(Element,Tree.pack(Element,tree)),tree,StorageLaws.reflect_packed(Element,tree))

type Observation<-Element: Data,-buffer: Storage.Buffer<Element>> is Data:
  Observation{model: Model<Element>,equation: {pack(~Element,model) == buffer : Storage.Buffer<Element>}}

def proposition(~Element: Data,length: Nat,-storage: Array<Element>,-tree: Tree.Tree<Element>) -> Type:
  @certificate: Certificate.Certificate<Element,storage> -> Observation<Element,Storage.Buffer{storage,length,certificate}>

def observe(~Element: Data,buffer: Storage.Buffer<Element>) -> Observation<Element,buffer>:
  Storage.Buffer{storage,+length,certificate} = buffer
  StorageLaws.lift_observation(~Element,~Nat,~proposition(~Element),length,storage,StorageLaws.observe(Element,storage),
    tree => certificate => Observation{Model{tree,length,certificate},{==}})(certificate)

def lift(~Element: Data,~Context: Data,~Proposition: Context -> @-buffer: Storage.Buffer<Element> -> Type,+context: Context,-buffer: Storage.Buffer<Element>,
  observation: Observation<Element,buffer>,canonical: @+model: Model<Element> -> Proposition(context,pack(~Element,model))) -> Proposition(context,buffer):
  Observation{model,equation} = observation
  %equation : Proposition(context,_)
  canonical(model)
