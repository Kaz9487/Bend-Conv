# Structural proof data indexed by an erased affine owner. Runtime owners carry
# flat certificates; this tree is used only by proofs, never by numerical loops.
import Base
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape

law Witness:
  for -Element: Data
  for depth: Nat
  for storage: Array<Element>
  Data

type LeafWitness<-Element: Data,-storage: Array<Element>> is Data:
  Leaf{-value: Element,owner: {storage == ALeaf{value} : Array<Element>}}

type NodeWitness<-Element: Data,-depth: Nat,-storage: Array<Element>> is Data:
  Node{-left: Array<Element>,-right: Array<Element>,owner: {storage == ANode{left,right} : Array<Element>},
    left_balanced: Witness(Element,depth,left),right_balanced: Witness(Element,depth,right)}

def Witness(Element,depth,storage):
  match depth:
    case 0n: LeafWitness<Element,storage>
    case 1n+rest: NodeWitness<Element,rest,storage>

def balanced(-Element: Data,depth: Nat,-storage: Array<Element>,witness: Witness(Element,depth,storage)) ->
  Shape.Balanced(Element,depth,Tree.reflect(Element,storage)):
  match depth witness:
    case 0n Leaf{value,owner}:
      %Equal.sym(Array<Element>,storage,ALeaf{value},owner) : Shape.Balanced(Element,0n,Tree.reflect(Element,_))
      Unit{}
    case 1n+ +rest Node{left,right,owner,left_balanced,right_balanced}:
      %Equal.sym(Array<Element>,storage,ANode{left,right},owner) : Shape.Balanced(Element,1n+rest,Tree.reflect(Element,_))
      Shape.Both{balanced(Element,rest,left,left_balanced),balanced(Element,rest,right,right_balanced)}

def from_shape(-Element: Data,+depth: Nat,+tree: Tree.Tree<Element>,balanced: Shape.Balanced(Element,depth,tree)) ->
  Witness(Element,depth,Tree.pack(Element,tree)):
  match depth tree:
    case 0n Tree.Leaf{value}: Leaf{value,{==}}
    case 0n Tree.Node{left,right}: match balanced:
    case 1n+rest Tree.Leaf{value}: match balanced:
    case 1n+ +rest Tree.Node{+left,+right}:
      Shape.Both{left_balanced,right_balanced} = balanced
      Node{Tree.pack(Element,left),Tree.pack(Element,right),{==},from_shape(Element,rest,left,left_balanced),from_shape(Element,rest,right,right_balanced)}

def fresh_model(-Element: Data,+depth: Nat,+value: Element) ->
  {Array.new(Element,depth,value) == Tree.pack(Element,Tree.fresh(Element,depth,value)) : Array<Element>}:
  match depth:
    case 0n: {==}
    case 1n+ +rest:
      %fresh_model(Element,rest,value) : {Array.new(Element,1n+rest,value) == ANode{_,Tree.pack(Element,Tree.fresh(Element,rest,value))} : Array<Element>}
      %fresh_model(Element,rest,value) : {Array.new(Element,1n+rest,value) == ANode{Array.new(Element,rest,value),_} : Array<Element>}
      {==}

def fresh(-Element: Data,+depth: Nat,+value: Element) -> Witness(Element,depth,Array.new(Element,depth,value)):
  %Equal.sym(Array<Element>,Array.new(Element,depth,value),Tree.pack(Element,Tree.fresh(Element,depth,value)),fresh_model(Element,depth,value)) : Witness(Element,depth,_)
  from_shape(Element,depth,Tree.fresh(Element,depth,value),Shape.fresh(Element,depth,value))

def size(-Element: Data,+depth: Nat,-storage: Array<Element>,witness: Witness(Element,depth,storage)) ->
  {Array.size(Element,storage) == (storage,U32.shln(1,depth)) : Array<Element> & U32}:
  match depth witness:
    case 0n Leaf{value,owner}:
      %Equal.sym(Array<Element>,storage,ALeaf{value},owner) : {Array.size(Element,_) == (_,U32.shln(1,0n)) : Array<Element> & U32}
      {==}
    case 1n+ +rest Node{left,right,owner,left_balanced,right_balanced}:
      %Equal.sym(Array<Element>,storage,ANode{left,right},owner) : {Array.size(Element,_) == (_,U32.shln(1,1n+rest)) : Array<Element> & U32}
      %size(Element,rest,left,left_balanced) :
        {Array.size.node(Element,right,Array.size(Element,left)) == Array.size.node(Element,right,_) : Array<Element> & U32}
      {==}

type Swapped<-Element: Data,-depth: Nat,-result: Array<Element> & Element> is Data:
  Swapped{-storage: Array<Element>,-previous: Element,
    equation: {result == (storage,previous) : Array<Element> & Element},balanced: Witness(Element,depth,storage)}

def swap_left(-Element: Data,-depth: Nat,-right: Array<Element>,-result: Array<Element> & Element,
  swapped: Swapped<Element,depth,result>,right_balanced: Witness(Element,depth,right)) ->
  Swapped<Element,1n+depth,Array.swap.lo(Element,right,result)>:
  match swapped:
    case Swapped{storage,previous,equation,left_balanced}:
      Swapped{ANode{storage,right},previous,
        Equal.cong(Array<Element> & Element,Array<Element> & Element,
          pair => Array.swap.lo(Element,right,pair),result,(storage,previous),equation),
        Node{storage,right,{==},left_balanced,right_balanced}}

def swap_right(-Element: Data,-depth: Nat,-left: Array<Element>,-result: Array<Element> & Element,
  left_balanced: Witness(Element,depth,left),swapped: Swapped<Element,depth,result>) ->
  Swapped<Element,1n+depth,Array.swap.hi(Element,left,result)>:
  match swapped:
    case Swapped{storage,previous,equation,right_balanced}:
      Swapped{ANode{left,storage},previous,
        Equal.cong(Array<Element> & Element,Array<Element> & Element,
          pair => Array.swap.hi(Element,left,pair),result,(storage,previous),equation),
        Node{left,storage,{==},left_balanced,right_balanced}}

# Base passes the branch z (i < n/2) computed by its caller; any z is covered
# here, and each child receives the comparison against its own half.
def swap_choose(-Element: Data,-depth: Nat,-left: Array<Element>,-right: Array<Element>,
  +capacity: U32,+index: U32,-value: Element,choose_left: Bool,
  left_balanced: Witness(Element,depth,left),right_balanced: Witness(Element,depth,right),
  left_result: Swapped<Element,depth,Array.swap.go(Element,left,U32.shr(capacity),index,value,U32.is_lt(index,U32.shr(U32.shr(capacity))))>,
  right_result: Swapped<Element,depth,Array.swap.go(Element,right,U32.shr(capacity),U32.sub(index,U32.shr(capacity)),value,
    U32.is_lt(U32.sub(index,U32.shr(capacity)),U32.shr(U32.shr(capacity))))>) ->
  Swapped<Element,1n+depth,Array.swap.go(Element,ANode{left,right},capacity,index,value,choose_left)>:
  match choose_left:
    case True{}:
      swap_left(Element,depth,right,Array.swap.go(Element,left,U32.shr(capacity),index,value,U32.is_lt(index,U32.shr(U32.shr(capacity)))),
        left_result,right_balanced)
    case False{}:
      swap_right(Element,depth,left,Array.swap.go(Element,right,U32.shr(capacity),U32.sub(index,U32.shr(capacity)),value,
        U32.is_lt(U32.sub(index,U32.shr(capacity)),U32.shr(U32.shr(capacity)))),left_balanced,right_result)

def swap(-Element: Data,+depth: Nat,-storage: Array<Element>,+capacity: U32,+index: U32,-value: Element,choose_left: Bool,
  witness: Witness(Element,depth,storage)) -> Swapped<Element,depth,Array.swap.go(Element,storage,capacity,index,value,choose_left)>:
  match depth witness:
    case 0n Leaf{previous,owner}:
      %Equal.sym(Array<Element>,storage,ALeaf{previous},owner) : Swapped<Element,0n,Array.swap.go(Element,_,capacity,index,value,choose_left)>
      Swapped{ALeaf{value},previous,{==},Leaf{value,{==}}}
    case 1n+ +rest Node{left,right,owner,+left_balanced,+right_balanced}:
      %Equal.sym(Array<Element>,storage,ANode{left,right},owner) : Swapped<Element,1n+rest,Array.swap.go(Element,_,capacity,index,value,choose_left)>
      swap_choose(Element,rest,left,right,capacity,index,value,choose_left,left_balanced,right_balanced,
        swap(Element,rest,left,U32.shr(capacity),index,value,U32.is_lt(index,U32.shr(U32.shr(capacity))),left_balanced),
        swap(Element,rest,right,U32.shr(capacity),U32.sub(index,U32.shr(capacity)),value,
          U32.is_lt(U32.sub(index,U32.shr(capacity)),U32.shr(U32.shr(capacity))),right_balanced))

def finish(-Element: Data,-depth: Nat,-result: Array<Element> & Element,swapped: Swapped<Element,depth,result>) ->
  Witness(Element,depth,Array.set.fin(Element,result)):
  match swapped:
    case Swapped{storage,previous,equation,witness}:
      %Equal.sym(Array<Element> & Element,result,(storage,previous),equation) : Witness(Element,depth,Array.set.fin(Element,_))
      witness

def write(-Element: Data,+depth: Nat,-storage: Array<Element>,+index: U32,-value: Element,
  +witness: Witness(Element,depth,storage)) -> Witness(Element,depth,Array.set(Element,storage,index,value)):
  %Equal.sym(Array<Element> & U32,Array.size(Element,storage),(storage,U32.shln(1,depth)),size(Element,depth,storage,witness)) :
    Witness(Element,depth,Array.set.fin(Element,Array.swap.at(Element,index,value,_)))
  +start = U32.and(index,U32.sub(U32.shln(1,depth),1))
  +choose_left = U32.is_lt(start,U32.shr(U32.shln(1,depth)))
  finish(Element,depth,Array.swap.go(Element,storage,U32.shln(1,depth),start,value,choose_left),
    swap(Element,depth,storage,U32.shln(1,depth),start,value,choose_left,witness))

# Reading returns precisely the original affine owner. The observed element is
# an erased index of this proof; no runtime sample is copied into evidence.
type Read<-Element: Data,-storage: Array<Element>,-result: Array<Element> & Element> is Data:
  Read{-value: Element,equation: {result == (storage,value) : Array<Element> & Element}}

def read_left(-Element: Data,-left: Array<Element>,-right: Array<Element>,-result: Array<Element> & Element,
  read: Read<Element,left,result>) -> Read<Element,ANode{left,right},Array.swap.lo(Element,right,result)>:
  match read:
    case Read{value,equation}:
      Read{value,Equal.cong(Array<Element> & Element,Array<Element> & Element,
        pair => Array.swap.lo(Element,right,pair),result,(left,value),equation)}

def read_right(-Element: Data,-left: Array<Element>,-right: Array<Element>,-result: Array<Element> & Element,
  read: Read<Element,right,result>) -> Read<Element,ANode{left,right},Array.swap.hi(Element,left,result)>:
  match read:
    case Read{value,equation}:
      Read{value,Equal.cong(Array<Element> & Element,Array<Element> & Element,
        pair => Array.swap.hi(Element,left,pair),result,(right,value),equation)}

def read_choose(-Element: Data,-left: Array<Element>,-right: Array<Element>,+capacity: U32,+index: U32,choose_left: Bool,
  left_read: Read<Element,left,Array.get.go(Element,left,U32.shr(capacity),index,U32.is_lt(index,U32.shr(U32.shr(capacity))))>,
  right_read: Read<Element,right,Array.get.go(Element,right,U32.shr(capacity),U32.sub(index,U32.shr(capacity)),
    U32.is_lt(U32.sub(index,U32.shr(capacity)),U32.shr(U32.shr(capacity))))>) ->
  Read<Element,ANode{left,right},Array.get.go(Element,ANode{left,right},capacity,index,choose_left)>:
  match choose_left:
    case True{}:
      read_left(Element,left,right,Array.get.go(Element,left,U32.shr(capacity),index,U32.is_lt(index,U32.shr(U32.shr(capacity)))),left_read)
    case False{}:
      read_right(Element,left,right,Array.get.go(Element,right,U32.shr(capacity),U32.sub(index,U32.shr(capacity)),
        U32.is_lt(U32.sub(index,U32.shr(capacity)),U32.shr(U32.shr(capacity)))),right_read)

def read_tree(-Element: Data,+depth: Nat,-storage: Array<Element>,+capacity: U32,+index: U32,choose_left: Bool,
  witness: Witness(Element,depth,storage)) -> Read<Element,storage,Array.get.go(Element,storage,capacity,index,choose_left)>:
  match depth witness:
    case 0n Leaf{value,owner}:
      %Equal.sym(Array<Element>,storage,ALeaf{value},owner) : Read<Element,_,Array.get.go(Element,_,capacity,index,choose_left)>
      Read{value,{==}}
    case 1n+ +rest Node{left,right,owner,left_balanced,right_balanced}:
      %Equal.sym(Array<Element>,storage,ANode{left,right},owner) : Read<Element,_,Array.get.go(Element,_,capacity,index,choose_left)>
      read_choose(Element,left,right,capacity,index,choose_left,
        read_tree(Element,rest,left,U32.shr(capacity),index,U32.is_lt(index,U32.shr(U32.shr(capacity))),left_balanced),
        read_tree(Element,rest,right,U32.shr(capacity),U32.sub(index,U32.shr(capacity)),
          U32.is_lt(U32.sub(index,U32.shr(capacity)),U32.shr(U32.shr(capacity))),right_balanced))

def read(-Element: Data,+depth: Nat,-storage: Array<Element>,+index: U32,+witness: Witness(Element,depth,storage)) ->
  Read<Element,storage,Array.get(Element,storage,index)>:
  %Equal.sym(Array<Element> & U32,Array.size(Element,storage),(storage,U32.shln(1,depth)),size(Element,depth,storage,witness)) :
    Read<Element,storage,Array.get.at(Element,index,_)>
  +start = U32.and(index,U32.sub(U32.shln(1,depth),1))
  read_tree(Element,depth,storage,U32.shln(1,depth),start,U32.is_lt(start,U32.shr(U32.shln(1,depth))),witness)

# Existential depth travels with proof data, not numerical storage.
type ArrayWitness<-Element: Data,-storage: Array<Element>> is Data:
  ArrayWitness{depth: Nat,witness: Witness(Element,depth,storage)}

def allocated(-Element: Data,+depth: Nat,value: Element) -> ArrayWitness<Element,Array.new(Element,depth,value)>:
  ArrayWitness{depth,fresh(Element,depth,value)}

def stored(-Element: Data,-storage: Array<Element>,+index: U32,-value: Element,evidence: ArrayWitness<Element,storage>) ->
  ArrayWitness<Element,Array.set(Element,storage,index,value)>:
  ArrayWitness{+depth,witness} = evidence
  ArrayWitness{depth,write(Element,depth,storage,index,value,witness)}

def observed(-Element: Data,-storage: Array<Element>,index: U32,evidence: ArrayWitness<Element,storage>) ->
  Read<Element,storage,Array.get(Element,storage,index)>:
  ArrayWitness{depth,witness} = evidence
  read(Element,depth,storage,index,witness)

# The native clone duplicates payload ownership, while its source observation
# and balance depth are unchanged. This theorem is consumed only by rewrites.
def clone(-Element: Data,+depth: Nat,-storage: Array<Element>,witness: Witness(Element,depth,storage)) ->
  {Array.clone(Element,storage) == (storage,storage) : Array<Element> & Array<Element>}:
  match depth witness:
    case 0n Leaf{value,owner}:
      %Equal.sym(Array<Element>,storage,ALeaf{value},owner) : {Array.clone(Element,_) == (_,_) : Array<Element> & Array<Element>}
      {==}
    case 1n+ +rest Node{left,right,owner,left_balanced,right_balanced}:
      %Equal.sym(Array<Element>,storage,ANode{left,right},owner) : {Array.clone(Element,_) == (_,_) : Array<Element> & Array<Element>}
      %Equal.sym(Array<Element> & Array<Element>,Array.clone(Element,left),(left,left),clone(Element,rest,left,left_balanced)) :
        {Array.clone.node(Element,_,Array.clone(Element,right)) == (ANode{left,right},ANode{left,right}) : Array<Element> & Array<Element>}
      %Equal.sym(Array<Element> & Array<Element>,Array.clone(Element,right),(right,right),clone(Element,rest,right,right_balanced)) :
        {Array.clone.node(Element,(left,left),_) == (ANode{left,right},ANode{left,right}) : Array<Element> & Array<Element>}
      {==}
