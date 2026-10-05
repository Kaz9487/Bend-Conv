# Shape invariants depend on tree structure, never on the stored element type.
# Native capacity is a U32; extent is unbounded and does not assert no overflow.
import Base
import ./storage_tree_model.bend as Tree

type Both<-Left: Data,-Right: Data> is Data:
  Both{left: Left,right: Right}

def first(-Left: Data,-Right: Data,evidence: Both<Left,Right>) -> Left:
  Both{left,right} = evidence
  left

def second(-Left: Data,-Right: Data,evidence: Both<Left,Right>) -> Right:
  Both{left,right} = evidence
  right

law Balanced:
  for -Element: Data
  for depth: Nat
  for tree: Tree.Tree<Element>
  Data

def Balanced(Element,depth,tree):
  match depth tree:
    case 0n Tree.Leaf{value}: Unit
    case 0n Tree.Node{left,right}: Empty
    case 1n+rest Tree.Leaf{value}: Empty
    case 1n+ +rest Tree.Node{left,right}:
      Both<Balanced(Element,rest,left),Balanced(Element,rest,right)>

def extent(depth: Nat) -> Nat:
  match depth:
    case 0n: 1n
    case 1n+rest: Nat.double(extent(rest))

def fresh(-Element: Data,+depth: Nat,+value: Element) -> Balanced(Element,depth,Tree.fresh(Element,depth,value)):
  match depth:
    case 0n: Unit{}
    case 1n+ +rest: Both{fresh(Element,rest,value),fresh(Element,rest,value)}

def store_node(-Element: Data,+depth: Nat,+left: Tree.Tree<Element>,+right: Tree.Tree<Element>,
  +half: U32,+index: U32,+value: Element,+choose_left: Bool,
  left_balanced: Balanced(Element,depth,left),right_balanced: Balanced(Element,depth,right),
  left_stored: Balanced(Element,depth,Tree.store(Element,left,half,index,value)),
  right_stored: Balanced(Element,depth,Tree.store(Element,right,half,(index - half : U32),value))) ->
  Balanced(Element,1n+depth,Bool.pick(Tree.Tree<Element>,choose_left,
    Tree.Node{Tree.store(Element,left,half,index,value),right},
    Tree.Node{left,Tree.store(Element,right,half,(index - half : U32),value)})):
  match choose_left:
    case True{}: Both{left_stored,right_balanced}
    case False{}: Both{left_balanced,right_stored}

def store(-Element: Data,+depth: Nat,+tree: Tree.Tree<Element>,+capacity: U32,+index: U32,+value: Element,
  balanced: Balanced(Element,depth,tree)) -> Balanced(Element,depth,Tree.store(Element,tree,capacity,index,value)):
  match depth tree:
    case 0n Tree.Leaf{previous}: Unit{}
    case 0n Tree.Node{left,right}: match balanced:
    case 1n+rest Tree.Leaf{previous}: match balanced:
    case 1n+ +rest Tree.Node{+left,+right}:
      Both{+left_balanced,+right_balanced} = balanced
      store_node(Element,rest,left,right,U32.shr(capacity),index,value,U32.is_lt(index,U32.shr(capacity)),
        left_balanced,right_balanced,
        store(Element,rest,left,U32.shr(capacity),index,value,left_balanced),
        store(Element,rest,right,U32.shr(capacity),(index - U32.shr(capacity) : U32),value,right_balanced))

def write(-Element: Data,+depth: Nat,+tree: Tree.Tree<Element>,+index: U32,+value: Element,
  balanced: Balanced(Element,depth,tree)) -> Balanced(Element,depth,Tree.write(Element,tree,index,value)):
  store(Element,depth,tree,Tree.capacity(Element,tree),U32.and(index,U32.sub(Tree.capacity(Element,tree),1)),value,balanced)

def capacity(-Element: Data,+depth: Nat,+tree: Tree.Tree<Element>,balanced: Balanced(Element,depth,tree)) ->
  {Tree.capacity(Element,tree) == U32.shln(1,depth) : U32}:
  match depth tree:
    case 0n Tree.Leaf{value}: {==}
    case 0n Tree.Node{left,right}: match balanced:
    case 1n+rest Tree.Leaf{value}: match balanced:
    case 1n+ +rest Tree.Node{+left,+right}:
      Both{left_balanced,right_balanced} = balanced
      %capacity(Element,rest,left,left_balanced) : {U32.shl(Tree.capacity(Element,left)) == U32.shl(_) : U32}
      {==}

def unique_depth(-Element: Data,tree: Tree.Tree<Element>,left_depth: Nat,right_depth: Nat,
  left: Balanced(Element,left_depth,tree),right: Balanced(Element,right_depth,tree)) -> {left_depth == right_depth : Nat}:
  match tree left_depth right_depth:
    case Tree.Leaf{value} 0n 0n: {==}
    case Tree.Leaf{value} 0n 1n+right_depth: match right:
    case Tree.Leaf{value} 1n+left_depth right_depth: match left:
    case Tree.Node{left_tree,right_tree} 0n right_depth: match left:
    case Tree.Node{left_tree,right_tree} 1n+left_depth 0n: match right:
    case Tree.Node{+left_tree,right_tree} 1n+ +left_depth 1n+ +right_depth:
      Both{left_balanced,left_other} = left
      Both{right_balanced,right_other} = right
      Equal.cong(Nat,Nat,depth => 1n+depth,left_depth,right_depth,unique_depth(Element,left_tree,left_depth,right_depth,left_balanced,right_balanced))
