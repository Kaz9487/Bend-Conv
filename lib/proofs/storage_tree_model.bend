# Proof-only Data model of native arrays, parameterized by element type.
# Addresses below intentionally retain the runtime's U32 wrap semantics.
import Base

type Tree<-T: Data> is Data:
  Leaf{value: T}
  Node{left: Tree<T>,right: Tree<T>}

def pack(-T: Data,tree: Tree<T>) -> Array<T>:
  match tree:
    case Leaf{v}:
      ALeaf{v}
    case Node{l,r}:
      ANode{pack(T,l),pack(T,r)}

# Proof observation of an existing owner, not a runtime conversion. Callers use
# it only in erased propositions; numeric code continues to use Array directly.
def reflect(-Element: Data,storage: Array<Element>) -> Tree<Element>:
  match storage:
    case ALeaf{value}: Leaf{value}
    case ANode{left,right}: Node{reflect(Element,left),reflect(Element,right)}
def capacity(-T: Data,tree: Tree<T>) -> U32:
  match tree:
    case Leaf{v}:
      1
    case Node{l,r}:
      U32.shl(capacity(T,l))
def fresh(-T: Data,n: Nat,+v: T) -> Tree<T>:
  match n:
    case 0n:
      Leaf{v}
    case 1n+ +rest:
      Node{fresh(T,rest,v),fresh(T,rest,v)}
def fetch(-T: Data,tree: Tree<T>,+n: U32,+i: U32) -> T:
  match tree:
    case Leaf{v}:
      v
    case Node{l,r}:
      Bool.pick(T,U32.is_lt(i,U32.shr(n)),fetch(T,l,U32.shr(n),i),fetch(T,r,U32.shr(n),U32.sub(i,U32.shr(n))))
def store(-T: Data,tree: Tree<T>,+n: U32,+i: U32,+v: T) -> Tree<T>:
  match tree:
    case Leaf{old}:
      Leaf{v}
    case Node{+l,+r}:
      Bool.pick(Tree<T>,U32.is_lt(i,U32.shr(n)),Node{store(T,l,U32.shr(n),i,v),r},Node{l,store(T,r,U32.shr(n),U32.sub(i,U32.shr(n)),v)})
def read(-T: Data,+tree: Tree<T>,i: U32) -> T:
  fetch(T,tree,capacity(T,tree),U32.and(i,U32.sub(capacity(T,tree),1)))
def write(-T: Data,+tree: Tree<T>,i: U32,v: T) -> Tree<T>:
  store(T,tree,capacity(T,tree),U32.and(i,U32.sub(capacity(T,tree),1)),v)

def write_sequence(~Element: Data,values: List<&2,Element>,+offset: U32,tree: Tree<Element>) -> Tree<Element>:
  match values:
    case Nil{}: tree
    case head <> tail:
      write_sequence(~Element,tail,(offset + 1 : U32),write(Element,tree,offset,head))

def read_sequence(~Element: Data,count: Nat,+offset: U32,+tree: Tree<Element>) -> List<&2,Element>:
  match count:
    case 0n: Nil{}
    case 1n+rest:
      read(Element,tree,offset) <> read_sequence(~Element,rest,(offset + 1 : U32),tree)
