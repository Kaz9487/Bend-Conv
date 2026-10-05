import Base
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_shape.bend as Shape
import ./storage_buffer_model.bend as Buffer

type Assignment<-T: Data> is Data:
  Put{index: U32,value: T}

def native(-T: Data,writes: List<&2,Assignment<T>>,out: Array<T>) -> Array<T>:
  match writes:
    case []:
      out
    case Put{i,v} <> rest:
      native(T,rest,Array.set(T,out,i,v))

def pure(-T: Data,writes: List<&2,Assignment<T>>,out: Tree.Tree<T>) -> Tree.Tree<T>:
  match writes:
    case []:
      out
    case Put{i,v} <> rest:
      pure(T,rest,Tree.write(T,out,i,v))

def concatenated_writes(-T: Data,+first: List<&2,Assignment<T>>,+second: List<&2,Assignment<T>>,+output: Tree.Tree<T>) ->
  {pure(T,List.append(&2,Assignment<T>,first,second),output) == pure(T,second,pure(T,first,output)) : Tree.Tree<T>}:
  match first:
    case Nil{}: {==}
    case Put{+address,+value} <> +rest:
      concatenated_writes(T,rest,second,Tree.write(T,output,address,value))

law refinement:
  for -T: Data
  for +writes: List<&2,Assignment<T>>
  for +out: Tree.Tree<T>
  {native(T,writes,Tree.pack(T,out)) == Tree.pack(T,pure(T,writes,out)) : Array<T>}
def refinement(T,writes,out):
  match writes:
    case []:
      {==}
    case Put{+i,+v} <> +rest:
      %Equal.sym(Array<T>,Array.set(T,Tree.pack(T,out),i,v),Tree.pack(T,Tree.write(T,out,i,v)),
        StorageLaws.write(T,out,i,v)) : {native(T,rest,_) == Tree.pack(T,pure(T,rest,Tree.write(T,out,i,v))) : Array<T>}
      refinement(T,rest,Tree.write(T,out,i,v))

# Keep consecutive stores as syntax until a semantic observation is needed.
# This avoids exponential unfolding of tree capacity during conversion checking.
type Program<-T: Data> is Data:
  Initial{tree: Tree.Tree<T>}
  Append{writes: List<&2,Assignment<T>>,previous: Program<T>}

def evaluate(-T: Data,program: Program<T>) -> Tree.Tree<T>:
  match program:
    case Initial{tree}:
      tree
    case Append{writes,previous}:
      pure(T,writes,evaluate(T,previous))

def initial_tree(-T: Data,program: Program<T>) -> Tree.Tree<T>:
  match program:
    case Initial{tree}: tree
    case Append{writes,previous}: initial_tree(T,previous)

# Structural invariants do not depend on the values being written.
def writes_preserve_balance(-Element: Data,+writes: List<&2,Assignment<Element>>,+tree: Tree.Tree<Element>,+depth: Nat,
  balanced: Shape.Balanced(Element,depth,tree)) -> Shape.Balanced(Element,depth,pure(Element,writes,tree)):
  match writes:
    case Nil{}: balanced
    case Put{+index,+value} <> +rest:
      writes_preserve_balance(Element,rest,Tree.write(Element,tree,index,value),depth,Shape.write(Element,depth,tree,index,value,balanced))

def writes_preserve_capacity(-Element: Data,+writes: List<&2,Assignment<Element>>,+tree: Tree.Tree<Element>,+depth: Nat,
  +balanced: Shape.Balanced(Element,depth,tree)) ->
  {Tree.capacity(Element,pure(Element,writes,tree)) == Tree.capacity(Element,tree) : U32}:
  Equal.trans(U32,Tree.capacity(Element,pure(Element,writes,tree)),U32.shln(1,depth),Tree.capacity(Element,tree),
    Shape.capacity(Element,depth,pure(Element,writes,tree),writes_preserve_balance(Element,writes,tree,depth,balanced)),
    Equal.sym(U32,Tree.capacity(Element,tree),U32.shln(1,depth),Shape.capacity(Element,depth,tree,balanced)))

def content_ready(~Element: Data,+writes: List<&2,Assignment<Element>>,+tree: Tree.Tree<Element>,+length: Nat,
  evidence: Buffer.ContentReady<Element,Buffer.Content{tree,length}>) -> Buffer.ContentReady<Element,Buffer.Content{pure(Element,writes,tree),length}>:
  Buffer.ContentReady{+depth,+balanced,depth_ok,room} = evidence
  Buffer.ContentReady{depth,writes_preserve_balance(Element,writes,tree,depth,balanced),depth_ok,
    %Equal.sym(U32,Tree.capacity(Element,pure(Element,writes,tree)),Tree.capacity(Element,tree),writes_preserve_capacity(Element,writes,tree,depth,balanced)) :
      {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool} room}

def program_preserves_balance(-Element: Data,+program: Program<Element>,+depth: Nat,
  balanced: Shape.Balanced(Element,depth,initial_tree(Element,program))) -> Shape.Balanced(Element,depth,evaluate(Element,program)):
  match program:
    case Initial{tree}: balanced
    case Append{+writes,+previous}:
      writes_preserve_balance(Element,writes,evaluate(Element,previous),depth,program_preserves_balance(Element,previous,depth,balanced))

def program_preserves_capacity(-Element: Data,+program: Program<Element>,+depth: Nat,
  +balanced: Shape.Balanced(Element,depth,initial_tree(Element,program))) ->
  {Tree.capacity(Element,evaluate(Element,program)) == Tree.capacity(Element,initial_tree(Element,program)) : U32}:
  Equal.trans(U32,Tree.capacity(Element,evaluate(Element,program)),U32.shln(1,depth),Tree.capacity(Element,initial_tree(Element,program)),
    Shape.capacity(Element,depth,evaluate(Element,program),program_preserves_balance(Element,program,depth,balanced)),
    Equal.sym(U32,Tree.capacity(Element,initial_tree(Element,program)),U32.shln(1,depth),Shape.capacity(Element,depth,initial_tree(Element,program),balanced)))
