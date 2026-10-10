# An arbitrary element observation commutes with every tree storage operation.
# Context may contain a vector index of any width, or any other read-only data.
import Base
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape

def map(~Element: Data,~Result: Data,~Context: Data,~observe: Context -> Element -> Result,
  +context: Context,tree: Tree.Tree<Element>) -> Tree.Tree<Result>:
  match tree:
    case Tree.Leaf{value}: Tree.Leaf{observe(context,value)}
    case Tree.Node{left,right}: Tree.Node{map(~Element,~Result,~Context,~observe,context,left),map(~Element,~Result,~Context,~observe,context,right)}

def map_preserves_balance(~Element: Data,~Result: Data,~Context: Data,~observe: Context -> Element -> Result,
  +context: Context,+depth: Nat,+tree: Tree.Tree<Element>,balanced: Shape.Balanced(Element,depth,tree)) -> Shape.Balanced(Result,depth,map(~Element,~Result,~Context,~observe,context,tree)):
  match depth tree:
    case 0n Tree.Leaf{value}: Unit{}
    case 0n Tree.Node{left,right}: balanced
    case 1n+rest Tree.Leaf{value}: balanced
    case 1n+ +rest Tree.Node{+left,+right}:
      Shape.Both{left_balanced,right_balanced} = balanced
      Shape.Both{map_preserves_balance(~Element,~Result,~Context,~observe,context,rest,left,left_balanced),
        map_preserves_balance(~Element,~Result,~Context,~observe,context,rest,right,right_balanced)}

def map_fresh(~Element: Data,~Result: Data,~Context: Data,~observe: Context -> Element -> Result,
  +context: Context,+depth: Nat,+value: Element) ->
  {map(~Element,~Result,~Context,~observe,context,Tree.fresh(Element,depth,value)) == Tree.fresh(Result,depth,observe(context,value)) : Tree.Tree<Result>}:
  match depth:
    case 0n: {==}
    case 1n+ +rest:
      %map_fresh(~Element,~Result,~Context,~observe,context,rest,value) :
        {Tree.Node{map(~Element,~Result,~Context,~observe,context,Tree.fresh(Element,rest,value)),map(~Element,~Result,~Context,~observe,context,Tree.fresh(Element,rest,value))} == Tree.Node{_,_} : Tree.Tree<Result>}
      {==}

def map_preserves_capacity(~Element: Data,~Result: Data,~Context: Data,~observe: Context -> Element -> Result,
  +context: Context,+tree: Tree.Tree<Element>) ->
  {Tree.capacity(Result,map(~Element,~Result,~Context,~observe,context,tree)) == Tree.capacity(Element,tree) : U32}:
  match tree:
    case Tree.Leaf{value}: {==}
    case Tree.Node{+left,right}:
      %map_preserves_capacity(~Element,~Result,~Context,~observe,context,left) :
        {U32.shl(Tree.capacity(Result,map(~Element,~Result,~Context,~observe,context,left))) == U32.shl(_) : U32}
      {==}

def map_fetch_branch(~Element: Data,~Result: Data,~Context: Data,~observe: Context -> Element -> Result,
  +context: Context,+condition: Bool,+left: Tree.Tree<Element>,+right: Tree.Tree<Element>,+half: U32,+index: U32,
  left_read: {Tree.fetch(Result,map(~Element,~Result,~Context,~observe,context,left),half,index) == observe(context,Tree.fetch(Element,left,half,index)) : Result},
  right_read: {Tree.fetch(Result,map(~Element,~Result,~Context,~observe,context,right),half,(index - half : U32)) == observe(context,Tree.fetch(Element,right,half,(index - half : U32))) : Result}) ->
  {Bool.pick(Result,condition,Tree.fetch(Result,map(~Element,~Result,~Context,~observe,context,left),half,index),Tree.fetch(Result,map(~Element,~Result,~Context,~observe,context,right),half,(index - half : U32))) ==
    observe(context,Bool.pick(Element,condition,Tree.fetch(Element,left,half,index),Tree.fetch(Element,right,half,(index - half : U32)))) : Result}:
  match condition:
    case True{}: left_read
    case False{}: right_read

def map_fetch(~Element: Data,~Result: Data,~Context: Data,~observe: Context -> Element -> Result,
  +context: Context,+tree: Tree.Tree<Element>,+capacity: U32,+index: U32) ->
  {Tree.fetch(Result,map(~Element,~Result,~Context,~observe,context,tree),capacity,index) == observe(context,Tree.fetch(Element,tree,capacity,index)) : Result}:
  match tree:
    case Tree.Leaf{value}: {==}
    case Tree.Node{+left,+right}:
      map_fetch_branch(~Element,~Result,~Context,~observe,context,U32.is_lt(index,U32.shr(capacity)),left,right,U32.shr(capacity),index,
        map_fetch(~Element,~Result,~Context,~observe,context,left,U32.shr(capacity),index),
        map_fetch(~Element,~Result,~Context,~observe,context,right,U32.shr(capacity),(index - U32.shr(capacity) : U32)))

def map_store_branch(~Element: Data,~Result: Data,~Context: Data,~observe: Context -> Element -> Result,
  +context: Context,+condition: Bool,+left: Tree.Tree<Element>,+right: Tree.Tree<Element>,+half: U32,+index: U32,+value: Element,
  left_step: {map(~Element,~Result,~Context,~observe,context,Tree.store(Element,left,half,index,value)) == Tree.store(Result,map(~Element,~Result,~Context,~observe,context,left),half,index,observe(context,value)) : Tree.Tree<Result>},
  right_step: {map(~Element,~Result,~Context,~observe,context,Tree.store(Element,right,half,(index - half : U32),value)) == Tree.store(Result,map(~Element,~Result,~Context,~observe,context,right),half,(index - half : U32),observe(context,value)) : Tree.Tree<Result>}) ->
  {map(~Element,~Result,~Context,~observe,context,Bool.pick(Tree.Tree<Element>,condition,Tree.Node{Tree.store(Element,left,half,index,value),right},Tree.Node{left,Tree.store(Element,right,half,(index - half : U32),value)})) ==
    Bool.pick(Tree.Tree<Result>,condition,Tree.Node{Tree.store(Result,map(~Element,~Result,~Context,~observe,context,left),half,index,observe(context,value)),map(~Element,~Result,~Context,~observe,context,right)},Tree.Node{map(~Element,~Result,~Context,~observe,context,left),Tree.store(Result,map(~Element,~Result,~Context,~observe,context,right),half,(index - half : U32),observe(context,value))}) : Tree.Tree<Result>}:
  match condition:
    case True{}:
      %left_step : {Tree.Node{map(~Element,~Result,~Context,~observe,context,Tree.store(Element,left,half,index,value)),map(~Element,~Result,~Context,~observe,context,right)} == Tree.Node{_,map(~Element,~Result,~Context,~observe,context,right)} : Tree.Tree<Result>}
      {==}
    case False{}:
      %right_step : {Tree.Node{map(~Element,~Result,~Context,~observe,context,left),map(~Element,~Result,~Context,~observe,context,Tree.store(Element,right,half,(index - half : U32),value))} == Tree.Node{map(~Element,~Result,~Context,~observe,context,left),_} : Tree.Tree<Result>}
      {==}

def map_store(~Element: Data,~Result: Data,~Context: Data,~observe: Context -> Element -> Result,
  +context: Context,+tree: Tree.Tree<Element>,+capacity: U32,+index: U32,+value: Element) ->
  {map(~Element,~Result,~Context,~observe,context,Tree.store(Element,tree,capacity,index,value)) == Tree.store(Result,map(~Element,~Result,~Context,~observe,context,tree),capacity,index,observe(context,value)) : Tree.Tree<Result>}:
  match tree:
    case Tree.Leaf{previous}: {==}
    case Tree.Node{+left,+right}:
      map_store_branch(~Element,~Result,~Context,~observe,context,U32.is_lt(index,U32.shr(capacity)),left,right,U32.shr(capacity),index,value,
        map_store(~Element,~Result,~Context,~observe,context,left,U32.shr(capacity),index,value),
        map_store(~Element,~Result,~Context,~observe,context,right,U32.shr(capacity),(index - U32.shr(capacity) : U32),value))

def map_read(~Element: Data,~Result: Data,~Context: Data,~observe: Context -> Element -> Result,
  +context: Context,+tree: Tree.Tree<Element>,+index: U32) ->
  {Tree.read(Result,map(~Element,~Result,~Context,~observe,context,tree),index) == observe(context,Tree.read(Element,tree,index)) : Result}:
  %Equal.sym(U32,Tree.capacity(Result,map(~Element,~Result,~Context,~observe,context,tree)),Tree.capacity(Element,tree),map_preserves_capacity(~Element,~Result,~Context,~observe,context,tree)) :
    {Tree.fetch(Result,map(~Element,~Result,~Context,~observe,context,tree),_,U32.and(index,(_ - 1 : U32))) == observe(context,Tree.read(Element,tree,index)) : Result}
  map_fetch(~Element,~Result,~Context,~observe,context,tree,Tree.capacity(Element,tree),U32.and(index,(Tree.capacity(Element,tree) - 1 : U32)))

def map_write(~Element: Data,~Result: Data,~Context: Data,~observe: Context -> Element -> Result,
  +context: Context,+tree: Tree.Tree<Element>,+index: U32,+value: Element) ->
  {map(~Element,~Result,~Context,~observe,context,Tree.write(Element,tree,index,value)) == Tree.write(Result,map(~Element,~Result,~Context,~observe,context,tree),index,observe(context,value)) : Tree.Tree<Result>}:
  %Equal.sym(U32,Tree.capacity(Result,map(~Element,~Result,~Context,~observe,context,tree)),Tree.capacity(Element,tree),map_preserves_capacity(~Element,~Result,~Context,~observe,context,tree)) :
    {map(~Element,~Result,~Context,~observe,context,Tree.write(Element,tree,index,value)) == Tree.store(Result,map(~Element,~Result,~Context,~observe,context,tree),_,U32.and(index,(_ - 1 : U32)),observe(context,value)) : Tree.Tree<Result>}
  map_store(~Element,~Result,~Context,~observe,context,tree,Tree.capacity(Element,tree),U32.and(index,(Tree.capacity(Element,tree) - 1 : U32)),value)
