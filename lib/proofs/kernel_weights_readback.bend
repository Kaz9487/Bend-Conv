import Base
import ../kernel_weights.bend as Weights
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws

def value(+tree: Tree.Tree<F32>,rows: U32,+inner: U32,+row: U32,index: U32) -> F32:
  Bool.pick(F32,U32.is_lt(row,rows),Tree.read(F32,tree,(row * inner + index : U32)),0.0)

def sample(tree: Tree.Tree<F32>,context: Weights.Context,index: U32) -> Tile.Tile8:
  Weights.Context{+rows,+inner,+row,destination} = context
  +tree = tree
  +index = index
  Tile.Tile{value(tree,rows,inner,row,index),value(tree,rows,inner,(row + 1 : U32),index),value(tree,rows,inner,(row + 2 : U32),index),value(tree,rows,inner,(row + 3 : U32),index),value(tree,rows,inner,(row + 4 : U32),index),value(tree,rows,inner,(row + 5 : U32),index),value(tree,rows,inner,(row + 6 : U32),index),value(tree,rows,inner,(row + 7 : U32),index)}

def conditional(+tree: Tree.Tree<F32>,+valid: Bool,+index: U32) ->
  {Storage.read_when(~F32,0.0,valid,Tree.pack(F32,tree),index) ==
    (Tree.pack(F32,tree),Bool.pick(F32,valid,Tree.read(F32,tree,index),0.0)) : Array<F32> & F32}:
  match valid:
    case False{}: {==}
    case True{}: StorageLaws.read(F32,tree,index)

def seventh(+tree: Tree.Tree<F32>,+rows: U32,+inner: U32,+row: U32,+index: U32,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,+f: F32,+g: F32) ->
  {Weights.seventh(rows,inner,row,index,a,b,c,d,e,f,(Tree.pack(F32,tree),g)) ==
    (Tree.pack(F32,tree),Tile.Tile{a,b,c,d,e,f,g,value(tree,rows,inner,(row + 7 : U32),index)}) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,Storage.read_when(~F32,0.0,U32.is_lt((row + 7 : U32),rows),Tree.pack(F32,tree),((row + 7) * inner + index : U32)),
    (Tree.pack(F32,tree),value(tree,rows,inner,(row + 7 : U32),index)),conditional(tree,U32.is_lt((row + 7 : U32),rows),((row + 7) * inner + index : U32))) :
    {Weights.eighth(a,b,c,d,e,f,g,_) == (Tree.pack(F32,tree),Tile.Tile{a,b,c,d,e,f,g,value(tree,rows,inner,(row + 7 : U32),index)}) : Array<F32> & Tile.Tile8}
  {==}

def sixth(+tree: Tree.Tree<F32>,+rows: U32,+inner: U32,+row: U32,+index: U32,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,+f: F32) ->
  {Weights.sixth(rows,inner,row,index,a,b,c,d,e,(Tree.pack(F32,tree),f)) ==
    (Tree.pack(F32,tree),Tile.Tile{a,b,c,d,e,f,value(tree,rows,inner,(row + 6 : U32),index),value(tree,rows,inner,(row + 7 : U32),index)}) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,Storage.read_when(~F32,0.0,U32.is_lt((row + 6 : U32),rows),Tree.pack(F32,tree),((row + 6) * inner + index : U32)),
    (Tree.pack(F32,tree),value(tree,rows,inner,(row + 6 : U32),index)),conditional(tree,U32.is_lt((row + 6 : U32),rows),((row + 6) * inner + index : U32))) :
    {Weights.seventh(rows,inner,row,index,a,b,c,d,e,f,_) == (Tree.pack(F32,tree),Tile.Tile{a,b,c,d,e,f,value(tree,rows,inner,(row + 6 : U32),index),value(tree,rows,inner,(row + 7 : U32),index)}) : Array<F32> & Tile.Tile8}
  seventh(tree,rows,inner,row,index,a,b,c,d,e,f,value(tree,rows,inner,(row + 6 : U32),index))

def fifth(+tree: Tree.Tree<F32>,+rows: U32,+inner: U32,+row: U32,+index: U32,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32) ->
  {Weights.fifth(rows,inner,row,index,a,b,c,d,(Tree.pack(F32,tree),e)) ==
    (Tree.pack(F32,tree),Tile.Tile{a,b,c,d,e,value(tree,rows,inner,(row + 5 : U32),index),value(tree,rows,inner,(row + 6 : U32),index),value(tree,rows,inner,(row + 7 : U32),index)}) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,Storage.read_when(~F32,0.0,U32.is_lt((row + 5 : U32),rows),Tree.pack(F32,tree),((row + 5) * inner + index : U32)),
    (Tree.pack(F32,tree),value(tree,rows,inner,(row + 5 : U32),index)),conditional(tree,U32.is_lt((row + 5 : U32),rows),((row + 5) * inner + index : U32))) :
    {Weights.sixth(rows,inner,row,index,a,b,c,d,e,_) == (Tree.pack(F32,tree),Tile.Tile{a,b,c,d,e,value(tree,rows,inner,(row + 5 : U32),index),value(tree,rows,inner,(row + 6 : U32),index),value(tree,rows,inner,(row + 7 : U32),index)}) : Array<F32> & Tile.Tile8}
  sixth(tree,rows,inner,row,index,a,b,c,d,e,value(tree,rows,inner,(row + 5 : U32),index))

def fourth(+tree: Tree.Tree<F32>,+rows: U32,+inner: U32,+row: U32,+index: U32,+a: F32,+b: F32,+c: F32,+d: F32) ->
  {Weights.fourth(rows,inner,row,index,a,b,c,(Tree.pack(F32,tree),d)) ==
    (Tree.pack(F32,tree),Tile.Tile{a,b,c,d,value(tree,rows,inner,(row + 4 : U32),index),value(tree,rows,inner,(row + 5 : U32),index),value(tree,rows,inner,(row + 6 : U32),index),value(tree,rows,inner,(row + 7 : U32),index)}) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,Storage.read_when(~F32,0.0,U32.is_lt((row + 4 : U32),rows),Tree.pack(F32,tree),((row + 4) * inner + index : U32)),
    (Tree.pack(F32,tree),value(tree,rows,inner,(row + 4 : U32),index)),conditional(tree,U32.is_lt((row + 4 : U32),rows),((row + 4) * inner + index : U32))) :
    {Weights.fifth(rows,inner,row,index,a,b,c,d,_) == (Tree.pack(F32,tree),Tile.Tile{a,b,c,d,value(tree,rows,inner,(row + 4 : U32),index),value(tree,rows,inner,(row + 5 : U32),index),value(tree,rows,inner,(row + 6 : U32),index),value(tree,rows,inner,(row + 7 : U32),index)}) : Array<F32> & Tile.Tile8}
  fifth(tree,rows,inner,row,index,a,b,c,d,value(tree,rows,inner,(row + 4 : U32),index))

def third(+tree: Tree.Tree<F32>,+rows: U32,+inner: U32,+row: U32,+index: U32,+a: F32,+b: F32,+c: F32) ->
  {Weights.third(rows,inner,row,index,a,b,(Tree.pack(F32,tree),c)) ==
    (Tree.pack(F32,tree),Tile.Tile{a,b,c,value(tree,rows,inner,(row + 3 : U32),index),value(tree,rows,inner,(row + 4 : U32),index),value(tree,rows,inner,(row + 5 : U32),index),value(tree,rows,inner,(row + 6 : U32),index),value(tree,rows,inner,(row + 7 : U32),index)}) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,Storage.read_when(~F32,0.0,U32.is_lt((row + 3 : U32),rows),Tree.pack(F32,tree),((row + 3) * inner + index : U32)),
    (Tree.pack(F32,tree),value(tree,rows,inner,(row + 3 : U32),index)),conditional(tree,U32.is_lt((row + 3 : U32),rows),((row + 3) * inner + index : U32))) :
    {Weights.fourth(rows,inner,row,index,a,b,c,_) == (Tree.pack(F32,tree),Tile.Tile{a,b,c,value(tree,rows,inner,(row + 3 : U32),index),value(tree,rows,inner,(row + 4 : U32),index),value(tree,rows,inner,(row + 5 : U32),index),value(tree,rows,inner,(row + 6 : U32),index),value(tree,rows,inner,(row + 7 : U32),index)}) : Array<F32> & Tile.Tile8}
  fourth(tree,rows,inner,row,index,a,b,c,value(tree,rows,inner,(row + 3 : U32),index))

def second(+tree: Tree.Tree<F32>,+rows: U32,+inner: U32,+row: U32,+index: U32,+a: F32,+b: F32) ->
  {Weights.second(rows,inner,row,index,a,(Tree.pack(F32,tree),b)) ==
    (Tree.pack(F32,tree),Tile.Tile{a,b,value(tree,rows,inner,(row + 2 : U32),index),value(tree,rows,inner,(row + 3 : U32),index),value(tree,rows,inner,(row + 4 : U32),index),value(tree,rows,inner,(row + 5 : U32),index),value(tree,rows,inner,(row + 6 : U32),index),value(tree,rows,inner,(row + 7 : U32),index)}) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,Storage.read_when(~F32,0.0,U32.is_lt((row + 2 : U32),rows),Tree.pack(F32,tree),((row + 2) * inner + index : U32)),
    (Tree.pack(F32,tree),value(tree,rows,inner,(row + 2 : U32),index)),conditional(tree,U32.is_lt((row + 2 : U32),rows),((row + 2) * inner + index : U32))) :
    {Weights.third(rows,inner,row,index,a,b,_) == (Tree.pack(F32,tree),Tile.Tile{a,b,value(tree,rows,inner,(row + 2 : U32),index),value(tree,rows,inner,(row + 3 : U32),index),value(tree,rows,inner,(row + 4 : U32),index),value(tree,rows,inner,(row + 5 : U32),index),value(tree,rows,inner,(row + 6 : U32),index),value(tree,rows,inner,(row + 7 : U32),index)}) : Array<F32> & Tile.Tile8}
  third(tree,rows,inner,row,index,a,b,value(tree,rows,inner,(row + 2 : U32),index))

def first(+tree: Tree.Tree<F32>,+rows: U32,+inner: U32,+row: U32,+index: U32,+a: F32) ->
  {Weights.first(rows,inner,row,index,(Tree.pack(F32,tree),a)) ==
    (Tree.pack(F32,tree),Tile.Tile{a,value(tree,rows,inner,(row + 1 : U32),index),value(tree,rows,inner,(row + 2 : U32),index),value(tree,rows,inner,(row + 3 : U32),index),value(tree,rows,inner,(row + 4 : U32),index),value(tree,rows,inner,(row + 5 : U32),index),value(tree,rows,inner,(row + 6 : U32),index),value(tree,rows,inner,(row + 7 : U32),index)}) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,Storage.read_when(~F32,0.0,U32.is_lt((row + 1 : U32),rows),Tree.pack(F32,tree),((row + 1) * inner + index : U32)),
    (Tree.pack(F32,tree),value(tree,rows,inner,(row + 1 : U32),index)),conditional(tree,U32.is_lt((row + 1 : U32),rows),((row + 1) * inner + index : U32))) :
    {Weights.second(rows,inner,row,index,a,_) == (Tree.pack(F32,tree),Tile.Tile{a,value(tree,rows,inner,(row + 1 : U32),index),value(tree,rows,inner,(row + 2 : U32),index),value(tree,rows,inner,(row + 3 : U32),index),value(tree,rows,inner,(row + 4 : U32),index),value(tree,rows,inner,(row + 5 : U32),index),value(tree,rows,inner,(row + 6 : U32),index),value(tree,rows,inner,(row + 7 : U32),index)}) : Array<F32> & Tile.Tile8}
  second(tree,rows,inner,row,index,a,value(tree,rows,inner,(row + 1 : U32),index))

def read(+tree: Tree.Tree<F32>,+context: Weights.Context,+index: U32) ->
  {Weights.read(Tree.pack(F32,tree),context,index) == (Tree.pack(F32,tree),sample(tree,context,index)) : Array<F32> & Tile.Tile8}:
  Weights.Context{+rows,+inner,+row,destination} = context
  %Equal.sym(Array<F32> & F32,Storage.read_when(~F32,0.0,U32.is_lt(row,rows),Tree.pack(F32,tree),(row * inner + index : U32)),
    (Tree.pack(F32,tree),value(tree,rows,inner,row,index)),conditional(tree,U32.is_lt(row,rows),(row * inner + index : U32))) :
    {Weights.first(rows,inner,row,index,_) == (Tree.pack(F32,tree),sample(tree,Weights.Context{rows,inner,row,destination},index)) : Array<F32> & Tile.Tile8}
  first(tree,rows,inner,row,index,value(tree,rows,inner,row,index))
