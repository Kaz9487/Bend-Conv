# The FP32 cells of Cells.map: eight consecutive values as one Tile8.
import Base
import ./kernel_tile.bend as Tile
import ./traversal_cells.bend as Cells
import ./traversal_array.bend as Traversal

def gathered(~Context: Data,~map: Context -> F32 -> F32,+context: Context,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,+f: F32,+g: F32,
  read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (values,h) = read
  (values,Tile.Tile{map(context,a),map(context,b),map(context,c),map(context,d),map(context,e),map(context,f),map(context,g),map(context,h)})

def gather7(~Context: Data,~map: Context -> F32 -> F32,+position: U32,+context: Context,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,+f: F32,
  read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (values,g) = read
  gathered(~Context,~map,context,a,b,c,d,e,f,g,Array.get(F32,values,(position + 1 : U32)))

def gather6(~Context: Data,~map: Context -> F32 -> F32,+position: U32,+context: Context,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,
  read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (values,f) = read
  gather7(~Context,~map,(position + 1 : U32),context,a,b,c,d,e,f,Array.get(F32,values,(position + 1 : U32)))

def gather5(~Context: Data,~map: Context -> F32 -> F32,+position: U32,+context: Context,+a: F32,+b: F32,+c: F32,+d: F32,
  read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (values,e) = read
  gather6(~Context,~map,(position + 1 : U32),context,a,b,c,d,e,Array.get(F32,values,(position + 1 : U32)))

def gather4(~Context: Data,~map: Context -> F32 -> F32,+position: U32,+context: Context,+a: F32,+b: F32,+c: F32,
  read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (values,d) = read
  gather5(~Context,~map,(position + 1 : U32),context,a,b,c,d,Array.get(F32,values,(position + 1 : U32)))

def gather3(~Context: Data,~map: Context -> F32 -> F32,+position: U32,+context: Context,+a: F32,+b: F32,
  read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (values,c) = read
  gather4(~Context,~map,(position + 1 : U32),context,a,b,c,Array.get(F32,values,(position + 1 : U32)))

def gather2(~Context: Data,~map: Context -> F32 -> F32,+position: U32,+context: Context,+a: F32,
  read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (values,b) = read
  gather3(~Context,~map,(position + 1 : U32),context,a,b,Array.get(F32,values,(position + 1 : U32)))

def gather1(~Context: Data,~map: Context -> F32 -> F32,+position: U32,+context: Context,
  read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (values,a) = read
  gather2(~Context,~map,(position + 1 : U32),context,a,Array.get(F32,values,(position + 1 : U32)))

# The mapped values at position, position + 1, ... as one tile; the array is only read.
def gather(~Context: Data,~map: Context -> F32 -> F32,values: Array<F32>,position: U32,context: Context) -> Array<F32> & Tile.Tile8:
  +position = position
  gather1(~Context,~map,position,context,Array.get(F32,values,position))

# The lanes of a tile to position, position + 1, ...
def scatter(values: Array<F32>,position: U32,tile: Tile.Tile8) -> Array<F32>:
  +tile = tile
  +position = position
  +p1 = (position + 1 : U32)
  +p2 = (p1 + 1 : U32)
  +p3 = (p2 + 1 : U32)
  +p4 = (p3 + 1 : U32)
  +p5 = (p4 + 1 : U32)
  +p6 = (p5 + 1 : U32)
  Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,values,position,Tile.lane0(tile)),p1,Tile.lane1(tile)),
    p2,Tile.lane2(tile)),p3,Tile.lane3(tile)),p4,Tile.lane4(tile)),p5,Tile.lane5(tile)),p6,Tile.lane6(tile)),(p6 + 1 : U32),Tile.lane7(tile))

def scratch() -> Array<Tile.Tile8>:
  Array.new(Tile.Tile8,6n,Tile.filled(0.0))

# The values the rounds leave: fewer than eight, mapped one by one.
def finish(~Context: Data,~map: Context -> F32 -> F32,rest: Nat,state: Cells.State<Array<F32>,Tile.Tile8,Context>) -> Array<F32>:
  Cells.State{values,scratch,position,slot,context} = state
  Traversal.map_in_place(~F32,~Context,~map,rest,position,context,values)

# One eight-value loop for any owned input bundle; gather and scatter decide
# which owners are read or updated. The unary path is its Array specialization.
def rounds_with(~Owner: Type,~Context: Data,~gather: Owner -> U32 -> Context -> Owner & Tile.Tile8,
  ~scatter: Owner -> U32 -> Tile.Tile8 -> Owner,~finish: Nat -> Cells.State<Owner,Tile.Tile8,Context> -> Owner,
  count: Nat,state: Cells.State<Owner,Tile.Tile8,Context>) -> Owner:
  match count:
    case 1n+1n+1n+1n+1n+1n+1n+1n+rest:
      rounds_with(~Owner,~Context,~gather,~scatter,~finish,rest,Cells.step(~Owner,~Tile.Tile8,~Context,~gather,~scatter,~8,state))
    case few: finish(few,state)

def rounds(~Context: Data,~map: Context -> F32 -> F32,count: Nat,state: Cells.State<Array<F32>,Tile.Tile8,Context>) -> Array<F32>:
  rounds_with(~Array<F32>,~Context,~gather(~Context,~map),~scatter,~finish(~Context,~map),count,state)

# map over the count values from position, in order: Traversal.map_in_place
# with the values of each round computed together.
def map(~Context: Data,~map: Context -> F32 -> F32,count: Nat,position: U32,context: Context,values: Array<F32>) -> Array<F32>:
  rounds(~Context,~map,count,Cells.State{values,scratch(),position,0,context})

# An array of 2^depth values holds the count positions from position, within
# the depths the refinement proof covers.
def covered(+depth: Nat,+position: U32,+count: Nat) -> Bool:
  Nat.is_le(depth,30n) && Nat.is_le(Nat.add(U32.to_nat(position),count),U32.to_nat(U32.shln(1,depth)))

# map over count values from position when the array is known to hold them,
# and the scalar map otherwise: the same array either way.
def map_when(~Context: Data,~operation: Context -> F32 -> F32,covered: Bool,count: Nat,position: U32,context: Context,values: Array<F32>) -> Array<F32>:
  match covered:
    case True{}: map(~Context,~operation,count,position,context,values)
    case False{}: Traversal.map_in_place(~F32,~Context,~operation,count,position,context,values)
