# Owned inputs move through cells holding `width` consecutive output values.
# Each round gathers a cell, stores it in scratch, reads it back and scatters
# it into the owners. Storing the whole cell lets the compiler compute its
# elements together. The Array specialization refines Traversal.map_in_place.
import Base
import ./traversal_loop.bend as Loop

type State<-Owner: Type,-Cell: Data,-Context: Data> is Type:
  State{values: Owner,scratch: Array<Cell>,position: U32,slot: U32,context: Context}

def scattered(~Owner: Type,~Cell: Data,~Context: Data,~scatter: Owner -> U32 -> Cell -> Owner,~width: U32,
  +position: U32,+slot: U32,context: Context,values: Owner,read: Array<Cell> & Cell) -> State<Owner,Cell,Context>:
  (scratch,cell) = read
  State{scatter(values,position,cell),scratch,(position + width : U32),U32.and((slot + 1 : U32),63),context}

def stored(~Owner: Type,~Cell: Data,~Context: Data,~scatter: Owner -> U32 -> Cell -> Owner,~width: U32,
  +position: U32,+slot: U32,context: Context,scratch: Array<Cell>,gathered: Owner & Cell) -> State<Owner,Cell,Context>:
  (values,cell) = gathered
  scattered(~Owner,~Cell,~Context,~scatter,~width,position,slot,context,values,Array.get(Cell,Array.set(Cell,scratch,slot,cell),slot))

def step(~Owner: Type,~Cell: Data,~Context: Data,~gather: Owner -> U32 -> Context -> Owner & Cell,
  ~scatter: Owner -> U32 -> Cell -> Owner,~width: U32,state: State<Owner,Cell,Context>) -> State<Owner,Cell,Context>:
  State{values,scratch,+position,+slot,+context} = state
  stored(~Owner,~Cell,~Context,~scatter,~width,position,slot,context,scratch,gather(values,position,context))

def output(~Owner: Type,~Cell: Data,~Context: Data,state: State<Owner,Cell,Context>) -> Owner:
  State{values,scratch,position,slot,context} = state
  values

# `cells` rounds from `position`; the scratch array holds 64 cells.
def map(~Owner: Type,~Cell: Data,~Context: Data,~gather: Owner -> U32 -> Context -> Owner & Cell,
  ~scatter: Owner -> U32 -> Cell -> Owner,~width: U32,
  cells: Nat,position: U32,context: Context,values: Owner,scratch: Array<Cell>) -> Owner:
  output(~Owner,~Cell,~Context,Loop.run(~State<Owner,Cell,Context>,~step(~Owner,~Cell,~Context,~gather,~scatter,~width),
    cells,State{values,scratch,position,0,context}))
