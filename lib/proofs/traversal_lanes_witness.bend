# Shape evidence for Lanes.map: every round reads the values array, writes a
# scratch tile, reads it back and stores eight values, so the array keeps its
# depth. Numerical loops call none of these functions.
import Base
import ../traversal_lanes.bend as Lanes
import ../traversal_cells.bend as Cells
import ../traversal_array.bend as Traversal
import ../kernel_tile.bend as Tile
import ./storage_witness.bend as Witness
import ./traversal_witness.bend as TraversalWitness

type Gathered<-values: Array<F32>,-result: Array<F32> & Tile.Tile8> is Data:
  Gathered{-cell: Tile.Tile8,equation: {result == (values,cell) : Array<F32> & Tile.Tile8}}

def gathered(~Context: Data,~map: Context -> F32 -> F32,-values: Array<F32>,+position: U32,+context: Context,
  r0: Witness.Read<F32,values,Array.get(F32,values,position)>,
  r1: Witness.Read<F32,values,Array.get(F32,values,(position + 1 : U32))>,
  r2: Witness.Read<F32,values,Array.get(F32,values,((position + 1 : U32) + 1 : U32))>,
  r3: Witness.Read<F32,values,Array.get(F32,values,(((position + 1 : U32) + 1 : U32) + 1 : U32))>,
  r4: Witness.Read<F32,values,Array.get(F32,values,((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32))>,
  r5: Witness.Read<F32,values,Array.get(F32,values,(((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32))>,
  r6: Witness.Read<F32,values,Array.get(F32,values,((((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32))>,
  r7: Witness.Read<F32,values,Array.get(F32,values,(((((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32))>) ->
  Gathered<values,Lanes.gather(~Context,~map,values,position,context)>:
  match r0 r1 r2 r3 r4 r5 r6 r7:
    case Witness.Read{a,ea} Witness.Read{b,eb} Witness.Read{c,ec} Witness.Read{d,ed} Witness.Read{e,ee} Witness.Read{f,ef} Witness.Read{g,eg} Witness.Read{h,eh}:
      +p1 = (position + 1 : U32)
      +p2 = (p1 + 1 : U32)
      +p3 = (p2 + 1 : U32)
      +p4 = (p3 + 1 : U32)
      +p5 = (p4 + 1 : U32)
      +p6 = (p5 + 1 : U32)
      +p7 = (p6 + 1 : U32)
      %Equal.sym(Array<F32> & F32,Array.get(F32,values,position),(values,a),ea) : Gathered<values,Lanes.gather1(~Context,~map,position,context,_)>
      %Equal.sym(Array<F32> & F32,Array.get(F32,values,p1),(values,b),eb) : Gathered<values,Lanes.gather2(~Context,~map,p1,context,a,_)>
      %Equal.sym(Array<F32> & F32,Array.get(F32,values,p2),(values,c),ec) : Gathered<values,Lanes.gather3(~Context,~map,p2,context,a,b,_)>
      %Equal.sym(Array<F32> & F32,Array.get(F32,values,p3),(values,d),ed) : Gathered<values,Lanes.gather4(~Context,~map,p3,context,a,b,c,_)>
      %Equal.sym(Array<F32> & F32,Array.get(F32,values,p4),(values,e),ee) : Gathered<values,Lanes.gather5(~Context,~map,p4,context,a,b,c,d,_)>
      %Equal.sym(Array<F32> & F32,Array.get(F32,values,p5),(values,f),ef) : Gathered<values,Lanes.gather6(~Context,~map,p5,context,a,b,c,d,e,_)>
      %Equal.sym(Array<F32> & F32,Array.get(F32,values,p6),(values,g),eg) : Gathered<values,Lanes.gather7(~Context,~map,p6,context,a,b,c,d,e,f,_)>
      %Equal.sym(Array<F32> & F32,Array.get(F32,values,p7),(values,h),eh) : Gathered<values,Lanes.gathered(~Context,~map,context,a,b,c,d,e,f,g,_)>
      Gathered{Tile.Tile{map(context,a),map(context,b),map(context,c),map(context,d),map(context,e),map(context,f),map(context,g),map(context,h)},{==}}

def gather(~Context: Data,~map: Context -> F32 -> F32,+depth: Nat,-values: Array<F32>,+position: U32,+context: Context,
  +witness: Witness.Witness(F32,depth,values)) -> Gathered<values,Lanes.gather(~Context,~map,values,position,context)>:
  +p1 = (position + 1 : U32)
  +p2 = (p1 + 1 : U32)
  +p3 = (p2 + 1 : U32)
  +p4 = (p3 + 1 : U32)
  +p5 = (p4 + 1 : U32)
  +p6 = (p5 + 1 : U32)
  +p7 = (p6 + 1 : U32)
  gathered(~Context,~map,values,position,context,Witness.read(F32,depth,values,position,witness),Witness.read(F32,depth,values,p1,witness),
    Witness.read(F32,depth,values,p2,witness),Witness.read(F32,depth,values,p3,witness),Witness.read(F32,depth,values,p4,witness),
    Witness.read(F32,depth,values,p5,witness),Witness.read(F32,depth,values,p6,witness),Witness.read(F32,depth,values,p7,witness))

def scatter(+depth: Nat,-values: Array<F32>,+position: U32,-tile: Tile.Tile8,witness: Witness.Witness(F32,depth,values)) ->
  Witness.Witness(F32,depth,Lanes.scatter(values,position,tile)):
  +p1 = (position + 1 : U32)
  +p2 = (p1 + 1 : U32)
  +p3 = (p2 + 1 : U32)
  +p4 = (p3 + 1 : U32)
  +p5 = (p4 + 1 : U32)
  +p6 = (p5 + 1 : U32)
  +p7 = (p6 + 1 : U32)
  -o1 = Array.set(F32,values,position,Tile.lane0(tile))
  -o2 = Array.set(F32,o1,p1,Tile.lane1(tile))
  -o3 = Array.set(F32,o2,p2,Tile.lane2(tile))
  -o4 = Array.set(F32,o3,p3,Tile.lane3(tile))
  -o5 = Array.set(F32,o4,p4,Tile.lane4(tile))
  -o6 = Array.set(F32,o5,p5,Tile.lane5(tile))
  -o7 = Array.set(F32,o6,p6,Tile.lane6(tile))
  witness = Witness.write(F32,depth,values,position,Tile.lane0(tile),witness)
  witness = Witness.write(F32,depth,o1,p1,Tile.lane1(tile),witness)
  witness = Witness.write(F32,depth,o2,p2,Tile.lane2(tile),witness)
  witness = Witness.write(F32,depth,o3,p3,Tile.lane3(tile),witness)
  witness = Witness.write(F32,depth,o4,p4,Tile.lane4(tile),witness)
  witness = Witness.write(F32,depth,o5,p5,Tile.lane5(tile),witness)
  witness = Witness.write(F32,depth,o6,p6,Tile.lane6(tile),witness)
  Witness.write(F32,depth,o7,p7,Tile.lane7(tile),witness)

type Evidence<-Context: Data,-depth: Nat,-state: Cells.State<Array<F32>,Tile.Tile8,Context>> is Data:
  Evidence{-values: Array<F32>,-scratch: Array<Tile.Tile8>,position: U32,slot: U32,context: Context,
    equation: {state == Cells.State{values,scratch,position,slot,context} : Cells.State<Array<F32>,Tile.Tile8,Context>},
    values_witness: Witness.Witness(F32,depth,values),scratch_witness: Witness.Witness(Tile.Tile8,6n,scratch)}

def scattered(~Context: Data,+depth: Nat,-values: Array<F32>,-scratch: Array<Tile.Tile8>,+position: U32,+slot: U32,+context: Context,
  values_witness: Witness.Witness(F32,depth,values),scratch_witness: Witness.Witness(Tile.Tile8,6n,scratch),
  read: Witness.Read<Tile.Tile8,scratch,Array.get(Tile.Tile8,scratch,slot)>) ->
  Evidence<Context,depth,Cells.scattered(~Array<F32>,~Tile.Tile8,~Context,~Lanes.scatter,~8,position,slot,context,values,Array.get(Tile.Tile8,scratch,slot))>:
  match read:
    case Witness.Read{tile,equation}:
      %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,scratch,slot),(scratch,tile),equation) :
        Evidence<Context,depth,Cells.scattered(~Array<F32>,~Tile.Tile8,~Context,~Lanes.scatter,~8,position,slot,context,values,_)>
      Evidence{Lanes.scatter(values,position,tile),scratch,(position + 8 : U32),U32.and((slot + 1 : U32),63),context,{==},
        scatter(depth,values,position,tile,values_witness),scratch_witness}

def stored(~Context: Data,~map: Context -> F32 -> F32,+depth: Nat,-values: Array<F32>,-scratch: Array<Tile.Tile8>,+position: U32,+slot: U32,+context: Context,
  values_witness: Witness.Witness(F32,depth,values),scratch_witness: Witness.Witness(Tile.Tile8,6n,scratch),
  gathered: Gathered<values,Lanes.gather(~Context,~map,values,position,context)>) ->
  Evidence<Context,depth,Cells.stored(~Array<F32>,~Tile.Tile8,~Context,~Lanes.scatter,~8,position,slot,context,scratch,Lanes.gather(~Context,~map,values,position,context))>:
  match gathered:
    case Gathered{cell,equation}:
      +written = Witness.write(Tile.Tile8,6n,scratch,slot,cell,scratch_witness)
      %Equal.sym(Array<F32> & Tile.Tile8,Lanes.gather(~Context,~map,values,position,context),(values,cell),equation) :
        Evidence<Context,depth,Cells.stored(~Array<F32>,~Tile.Tile8,~Context,~Lanes.scatter,~8,position,slot,context,scratch,_)>
      scattered(~Context,depth,values,Array.set(Tile.Tile8,scratch,slot,cell),position,slot,context,values_witness,written,
        Witness.read(Tile.Tile8,6n,Array.set(Tile.Tile8,scratch,slot,cell),slot,written))

def step(~Context: Data,~map: Context -> F32 -> F32,+depth: Nat,-state: Cells.State<Array<F32>,Tile.Tile8,Context>,evidence: Evidence<Context,depth,state>) ->
  Evidence<Context,depth,Cells.step(~Array<F32>,~Tile.Tile8,~Context,~Lanes.gather(~Context,~map),~Lanes.scatter,~8,state)>:
  match evidence:
    case Evidence{values,scratch,+position,+slot,+context,equation,+values_witness,scratch_witness}:
      %Equal.sym(Cells.State<Array<F32>,Tile.Tile8,Context>,state,Cells.State{values,scratch,position,slot,context},equation) :
        Evidence<Context,depth,Cells.step(~Array<F32>,~Tile.Tile8,~Context,~Lanes.gather(~Context,~map),~Lanes.scatter,~8,_)>
      stored(~Context,~map,depth,values,scratch,position,slot,context,values_witness,scratch_witness,
        gather(~Context,~map,depth,values,position,context,values_witness))

def finish(~Context: Data,~map: Context -> F32 -> F32,+depth: Nat,count: Nat,-state: Cells.State<Array<F32>,Tile.Tile8,Context>,evidence: Evidence<Context,depth,state>) ->
  Witness.Witness(F32,depth,Lanes.finish(~Context,~map,count,state)):
  match evidence:
    case Evidence{values,scratch,position,slot,context,equation,values_witness,scratch_witness}:
      %Equal.sym(Cells.State<Array<F32>,Tile.Tile8,Context>,state,Cells.State{values,scratch,position,slot,context},equation) :
        Witness.Witness(F32,depth,Lanes.finish(~Context,~map,count,_))
      TraversalWitness.map_in_place(~F32,~Context,~map,depth,count,position,context,values,values_witness)

def rounds(~Context: Data,~map: Context -> F32 -> F32,+depth: Nat,count: Nat,-state: Cells.State<Array<F32>,Tile.Tile8,Context>,evidence: Evidence<Context,depth,state>) ->
  Witness.Witness(F32,depth,Lanes.rounds(~Context,~map,count,state)):
  match count:
    case 0n: finish(~Context,~map,depth,0n,state,evidence)
    case 1n: finish(~Context,~map,depth,1n,state,evidence)
    case 2n: finish(~Context,~map,depth,2n,state,evidence)
    case 3n: finish(~Context,~map,depth,3n,state,evidence)
    case 4n: finish(~Context,~map,depth,4n,state,evidence)
    case 5n: finish(~Context,~map,depth,5n,state,evidence)
    case 6n: finish(~Context,~map,depth,6n,state,evidence)
    case 7n: finish(~Context,~map,depth,7n,state,evidence)
    case 1n+1n+1n+1n+1n+1n+1n+1n+rest:
      rounds(~Context,~map,depth,rest,Cells.step(~Array<F32>,~Tile.Tile8,~Context,~Lanes.gather(~Context,~map),~Lanes.scatter,~8,state),
        step(~Context,~map,depth,state,evidence))

def map(~Context: Data,~map: Context -> F32 -> F32,+depth: Nat,count: Nat,+position: U32,+context: Context,-values: Array<F32>,witness: Witness.Witness(F32,depth,values)) ->
  Witness.Witness(F32,depth,Lanes.map(~Context,~map,count,position,context,values)):
  rounds(~Context,~map,depth,count,Cells.State{values,Lanes.scratch(),position,0,context},
    Evidence{values,Lanes.scratch(),position,0,context,{==},witness,Witness.fresh(Tile.Tile8,6n,Tile.filled(0.0))})

def map_when(~Context: Data,~operation: Context -> F32 -> F32,covered: Bool,+depth: Nat,count: Nat,+position: U32,+context: Context,-values: Array<F32>,
  witness: Witness.Witness(F32,depth,values)) -> Witness.Witness(F32,depth,Lanes.map_when(~Context,~operation,covered,count,position,context,values)):
  match covered:
    case True{}: map(~Context,~operation,depth,count,position,context,values,witness)
    case False{}: TraversalWitness.map_in_place(~F32,~Context,~operation,depth,count,position,context,values,witness)
