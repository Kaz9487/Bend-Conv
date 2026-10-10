# The live certificate of Lanes.map_when, and its refinement: with the cover
# test of the certificate's depth it is Traversal.map_in_place.
import Base
import ../kernel_tile.bend as Tile
import ./traversal_witness.bend as TraversalWitness
import ../traversal_lanes.bend as Lanes
import ../traversal_array.bend as Traversal
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./storage_shape.bend as Shape
import ./traversal_lanes_witness.bend as LanesWitness
import ./traversal_lanes_refinement.bend as LanesLaws
import ./boolean_logic.bend as Boolean
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./storage_refinement.bend as StorageLaws
import ./storage_witness.bend as Witness

def map_when(~Context: Data,~operation: Context -> F32 -> F32,+covered: Bool,+count: Nat,+position: U32,+context: Context,-values: Array<F32>,
  certificate: Certificate.Certificate<F32,values>) -> Certificate.Certificate<F32,Lanes.map_when(~Context,~operation,covered,count,position,context,values)>:
  Certificate.Certificate{+depth,equation} = certificate
  Certificate.Certificate{depth,
    %Certificate.restored(F32,depth,Lanes.map_when(~Context,~operation,covered,count,position,context,values),
      LanesWitness.map_when(~Context,~operation,covered,depth,count,position,context,values,Certificate.witness_at(F32,depth,values,equation))) :
      {_ == Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,Lanes.map_when(~Context,~operation,covered,count,position,context,values)))) : Array<F32>}
    {==}}

# The cover test of a balanced tree gives the bounds of the refinement.
def refines(~Context: Data,~operation: Context -> F32 -> F32,+covered: Bool,+count: Nat,+position: U32,+context: Context,+values: Tree.Tree<F32>,+depth: Nat,
  +balanced: Shape.Balanced(F32,depth,values),+guard: {Lanes.covered(depth,position,count) == covered : Bool}) ->
  {Lanes.map_when(~Context,~operation,covered,count,position,context,Tree.pack(F32,values)) ==
    Traversal.map_in_place(~F32,~Context,~operation,count,position,context,Tree.pack(F32,values)) : Array<F32>}:
  match covered:
    case False{}: {==}
    case True{}:
      +room = Nat.is_le(Nat.add(U32.to_nat(position),count),U32.to_nat(U32.shln(1,depth)))
      %Arithmetic.native_roundtrip(position) :
        {Lanes.map(~Context,~operation,count,_,context,Tree.pack(F32,values)) ==
          Traversal.map_in_place(~F32,~Context,~operation,count,_,context,Tree.pack(F32,values)) : Array<F32>}
      LanesLaws.map(~Context,~operation,count,U32.to_nat(position),context,values,depth,balanced,Boolean.left(Nat.is_le(depth,30n),room,guard),
        Boolean.right(Nat.is_le(depth,30n),room,guard))

def observed(~Context: Data,~operation: Context -> F32 -> F32,+count: Nat,+position: U32,+context: Context,-values: Array<F32>,+depth: Nat,
  balanced: Shape.Balanced(F32,depth,Tree.reflect(F32,values)),observation: StorageLaws.Observation<F32,values>) ->
  {Lanes.map_when(~Context,~operation,Lanes.covered(depth,position,count),count,position,context,values) ==
    Traversal.map_in_place(~F32,~Context,~operation,count,position,context,values) : Array<F32>}:
  StorageLaws.Observation{+tree,packed,reflected} = observation
  %packed :
    {Lanes.map_when(~Context,~operation,Lanes.covered(depth,position,count),count,position,context,_) ==
      Traversal.map_in_place(~F32,~Context,~operation,count,position,context,_) : Array<F32>}
  refines(~Context,~operation,Lanes.covered(depth,position,count),count,position,context,tree,depth,
    %Equal.sym(Tree.Tree<F32>,tree,Tree.reflect(F32,values),reflected) : Shape.Balanced(F32,depth,_) balanced,{==})

# On any certified array the map through tiles is the ordered map.
def certified(~Context: Data,~operation: Context -> F32 -> F32,+count: Nat,+position: U32,+context: Context,values: Array<F32>,+depth: Nat,
  equation: {values == Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,values))) : Array<F32>}) ->
  {Lanes.map_when(~Context,~operation,Lanes.covered(depth,position,count),count,position,context,values) ==
    Traversal.map_in_place(~F32,~Context,~operation,count,position,context,values) : Array<F32>}:
  observed(~Context,~operation,count,position,context,values,depth,Witness.balanced(F32,depth,values,Certificate.witness_at(F32,depth,values,equation)),
    StorageLaws.observe(F32,values))

# Reuse the existing gather/scatter witnesses when a cell owns certified
# buffers rather than passing flat certificates outside the whole traversal.
def gathered_owner(~Context: Data,~map: Context -> F32 -> F32,-values: Array<F32>,position: U32,context: Context,
  evidence: LanesWitness.Gathered<values,Lanes.gather(~Context,~map,values,position,context)>) ->
  {TraversalWitness.first(~Array<F32>,~Tile.Tile8,Lanes.gather(~Context,~map,values,position,context)) == values : Array<F32>}:
  LanesWitness.Gathered{cell,equation} = evidence
  %Equal.sym(Array<F32> & Tile.Tile8,Lanes.gather(~Context,~map,values,position,context),(values,cell),equation) :
    {TraversalWitness.first(~Array<F32>,~Tile.Tile8,_) == values : Array<F32>}
  {==}

def gather(~Context: Data,~map: Context -> F32 -> F32,-values: Array<F32>,+position: U32,+context: Context,
  certificate: Certificate.Certificate<F32,values>) ->
  Certificate.Certificate<F32,TraversalWitness.first(~Array<F32>,~Tile.Tile8,Lanes.gather(~Context,~map,values,position,context))>:
  Certificate.Certificate{+depth,+equation} = certificate
  %Equal.sym(Array<F32>,TraversalWitness.first(~Array<F32>,~Tile.Tile8,Lanes.gather(~Context,~map,values,position,context)),values,
    gathered_owner(~Context,~map,values,position,context,
      LanesWitness.gather(~Context,~map,depth,values,position,context,Certificate.witness_at(F32,depth,values,equation)))) :
    Certificate.Certificate<F32,_>
  Certificate.Certificate{depth,equation}

def scatter(-values: Array<F32>,+position: U32,-tile: Tile.Tile8,certificate: Certificate.Certificate<F32,values>) ->
  Certificate.Certificate<F32,Lanes.scatter(values,position,tile)>:
  Certificate.Certificate{+depth,equation} = certificate
  Certificate.Certificate{depth,
    %Certificate.restored(F32,depth,Lanes.scatter(values,position,tile),
      LanesWitness.scatter(depth,values,position,tile,Certificate.witness_at(F32,depth,values,equation))) :
      {_ == Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,Lanes.scatter(values,position,tile)))) : Array<F32>}
    {==}}
