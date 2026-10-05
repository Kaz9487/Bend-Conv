# Project the actual weight traversal onto a contiguous scalar write sequence.
# Bounds and coordinate coverage are discharged at the allocated caller.
import Base
import ../specification.bend as ConvSpec
import ../../lib/kernel_weights.bend as Weights
import ../../lib/kernel_tile.bend as Tile
import ../../lib/math_sequence.bend as Sequence
import ../../lib/traversal_array.bend as Traversal
import ../../lib/traversal_loop.bend as Loop
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/traversal_array_refinement.bend as ArrayLaws
import ../../lib/proofs/kernel_weights_refinement.bend as WeightsLaws
import ../../lib/proofs/kernel_weights_readback.bend as WeightsReadback
import ../../lib/proofs/loop_projection.bend as Projection
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ./tile_array_projection.bend as ArrayProjection
import ./dense_array_initialization.bend as DenseArrayInitialization
import ../../lib/proofs/nat_interval.bend as Interval
import ../../lib/proofs/storage_shape.bend as Shape

def fill(count: Nat,+first: U32,+context: Weights.Context,+input: Tree.Tree<F32>,output: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  match count:
    case 0n: output
    case 1n+rest:
      fill(rest,(first + 1 : U32),context,input,Tree.write(Tile.Tile8,output,Weights.destination(context,first),WeightsReadback.sample(input,context,first)))

def loop_matches_fill(+count: Nat,+first: U32,+context: Weights.Context,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>) ->
  {WeightsLaws.finish(Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.Context>,
    ~ArrayLaws.model_step(~Tree.Tree<F32>,~Tile.Tile8,~Weights.Context,~WeightsReadback.sample,~Weights.destination),count,Traversal.State{input,output,first,context})) ==
    WeightsLaws.Owners{input,fill(count,first,context,input,output)} : WeightsLaws.Owners}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      loop_matches_fill(rest,(first + 1 : U32),context,input,Tree.write(Tile.Tile8,output,Weights.destination(context,first),WeightsReadback.sample(input,context,first)))

def block_matches_fill(+block: U32,+rows: U32,+inner: U32,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>) ->
  {WeightsLaws.block(block,Weights.Layout{rows,inner},WeightsLaws.Owners{input,output}) ==
    WeightsLaws.Owners{input,fill(U32.to_nat(inner),0,Weights.Context{rows,inner,(block * 8 : U32),(block * inner : U32)},input,output)} : WeightsLaws.Owners}:
  loop_matches_fill(U32.to_nat(inner),0,Weights.Context{rows,inner,(block * 8 : U32),(block * inner : U32)},input,output)

def values(count: Nat,+first: U32,+context: Weights.Context,+input: Tree.Tree<F32>,+lane: Sequence.Index(8n)) -> List<&2,F32>:
  match count:
    case 0n: []
    case 1n+rest:
      Tile.element_at(lane,WeightsReadback.sample(input,context,first)) <> values(rest,(first + 1 : U32),context,input,lane)

def values_length(+count: Nat,+first: U32,+context: Weights.Context,+input: Tree.Tree<F32>,+lane: Sequence.Index(8n)) ->
  {ConvSpec.length(values(count,first,context,input,lane)) == count : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %values_length(rest,(first + 1 : U32),context,input,lane) : {1n+ConvSpec.length(values(rest,(first + 1 : U32),context,input,lane)) == 1n+_ : Nat}
      {==}

def destination_next(+context: Weights.Context,+first: U32) ->
  {Weights.destination(context,(first + 1 : U32)) == (Weights.destination(context,first) + 1 : U32) : U32}:
  Weights.Context{rows,inner,row,+destination} = context
  Equal.sym(U32,((destination + first : U32) + 1 : U32),(destination + (first + 1 : U32) : U32),Arithmetic.add_associative(destination,first,1))

def projected_fill(+count: Nat,+first: U32,+context: Weights.Context,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+lane: Sequence.Index(8n)) ->
  {ArrayProjection.project(fill(count,first,context,input,output),lane) ==
    Tree.write_sequence(~F32,values(count,first,context,input,lane),Weights.destination(context,first),ArrayProjection.project(output,lane)) : Tree.Tree<F32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +sample = WeightsReadback.sample(input,context,first)
      +destination = Weights.destination(context,first)
      +updated = Tree.write(Tile.Tile8,output,destination,sample)
      %Equal.sym(Tree.Tree<F32>,ArrayProjection.project(fill(rest,(first + 1 : U32),context,input,updated),lane),
        Tree.write_sequence(~F32,values(rest,(first + 1 : U32),context,input,lane),Weights.destination(context,(first + 1 : U32)),ArrayProjection.project(updated,lane)),
        projected_fill(rest,(first + 1 : U32),context,input,updated,lane)) :
        {_ == Tree.write_sequence(~F32,values(rest,(first + 1 : U32),context,input,lane),(destination + 1 : U32),
          Tree.write(F32,ArrayProjection.project(output,lane),destination,Tile.element_at(lane,sample))) : Tree.Tree<F32>}
      %Equal.sym(U32,Weights.destination(context,(first + 1 : U32)),(destination + 1 : U32),destination_next(context,first)) :
        {Tree.write_sequence(~F32,values(rest,(first + 1 : U32),context,input,lane),_,ArrayProjection.project(updated,lane)) ==
          Tree.write_sequence(~F32,values(rest,(first + 1 : U32),context,input,lane),(destination + 1 : U32),
            Tree.write(F32,ArrayProjection.project(output,lane),destination,Tile.element_at(lane,sample))) : Tree.Tree<F32>}
      %Equal.sym(Tree.Tree<F32>,ArrayProjection.project(updated,lane),Tree.write(F32,ArrayProjection.project(output,lane),destination,Tile.element_at(lane,sample)),
        ArrayProjection.write_projected(output,lane,destination,sample)) :
        {Tree.write_sequence(~F32,values(rest,(first + 1 : U32),context,input,lane),(destination + 1 : U32),_) ==
          Tree.write_sequence(~F32,values(rest,(first + 1 : U32),context,input,lane),(destination + 1 : U32),
            Tree.write(F32,ArrayProjection.project(output,lane),destination,Tile.element_at(lane,sample))) : Tree.Tree<F32>}
      {==}

def value_at(+count: Nat,+first: U32,+context: Weights.Context,+input: Tree.Tree<F32>,+lane: Sequence.Index(8n),+index: Nat,
  inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {ConvSpec.at(index,values(count,first,context,input,lane)) == Tile.element_at(lane,WeightsReadback.sample(input,context,(first + U32.from_nat(index) : U32))) : F32}:
  match count:
    case 0n:
      Empty.absurd({ConvSpec.at(index,[]) == Tile.element_at(lane,WeightsReadback.sample(input,context,(first + U32.from_nat(index) : U32))) : F32},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Division.lt_zero(index)),inside)))
    case 1n+ +rest:
      match index:
        case 0n:
          %Equal.sym(U32,(first + 0 : U32),first,Arithmetic.add_zero(first)) :
            {Tile.element_at(lane,WeightsReadback.sample(input,context,first)) == Tile.element_at(lane,WeightsReadback.sample(input,context,_)) : F32}
          {==}
        case 1n+ +previous:
          %Equal.sym(U32,(first + U32.from_nat(1n+previous) : U32),((first + 1 : U32) + U32.from_nat(previous) : U32),Projection.indexed_successor(first,previous)) :
            {ConvSpec.at(previous,values(rest,(first + 1 : U32),context,input,lane)) == Tile.element_at(lane,WeightsReadback.sample(input,context,_)) : F32}
          value_at(rest,(first + 1 : U32),context,input,lane,previous,inside)

def balanced(+count: Nat,+first: U32,+context: Weights.Context,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+lane: Sequence.Index(8n),+depth: Nat,
  shape: Shape.Balanced(F32,depth,ArrayProjection.project(output,lane))) ->
  Shape.Balanced(F32,depth,ArrayProjection.project(fill(count,first,context,input,output),lane)):
  %Equal.sym(Tree.Tree<F32>,ArrayProjection.project(fill(count,first,context,input,output),lane),
    Tree.write_sequence(~F32,values(count,first,context,input,lane),Weights.destination(context,first),ArrayProjection.project(output,lane)),
    projected_fill(count,first,context,input,output,lane)) : Shape.Balanced(F32,depth,_)
  DenseArrayInitialization.balanced(values(count,first,context,input,lane),Weights.destination(context,first),ArrayProjection.project(output,lane),depth,shape)

def capacity(+count: Nat,+first: U32,+context: Weights.Context,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+lane: Sequence.Index(8n),+depth: Nat,
  shape: Shape.Balanced(F32,depth,ArrayProjection.project(output,lane))) ->
  {Tree.capacity(F32,ArrayProjection.project(fill(count,first,context,input,output),lane)) == Tree.capacity(F32,ArrayProjection.project(output,lane)) : U32}:
  %Equal.sym(Tree.Tree<F32>,ArrayProjection.project(fill(count,first,context,input,output),lane),
    Tree.write_sequence(~F32,values(count,first,context,input,lane),Weights.destination(context,first),ArrayProjection.project(output,lane)),
    projected_fill(count,first,context,input,output,lane)) : {Tree.capacity(F32,_) == Tree.capacity(F32,ArrayProjection.project(output,lane)) : U32}
  DenseArrayInitialization.capacity(values(count,first,context,input,lane),Weights.destination(context,first),ArrayProjection.project(output,lane),depth,shape)

def readback(+count: Nat,+first: U32,+context: Weights.Context,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+lane: Sequence.Index(8n),+depth: Nat,+query: U32,+index: Nat,
  +shape: Shape.Balanced(F32,depth,ArrayProjection.project(output,lane)),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +query_exact: {U32.to_nat(query) == Nat.add(U32.to_nat(Weights.destination(context,first)),index) : Nat},
  +room: {Nat.is_le(Nat.add(U32.to_nat(Weights.destination(context,first)),count),U32.to_nat(Tree.capacity(F32,ArrayProjection.project(output,lane)))) == True{} : Bool},
  +inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Tile.element_at(lane,Tree.read(Tile.Tile8,fill(count,first,context,input,output),query)) ==
    Tile.element_at(lane,WeightsReadback.sample(input,context,(first + U32.from_nat(index) : U32))) : F32}:
  %ArrayProjection.read_projected(fill(count,first,context,input,output),lane,query) :
    {_ == Tile.element_at(lane,WeightsReadback.sample(input,context,(first + U32.from_nat(index) : U32))) : F32}
  %Equal.sym(Tree.Tree<F32>,ArrayProjection.project(fill(count,first,context,input,output),lane),
    Tree.write_sequence(~F32,values(count,first,context,input,lane),Weights.destination(context,first),ArrayProjection.project(output,lane)),
    projected_fill(count,first,context,input,output,lane)) :
    {Tree.read(F32,_,query) == Tile.element_at(lane,WeightsReadback.sample(input,context,(first + U32.from_nat(index) : U32))) : F32}
  %value_at(count,first,context,input,lane,index,inside) :
    {Tree.read(F32,Tree.write_sequence(~F32,values(count,first,context,input,lane),Weights.destination(context,first),ArrayProjection.project(output,lane)),query) == _ : F32}
  DenseArrayInitialization.read_filled_range(values(count,first,context,input,lane),Weights.destination(context,first),ArrayProjection.project(output,lane),depth,query,index,shape,depth_ok,query_exact,
    %Equal.sym(Nat,ConvSpec.length(values(count,first,context,input,lane)),count,values_length(count,first,context,input,lane)) :
      {Nat.is_le(Nat.add(U32.to_nat(Weights.destination(context,first)),_),U32.to_nat(Tree.capacity(F32,ArrayProjection.project(output,lane)))) == True{} : Bool}
    room,
    %Equal.sym(Nat,ConvSpec.length(values(count,first,context,input,lane)),count,values_length(count,first,context,input,lane)) : {Nat.is_lt(index,_) == True{} : Bool}
    inside)

def preserves_outside(+count: Nat,+first: U32,+context: Weights.Context,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+lane: Sequence.Index(8n),+depth: Nat,+query: U32,
  +shape: Shape.Balanced(F32,depth,ArrayProjection.project(output,lane)),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(Weights.destination(context,first)),count),U32.to_nat(Tree.capacity(F32,ArrayProjection.project(output,lane)))) == True{} : Bool},
  +inside: {U32.is_lt(query,Tree.capacity(F32,ArrayProjection.project(output,lane))) == True{} : Bool},
  outside: Interval.Outside<U32.to_nat(Weights.destination(context,first)),count,U32.to_nat(query)>) ->
  {Tile.element_at(lane,Tree.read(Tile.Tile8,fill(count,first,context,input,output),query)) == Tile.element_at(lane,Tree.read(Tile.Tile8,output,query)) : F32}:
  %ArrayProjection.read_projected(output,lane,query) :
    {Tile.element_at(lane,Tree.read(Tile.Tile8,fill(count,first,context,input,output),query)) == _ : F32}
  %ArrayProjection.read_projected(fill(count,first,context,input,output),lane,query) :
    {_ == Tree.read(F32,ArrayProjection.project(output,lane),query) : F32}
  %Equal.sym(Tree.Tree<F32>,ArrayProjection.project(fill(count,first,context,input,output),lane),
    Tree.write_sequence(~F32,values(count,first,context,input,lane),Weights.destination(context,first),ArrayProjection.project(output,lane)),
    projected_fill(count,first,context,input,output,lane)) :
    {Tree.read(F32,_,query) == Tree.read(F32,ArrayProjection.project(output,lane),query) : F32}
  DenseArrayInitialization.read_filled_outside(values(count,first,context,input,lane),Weights.destination(context,first),ArrayProjection.project(output,lane),depth,query,shape,depth_ok,
    %Equal.sym(Nat,ConvSpec.length(values(count,first,context,input,lane)),count,values_length(count,first,context,input,lane)) :
      {Nat.is_le(Nat.add(U32.to_nat(Weights.destination(context,first)),_),U32.to_nat(Tree.capacity(F32,ArrayProjection.project(output,lane)))) == True{} : Bool}
    room,inside,
    %Equal.sym(Nat,ConvSpec.length(values(count,first,context,input,lane)),count,values_length(count,first,context,input,lane)) :
      Interval.Outside<U32.to_nat(Weights.destination(context,first)),_,U32.to_nat(query)>
    outside)
