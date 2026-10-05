import Base
import ../kernel_block.bend as Block
import ../kernel_tile.bend as Tile
import ./traversal_witness.bend as TraversalWitness

# Pair composition consumes already established observations. Context and
# values may be erased, so a cached head need not become a live proof field.
def pair(~State: Type,~Context: Data,~Element: Data,~read: State -> Context -> Nat -> State & Element,~stride: Nat,
  -state: State,-context: Context,+index: Nat,
  first: TraversalWitness.Sample<State,Element,state,read(state,context,index)>,
  second: TraversalWitness.Sample<State,Element,state,read(state,context,Nat.add(index,stride))>) ->
  TraversalWitness.Sample<State,Block.Twin<Element>,state,Block.read_pair(~State,~Context,~Element,~read,~stride,state,context,index)>:
  TraversalWitness.Sample{left,left_equation} = first
  TraversalWitness.Sample{right,right_equation} = second
  %Equal.sym(State & Element,read(state,context,index),(state,left),left_equation) :
    TraversalWitness.Sample<State,Block.Twin<Element>,state,Block.read_second(~State,~Context,~Element,~Nat,~read,_,context,Nat.add(index,stride))>
  %Equal.sym(State & Element,read(state,context,Nat.add(index,stride)),(state,right),right_equation) :
    TraversalWitness.Sample<State,Block.Twin<Element>,state,Block.finish_pair(~State,~Element,left,_)>
  TraversalWitness.Sample{Block.Twin{left,right},{==}}

def finish(~State: Type,-state: State,-result: State & Block.Twin<Block.Twin<Block.Twin<F32>>>,
  evidence: TraversalWitness.Sample<State,Block.Twin<Block.Twin<Block.Twin<F32>>>,state,result>) ->
  TraversalWitness.Sample<State,Tile.Tile8,state,Tile.finish_pairs(~State,result)>:
  TraversalWitness.Sample{value,equation} = evidence
  %Equal.sym(State & Block.Twin<Block.Twin<Block.Twin<F32>>>,result,(state,value),equation) :
    TraversalWitness.Sample<State,Tile.Tile8,state,Tile.finish_pairs(~State,_)>
  TraversalWitness.Sample{Tile.pairs_value(value),{==}}

def with_head(~State: Type,~Context: Data,~Evidence: @-state: State -> Data,
  ~read: State -> Context -> Nat -> State & F32,
  ~read_preserves: @-state: State -> @context: Context -> @index: Nat -> Evidence(state) -> TraversalWitness.Sample<State,F32,state,read(state,context,index)>,
  -state: State,+context: Context,-head: F32,+index: Nat,evidence: Evidence(state)) ->
  TraversalWitness.Sample<State,F32,state,Tile.read_with_head(~State,~Context,~read,state,Tile.HeadContext{head,context},index)>:
  match index:
    case 0n: TraversalWitness.Sample{head,{==}}
    case 1n+rest: read_preserves(state,context,1n+rest,evidence)

def pair_head(~State: Type,~Context: Data,~Evidence: @-state: State -> Data,
  ~read: State -> Context -> Nat -> State & F32,
  ~read_preserves: @-state: State -> @context: Context -> @index: Nat -> Evidence(state) -> TraversalWitness.Sample<State,F32,state,read(state,context,index)>,
  -state: State,+context: Context,-head: F32,+index: Nat,+evidence: Evidence(state)) ->
  TraversalWitness.Sample<State,Block.Twin<F32>,state,
    Block.read_pair(~State,~Tile.HeadContext<Context>,~F32,~Tile.read_with_head(~State,~Context,~read),~1n,state,Tile.HeadContext{head,context},index)>:
  pair(~State,~Tile.HeadContext<Context>,~F32,~Tile.read_with_head(~State,~Context,~read),~1n,state,Tile.HeadContext{head,context},index,
    with_head(~State,~Context,~Evidence,~read,~read_preserves,state,context,head,index,evidence),
    with_head(~State,~Context,~Evidence,~read,~read_preserves,state,context,head,Nat.add(index,1n),evidence))

def four_head(~State: Type,~Context: Data,~Evidence: @-state: State -> Data,
  ~read: State -> Context -> Nat -> State & F32,
  ~read_preserves: @-state: State -> @context: Context -> @index: Nat -> Evidence(state) -> TraversalWitness.Sample<State,F32,state,read(state,context,index)>,
  -state: State,+context: Context,-head: F32,+index: Nat,+evidence: Evidence(state)) ->
  TraversalWitness.Sample<State,Block.Twin<Block.Twin<F32>>,state,
    Block.read_pair(~State,~Tile.HeadContext<Context>,~Block.Twin<F32>,
      ~Block.read_pair(~State,~Tile.HeadContext<Context>,~F32,~Tile.read_with_head(~State,~Context,~read),~1n),~2n,state,Tile.HeadContext{head,context},index)>:
  pair(~State,~Tile.HeadContext<Context>,~Block.Twin<F32>,
    ~Block.read_pair(~State,~Tile.HeadContext<Context>,~F32,~Tile.read_with_head(~State,~Context,~read),~1n),~2n,state,Tile.HeadContext{head,context},index,
    pair_head(~State,~Context,~Evidence,~read,~read_preserves,state,context,head,index,evidence),
    pair_head(~State,~Context,~Evidence,~read,~read_preserves,state,context,head,Nat.add(index,2n),evidence))

def eight_head(~State: Type,~Context: Data,~Evidence: @-state: State -> Data,
  ~read: State -> Context -> Nat -> State & F32,
  ~read_preserves: @-state: State -> @context: Context -> @index: Nat -> Evidence(state) -> TraversalWitness.Sample<State,F32,state,read(state,context,index)>,
  -state: State,+context: Context,-head: F32,+evidence: Evidence(state)) ->
  TraversalWitness.Sample<State,Block.Twin<Block.Twin<Block.Twin<F32>>>,state,
    Block.read_pair(~State,~Tile.HeadContext<Context>,~Block.Twin<Block.Twin<F32>>,
      ~Block.read_pair(~State,~Tile.HeadContext<Context>,~Block.Twin<F32>,
        ~Block.read_pair(~State,~Tile.HeadContext<Context>,~F32,~Tile.read_with_head(~State,~Context,~read),~1n),~2n),~4n,state,Tile.HeadContext{head,context},0n)>:
  pair(~State,~Tile.HeadContext<Context>,~Block.Twin<Block.Twin<F32>>,
    ~Block.read_pair(~State,~Tile.HeadContext<Context>,~Block.Twin<F32>,
      ~Block.read_pair(~State,~Tile.HeadContext<Context>,~F32,~Tile.read_with_head(~State,~Context,~read),~1n),~2n),~4n,state,Tile.HeadContext{head,context},0n,
    four_head(~State,~Context,~Evidence,~read,~read_preserves,state,context,head,0n,evidence),
    four_head(~State,~Context,~Evidence,~read,~read_preserves,state,context,head,4n,evidence))

def tail(~State: Type,~Context: Data,~Evidence: @-state: State -> Data,
  ~read: State -> Context -> Nat -> State & F32,
  ~read_preserves: @-state: State -> @context: Context -> @index: Nat -> Evidence(state) -> TraversalWitness.Sample<State,F32,state,read(state,context,index)>,
  -state: State,+context: Context,-first: State & F32,evidence: Evidence(state),sample: TraversalWitness.Sample<State,F32,state,first>) ->
  TraversalWitness.Sample<State,Tile.Tile8,state,Tile.read_tail(~State,~Context,~read,first,context)>:
  TraversalWitness.Sample{head,equation} = sample
  %Equal.sym(State & F32,first,(state,head),equation) :
    TraversalWitness.Sample<State,Tile.Tile8,state,Tile.read_tail(~State,~Context,~read,_,context)>
  finish(~State,state,
    Block.read_pair(~State,~Tile.HeadContext<Context>,~Block.Twin<Block.Twin<F32>>,
      ~Block.read_pair(~State,~Tile.HeadContext<Context>,~Block.Twin<F32>,
        ~Block.read_pair(~State,~Tile.HeadContext<Context>,~F32,~Tile.read_with_head(~State,~Context,~read),~1n),~2n),~4n,state,Tile.HeadContext{head,context},0n),
    eight_head(~State,~Context,~Evidence,~read,~read_preserves,state,context,head,evidence))
