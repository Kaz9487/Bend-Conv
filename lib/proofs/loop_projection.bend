# One induction for every state representation, output index and step count.
import Base
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ../traversal_loop.bend as Loop

def iterate(~State: Data,~step: State -> State,count: Nat,state: State) -> State:
  Loop.run(~State,~step,count,state)

# A single commuting step lifts to every count. RuntimeState may own arrays;
# only the model is duplicated in the proof. No runtime array is cloned.
def iteration_refines_model(
  ~RuntimeState: Type,~ModelState: Data,
  ~runtime_step: RuntimeState -> RuntimeState,~model_step: ModelState -> ModelState,
  ~represent: ModelState -> RuntimeState,
  ~step_matches: @+state: ModelState -> {runtime_step(represent(state)) == represent(model_step(state)) : RuntimeState},
  +count: Nat,+state: ModelState
) -> {Loop.run(~RuntimeState,~runtime_step,count,represent(state)) == represent(Loop.run(~ModelState,~model_step,count,state)) : RuntimeState}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(RuntimeState,runtime_step(represent(state)),represent(model_step(state)),step_matches(state)) :
        {Loop.run(~RuntimeState,~runtime_step,rest,_) == represent(Loop.run(~ModelState,~model_step,rest,model_step(state))) : RuntimeState}
      iteration_refines_model(~RuntimeState,~ModelState,~runtime_step,~model_step,~represent,~step_matches,rest,model_step(state))

# A read-only observation is preserved for every number of iterations.
def preserves_observation(
  ~State: Data,~Observation: Data,~step: State -> State,~observe: State -> Observation,
  ~preserved: @+state: State -> {observe(step(state)) == observe(state) : Observation},
  +count: Nat,+state: State
) -> {observe(iterate(~State,~step,count,state)) == observe(state) : Observation}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %preserved(state) : {observe(iterate(~State,~step,rest,step(state))) == _ : Observation}
      preserves_observation(~State,~Observation,~step,~observe,~preserved,rest,step(state))

def iteration_matches_denotation(
  ~Observation: Data,~State: Data,
  ~Index: Data,
  ~step: State -> State,
  ~project: Index -> State -> Observation,
  ~denote: Index -> Nat -> State -> Observation,
  ~at_zero: @+index: Index -> @+state: State -> {project(index,state) == denote(index,0n,state) : Observation},
  ~after_step: @+index: Index -> @+count: Nat -> @+state: State -> {denote(index,count,step(state)) == denote(index,1n+count,state) : Observation},
  +count: Nat,+index: Index,+state: State
) -> {project(index,iterate(~State,~step,count,state)) == denote(index,count,state) : Observation}:
  match count:
    case 0n: at_zero(index,state)
    case 1n+ +rest:
      %after_step(index,rest,state) : {project(index,iterate(~State,~step,rest,step(state))) == _ : Observation}
      iteration_matches_denotation(~Observation,~State,~Index,~step,~project,~denote,~at_zero,~after_step,rest,index,step(state))

type Measured<-State: Data> is Data:
  Measured{result: State,work: Nat}

def add_work(-State: Data,work: Nat,tail: Measured<State>) -> Measured<State>:
  Measured{result,remaining} = tail
  Measured{result,Nat.add(work,remaining)}

def measured_iteration(~State: Data,~step: State -> State,~cost: State -> Nat,count: Nat,+state: State) -> Measured<State>:
  match count:
    case 0n: Measured{state,0n}
    case 1n+rest:
      add_work(State,cost(state),measured_iteration(~State,~step,~cost,rest,step(state)))

# Concrete adapters must establish cost_per_step; it is not a heap-cost claim.
def iteration_result_and_work(
  ~State: Data,~step: State -> State,~cost: State -> Nat,
  ~width: Nat,
  ~cost_per_step: @+state: State -> {cost(state) == width : Nat},
  +count: Nat,+state: State
) -> {measured_iteration(~State,~step,~cost,count,state) == Measured{iterate(~State,~step,count,state),Nat.mul(count,width)} : Measured<State>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %cost_per_step(state) : {add_work(State,cost(state),measured_iteration(~State,~step,~cost,rest,step(state))) == add_work(State,_,
          Measured{iterate(~State,~step,rest,step(state)),Nat.mul(rest,width)}) : Measured<State>}
      %iteration_result_and_work(~State,~step,~cost,~width,~cost_per_step,rest,step(state)) : {add_work(State,cost(state),
          measured_iteration(~State,~step,~cost,rest,step(state))) == add_work(State,cost(state),_) : Measured<State>}
      {==}

# Machine counters share the same modular progress theorem for any step size.
# Non-wrapping coordinate bounds are separate obligations of the caller.

def indexed_successor(+position: U32,+offset: Nat) ->
  {(position + U32.from_nat(1n+offset) : U32) == ((position + 1 : U32) + U32.from_nat(offset) : U32) : U32}:
  %WordArithmetic.native_add(1n,offset) :
    {(position + _ : U32) == ((position + 1 : U32) + U32.from_nat(offset) : U32) : U32}
  Equal.sym(U32,((position + 1 : U32) + U32.from_nat(offset) : U32),(position + (1 + U32.from_nat(offset) : U32) : U32),
    Arithmetic.add_associative(position,1,U32.from_nat(offset)))
def add_counted_stride(+position: U32,+stride: Nat,+count: Nat) ->
  {((position + U32.from_nat(stride) : U32) + U32.from_nat(Nat.mul(count,stride)) : U32) ==
    (position + U32.from_nat(Nat.mul(1n+count,stride)) : U32) : U32}:
  Equal.trans(U32,((position + U32.from_nat(stride) : U32) + U32.from_nat(Nat.mul(count,stride)) : U32),
    (position + (U32.from_nat(stride) + U32.from_nat(Nat.mul(count,stride)) : U32) : U32),
    (position + U32.from_nat(Nat.mul(1n+count,stride)) : U32),Arithmetic.add_associative(position,U32.from_nat(stride),U32.from_nat(Nat.mul(count,stride))),
    Equal.cong(U32,U32,value => (position + value : U32),(U32.from_nat(stride) + U32.from_nat(Nat.mul(count,stride)) : U32),
      U32.from_nat(Nat.mul(1n+count,stride)),WordArithmetic.native_add(stride,Nat.mul(count,stride))))

def counter_progress(~State: Data,~step: State -> State,~position: State -> U32,~stride: Nat,
  ~advances: @+state: State -> {position(step(state)) == (position(state) + U32.from_nat(stride) : U32) : U32},
  +count: Nat,+state: State) ->
  {position(iterate(~State,~step,count,state)) == (position(state) + U32.from_nat(Nat.mul(count,stride)) : U32) : U32}:
  match count:
    case 0n: Equal.sym(U32,(position(state) + 0 : U32),position(state),Arithmetic.add_zero(position(state)))
    case 1n+ +rest:
      %add_counted_stride(position(state),stride,rest) : {position(iterate(~State,~step,rest,step(state))) == _ : U32}
      %advances(state) : {position(iterate(~State,~step,rest,step(state))) == (_ + U32.from_nat(Nat.mul(rest,stride)) : U32) : U32}
      counter_progress(~State,~step,~position,~stride,~advances,rest,step(state))

# The final step is applied once, to the state the loop ends in.
def run_then_finishes(~State: Type,~Result: Type,~step: State -> State,~finish: State -> Result,count: Nat,-state: State) ->
  {Loop.run_then(~State,~Result,~step,~finish,count,state) == finish(Loop.run(~State,~step,count,state)) : Result}:
  match count:
    case 0n: {==}
    case 1n+rest: run_then_finishes(~State,~Result,~step,~finish,rest,step(state))
