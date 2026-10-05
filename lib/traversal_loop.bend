# One sequential traversal for affine runtime owners and reusable proof states.
# The step is a compile-time template: no closure allocation in the loop.
import Base

def run(~State: Type,~step: State -> State,count: Nat,state: State) -> State:
  match count:
    case 0n: state
    case 1n+rest: run(~State,~step,rest,step(state))

# The same loop handing its final state to finish in the base case. A caller
# that stores its accumulators in finish keeps them out of the loop's return
# value, so an outlined loop never passes accumulators back by value.
def run_then(~State: Type,~Result: Type,~step: State -> State,~finish: State -> Result,count: Nat,state: State) -> Result:
  match count:
    case 0n: finish(state)
    case 1n+rest: run_then(~State,~Result,~step,~finish,rest,step(state))
