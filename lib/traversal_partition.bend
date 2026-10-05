# A work partition is independent of its element representation and ownership.
# Storage remains in the leaf payload; offset/span describe its logical range.
import Base
import ./storage_buffer.bend as Storage

type Tree<ownership,-Block: Kind(ownership),-Details: Data> is Kind(ownership):
  Leaf{data: Block,offset: U32,span: U32,details: Details}
  Fork{left: Tree<ownership,Block,Details>,right: Tree<ownership,Block,Details>}

# Each binary fork gives both children independent owners; its leaf callback
# finishes all local stages before returning. The input split sees the left
# child's range, so it can copy only what that child reads. Representation
# adapters supply direct templates, so Array and certified Buffer share it.
# The last split calls its two leaves directly: a leaf that neither forks nor
# calls a closure is then its own task, which the runtime keeps for the work
# turn; a leaf reached through the forking recursion would run in the grow
# turn on one worker per row.
def run(~Input: Type,~Weights: Type,~Bias: Type,~Output: Type,~Context: Data,
  ~split_input: Context -> U32 -> U32 -> Input -> Input & Input,~clone_weights: Weights -> Weights & Weights,~clone_bias: Bias -> Bias & Bias,
  ~leaf: Context -> U32 -> U32 -> Input -> Weights -> Bias -> Tree<&1,Output,U32>,
  depth: Nat,+context: Context,+offset: U32,+span: U32,input: Input,weights: Weights,bias: Bias) -> Tree<&1,Output,U32>:
  match depth:
    case 0n: leaf(context,offset,span,input,weights,bias)
    case 1n:
      +half = (span / 16 * 8 : U32)
      Storage.bind_pair(~Input,~Input,~Tree<&1,Output,U32>,split_input(context,offset,half,input),left_input => right_input =>
        Storage.bind_pair(~Weights,~Weights,~Tree<&1,Output,U32>,clone_weights(weights),left_weights => right_weights =>
          Storage.bind_pair(~Bias,~Bias,~Tree<&1,Output,U32>,clone_bias(bias),left_bias => right_bias =>
            left right = leaf(context,offset,half,left_input,left_weights,left_bias) leaf(context,(offset + half : U32),(span - half : U32),right_input,right_weights,right_bias)
            Fork{left,right})))
    case 2n+ +rest:
      +half = (span / 16 * 8 : U32)
      Storage.bind_pair(~Input,~Input,~Tree<&1,Output,U32>,split_input(context,offset,half,input),left_input => right_input =>
        Storage.bind_pair(~Weights,~Weights,~Tree<&1,Output,U32>,clone_weights(weights),left_weights => right_weights =>
          Storage.bind_pair(~Bias,~Bias,~Tree<&1,Output,U32>,clone_bias(bias),left_bias => right_bias =>
            left right = run(~Input,~Weights,~Bias,~Output,~Context,~split_input,~clone_weights,~clone_bias,~leaf,1n+rest,context,offset,half,left_input,left_weights,left_bias) run(~Input,~Weights,~Bias,~Output,~Context,~split_input,~clone_weights,~clone_bias,~leaf,1n+rest,context,(offset + half : U32),(span - half : U32),right_input,right_weights,right_bias)
            Fork{left,right})))

def gather(~Output: Type,~copy: U32 -> U32 -> U32 -> U32 -> Output -> Output -> Output,
  partitions: Tree<&1,Output,U32>,output: Output,+positions: U32) -> Output:
  match partitions:
    case Leaf{input,offset,span,rows}: copy(rows,offset,positions,span,input,output)
    case Fork{left,right}: gather(~Output,~copy,right,gather(~Output,~copy,left,output,positions),positions)
