# One storage refinement for the common fork schedule. Leaf and input-split
# obligations are supplied by actual kernel refinements; Array.clone of the
# replicated weight and bias owners is discharged here.
import Base
import ../traversal_partition.bend as Partition
import ../storage_buffer.bend as Storage
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws

def represent(~Output: Data,tree: Partition.Tree<&2,Tree.Tree<Output>,U32>) -> Partition.Tree<&1,Array<Output>,U32>:
  match tree:
    case Partition.Leaf{data,offset,span,details}: Partition.Leaf{Tree.pack(Output,data),offset,span,details}
    case Partition.Fork{left,right}: Partition.Fork{represent(~Output,left),represent(~Output,right)}

# A model split names both children; the native split returns their owners.
type Split<-Input: Data> is Data:
  Split{left: Input,right: Input}

def first(~Input: Data,pair: Split<Input>) -> Input:
  Split{left,right} = pair
  left

def second(~Input: Data,pair: Split<Input>) -> Input:
  Split{left,right} = pair
  right

def split_eta(~Input: Data,+pair: Split<Input>) -> {pair == Split{first(~Input,pair),second(~Input,pair)} : Split<Input>}:
  Split{left,right} = pair
  {==}

def partition(~Input: Data,~Weight: Data,~Bias: Data,~Output: Data,~Context: Data,
  ~split: Context -> U32 -> U32 -> Input -> Split<Input>,
  ~leaf: Context -> U32 -> U32 -> Input -> Tree.Tree<Weight> -> Tree.Tree<Bias> -> Partition.Tree<&2,Tree.Tree<Output>,U32>,
  depth: Nat,+context: Context,+offset: U32,+span: U32,+input: Input,+weights: Tree.Tree<Weight>,+bias: Tree.Tree<Bias>) -> Partition.Tree<&2,Tree.Tree<Output>,U32>:
  match depth:
    case 0n: leaf(context,offset,span,input,weights,bias)
    case 1n+ +rest:
      +half = (span / 16 * 8 : U32)
      +inputs = split(context,offset,half,input)
      Partition.Fork{partition(~Input,~Weight,~Bias,~Output,~Context,~split,~leaf,rest,context,offset,half,first(~Input,inputs),weights,bias),
        partition(~Input,~Weight,~Bias,~Output,~Context,~split,~leaf,rest,context,(offset + half : U32),(span - half : U32),second(~Input,inputs),weights,bias)}

def native_children(~Native: Type,~Weight: Data,~Bias: Data,~Output: Data,~Context: Data,
  ~split: Context -> U32 -> U32 -> Native -> Native & Native,
  ~leaf: Context -> U32 -> U32 -> Native -> Array<Weight> -> Array<Bias> -> Partition.Tree<&1,Array<Output>,U32>,
  +depth: Nat,+context: Context,+offset: U32,+span: U32,inputs: Native & Native,weights: Array<Weight> & Array<Weight>,bias: Array<Bias> & Array<Bias>) -> Partition.Tree<&1,Array<Output>,U32>:
  +half = (span / 16 * 8 : U32)
  Storage.bind_pair(~Native,~Native,~Partition.Tree<&1,Array<Output>,U32>,inputs,left_input => right_input =>
    Storage.bind_pair(~Array<Weight>,~Array<Weight>,~Partition.Tree<&1,Array<Output>,U32>,weights,left_weights => right_weights =>
      Storage.bind_pair(~Array<Bias>,~Array<Bias>,~Partition.Tree<&1,Array<Output>,U32>,bias,left_bias => right_bias =>
        left right = Partition.run(~Native,~Array<Weight>,~Array<Bias>,~Array<Output>,~Context,~split,~Array.clone(Weight),~Array.clone(Bias),~leaf,
          depth,context,offset,half,left_input,left_weights,left_bias) Partition.run(~Native,~Array<Weight>,~Array<Bias>,~Array<Output>,~Context,~split,~Array.clone(Weight),~Array.clone(Bias),~leaf,
          depth,context,(offset + half : U32),(span - half : U32),right_input,right_weights,right_bias)
        Partition.Fork{left,right})))

# One split, whatever lies below it: the last split calls its leaves directly
# and an earlier one recurses, and both equal the two child runs.
def step(~Native: Type,~Weight: Data,~Bias: Data,~Output: Data,~Context: Data,
  ~split: Context -> U32 -> U32 -> Native -> Native & Native,
  ~leaf: Context -> U32 -> U32 -> Native -> Array<Weight> -> Array<Bias> -> Partition.Tree<&1,Array<Output>,U32>,
  +rest: Nat,+context: Context,+offset: U32,+span: U32,-input: Native,-weights: Array<Weight>,-bias: Array<Bias>) ->
  {Partition.run(~Native,~Array<Weight>,~Array<Bias>,~Array<Output>,~Context,~split,~Array.clone(Weight),~Array.clone(Bias),~leaf,1n+rest,context,offset,span,input,weights,bias) ==
    native_children(~Native,~Weight,~Bias,~Output,~Context,~split,~leaf,rest,context,offset,span,split(context,offset,(span / 16 * 8 : U32),input),Array.clone(Weight,weights),Array.clone(Bias,bias)) :
    Partition.Tree<&1,Array<Output>,U32>}:
  match rest:
    case 0n: {==}
    case 1n+more: {==}

def represent_pair(~Native: Type,~Input: Data,~represent_input: Input -> Native,pair: Split<Input>) -> Native & Native:
  Split{left,right} = pair
  (represent_input(left),represent_input(right))

def run_matches(~Native: Type,~Input: Data,~Weight: Data,~Bias: Data,~Output: Data,~Context: Data,
  ~represent_input: Input -> Native,
  ~native_split: Context -> U32 -> U32 -> Native -> Native & Native,
  ~model_split: Context -> U32 -> U32 -> Input -> Split<Input>,
  ~native_leaf: Context -> U32 -> U32 -> Native -> Array<Weight> -> Array<Bias> -> Partition.Tree<&1,Array<Output>,U32>,
  ~model_leaf: Context -> U32 -> U32 -> Input -> Tree.Tree<Weight> -> Tree.Tree<Bias> -> Partition.Tree<&2,Tree.Tree<Output>,U32>,
  ~split_matches: @+context: Context -> @+offset: U32 -> @+half: U32 -> @+input: Input ->
    {native_split(context,offset,half,represent_input(input)) == represent_pair(~Native,~Input,~represent_input,model_split(context,offset,half,input)) : Native & Native},
  ~leaf_matches: @+context: Context -> @+offset: U32 -> @+span: U32 -> @+input: Input -> @+weights: Tree.Tree<Weight> -> @+bias: Tree.Tree<Bias> ->
    {native_leaf(context,offset,span,represent_input(input),Tree.pack(Weight,weights),Tree.pack(Bias,bias)) == represent(~Output,model_leaf(context,offset,span,input,weights,bias)) : Partition.Tree<&1,Array<Output>,U32>},
  +depth: Nat,+context: Context,+offset: U32,+span: U32,+input: Input,+weights: Tree.Tree<Weight>,+bias: Tree.Tree<Bias>) ->
  {Partition.run(~Native,~Array<Weight>,~Array<Bias>,~Array<Output>,~Context,~native_split,~Array.clone(Weight),~Array.clone(Bias),~native_leaf,
      depth,context,offset,span,represent_input(input),Tree.pack(Weight,weights),Tree.pack(Bias,bias)) ==
    represent(~Output,partition(~Input,~Weight,~Bias,~Output,~Context,~model_split,~model_leaf,depth,context,offset,span,input,weights,bias)) : Partition.Tree<&1,Array<Output>,U32>}:
  match depth:
    case 0n: leaf_matches(context,offset,span,input,weights,bias)
    case 1n+ +rest:
      +half = (span / 16 * 8 : U32)
      +inputs = model_split(context,offset,half,input)
      +left_input = first(~Input,inputs)
      +right_input = second(~Input,inputs)
      +left = partition(~Input,~Weight,~Bias,~Output,~Context,~model_split,~model_leaf,rest,context,offset,half,left_input,weights,bias)
      +right = partition(~Input,~Weight,~Bias,~Output,~Context,~model_split,~model_leaf,rest,context,(offset + half : U32),(span - half : U32),right_input,weights,bias)
      %Equal.sym(Partition.Tree<&1,Array<Output>,U32>,
        Partition.run(~Native,~Array<Weight>,~Array<Bias>,~Array<Output>,~Context,~native_split,~Array.clone(Weight),~Array.clone(Bias),~native_leaf,
          1n+rest,context,offset,span,represent_input(input),Tree.pack(Weight,weights),Tree.pack(Bias,bias)),
        native_children(~Native,~Weight,~Bias,~Output,~Context,~native_split,~native_leaf,rest,context,offset,span,native_split(context,offset,half,represent_input(input)),
          Array.clone(Weight,Tree.pack(Weight,weights)),Array.clone(Bias,Tree.pack(Bias,bias))),
        step(~Native,~Weight,~Bias,~Output,~Context,~native_split,~native_leaf,rest,context,offset,span,represent_input(input),Tree.pack(Weight,weights),Tree.pack(Bias,bias))) :
        {_ == represent(~Output,Partition.Fork{left,right}) : Partition.Tree<&1,Array<Output>,U32>}
      %Equal.sym(Native & Native,native_split(context,offset,half,represent_input(input)),represent_pair(~Native,~Input,~represent_input,inputs),
        split_matches(context,offset,half,input)) :
        {native_children(~Native,~Weight,~Bias,~Output,~Context,~native_split,~native_leaf,rest,context,offset,span,_,Array.clone(Weight,Tree.pack(Weight,weights)),Array.clone(Bias,Tree.pack(Bias,bias))) == represent(~Output,Partition.Fork{left,right}) : Partition.Tree<&1,Array<Output>,U32>}
      %Equal.sym(Split<Input>,inputs,Split{left_input,right_input},split_eta(~Input,inputs)) :
        {native_children(~Native,~Weight,~Bias,~Output,~Context,~native_split,~native_leaf,rest,context,offset,span,represent_pair(~Native,~Input,~represent_input,_),Array.clone(Weight,Tree.pack(Weight,weights)),Array.clone(Bias,Tree.pack(Bias,bias))) == represent(~Output,Partition.Fork{left,right}) : Partition.Tree<&1,Array<Output>,U32>}
      %Equal.sym(Array<Weight> & Array<Weight>,Array.clone(Weight,Tree.pack(Weight,weights)),(Tree.pack(Weight,weights),Tree.pack(Weight,weights)),StorageLaws.clone(Weight,weights)) :
        {native_children(~Native,~Weight,~Bias,~Output,~Context,~native_split,~native_leaf,rest,context,offset,span,(represent_input(left_input),represent_input(right_input)),_,Array.clone(Bias,Tree.pack(Bias,bias))) == represent(~Output,Partition.Fork{left,right}) : Partition.Tree<&1,Array<Output>,U32>}
      %Equal.sym(Array<Bias> & Array<Bias>,Array.clone(Bias,Tree.pack(Bias,bias)),(Tree.pack(Bias,bias),Tree.pack(Bias,bias)),StorageLaws.clone(Bias,bias)) :
        {native_children(~Native,~Weight,~Bias,~Output,~Context,~native_split,~native_leaf,rest,context,offset,span,(represent_input(left_input),represent_input(right_input)),(Tree.pack(Weight,weights),Tree.pack(Weight,weights)),_) == represent(~Output,Partition.Fork{left,right}) : Partition.Tree<&1,Array<Output>,U32>}
      %Equal.sym(Partition.Tree<&1,Array<Output>,U32>,
        Partition.run(~Native,~Array<Weight>,~Array<Bias>,~Array<Output>,~Context,~native_split,~Array.clone(Weight),~Array.clone(Bias),~native_leaf,
          rest,context,offset,half,represent_input(left_input),Tree.pack(Weight,weights),Tree.pack(Bias,bias)),represent(~Output,left),
        run_matches(~Native,~Input,~Weight,~Bias,~Output,~Context,~represent_input,~native_split,~model_split,~native_leaf,~model_leaf,~split_matches,~leaf_matches,rest,context,offset,half,left_input,weights,bias)) :
        {Partition.Fork{_,Partition.run(~Native,~Array<Weight>,~Array<Bias>,~Array<Output>,~Context,~native_split,~Array.clone(Weight),~Array.clone(Bias),~native_leaf,
          rest,context,(offset + half : U32),(span - half : U32),represent_input(right_input),Tree.pack(Weight,weights),Tree.pack(Bias,bias))} == represent(~Output,Partition.Fork{left,right}) : Partition.Tree<&1,Array<Output>,U32>}
      %Equal.sym(Partition.Tree<&1,Array<Output>,U32>,
        Partition.run(~Native,~Array<Weight>,~Array<Bias>,~Array<Output>,~Context,~native_split,~Array.clone(Weight),~Array.clone(Bias),~native_leaf,
          rest,context,(offset + half : U32),(span - half : U32),represent_input(right_input),Tree.pack(Weight,weights),Tree.pack(Bias,bias)),represent(~Output,right),
        run_matches(~Native,~Input,~Weight,~Bias,~Output,~Context,~represent_input,~native_split,~model_split,~native_leaf,~model_leaf,~split_matches,~leaf_matches,rest,context,(offset + half : U32),(span - half : U32),right_input,weights,bias)) :
        {Partition.Fork{represent(~Output,left),_} == represent(~Output,Partition.Fork{left,right}) : Partition.Tree<&1,Array<Output>,U32>}
      {==}

