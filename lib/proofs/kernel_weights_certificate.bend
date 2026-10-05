import Base
import ../kernel_weights.bend as Weights
import ../kernel_tile.bend as Tile
import ../traversal_array.bend as Traversal
import ../storage_allocation.bend as Allocation
import ../tensor.bend as Tensors
import ./kernel_weights_witness.bend as WeightsWitness
import ./traversal_witness.bend as TraversalWitness
import ./storage_witness.bend as Witness
import ./storage_certificate.bend as Certificate
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_tree_model.bend as Tree

def family(-input: Array<F32>) -> Data:
  Witness.ArrayWitness<F32,input>

def block(+depth: Nat,+block: U32,+layout: Weights.Layout,-input: Array<F32>,-owners: Array<F32> & Array<Tile.Tile8>,
  input_evidence: Witness.ArrayWitness<F32,input>,evidence: TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,owners>) ->
  TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,Weights.copy_block(block,layout,owners)>:
  Weights.Layout{+rows,+inner} = layout
  TraversalWitness.ResultEvidence{output,equation,output_evidence} = evidence
  %Equal.sym(Array<F32> & Array<Tile.Tile8>,owners,(input,output),equation) :
    TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,Weights.copy_block(block,Weights.Layout{rows,inner},_)>
  +context = {Weights.Context{rows,inner,(block * 8 : U32),(block * inner : U32)} : Weights.Context}
  -state = {Traversal.State{input,output,0,context} : Traversal.State<&1,&1,Array<F32>,Array<Tile.Tile8>,Weights.Context>}
  TraversalWitness.finish(~Array<F32>,~Tile.Tile8,~Weights.Context,input,depth,
    Traversal.run(~Array<F32>,~Tile.Tile8,~Weights.Context,~Weights.read,~Weights.destination,U32.to_nat(inner),state),
    TraversalWitness.run(~Array<F32>,~Tile.Tile8,~Weights.Context,~family,~Weights.read,~Weights.destination,~WeightsWitness.read,
      depth,U32.to_nat(inner),input,state,input_evidence,TraversalWitness.StateEvidence{output,0,context,{==},output_evidence}))

def traverse(+depth: Nat,count: Nat,+index: U32,+layout: Weights.Layout,-input: Array<F32>,-owners: Array<F32> & Array<Tile.Tile8>,
  +input_evidence: Witness.ArrayWitness<F32,input>,evidence: TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,owners>) ->
  TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,
    Traversal.indexed_fold(~(Array<F32> & Array<Tile.Tile8>),~Weights.Layout,~Weights.copy_block,count,index,layout,owners)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      traverse(depth,rest,(index + 1 : U32),layout,input,Weights.copy_block(index,layout,owners),input_evidence,
        block(depth,index,layout,input,owners,input_evidence,evidence))

def finish(-depth: Nat,-input: Array<F32>,-owners: Array<F32> & Array<Tile.Tile8>,
  evidence: TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,owners>) -> Witness.Witness(Tile.Tile8,depth,Weights.reordered(owners)):
  TraversalWitness.ResultEvidence{output,equation,output_evidence} = evidence
  %Equal.sym(Array<F32> & Array<Tile.Tile8>,owners,(input,output),equation) : Witness.Witness(Tile.Tile8,depth,Weights.reordered(_))
  output_evidence

def allocated(+depth: Nat) -> Witness.Witness(Tile.Tile8,depth,Allocation.array(~Tile.Tile8,depth,Tile.filled(0.0))):
  %Equal.sym(Array<Tile.Tile8>,Allocation.array(~Tile.Tile8,depth,Tile.filled(0.0)),Array.new(Tile.Tile8,depth,Tile.filled(0.0)),
    AllocationLaws.array_same(Tile.Tile8,depth,Tile.filled(0.0))) : Witness.Witness(Tile.Tile8,depth,_)
  Witness.fresh(Tile.Tile8,depth,Tile.filled(0.0))

def run(+rows: U32,+inner: U32,-input: Array<F32>,evidence: Witness.ArrayWitness<F32,input>) ->
  Witness.Witness(Tile.Tile8,Tensors.depth_for((((rows + 7) / 8) * inner : U32)),Weights.run(rows,inner,input)):
  +blocks = ((rows + 7) / 8 : U32)
  +depth = Tensors.depth_for((blocks * inner : U32))
  +layout = {Weights.Layout{rows,inner} : Weights.Layout}
  -output = Allocation.array(~Tile.Tile8,depth,Tile.filled(0.0))
  finish(depth,input,Traversal.indexed_fold(~(Array<F32> & Array<Tile.Tile8>),~Weights.Layout,~Weights.copy_block,U32.to_nat(blocks),0,layout,(input,output)),
    traverse(depth,U32.to_nat(blocks),0,layout,input,(input,output),evidence,TraversalWitness.ResultEvidence{output,{==},allocated(depth)}))

def equation(+rows: U32,+inner: U32,-input: Array<F32>,certificate: Certificate.Certificate<F32,input>) ->
  {Weights.run(rows,inner,input) == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,Tensors.depth_for((((rows + 7) / 8) * inner : U32)),
    Tree.reflect(Tile.Tile8,Weights.run(rows,inner,input)))) : Array<Tile.Tile8>}:
  %Certificate.restored(Tile.Tile8,Tensors.depth_for((((rows + 7) / 8) * inner : U32)),Weights.run(rows,inner,input),
    run(rows,inner,input,Certificate.witness(F32,input,certificate))) :
    {_ == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,Tensors.depth_for((((rows + 7) / 8) * inner : U32)),
      Tree.reflect(Tile.Tile8,Weights.run(rows,inner,input)))) : Array<Tile.Tile8>}
  {==}

def certificate(+rows: U32,+inner: U32,-input: Array<F32>,certificate: Certificate.Certificate<F32,input>) ->
  Certificate.Certificate<Tile.Tile8,Weights.run(rows,inner,input)>:
  Certificate.Certificate{Tensors.depth_for((((rows + 7) / 8) * inner : U32)),equation(rows,inner,input,certificate)}
