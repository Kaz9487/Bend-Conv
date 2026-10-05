import Base
import ../kernel_gather.bend as Gather
import ../traversal_array.bend as Traversal
import ./storage_witness.bend as Witness
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./traversal_witness.bend as TraversalWitness

def channel(+depth: Nat,+channel: U32,+width: U32,+offset: U32,+positions: U32,+span: U32,-input: Array<F32>,-owners: Array<F32> & Array<F32>,
  input_evidence: Witness.ArrayWitness<F32,input>,evidence: TraversalWitness.ResultEvidence<Array<F32>,F32,input,depth,owners>) ->
  TraversalWitness.ResultEvidence<Array<F32>,F32,input,depth,Gather.gather_channel(channel,width,offset,positions,span,owners)>:
  TraversalWitness.ResultEvidence{output,equation,balanced} = evidence
  %Equal.sym(Array<F32> & Array<F32>,owners,(input,output),equation) :
    TraversalWitness.ResultEvidence<Array<F32>,F32,input,depth,Gather.gather_channel(channel,width,offset,positions,span,_)>
  +context = {Traversal.Offsets{(channel * span : U32),(channel * positions + offset : U32)} : Traversal.Offsets}
  -state = {Traversal.State{input,output,0,context} : Traversal.State<&1,&1,Array<F32>,Array<F32>,Traversal.Offsets>}
  TraversalWitness.finish(~Array<F32>,~F32,~Traversal.Offsets,input,depth,
    Traversal.run(~Array<F32>,~F32,~Traversal.Offsets,~Traversal.source_offset(~F32),~Traversal.destination_offset,U32.to_nat(width),state),
    TraversalWitness.run(~Array<F32>,~F32,~Traversal.Offsets,~TraversalWitness.array_evidence(~F32),~Traversal.source_offset(~F32),~Traversal.destination_offset,~TraversalWitness.offset_read(~F32),
      depth,U32.to_nat(width),input,state,input_evidence,TraversalWitness.StateEvidence{output,0,context,{==},balanced}))

def channels(+depth: Nat,count: Nat,+index: U32,+width: U32,+offset: U32,+positions: U32,+span: U32,-input: Array<F32>,-owners: Array<F32> & Array<F32>,
  +input_evidence: Witness.ArrayWitness<F32,input>,evidence: TraversalWitness.ResultEvidence<Array<F32>,F32,input,depth,owners>) ->
  TraversalWitness.ResultEvidence<Array<F32>,F32,input,depth,Gather.gather_channels(count,index,width,offset,positions,span,owners)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      channels(depth,rest,(index + 1 : U32),width,offset,positions,span,input,Gather.gather_channel(index,width,offset,positions,span,owners),input_evidence,
        channel(depth,index,width,offset,positions,span,input,owners,input_evidence,evidence))

def finish(-depth: Nat,-input: Array<F32>,-owners: Array<F32> & Array<F32>,evidence: TraversalWitness.ResultEvidence<Array<F32>,F32,input,depth,owners>) ->
  Witness.Witness(F32,depth,Gather.gathered_output(owners)):
  TraversalWitness.ResultEvidence{output,equation,balanced} = evidence
  %Equal.sym(Array<F32> & Array<F32>,owners,(input,output),equation) : Witness.Witness(F32,depth,Gather.gathered_output(_))
  balanced

def block(+depth: Nat,+count: U32,+offset: U32,+positions: U32,+span: U32,-input: Array<F32>,-output: Array<F32>,
  input_evidence: Witness.ArrayWitness<F32,input>,balanced: Witness.Witness(F32,depth,output)) ->
  Witness.Witness(F32,depth,Gather.gather_block(count,offset,positions,span,input,output)):
  +width = Gather.gather_width(offset,positions,span)
  finish(depth,input,Gather.gather_channels(U32.to_nat(count),0,width,offset,positions,span,(input,output)),
    channels(depth,U32.to_nat(count),0,width,offset,positions,span,input,(input,output),input_evidence,TraversalWitness.ResultEvidence{output,{==},balanced}))

def equation(+depth: Nat,count: U32,offset: U32,positions: U32,span: U32,-input: Array<F32>,-output: Array<F32>,
  input_certificate: Certificate.Certificate<F32,input>,balanced: {output == Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,output))) : Array<F32>}) ->
  {Gather.gather_block(count,offset,positions,span,input,output) ==
    Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,Gather.gather_block(count,offset,positions,span,input,output)))) : Array<F32>}:
  %Certificate.restored(F32,depth,Gather.gather_block(count,offset,positions,span,input,output),
    block(depth,count,offset,positions,span,input,output,Certificate.witness(F32,input,input_certificate),Certificate.witness_at(F32,depth,output,balanced))) :
    {_ == Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,Gather.gather_block(count,offset,positions,span,input,output)))) : Array<F32>}
  {==}

def certificate(count: U32,offset: U32,positions: U32,span: U32,-input: Array<F32>,-output: Array<F32>,
  input_certificate: Certificate.Certificate<F32,input>,output_certificate: Certificate.Certificate<F32,output>) ->
  Certificate.Certificate<F32,Gather.gather_block(count,offset,positions,span,input,output)>:
  Certificate.Certificate{+depth,balanced} = output_certificate
  Certificate.Certificate{depth,equation(depth,count,offset,positions,span,input,output,input_certificate,balanced)}
