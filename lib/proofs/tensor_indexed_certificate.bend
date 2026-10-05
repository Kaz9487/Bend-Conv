import Base
import ../tensor_indexed.bend as Indexed
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./storage_witness.bend as Witness
import ./traversal_witness.bend as TraversalWitness
import ./traversal_certificate.bend as TraversalCertificate

def sample(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~operation: U32 -> Element -> Element,
  -input: Array<Element>,+context: Context,+position: U32,-result: Array<Element> & Element,
  evidence: TraversalWitness.Sample<Array<Element>,Element,input,result>) ->
  TraversalWitness.Sample<Array<Element>,Element,input,Indexed.value(~Element,~Context,~index,~operation,context,position,result)>:
  match evidence:
    case TraversalWitness.Sample{value,equation}:
      %Equal.sym(Array<Element> & Element,result,(input,value),equation) :
        TraversalWitness.Sample<Array<Element>,Element,input,Indexed.value(~Element,~Context,~index,~operation,context,position,_)>
      TraversalWitness.Sample{operation(index(context,position),value),{==}}

def read(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~operation: U32 -> Element -> Element,
  -input: Array<Element>,context: Context,position: U32,evidence: Witness.ArrayWitness<Element,input>) ->
  TraversalWitness.Sample<Array<Element>,Element,input,Indexed.read(~Element,~Context,~index,~operation,input,context,position)>:
  +position = position
  sample(~Element,~Context,~index,~operation,input,context,position,Array.get(Element,input,position),
    TraversalWitness.array_sample(~Element,input,Array.get(Element,input,position),Witness.observed(Element,input,position,evidence)))

def preserves(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~operation: U32 -> Element -> Element,
  +depth: Nat,call: Indexed.Call<Context>,-input: Array<Element>,-output: Array<Element>,input_certificate: Certificate.Certificate<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,Indexed.run(~Element,~Context,~index,~operation,call,input,output)>:
  Indexed.Call{count,context} = call
  TraversalCertificate.completed(~Array<Element>,~Element,~Context,~TraversalWitness.array_evidence(~Element),
    ~Traversal.run(~Array<Element>,~Element,~Context,~Indexed.read(~Element,~Context,~index,~operation),~Indexed.destination(~Context)),
    ~TraversalWitness.run(~Array<Element>,~Element,~Context,~TraversalWitness.array_evidence(~Element),~Indexed.read(~Element,~Context,~index,~operation),
      ~Indexed.destination(~Context),~read(~Element,~Context,~index,~operation)),
    count,0,context,input,output,Certificate.witness(Element,input,input_certificate),depth,equation)

def certificates(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~operation: U32 -> Element -> Element,
  +call: Indexed.Call<Context>,-input: Array<Element>,-output: Array<Element>,input_certificate: Certificate.Certificate<Element,input>,
  output_certificate: Certificate.Certificate<Element,output>) ->
  TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),Indexed.run(~Element,~Context,~index,~operation,call,input,output)>:
  Certificate.Certificate{+depth,+equation} = input_certificate
  TraversalCertificate.operation(~Array<Element>,~Element,~Indexed.Call<Context>,~Certificate.family(~Element),
    ~Indexed.run(~Element,~Context,~index,~operation),~preserves(~Element,~Context,~index,~operation),
    call,input,output,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},output_certificate)

def next_block(~Element: Data,~Context: Data,+depth: Nat,block: U32,size: U32,context: Context,-input: Array<Element>,-state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Indexed.Block>,
  evidence: TraversalWitness.StateEvidence<Array<Element>,Element,Indexed.Block,input,depth,state>) ->
  TraversalWitness.StateEvidence<Array<Element>,Element,Indexed.Blocks<Context>,input,depth,Indexed.next_block(~Element,~Context,block,size,context,state)>:
  match evidence:
    case TraversalWitness.StateEvidence{output,position,inner,equation,output_evidence}:
      %Equal.sym(Traversal.State<&1,&1,Array<Element>,Array<Element>,Indexed.Block>,state,Traversal.State{input,output,position,inner},equation) :
        TraversalWitness.StateEvidence<Array<Element>,Element,Indexed.Blocks<Context>,input,depth,Indexed.next_block(~Element,~Context,block,size,context,_)>
      TraversalWitness.StateEvidence{output,position,Indexed.Blocks{(block + 1 : U32),size,context},{==},output_evidence}

def block_step(~Element: Data,~Context: Data,~base: Context -> U32 -> U32,~operation: U32 -> Element -> Element,+depth: Nat,-input: Array<Element>,-state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Indexed.Blocks<Context>>,
  +input_evidence: Witness.ArrayWitness<Element,input>,evidence: TraversalWitness.StateEvidence<Array<Element>,Element,Indexed.Blocks<Context>,input,depth,state>) ->
  TraversalWitness.StateEvidence<Array<Element>,Element,Indexed.Blocks<Context>,input,depth,Indexed.block_step(~Element,~Context,~base,~operation,state)>:
  match evidence:
    case TraversalWitness.StateEvidence{output,+position,Indexed.Blocks{+block,+size,+context},equation,output_evidence}:
      %Equal.sym(Traversal.State<&1,&1,Array<Element>,Array<Element>,Indexed.Blocks<Context>>,state,Traversal.State{input,output,position,Indexed.Blocks{block,size,context}},equation) :
        TraversalWitness.StateEvidence<Array<Element>,Element,Indexed.Blocks<Context>,input,depth,Indexed.block_step(~Element,~Context,~base,~operation,_)>
      next_block(~Element,~Context,depth,block,size,context,input,
        Traversal.run(~Array<Element>,~Element,~Indexed.Block,~Indexed.read(~Element,~Indexed.Block,~Indexed.block_index,~operation),~Indexed.destination(~Indexed.Block),U32.to_nat(size),
          Traversal.State{input,output,position,Indexed.Block{base(context,block),position}}),
        TraversalWitness.run(~Array<Element>,~Element,~Indexed.Block,~TraversalWitness.array_evidence(~Element),~Indexed.read(~Element,~Indexed.Block,~Indexed.block_index,~operation),
          ~Indexed.destination(~Indexed.Block),~read(~Element,~Indexed.Block,~Indexed.block_index,~operation),depth,U32.to_nat(size),input,
          Traversal.State{input,output,position,Indexed.Block{base(context,block),position}},input_evidence,
          TraversalWitness.StateEvidence{output,position,Indexed.Block{base(context,block),position},{==},output_evidence}))

def block_loop(~Element: Data,~Context: Data,~base: Context -> U32 -> U32,~operation: U32 -> Element -> Element,+depth: Nat,count: Nat,-input: Array<Element>,-state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Indexed.Blocks<Context>>,
  +input_evidence: Witness.ArrayWitness<Element,input>,evidence: TraversalWitness.StateEvidence<Array<Element>,Element,Indexed.Blocks<Context>,input,depth,state>) ->
  TraversalWitness.StateEvidence<Array<Element>,Element,Indexed.Blocks<Context>,input,depth,
    Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Indexed.Blocks<Context>>,~Indexed.block_step(~Element,~Context,~base,~operation),count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      block_loop(~Element,~Context,~base,~operation,depth,rest,input,Indexed.block_step(~Element,~Context,~base,~operation,state),input_evidence,
        block_step(~Element,~Context,~base,~operation,depth,input,state,input_evidence,evidence))

type Tiling<-Context: Data> is Data:
  Tiling{total: U32,context: Context}

def tiled_operation(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~base: Context -> U32 -> U32,~blocks: Context -> U32,~size: Context -> U32,~operation: U32 -> Element -> Element,
  tiling: Tiling<Context>,input: Array<Element>,output: Array<Element>) -> Array<Element> & Array<Element>:
  Tiling{total,context} = tiling
  Indexed.run_tiled(~Element,~Context,~index,~base,~blocks,~size,~operation,total,context,input,output)

def whole_preserves(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~base: Context -> U32 -> U32,~operation: U32 -> Element -> Element,
  +depth: Nat,whole: Bool,blocks: U32,size: U32,total: U32,context: Context,-input: Array<Element>,-output: Array<Element>,input_certificate: Certificate.Certificate<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,Indexed.run_whole(~Element,~Context,~index,~base,~operation,whole,blocks,size,total,context,input,output)>:
  match whole:
    case True{}:
      TraversalCertificate.completed(~Array<Element>,~Element,~Indexed.Blocks<Context>,~TraversalWitness.array_evidence(~Element),
        ~Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Indexed.Blocks<Context>>,~Indexed.block_step(~Element,~Context,~base,~operation)),~block_loop(~Element,~Context,~base,~operation),
        U32.to_nat(blocks),0,Indexed.Blocks{0,size,context},input,output,Certificate.witness(Element,input,input_certificate),depth,equation)
    case False{}:
      preserves(~Element,~Context,~index,~operation,depth,Indexed.Call{U32.to_nat(total),context},input,output,input_certificate,equation)

def tiled_preserves(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~base: Context -> U32 -> U32,~blocks: Context -> U32,~size: Context -> U32,~operation: U32 -> Element -> Element,
  +depth: Nat,tiling: Tiling<Context>,-input: Array<Element>,-output: Array<Element>,input_certificate: Certificate.Certificate<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,tiled_operation(~Element,~Context,~index,~base,~blocks,~size,~operation,tiling,input,output)>:
  Tiling{+total,+context} = tiling
  whole_preserves(~Element,~Context,~index,~base,~operation,depth,U32.is_le(total,Indexed.block_limit()) && Nat.is_eq(Nat.mul(U32.to_nat(blocks(context)),U32.to_nat(size(context))),U32.to_nat(total)),
    blocks(context),size(context),total,context,input,output,input_certificate,equation)

def tiled_certificates(~Element: Data,~Context: Data,~index: Context -> U32 -> U32,~base: Context -> U32 -> U32,~blocks: Context -> U32,~size: Context -> U32,~operation: U32 -> Element -> Element,
  +total: U32,+context: Context,-input: Array<Element>,-output: Array<Element>,input_certificate: Certificate.Certificate<Element,input>,
  output_certificate: Certificate.Certificate<Element,output>) ->
  TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),Indexed.run_tiled(~Element,~Context,~index,~base,~blocks,~size,~operation,total,context,input,output)>:
  Certificate.Certificate{+depth,+equation} = input_certificate
  TraversalCertificate.operation(~Array<Element>,~Element,~Tiling<Context>,~Certificate.family(~Element),
    ~tiled_operation(~Element,~Context,~index,~base,~blocks,~size,~operation),~tiled_preserves(~Element,~Context,~index,~base,~blocks,~size,~operation),
    Tiling{total,context},input,output,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},output_certificate)
