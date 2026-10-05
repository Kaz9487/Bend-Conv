# Certified contiguous ranges have the raw traversal's exact write program.
# The Data observation retains tree contents and lengths, erasing certificates.
import Base
import ../storage_buffer.bend as Storage
import ../storage_copy.bend as Copy
import ../tensor_operations.bend as Operations
import ../tensor_access.bend as Access
import ../tensor_view.bend as Views
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_certificate.bend as Certificate
import ./traversal_certificate.bend as TraversalCertificate
import ./tensor_access_certificate.bend as AccessCertificate
import ./contiguous_copy.bend as Contiguous
import ./write_program.bend as Writes

type Owners<-Element: Data> is Data:
  Owners{source: Tree.Tree<Element>,source_length: Nat,output: Tree.Tree<Element>,output_length: Nat}

type Call is Data:
  Call{count: U32,source: U32,destination: U32,source_length: Nat,output_length: Nat}

def call_count(call: Call) -> U32:
  Call{count,source,destination,input_length,output_length} = call
  count

def call_source(call: Call) -> U32:
  Call{count,source,destination,input_length,output_length} = call
  source

def call_destination(call: Call) -> U32:
  Call{count,source,destination,input_length,output_length} = call
  destination

def call_input_length(call: Call) -> Nat:
  Call{count,source,destination,input_length,output_length} = call
  input_length

def call_output_length(call: Call) -> Nat:
  Call{count,source,destination,input_length,output_length} = call
  output_length

def observe(~Element: Data,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Owners<Element>:
  (Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate}) = owners
  Owners{Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length}

def values(~Element: Data,input_length: Nat,output_length: Nat,result: Array<Element> & Array<Element>) -> Owners<Element>:
  (input,output) = result
  Owners{Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length}

def finished(~Element: Data,+input_length: Nat,+output_length: Nat,result: Array<Element> & Array<Element>,
  certificates: TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),result>) ->
  {observe(~Element,Operations.pair_buffers(~Element,input_length,output_length,result,certificates)) == values(~Element,input_length,output_length,result) : Owners<Element>}:
  (input,output) = result
  TraversalCertificate.Certificates{input_certificate,output_certificate} = certificates
  {==}

def represented_values(~Element: Data,+input: Tree.Tree<Element>,+input_length: Nat,+output: Tree.Tree<Element>,+output_length: Nat) ->
  {values(~Element,input_length,output_length,(Tree.pack(Element,input),Tree.pack(Element,output))) == Owners{input,input_length,output,output_length} : Owners<Element>}:
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,input)),input,StorageLaws.reflect_packed(Element,input)) :
    {Owners{_,input_length,Tree.reflect(Element,Tree.pack(Element,output)),output_length} == Owners{input,input_length,output,output_length} : Owners<Element>}
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,output)),output,StorageLaws.reflect_packed(Element,output)) :
    {Owners{input,input_length,_,output_length} == Owners{input,input_length,output,output_length} : Owners<Element>}
  {==}

def model(~Element: Data,count: U32,source: U32,destination: U32,owners: Owners<Element>) -> Owners<Element>:
  Owners{+input,input_length,output,output_length} = owners
  Owners{input,input_length,Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),input,0,source,destination),output),output_length}

def packed(~Element: Data,+call: Call,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,
  +input_certificate: Certificate.Certificate<Element,Tree.pack(Element,input)>,+output_certificate: Certificate.Certificate<Element,Tree.pack(Element,output)>) ->
  {observe(~Element,Copy.range(~Element,call_count(call),call_source(call),call_destination(call),
      Storage.Buffer{Tree.pack(Element,input),call_input_length(call),input_certificate},Storage.Buffer{Tree.pack(Element,output),call_output_length(call),output_certificate})) ==
    model(~Element,call_count(call),call_source(call),call_destination(call),Owners{input,call_input_length(call),output,call_output_length(call)}) : Owners<Element>}:
  Call{+count,+source,+destination,+input_length,+output_length} = call
  +view = {Views.View{[Views.Axis{count,1}],source} : Views.View}
  %Equal.sym(Owners<Element>,observe(~Element,Operations.pair_buffers(~Element,input_length,output_length,
      Access.copy_run(~Element,True{},U32.to_nat(count),view,destination,Tree.pack(Element,input),Tree.pack(Element,output)),
      AccessCertificate.copy(~Element,True{},U32.to_nat(count),view,destination,Tree.pack(Element,input),Tree.pack(Element,output),input_certificate,output_certificate))),
    values(~Element,input_length,output_length,Access.copy_run(~Element,True{},U32.to_nat(count),view,destination,Tree.pack(Element,input),Tree.pack(Element,output))),
    finished(~Element,input_length,output_length,Access.copy_run(~Element,True{},U32.to_nat(count),view,destination,Tree.pack(Element,input),Tree.pack(Element,output)),
      AccessCertificate.copy(~Element,True{},U32.to_nat(count),view,destination,Tree.pack(Element,input),Tree.pack(Element,output),input_certificate,output_certificate))) :
    {_ == model(~Element,count,source,destination,Owners{input,input_length,output,output_length}) : Owners<Element>}
  %Equal.sym(Array<Element> & Array<Element>,Access.copy_run(~Element,True{},U32.to_nat(count),view,destination,Tree.pack(Element,input),Tree.pack(Element,output)),
    (Tree.pack(Element,input),Tree.pack(Element,Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),input,0,source,destination),output))),Contiguous.copy(~Element,U32.to_nat(count),input,output,source,destination)) :
    {values(~Element,input_length,output_length,_) == model(~Element,count,source,destination,Owners{input,input_length,output,output_length}) : Owners<Element>}
  represented_values(~Element,input,input_length,Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),input,0,source,destination),output),output_length)

def proposition(~Element: Data,call: Call,-input: Array<Element>,-output: Array<Element>,-input_tree: Tree.Tree<Element>,-output_tree: Tree.Tree<Element>) -> Type:
  +call = call
  @input_certificate: Certificate.Certificate<Element,input> -> @output_certificate: Certificate.Certificate<Element,output> ->
  {observe(~Element,Copy.range(~Element,call_count(call),call_source(call),call_destination(call),
      Storage.Buffer{input,call_input_length(call),input_certificate},Storage.Buffer{output,call_output_length(call),output_certificate})) ==
    model(~Element,call_count(call),call_source(call),call_destination(call),Owners{input_tree,call_input_length(call),output_tree,call_output_length(call)}) : Owners<Element>}

def range_pair(~Element: Data,count: U32,source: U32,destination: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  (input,output) = owners
  Copy.range(~Element,count,source,destination,input,output)

def range(~Element: Data,+count: U32,+source: U32,+destination: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  {observe(~Element,range_pair(~Element,count,source,destination,owners)) == model(~Element,count,source,destination,observe(~Element,owners)) : Owners<Element>}:
  (Storage.Buffer{input,+input_length,input_certificate},Storage.Buffer{output,+output_length,output_certificate}) = owners
  +call = {Call{count,source,destination,input_length,output_length} : Call}
  StorageLaws.lift_pair_observation(~Element,~Call,~proposition(~Element),call,input,output,StorageLaws.observe(Element,input),StorageLaws.observe(Element,output),
    input_tree => output_tree => input_certificate => output_certificate => packed(~Element,call,input_tree,output_tree,input_certificate,output_certificate))(input_certificate)(output_certificate)
