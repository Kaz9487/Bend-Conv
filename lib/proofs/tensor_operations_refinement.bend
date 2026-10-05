# Public-owner observations erase certificate representation while retaining
# storage and logical lengths. Kernel equations use the independent tree model.
import Base
import ./storage_refinement.bend as StorageLaws
import ../storage_buffer.bend as Storage
import ../tensor_operations.bend as Operations
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ../tensor_access.bend as Access
import ../traversal_array.bend as Traversal
import ../kernel_block.bend as Block
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./traversal_certificate.bend as TraversalCertificate
import ./traversal_witness.bend as TraversalWitness
import ./tensor_access_certificate.bend as AccessCertificate
import ./tensor_access_refinement.bend as AccessLaws
import ./traversal_array_refinement.bend as ArrayLaws
import ./tensor_operation_bounds.bend as OperationBounds
import ./tensor_cursor_order.bend as CursorOrder

type Owner<kind,-Value: Kind(kind)> is Kind(kind):
  Owner{storage: Value,length: Nat}

def PairOwners(-Element: Data) -> Type:
  Owner<&1,Array<Element>> & Owner<&1,Array<Element>>

def observe_owner(~Element: Data,buffer: Storage.Buffer<Element>) -> Owner<&1,Array<Element>>:
  Storage.Buffer{storage,length,certificate} = buffer
  Owner{storage,length}

def observe_pair(~Element: Data,pair: Storage.Buffer<Element> & Storage.Buffer<Element>) -> PairOwners(Element):
  (input,output) = pair
  (observe_owner(~Element,input),observe_owner(~Element,output))

def observe_pair_outcome(~Element: Data,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) -> Operations.Outcome<PairOwners(Element)>:
  match result:
    case Operations.Accepted{value}: Operations.Accepted{observe_pair(~Element,value)}
    case Operations.Rejected{value}: Operations.Rejected{observe_pair(~Element,value)}

def owned_pair(~Element: Data,input_length: Nat,output_length: Nat,result: Array<Element> & Array<Element>) -> PairOwners(Element):
  (input,output) = result
  (Owner{input,input_length},Owner{output,output_length})

def modeled_pair(~Element: Data,input_length: Nat,output_length: Nat,result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) -> PairOwners(Element):
  AccessLaws.ModelResult{input,output} = result
  (Owner{Tree.pack(Element,input),input_length},Owner{Tree.pack(Element,output),output_length})

def pair_finished(~Element: Data,+input_length: Nat,+output_length: Nat,result: Array<Element> & Array<Element>,
  certificates: TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),result>,
  +model: AccessLaws.ModelResult<Tree.Tree<Element>,Element>,
  equation: {result == AccessLaws.pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),model) : Array<Element> & Array<Element>}) ->
  {observe_pair(~Element,Operations.pair_buffers(~Element,input_length,output_length,result,certificates)) ==
    modeled_pair(~Element,input_length,output_length,model) : PairOwners(Element)}:
  (input,output) = result
  TraversalCertificate.Certificates{input_certificate,output_certificate} = certificates
  AccessLaws.ModelResult{+input_tree,+output_tree} = model
  Equal.cong(Array<Element> & Array<Element>,PairOwners(Element),result => owned_pair(~Element,input_length,output_length,result),
    (input,output),(Tree.pack(Element,input_tree),Tree.pack(Element,output_tree)),equation)

def copy_model(~Element: Data,valid: Bool,count: Maybe<&2,U32>,source: Views.View,target: Views.View,
  input: Tree.Tree<Element>,input_length: Nat,output: Tree.Tree<Element>,output_length: Nat) -> Operations.Outcome<PairOwners(Element)>:
  match valid count:
    case True{} Some{count}:
      Views.View{axes,destination} = target
      +source = source
      Operations.Accepted{modeled_pair(~Element,input_length,output_length,
        AccessLaws.copy_model(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input,output))}
    case valid count:
      Operations.Rejected{(Owner{Tree.pack(Element,input),input_length},Owner{Tree.pack(Element,output),output_length})}

def packed_copy(~Element: Data,+valid: Bool,+count: Maybe<&2,U32>,+source: Views.View,+target: Views.View,
  +input: Tree.Tree<Element>,+input_length: Nat,+output: Tree.Tree<Element>,+output_length: Nat,
  +input_certificate: Certificate.Certificate<Element,Tree.pack(Element,input)>,
  +output_certificate: Certificate.Certificate<Element,Tree.pack(Element,output)>) ->
  {observe_pair_outcome(~Element,Operations.copy_when(~Element,valid,count,source,target,
      Storage.Buffer{Tree.pack(Element,input),input_length,input_certificate},Storage.Buffer{Tree.pack(Element,output),output_length,output_certificate})) ==
    copy_model(~Element,valid,count,source,target,input,input_length,output,output_length) : Operations.Outcome<PairOwners(Element)>}:
  match valid count:
    case False{} None{}: {==}
    case False{} Some{count}: {==}
    case True{} None{}: {==}
    case True{} Some{+count}:
      Views.View{+axes,+destination} = target
      Equal.cong(PairOwners(Element),Operations.Outcome<PairOwners(Element)>,value => Operations.Accepted{value},
        observe_pair(~Element,Operations.pair_buffers(~Element,input_length,output_length,
          Access.copy_run(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,Tree.pack(Element,input),Tree.pack(Element,output)),
          AccessCertificate.copy(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,Tree.pack(Element,input),Tree.pack(Element,output),input_certificate,output_certificate))),
        modeled_pair(~Element,input_length,output_length,AccessLaws.copy_model(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input,output)),
        pair_finished(~Element,input_length,output_length,
          Access.copy_run(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,Tree.pack(Element,input),Tree.pack(Element,output)),
          AccessCertificate.copy(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,Tree.pack(Element,input),Tree.pack(Element,output),input_certificate,output_certificate),
          AccessLaws.copy_model(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input,output),
          AccessLaws.copy(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input,output)))

def observed_copy(~Element: Data,+valid: Bool,+count: Maybe<&2,U32>,+source: Views.View,+target: Views.View,
  -input: Array<Element>,+input_length: Nat,-output: Array<Element>,+output_length: Nat,
  input_observation: StorageLaws.Observation<Element,input>,output_observation: StorageLaws.Observation<Element,output>) ->
  @input_certificate: Certificate.Certificate<Element,input> -> @output_certificate: Certificate.Certificate<Element,output> ->
  {observe_pair_outcome(~Element,Operations.copy_when(~Element,valid,count,source,target,
      Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})) ==
    copy_model(~Element,valid,count,source,target,Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length) : Operations.Outcome<PairOwners(Element)>}:
  StorageLaws.Observation{+input_tree,input_packed,input_reflected} = input_observation
  StorageLaws.Observation{+output_tree,output_packed,output_reflected} = output_observation
  %input_reflected :
    @input_certificate: Certificate.Certificate<Element,input> -> @output_certificate: Certificate.Certificate<Element,output> ->
    {observe_pair_outcome(~Element,Operations.copy_when(~Element,valid,count,source,target,
        Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})) ==
      copy_model(~Element,valid,count,source,target,_,input_length,Tree.reflect(Element,output),output_length) : Operations.Outcome<PairOwners(Element)>}
  %output_reflected :
    @input_certificate: Certificate.Certificate<Element,input> -> @output_certificate: Certificate.Certificate<Element,output> ->
    {observe_pair_outcome(~Element,Operations.copy_when(~Element,valid,count,source,target,
        Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})) ==
      copy_model(~Element,valid,count,source,target,input_tree,input_length,_,output_length) : Operations.Outcome<PairOwners(Element)>}
  %input_packed :
    @input_certificate: Certificate.Certificate<Element,_> -> @output_certificate: Certificate.Certificate<Element,output> ->
    {observe_pair_outcome(~Element,Operations.copy_when(~Element,valid,count,source,target,
        Storage.Buffer{_,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})) ==
      copy_model(~Element,valid,count,source,target,input_tree,input_length,output_tree,output_length) : Operations.Outcome<PairOwners(Element)>}
  %output_packed :
    @input_certificate: Certificate.Certificate<Element,Tree.pack(Element,input_tree)> -> @output_certificate: Certificate.Certificate<Element,_> ->
    {observe_pair_outcome(~Element,Operations.copy_when(~Element,valid,count,source,target,
        Storage.Buffer{Tree.pack(Element,input_tree),input_length,input_certificate},Storage.Buffer{_,output_length,output_certificate})) ==
      copy_model(~Element,valid,count,source,target,input_tree,input_length,output_tree,output_length) : Operations.Outcome<PairOwners(Element)>}
  input_certificate => output_certificate =>
    packed_copy(~Element,valid,count,source,target,input_tree,input_length,output_tree,output_length,input_certificate,output_certificate)

def copy_denotation(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,output: Storage.Buffer<Element>,+target: Views.View) -> Operations.Outcome<PairOwners(Element)>:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Storage.Buffer{output,+output_length,output_certificate} = output
  copy_model(~Element,Shapes.same(Views.shape(source),Views.shape(target)) && Views.is_contiguous(target) &&
    Operations.valid_buffer(input_length,Storage.capacity(~Element,input,input_certificate),source) &&
    Operations.valid_buffer(output_length,Storage.capacity(~Element,output,output_certificate),target),Views.numel(target),source,target,
    Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length)

def copy(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,output: Storage.Buffer<Element>,+target: Views.View) ->
  {observe_pair_outcome(~Element,Operations.copy(~Element,input,source,output,target)) == copy_denotation(~Element,input,source,output,target) : Operations.Outcome<PairOwners(Element)>}:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Certificate.Certificate{+input_certificate_depth,+input_certificate_equation} = input_certificate
  Storage.Buffer{output,+output_length,output_certificate} = output
  Certificate.Certificate{+output_certificate_depth,+output_certificate_equation} = output_certificate
  observed_copy(~Element,Shapes.same(Views.shape(source),Views.shape(target)) && Views.is_contiguous(target) &&
    Operations.valid_buffer(input_length,Storage.capacity(~Element,input,Certificate.Certificate{input_certificate_depth,input_certificate_equation}),source) &&
    Operations.valid_buffer(output_length,Storage.capacity(~Element,output,Certificate.Certificate{output_certificate_depth,output_certificate_equation}),target),Views.numel(target),source,target,
    input,input_length,output,output_length,StorageLaws.observe(Element,input),StorageLaws.observe(Element,output))(Certificate.Certificate{input_certificate_depth,input_certificate_equation})(Certificate.Certificate{output_certificate_depth,output_certificate_equation})
# A scatter: the contiguous source written through the target view.
def scatter_model(~Element: Data,valid: Bool,count: Maybe<&2,U32>,source: Views.View,target: Views.View,
  input: Tree.Tree<Element>,input_length: Nat,output: Tree.Tree<Element>,output_length: Nat) -> Operations.Outcome<PairOwners(Element)>:
  match valid count:
    case True{} Some{count}:
      Views.View{axes,offset} = source
      Operations.Accepted{modeled_pair(~Element,input_length,output_length,
        AccessLaws.scatter_model(~Element,U32.to_nat(count),target,offset,input,output))}
    case valid count:
      Operations.Rejected{(Owner{Tree.pack(Element,input),input_length},Owner{Tree.pack(Element,output),output_length})}

def packed_scatter(~Element: Data,+valid: Bool,+count: Maybe<&2,U32>,+source: Views.View,+target: Views.View,
  +input: Tree.Tree<Element>,+input_length: Nat,+output: Tree.Tree<Element>,+output_length: Nat,
  +input_certificate: Certificate.Certificate<Element,Tree.pack(Element,input)>,
  +output_certificate: Certificate.Certificate<Element,Tree.pack(Element,output)>) ->
  {observe_pair_outcome(~Element,Operations.scatter_when(~Element,valid,count,source,target,
      Storage.Buffer{Tree.pack(Element,input),input_length,input_certificate},Storage.Buffer{Tree.pack(Element,output),output_length,output_certificate})) ==
    scatter_model(~Element,valid,count,source,target,input,input_length,output,output_length) : Operations.Outcome<PairOwners(Element)>}:
  match valid count:
    case False{} None{}: {==}
    case False{} Some{count}: {==}
    case True{} None{}: {==}
    case True{} Some{+count}:
      Views.View{+axes,+offset} = source
      Equal.cong(PairOwners(Element),Operations.Outcome<PairOwners(Element)>,value => Operations.Accepted{value},
        observe_pair(~Element,Operations.pair_buffers(~Element,input_length,output_length,
          Access.scatter_run(~Element,U32.to_nat(count),target,offset,Tree.pack(Element,input),Tree.pack(Element,output)),
          AccessCertificate.scatter(~Element,U32.to_nat(count),target,offset,Tree.pack(Element,input),Tree.pack(Element,output),input_certificate,output_certificate))),
        modeled_pair(~Element,input_length,output_length,AccessLaws.scatter_model(~Element,U32.to_nat(count),target,offset,input,output)),
        pair_finished(~Element,input_length,output_length,
          Access.scatter_run(~Element,U32.to_nat(count),target,offset,Tree.pack(Element,input),Tree.pack(Element,output)),
          AccessCertificate.scatter(~Element,U32.to_nat(count),target,offset,Tree.pack(Element,input),Tree.pack(Element,output),input_certificate,output_certificate),
          AccessLaws.scatter_model(~Element,U32.to_nat(count),target,offset,input,output),
          AccessLaws.scatter(~Element,U32.to_nat(count),target,offset,input,output)))

def observed_scatter(~Element: Data,+valid: Bool,+count: Maybe<&2,U32>,+source: Views.View,+target: Views.View,
  -input: Array<Element>,+input_length: Nat,-output: Array<Element>,+output_length: Nat,
  input_observation: StorageLaws.Observation<Element,input>,output_observation: StorageLaws.Observation<Element,output>) ->
  @input_certificate: Certificate.Certificate<Element,input> -> @output_certificate: Certificate.Certificate<Element,output> ->
  {observe_pair_outcome(~Element,Operations.scatter_when(~Element,valid,count,source,target,
      Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})) ==
    scatter_model(~Element,valid,count,source,target,Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length) : Operations.Outcome<PairOwners(Element)>}:
  StorageLaws.Observation{+input_tree,input_packed,input_reflected} = input_observation
  StorageLaws.Observation{+output_tree,output_packed,output_reflected} = output_observation
  %input_reflected :
    @input_certificate: Certificate.Certificate<Element,input> -> @output_certificate: Certificate.Certificate<Element,output> ->
    {observe_pair_outcome(~Element,Operations.scatter_when(~Element,valid,count,source,target,
        Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})) ==
      scatter_model(~Element,valid,count,source,target,_,input_length,Tree.reflect(Element,output),output_length) : Operations.Outcome<PairOwners(Element)>}
  %output_reflected :
    @input_certificate: Certificate.Certificate<Element,input> -> @output_certificate: Certificate.Certificate<Element,output> ->
    {observe_pair_outcome(~Element,Operations.scatter_when(~Element,valid,count,source,target,
        Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})) ==
      scatter_model(~Element,valid,count,source,target,input_tree,input_length,_,output_length) : Operations.Outcome<PairOwners(Element)>}
  %input_packed :
    @input_certificate: Certificate.Certificate<Element,_> -> @output_certificate: Certificate.Certificate<Element,output> ->
    {observe_pair_outcome(~Element,Operations.scatter_when(~Element,valid,count,source,target,
        Storage.Buffer{_,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})) ==
      scatter_model(~Element,valid,count,source,target,input_tree,input_length,output_tree,output_length) : Operations.Outcome<PairOwners(Element)>}
  %output_packed :
    @input_certificate: Certificate.Certificate<Element,Tree.pack(Element,input_tree)> -> @output_certificate: Certificate.Certificate<Element,_> ->
    {observe_pair_outcome(~Element,Operations.scatter_when(~Element,valid,count,source,target,
        Storage.Buffer{Tree.pack(Element,input_tree),input_length,input_certificate},Storage.Buffer{_,output_length,output_certificate})) ==
      scatter_model(~Element,valid,count,source,target,input_tree,input_length,output_tree,output_length) : Operations.Outcome<PairOwners(Element)>}
  input_certificate => output_certificate =>
    packed_scatter(~Element,valid,count,source,target,input_tree,input_length,output_tree,output_length,input_certificate,output_certificate)

def scatter_denotation(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,output: Storage.Buffer<Element>,+target: Views.View) -> Operations.Outcome<PairOwners(Element)>:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Storage.Buffer{output,+output_length,output_certificate} = output
  scatter_model(~Element,Shapes.same(Views.shape(source),Views.shape(target)) && Views.is_contiguous(source) &&
    Operations.valid_buffer(input_length,Storage.capacity(~Element,input,input_certificate),source) &&
    Operations.valid_buffer(output_length,Storage.capacity(~Element,output,output_certificate),target),Views.numel(target),source,target,
    Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length)

def scatter(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,output: Storage.Buffer<Element>,+target: Views.View) ->
  {observe_pair_outcome(~Element,Operations.scatter(~Element,input,source,output,target)) == scatter_denotation(~Element,input,source,output,target) : Operations.Outcome<PairOwners(Element)>}:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Certificate.Certificate{+input_certificate_depth,+input_certificate_equation} = input_certificate
  Storage.Buffer{output,+output_length,output_certificate} = output
  Certificate.Certificate{+output_certificate_depth,+output_certificate_equation} = output_certificate
  observed_scatter(~Element,Shapes.same(Views.shape(source),Views.shape(target)) && Views.is_contiguous(source) &&
    Operations.valid_buffer(input_length,Storage.capacity(~Element,input,Certificate.Certificate{input_certificate_depth,input_certificate_equation}),source) &&
    Operations.valid_buffer(output_length,Storage.capacity(~Element,output,Certificate.Certificate{output_certificate_depth,output_certificate_equation}),target),Views.numel(target),source,target,
    input,input_length,output,output_length,StorageLaws.observe(Element,input),StorageLaws.observe(Element,output))(Certificate.Certificate{input_certificate_depth,input_certificate_equation})(Certificate.Certificate{output_certificate_depth,output_certificate_equation})
type PairCall is Data:
  PairCall{valid: Bool,count: Maybe<&2,U32>,source: Views.View,target: Views.View,input_length: Nat,output_length: Nat}

def combined_pair(~Element: Data,left_length: Nat,right_length: Nat,result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) -> PairOwners(Element):
  AccessLaws.ModelResult{right,left} = result
  (Owner{Tree.pack(Element,left),left_length},Owner{Tree.pack(Element,right),right_length})

def combine_owned(~Element: Data,left_length: Nat,right_length: Nat,result: Array<Element> & Array<Element>) -> PairOwners(Element):
  (right,left) = result
  (Owner{left,left_length},Owner{right,right_length})

def combine_finished(~Element: Data,+left_length: Nat,+right_length: Nat,result: Array<Element> & Array<Element>,
  certificates: TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),result>,+model: AccessLaws.ModelResult<Tree.Tree<Element>,Element>,
  equation: {result == AccessLaws.pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),model) : Array<Element> & Array<Element>}) ->
  {observe_pair(~Element,Operations.combine_buffers(~Element,left_length,right_length,result,certificates)) == combined_pair(~Element,left_length,right_length,model) : PairOwners(Element)}:
  (right,left) = result
  TraversalCertificate.Certificates{right_certificate,left_certificate} = certificates
  AccessLaws.ModelResult{+right_tree,+left_tree} = model
  Equal.cong(Array<Element> & Array<Element>,PairOwners(Element),result => combine_owned(~Element,left_length,right_length,result),
    (right,left),(Tree.pack(Element,right_tree),Tree.pack(Element,left_tree)),equation)

def combine_model(~Element: Data,~operation: Element -> Element -> Element,call: PairCall,right: Tree.Tree<Element>,left: Tree.Tree<Element>) -> Operations.Outcome<PairOwners(Element)>:
  PairCall{valid,count,source,target,right_length,left_length} = call
  match valid count:
    case True{} Some{count}:
      Views.View{axes,+destination} = target
      +source = source
      Operations.Accepted{combined_pair(~Element,left_length,right_length,AccessLaws.combine_model(~Element,~operation,
        Access.zero_contiguous(source) && U32.is_zero(destination),U32.to_nat(count),source,destination,right,left))}
    case valid count: Operations.Rejected{(Owner{Tree.pack(Element,left),left_length},Owner{Tree.pack(Element,right),right_length})}

def combine_proposition(~Element: Data,~operation: Element -> Element -> Element,call: PairCall,-right: Array<Element>,-left: Array<Element>,
  -right_tree: Tree.Tree<Element>,-left_tree: Tree.Tree<Element>) -> Type:
  PairCall{valid,count,source,target,right_length,left_length} = call
  @right_certificate: Certificate.Certificate<Element,right> -> @left_certificate: Certificate.Certificate<Element,left> ->
  {observe_pair_outcome(~Element,Operations.combine_when(~Element,~operation,valid,count,source,target,
      Storage.Buffer{left,left_length,left_certificate},Storage.Buffer{right,right_length,right_certificate})) ==
    combine_model(~Element,~operation,PairCall{valid,count,source,target,right_length,left_length},right_tree,left_tree) : Operations.Outcome<PairOwners(Element)>}

def packed_combine(~Element: Data,~operation: Element -> Element -> Element,+call: PairCall,+right: Tree.Tree<Element>,+left: Tree.Tree<Element>) ->
  combine_proposition(~Element,~operation,call,Tree.pack(Element,right),Tree.pack(Element,left),right,left):
  PairCall{valid,count,+source,target,+right_length,+left_length} = call
  match valid count:
    case False{} None{}: right_certificate => left_certificate => {==}
    case False{} Some{count}: right_certificate => left_certificate => {==}
    case True{} None{}: right_certificate => left_certificate => {==}
    case True{} Some{+count}:
      Views.View{axes,+destination} = target
      +fast = Access.zero_contiguous(source) && U32.is_zero(destination)
      right_certificate => left_certificate =>
        Equal.cong(PairOwners(Element),Operations.Outcome<PairOwners(Element)>,value => Operations.Accepted{value},
        observe_pair(~Element,Operations.combine_buffers(~Element,left_length,right_length,
          Access.combine_run(~Element,~operation,fast,U32.to_nat(count),source,destination,Tree.pack(Element,right),Tree.pack(Element,left)),
          AccessCertificate.combine(~Element,~operation,fast,U32.to_nat(count),source,destination,Tree.pack(Element,right),Tree.pack(Element,left),right_certificate,left_certificate))),
        combined_pair(~Element,left_length,right_length,AccessLaws.combine_model(~Element,~operation,fast,U32.to_nat(count),source,destination,right,left)),
        combine_finished(~Element,left_length,right_length,
          Access.combine_run(~Element,~operation,fast,U32.to_nat(count),source,destination,Tree.pack(Element,right),Tree.pack(Element,left)),
          AccessCertificate.combine(~Element,~operation,fast,U32.to_nat(count),source,destination,Tree.pack(Element,right),Tree.pack(Element,left),right_certificate,left_certificate),
          AccessLaws.combine_model(~Element,~operation,fast,U32.to_nat(count),source,destination,right,left),AccessLaws.combine(~Element,~operation,fast,U32.to_nat(count),source,destination,right,left)))

def combine_denotation(~Element: Data,~operation: Element -> Element -> Element,left: Storage.Buffer<Element>,+left_view: Views.View,
  right: Storage.Buffer<Element>,+right_view: Views.View) -> Operations.Outcome<PairOwners(Element)>:
  Storage.Buffer{left,+left_length,left_certificate} = left
  Storage.Buffer{right,+right_length,right_certificate} = right
  combine_model(~Element,~operation,PairCall{Shapes.same(Views.shape(left_view),Views.shape(right_view)) && Views.is_contiguous(left_view) &&
    Operations.valid_buffer(left_length,Storage.capacity(~Element,left,left_certificate),left_view) &&
    Operations.valid_buffer(right_length,Storage.capacity(~Element,right,right_certificate),right_view),Views.numel(left_view),right_view,left_view,right_length,left_length},
    Tree.reflect(Element,right),Tree.reflect(Element,left))

def combine(~Element: Data,~operation: Element -> Element -> Element,left: Storage.Buffer<Element>,+left_view: Views.View,
  right: Storage.Buffer<Element>,+right_view: Views.View) ->
  {observe_pair_outcome(~Element,Operations.combine(~Element,~operation,left,left_view,right,right_view)) ==
    combine_denotation(~Element,~operation,left,left_view,right,right_view) : Operations.Outcome<PairOwners(Element)>}:
  Storage.Buffer{left,+left_length,left_certificate} = left
  Certificate.Certificate{+left_certificate_depth,+left_certificate_equation} = left_certificate
  Storage.Buffer{right,+right_length,right_certificate} = right
  Certificate.Certificate{+right_certificate_depth,+right_certificate_equation} = right_certificate
  +call = {PairCall{Shapes.same(Views.shape(left_view),Views.shape(right_view)) && Views.is_contiguous(left_view) &&
    Operations.valid_buffer(left_length,Storage.capacity(~Element,left,Certificate.Certificate{left_certificate_depth,left_certificate_equation}),left_view) &&
    Operations.valid_buffer(right_length,Storage.capacity(~Element,right,Certificate.Certificate{right_certificate_depth,right_certificate_equation}),right_view),Views.numel(left_view),right_view,left_view,right_length,left_length} : PairCall}
  StorageLaws.lift_pair_observation(~Element,~PairCall,~combine_proposition(~Element,~operation),call,right,left,
    StorageLaws.observe(Element,right),StorageLaws.observe(Element,left),right => left => packed_combine(~Element,~operation,call,right,left))(Certificate.Certificate{right_certificate_depth,right_certificate_equation})(Certificate.Certificate{left_certificate_depth,left_certificate_equation})

type UnaryCall is Data:
  UnaryCall{valid: Bool,count: Maybe<&2,U32>,view: Views.View,length: Nat}

def observe_outcome(~Element: Data,result: Operations.Outcome<Storage.Buffer<Element>>) -> Operations.Outcome<Owner<&1,Array<Element>>>:
  match result:
    case Operations.Accepted{value}: Operations.Accepted{observe_owner(~Element,value)}
    case Operations.Rejected{value}: Operations.Rejected{observe_owner(~Element,value)}

def apply_model(~Element: Data,~operation: Element -> Element,call: UnaryCall,tree: Tree.Tree<Element>) -> Operations.Outcome<Owner<&1,Array<Element>>>:
  UnaryCall{valid,count,view,length} = call
  match valid count:
    case True{} Some{count}:
      Views.View{axes,offset} = view
      Operations.Accepted{Owner{Tree.pack(Element,AccessLaws.apply_model(~Element,~operation,U32.to_nat(count),offset,tree)),length}}
    case valid count: Operations.Rejected{Owner{Tree.pack(Element,tree),length}}

def apply_proposition(~Element: Data,~operation: Element -> Element,call: UnaryCall,-storage: Array<Element>,-tree: Tree.Tree<Element>) -> Type:
  UnaryCall{valid,count,view,length} = call
  @certificate: Certificate.Certificate<Element,storage> ->
  {observe_outcome(~Element,Operations.apply_when(~Element,~operation,valid,count,view,Storage.Buffer{storage,length,certificate})) ==
    apply_model(~Element,~operation,UnaryCall{valid,count,view,length},tree) : Operations.Outcome<Owner<&1,Array<Element>>>}

def packed_apply(~Element: Data,~operation: Element -> Element,+call: UnaryCall,+tree: Tree.Tree<Element>) ->
  apply_proposition(~Element,~operation,call,Tree.pack(Element,tree),tree):
  UnaryCall{valid,count,view,+length} = call
  match valid count:
    case False{} None{}: certificate => {==}
    case False{} Some{count}: certificate => {==}
    case True{} None{}: certificate => {==}
    case True{} Some{+count}:
      Views.View{axes,+offset} = view
      certificate => Equal.cong(Array<Element>,Operations.Outcome<Owner<&1,Array<Element>>>,storage => Operations.Accepted{Owner{storage,length}},
        Traversal.map_in_place(~Element,~Unit,~Access.apply_element(~Element,~operation),U32.to_nat(count),offset,Unit{},Tree.pack(Element,tree)),
        Tree.pack(Element,AccessLaws.apply_model(~Element,~operation,U32.to_nat(count),offset,tree)),AccessLaws.apply(~Element,~operation,U32.to_nat(count),offset,tree))

def apply_denotation(~Element: Data,~operation: Element -> Element,buffer: Storage.Buffer<Element>,+view: Views.View) -> Operations.Outcome<Owner<&1,Array<Element>>>:
  Storage.Buffer{storage,+length,certificate} = buffer
  apply_model(~Element,~operation,UnaryCall{Views.is_contiguous(view) && Operations.valid_buffer(length,Storage.capacity(~Element,storage,certificate),view),Views.numel(view),view,length},Tree.reflect(Element,storage))

def apply(~Element: Data,~operation: Element -> Element,buffer: Storage.Buffer<Element>,+view: Views.View) ->
  {observe_outcome(~Element,Operations.apply(~Element,~operation,buffer,view)) == apply_denotation(~Element,~operation,buffer,view) : Operations.Outcome<Owner<&1,Array<Element>>>}:
  Storage.Buffer{storage,+length,certificate} = buffer
  Certificate.Certificate{+certificate_depth,+certificate_equation} = certificate
  +call = {UnaryCall{Views.is_contiguous(view) && Operations.valid_buffer(length,Storage.capacity(~Element,storage,Certificate.Certificate{certificate_depth,certificate_equation}),view),Views.numel(view),view,length} : UnaryCall}
  StorageLaws.lift_observation(~Element,~UnaryCall,~apply_proposition(~Element,~operation),call,storage,StorageLaws.observe(Element,storage),
    tree => packed_apply(~Element,~operation,call,tree))(Certificate.Certificate{certificate_depth,certificate_equation})

type ReductionCall<-Accumulator: Data> is Data:
  ReductionCall{call: UnaryCall,initial: Accumulator}

def ReductionOwner(-Element: Data,-Accumulator: Data) -> Type:
  Owner<&1,Array<Element>> & Accumulator

def observe_reduction(~Element: Data,~Accumulator: Data,pair: Storage.Buffer<Element> & Accumulator) -> ReductionOwner(Element,Accumulator):
  (buffer,value) = pair
  (observe_owner(~Element,buffer),value)

def observe_reduction_outcome(~Element: Data,~Accumulator: Data,result: Operations.Outcome<(Storage.Buffer<Element> & Accumulator)>) -> Operations.Outcome<ReductionOwner(Element,Accumulator)>:
  match result:
    case Operations.Accepted{value}: Operations.Accepted{observe_reduction(~Element,~Accumulator,value)}
    case Operations.Rejected{value}: Operations.Rejected{observe_reduction(~Element,~Accumulator,value)}

def reduced_owner(~Element: Data,~Accumulator: Data,length: Nat,result: AccessLaws.FoldResult<Tree.Tree<Element>,Accumulator>) -> ReductionOwner(Element,Accumulator):
  AccessLaws.FoldResult{tree,value} = result
  (Owner{Tree.pack(Element,tree),length},value)

def reduce_owned(~Element: Data,~Accumulator: Data,length: Nat,result: Array<Element> & Accumulator) -> ReductionOwner(Element,Accumulator):
  (storage,value) = result
  (Owner{storage,length},value)

def reduce_finished(~Element: Data,~Accumulator: Data,+length: Nat,result: Array<Element> & Accumulator,
  certificate: Certificate.Certificate<Element,TraversalWitness.first(~Array<Element>,~Accumulator,result)>,+model: AccessLaws.FoldResult<Tree.Tree<Element>,Accumulator>,
  equation: {result == AccessLaws.pack_fold(~Array<Element>,~Tree.Tree<Element>,~Accumulator,~ArrayLaws.pack_tree(~Element),model) : Array<Element> & Accumulator}) ->
  {observe_reduction(~Element,~Accumulator,Operations.reduce_buffer(~Element,~Accumulator,length,result,certificate)) == reduced_owner(~Element,~Accumulator,length,model) : ReductionOwner(Element,Accumulator)}:
  (storage,value) = result
  AccessLaws.FoldResult{+tree,+model_value} = model
  Equal.cong(Array<Element> & Accumulator,ReductionOwner(Element,Accumulator),result => reduce_owned(~Element,~Accumulator,length,result),
    (storage,value),(Tree.pack(Element,tree),model_value),equation)

def reduce_model(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,request: ReductionCall<Accumulator>,tree: Tree.Tree<Element>) -> Operations.Outcome<ReductionOwner(Element,Accumulator)>:
  ReductionCall{UnaryCall{valid,count,view,length},initial} = request
  match valid count:
    case True{} Some{count}:
      +view = view
      Operations.Accepted{reduced_owner(~Element,~Accumulator,length,AccessLaws.reduce_model(~Element,~Accumulator,~operation,Views.is_contiguous(view),U32.to_nat(count),view,tree,initial))}
    case valid count: Operations.Rejected{(Owner{Tree.pack(Element,tree),length},initial)}

def reduce_proposition(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,
  request: ReductionCall<Accumulator>,-storage: Array<Element>,-tree: Tree.Tree<Element>) -> Type:
  ReductionCall{UnaryCall{valid,count,view,length},initial} = request
  @certificate: Certificate.Certificate<Element,storage> ->
  {observe_reduction_outcome(~Element,~Accumulator,Operations.reduce_when(~Element,~Accumulator,~operation,valid,count,view,Storage.Buffer{storage,length,certificate},initial)) ==
    reduce_model(~Element,~Accumulator,~operation,ReductionCall{UnaryCall{valid,count,view,length},initial},tree) : Operations.Outcome<ReductionOwner(Element,Accumulator)>}

def packed_reduce(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,+request: ReductionCall<Accumulator>,+tree: Tree.Tree<Element>) ->
  reduce_proposition(~Element,~Accumulator,~operation,request,Tree.pack(Element,tree),tree):
  ReductionCall{UnaryCall{valid,count,+view,+length},+initial} = request
  match valid count:
    case False{} None{}: certificate => {==}
    case False{} Some{count}: certificate => {==}
    case True{} None{}: certificate => {==}
    case True{} Some{+count}:
      +fast = Views.is_contiguous(view)
      certificate => Equal.cong(ReductionOwner(Element,Accumulator),Operations.Outcome<ReductionOwner(Element,Accumulator)>,value => Operations.Accepted{value},
        observe_reduction(~Element,~Accumulator,Operations.reduce_buffer(~Element,~Accumulator,length,
          Access.reduce_run(~Element,~Accumulator,~operation,fast,U32.to_nat(count),view,Tree.pack(Element,tree),initial),
          AccessCertificate.reduce(~Element,~Accumulator,~operation,fast,U32.to_nat(count),view,Tree.pack(Element,tree),initial,certificate))),
        reduced_owner(~Element,~Accumulator,length,AccessLaws.reduce_model(~Element,~Accumulator,~operation,fast,U32.to_nat(count),view,tree,initial)),
        reduce_finished(~Element,~Accumulator,length,Access.reduce_run(~Element,~Accumulator,~operation,fast,U32.to_nat(count),view,Tree.pack(Element,tree),initial),
          AccessCertificate.reduce(~Element,~Accumulator,~operation,fast,U32.to_nat(count),view,Tree.pack(Element,tree),initial,certificate),
          AccessLaws.reduce_model(~Element,~Accumulator,~operation,fast,U32.to_nat(count),view,tree,initial),AccessLaws.reduce(~Element,~Accumulator,~operation,fast,U32.to_nat(count),view,tree,initial)))

def reduce_denotation(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,
  buffer: Storage.Buffer<Element>,+view: Views.View,initial: Accumulator) -> Operations.Outcome<ReductionOwner(Element,Accumulator)>:
  Storage.Buffer{storage,+length,certificate} = buffer
  reduce_model(~Element,~Accumulator,~operation,ReductionCall{UnaryCall{Operations.valid_buffer(length,Storage.capacity(~Element,storage,certificate),view),Views.numel(view),view,length},initial},Tree.reflect(Element,storage))

def reduce(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,buffer: Storage.Buffer<Element>,+view: Views.View,+initial: Accumulator) ->
  {observe_reduction_outcome(~Element,~Accumulator,Operations.reduce(~Element,~Accumulator,~operation,buffer,view,initial)) ==
    reduce_denotation(~Element,~Accumulator,~operation,buffer,view,initial) : Operations.Outcome<ReductionOwner(Element,Accumulator)>}:
  Storage.Buffer{storage,+length,certificate} = buffer
  Certificate.Certificate{+certificate_depth,+certificate_equation} = certificate
  +request = {ReductionCall{UnaryCall{Operations.valid_buffer(length,Storage.capacity(~Element,storage,Certificate.Certificate{certificate_depth,certificate_equation}),view),Views.numel(view),view,length},initial} : ReductionCall<Accumulator>}
  StorageLaws.lift_observation(~Element,~ReductionCall<Accumulator>,~reduce_proposition(~Element,~Accumulator,~operation),request,storage,StorageLaws.observe(Element,storage),
    tree => packed_reduce(~Element,~Accumulator,~operation,request,tree))(Certificate.Certificate{certificate_depth,certificate_equation})
def apply_indexed_model(~Element: Data,~operation: U32 -> Element -> Element,valid: Bool,count: Maybe<&2,U32>,view: Views.View,
  input: Tree.Tree<Element>,input_length: Nat,output: Tree.Tree<Element>,output_length: Nat) -> Operations.Outcome<PairOwners(Element)>:
  match valid count:
    case True{} Some{count}:
      Operations.Accepted{modeled_pair(~Element,input_length,output_length,AccessLaws.apply_indexed_model(~Element,~operation,U32.to_nat(count),view,input,output))}
    case valid count:
      Operations.Rejected{(Owner{Tree.pack(Element,input),input_length},Owner{Tree.pack(Element,output),output_length})}

def packed_apply_indexed(~Element: Data,~operation: U32 -> Element -> Element,+valid: Bool,+count: Maybe<&2,U32>,+view: Views.View,
  +input: Tree.Tree<Element>,+input_length: Nat,+output: Tree.Tree<Element>,+output_length: Nat,
  +input_certificate: Certificate.Certificate<Element,Tree.pack(Element,input)>,
  +output_certificate: Certificate.Certificate<Element,Tree.pack(Element,output)>) ->
  {observe_pair_outcome(~Element,Operations.apply_indexed_when(~Element,~operation,valid,count,view,
      Storage.Buffer{Tree.pack(Element,input),input_length,input_certificate},Storage.Buffer{Tree.pack(Element,output),output_length,output_certificate})) ==
    apply_indexed_model(~Element,~operation,valid,count,view,input,input_length,output,output_length) : Operations.Outcome<PairOwners(Element)>}:
  match valid count:
    case False{} None{}: {==}
    case False{} Some{count}: {==}
    case True{} None{}: {==}
    case True{} Some{+count}:
      Equal.cong(PairOwners(Element),Operations.Outcome<PairOwners(Element)>,value => Operations.Accepted{value},
        observe_pair(~Element,Operations.pair_buffers(~Element,input_length,output_length,
          Access.apply_indexed_run(~Element,~operation,U32.to_nat(count),view,Tree.pack(Element,input),Tree.pack(Element,output)),
          AccessCertificate.apply_indexed(~Element,~operation,U32.to_nat(count),view,Tree.pack(Element,input),Tree.pack(Element,output),input_certificate,output_certificate))),
        modeled_pair(~Element,input_length,output_length,AccessLaws.apply_indexed_model(~Element,~operation,U32.to_nat(count),view,input,output)),
        pair_finished(~Element,input_length,output_length,
          Access.apply_indexed_run(~Element,~operation,U32.to_nat(count),view,Tree.pack(Element,input),Tree.pack(Element,output)),
          AccessCertificate.apply_indexed(~Element,~operation,U32.to_nat(count),view,Tree.pack(Element,input),Tree.pack(Element,output),input_certificate,output_certificate),
          AccessLaws.apply_indexed_model(~Element,~operation,U32.to_nat(count),view,input,output),
          AccessLaws.apply_indexed(~Element,~operation,U32.to_nat(count),view,input,output)))

def observed_apply_indexed(~Element: Data,~operation: U32 -> Element -> Element,+valid: Bool,+count: Maybe<&2,U32>,+view: Views.View,
  -input: Array<Element>,+input_length: Nat,-output: Array<Element>,+output_length: Nat,
  input_observation: StorageLaws.Observation<Element,input>,output_observation: StorageLaws.Observation<Element,output>) ->
  @input_certificate: Certificate.Certificate<Element,input> -> @output_certificate: Certificate.Certificate<Element,output> ->
  {observe_pair_outcome(~Element,Operations.apply_indexed_when(~Element,~operation,valid,count,view,
      Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})) ==
    apply_indexed_model(~Element,~operation,valid,count,view,Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length) :
    Operations.Outcome<PairOwners(Element)>}:
  StorageLaws.Observation{+input_tree,input_packed,input_reflected} = input_observation
  StorageLaws.Observation{+output_tree,output_packed,output_reflected} = output_observation
  %input_reflected :
    @input_certificate: Certificate.Certificate<Element,input> -> @output_certificate: Certificate.Certificate<Element,output> ->
    {observe_pair_outcome(~Element,Operations.apply_indexed_when(~Element,~operation,valid,count,view,
        Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})) ==
      apply_indexed_model(~Element,~operation,valid,count,view,_,input_length,Tree.reflect(Element,output),output_length) : Operations.Outcome<PairOwners(Element)>}
  %output_reflected :
    @input_certificate: Certificate.Certificate<Element,input> -> @output_certificate: Certificate.Certificate<Element,output> ->
    {observe_pair_outcome(~Element,Operations.apply_indexed_when(~Element,~operation,valid,count,view,
        Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})) ==
      apply_indexed_model(~Element,~operation,valid,count,view,input_tree,input_length,_,output_length) : Operations.Outcome<PairOwners(Element)>}
  %input_packed :
    @input_certificate: Certificate.Certificate<Element,_> -> @output_certificate: Certificate.Certificate<Element,output> ->
    {observe_pair_outcome(~Element,Operations.apply_indexed_when(~Element,~operation,valid,count,view,
        Storage.Buffer{_,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})) ==
      apply_indexed_model(~Element,~operation,valid,count,view,input_tree,input_length,output_tree,output_length) : Operations.Outcome<PairOwners(Element)>}
  %output_packed :
    @input_certificate: Certificate.Certificate<Element,Tree.pack(Element,input_tree)> -> @output_certificate: Certificate.Certificate<Element,_> ->
    {observe_pair_outcome(~Element,Operations.apply_indexed_when(~Element,~operation,valid,count,view,
        Storage.Buffer{Tree.pack(Element,input_tree),input_length,input_certificate},Storage.Buffer{_,output_length,output_certificate})) ==
      apply_indexed_model(~Element,~operation,valid,count,view,input_tree,input_length,output_tree,output_length) : Operations.Outcome<PairOwners(Element)>}
  input_certificate => output_certificate =>
    packed_apply_indexed(~Element,~operation,valid,count,view,input_tree,input_length,output_tree,output_length,input_certificate,output_certificate)

def apply_indexed_denotation(~Element: Data,~operation: U32 -> Element -> Element,input: Storage.Buffer<Element>,+view: Views.View,
  output: Storage.Buffer<Element>) -> Operations.Outcome<PairOwners(Element)>:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Storage.Buffer{output,+output_length,output_certificate} = output
  apply_indexed_model(~Element,~operation,Views.is_contiguous(view) &&
    Operations.valid_buffer(input_length,Storage.capacity(~Element,input,input_certificate),view) &&
    Operations.holds(Views.numel(view),output_length,Storage.capacity(~Element,output,output_certificate)),Views.numel(view),view,
    Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length)

# Output element i is operation(i, input element i) whenever the guard holds.
def apply_indexed(~Element: Data,~operation: U32 -> Element -> Element,input: Storage.Buffer<Element>,+view: Views.View,output: Storage.Buffer<Element>) ->
  {observe_pair_outcome(~Element,Operations.apply_indexed(~Element,~operation,input,view,output)) ==
    apply_indexed_denotation(~Element,~operation,input,view,output) : Operations.Outcome<PairOwners(Element)>}:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Certificate.Certificate{+input_certificate_depth,+input_certificate_equation} = input_certificate
  Storage.Buffer{output,+output_length,output_certificate} = output
  Certificate.Certificate{+output_certificate_depth,+output_certificate_equation} = output_certificate
  observed_apply_indexed(~Element,~operation,Views.is_contiguous(view) &&
    Operations.valid_buffer(input_length,Storage.capacity(~Element,input,Certificate.Certificate{input_certificate_depth,input_certificate_equation}),view) &&
    Operations.holds(Views.numel(view),output_length,Storage.capacity(~Element,output,Certificate.Certificate{output_certificate_depth,output_certificate_equation})),
    Views.numel(view),view,input,input_length,output,output_length,StorageLaws.observe(Element,input),StorageLaws.observe(Element,output))(Certificate.Certificate{input_certificate_depth,input_certificate_equation})(Certificate.Certificate{output_certificate_depth,output_certificate_equation})
