# Binary band callers: aligned layouts, owners, dense and empty operands,
# fallbacks, status and the public contracts.
import Base
import ./tensor_band_combine_leaf.bend as BandCombineLeaf
import ./tensor_band_model.bend as Model
import ../tensor_band.bend as Bands
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ./tensor_band_dense.bend as BandDense
import ./storage_shape.bend as Shape
import ./storage_buffer_model.bend as Buffer
import ./tensor_band_inspection.bend as Inspection
import ../machine_limits.bend as Machine
import ./storage_tree_model.bend as Tree
import ./tensor_band_plan.bend as BandPlan
import ../tensor.bend as Tensors
import ./tensor_band_count.bend as BandCount
import ../default_machine_limits.bend as Default
import ./tensor_band_gather.bend as BandGather
import ../tensor_band_schedule.bend as BandSchedule
import ../tensor_shape.bend as Shapes
import ./tensor_band_map.bend as BandMap
import ./tensor_band_shape_count.bend as BandShapeCount
import ./tensor_count_refinement.bend as Count
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ./boolean_logic.bend as Boolean
import ./storage_refinement.bend as StorageLaws
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_vector_descriptor_refinement.bend as TensorVectorDescriptorRefinement
import ./tensor_vector_view_refinement.bend as Vector
import ../storage_buffer.bend as Storage
import ../tensor_window.bend as Window

# The actual binary leaf obtains its checked count from the aligned owners'
# metadata and Ready evidence, then inherits pointwise values and allocation.
# Successful descriptor computations follow the actual owners' count evidence.
def leaf_contract_count_exact(+geometry: Bands.Geometry,+rows: U32,+length: Nat,evidence: BandShape.Count<geometry,rows,length>) ->
  {U32.to_nat(Bands.local_count(geometry,rows)) == length : Nat}:
  BandShape.Count{exact,allocatable} = evidence
  exact

def equation(~Element: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+geometry: Bands.Geometry,+first: U32,+rows: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},
  +left_metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,left)),+right_metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,right)),
  +left_ready: Buffer.Sized<Element,left>,+right_ready: Buffer.Sized<Element,right>) ->
  {Buffer.contents(~Element,Bands.combine_leaf_with(~Element,~kernel,Bands.local_count(geometry,rows),Buffer.pack(~Element,left),Buffer.pack(~Element,right))) ==
    BandCombineLeaf.leaf_content(~Element,~operation,Buffer.content(~Element,left),Buffer.content(~Element,right)) : Buffer.Content<Element>}:
  +left_exact = leaf_contract_count_exact(geometry,rows,Buffer.length(~Element,left),BandShape.count(~Element,geometry,first,rows,left,width_positive,left_metadata,left_ready))
  +right_exact = leaf_contract_count_exact(geometry,rows,Buffer.length(~Element,right),BandShape.count(~Element,geometry,first,rows,right,width_positive,right_metadata,right_ready))
  +descriptor = TensorVectorDescriptorRefinement.from_count(geometry,rows,Buffer.length(~Element,left),BandShape.count(~Element,geometry,first,rows,left,width_positive,left_metadata,left_ready))
  Equal.trans(Buffer.Content<Element>,
    Buffer.contents(~Element,Bands.combine_leaf_with(~Element,~kernel,Bands.local_count(geometry,rows),Buffer.pack(~Element,left),Buffer.pack(~Element,right))),
    BandCombineLeaf.refinement_leaf_model(~Element,~operation,Bands.local_count(geometry,rows),left,right),
    BandCombineLeaf.leaf_content(~Element,~operation,Buffer.content(~Element,left),Buffer.content(~Element,right)),
    BandCombineLeaf.refinement_leaf(~Element,~operation,~kernel,~kernel_law,Bands.local_count(geometry,rows),left,right),
    BandCombineLeaf.checked_leaf_model(~Element,~operation,Bands.local_count(geometry,rows),left,right,left_exact,right_exact,left_ready,right_ready,descriptor))

def LeafContract(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+content: Buffer.Content<Element>) -> Data:
  Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,Model.RowBand{content,first,rows}),
    Shape.Both<Buffer.ContentReady<Element,content>,BandShape.Leaf(geometry,first,rows,Buffer.content_length(~Element,content))>>

def leaf(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+first: U32,+rows: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},
  +left_metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,left)),+right_metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,right)),
  +left_ready: Buffer.Sized<Element,left>,+right_ready: Buffer.Sized<Element,right>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.RowBand{Buffer.content(~Element,left),first,rows}),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.RowBand{Buffer.content(~Element,right),first,rows})) ->
  LeafContract(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,first,rows,
    Buffer.contents(~Element,Bands.combine_leaf_with(~Element,~kernel,Bands.local_count(geometry,rows),Buffer.pack(~Element,left),Buffer.pack(~Element,right)))):
  match left right:
    case Buffer.Model{+left_tree,+left_length,+left_certificate} Buffer.Model{+right_tree,+right_length,+right_certificate}:
      +left = {Buffer.Model{left_tree,left_length,left_certificate} : Buffer.Model<Element>}
      +right = {Buffer.Model{right_tree,right_length,right_certificate} : Buffer.Model<Element>}
      %Equal.sym(Buffer.Content<Element>,
        Buffer.contents(~Element,Bands.combine_leaf_with(~Element,~kernel,Bands.local_count(geometry,rows),Buffer.pack(~Element,left),Buffer.pack(~Element,right))),
        BandCombineLeaf.leaf_content(~Element,~operation,Buffer.content(~Element,left),Buffer.content(~Element,right)),
        equation(~Element,~operation,~kernel,~kernel_law,geometry,first,rows,left,right,width_positive,left_metadata,right_metadata,left_ready,right_ready)) :
        LeafContract(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,first,rows,_)
      Shape.Both{BandCombineLeaf.semantics_leaf(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,geometry,first,rows,left,right,
          left_ready,left_values,right_values),
        Shape.Both{BandCombineLeaf.leaf_ready(~Element,~operation,left,right,left_ready),left_metadata}}

# Common aligned layouts discharge each leaf's real count and descriptor.
# The complete native tree retains exact values, storage readiness and shape.
def PipelineContract(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+content: Model.Content<Element>) -> Data:
  Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,content),
    Shape.Both<Model.ContentReady(~Element,content),BandShape.Contents(~Element,geometry,content)>>

def joined(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+left: Model.Content<Element>,+right: Model.Content<Element>,
  left_evidence: PipelineContract(~Element,~Context,~value,context,geometry,left),right_evidence: PipelineContract(~Element,~Context,~value,context,geometry,right)) ->
  PipelineContract(~Element,~Context,~value,context,geometry,Model.Joined{left,right}):
  Shape.Both{left_values,Shape.Both{left_ready,left_metadata}} = left_evidence
  Shape.Both{right_values,Shape.Both{right_ready,right_metadata}} = right_evidence
  Shape.Both{Shape.Both{left_values,right_values},Shape.Both{Shape.Both{left_ready,right_ready},Shape.Both{left_metadata,right_metadata}}}

def leaf_contract(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,content: Buffer.Content<Element>,
  evidence: LeafContract(~Element,~Context,~value,context,geometry,first,rows,content)) ->
  PipelineContract(~Element,~Context,~value,context,geometry,Model.RowBand{content,first,rows}):
  Buffer.Content{tree,length} = content
  evidence

def pipeline_model(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},common: BandCombineLeaf.Common(~Element,left,right),
  left_metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,left)),right_metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,right)),
  left_ready: Model.Ready(~Element,left),right_ready: Model.Ready(~Element,right),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left)),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right))) ->
  PipelineContract(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
    BandCombineLeaf.model(~Element,~operation,geometry,left,right)):
  match left right:
    case Model.Leaf{Buffer.Model{+left_tree,+left_length,+left_certificate},+first,+rows}
      Model.Leaf{Buffer.Model{+right_tree,+right_length,+right_certificate},+other_first,+other_rows}:
      Shape.Both{+same_first,+same_rows} = common
      +left = {Buffer.Model{left_tree,left_length,left_certificate} : Buffer.Model<Element>}
      +right = {Buffer.Model{right_tree,right_length,right_certificate} : Buffer.Model<Element>}
      +right_metadata = {
        %Equal.sym(U32,first,other_first,same_first) : BandShape.Leaf(geometry,_,other_rows,Buffer.length(~Element,right))
        right_metadata : BandShape.Leaf(geometry,first,other_rows,Buffer.length(~Element,right))}
      +right_metadata = {
        %Equal.sym(U32,rows,other_rows,same_rows) : BandShape.Leaf(geometry,first,_,Buffer.length(~Element,right))
        right_metadata : BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,right))}
      +right_values = {
        %Equal.sym(U32,first,other_first,same_first) : BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.RowBand{Buffer.content(~Element,right),_,other_rows})
        right_values : BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.RowBand{Buffer.content(~Element,right),first,other_rows})}
      +right_values = {
        %Equal.sym(U32,rows,other_rows,same_rows) : BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.RowBand{Buffer.content(~Element,right),first,_})
        right_values : BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.RowBand{Buffer.content(~Element,right),first,rows})}
      leaf_contract(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,first,rows,
        BandCombineLeaf.refinement_leaf_model(~Element,~operation,Bands.local_count(geometry,rows),left,right),
        %BandCombineLeaf.refinement_leaf(~Element,~operation,~kernel,~kernel_law,Bands.local_count(geometry,rows),left,right) :
          LeafContract(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,first,rows,_)
        leaf(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,first,rows,left,right,
          width_positive,left_metadata,right_metadata,left_ready,right_ready,left_values,right_values))
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      Shape.Both{left_common,right_common} = common
      Shape.Both{left_left_metadata,left_right_metadata} = left_metadata
      Shape.Both{right_left_metadata,right_right_metadata} = right_metadata
      Shape.Both{left_left_ready,left_right_ready} = left_ready
      Shape.Both{right_left_ready,right_right_ready} = right_ready
      Shape.Both{left_left_values,left_right_values} = left_values
      Shape.Both{right_left_values,right_right_values} = right_values
      joined(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
        BandCombineLeaf.model(~Element,~operation,geometry,left,other_left),BandCombineLeaf.model(~Element,~operation,geometry,right,other_right),
        pipeline_model(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,left,other_left,width_positive,left_common,
          left_left_metadata,right_left_metadata,left_left_ready,right_left_ready,left_left_values,right_left_values),
        pipeline_model(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,right,other_right,width_positive,right_common,
          left_right_metadata,right_right_metadata,left_right_ready,right_right_ready,left_right_values,right_right_values))
    case Model.Leaf{+buffer,+first,+rows} Model.Fork{+other_left,+other_right}:
      Empty.absurd(PipelineContract(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
        BandCombineLeaf.model(~Element,~operation,geometry,Model.Leaf{buffer,first,rows},Model.Fork{other_left,other_right})),common)
    case Model.Fork{+left,+right} Model.Leaf{+buffer,+first,+rows}:
      Empty.absurd(PipelineContract(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
        BandCombineLeaf.model(~Element,~operation,geometry,Model.Fork{left,right},Model.Leaf{buffer,first,rows})),common)

def pipeline_combine(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+parallel: Bool,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},same: {Model.layout(~Element,left) == Model.layout(~Element,right) : Bands.Plan},
  left_metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,left)),right_metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,right)),
  left_ready: Model.Ready(~Element,left),right_ready: Model.Ready(~Element,right),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left)),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right))) ->
  PipelineContract(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
    Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right)))):
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right))),
    BandCombineLeaf.model(~Element,~operation,geometry,left,right),BandCombineLeaf.combine(~Element,~operation,~kernel,~kernel_law,geometry,parallel,left,right)) :
    PipelineContract(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,_)
  pipeline_model(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,left,right,width_positive,BandCombineLeaf.matched(~Element,left,right,same),
    left_metadata,right_metadata,left_ready,right_ready,left_values,right_values)

def well_formed(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+content: Model.Content<Element>,
  evidence: PipelineContract(~Element,~Context,~value,context,geometry,content),coverage: BandPlan.Coverage(Model.content_layout(~Element,content),0n,Inspection.total_rows(geometry))) ->
  Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,content),BandShape.WellFormed<Element,geometry,content>>:
  Shape.Both{values,Shape.Both{ready,metadata}} = evidence
  Shape.Both{values,BandShape.WellFormed{metadata,ready,coverage}}

def combine_well_formed(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+parallel: Bool,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},same: {Model.layout(~Element,left) == Model.layout(~Element,right) : Bands.Plan},
  left_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,right_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,right)>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left)),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right))) ->
  Shape.Both<BandValues.Values(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
      Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right)))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right)))>>:
  BandShape.WellFormed{left_metadata,left_ready,coverage} = left_shape
  BandShape.WellFormed{right_metadata,right_ready,right_coverage} = right_shape
  well_formed(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
    Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right))),
    pipeline_combine(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,parallel,left,right,width_positive,same,
      left_metadata,right_metadata,Model.ready(~Element,left,left_ready),Model.ready(~Element,right,right_ready),left_values,right_values),
    %Equal.sym(Bands.Plan,Model.content_layout(~Element,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right)))),
      Model.layout(~Element,left),BandCombineLeaf.combine_layout(~Element,~operation,~kernel,~kernel_law,geometry,parallel,left,right)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
    %Equal.sym(Bands.Plan,Model.layout(~Element,left),Model.content_layout(~Element,Model.content(~Element,left)),Model.layout_matches_content(~Element,left)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
    coverage)

# Observe the actual affine owners once, then transport their checked shape
# and values into the complete binary refinement. No leaf rule is supplied.
def observed_contents(~Element: Data,+model: Model.Model<Element>,-owner: Bands.Bands(~Element),equation: {Model.pack(~Element,model) == owner : Bands.Bands(~Element)}) ->
  {Model.contents(~Element,owner) == Model.content(~Element,model) : Model.Content<Element>}:
  Equal.trans(Model.Content<Element>,Model.contents(~Element,owner),Model.contents(~Element,Model.pack(~Element,model)),Model.content(~Element,model),
    Equal.cong(Bands.Bands(~Element),Model.Content<Element>,bands => Model.contents(~Element,bands),owner,Model.pack(~Element,model),Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),owner,equation)),
    Model.packed_contents(~Element,model))

def observed_layout(~Element: Data,+model: Model.Model<Element>,-owner: Bands.Bands(~Element),equation: {Model.pack(~Element,model) == owner : Bands.Bands(~Element)}) ->
  {Model.layout(~Element,model) == Model.content_layout(~Element,Model.contents(~Element,owner)) : Bands.Plan}:
  Equal.trans(Bands.Plan,Model.layout(~Element,model),Model.content_layout(~Element,Model.content(~Element,model)),Model.content_layout(~Element,Model.contents(~Element,owner)),
    Model.layout_matches_content(~Element,model),
    Equal.sym(Bands.Plan,Model.content_layout(~Element,Model.contents(~Element,owner)),Model.content_layout(~Element,Model.content(~Element,model)),
      Equal.cong(Model.Content<Element>,Bands.Plan,content => Model.content_layout(~Element,content),Model.contents(~Element,owner),Model.content(~Element,model),observed_contents(~Element,model,owner,equation))))

def owners_observed(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+parallel: Bool,-left: Bands.Bands(~Element),-right: Bands.Bands(~Element),
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},+same: {Model.content_layout(~Element,Model.contents(~Element,left)) == Model.content_layout(~Element,Model.contents(~Element,right)) : Bands.Plan},
  left_shape: BandShape.WellFormed<Element,geometry,Model.contents(~Element,left)>,right_shape: BandShape.WellFormed<Element,geometry,Model.contents(~Element,right)>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.contents(~Element,left)),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.contents(~Element,right)),
  left_observation: Model.Observation<Element,left>,right_observation: Model.Observation<Element,right>) ->
  Shape.Both<BandValues.Values(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
      Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))>>:
  Model.Observation{+left_model,+left_equation} = left_observation
  Model.Observation{+right_model,+right_equation} = right_observation
  +left_content = observed_contents(~Element,left_model,left,left_equation)
  +right_content = observed_contents(~Element,right_model,right,right_equation)
  +left_shape = {
    %left_content : BandShape.WellFormed<Element,geometry,_>
    left_shape : BandShape.WellFormed<Element,geometry,Model.content(~Element,left_model)>}
  +right_shape = {
    %right_content : BandShape.WellFormed<Element,geometry,_>
    right_shape : BandShape.WellFormed<Element,geometry,Model.content(~Element,right_model)>}
  +left_values = {
    %left_content : BandValues.Values(~Element,~Left,~left_value,left_context,geometry,_)
    left_values : BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left_model))}
  +right_values = {
    %right_content : BandValues.Values(~Element,~Right,~right_value,right_context,geometry,_)
    right_values : BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right_model))}
  +matched = Equal.trans(Bands.Plan,Model.layout(~Element,left_model),Model.content_layout(~Element,Model.contents(~Element,left)),Model.layout(~Element,right_model),
    observed_layout(~Element,left_model,left,left_equation),
    Equal.trans(Bands.Plan,Model.content_layout(~Element,Model.contents(~Element,left)),Model.content_layout(~Element,Model.contents(~Element,right)),Model.layout(~Element,right_model),
      same,Equal.sym(Bands.Plan,Model.layout(~Element,right_model),Model.content_layout(~Element,Model.contents(~Element,right)),observed_layout(~Element,right_model,right,right_equation))))
  %left_equation :
    Shape.Both<BandValues.Values(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
        Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,_,right))),
      BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,_,right))>>
  %right_equation :
    Shape.Both<BandValues.Values(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
        Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left_model),_))),
      BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left_model),_))>>
  combine_well_formed(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,parallel,left_model,right_model,width_positive,matched,
    left_shape,right_shape,left_values,right_values)

def combine_owners(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+parallel: Bool,left: Bands.Bands(~Element),right: Bands.Bands(~Element),
  width_positive: {BandShape.width_positive(geometry) == True{} : Bool},same: {Model.content_layout(~Element,Model.contents(~Element,left)) == Model.content_layout(~Element,Model.contents(~Element,right)) : Bands.Plan},
  left_shape: BandShape.WellFormed<Element,geometry,Model.contents(~Element,left)>,right_shape: BandShape.WellFormed<Element,geometry,Model.contents(~Element,right)>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.contents(~Element,left)),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.contents(~Element,right))) ->
  Shape.Both<BandValues.Values(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
      Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))>>:
  owners_observed(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,parallel,left,right,width_positive,same,left_shape,right_shape,left_values,right_values,
    Model.observe(~Element,left),Model.observe(~Element,right))

# Both real align calls establish the common layout before any binary leaf.
# The chosen layout obtains allocation/copy facts from an existing owner tree.
def alignment_width_positive(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>) ->
  {BandShape.width_positive(geometry) == True{} : Bool}:
  Bands.Geometry{images,channels,height,width} = geometry
  BandGather.Domain{capacity_ok,channels_positive,height_positive,alignment_width_positive,machine_room,allocation_room} = domain
  alignment_width_positive

def alignment_aligned(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+parallel: Bool,+left_fill: Element,+right_fill: Element,+layout: Bands.Plan,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,
  +left_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,+right_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,right)>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left)),right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right)),
  +target: BandDense.Fits(geometry,layout,capacity,Machine.half_word(limits)),+coverage: BandPlan.Coverage(layout,0n,Inspection.total_rows(geometry))) ->
  Shape.Both<BandValues.Values(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
      Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Bands.align(~Element,geometry,left_fill,layout,Model.pack(~Element,left)),Bands.align(~Element,geometry,right_fill,layout,Model.pack(~Element,right))))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Bands.align(~Element,geometry,left_fill,layout,Model.pack(~Element,left)),Bands.align(~Element,geometry,right_fill,layout,Model.pack(~Element,right))))>>:
  combine_owners(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,parallel,
    Bands.align(~Element,geometry,left_fill,layout,Model.pack(~Element,left)),Bands.align(~Element,geometry,right_fill,layout,Model.pack(~Element,right)),alignment_width_positive(geometry,limits,capacity,domain),
    Equal.trans(Bands.Plan,
      Model.content_layout(~Element,Model.contents(~Element,Bands.align(~Element,geometry,left_fill,layout,Model.pack(~Element,left)))),layout,
      Model.content_layout(~Element,Model.contents(~Element,Bands.align(~Element,geometry,right_fill,layout,Model.pack(~Element,right)))),
      BandCombineLeaf.aligned_layout(~Element,geometry,left_fill,layout,left),Equal.sym(Bands.Plan,
        Model.content_layout(~Element,Model.contents(~Element,Bands.align(~Element,geometry,right_fill,layout,Model.pack(~Element,right)))),layout,BandCombineLeaf.aligned_layout(~Element,geometry,right_fill,layout,right))),
    BandCombineLeaf.aligned_well_formed(~Element,geometry,left_fill,layout,left,capacity,Machine.half_word(limits),left_shape,target,coverage),
    BandCombineLeaf.aligned_well_formed(~Element,geometry,right_fill,layout,right,capacity,Machine.half_word(limits),right_shape,target,coverage),
    BandCombineLeaf.aligned_values(~Element,~Left,~left_value,left_context,geometry,left_fill,layout,left,limits,capacity,domain,left_shape,left_values,target),
    BandCombineLeaf.aligned_values(~Element,~Right,~right_value,right_context,geometry,right_fill,layout,right,limits,capacity,domain,right_shape,right_values,target))

def Choice(+geometry: Bands.Geometry,+layout: Bands.Plan,+limits: Machine.Limits,+capacity: U32) -> Data:
  Shape.Both<BandDense.Fits(geometry,layout,capacity,Machine.half_word(limits)),BandPlan.Coverage(layout,0n,Inspection.total_rows(geometry))>

def alignment_selected(~Element: Data,choose_left: Bool,+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +domain: BandGather.Domain<geometry,limits,capacity>,left_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,right_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,right)>) ->
  Choice(geometry,Bool.pick(Bands.Plan,choose_left,Model.layout(~Element,left),Model.layout(~Element,right)),limits,capacity):
  match choose_left:
    case True{}:
      BandShape.WellFormed{metadata,ready,coverage} = left_shape
      Shape.Both{BandDense.target(~Element,geometry,limits,capacity,left,domain,metadata,Model.ready(~Element,left,ready)),
        %Equal.sym(Bands.Plan,Model.layout(~Element,left),Model.content_layout(~Element,Model.content(~Element,left)),Model.layout_matches_content(~Element,left)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
        coverage}
    case False{}:
      BandShape.WellFormed{metadata,ready,coverage} = right_shape
      Shape.Both{BandDense.target(~Element,geometry,limits,capacity,right,domain,metadata,Model.ready(~Element,right,ready)),
        %Equal.sym(Bands.Plan,Model.layout(~Element,right),Model.content_layout(~Element,Model.content(~Element,right)),Model.layout_matches_content(~Element,right)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
        coverage}

def alignment_from_choice(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+parallel: Bool,+left_fill: Element,+right_fill: Element,+layout: Bands.Plan,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,
  left_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,right_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,right)>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left)),right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right)),
  choice: Choice(geometry,layout,limits,capacity)) ->
  Shape.Both<BandValues.Values(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
      Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Bands.align(~Element,geometry,left_fill,layout,Model.pack(~Element,left)),Bands.align(~Element,geometry,right_fill,layout,Model.pack(~Element,right))))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Bands.align(~Element,geometry,left_fill,layout,Model.pack(~Element,left)),Bands.align(~Element,geometry,right_fill,layout,Model.pack(~Element,right))))>>:
  Shape.Both{target,coverage} = choice
  alignment_aligned(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,parallel,left_fill,right_fill,layout,left,right,limits,capacity,domain,left_shape,right_shape,left_values,right_values,target,coverage)

def alignment_chosen(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+parallel: Bool,+left_fill: Element,+right_fill: Element,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,
  +left_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,+right_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,right)>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left)),right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right))) ->
  Shape.Both<BandValues.Values(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
      Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,
        Bands.align(~Element,geometry,left_fill,BandSchedule.alignment(geometry,Model.layout(~Element,left),Model.layout(~Element,right)),Model.pack(~Element,left)),
        Bands.align(~Element,geometry,right_fill,BandSchedule.alignment(geometry,Model.layout(~Element,left),Model.layout(~Element,right)),Model.pack(~Element,right))))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,
      Bands.align(~Element,geometry,left_fill,BandSchedule.alignment(geometry,Model.layout(~Element,left),Model.layout(~Element,right)),Model.pack(~Element,left)),
      Bands.align(~Element,geometry,right_fill,BandSchedule.alignment(geometry,Model.layout(~Element,left),Model.layout(~Element,right)),Model.pack(~Element,right))))>>:
  alignment_from_choice(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,parallel,left_fill,right_fill,
    BandSchedule.alignment(geometry,Model.layout(~Element,left),Model.layout(~Element,right)),left,right,limits,capacity,domain,left_shape,right_shape,left_values,right_values,
    alignment_selected(~Element,Bands.same_plan(Model.layout(~Element,left),Model.layout(~Element,right)) ||
      Nat.is_le(BandSchedule.allocation_elements(Model.layout(~Element,left),geometry),BandSchedule.allocation_elements(Model.layout(~Element,right),geometry)),
      geometry,limits,capacity,left,right,domain,left_shape,right_shape))

# Common representation and value contracts for internal binary results.
# Public native callers produce the stronger Global.Normal contract.
def same_shape(+shape: List<&2,U32>) -> {Shapes.same_extents(shape,shape) == True{} : Bool}:
  match shape:
    case Nil{}: {==}
    case +head <> +tail:
      %Equal.sym(Bool,U32.is_eq(head,head),True{},BandShape.equal_words(head,head,{==})) : {_ && Shapes.same_extents(tail,tail) == True{} : Bool}
      same_shape(tail)

def CombineContract(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,tensor: Tensors.Tensor<Element>) -> Data:
  Shape.Both<BandMap.WellFormed(~Element,tensor),
    BandMap.LogicalValues(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},tensor)>

def contract_banded(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,-bands: Bands.Bands(~Element),
  evidence: Shape.Both<BandValues.Values(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,Model.contents(~Element,bands)),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,bands)>>) ->
  CombineContract(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,Tensors.Banded{bands,geometry,shape,fill,Tensors.Usable{},workers}):
  Shape.Both{values,well_formed} = evidence
  Shape.Both{well_formed,values}

# The mixed binary caller copies dense storage directly into the existing
# band layout. Its machine envelope comes from the real dense owner and count.
def count_domain(~Element: Data,+geometry: Bands.Geometry,+input: Buffer.Model<Element>,+count: U32,
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  length_exact: {U32.to_nat(count) == Buffer.length(~Element,input) : Nat},ready: Buffer.Sized<Element,input>,
  counted: Count.Count<ShapeLaws.elements(BandCount.extents(geometry)),count>) ->
  BandGather.Domain<geometry,Default.limits(),Tree.capacity(Element,Buffer.tree(~Element,input))>:
  match ready counted:
    case Buffer.Sized{+depth_ok,+room} Count.Count{+exact,+within}:
      +expected_length = Equal.trans(Nat,BandGather.elements(geometry),ShapeLaws.elements(BandCount.extents(geometry)),Buffer.length(~Element,input),
        Equal.sym(Nat,ShapeLaws.elements(BandCount.extents(geometry)),BandGather.elements(geometry),BandCount.domain_elements(geometry)),
        Equal.trans(Nat,ShapeLaws.elements(BandCount.extents(geometry)),U32.to_nat(count),Buffer.length(~Element,input),exact,length_exact))
      BandDense.owned(~Element,geometry,Default.limits(),Machine.capacity(Default.limits()),input,expected_length,Buffer.Sized{depth_ok,room},
        BandCount.evidence(geometry,count,channels_positive,height_positive,width_positive,Count.Count{exact,within}))

def converted_well_formed(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+input: Buffer.Model<Element>,+division_capacity: U32,
  +target: BandDense.Fits(geometry,layout,Tree.capacity(Element,Buffer.tree(~Element,input)),division_capacity),coverage: BandPlan.Coverage(layout,0n,Inspection.total_rows(geometry))) ->
  BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.from_dense(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))>:
  BandShape.WellFormed{BandDense.from_dense_shape(~Element,geometry,fill,layout,input,division_capacity,target),BandDense.from_dense_ready(~Element,geometry,fill,layout,input,division_capacity,target),
    %Equal.sym(Bands.Plan,Model.content_layout(~Element,Model.contents(~Element,Bands.from_dense(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))),layout,
      BandDense.from_dense_layout(~Element,geometry,fill,layout,input)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
    coverage}

def band_left(~Element: Data,~Left: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,
  +left_context: Left,+geometry: Bands.Geometry,+parallel: Bool,+left_fill: Element,+right_fill: Element,+left: Model.Model<Element>,+right: Buffer.Model<Element>,
  +whole: BandGather.Domain<geometry,Default.limits(),Tree.capacity(Element,Buffer.tree(~Element,right))>,+left_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left))) ->
  Shape.Both<BandValues.Values(~Element,~BandCombineLeaf.Context<Left,Tree.Tree<Element>>,~BandCombineLeaf.value(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element)),BandCombineLeaf.Context{left_context,Buffer.tree(~Element,right)},geometry,
      Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,
        Bands.align(~Element,geometry,left_fill,Model.layout(~Element,left),Model.pack(~Element,left)),Bands.from_dense(~Element,geometry,right_fill,Model.layout(~Element,left),Buffer.pack(~Element,right))))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,
      Bands.align(~Element,geometry,left_fill,Model.layout(~Element,left),Model.pack(~Element,left)),Bands.from_dense(~Element,geometry,right_fill,Model.layout(~Element,left),Buffer.pack(~Element,right))))>>:
  BandShape.WellFormed{metadata,ready,+coverage} = left_shape
  +target = BandDense.target(~Element,geometry,Default.limits(),Tree.capacity(Element,Buffer.tree(~Element,right)),left,whole,metadata,Model.ready(~Element,left,ready))
  +coverage = {
    %Equal.sym(Bands.Plan,Model.layout(~Element,left),Model.content_layout(~Element,Model.content(~Element,left)),Model.layout_matches_content(~Element,left)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
    coverage : BandPlan.Coverage(Model.layout(~Element,left),0n,Inspection.total_rows(geometry))}
  combine_owners(~Element,~Left,~Tree.Tree<Element>,~operation,~kernel,~kernel_law,~left_value,~BandValues.dense_value(~Element),left_context,Buffer.tree(~Element,right),geometry,parallel,
    Bands.align(~Element,geometry,left_fill,Model.layout(~Element,left),Model.pack(~Element,left)),Bands.from_dense(~Element,geometry,right_fill,Model.layout(~Element,left),Buffer.pack(~Element,right)),
    alignment_width_positive(geometry,Default.limits(),Tree.capacity(Element,Buffer.tree(~Element,right)),whole),
    Equal.trans(Bands.Plan,Model.content_layout(~Element,Model.contents(~Element,Bands.align(~Element,geometry,left_fill,Model.layout(~Element,left),Model.pack(~Element,left)))),Model.layout(~Element,left),
      Model.content_layout(~Element,Model.contents(~Element,Bands.from_dense(~Element,geometry,right_fill,Model.layout(~Element,left),Buffer.pack(~Element,right)))),
      BandCombineLeaf.aligned_layout(~Element,geometry,left_fill,Model.layout(~Element,left),left),
      Equal.sym(Bands.Plan,Model.content_layout(~Element,Model.contents(~Element,Bands.from_dense(~Element,geometry,right_fill,Model.layout(~Element,left),Buffer.pack(~Element,right)))),Model.layout(~Element,left),
        BandDense.from_dense_layout(~Element,geometry,right_fill,Model.layout(~Element,left),right))),
    BandCombineLeaf.aligned_well_formed(~Element,geometry,left_fill,Model.layout(~Element,left),left,Tree.capacity(Element,Buffer.tree(~Element,right)),Machine.half_word(Default.limits()),left_shape,target,coverage),
    converted_well_formed(~Element,geometry,right_fill,Model.layout(~Element,left),right,Machine.half_word(Default.limits()),target,coverage),
    BandCombineLeaf.aligned_values(~Element,~Left,~left_value,left_context,geometry,left_fill,Model.layout(~Element,left),left,Default.limits(),Tree.capacity(Element,Buffer.tree(~Element,right)),whole,left_shape,left_values,target),
    BandDense.semantics_from_dense(~Element,geometry,right_fill,Model.layout(~Element,left),right,Machine.half_word(Default.limits()),target))

# Dense storage on the left is converted directly into the right owner's
# layout. The binary operation still receives the original left value first.
def combined(~Element: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~right_value: Right -> U32 -> Element,
  +right_context: Right,+geometry: Bands.Geometry,+parallel: Bool,+left_fill: Element,+right_fill: Element,+left: Buffer.Model<Element>,+right: Model.Model<Element>,
  +whole: BandGather.Domain<geometry,Default.limits(),Tree.capacity(Element,Buffer.tree(~Element,left))>,+right_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,right)>,
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right))) ->
  Shape.Both<BandValues.Values(~Element,~BandCombineLeaf.Context<Tree.Tree<Element>,Right>,~BandCombineLeaf.value(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value),
      BandCombineLeaf.Context{Buffer.tree(~Element,left),right_context},geometry,
      Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,
        Bands.from_dense(~Element,geometry,left_fill,Model.layout(~Element,right),Buffer.pack(~Element,left)),Bands.align(~Element,geometry,right_fill,Model.layout(~Element,right),Model.pack(~Element,right))))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,
      Bands.from_dense(~Element,geometry,left_fill,Model.layout(~Element,right),Buffer.pack(~Element,left)),Bands.align(~Element,geometry,right_fill,Model.layout(~Element,right),Model.pack(~Element,right))))>>:
  BandShape.WellFormed{metadata,ready,+coverage} = right_shape
  +target = BandDense.target(~Element,geometry,Default.limits(),Tree.capacity(Element,Buffer.tree(~Element,left)),right,whole,metadata,Model.ready(~Element,right,ready))
  +coverage = {
    %Equal.sym(Bands.Plan,Model.layout(~Element,right),Model.content_layout(~Element,Model.content(~Element,right)),Model.layout_matches_content(~Element,right)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
    coverage : BandPlan.Coverage(Model.layout(~Element,right),0n,Inspection.total_rows(geometry))}
  combine_owners(~Element,~Tree.Tree<Element>,~Right,~operation,~kernel,~kernel_law,~BandValues.dense_value(~Element),~right_value,Buffer.tree(~Element,left),right_context,geometry,parallel,
    Bands.from_dense(~Element,geometry,left_fill,Model.layout(~Element,right),Buffer.pack(~Element,left)),Bands.align(~Element,geometry,right_fill,Model.layout(~Element,right),Model.pack(~Element,right)),
    alignment_width_positive(geometry,Default.limits(),Tree.capacity(Element,Buffer.tree(~Element,left)),whole),
    Equal.trans(Bands.Plan,Model.content_layout(~Element,Model.contents(~Element,Bands.from_dense(~Element,geometry,left_fill,Model.layout(~Element,right),Buffer.pack(~Element,left)))),Model.layout(~Element,right),
      Model.content_layout(~Element,Model.contents(~Element,Bands.align(~Element,geometry,right_fill,Model.layout(~Element,right),Model.pack(~Element,right)))),
      BandDense.from_dense_layout(~Element,geometry,left_fill,Model.layout(~Element,right),left),
      Equal.sym(Bands.Plan,Model.content_layout(~Element,Model.contents(~Element,Bands.align(~Element,geometry,right_fill,Model.layout(~Element,right),Model.pack(~Element,right)))),Model.layout(~Element,right),
        BandCombineLeaf.aligned_layout(~Element,geometry,right_fill,Model.layout(~Element,right),right))),
    converted_well_formed(~Element,geometry,left_fill,Model.layout(~Element,right),left,Machine.half_word(Default.limits()),target,coverage),
    BandCombineLeaf.aligned_well_formed(~Element,geometry,right_fill,Model.layout(~Element,right),right,Tree.capacity(Element,Buffer.tree(~Element,left)),Machine.half_word(Default.limits()),right_shape,target,coverage),
    BandDense.semantics_from_dense(~Element,geometry,left_fill,Model.layout(~Element,right),left,Machine.half_word(Default.limits()),target),
    BandCombineLeaf.aligned_values(~Element,~Right,~right_value,right_context,geometry,right_fill,Model.layout(~Element,right),right,Default.limits(),Tree.capacity(Element,Buffer.tree(~Element,left)),whole,right_shape,right_values,target))

# A zero-count binary traversal preserves its left content even when the
# checked call rejects. Empty geometries derive these counts from source data.
def empty_refinement_selected(~Element: Data,~operation: Element -> Element -> Element,accepted: Bool,+left: Tree.Tree<Element>,+right: Tree.Tree<Element>,+left_length: Nat,+right_length: Nat) ->
  {BandCombineLeaf.left_content(~Element,OperationsLaws.combine_model(~Element,~operation,
      OperationsLaws.PairCall{accepted,Some{0},Vector.vector(0),Vector.vector(0),right_length,left_length},right,left)) == Buffer.Content{left,left_length} : Buffer.Content<Element>}:
  match accepted:
    case False{}:
      Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,left_length},Tree.reflect(Element,Tree.pack(Element,left)),left,StorageLaws.reflect_packed(Element,left))
    case True{}:
      Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,left_length},Tree.reflect(Element,Tree.pack(Element,left)),left,StorageLaws.reflect_packed(Element,left))

def leaf_model(~Element: Data,~operation: Element -> Element -> Element,left: Buffer.Model<Element>,right: Buffer.Model<Element>) ->
  {BandCombineLeaf.refinement_leaf_model(~Element,~operation,0,left,right) == Buffer.content(~Element,left) : Buffer.Content<Element>}:
  match left right:
    case Buffer.Model{+left_tree,+left_length,+left_certificate} Buffer.Model{+right_tree,+right_length,+right_certificate}:
      +left = {Buffer.Model{left_tree,left_length,left_certificate} : Buffer.Model<Element>}
      +right = {Buffer.Model{right_tree,right_length,right_certificate} : Buffer.Model<Element>}
      Equal.trans(Buffer.Content<Element>,BandCombineLeaf.refinement_leaf_model(~Element,~operation,0,left,right),Buffer.Content{Tree.reflect(Element,Tree.pack(Element,left_tree)),left_length},Buffer.Content{left_tree,left_length},
        empty_refinement_selected(~Element,~operation,BandCombineLeaf.guard(~Element,0,left,right),Tree.reflect(Element,Tree.pack(Element,left_tree)),Tree.reflect(Element,Tree.pack(Element,right_tree)),left_length,right_length),
        Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,left_length},Tree.reflect(Element,Tree.pack(Element,left_tree)),left_tree,StorageLaws.reflect_packed(Element,left_tree)))

def Counts(~Element: Data,+geometry: Bands.Geometry,bands: Model.Model<Element>) -> Data:
  match bands:
    case Model.Leaf{buffer,first,rows}: {Bands.local_count(geometry,rows) == 0 : U32}
    case Model.Fork{left,right}: Shape.Both<Counts(~Element,geometry,left),Counts(~Element,geometry,right)>

def width_counts(~Element: Data,+images: U32,+channels: U32,+height: U32,bands: Model.Model<Element>) ->
  Counts(~Element,Bands.Geometry{images,channels,height,0},bands):
  match bands:
    case Model.Leaf{buffer,first,rows}: BandCount.width_native(images,channels,height,rows)
    case Model.Fork{+left,+right}: Shape.Both{width_counts(~Element,images,channels,height,left),width_counts(~Element,images,channels,height,right)}

def channel_counts(~Element: Data,+images: U32,+height: U32,+width: U32,bands: Model.Model<Element>) ->
  Counts(~Element,Bands.Geometry{images,0,height,width},bands):
  match bands:
    case Model.Leaf{buffer,first,rows}: BandCount.channels_native(images,height,width,rows)
    case Model.Fork{+left,+right}: Shape.Both{channel_counts(~Element,images,height,width,left),channel_counts(~Element,images,height,width,right)}

def no_rows_leaf(+geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,
  total_zero: {Inspection.total_rows(geometry) == 0n : Nat},metadata: BandShape.Leaf(geometry,first,rows,length)) -> {Bands.local_count(geometry,rows) == 0 : U32}:
  Bands.Geometry{images,channels,height,width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  BandDense.empty_count(geometry,rows,
    BandDense.no_rows_natural(U32.to_nat(first),U32.to_nat(rows),%total_zero : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),_) == True{} : Bool} rows_fit))

def no_rows_counts(~Element: Data,+geometry: Bands.Geometry,bands: Model.Model<Element>,+total_zero: {Inspection.total_rows(geometry) == 0n : Nat},
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands))) -> Counts(~Element,geometry,bands):
  match bands:
    case Model.Leaf{Buffer.Model{tree,+length,certificate},+first,+rows}: no_rows_leaf(geometry,first,rows,length,total_zero,metadata)
    case Model.Fork{+left,+right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{no_rows_counts(~Element,geometry,left,total_zero,left_metadata),no_rows_counts(~Element,geometry,right,total_zero,right_metadata)}

def empty_refinement_model(~Element: Data,~operation: Element -> Element -> Element,+geometry: Bands.Geometry,left: Model.Model<Element>,right: Model.Model<Element>,counts: Counts(~Element,geometry,left)) ->
  {BandCombineLeaf.model(~Element,~operation,geometry,left,right) == Model.content(~Element,left) : Model.Content<Element>}:
  match left right:
    case Model.Leaf{+left,+first,+rows} Model.Leaf{+right,other_first,other_rows}:
      Equal.cong(Buffer.Content<Element>,Model.Content<Element>,buffer => Model.RowBand{buffer,first,rows},BandCombineLeaf.refinement_leaf_model(~Element,~operation,Bands.local_count(geometry,rows),left,right),Buffer.content(~Element,left),
        %Equal.sym(U32,Bands.local_count(geometry,rows),0,counts) : {BandCombineLeaf.refinement_leaf_model(~Element,~operation,_,left,right) == Buffer.content(~Element,left) : Buffer.Content<Element>}
        leaf_model(~Element,~operation,left,right))
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      Shape.Both{left_counts,right_counts} = counts
      %Equal.sym(Model.Content<Element>,BandCombineLeaf.model(~Element,~operation,geometry,left,other_left),Model.content(~Element,left),empty_refinement_model(~Element,~operation,geometry,left,other_left,left_counts)) :
        {Model.Joined{_,BandCombineLeaf.model(~Element,~operation,geometry,right,other_right)} == Model.Joined{Model.content(~Element,left),Model.content(~Element,right)} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,BandCombineLeaf.model(~Element,~operation,geometry,right,other_right),Model.content(~Element,right),empty_refinement_model(~Element,~operation,geometry,right,other_right,right_counts)) :
        {Model.Joined{Model.content(~Element,left),_} == Model.Joined{Model.content(~Element,left),Model.content(~Element,right)} : Model.Content<Element>}
      {==}
    case Model.Leaf{buffer,first,rows} Model.Fork{other_left,other_right}: {==}
    case Model.Fork{left,right} Model.Leaf{buffer,first,rows}: {==}

def empty_refinement_combine(~Element: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+geometry: Bands.Geometry,+parallel: Bool,+left: Model.Model<Element>,+right: Model.Model<Element>,counts: Counts(~Element,geometry,left)) ->
  {Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right))) == Model.content(~Element,left) : Model.Content<Element>}:
  Equal.trans(Model.Content<Element>,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right))),
    BandCombineLeaf.model(~Element,~operation,geometry,left,right),Model.content(~Element,left),BandCombineLeaf.combine(~Element,~operation,~kernel,~kernel_law,geometry,parallel,left,right),empty_refinement_model(~Element,~operation,geometry,left,right,counts))

# Source metadata and Ready discharge empty local lengths. The actual empty
# binary preserves content, so its complete WF and pointwise values follow.
def zero_length(+geometry: Bands.Geometry,+rows: U32,+length: Nat,
  native_zero: {Bands.local_count(geometry,rows) == 0 : U32},counted: BandShape.Count<geometry,rows,length>) -> {length == 0n : Nat}:
  Equal.trans(Nat,length,U32.to_nat(Bands.local_count(geometry,rows)),0n,
    Equal.sym(Nat,U32.to_nat(Bands.local_count(geometry,rows)),length,BandShapeCount.count_refinement_exact(geometry,rows,length,counted)),
    Equal.cong(U32,Nat,U32.to_nat,Bands.local_count(geometry,rows),0,native_zero))

def values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,bands: Model.Model<Element>,counts: Counts(~Element,geometry,bands),
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands)),ready: Model.Ready(~Element,bands)) ->
  BandValues.Values(~Element,~Context,~value,context,geometry,Model.content(~Element,bands)):
  match bands:
    case Model.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+rows}:
      +input = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
      +metadata = metadata
      BandValues.empty_leaf(~Element,~Context,~value,context,geometry,first,rows,tree,length,
        BandValues.empty_count(geometry,first,rows,length,metadata,zero_length(geometry,rows,length,counts,BandShapeCount.counted(~Element,geometry,first,rows,input,metadata,ready))))
    case Model.Fork{+left,+right}:
      Shape.Both{left_counts,right_counts} = counts
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{values(~Element,~Context,~value,context,geometry,left,left_counts,left_metadata,left_ready),values(~Element,~Context,~value,context,geometry,right,right_counts,right_metadata,right_ready)}

def combined_well_formed(~Element: Data,~Context: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+parallel: Bool,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +counts: Counts(~Element,geometry,left),source: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>) ->
  Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right)))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right)))>>:
  BandShape.WellFormed{+metadata,+ready,coverage} = source
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right))),Model.content(~Element,left),
    empty_refinement_combine(~Element,~operation,~kernel,~kernel_law,geometry,parallel,left,right,counts)) :
    Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,_),BandShape.WellFormed<Element,geometry,_>>
  Shape.Both{values(~Element,~Context,~value,context,geometry,left,counts,metadata,Model.ready(~Element,left,ready)),BandShape.WellFormed{metadata,ready,coverage}}

# Empty public shapes obtain target facts from an actual source layout.
# Observing aligned owners connects zero-count execution to full WF/Values.
def empty_alignment_target(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+capacity: U32,+division_capacity: U32,empty: BandShapeCount.EmptyGeometry<geometry>,
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands))) -> BandDense.Fits(geometry,Model.layout(~Element,bands),capacity,division_capacity):
  match geometry empty:
    case Bands.Geometry{+images,+channels,+height,+width} BandShapeCount.Width{+zero}:
      +metadata_zero = {
        %zero : BandShape.Contents(~Element,Bands.Geometry{images,channels,height,_},Model.content(~Element,bands))
        metadata : BandShape.Contents(~Element,Bands.Geometry{images,channels,height,0},Model.content(~Element,bands))}
      %Equal.sym(U32,width,0,zero) : BandDense.Fits(Bands.Geometry{images,channels,height,_},Model.layout(~Element,bands),capacity,division_capacity)
      BandDense.width(~Element,images,channels,height,bands,capacity,division_capacity,metadata_zero)
    case Bands.Geometry{+images,+channels,+height,+width} BandShapeCount.Channels{+zero}:
      +metadata_zero = {
        %zero : BandShape.Contents(~Element,Bands.Geometry{images,_,height,width},Model.content(~Element,bands))
        metadata : BandShape.Contents(~Element,Bands.Geometry{images,0,height,width},Model.content(~Element,bands))}
      %Equal.sym(U32,channels,0,zero) : BandDense.Fits(Bands.Geometry{images,_,height,width},Model.layout(~Element,bands),capacity,division_capacity)
      BandDense.channels(~Element,images,height,width,bands,capacity,division_capacity,metadata_zero)
    case Bands.Geometry{images,channels,height,width} BandShapeCount.Rows{zero}: BandDense.no_rows(~Element,geometry,bands,capacity,division_capacity,zero,metadata)

def counts(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,empty: BandShapeCount.EmptyGeometry<geometry>,
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands))) -> Counts(~Element,geometry,bands):
  match geometry empty:
    case Bands.Geometry{+images,+channels,+height,+width} BandShapeCount.Width{zero}:
      %Equal.sym(U32,width,0,zero) : Counts(~Element,Bands.Geometry{images,channels,height,_},bands)
      width_counts(~Element,images,channels,height,bands)
    case Bands.Geometry{+images,+channels,+height,+width} BandShapeCount.Channels{zero}:
      %Equal.sym(U32,channels,0,zero) : Counts(~Element,Bands.Geometry{images,_,height,width},bands)
      channel_counts(~Element,images,height,width,bands)
    case Bands.Geometry{images,channels,height,width} BandShapeCount.Rows{zero}: no_rows_counts(~Element,geometry,bands,zero,metadata)

def modeled(~Element: Data,~Context: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~value: Context -> U32 -> Element,
  +context: Context,+geometry: Bands.Geometry,+parallel: Bool,+left: Model.Model<Element>,+right: Model.Model<Element>,empty: BandShapeCount.EmptyGeometry<geometry>,
  source: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>) ->
  Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right)))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right)))>>:
  BandShape.WellFormed{+metadata,ready,coverage} = source
  combined_well_formed(~Element,~Context,~operation,~kernel,~kernel_law,~value,context,geometry,parallel,left,right,
    counts(~Element,geometry,left,empty,metadata),BandShape.WellFormed{metadata,ready,coverage})

def empty_alignment_observed(~Element: Data,~Context: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~value: Context -> U32 -> Element,
  +context: Context,+geometry: Bands.Geometry,+parallel: Bool,-left: Bands.Bands(~Element),-right: Bands.Bands(~Element),+empty: BandShapeCount.EmptyGeometry<geometry>,
  source: BandShape.WellFormed<Element,geometry,Model.contents(~Element,left)>,left_observation: Model.Observation<Element,left>,right_observation: Model.Observation<Element,right>) ->
  Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))>>:
  Model.Observation{+left_model,+left_equation} = left_observation
  Model.Observation{+right_model,+right_equation} = right_observation
  +left_content = observed_contents(~Element,left_model,left,left_equation)
  +canonical = {
    %left_content : BandShape.WellFormed<Element,geometry,_>
    source : BandShape.WellFormed<Element,geometry,Model.content(~Element,left_model)>}
  %left_equation :
    Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,_,right))),
      BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,_,right))>>
  %right_equation :
    Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left_model),_))),
      BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left_model),_))>>
  modeled(~Element,~Context,~operation,~kernel,~kernel_law,~value,context,geometry,parallel,left_model,right_model,empty,canonical)

def empty_alignment_owners(~Element: Data,~Context: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~value: Context -> U32 -> Element,
  +context: Context,+geometry: Bands.Geometry,+parallel: Bool,left: Bands.Bands(~Element),right: Bands.Bands(~Element),empty: BandShapeCount.EmptyGeometry<geometry>,
  source: BandShape.WellFormed<Element,geometry,Model.contents(~Element,left)>) ->
  Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))>>:
  empty_alignment_observed(~Element,~Context,~operation,~kernel,~kernel_law,~value,context,geometry,parallel,left,right,empty,source,Model.observe(~Element,left),Model.observe(~Element,right))

def empty_alignment_aligned(~Element: Data,~Context: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~value: Context -> U32 -> Element,
  +context: Context,+geometry: Bands.Geometry,+parallel: Bool,+left_fill: Element,+right_fill: Element,+layout: Bands.Plan,+left: Model.Model<Element>,+right: Model.Model<Element>,+limits: Machine.Limits,+capacity: U32,
  empty: BandShapeCount.EmptyGeometry<geometry>,source: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,
  empty_alignment_target: BandDense.Fits(geometry,layout,capacity,Machine.half_word(limits)),coverage: BandPlan.Coverage(layout,0n,Inspection.total_rows(geometry))) ->
  Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,
      Bands.align(~Element,geometry,left_fill,layout,Model.pack(~Element,left)),Bands.align(~Element,geometry,right_fill,layout,Model.pack(~Element,right))))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,
      Bands.align(~Element,geometry,left_fill,layout,Model.pack(~Element,left)),Bands.align(~Element,geometry,right_fill,layout,Model.pack(~Element,right))))>>:
  empty_alignment_owners(~Element,~Context,~operation,~kernel,~kernel_law,~value,context,geometry,parallel,Bands.align(~Element,geometry,left_fill,layout,Model.pack(~Element,left)),
    Bands.align(~Element,geometry,right_fill,layout,Model.pack(~Element,right)),empty,
    BandCombineLeaf.aligned_well_formed(~Element,geometry,left_fill,layout,left,capacity,Machine.half_word(limits),source,empty_alignment_target,coverage))

def empty_alignment_selected(~Element: Data,choose_left: Bool,+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+left: Model.Model<Element>,+right: Model.Model<Element>,+empty: BandShapeCount.EmptyGeometry<geometry>,
  left_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,right_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,right)>) ->
  Choice(geometry,Bool.pick(Bands.Plan,choose_left,Model.layout(~Element,left),Model.layout(~Element,right)),limits,capacity):
  match choose_left:
    case True{}:
      BandShape.WellFormed{metadata,ready,coverage} = left_shape
      Shape.Both{empty_alignment_target(~Element,geometry,left,capacity,Machine.half_word(limits),empty,metadata),
        %Equal.sym(Bands.Plan,Model.layout(~Element,left),Model.content_layout(~Element,Model.content(~Element,left)),Model.layout_matches_content(~Element,left)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
        coverage}
    case False{}:
      BandShape.WellFormed{metadata,ready,coverage} = right_shape
      Shape.Both{empty_alignment_target(~Element,geometry,right,capacity,Machine.half_word(limits),empty,metadata),
        %Equal.sym(Bands.Plan,Model.layout(~Element,right),Model.content_layout(~Element,Model.content(~Element,right)),Model.layout_matches_content(~Element,right)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
        coverage}

def empty_alignment_from_choice(~Element: Data,~Context: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~value: Context -> U32 -> Element,
  +context: Context,+geometry: Bands.Geometry,+parallel: Bool,+left_fill: Element,+right_fill: Element,+layout: Bands.Plan,+left: Model.Model<Element>,+right: Model.Model<Element>,+limits: Machine.Limits,+capacity: U32,
  empty: BandShapeCount.EmptyGeometry<geometry>,source: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,choice: Choice(geometry,layout,limits,capacity)) ->
  Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,
      Bands.align(~Element,geometry,left_fill,layout,Model.pack(~Element,left)),Bands.align(~Element,geometry,right_fill,layout,Model.pack(~Element,right))))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,
      Bands.align(~Element,geometry,left_fill,layout,Model.pack(~Element,left)),Bands.align(~Element,geometry,right_fill,layout,Model.pack(~Element,right))))>>:
  Shape.Both{empty_alignment_target,coverage} = choice
  empty_alignment_aligned(~Element,~Context,~operation,~kernel,~kernel_law,~value,context,geometry,parallel,left_fill,right_fill,layout,left,right,limits,capacity,empty,source,empty_alignment_target,coverage)

def empty_alignment_chosen(~Element: Data,~Context: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~value: Context -> U32 -> Element,
  +context: Context,+geometry: Bands.Geometry,+parallel: Bool,+left_fill: Element,+right_fill: Element,+left: Model.Model<Element>,+right: Model.Model<Element>,+limits: Machine.Limits,+capacity: U32,+empty: BandShapeCount.EmptyGeometry<geometry>,
  +left_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,right_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,right)>) ->
  Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,
      Bands.align(~Element,geometry,left_fill,BandSchedule.alignment(geometry,Model.layout(~Element,left),Model.layout(~Element,right)),Model.pack(~Element,left)),
      Bands.align(~Element,geometry,right_fill,BandSchedule.alignment(geometry,Model.layout(~Element,left),Model.layout(~Element,right)),Model.pack(~Element,right))))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,
      Bands.align(~Element,geometry,left_fill,BandSchedule.alignment(geometry,Model.layout(~Element,left),Model.layout(~Element,right)),Model.pack(~Element,left)),
      Bands.align(~Element,geometry,right_fill,BandSchedule.alignment(geometry,Model.layout(~Element,left),Model.layout(~Element,right)),Model.pack(~Element,right))))>>:
  empty_alignment_from_choice(~Element,~Context,~operation,~kernel,~kernel_law,~value,context,geometry,parallel,left_fill,right_fill,BandSchedule.alignment(geometry,Model.layout(~Element,left),Model.layout(~Element,right)),left,right,limits,capacity,
    empty,left_shape,empty_alignment_selected(~Element,Bands.same_plan(Model.layout(~Element,left),Model.layout(~Element,right)) ||
      Nat.is_le(BandSchedule.allocation_elements(Model.layout(~Element,left),geometry),BandSchedule.allocation_elements(Model.layout(~Element,right),geometry)),geometry,limits,capacity,left,right,empty,left_shape,right_shape))

# Empty binary execution contracts use the selected band layout. Their
# zero-count content contract includes every zero axis and both mixed orders.
def empty_contract_pair_planned(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+name: String,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,+workers: U32,+left: Model.Model<Element>,+right: Model.Model<Element>,
  empty: {Shapes.has_zero(BandCount.extents(geometry)) == True{} : Bool},left_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,right_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,right)>) ->
  CombineContract(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,
    Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,geometry,shape,shape,left_fill,right_fill,workers,
      (Model.pack(~Element,left),Model.layout(~Element,left)),(Model.pack(~Element,right),Model.layout(~Element,right)))):
  +layout = BandSchedule.alignment(geometry,Model.layout(~Element,left),Model.layout(~Element,right))
  +parallel = BandSchedule.parallel(geometry,workers,12.0,operations,layout)
  contract_banded(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,geometry,shape,left_fill,workers,
    Bands.combine_with(~Element,~kernel,geometry,parallel,Bands.align(~Element,geometry,left_fill,layout,Model.pack(~Element,left)),Bands.align(~Element,geometry,right_fill,layout,Model.pack(~Element,right))),
    empty_alignment_chosen(~Element,~BandCombineLeaf.Context<Left,Right>,~operation,~kernel,~kernel_law,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},
      geometry,parallel,left_fill,right_fill,left,right,Default.limits(),Machine.capacity(Default.limits()),BandShapeCount.empty_geometry_classified(geometry,empty),left_shape,right_shape))

def empty_contract_band_left_planned(~Element: Data,~Left: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,
  +left_context: Left,+name: String,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,+workers: U32,+right_workers: U32,+left: Model.Model<Element>,+right: Buffer.Model<Element>,
  empty: {Shapes.has_zero(BandCount.extents(geometry)) == True{} : Bool},+source: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>) ->
  CombineContract(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element),left_context,Buffer.tree(~Element,right),
    Tensors.combine_bands_plan_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,shape,shape,left_fill,workers,
      Tensors.Tensor{Buffer.pack(~Element,right),shape,right_fill,Tensors.Usable{},right_workers},(Model.pack(~Element,left),Model.layout(~Element,left)))):
  BandShape.WellFormed{metadata,ready,+coverage} = source
  +kind = BandShapeCount.empty_geometry_classified(geometry,empty)
  +target = empty_alignment_target(~Element,geometry,left,Machine.capacity(Default.limits()),Machine.half_word(Default.limits()),kind,metadata)
  +covered = {
    %Equal.sym(Bands.Plan,Model.layout(~Element,left),Model.content_layout(~Element,Model.content(~Element,left)),Model.layout_matches_content(~Element,left)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
    coverage : BandPlan.Coverage(Model.layout(~Element,left),0n,Inspection.total_rows(geometry))}
  +parallel = BandSchedule.parallel(geometry,workers,12.0,operations,Model.layout(~Element,left))
  contract_banded(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element),left_context,Buffer.tree(~Element,right),geometry,shape,left_fill,workers,
    Bands.combine_with(~Element,~kernel,geometry,parallel,Bands.align(~Element,geometry,left_fill,Model.layout(~Element,left),Model.pack(~Element,left)),Bands.from_dense(~Element,geometry,right_fill,Model.layout(~Element,left),Buffer.pack(~Element,right))),
    empty_alignment_owners(~Element,~BandCombineLeaf.Context<Left,Tree.Tree<Element>>,~operation,~kernel,~kernel_law,~BandCombineLeaf.value(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element)),BandCombineLeaf.Context{left_context,Buffer.tree(~Element,right)},geometry,parallel,
      Bands.align(~Element,geometry,left_fill,Model.layout(~Element,left),Model.pack(~Element,left)),Bands.from_dense(~Element,geometry,right_fill,Model.layout(~Element,left),Buffer.pack(~Element,right)),kind,
      BandCombineLeaf.aligned_well_formed(~Element,geometry,left_fill,Model.layout(~Element,left),left,Machine.capacity(Default.limits()),Machine.half_word(Default.limits()),source,target,covered)))

def empty_contract_dense_left_planned(~Element: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~right_value: Right -> U32 -> Element,
  +right_context: Right,+name: String,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,+workers: U32,+left: Buffer.Model<Element>,+right: Model.Model<Element>,
  empty: {Shapes.has_zero(BandCount.extents(geometry)) == True{} : Bool},source: BandShape.WellFormed<Element,geometry,Model.content(~Element,right)>) ->
  CombineContract(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value,Buffer.tree(~Element,left),right_context,
    Tensors.combine_right_plan_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,shape,shape,left_fill,right_fill,workers,
      Buffer.pack(~Element,left),(Model.pack(~Element,right),Model.layout(~Element,right)))):
  BandShape.WellFormed{metadata,ready,coverage} = source
  +kind = BandShapeCount.empty_geometry_classified(geometry,empty)
  +target = empty_alignment_target(~Element,geometry,right,Tree.capacity(Element,Buffer.tree(~Element,left)),Machine.half_word(Default.limits()),kind,metadata)
  +covered = {
    %Equal.sym(Bands.Plan,Model.layout(~Element,right),Model.content_layout(~Element,Model.content(~Element,right)),Model.layout_matches_content(~Element,right)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
    coverage : BandPlan.Coverage(Model.layout(~Element,right),0n,Inspection.total_rows(geometry))}
  +parallel = BandSchedule.parallel(geometry,workers,12.0,operations,Model.layout(~Element,right))
  contract_banded(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value,Buffer.tree(~Element,left),right_context,geometry,shape,left_fill,workers,
    Bands.combine_with(~Element,~kernel,geometry,parallel,Bands.from_dense(~Element,geometry,left_fill,Model.layout(~Element,right),Buffer.pack(~Element,left)),Bands.align(~Element,geometry,right_fill,Model.layout(~Element,right),Model.pack(~Element,right))),
    empty_alignment_owners(~Element,~BandCombineLeaf.Context<Tree.Tree<Element>,Right>,~operation,~kernel,~kernel_law,~BandCombineLeaf.value(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value),BandCombineLeaf.Context{Buffer.tree(~Element,left),right_context},geometry,parallel,
      Bands.from_dense(~Element,geometry,left_fill,Model.layout(~Element,right),Buffer.pack(~Element,left)),Bands.align(~Element,geometry,right_fill,Model.layout(~Element,right),Model.pack(~Element,right)),kind,
      converted_well_formed(~Element,geometry,left_fill,Model.layout(~Element,right),left,Machine.half_word(Default.limits()),target,covered)))

# Failed status and unequal shapes retain the actual left representation.
# These branches require neither a successful descriptor nor machine room.
def Preserved(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,tensor: Tensors.Tensor<Element>) -> Data:
  Shape.Both<BandMap.WellFormed(~Element,tensor),BandMap.LogicalValues(~Element,~Logical,~original,logical,tensor)>

def unchanged(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,+bands: Model.Model<Element>,
  well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands))) ->
  Preserved(~Element,~Logical,~original,logical,Tensors.Banded{Model.pack(~Element,bands),geometry,shape,fill,status,workers}):
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,bands)),Model.content(~Element,bands),Model.packed_contents(~Element,bands)) :
    Shape.Both<BandShape.WellFormed<Element,geometry,_>,BandValues.Values(~Element,~Logical,~original,logical,geometry,_)>
  Shape.Both{well_formed,values}

def failed_checked(~Element: Data,~Logical: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~original: Logical -> U32 -> Element,
  +logical: Logical,+name: String,+operations: F32,+problem: Tensors.Problem,+same: Bool,+geometry: Bands.Geometry,+layout: Bands.Plan,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+fill: Element,+workers: U32,
  +source_geometry: Bands.Geometry,+source_shape: List<&2,U32>,+source_fill: Element,+source_status: Tensors.Status,+source_workers: U32,+left: Model.Model<Element>,right: Tensors.Tensor<Element>,
  well_formed: BandShape.WellFormed<Element,source_geometry,Model.content(~Element,left)>,values: BandValues.Values(~Element,~Logical,~original,logical,source_geometry,Model.content(~Element,left))) ->
  Preserved(~Element,~Logical,~original,logical,
    Tensors.combine_bands_checked_with(~Element,~kernel,name,operations,Tensors.Failed{problem},same,geometry,layout,left_shape,right_shape,fill,workers,
      Tensors.Banded{Model.pack(~Element,left),source_geometry,source_shape,source_fill,source_status,source_workers},right)):
  unchanged(~Element,~Logical,~original,logical,source_geometry,source_shape,source_fill,Tensors.Failed{problem},source_workers,left,well_formed,values)

# Binary callers retain the checked count of the actual public shape.
# Their alignment semantics supply output Values and representation readiness;
# the unchanged output shape supplies its whole-tensor machine envelope.
def normal_count_exact(+expected: Nat,+count: U32,evidence: Count.Count<expected,count>) -> {expected == U32.to_nat(count) : Nat}:
  Count.Count{exact,bounded} = evidence
  exact

def NormalContract(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,-tensor: Tensors.Tensor<Element>) -> Data:
  Shape.Both<Global.Normal(~Element,tensor),
    BandMap.LogicalValues(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},tensor)>

def normal_banded(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,-bands: Bands.Bands(~Element),
  counted: Global.Counted<shape>,coherent: {Window.band_shape(shape,geometry) == True{} : Bool},
  evidence: Shape.Both<BandValues.Values(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,Model.contents(~Element,bands)),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,bands)>>) ->
  NormalContract(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,Tensors.Banded{bands,geometry,shape,fill,Tensors.Usable{},workers}):
  Shape.Both{values,well_formed} = evidence
  Shape.Both{Shape.Both{well_formed,Shape.Both{counted,coherent}},values}

def normal_pair_planned(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+name: String,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,+workers: U32,
  +left: Model.Model<Element>,+right: Model.Model<Element>,+counted: Global.Counted<shape>,+coherent: {Window.band_shape(shape,geometry) == True{} : Bool},
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  left_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,right_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,right)>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left)),right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right))) ->
  NormalContract(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,
    Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,geometry,shape,shape,left_fill,right_fill,workers,
      (Model.pack(~Element,left),Model.layout(~Element,left)),(Model.pack(~Element,right),Model.layout(~Element,right)))):
  +layout = BandSchedule.alignment(geometry,Model.layout(~Element,left),Model.layout(~Element,right))
  +parallel = BandSchedule.parallel(geometry,workers,12.0,operations,layout)
  normal_banded(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,geometry,shape,left_fill,workers,
    Bands.combine_with(~Element,~kernel,geometry,parallel,Bands.align(~Element,geometry,left_fill,layout,Model.pack(~Element,left)),Bands.align(~Element,geometry,right_fill,layout,Model.pack(~Element,right))),
    counted,coherent,
    alignment_chosen(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,parallel,left_fill,right_fill,left,right,
      Default.limits(),Machine.capacity(Default.limits()),Global.domain(shape,geometry,counted,coherent,channels_positive,height_positive,width_positive),left_shape,right_shape,left_values,right_values))

# The dense owner contributes its physical room. Its length equals the actual
# public shape; full axis coherence relates that shape to the selected geometry.
def dense_domain(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+input: Buffer.Model<Element>,+counted: Global.Counted<shape>,
  coherent: {Window.band_shape(shape,geometry) == True{} : Bool},
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  length_exact: {Buffer.length(~Element,input) == ShapeLaws.elements(shape) : Nat},ready: Buffer.Sized<Element,input>) ->
  BandGather.Domain<geometry,Default.limits(),Tree.capacity(Element,Buffer.tree(~Element,input))>:
  +evidence = Global.evidence(shape,counted)
  +geometry_evidence = Global.geometry_evidence(shape,geometry,counted,coherent)
  +expected_length = Equal.trans(Nat,U32.to_nat(Global.count(shape,counted)),ShapeLaws.elements(shape),Buffer.length(~Element,input),
    Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(Global.count(shape,counted)),normal_count_exact(ShapeLaws.elements(shape),Global.count(shape,counted),evidence)),
    Equal.sym(Nat,Buffer.length(~Element,input),ShapeLaws.elements(shape),length_exact))
  count_domain(~Element,geometry,input,Global.count(shape,counted),channels_positive,height_positive,width_positive,expected_length,ready,
    %Equal.sym(Nat,ShapeLaws.elements(BandCount.extents(geometry)),BandGather.elements(geometry),BandCount.domain_elements(geometry)) : Count.Count<_,Global.count(shape,counted)>
    geometry_evidence)

def normal_band_left_planned(~Element: Data,~Left: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,
  +left_context: Left,+name: String,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,+workers: U32,+right_workers: U32,
  +left: Model.Model<Element>,+right: Buffer.Model<Element>,+counted: Global.Counted<shape>,+coherent: {Window.band_shape(shape,geometry) == True{} : Bool},
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  length_exact: {Buffer.length(~Element,right) == ShapeLaws.elements(shape) : Nat},right_ready: Buffer.Sized<Element,right>,
  left_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left))) ->
  NormalContract(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element),left_context,Buffer.tree(~Element,right),
    Tensors.combine_bands_plan_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,shape,shape,left_fill,workers,
      Tensors.Tensor{Buffer.pack(~Element,right),shape,right_fill,Tensors.Usable{},right_workers},(Model.pack(~Element,left),Model.layout(~Element,left)))):
  +parallel = BandSchedule.parallel(geometry,workers,12.0,operations,Model.layout(~Element,left))
  normal_banded(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element),left_context,Buffer.tree(~Element,right),geometry,shape,left_fill,workers,
    Bands.combine_with(~Element,~kernel,geometry,parallel,Bands.align(~Element,geometry,left_fill,Model.layout(~Element,left),Model.pack(~Element,left)),
      Bands.from_dense(~Element,geometry,right_fill,Model.layout(~Element,left),Buffer.pack(~Element,right))),counted,coherent,
    band_left(~Element,~Left,~operation,~kernel,~kernel_law,~left_value,left_context,geometry,parallel,left_fill,right_fill,left,right,
      dense_domain(~Element,geometry,shape,right,counted,coherent,channels_positive,height_positive,width_positive,length_exact,right_ready),left_shape,left_values))

def normal_dense_left_planned(~Element: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~right_value: Right -> U32 -> Element,
  +right_context: Right,+name: String,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,+workers: U32,
  +left: Buffer.Model<Element>,+right: Model.Model<Element>,+counted: Global.Counted<shape>,+coherent: {Window.band_shape(shape,geometry) == True{} : Bool},
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  length_exact: {Buffer.length(~Element,left) == ShapeLaws.elements(shape) : Nat},left_ready: Buffer.Sized<Element,left>,
  right_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,right)>,right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right))) ->
  NormalContract(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value,Buffer.tree(~Element,left),right_context,
    Tensors.combine_right_plan_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,shape,shape,left_fill,right_fill,workers,
      Buffer.pack(~Element,left),(Model.pack(~Element,right),Model.layout(~Element,right)))):
  +parallel = BandSchedule.parallel(geometry,workers,12.0,operations,Model.layout(~Element,right))
  normal_banded(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value,Buffer.tree(~Element,left),right_context,geometry,shape,left_fill,workers,
    Bands.combine_with(~Element,~kernel,geometry,parallel,Bands.from_dense(~Element,geometry,left_fill,Model.layout(~Element,right),Buffer.pack(~Element,left)),
      Bands.align(~Element,geometry,right_fill,Model.layout(~Element,right),Model.pack(~Element,right))),counted,coherent,
    combined(~Element,~Right,~operation,~kernel,~kernel_law,~right_value,right_context,geometry,parallel,left_fill,right_fill,left,right,
      dense_domain(~Element,geometry,shape,left,counted,coherent,channels_positive,height_positive,width_positive,length_exact,left_ready),right_shape,right_values))

def pair_invocation(~Element: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+name: String,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,
  +left_fill: Element,+right_fill: Element,+left_workers: U32,+right_workers: U32,+left: Model.Model<Element>,+right: Model.Model<Element>) ->
  {Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Banded{Model.pack(~Element,left),geometry,shape,left_fill,Tensors.Usable{},left_workers},
      Tensors.Banded{Model.pack(~Element,right),geometry,shape,right_fill,Tensors.Usable{},right_workers}) ==
    Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,geometry,shape,shape,left_fill,right_fill,U32.max(left_workers,right_workers),
      (Model.pack(~Element,left),Model.layout(~Element,left)),(Model.pack(~Element,right),Model.layout(~Element,right))) : Tensors.Tensor<Element>}:
  %Equal.sym(Bool,Shapes.same_extents(shape,shape),True{},same_shape(shape)) :
    {Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,Tensors.Usable{},_,geometry,geometry,shape,shape,left_fill,right_fill,U32.max(left_workers,right_workers),
      Bands.plan(~Element,Model.pack(~Element,left)),Bands.plan(~Element,Model.pack(~Element,right))) ==
      Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,geometry,shape,shape,left_fill,right_fill,U32.max(left_workers,right_workers),
        (Model.pack(~Element,left),Model.layout(~Element,left)),(Model.pack(~Element,right),Model.layout(~Element,right))) : Tensors.Tensor<Element>}
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,left)),(Model.pack(~Element,left),Model.layout(~Element,left)),Model.plan(~Element,left)) :
    {Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,geometry,shape,shape,left_fill,right_fill,U32.max(left_workers,right_workers),
      _,Bands.plan(~Element,Model.pack(~Element,right))) ==
      Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,geometry,shape,shape,left_fill,right_fill,U32.max(left_workers,right_workers),
        (Model.pack(~Element,left),Model.layout(~Element,left)),(Model.pack(~Element,right),Model.layout(~Element,right))) : Tensors.Tensor<Element>}
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,right)),(Model.pack(~Element,right),Model.layout(~Element,right)),Model.plan(~Element,right)) :
    {Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,geometry,shape,shape,left_fill,right_fill,U32.max(left_workers,right_workers),
      (Model.pack(~Element,left),Model.layout(~Element,left)),_) ==
      Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,geometry,shape,shape,left_fill,right_fill,U32.max(left_workers,right_workers),
        (Model.pack(~Element,left),Model.layout(~Element,left)),(Model.pack(~Element,right),Model.layout(~Element,right))) : Tensors.Tensor<Element>}
  {==}

# Empty callers already supply their complete representation/value result.
# Adding the unchanged actual shape receipt gives the same normal contract.
def promote_banded(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,-bands: Bands.Bands(~Element),
  counted: Global.Counted<shape>,coherent: {Window.band_shape(shape,geometry) == True{} : Bool},
  evidence: CombineContract(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,Tensors.Banded{bands,geometry,shape,fill,Tensors.Usable{},workers})) ->
  NormalContract(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,Tensors.Banded{bands,geometry,shape,fill,Tensors.Usable{},workers}):
  Shape.Both{well_formed,values} = evidence
  Shape.Both{Shape.Both{well_formed,Shape.Both{counted,coherent}},values}

def empty_pair_planned(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+name: String,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,+workers: U32,
  +left: Model.Model<Element>,+right: Model.Model<Element>,counted: Global.Counted<shape>,coherent: {Window.band_shape(shape,geometry) == True{} : Bool},
  empty: {Shapes.has_zero(BandCount.extents(geometry)) == True{} : Bool},
  left_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,right_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,right)>) ->
  NormalContract(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,
    Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,geometry,shape,shape,left_fill,right_fill,workers,
      (Model.pack(~Element,left),Model.layout(~Element,left)),(Model.pack(~Element,right),Model.layout(~Element,right)))):
  +layout = BandSchedule.alignment(geometry,Model.layout(~Element,left),Model.layout(~Element,right))
  +parallel = BandSchedule.parallel(geometry,workers,12.0,operations,layout)
  promote_banded(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,geometry,shape,left_fill,workers,
    Bands.combine_with(~Element,~kernel,geometry,parallel,Bands.align(~Element,geometry,left_fill,layout,Model.pack(~Element,left)),
      Bands.align(~Element,geometry,right_fill,layout,Model.pack(~Element,right))),counted,coherent,
    empty_contract_pair_planned(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,name,operations,geometry,shape,left_fill,right_fill,workers,left,right,
      empty,left_shape,right_shape))

def empty_band_left_planned(~Element: Data,~Left: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,
  +left_context: Left,+name: String,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,+workers: U32,+right_workers: U32,
  +left: Model.Model<Element>,+right: Buffer.Model<Element>,counted: Global.Counted<shape>,coherent: {Window.band_shape(shape,geometry) == True{} : Bool},
  empty: {Shapes.has_zero(BandCount.extents(geometry)) == True{} : Bool},left_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>) ->
  NormalContract(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element),left_context,Buffer.tree(~Element,right),
    Tensors.combine_bands_plan_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,shape,shape,left_fill,workers,
      Tensors.Tensor{Buffer.pack(~Element,right),shape,right_fill,Tensors.Usable{},right_workers},(Model.pack(~Element,left),Model.layout(~Element,left)))):
  +parallel = BandSchedule.parallel(geometry,workers,12.0,operations,Model.layout(~Element,left))
  promote_banded(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element),left_context,Buffer.tree(~Element,right),geometry,shape,left_fill,workers,
    Bands.combine_with(~Element,~kernel,geometry,parallel,Bands.align(~Element,geometry,left_fill,Model.layout(~Element,left),Model.pack(~Element,left)),
      Bands.from_dense(~Element,geometry,right_fill,Model.layout(~Element,left),Buffer.pack(~Element,right))),counted,coherent,
    empty_contract_band_left_planned(~Element,~Left,~operation,~kernel,~kernel_law,~left_value,left_context,name,operations,geometry,shape,left_fill,right_fill,workers,right_workers,left,right,empty,left_shape))

def empty_dense_left_planned(~Element: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~right_value: Right -> U32 -> Element,
  +right_context: Right,+name: String,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,+workers: U32,
  +left: Buffer.Model<Element>,+right: Model.Model<Element>,counted: Global.Counted<shape>,coherent: {Window.band_shape(shape,geometry) == True{} : Bool},
  empty: {Shapes.has_zero(BandCount.extents(geometry)) == True{} : Bool},right_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,right)>) ->
  NormalContract(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value,Buffer.tree(~Element,left),right_context,
    Tensors.combine_right_plan_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,shape,shape,left_fill,right_fill,workers,
      Buffer.pack(~Element,left),(Model.pack(~Element,right),Model.layout(~Element,right)))):
  +parallel = BandSchedule.parallel(geometry,workers,12.0,operations,Model.layout(~Element,right))
  promote_banded(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value,Buffer.tree(~Element,left),right_context,geometry,shape,left_fill,workers,
    Bands.combine_with(~Element,~kernel,geometry,parallel,Bands.from_dense(~Element,geometry,left_fill,Model.layout(~Element,right),Buffer.pack(~Element,left)),
      Bands.align(~Element,geometry,right_fill,Model.layout(~Element,right),Model.pack(~Element,right))),counted,coherent,
    empty_contract_dense_left_planned(~Element,~Right,~operation,~kernel,~kernel_law,~right_value,right_context,name,operations,geometry,shape,left_fill,right_fill,workers,left,right,empty,right_shape))

def dense_length(~Element: Data,+shape: List<&2,U32>,input: Buffer.Model<Element>,
  exact: {Buffer.content_length(~Element,Buffer.content(~Element,input)) == ShapeLaws.elements(shape) : Nat}) ->
  {Buffer.length(~Element,input) == ShapeLaws.elements(shape) : Nat}:
  Buffer.Model{tree,length,certificate} = input
  exact

def band_left_nonempty_ready(~Element: Data,~Left: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,
  +left_context: Left,+name: String,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,+left_workers: U32,+right_workers: U32,
  +left: Model.Model<Element>,+right: Buffer.Model<Element>,
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  left_normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,left),geometry,shape,left_fill,Tensors.Usable{},left_workers}),
  right_normal: Global.Normal(~Element,Tensors.Tensor{Buffer.pack(~Element,right),shape,right_fill,Tensors.Usable{},right_workers}),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left))) ->
  NormalContract(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element),left_context,Buffer.tree(~Element,right),
    Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Banded{Model.pack(~Element,left),geometry,shape,left_fill,Tensors.Usable{},left_workers},
      Tensors.Tensor{Buffer.pack(~Element,right),shape,right_fill,Tensors.Usable{},right_workers})):
  Shape.Both{left_shape,Shape.Both{+counted,+coherent}} = left_normal
  Shape.Both{right_ready,Shape.Both{right_counted,length_exact}} = right_normal
  +right_ready = Buffer.ready_content(~Element,right,
    %Buffer.packed_contents(~Element,right) : Buffer.ContentReady<Element,_> right_ready)
  +length_exact = dense_length(~Element,shape,right,
    %Buffer.packed_contents(~Element,right) : {Buffer.content_length(~Element,_) == ShapeLaws.elements(shape) : Nat} length_exact)
  %Equal.sym(Bool,Shapes.same_extents(shape,shape),True{},same_shape(shape)) :
    NormalContract(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element),left_context,Buffer.tree(~Element,right),
      Tensors.combine_bands_plan_with(~Element,~kernel,name,operations,Tensors.Usable{},_,geometry,shape,shape,left_fill,U32.max(left_workers,right_workers),
        Tensors.Tensor{Buffer.pack(~Element,right),shape,right_fill,Tensors.Usable{},right_workers},Bands.plan(~Element,Model.pack(~Element,left))))
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,left)),(Model.pack(~Element,left),Model.layout(~Element,left)),Model.plan(~Element,left)) :
    NormalContract(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element),left_context,Buffer.tree(~Element,right),
      Tensors.combine_bands_plan_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,shape,shape,left_fill,U32.max(left_workers,right_workers),
        Tensors.Tensor{Buffer.pack(~Element,right),shape,right_fill,Tensors.Usable{},right_workers},_))
  normal_band_left_planned(~Element,~Left,~operation,~kernel,~kernel_law,~left_value,left_context,name,operations,geometry,shape,left_fill,right_fill,U32.max(left_workers,right_workers),right_workers,
    left,right,counted,coherent,channels_positive,height_positive,width_positive,length_exact,right_ready,
    %Model.packed_contents(~Element,left) : BandShape.WellFormed<Element,geometry,_> left_shape,left_values)

def dense_left_nonempty_ready(~Element: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~right_value: Right -> U32 -> Element,
  +right_context: Right,+name: String,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,+left_workers: U32,+right_workers: U32,
  +left: Buffer.Model<Element>,+right: Model.Model<Element>,
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  left_normal: Global.Normal(~Element,Tensors.Tensor{Buffer.pack(~Element,left),shape,left_fill,Tensors.Usable{},left_workers}),
  right_normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,right),geometry,shape,right_fill,Tensors.Usable{},right_workers}),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right))) ->
  NormalContract(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value,Buffer.tree(~Element,left),right_context,
    Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Tensor{Buffer.pack(~Element,left),shape,left_fill,Tensors.Usable{},left_workers},
      Tensors.Banded{Model.pack(~Element,right),geometry,shape,right_fill,Tensors.Usable{},right_workers})):
  Shape.Both{left_ready,Shape.Both{left_counted,length_exact}} = left_normal
  Shape.Both{right_shape,Shape.Both{+counted,+coherent}} = right_normal
  +left_ready = Buffer.ready_content(~Element,left,
    %Buffer.packed_contents(~Element,left) : Buffer.ContentReady<Element,_> left_ready)
  +length_exact = dense_length(~Element,shape,left,
    %Buffer.packed_contents(~Element,left) : {Buffer.content_length(~Element,_) == ShapeLaws.elements(shape) : Nat} length_exact)
  %Equal.sym(Bool,Shapes.same_extents(shape,shape),True{},same_shape(shape)) :
    NormalContract(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value,Buffer.tree(~Element,left),right_context,
      Tensors.combine_right_plan_with(~Element,~kernel,name,operations,Tensors.Usable{},_,geometry,shape,shape,left_fill,right_fill,U32.max(left_workers,right_workers),
        Buffer.pack(~Element,left),Bands.plan(~Element,Model.pack(~Element,right))))
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,right)),(Model.pack(~Element,right),Model.layout(~Element,right)),Model.plan(~Element,right)) :
    NormalContract(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value,Buffer.tree(~Element,left),right_context,
      Tensors.combine_right_plan_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,shape,shape,left_fill,right_fill,U32.max(left_workers,right_workers),
        Buffer.pack(~Element,left),_))
  normal_dense_left_planned(~Element,~Right,~operation,~kernel,~kernel_law,~right_value,right_context,name,operations,geometry,shape,left_fill,right_fill,U32.max(left_workers,right_workers),
    left,right,counted,coherent,channels_positive,height_positive,width_positive,length_exact,left_ready,
    %Model.packed_contents(~Element,right) : BandShape.WellFormed<Element,geometry,_> right_shape,right_values)

type Dimensions<-geometry: Bands.Geometry> is Data:
  Dimensions{channels: {BandGather.channels_positive(geometry) == True{} : Bool},height: {BandGather.height_positive(geometry) == True{} : Bool},width: {BandGather.width_positive(geometry) == True{} : Bool}}

def nonempty_dimensions(+geometry: Bands.Geometry,nonempty: {Shapes.has_zero(BandCount.extents(geometry)) == False{} : Bool}) -> Dimensions<geometry>:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  +remaining = Boolean.or_false_right(U32.is_zero(images),U32.is_zero(channels) || (U32.is_zero(height) || (U32.is_zero(width) || False{})),nonempty)
  +channels_nonzero = Boolean.or_false_left(U32.is_zero(channels),U32.is_zero(height) || (U32.is_zero(width) || False{}),remaining)
  +remaining = Boolean.or_false_right(U32.is_zero(channels),U32.is_zero(height) || (U32.is_zero(width) || False{}),remaining)
  +height_nonzero = Boolean.or_false_left(U32.is_zero(height),U32.is_zero(width) || False{},remaining)
  +remaining = Boolean.or_false_right(U32.is_zero(height),U32.is_zero(width) || False{},remaining)
  Dimensions{CountCompletion.positive_word(channels,channels_nonzero),CountCompletion.positive_word(height,height_nonzero),
    CountCompletion.positive_word(width,Boolean.or_false_left(U32.is_zero(width),False{},remaining))}

def DimensionChoice(+geometry: Bands.Geometry) -> Data:
  Either<&2,&2,{Shapes.has_zero(BandCount.extents(geometry)) == True{} : Bool},Dimensions<geometry>>

def dimensions_selected(empty: Bool,+geometry: Bands.Geometry,classified: {Shapes.has_zero(BandCount.extents(geometry)) == empty : Bool}) -> DimensionChoice(geometry):
  match empty:
    case True{}: Inl{classified}
    case False{}: Inr{nonempty_dimensions(geometry,classified)}

def dimension_choice(+geometry: Bands.Geometry) -> DimensionChoice(geometry):
  dimensions_selected(Shapes.has_zero(BandCount.extents(geometry)),geometry,{==})

def pair_nonempty_planned(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+name: String,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,+workers: U32,
  +left: Model.Model<Element>,+right: Model.Model<Element>,counted: Global.Counted<shape>,coherent: {Window.band_shape(shape,geometry) == True{} : Bool},dimensions: Dimensions<geometry>,
  left_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,right_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,right)>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left)),right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right))) ->
  NormalContract(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,
    Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,geometry,shape,shape,left_fill,right_fill,workers,
      (Model.pack(~Element,left),Model.layout(~Element,left)),(Model.pack(~Element,right),Model.layout(~Element,right)))):
  Dimensions{channels_positive,height_positive,width_positive} = dimensions
  normal_pair_planned(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,name,operations,geometry,shape,left_fill,right_fill,workers,left,right,
    counted,coherent,channels_positive,height_positive,width_positive,left_shape,right_shape,left_values,right_values)

def pair_selected(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  empty: Bool,+left_context: Left,+right_context: Right,+name: String,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,+workers: U32,
  +left: Model.Model<Element>,+right: Model.Model<Element>,counted: Global.Counted<shape>,coherent: {Window.band_shape(shape,geometry) == True{} : Bool},
  classified: {Shapes.has_zero(BandCount.extents(geometry)) == empty : Bool},
  left_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,right_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,right)>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left)),right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right))) ->
  NormalContract(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,
    Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,geometry,shape,shape,left_fill,right_fill,workers,
      (Model.pack(~Element,left),Model.layout(~Element,left)),(Model.pack(~Element,right),Model.layout(~Element,right)))):
  match empty:
    case True{}: empty_pair_planned(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,name,operations,geometry,shape,left_fill,right_fill,workers,left,right,
      counted,coherent,classified,left_shape,right_shape)
    case False{}: pair_nonempty_planned(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,name,operations,geometry,shape,left_fill,right_fill,workers,left,right,
      counted,coherent,nonempty_dimensions(geometry,classified),left_shape,right_shape,left_values,right_values)

def normal_pair_ready(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+name: String,+operations: F32,+geometry: Bands.Geometry,+right_geometry: Bands.Geometry,+shape: List<&2,U32>,
  +left_fill: Element,+right_fill: Element,+left_workers: U32,+right_workers: U32,+left: Model.Model<Element>,+right: Model.Model<Element>,
  left_normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,left),geometry,shape,left_fill,Tensors.Usable{},left_workers}),
  right_normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,right),right_geometry,shape,right_fill,Tensors.Usable{},right_workers}),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left)),right_values: BandValues.Values(~Element,~Right,~right_value,right_context,right_geometry,Model.content(~Element,right))) ->
  NormalContract(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,
    Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Banded{Model.pack(~Element,left),geometry,shape,left_fill,Tensors.Usable{},left_workers},
      Tensors.Banded{Model.pack(~Element,right),right_geometry,shape,right_fill,Tensors.Usable{},right_workers})):
  Shape.Both{left_shape,Shape.Both{+counted,+left_coherent}} = left_normal
  Shape.Both{right_shape,Shape.Both{right_counted,+right_coherent}} = right_normal
  +same_geometry = BandCount.unique(shape,geometry,right_geometry,left_coherent,right_coherent)
  +right_shape = {
    %Model.packed_contents(~Element,right) : BandShape.WellFormed<Element,right_geometry,_>
    right_shape : BandShape.WellFormed<Element,right_geometry,Model.content(~Element,right)>}
  +right_shape = {
    %Equal.sym(Bands.Geometry,geometry,right_geometry,same_geometry) : BandShape.WellFormed<Element,_,Model.content(~Element,right)>
    right_shape : BandShape.WellFormed<Element,geometry,Model.content(~Element,right)>}
  +right_values = {%Equal.sym(Bands.Geometry,geometry,right_geometry,same_geometry) : BandValues.Values(~Element,~Right,~right_value,right_context,_,Model.content(~Element,right)) right_values :
    BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right))}
  %same_geometry : NormalContract(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,
    Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Banded{Model.pack(~Element,left),geometry,shape,left_fill,Tensors.Usable{},left_workers},
      Tensors.Banded{Model.pack(~Element,right),_,shape,right_fill,Tensors.Usable{},right_workers}))
  %Equal.sym(Tensors.Tensor<Element>,
    Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Banded{Model.pack(~Element,left),geometry,shape,left_fill,Tensors.Usable{},left_workers},
      Tensors.Banded{Model.pack(~Element,right),geometry,shape,right_fill,Tensors.Usable{},right_workers}),
    Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,geometry,shape,shape,left_fill,right_fill,U32.max(left_workers,right_workers),
      (Model.pack(~Element,left),Model.layout(~Element,left)),(Model.pack(~Element,right),Model.layout(~Element,right))),
    pair_invocation(~Element,~operation,~kernel,~kernel_law,name,operations,geometry,shape,left_fill,right_fill,left_workers,right_workers,left,right)) :
    NormalContract(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,_)
  pair_selected(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,Shapes.has_zero(BandCount.extents(geometry)),left_context,right_context,name,operations,
    geometry,shape,left_fill,right_fill,U32.max(left_workers,right_workers),left,right,counted,left_coherent,{==},
    %Model.packed_contents(~Element,left) : BandShape.WellFormed<Element,geometry,_>
    left_shape,right_shape,left_values,right_values)

def band_left_ready_selected(~Element: Data,~Left: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,
  +geometry: Bands.Geometry,choice: DimensionChoice(geometry),+left_context: Left,+name: String,+operations: F32,+shape: List<&2,U32>,
  +left_fill: Element,+right_fill: Element,+left_workers: U32,+right_workers: U32,+left: Model.Model<Element>,+right: Buffer.Model<Element>,
  left_normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,left),geometry,shape,left_fill,Tensors.Usable{},left_workers}),
  right_normal: Global.Normal(~Element,Tensors.Tensor{Buffer.pack(~Element,right),shape,right_fill,Tensors.Usable{},right_workers}),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left))) ->
  NormalContract(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element),left_context,Buffer.tree(~Element,right),
    Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Banded{Model.pack(~Element,left),geometry,shape,left_fill,Tensors.Usable{},left_workers},
      Tensors.Tensor{Buffer.pack(~Element,right),shape,right_fill,Tensors.Usable{},right_workers})):
  match choice:
    case Inr{dimensions}:
      Dimensions{channels_positive,height_positive,width_positive} = dimensions
      band_left_nonempty_ready(~Element,~Left,~operation,~kernel,~kernel_law,~left_value,left_context,name,operations,geometry,shape,left_fill,right_fill,left_workers,right_workers,left,right,
        channels_positive,height_positive,width_positive,left_normal,right_normal,left_values)
    case Inl{empty}:
      Shape.Both{left_shape,Shape.Both{counted,coherent}} = left_normal
      +workers = U32.max(left_workers,right_workers)
      %Equal.sym(Bool,Shapes.same_extents(shape,shape),True{},same_shape(shape)) :
        NormalContract(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element),left_context,Buffer.tree(~Element,right),
          Tensors.combine_bands_plan_with(~Element,~kernel,name,operations,Tensors.Usable{},_,geometry,shape,shape,left_fill,workers,
            Tensors.Tensor{Buffer.pack(~Element,right),shape,right_fill,Tensors.Usable{},right_workers},Bands.plan(~Element,Model.pack(~Element,left))))
      %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,left)),(Model.pack(~Element,left),Model.layout(~Element,left)),Model.plan(~Element,left)) :
        NormalContract(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element),left_context,Buffer.tree(~Element,right),
          Tensors.combine_bands_plan_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,shape,shape,left_fill,workers,
            Tensors.Tensor{Buffer.pack(~Element,right),shape,right_fill,Tensors.Usable{},right_workers},_))
      empty_band_left_planned(~Element,~Left,~operation,~kernel,~kernel_law,~left_value,left_context,name,operations,geometry,shape,left_fill,right_fill,workers,right_workers,left,right,counted,coherent,empty,
        %Model.packed_contents(~Element,left) : BandShape.WellFormed<Element,geometry,_> left_shape)

def normal_band_left_ready(~Element: Data,~Left: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,
  +left_context: Left,+name: String,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,+left_workers: U32,+right_workers: U32,
  +left: Model.Model<Element>,+right: Buffer.Model<Element>,
  left_normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,left),geometry,shape,left_fill,Tensors.Usable{},left_workers}),
  right_normal: Global.Normal(~Element,Tensors.Tensor{Buffer.pack(~Element,right),shape,right_fill,Tensors.Usable{},right_workers}),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left))) ->
  NormalContract(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element),left_context,Buffer.tree(~Element,right),
    Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Banded{Model.pack(~Element,left),geometry,shape,left_fill,Tensors.Usable{},left_workers},
      Tensors.Tensor{Buffer.pack(~Element,right),shape,right_fill,Tensors.Usable{},right_workers})):
  band_left_ready_selected(~Element,~Left,~operation,~kernel,~kernel_law,~left_value,geometry,dimension_choice(geometry),left_context,name,operations,shape,left_fill,right_fill,left_workers,right_workers,left,right,
    left_normal,right_normal,left_values)

def dense_left_ready_selected(~Element: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~right_value: Right -> U32 -> Element,
  +geometry: Bands.Geometry,choice: DimensionChoice(geometry),+right_context: Right,+name: String,+operations: F32,+shape: List<&2,U32>,
  +left_fill: Element,+right_fill: Element,+left_workers: U32,+right_workers: U32,+left: Buffer.Model<Element>,+right: Model.Model<Element>,
  left_normal: Global.Normal(~Element,Tensors.Tensor{Buffer.pack(~Element,left),shape,left_fill,Tensors.Usable{},left_workers}),
  right_normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,right),geometry,shape,right_fill,Tensors.Usable{},right_workers}),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right))) ->
  NormalContract(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value,Buffer.tree(~Element,left),right_context,
    Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Tensor{Buffer.pack(~Element,left),shape,left_fill,Tensors.Usable{},left_workers},
      Tensors.Banded{Model.pack(~Element,right),geometry,shape,right_fill,Tensors.Usable{},right_workers})):
  match choice:
    case Inr{dimensions}:
      Dimensions{channels_positive,height_positive,width_positive} = dimensions
      dense_left_nonempty_ready(~Element,~Right,~operation,~kernel,~kernel_law,~right_value,right_context,name,operations,geometry,shape,left_fill,right_fill,left_workers,right_workers,left,right,
        channels_positive,height_positive,width_positive,left_normal,right_normal,right_values)
    case Inl{empty}:
      Shape.Both{right_shape,Shape.Both{counted,coherent}} = right_normal
      +workers = U32.max(left_workers,right_workers)
      %Equal.sym(Bool,Shapes.same_extents(shape,shape),True{},same_shape(shape)) :
        NormalContract(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value,Buffer.tree(~Element,left),right_context,
          Tensors.combine_right_plan_with(~Element,~kernel,name,operations,Tensors.Usable{},_,geometry,shape,shape,left_fill,right_fill,workers,
            Buffer.pack(~Element,left),Bands.plan(~Element,Model.pack(~Element,right))))
      %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,right)),(Model.pack(~Element,right),Model.layout(~Element,right)),Model.plan(~Element,right)) :
        NormalContract(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value,Buffer.tree(~Element,left),right_context,
          Tensors.combine_right_plan_with(~Element,~kernel,name,operations,Tensors.Usable{},True{},geometry,shape,shape,left_fill,right_fill,workers,
            Buffer.pack(~Element,left),_))
      empty_dense_left_planned(~Element,~Right,~operation,~kernel,~kernel_law,~right_value,right_context,name,operations,geometry,shape,left_fill,right_fill,workers,left,right,counted,coherent,empty,
        %Model.packed_contents(~Element,right) : BandShape.WellFormed<Element,geometry,_> right_shape)

def normal_dense_left_ready(~Element: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~right_value: Right -> U32 -> Element,
  +right_context: Right,+name: String,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,+left_workers: U32,+right_workers: U32,
  +left: Buffer.Model<Element>,+right: Model.Model<Element>,
  left_normal: Global.Normal(~Element,Tensors.Tensor{Buffer.pack(~Element,left),shape,left_fill,Tensors.Usable{},left_workers}),
  right_normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,right),geometry,shape,right_fill,Tensors.Usable{},right_workers}),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right))) ->
  NormalContract(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value,Buffer.tree(~Element,left),right_context,
    Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Tensor{Buffer.pack(~Element,left),shape,left_fill,Tensors.Usable{},left_workers},
      Tensors.Banded{Model.pack(~Element,right),geometry,shape,right_fill,Tensors.Usable{},right_workers})):
  dense_left_ready_selected(~Element,~Right,~operation,~kernel,~kernel_law,~right_value,geometry,dimension_choice(geometry),right_context,name,operations,shape,left_fill,right_fill,left_workers,right_workers,left,right,
    left_normal,right_normal,right_values)

# Native combine factories observe each owner once, then use canonical Data
# models. Normal retains the actual shape count, including zero dimensions.
# In mixed calls the dense context is its actual reflected storage tree;
# its logical value is Tree.read, rather than a separate supplied value map.
def pair_observed(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+operation_name: String,+operation_cost: F32,+left_geometry: Bands.Geometry,+right_geometry: Bands.Geometry,+shared_shape: List<&2,U32>,
  +left_fill: Element,+right_fill: Element,+left_workers: U32,+right_workers: U32,-left_bands: Bands.Bands(~Element),-right_bands: Bands.Bands(~Element),
  left_observation: Model.Observation<Element,left_bands>,right_observation: Model.Observation<Element,right_bands>,
  left_normal: Global.Normal(~Element,Tensors.Banded{left_bands,left_geometry,shared_shape,left_fill,Tensors.Usable{},left_workers}),
  right_normal: Global.Normal(~Element,Tensors.Banded{right_bands,right_geometry,shared_shape,right_fill,Tensors.Usable{},right_workers}),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,left_geometry,Model.contents(~Element,left_bands)),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,right_geometry,Model.contents(~Element,right_bands))) ->
  NormalContract(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,
    Tensors.combine_with(~Element,~kernel,operation_name,operation_cost,Tensors.Banded{left_bands,left_geometry,shared_shape,left_fill,Tensors.Usable{},left_workers},
      Tensors.Banded{right_bands,right_geometry,shared_shape,right_fill,Tensors.Usable{},right_workers})):
  Model.Observation{+left_model,+left_equation} = left_observation
  Model.Observation{+right_model,+right_equation} = right_observation
  +left_contents = Equal.trans(Model.Content<Element>,Model.contents(~Element,left_bands),Model.contents(~Element,Model.pack(~Element,left_model)),Model.content(~Element,left_model),
    Equal.cong(Bands.Bands(~Element),Model.Content<Element>,Model.contents(~Element),left_bands,Model.pack(~Element,left_model),Equal.sym(Bands.Bands(~Element),Model.pack(~Element,left_model),left_bands,left_equation)),Model.packed_contents(~Element,left_model))
  +right_contents = Equal.trans(Model.Content<Element>,Model.contents(~Element,right_bands),Model.contents(~Element,Model.pack(~Element,right_model)),Model.content(~Element,right_model),
    Equal.cong(Bands.Bands(~Element),Model.Content<Element>,Model.contents(~Element),right_bands,Model.pack(~Element,right_model),Equal.sym(Bands.Bands(~Element),Model.pack(~Element,right_model),right_bands,right_equation)),Model.packed_contents(~Element,right_model))
  %left_equation : NormalContract(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,
    Tensors.combine_with(~Element,~kernel,operation_name,operation_cost,Tensors.Banded{_,left_geometry,shared_shape,left_fill,Tensors.Usable{},left_workers},Tensors.Banded{right_bands,right_geometry,shared_shape,right_fill,Tensors.Usable{},right_workers}))
  %right_equation : NormalContract(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,
    Tensors.combine_with(~Element,~kernel,operation_name,operation_cost,Tensors.Banded{Model.pack(~Element,left_model),left_geometry,shared_shape,left_fill,Tensors.Usable{},left_workers},Tensors.Banded{_,right_geometry,shared_shape,right_fill,Tensors.Usable{},right_workers}))
  normal_pair_ready(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,operation_name,operation_cost,left_geometry,right_geometry,shared_shape,
    left_fill,right_fill,left_workers,right_workers,left_model,right_model,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,left_model),left_bands,left_equation) : Global.Normal(~Element,Tensors.Banded{_,left_geometry,shared_shape,left_fill,Tensors.Usable{},left_workers}) left_normal,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,right_model),right_bands,right_equation) : Global.Normal(~Element,Tensors.Banded{_,right_geometry,shared_shape,right_fill,Tensors.Usable{},right_workers}) right_normal,
    %left_contents : BandValues.Values(~Element,~Left,~left_value,left_context,left_geometry,_) left_values,
    %right_contents : BandValues.Values(~Element,~Right,~right_value,right_context,right_geometry,_) right_values)

def native_pair_ready(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+operation_name: String,+operation_cost: F32,+left_geometry: Bands.Geometry,+right_geometry: Bands.Geometry,+shared_shape: List<&2,U32>,
  +left_fill: Element,+right_fill: Element,+left_workers: U32,+right_workers: U32,left_bands: Bands.Bands(~Element),right_bands: Bands.Bands(~Element),
  left_normal: Global.Normal(~Element,Tensors.Banded{left_bands,left_geometry,shared_shape,left_fill,Tensors.Usable{},left_workers}),
  right_normal: Global.Normal(~Element,Tensors.Banded{right_bands,right_geometry,shared_shape,right_fill,Tensors.Usable{},right_workers}),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,left_geometry,Model.contents(~Element,left_bands)),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,right_geometry,Model.contents(~Element,right_bands))) ->
  NormalContract(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,
    Tensors.combine_with(~Element,~kernel,operation_name,operation_cost,Tensors.Banded{left_bands,left_geometry,shared_shape,left_fill,Tensors.Usable{},left_workers},
      Tensors.Banded{right_bands,right_geometry,shared_shape,right_fill,Tensors.Usable{},right_workers})):
  pair_observed(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,operation_name,operation_cost,left_geometry,right_geometry,shared_shape,
    left_fill,right_fill,left_workers,right_workers,left_bands,right_bands,Model.observe(~Element,left_bands),Model.observe(~Element,right_bands),left_normal,right_normal,left_values,right_values)

def band_left_observed(~Element: Data,~Left: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,
  +left_context: Left,+operation_name: String,+operation_cost: F32,+geometry: Bands.Geometry,+shared_shape: List<&2,U32>,
  +left_fill: Element,+right_fill: Element,+left_workers: U32,+right_workers: U32,-left_bands: Bands.Bands(~Element),-right_buffer: Storage.Buffer<Element>,
  left_observation: Model.Observation<Element,left_bands>,right_observation: Buffer.Observation<Element,right_buffer>,
  left_normal: Global.Normal(~Element,Tensors.Banded{left_bands,geometry,shared_shape,left_fill,Tensors.Usable{},left_workers}),
  right_normal: Global.Normal(~Element,Tensors.Tensor{right_buffer,shared_shape,right_fill,Tensors.Usable{},right_workers}),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.contents(~Element,left_bands))) ->
  NormalContract(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element),left_context,Buffer.content_tree(~Element,Buffer.contents(~Element,right_buffer)),
    Tensors.combine_with(~Element,~kernel,operation_name,operation_cost,Tensors.Banded{left_bands,geometry,shared_shape,left_fill,Tensors.Usable{},left_workers},
      Tensors.Tensor{right_buffer,shared_shape,right_fill,Tensors.Usable{},right_workers})):
  Model.Observation{+left_model,+left_equation} = left_observation
  Buffer.Observation{Buffer.Model{+right_tree,+right_length,+right_certificate},+right_equation} = right_observation
  +right_model = {Buffer.Model{right_tree,right_length,right_certificate} : Buffer.Model<Element>}
  +left_contents = Equal.trans(Model.Content<Element>,Model.contents(~Element,left_bands),Model.contents(~Element,Model.pack(~Element,left_model)),Model.content(~Element,left_model),
    Equal.cong(Bands.Bands(~Element),Model.Content<Element>,Model.contents(~Element),left_bands,Model.pack(~Element,left_model),Equal.sym(Bands.Bands(~Element),Model.pack(~Element,left_model),left_bands,left_equation)),Model.packed_contents(~Element,left_model))
  +right_contents = Equal.trans(Buffer.Content<Element>,Buffer.contents(~Element,right_buffer),Buffer.contents(~Element,Buffer.pack(~Element,right_model)),Buffer.content(~Element,right_model),
    Equal.cong(Storage.Buffer<Element>,Buffer.Content<Element>,Buffer.contents(~Element),right_buffer,Buffer.pack(~Element,right_model),Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,right_model),right_buffer,right_equation)),Buffer.packed_contents(~Element,right_model))
  +right_tree_equal = Equal.cong(Buffer.Content<Element>,Tree.Tree<Element>,Buffer.content_tree(~Element),Buffer.contents(~Element,right_buffer),Buffer.content(~Element,right_model),right_contents)
  %Equal.sym(Tree.Tree<Element>,Buffer.content_tree(~Element,Buffer.contents(~Element,right_buffer)),right_tree,right_tree_equal) :
    NormalContract(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element),left_context,_,
      Tensors.combine_with(~Element,~kernel,operation_name,operation_cost,Tensors.Banded{left_bands,geometry,shared_shape,left_fill,Tensors.Usable{},left_workers},Tensors.Tensor{right_buffer,shared_shape,right_fill,Tensors.Usable{},right_workers}))
  %left_equation : NormalContract(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element),left_context,right_tree,
    Tensors.combine_with(~Element,~kernel,operation_name,operation_cost,Tensors.Banded{_,geometry,shared_shape,left_fill,Tensors.Usable{},left_workers},Tensors.Tensor{right_buffer,shared_shape,right_fill,Tensors.Usable{},right_workers}))
  %right_equation : NormalContract(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element),left_context,right_tree,
    Tensors.combine_with(~Element,~kernel,operation_name,operation_cost,Tensors.Banded{Model.pack(~Element,left_model),geometry,shared_shape,left_fill,Tensors.Usable{},left_workers},Tensors.Tensor{_,shared_shape,right_fill,Tensors.Usable{},right_workers}))
  normal_band_left_ready(~Element,~Left,~operation,~kernel,~kernel_law,~left_value,left_context,operation_name,operation_cost,geometry,shared_shape,left_fill,right_fill,left_workers,right_workers,left_model,right_model,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,left_model),left_bands,left_equation) : Global.Normal(~Element,Tensors.Banded{_,geometry,shared_shape,left_fill,Tensors.Usable{},left_workers}) left_normal,
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,right_model),right_buffer,right_equation) : Global.Normal(~Element,Tensors.Tensor{_,shared_shape,right_fill,Tensors.Usable{},right_workers}) right_normal,
    %left_contents : BandValues.Values(~Element,~Left,~left_value,left_context,geometry,_) left_values)

def native_band_left_ready(~Element: Data,~Left: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,
  +left_context: Left,+operation_name: String,+operation_cost: F32,+geometry: Bands.Geometry,+shared_shape: List<&2,U32>,
  +left_fill: Element,+right_fill: Element,+left_workers: U32,+right_workers: U32,left_bands: Bands.Bands(~Element),right_buffer: Storage.Buffer<Element>,
  left_normal: Global.Normal(~Element,Tensors.Banded{left_bands,geometry,shared_shape,left_fill,Tensors.Usable{},left_workers}),
  right_normal: Global.Normal(~Element,Tensors.Tensor{right_buffer,shared_shape,right_fill,Tensors.Usable{},right_workers}),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.contents(~Element,left_bands))) ->
  NormalContract(~Element,~Left,~Tree.Tree<Element>,~operation,~left_value,~BandValues.dense_value(~Element),left_context,Buffer.content_tree(~Element,Buffer.contents(~Element,right_buffer)),
    Tensors.combine_with(~Element,~kernel,operation_name,operation_cost,Tensors.Banded{left_bands,geometry,shared_shape,left_fill,Tensors.Usable{},left_workers},
      Tensors.Tensor{right_buffer,shared_shape,right_fill,Tensors.Usable{},right_workers})):
  band_left_observed(~Element,~Left,~operation,~kernel,~kernel_law,~left_value,left_context,operation_name,operation_cost,geometry,shared_shape,left_fill,right_fill,left_workers,right_workers,
    left_bands,right_buffer,Model.observe(~Element,left_bands),Buffer.observe(~Element,right_buffer),left_normal,right_normal,left_values)

def dense_left_observed(~Element: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~right_value: Right -> U32 -> Element,
  +right_context: Right,+operation_name: String,+operation_cost: F32,+geometry: Bands.Geometry,+shared_shape: List<&2,U32>,
  +left_fill: Element,+right_fill: Element,+left_workers: U32,+right_workers: U32,-left_buffer: Storage.Buffer<Element>,-right_bands: Bands.Bands(~Element),
  left_observation: Buffer.Observation<Element,left_buffer>,right_observation: Model.Observation<Element,right_bands>,
  left_normal: Global.Normal(~Element,Tensors.Tensor{left_buffer,shared_shape,left_fill,Tensors.Usable{},left_workers}),
  right_normal: Global.Normal(~Element,Tensors.Banded{right_bands,geometry,shared_shape,right_fill,Tensors.Usable{},right_workers}),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.contents(~Element,right_bands))) ->
  NormalContract(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value,Buffer.content_tree(~Element,Buffer.contents(~Element,left_buffer)),right_context,
    Tensors.combine_with(~Element,~kernel,operation_name,operation_cost,Tensors.Tensor{left_buffer,shared_shape,left_fill,Tensors.Usable{},left_workers},
      Tensors.Banded{right_bands,geometry,shared_shape,right_fill,Tensors.Usable{},right_workers})):
  Buffer.Observation{Buffer.Model{+left_tree,+left_length,+left_certificate},+left_equation} = left_observation
  Model.Observation{+right_model,+right_equation} = right_observation
  +left_model = {Buffer.Model{left_tree,left_length,left_certificate} : Buffer.Model<Element>}
  +left_contents = Equal.trans(Buffer.Content<Element>,Buffer.contents(~Element,left_buffer),Buffer.contents(~Element,Buffer.pack(~Element,left_model)),Buffer.content(~Element,left_model),
    Equal.cong(Storage.Buffer<Element>,Buffer.Content<Element>,Buffer.contents(~Element),left_buffer,Buffer.pack(~Element,left_model),Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,left_model),left_buffer,left_equation)),Buffer.packed_contents(~Element,left_model))
  +right_contents = Equal.trans(Model.Content<Element>,Model.contents(~Element,right_bands),Model.contents(~Element,Model.pack(~Element,right_model)),Model.content(~Element,right_model),
    Equal.cong(Bands.Bands(~Element),Model.Content<Element>,Model.contents(~Element),right_bands,Model.pack(~Element,right_model),Equal.sym(Bands.Bands(~Element),Model.pack(~Element,right_model),right_bands,right_equation)),Model.packed_contents(~Element,right_model))
  +left_tree_equal = Equal.cong(Buffer.Content<Element>,Tree.Tree<Element>,Buffer.content_tree(~Element),Buffer.contents(~Element,left_buffer),Buffer.content(~Element,left_model),left_contents)
  %Equal.sym(Tree.Tree<Element>,Buffer.content_tree(~Element,Buffer.contents(~Element,left_buffer)),left_tree,left_tree_equal) :
    NormalContract(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value,_,right_context,
      Tensors.combine_with(~Element,~kernel,operation_name,operation_cost,Tensors.Tensor{left_buffer,shared_shape,left_fill,Tensors.Usable{},left_workers},Tensors.Banded{right_bands,geometry,shared_shape,right_fill,Tensors.Usable{},right_workers}))
  %left_equation : NormalContract(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value,left_tree,right_context,
    Tensors.combine_with(~Element,~kernel,operation_name,operation_cost,Tensors.Tensor{_,shared_shape,left_fill,Tensors.Usable{},left_workers},Tensors.Banded{right_bands,geometry,shared_shape,right_fill,Tensors.Usable{},right_workers}))
  %right_equation : NormalContract(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value,left_tree,right_context,
    Tensors.combine_with(~Element,~kernel,operation_name,operation_cost,Tensors.Tensor{Buffer.pack(~Element,left_model),shared_shape,left_fill,Tensors.Usable{},left_workers},Tensors.Banded{_,geometry,shared_shape,right_fill,Tensors.Usable{},right_workers}))
  normal_dense_left_ready(~Element,~Right,~operation,~kernel,~kernel_law,~right_value,right_context,operation_name,operation_cost,geometry,shared_shape,left_fill,right_fill,left_workers,right_workers,left_model,right_model,
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,left_model),left_buffer,left_equation) : Global.Normal(~Element,Tensors.Tensor{_,shared_shape,left_fill,Tensors.Usable{},left_workers}) left_normal,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,right_model),right_bands,right_equation) : Global.Normal(~Element,Tensors.Banded{_,geometry,shared_shape,right_fill,Tensors.Usable{},right_workers}) right_normal,
    %right_contents : BandValues.Values(~Element,~Right,~right_value,right_context,geometry,_) right_values)

def native_dense_left_ready(~Element: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~right_value: Right -> U32 -> Element,
  +right_context: Right,+operation_name: String,+operation_cost: F32,+geometry: Bands.Geometry,+shared_shape: List<&2,U32>,
  +left_fill: Element,+right_fill: Element,+left_workers: U32,+right_workers: U32,left_buffer: Storage.Buffer<Element>,right_bands: Bands.Bands(~Element),
  left_normal: Global.Normal(~Element,Tensors.Tensor{left_buffer,shared_shape,left_fill,Tensors.Usable{},left_workers}),
  right_normal: Global.Normal(~Element,Tensors.Banded{right_bands,geometry,shared_shape,right_fill,Tensors.Usable{},right_workers}),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.contents(~Element,right_bands))) ->
  NormalContract(~Element,~Tree.Tree<Element>,~Right,~operation,~BandValues.dense_value(~Element),~right_value,Buffer.content_tree(~Element,Buffer.contents(~Element,left_buffer)),right_context,
    Tensors.combine_with(~Element,~kernel,operation_name,operation_cost,Tensors.Tensor{left_buffer,shared_shape,left_fill,Tensors.Usable{},left_workers},
      Tensors.Banded{right_bands,geometry,shared_shape,right_fill,Tensors.Usable{},right_workers})):
  dense_left_observed(~Element,~Right,~operation,~kernel,~kernel_law,~right_value,right_context,operation_name,operation_cost,geometry,shared_shape,left_fill,right_fill,left_workers,right_workers,
    left_buffer,right_bands,Buffer.observe(~Element,left_buffer),Model.observe(~Element,right_bands),left_normal,right_normal,right_values)

# Public status and shape failures preserve the actual left owner through
# the plan-producing wrappers, including both mixed representations.
def left_failure(+problem: Tensors.Problem,+right: Tensors.Status) ->
  {Tensors.first_failure(Tensors.Failed{problem},right) == Tensors.Failed{problem} : Tensors.Status}:
  {==}

def right_failure(+problem: Tensors.Problem) ->
  {Tensors.first_failure(Tensors.Usable{},Tensors.Failed{problem}) == Tensors.Failed{problem} : Tensors.Status}:
  {==}

def band_tensor_failed(~Element: Data,~Logical: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~original: Logical -> U32 -> Element,
  +logical: Logical,+name: String,+operations: F32,+geometry: Bands.Geometry,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,
  +left_status: Tensors.Status,+right_status: Tensors.Status,+left_workers: U32,+right_workers: U32,+left: Model.Model<Element>,+right: Buffer.Model<Element>,+problem: Tensors.Problem,
  failure: {Tensors.first_failure(left_status,right_status) == Tensors.Failed{problem} : Tensors.Status},
  well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,left))) ->
  Preserved(~Element,~Logical,~original,logical,Tensors.combine_with(~Element,~kernel,name,operations,
    Tensors.Banded{Model.pack(~Element,left),geometry,left_shape,left_fill,left_status,left_workers},Tensors.Tensor{Buffer.pack(~Element,right),right_shape,right_fill,right_status,right_workers})):
  %Equal.sym(Tensors.Status,Tensors.first_failure(left_status,right_status),Tensors.Failed{problem},failure) :
    Preserved(~Element,~Logical,~original,logical,Tensors.combine_bands_plan_with(~Element,~kernel,name,operations,_,Shapes.same_extents(left_shape,right_shape),geometry,
      left_shape,right_shape,left_fill,U32.max(left_workers,right_workers),Tensors.Tensor{Buffer.pack(~Element,right),right_shape,right_fill,right_status,right_workers},Bands.plan(~Element,Model.pack(~Element,left))))
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,left)),(Model.pack(~Element,left),Model.layout(~Element,left)),Model.plan(~Element,left)) :
    Preserved(~Element,~Logical,~original,logical,Tensors.combine_bands_plan_with(~Element,~kernel,name,operations,Tensors.Failed{problem},Shapes.same_extents(left_shape,right_shape),geometry,
      left_shape,right_shape,left_fill,U32.max(left_workers,right_workers),Tensors.Tensor{Buffer.pack(~Element,right),right_shape,right_fill,right_status,right_workers},_))
  unchanged(~Element,~Logical,~original,logical,geometry,left_shape,left_fill,Tensors.Failed{problem},U32.max(left_workers,right_workers),left,well_formed,values)

def pair_failed(~Element: Data,~Logical: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~original: Logical -> U32 -> Element,
  +logical: Logical,+name: String,+operations: F32,+geometry: Bands.Geometry,+right_geometry: Bands.Geometry,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,
  +left_status: Tensors.Status,+right_status: Tensors.Status,+left_workers: U32,+right_workers: U32,+left: Model.Model<Element>,+right: Model.Model<Element>,+problem: Tensors.Problem,
  failure: {Tensors.first_failure(left_status,right_status) == Tensors.Failed{problem} : Tensors.Status},
  well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,left))) ->
  Preserved(~Element,~Logical,~original,logical,Tensors.combine_with(~Element,~kernel,name,operations,
    Tensors.Banded{Model.pack(~Element,left),geometry,left_shape,left_fill,left_status,left_workers},Tensors.Banded{Model.pack(~Element,right),right_geometry,right_shape,right_fill,right_status,right_workers})):
  %Equal.sym(Tensors.Status,Tensors.first_failure(left_status,right_status),Tensors.Failed{problem},failure) :
    Preserved(~Element,~Logical,~original,logical,Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,_,Shapes.same_extents(left_shape,right_shape),geometry,right_geometry,
      left_shape,right_shape,left_fill,right_fill,U32.max(left_workers,right_workers),Bands.plan(~Element,Model.pack(~Element,left)),Bands.plan(~Element,Model.pack(~Element,right))))
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,left)),(Model.pack(~Element,left),Model.layout(~Element,left)),Model.plan(~Element,left)) :
    Preserved(~Element,~Logical,~original,logical,Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,Tensors.Failed{problem},Shapes.same_extents(left_shape,right_shape),geometry,right_geometry,
      left_shape,right_shape,left_fill,right_fill,U32.max(left_workers,right_workers),_,Bands.plan(~Element,Model.pack(~Element,right))))
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,right)),(Model.pack(~Element,right),Model.layout(~Element,right)),Model.plan(~Element,right)) :
    Preserved(~Element,~Logical,~original,logical,Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,Tensors.Failed{problem},Shapes.same_extents(left_shape,right_shape),geometry,right_geometry,
      left_shape,right_shape,left_fill,right_fill,U32.max(left_workers,right_workers),(Model.pack(~Element,left),Model.layout(~Element,left)),_))
  unchanged(~Element,~Logical,~original,logical,geometry,left_shape,left_fill,Tensors.Failed{problem},U32.max(left_workers,right_workers),left,well_formed,values)

def tensor_band_failed(~Element: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  +name: String,+operations: F32,+geometry: Bands.Geometry,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,
  +left_status: Tensors.Status,+right_status: Tensors.Status,+left_workers: U32,+right_workers: U32,+left: Buffer.Model<Element>,+right: Model.Model<Element>,+problem: Tensors.Problem,
  failure: {Tensors.first_failure(left_status,right_status) == Tensors.Failed{problem} : Tensors.Status}) ->
  {Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Tensor{Buffer.pack(~Element,left),left_shape,left_fill,left_status,left_workers},
      Tensors.Banded{Model.pack(~Element,right),geometry,right_shape,right_fill,right_status,right_workers}) ==
    Tensors.Tensor{Buffer.pack(~Element,left),left_shape,left_fill,Tensors.Failed{problem},U32.max(left_workers,right_workers)} : Tensors.Tensor<Element>}:
  %Equal.sym(Tensors.Status,Tensors.first_failure(left_status,right_status),Tensors.Failed{problem},failure) :
    {Tensors.combine_right_plan_with(~Element,~kernel,name,operations,_,Shapes.same_extents(left_shape,right_shape),geometry,left_shape,right_shape,left_fill,right_fill,U32.max(left_workers,right_workers),
        Buffer.pack(~Element,left),Bands.plan(~Element,Model.pack(~Element,right))) ==
      Tensors.Tensor{Buffer.pack(~Element,left),left_shape,left_fill,Tensors.Failed{problem},U32.max(left_workers,right_workers)} : Tensors.Tensor<Element>}
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,right)),(Model.pack(~Element,right),Model.layout(~Element,right)),Model.plan(~Element,right)) :
    {Tensors.combine_right_plan_with(~Element,~kernel,name,operations,Tensors.Failed{problem},Shapes.same_extents(left_shape,right_shape),geometry,left_shape,right_shape,left_fill,right_fill,U32.max(left_workers,right_workers),
        Buffer.pack(~Element,left),_) == Tensors.Tensor{Buffer.pack(~Element,left),left_shape,left_fill,Tensors.Failed{problem},U32.max(left_workers,right_workers)} : Tensors.Tensor<Element>}
  {==}

# Operands of different extents leave the band layout: the call is the dense
# combination of the two tensors, which gathers a banded operand first and
# follows NumPy broadcasting (tensor_public_refinement).
def band_tensor_different(~Element: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  +name: String,+operations: F32,+geometry: Bands.Geometry,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,
  +left_workers: U32,+right_workers: U32,+left: Model.Model<Element>,+right: Buffer.Model<Element>,mismatch: {Shapes.same_extents(left_shape,right_shape) == False{} : Bool}) ->
  {Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Banded{Model.pack(~Element,left),geometry,left_shape,left_fill,Tensors.Usable{},left_workers},Tensors.Tensor{Buffer.pack(~Element,right),right_shape,right_fill,Tensors.Usable{},right_workers}) ==
    Tensors.combine_dense_with(~Element,~kernel,name,Tensors.Banded{Model.pack(~Element,left),geometry,left_shape,left_fill,Tensors.Usable{},U32.max(left_workers,right_workers)},Tensors.Tensor{Buffer.pack(~Element,right),right_shape,right_fill,Tensors.Usable{},right_workers}) : Tensors.Tensor<Element>}:
  %Equal.sym(Bool,Shapes.same_extents(left_shape,right_shape),False{},mismatch) :
    {Tensors.combine_bands_plan_with(~Element,~kernel,name,operations,Tensors.Usable{},_,geometry,left_shape,right_shape,left_fill,U32.max(left_workers,right_workers),Tensors.Tensor{Buffer.pack(~Element,right),right_shape,right_fill,Tensors.Usable{},right_workers},
        Bands.plan(~Element,Model.pack(~Element,left))) ==
      Tensors.combine_dense_with(~Element,~kernel,name,Tensors.Banded{Model.pack(~Element,left),geometry,left_shape,left_fill,Tensors.Usable{},U32.max(left_workers,right_workers)},Tensors.Tensor{Buffer.pack(~Element,right),right_shape,right_fill,Tensors.Usable{},right_workers}) : Tensors.Tensor<Element>}
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,left)),(Model.pack(~Element,left),Model.layout(~Element,left)),Model.plan(~Element,left)) :
    {Tensors.combine_bands_plan_with(~Element,~kernel,name,operations,Tensors.Usable{},False{},geometry,left_shape,right_shape,left_fill,U32.max(left_workers,right_workers),Tensors.Tensor{Buffer.pack(~Element,right),right_shape,right_fill,Tensors.Usable{},right_workers},_) ==
      Tensors.combine_dense_with(~Element,~kernel,name,Tensors.Banded{Model.pack(~Element,left),geometry,left_shape,left_fill,Tensors.Usable{},U32.max(left_workers,right_workers)},Tensors.Tensor{Buffer.pack(~Element,right),right_shape,right_fill,Tensors.Usable{},right_workers}) : Tensors.Tensor<Element>}
  {==}

def pair_different(~Element: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  +name: String,+operations: F32,+geometry: Bands.Geometry,+right_geometry: Bands.Geometry,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,
  +left_workers: U32,+right_workers: U32,+left: Model.Model<Element>,+right: Model.Model<Element>,mismatch: {Shapes.same_extents(left_shape,right_shape) == False{} : Bool}) ->
  {Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Banded{Model.pack(~Element,left),geometry,left_shape,left_fill,Tensors.Usable{},left_workers},Tensors.Banded{Model.pack(~Element,right),right_geometry,right_shape,right_fill,Tensors.Usable{},right_workers}) ==
    Tensors.combine_dense_with(~Element,~kernel,name,Tensors.Banded{Model.pack(~Element,left),geometry,left_shape,left_fill,Tensors.Usable{},U32.max(left_workers,right_workers)},
      Tensors.Banded{Model.pack(~Element,right),right_geometry,right_shape,right_fill,Tensors.Usable{},U32.max(left_workers,right_workers)}) : Tensors.Tensor<Element>}:
  %Equal.sym(Bool,Shapes.same_extents(left_shape,right_shape),False{},mismatch) :
    {Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,Tensors.Usable{},_,geometry,right_geometry,left_shape,right_shape,left_fill,right_fill,U32.max(left_workers,right_workers),
        Bands.plan(~Element,Model.pack(~Element,left)),Bands.plan(~Element,Model.pack(~Element,right))) ==
      Tensors.combine_dense_with(~Element,~kernel,name,Tensors.Banded{Model.pack(~Element,left),geometry,left_shape,left_fill,Tensors.Usable{},U32.max(left_workers,right_workers)},
        Tensors.Banded{Model.pack(~Element,right),right_geometry,right_shape,right_fill,Tensors.Usable{},U32.max(left_workers,right_workers)}) : Tensors.Tensor<Element>}
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,left)),(Model.pack(~Element,left),Model.layout(~Element,left)),Model.plan(~Element,left)) :
    {Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,Tensors.Usable{},False{},geometry,right_geometry,left_shape,right_shape,left_fill,right_fill,U32.max(left_workers,right_workers),
        _,Bands.plan(~Element,Model.pack(~Element,right))) ==
      Tensors.combine_dense_with(~Element,~kernel,name,Tensors.Banded{Model.pack(~Element,left),geometry,left_shape,left_fill,Tensors.Usable{},U32.max(left_workers,right_workers)},
        Tensors.Banded{Model.pack(~Element,right),right_geometry,right_shape,right_fill,Tensors.Usable{},U32.max(left_workers,right_workers)}) : Tensors.Tensor<Element>}
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,right)),(Model.pack(~Element,right),Model.layout(~Element,right)),Model.plan(~Element,right)) :
    {Tensors.combine_pair_plans_with(~Element,~kernel,name,operations,Tensors.Usable{},False{},geometry,right_geometry,left_shape,right_shape,left_fill,right_fill,U32.max(left_workers,right_workers),
        (Model.pack(~Element,left),Model.layout(~Element,left)),_) ==
      Tensors.combine_dense_with(~Element,~kernel,name,Tensors.Banded{Model.pack(~Element,left),geometry,left_shape,left_fill,Tensors.Usable{},U32.max(left_workers,right_workers)},
        Tensors.Banded{Model.pack(~Element,right),right_geometry,right_shape,right_fill,Tensors.Usable{},U32.max(left_workers,right_workers)}) : Tensors.Tensor<Element>}
  {==}

def tensor_band_different(~Element: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  +name: String,+operations: F32,+geometry: Bands.Geometry,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+left_fill: Element,+right_fill: Element,
  +left_workers: U32,+right_workers: U32,+left: Buffer.Model<Element>,+right: Model.Model<Element>,mismatch: {Shapes.same_extents(left_shape,right_shape) == False{} : Bool}) ->
  {Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Tensor{Buffer.pack(~Element,left),left_shape,left_fill,Tensors.Usable{},left_workers},Tensors.Banded{Model.pack(~Element,right),geometry,right_shape,right_fill,Tensors.Usable{},right_workers}) ==
    Tensors.combine_dense_with(~Element,~kernel,name,Tensors.Tensor{Buffer.pack(~Element,left),left_shape,left_fill,Tensors.Usable{},U32.max(left_workers,right_workers)},
      Tensors.Banded{Model.pack(~Element,right),geometry,right_shape,right_fill,Tensors.Usable{},U32.max(left_workers,right_workers)}) : Tensors.Tensor<Element>}:
  %Equal.sym(Bool,Shapes.same_extents(left_shape,right_shape),False{},mismatch) :
    {Tensors.combine_right_plan_with(~Element,~kernel,name,operations,Tensors.Usable{},_,geometry,left_shape,right_shape,left_fill,right_fill,U32.max(left_workers,right_workers),
        Buffer.pack(~Element,left),Bands.plan(~Element,Model.pack(~Element,right))) ==
      Tensors.combine_dense_with(~Element,~kernel,name,Tensors.Tensor{Buffer.pack(~Element,left),left_shape,left_fill,Tensors.Usable{},U32.max(left_workers,right_workers)},
        Tensors.Banded{Model.pack(~Element,right),geometry,right_shape,right_fill,Tensors.Usable{},U32.max(left_workers,right_workers)}) : Tensors.Tensor<Element>}
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,right)),(Model.pack(~Element,right),Model.layout(~Element,right)),Model.plan(~Element,right)) :
    {Tensors.combine_right_plan_with(~Element,~kernel,name,operations,Tensors.Usable{},False{},geometry,left_shape,right_shape,left_fill,right_fill,U32.max(left_workers,right_workers),
        Buffer.pack(~Element,left),_) ==
      Tensors.combine_dense_with(~Element,~kernel,name,Tensors.Tensor{Buffer.pack(~Element,left),left_shape,left_fill,Tensors.Usable{},U32.max(left_workers,right_workers)},
        Tensors.Banded{Model.pack(~Element,right),geometry,right_shape,right_fill,Tensors.Usable{},U32.max(left_workers,right_workers)}) : Tensors.Tensor<Element>}
  {==}
