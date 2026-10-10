# A window of a valid, supported problem is itself valid and supported when it
# keeps at least one row and the kernel still fits; its contract is the window
# tree for input and the original owners for weights and bias.
import Base
import ./blocked_pipeline_bounds.bend as BlockBounds
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../scalar_kernel.bend as ScalarConv
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/row_major_index.bend as RowMajor
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./input_storage.bend as Inputs
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ./window_band_values.bend as WindowBandValues
import ./window_samples.bend as Samples

def band(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat) -> WindowBandValues.Band:
  WindowBandValues.Band{Guards.input(problem),Guards.height(problem),Guards.width(problem),first,rows}

def count(+problem: ConvSpec.Problem,+rows: Nat) -> Nat:
  Nat.mul(Guards.input_channel_count(problem),Nat.mul(rows,Guards.width(problem)))

def valid_from(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+facts: Guards.Valid<problem>,
  +rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool},
  +kernel_fits: {Nat.is_le(Guards.kernel(problem),Nat.add(rows,Nat.mul(2n,Guards.pad(problem)))) == True{} : Bool}) ->
  {ConvSpec.valid(Samples.window(problem,first,rows)) == True{} : Bool}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Guards.Valid{+ci_positive,+co_positive,+height_positive,+width_positive,+kernel_positive,+stride_positive,+kernel_height,+kernel_width,
    +input_length,+weight_length,+bias_length} = facts
  +total = Nat.mul(ci,Nat.mul(rows,width))
  +windowed = {ConvSpec.Problem{ConvSpec.Shape{ci,co,rows,width,kernel,stride,padding},
    WindowBandValues.values(total,0n,WindowBandValues.Band{input,height,width,first,rows}),weights,bias} : ConvSpec.Problem}
  +window_length = {%Equal.sym(Nat,ConvSpec.length(WindowBandValues.values(total,0n,WindowBandValues.Band{input,height,width,first,rows})),total,
        WindowBandValues.length(total,0n,WindowBandValues.Band{input,height,width,first,rows})) : {Nat.is_eq(_,total) == True{} : Bool}
      Order.equal_reflexive(total) : {Guards.input_length_matches_shape(windowed) == True{} : Bool}}
  Boolean.all_hold(Guards.valid_conditions(windowed),
    Shape.Both{bias_length,Shape.Both{weight_length,Shape.Both{window_length,Shape.Both{kernel_width,Shape.Both{kernel_fits,Shape.Both{stride_positive,
      Shape.Both{kernel_positive,Shape.Both{width_positive,Shape.Both{rows_positive,Shape.Both{co_positive,Shape.Both{ci_positive,Unit{}}}}}}}}}}}})

def valid(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool},
  +kernel_fits: {Nat.is_le(Guards.kernel(problem),Nat.add(rows,Nat.mul(2n,Guards.pad(problem)))) == True{} : Bool}) ->
  {ConvSpec.valid(Samples.window(problem,first,rows)) == True{} : Bool}:
  valid_from(problem,first,rows,Guards.valid(problem,valid),rows_positive,kernel_fits)

def multiply_left_le(+factor: Nat,+smaller: Nat,+larger: Nat,+ordered: {Nat.is_le(smaller,larger) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(factor,smaller),Nat.mul(factor,larger)) == True{} : Bool}:
  %RowMajor.multiply_commutative(smaller,factor) : {Nat.is_le(_,Nat.mul(factor,larger)) == True{} : Bool}
  %RowMajor.multiply_commutative(larger,factor) : {Nat.is_le(Nat.mul(smaller,factor),_) == True{} : Bool}
  RowMajor.mul_le(smaller,larger,factor,ordered)

def axis_le(+smaller: Nat,+larger: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +ordered: {Nat.is_le(smaller,larger) == True{} : Bool}) ->
  {Nat.is_le(ConvSpec.axis_size(smaller,kernel,stride,padding),ConvSpec.axis_size(larger,kernel,stride,padding)) == True{} : Bool}:
  DivisionOrder.quotient_le(Nat.sub(Nat.add(smaller,Nat.mul(2n,padding)),kernel),Nat.sub(Nat.add(larger,Nat.mul(2n,padding)),kernel),stride,stride_positive,
    DivisionOrder.subtract_le(Nat.add(smaller,Nat.mul(2n,padding)),Nat.add(larger,Nat.mul(2n,padding)),kernel,
      Order.add_le(smaller,larger,Nat.mul(2n,padding),ordered)))

def scalar_from_formula(+limits: Machine.Limits,+problem: ConvSpec.Problem,
  +formula: {Guards.supported_formula(problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)) == True{} : Bool}) ->
  {ScalarConv.supported_with_limits(limits,problem) == True{} : Bool}:
  Machine.Limits{dimension,capacity,coordinate,half_word,facts} = limits
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  formula

def supported_from(+limits: Machine.Limits,+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+facts: Guards.Valid<problem>,
  +fast: Support.FastSupported<limits,problem>,+fits: {Nat.is_le(Nat.add(first,rows),Guards.height(problem)) == True{} : Bool}) ->
  {NativeConv.supports_packed_with_limits(limits,Samples.window(problem,first,rows)) == True{} : Bool}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Guards.Valid{ci_positive,co_positive,height_positive,width_positive,kernel_positive,+stride_positive,kernel_height,kernel_width,
    +input_length,weight_length,bias_length} = facts
  Support.FastSupported{+scalar,+packing,+output} = fast
  Guards.Supported{+ci_bound,+co_bound,+height_bound,+width_bound,+kernel_bound,+stride_bound,+padding_bound,+input_bound,+weight_bound,
    +output_bound,+vertical_bound,+horizontal_bound} = scalar
  +dimension = Machine.dimension(limits)
  +capacity = Machine.capacity(limits)
  +coordinate = Machine.coordinate(limits)
  +total = Nat.mul(ci,Nat.mul(rows,width))
  +values = WindowBandValues.values(total,0n,WindowBandValues.Band{input,height,width,first,rows})
  +windowed = {ConvSpec.Problem{ConvSpec.Shape{ci,co,rows,width,kernel,stride,padding},values,weights,bias} : ConvSpec.Problem}
  +rows_le = Order.transitive(rows,Nat.add(first,rows),height,Samples.le_after(first,rows),fits)
  +out_height = ConvSpec.axis_size(height,kernel,stride,padding)
  +out_rows = ConvSpec.axis_size(rows,kernel,stride,padding)
  +out_width = ConvSpec.axis_size(width,kernel,stride,padding)
  +axis = axis_le(rows,height,kernel,stride,padding,stride_positive,rows_le)
  +area = RowMajor.mul_le(out_rows,out_height,out_width,axis)
  +padded = BlockBounds.padded_le(Nat.mul(out_rows,out_width),Nat.mul(out_height,out_width),area)
  +input_fits = {%Equal.sym(Nat,ConvSpec.length(values),total,WindowBandValues.length(total,0n,WindowBandValues.Band{input,height,width,first,rows})) :
        {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
      Order.transitive(total,Nat.mul(ci,Nat.mul(height,width)),U32.to_nat(capacity),
        multiply_left_le(ci,Nat.mul(rows,width),Nat.mul(height,width),RowMajor.mul_le(rows,height,width,rows_le)),
        %Guards.reflect(ConvSpec.length(input),Nat.mul(ci,Nat.mul(height,width)),input_length) :
          {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
        input_bound) : {Guards.input_buffer_supported(windowed,dimension,capacity,coordinate) == True{} : Bool}}
  +formula = Boolean.all_hold(Guards.supported_conditions(windowed,dimension,capacity,coordinate),
    Shape.Both{horizontal_bound,
    Shape.Both{Order.transitive(Nat.add(Nat.mul(out_rows,stride),kernel),Nat.add(Nat.mul(out_height,stride),kernel),U32.to_nat(coordinate),
        Order.add_le(Nat.mul(out_rows,stride),Nat.mul(out_height,stride),kernel,RowMajor.mul_le(out_rows,out_height,stride,axis)),vertical_bound),
    Shape.Both{Order.transitive(Nat.mul(co,Nat.mul(out_rows,out_width)),Nat.mul(co,Nat.mul(out_height,out_width)),U32.to_nat(capacity),
        multiply_left_le(co,Nat.mul(out_rows,out_width),Nat.mul(out_height,out_width),area),output_bound),
    Shape.Both{weight_bound,
    Shape.Both{input_fits,
    Shape.Both{padding_bound,Shape.Both{stride_bound,Shape.Both{kernel_bound,Shape.Both{width_bound,
    Shape.Both{Order.transitive(rows,height,U32.to_nat(dimension),rows_le,height_bound),
    Shape.Both{co_bound,Shape.Both{ci_bound,Unit{}}}}}}}}}}}}})
  +window_scalar = scalar_from_formula(limits,windowed,formula)
  +inner = Nat.mul(ci,Nat.mul(kernel,kernel))
  +window_packing = {Order.transitive(Nat.mul(Nat.mul(Nat.div(Nat.add(Nat.mul(out_rows,out_width),7n),8n),8n),inner),
        Nat.mul(Nat.mul(Nat.div(Nat.add(Nat.mul(out_height,out_width),7n),8n),8n),inner),U32.to_nat(capacity),
        RowMajor.mul_le(Nat.mul(Nat.div(Nat.add(Nat.mul(out_rows,out_width),7n),8n),8n),Nat.mul(Nat.div(Nat.add(Nat.mul(out_height,out_width),7n),8n),8n),inner,padded),
        packing) : {Support.packed_bound(limits,windowed) == True{} : Bool}}
  +window_output = {Order.transitive(Nat.mul(Nat.mul(Nat.div(Nat.add(Nat.mul(out_rows,out_width),7n),8n),8n),co),
        Nat.mul(Nat.mul(Nat.div(Nat.add(Nat.mul(out_height,out_width),7n),8n),8n),co),U32.to_nat(capacity),
        RowMajor.mul_le(Nat.mul(Nat.div(Nat.add(Nat.mul(out_rows,out_width),7n),8n),8n),Nat.mul(Nat.div(Nat.add(Nat.mul(out_height,out_width),7n),8n),8n),co,padded),
        output) : {Support.output_bound(limits,windowed) == True{} : Bool}}
  %Equal.sym(Bool,NativeConv.supports_packed_with_limits(limits,windowed),
    ScalarConv.supported_with_limits(limits,windowed) && Support.packed_bound(limits,windowed) && Support.output_bound(limits,windowed),
    Support.packed_guard_matches_original_conditions(limits,windowed)) : {_ == True{} : Bool}
  Boolean.all_hold([Support.output_bound(limits,windowed),Support.packed_bound(limits,windowed),ScalarConv.supported_with_limits(limits,windowed)],
    Shape.Both{window_output,Shape.Both{window_packing,Shape.Both{window_scalar,Unit{}}}})

def supported(+limits: Machine.Limits,+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(first,rows),Guards.height(problem)) == True{} : Bool}) ->
  {NativeConv.supports_packed_with_limits(limits,Samples.window(problem,first,rows)) == True{} : Bool}:
  supported_from(limits,problem,first,rows,Guards.valid(problem,valid),Support.fast(limits,problem,supported),fits)

# The window tree carries the band; weights and bias keep their original owners.
def contract(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+first: Nat,+rows: Nat,+tree: Tree.Tree<F32>,
  +contents: WindowBandValues.Contents(count(problem,rows),band(problem,first,rows),tree),
  +room: {Nat.is_le(count(problem,rows),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool}) ->
  Inputs.Contract<Samples.window(problem,first,rows)>:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Inputs.Contract{input_evidence,+weight_evidence,+bias_evidence} = storage_contract
  Inputs.Contract{WindowBandValues.evidence(Nat.mul(ci,Nat.mul(rows,width)),WindowBandValues.Band{input,height,width,first,rows},tree,contents,room),weight_evidence,bias_evidence}

# A band window has no whole-image owner: its tree carries the band and the
# supplied evidence carries weights and bias.
def band_contract(+problem: ConvSpec.Problem,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(Guards.weights(problem)),Guards.weights(problem)>,
  +bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(Guards.bias(problem)),Guards.bias(problem)>,
  +first: Nat,+rows: Nat,+tree: Tree.Tree<F32>,+contents: WindowBandValues.Contents(count(problem,rows),band(problem,first,rows),tree),
  +room: {Nat.is_le(count(problem,rows),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool}) ->
  Inputs.Contract<Samples.window(problem,first,rows)>:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Inputs.Contract{WindowBandValues.evidence(Nat.mul(ci,Nat.mul(rows,width)),WindowBandValues.Band{input,height,width,first,rows},tree,contents,room),weight_evidence,bias_evidence}
