# A band that passes the pipeline coverage check, as natural-number facts:
# it lies inside its parent window and covers every read of its span.
import Base
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/kernel_window.bend as KernelWindow
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/u32_shift.bend as U32Shift
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/row_major_index.bend as RowMajor
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./reduction_coordinate_bounds.bend as Coordinates
import ./window_samples.bend as Samples
import ./window_positions.bend as Positions
import ./window_start.bend as Start
import ./window_contents.bend as Contents

def or_cases(+first: Bool,+second: Bool,+truth: {first || second == True{} : Bool}) ->
  Either<&2,&2,{first == True{} : Bool},{second == True{} : Bool}>:
  match first:
    case True{}: Inl{{==}}
    case False{}: Inr{truth}

def word_zero(+value: U32,+zero: {U32.is_zero(value) == True{} : Bool}) -> {U32.to_nat(value) == 0n : Nat}:
  Order.equal_of_is_eq(U32.to_nat(value),0n,
    %Comparison.eq(value,0) : {_ == True{} : Bool}
    zero)

def word_equal(+left: U32,+right: U32,+same: {U32.is_eq(left,right) == True{} : Bool}) -> {U32.to_nat(left) == U32.to_nat(right) : Nat}:
  Order.equal_of_is_eq(U32.to_nat(left),U32.to_nat(right),
    %Comparison.eq(left,right) : {_ == True{} : Bool}
    same)

def vertical_bound(+limits: Machine.Limits,+problem: ConvSpec.Problem,
  +support: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>) ->
  {Nat.is_le(Nat.add(Nat.mul(ConvSpec.out_height(problem),Guards.stride(problem)),Guards.kernel(problem)),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  Guards.Supported{ci_bound,co_bound,height_bound,width_bound,kernel_bound,stride_bound,padding_bound,input_bound,weight_bound,
    output_bound,+vertical,horizontal} = support
  vertical

# Every row coordinate of the padded image stays below the coordinate limit.
def padded_rows_bound(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +support: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>) ->
  {Nat.is_le(Nat.add(Guards.height(problem),Nat.mul(2n,Guards.pad(problem))),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +stride_positive = Contents.valid_condition(problem,valid,5n)
  +kernel_fits = Contents.valid_condition(problem,valid,4n)
  +whole = Nat.add(height,Nat.mul(2n,padding))
  +reach = Nat.sub(whole,kernel)
  +restored = {%Order.commutative(kernel,reach) : {_ == whole : Nat}
    Positions.add_sub(kernel,whole,kernel_fits) : {Nat.add(reach,kernel) == whole : Nat}}
  Order.transitive(whole,Nat.add(Nat.mul(1n+Nat.div(reach,stride),stride),kernel),U32.to_nat(Machine.coordinate(limits)),
    %restored : {Nat.is_le(_,Nat.add(Nat.mul(1n+Nat.div(reach,stride),stride),kernel)) == True{} : Bool}
    Order.add_le(reach,Nat.mul(1n+Nat.div(reach,stride),stride),kernel,
      Bounds.strict_le(reach,Nat.mul(1n+Nat.div(reach,stride),stride),DivisionOrder.next_above(reach,stride,stride_positive))),
    vertical_bound(limits,problem,support))

def out_width_positive(+problem: ConvSpec.Problem) -> {Nat.is_lt(0n,ConvSpec.out_width(problem)) == True{} : Bool}:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  ConvSpec.Shape{ci,co,height,width,kernel,stride,padding} = shape
  {==}

def conditions(+parent_first: U32,+parent_rows: U32,+height: U32,+kernel: U32,+stride: U32,+padding: U32,+output_width: U32,+offset: U32,+span: U32,
  +first: U32,+rows: U32) -> List<&2,Bool>:
  +parent_end = (parent_first + parent_rows : U32)
  +top = (offset / output_width * stride : U32)
  +bottom = ((offset + span - 1) / output_width * stride + kernel : U32)
  +end = (first + rows : U32)
  [U32.is_le(kernel,(rows + padding + padding : U32)),(U32.is_eq(end,height) || U32.is_le(bottom,(end + padding : U32))),
    (U32.is_zero(first) || U32.is_le((first + padding : U32),top)),U32.is_zero((first % stride : U32)),U32.is_le(rows,(parent_end - first : U32)),
    U32.is_le(first,parent_end),U32.is_le(parent_first,first),U32.is_lt(0,rows),U32.is_lt(0,span)]

def condition(+parent_first: U32,+parent_rows: U32,+height: U32,+kernel: U32,+stride: U32,+padding: U32,+output_width: U32,+offset: U32,+span: U32,
  +first: U32,+rows: U32,+index: Nat,
  +checked: {KernelWindow.covers(parent_first,parent_rows,height,kernel,stride,padding,output_width,offset,span,KernelWindow.Band{first,rows}) == True{} : Bool}) ->
  {Boolean.condition_at(conditions(parent_first,parent_rows,height,kernel,stride,padding,output_width,offset,span,first,rows),index) == True{} : Bool}:
  Boolean.every_condition_holds(conditions(parent_first,parent_rows,height,kernel,stride,padding,output_width,offset,span,first,rows),index,checked)

type CopyFacts<-problem: ConvSpec.Problem,-parent_first: U32,-parent_rows: U32,-first: U32,-rows: U32,-offset: U32,-span: U32> is Data:
  CopyFacts{cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
    after: {Nat.is_le(U32.to_nat(parent_first),U32.to_nat(first)) == True{} : Bool},
    within: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.add(U32.to_nat(parent_first),U32.to_nat(parent_rows))) == True{} : Bool}}

def top_cover(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,+first: U32,
  +width_blocks: Nat,+width: {Nat.mul(width_blocks,8n) == ConvSpec.out_width(problem) : Nat},
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +first_le: {Nat.is_le(U32.to_nat(first),Guards.height(problem)) == True{} : Bool},
  +zero_or_below: Either<&2,&2,{U32.is_zero(first) == True{} : Bool},
    {U32.is_le((first + U32.from_nat(Guards.pad(problem)) : U32),(offset / U32.from_nat(ConvSpec.out_width(problem)) * U32.from_nat(Guards.stride(problem)) : U32)) == True{} : Bool}>) ->
  Either<&2,&2,{U32.to_nat(first) == 0n : Nat},
    {Nat.is_le(Nat.add(U32.to_nat(first),Guards.pad(problem)),Nat.mul(Nat.div(U32.to_nat(offset),ConvSpec.out_width(problem)),Guards.stride(problem))) == True{} : Bool}>:
  match zero_or_below:
    case Inl{+zero}: Inl{word_zero(first,zero)}
    case Inr{+below}:
      +facts = Guards.valid(problem,valid)
      +fast = Support.fast(limits,problem,supported)
      +support = Padded.scalar(limits,problem,fast)
      +height = Guards.height(problem)
      +padding = Guards.pad(problem)
      +stride = Guards.stride(problem)
      +out_width = ConvSpec.out_width(problem)
      +out_height = ConvSpec.out_height(problem)
      +coordinate = Machine.coordinate(limits)
      +padding_word = U32.from_nat(padding)
      +width_word = U32.from_nat(out_width)
      +stride_word = U32.from_nat(stride)
      +start = U32.to_nat(offset)
      +pad_exact = SupportedCoordinateBounds.pad_exact(limits,problem,support)
      +width_exact = Coordinates.width_exact(limits,problem,facts,support)
      +stride_exact = Start.stride_word(limits,problem,support)
      +rows_bound = padded_rows_bound(limits,problem,valid,support)
      +first_room = Order.transitive(Nat.add(U32.to_nat(first),padding),Nat.add(height,Nat.mul(2n,padding)),U32.to_nat(coordinate),
        Order.transitive(Nat.add(U32.to_nat(first),padding),Nat.add(height,padding),Nat.add(height,Nat.mul(2n,padding)),Order.add_le(U32.to_nat(first),height,padding,first_le),
          %Equal.sym(Bool,Nat.is_le(Nat.add(height,padding),Nat.add(height,Nat.mul(2n,padding))),Nat.is_le(padding,Nat.mul(2n,padding)),
            Samples.le_offset(height,padding,Nat.mul(2n,padding))) : {_ == True{} : Bool}
          Order.le_add(padding,Nat.mul(1n,padding))),rows_bound)
      +first_exact = {%pad_exact : {U32.to_nat((first + padding_word : U32)) == Nat.add(U32.to_nat(first),_) : Nat}
        Arithmetic.add(first,padding_word,coordinate,Limits.coordinate_fits(limits),
          %Equal.sym(Nat,U32.to_nat(padding_word),padding,pad_exact) : {Nat.is_le(Nat.add(U32.to_nat(first),_),U32.to_nat(coordinate)) == True{} : Bool}
          first_room) : {U32.to_nat((first + padding_word : U32)) == Nat.add(U32.to_nat(first),padding) : Nat}}
      +grid = {%Positions.aligned_padded(problem,width_blocks,width) :
          {Nat.is_le(Nat.add(start,U32.to_nat(span)),_) == True{} : Bool}
        range : {Nat.is_le(Nat.add(start,U32.to_nat(span)),Nat.mul(out_height,out_width)) == True{} : Bool}}
      +row_le = {%Positions.exact_quotient(out_height,out_width,out_width_positive(problem)) : {Nat.is_le(Nat.div(start,out_width),_) == True{} : Bool}
        DivisionOrder.quotient_le(start,Nat.mul(out_height,out_width),out_width,out_width_positive(problem),
          Order.transitive(start,Nat.add(start,U32.to_nat(span)),Nat.mul(out_height,out_width),Order.le_add(start,U32.to_nat(span)),grid)) :
        {Nat.is_le(Nat.div(start,out_width),out_height) == True{} : Bool}}
      +quotient_word = (offset / width_word : U32)
      +quotient_exact = {Equal.trans(Nat,U32.to_nat(quotient_word),Nat.div(start,U32.to_nat(width_word)),Nat.div(start,out_width),
          U32Division.divide(offset,width_word,Machine.half_word(limits),Limits.half_word_fits(limits),
            %Equal.sym(Nat,U32.to_nat(width_word),out_width,width_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
            %Limits.capacity_doubles_to_half(limits) : {Nat.is_le(Nat.double(out_width),U32.to_nat(_)) == True{} : Bool}
            %Equal.sym(Nat,U32.to_nat(U32.shl(Machine.capacity(limits))),Nat.double(U32.to_nat(Machine.capacity(limits))),
              U32Shift.double(Machine.capacity(limits),Limits.capacity_below_half(limits))) : {Nat.is_le(Nat.double(out_width),_) == True{} : Bool}
            Order.double_le(out_width,U32.to_nat(Machine.capacity(limits)),Coordinates.width_bound(limits,problem,facts,support)),
            %Equal.sym(Nat,U32.to_nat(width_word),out_width,width_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
            out_width_positive(problem)),
          %Equal.sym(Nat,U32.to_nat(width_word),out_width,width_exact) : {Nat.div(start,_) == Nat.div(start,out_width) : Nat}
          {==}) : {U32.to_nat(quotient_word) == Nat.div(start,out_width) : Nat}}
      +top_room = Order.transitive(Nat.mul(Nat.div(start,out_width),stride),Nat.mul(out_height,stride),U32.to_nat(coordinate),
        RowMajor.mul_le(Nat.div(start,out_width),out_height,stride,row_le),
        Order.transitive(Nat.mul(out_height,stride),Nat.add(Nat.mul(out_height,stride),Guards.kernel(problem)),U32.to_nat(coordinate),
          Order.le_add(Nat.mul(out_height,stride),Guards.kernel(problem)),vertical_bound(limits,problem,support)))
      +top_exact = {%stride_exact : {U32.to_nat((quotient_word * stride_word : U32)) == Nat.mul(Nat.div(start,out_width),_) : Nat}
        %quotient_exact : {U32.to_nat((quotient_word * stride_word : U32)) == Nat.mul(_,U32.to_nat(stride_word)) : Nat}
        Arithmetic.multiply(quotient_word,stride_word,coordinate,Limits.coordinate_fits(limits),
          %Equal.sym(Nat,U32.to_nat(quotient_word),Nat.div(start,out_width),quotient_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(stride_word)),U32.to_nat(coordinate)) == True{} : Bool}
          %Equal.sym(Nat,U32.to_nat(stride_word),stride,stride_exact) : {Nat.is_le(Nat.mul(Nat.div(start,out_width),_),U32.to_nat(coordinate)) == True{} : Bool}
          top_room) : {U32.to_nat((quotient_word * stride_word : U32)) == Nat.mul(Nat.div(start,out_width),stride) : Nat}}
      Inr{%first_exact : {Nat.is_le(_,Nat.mul(Nat.div(start,out_width),stride)) == True{} : Bool}
        %top_exact : {Nat.is_le(U32.to_nat((first + padding_word : U32)),_) == True{} : Bool}
        Bounds.as_nat_le((first + padding_word : U32),(quotient_word * stride_word : U32),below)}

def height_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Nat.is_le(Guards.height(problem),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(Guards.height(problem),Nat.mul(Guards.height(problem),Guards.width(problem)),U32.to_nat(Machine.capacity(limits)),
    RowMajor.product_covers(Guards.height(problem),Guards.width(problem),Contents.valid_condition(problem,valid,7n)),
    Contents.area_room(limits,problem,valid,supported))

def bottom_cover(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,+first: U32,+rows: U32,
  +width_blocks: Nat,+width: {Nat.mul(width_blocks,8n) == ConvSpec.out_width(problem) : Nat},
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Guards.height(problem)) == True{} : Bool},
  +end_or_below: Either<&2,&2,{U32.is_eq((first + rows : U32),U32.from_nat(Guards.height(problem))) == True{} : Bool},
    {U32.is_le(((offset + span - 1) / U32.from_nat(ConvSpec.out_width(problem)) * U32.from_nat(Guards.stride(problem)) + U32.from_nat(Guards.kernel(problem)) : U32),
      ((first + rows : U32) + U32.from_nat(Guards.pad(problem)) : U32)) == True{} : Bool}>) ->
  Either<&2,&2,{Guards.height(problem) == Nat.add(U32.to_nat(first),U32.to_nat(rows)) : Nat},
    {Nat.is_le(Nat.add(Nat.mul(Nat.div(Nat.sub(Nat.add(U32.to_nat(offset),U32.to_nat(span)),1n),ConvSpec.out_width(problem)),Guards.stride(problem)),Guards.kernel(problem)),
      Nat.add(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Guards.pad(problem))) == True{} : Bool}>:
  match end_or_below:
    case Inl{+same}:
      +facts = Guards.valid(problem,valid)
      +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
      +capacity = Machine.capacity(limits)
      +end_word = (first + rows : U32)
      +end_exact = Arithmetic.add(first,rows,capacity,Limits.capacity_fits(limits),
        Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Guards.height(problem),U32.to_nat(capacity),fits,height_room(limits,problem,valid,supported)))
      Inl{Equal.trans(Nat,Guards.height(problem),U32.to_nat(U32.from_nat(Guards.height(problem))),Nat.add(U32.to_nat(first),U32.to_nat(rows)),
        Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.height(problem))),Guards.height(problem),SupportedCoordinateBounds.height_exact(limits,problem,support)),
        Equal.trans(Nat,U32.to_nat(U32.from_nat(Guards.height(problem))),U32.to_nat(end_word),Nat.add(U32.to_nat(first),U32.to_nat(rows)),
          Equal.sym(Nat,U32.to_nat(end_word),U32.to_nat(U32.from_nat(Guards.height(problem))),word_equal(end_word,U32.from_nat(Guards.height(problem)),same)),
          end_exact))}
    case Inr{+below}:
      +facts = Guards.valid(problem,valid)
      +fast = Support.fast(limits,problem,supported)
      +support = Padded.scalar(limits,problem,fast)
      +capacity = Machine.capacity(limits)
      +coordinate = Machine.coordinate(limits)
      +height = Guards.height(problem)
      +padding = Guards.pad(problem)
      +stride = Guards.stride(problem)
      +kernel = Guards.kernel(problem)
      +out_width = ConvSpec.out_width(problem)
      +out_height = ConvSpec.out_height(problem)
      +width_word = U32.from_nat(out_width)
      +stride_word = U32.from_nat(stride)
      +kernel_word = U32.from_nat(kernel)
      +padding_word = U32.from_nat(padding)
      +start = U32.to_nat(offset)
      +count = U32.to_nat(span)
      +cap = U32.to_nat(capacity)
      +width_exact = Coordinates.width_exact(limits,problem,facts,support)
      +stride_exact = Start.stride_word(limits,problem,support)
      +kernel_exact = SupportedCoordinateBounds.kernel_exact(limits,problem,support)
      +pad_exact = SupportedCoordinateBounds.pad_exact(limits,problem,support)
      +rows_bound = padded_rows_bound(limits,problem,valid,support)
      +vertical = vertical_bound(limits,problem,support)
      +grid = {%Positions.aligned_padded(problem,width_blocks,width) : {Nat.is_le(Nat.add(start,count),_) == True{} : Bool}
        range : {Nat.is_le(Nat.add(start,count),Nat.mul(out_height,out_width)) == True{} : Bool}}
      +whole_room = Order.transitive(Nat.add(start,count),Support.padded(problem),cap,range,Padded.padded_bound(limits,problem,facts,fast))
      +sum_word = (offset + span : U32)
      +sum_exact = Arithmetic.add(offset,span,capacity,Limits.capacity_fits(limits),whole_room)
      +whole_positive = Samples.lt_le(0n,count,Nat.add(start,count),span_positive,Samples.le_after(start,count))
      +last = Nat.sub(Nat.add(start,count),1n)
      +last_word = (sum_word - 1 : U32)
      +last_exact = {%sum_exact : {U32.to_nat(last_word) == Nat.sub(_,1n) : Nat}
        Arithmetic.subtract(sum_word,1,capacity,Limits.capacity_fits(limits),
          %Equal.sym(Nat,U32.to_nat(sum_word),Nat.add(start,count),sum_exact) : {Nat.is_le(1n,_) == True{} : Bool}
          Order.strict_follow(0n,Nat.add(start,count),whole_positive),
          %Equal.sym(Nat,U32.to_nat(sum_word),Nat.add(start,count),sum_exact) : {Nat.is_le(Nat.sub(_,1n),cap) == True{} : Bool}
          Order.transitive(last,Nat.add(start,count),cap,Order.sub_le(Nat.add(start,count),1n),whole_room)) :
        {U32.to_nat(last_word) == last : Nat}}
      +last_inside = {%Order.positive_has_last(Nat.add(start,count),whole_positive) : {Nat.is_lt(last,_) == True{} : Bool}
        Order.lt_succ(last,last,Order.reflexive(last)) : {Nat.is_lt(last,Nat.add(start,count)) == True{} : Bool}}
      +row = Nat.div(last,out_width)
      +row_below = Positions.quotient_lt(last,out_height,out_width,out_width_positive(problem),Samples.lt_le(last,Nat.add(start,count),Nat.mul(out_height,out_width),last_inside,grid))
      +quotient_word = (last_word / width_word : U32)
      +quotient_exact = {Equal.trans(Nat,U32.to_nat(quotient_word),Nat.div(U32.to_nat(last_word),U32.to_nat(width_word)),row,
          U32Division.divide(last_word,width_word,Machine.half_word(limits),Limits.half_word_fits(limits),
            %Equal.sym(Nat,U32.to_nat(width_word),out_width,width_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
            %Limits.capacity_doubles_to_half(limits) : {Nat.is_le(Nat.double(out_width),U32.to_nat(_)) == True{} : Bool}
            %Equal.sym(Nat,U32.to_nat(U32.shl(capacity)),Nat.double(cap),U32Shift.double(capacity,Limits.capacity_below_half(limits))) :
              {Nat.is_le(Nat.double(out_width),_) == True{} : Bool}
            Order.double_le(out_width,cap,Coordinates.width_bound(limits,problem,facts,support)),
            %Equal.sym(Nat,U32.to_nat(width_word),out_width,width_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
            out_width_positive(problem)),
          %Equal.sym(Nat,U32.to_nat(width_word),out_width,width_exact) : {Nat.div(U32.to_nat(last_word),_) == row : Nat}
          %Equal.sym(Nat,U32.to_nat(last_word),last,last_exact) : {Nat.div(_,out_width) == row : Nat}
          {==}) : {U32.to_nat(quotient_word) == row : Nat}}
      +product_room = Order.transitive(Nat.mul(row,stride),Nat.mul(out_height,stride),U32.to_nat(coordinate),
        RowMajor.mul_le(row,out_height,stride,Bounds.strict_le(row,out_height,row_below)),
        Order.transitive(Nat.mul(out_height,stride),Nat.add(Nat.mul(out_height,stride),kernel),U32.to_nat(coordinate),Order.le_add(Nat.mul(out_height,stride),kernel),vertical))
      +product_word = (quotient_word * stride_word : U32)
      +product_exact = {%stride_exact : {U32.to_nat(product_word) == Nat.mul(row,_) : Nat}
        %quotient_exact : {U32.to_nat(product_word) == Nat.mul(_,U32.to_nat(stride_word)) : Nat}
        Arithmetic.multiply(quotient_word,stride_word,coordinate,Limits.coordinate_fits(limits),
          %Equal.sym(Nat,U32.to_nat(quotient_word),row,quotient_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(stride_word)),U32.to_nat(coordinate)) == True{} : Bool}
          %Equal.sym(Nat,U32.to_nat(stride_word),stride,stride_exact) : {Nat.is_le(Nat.mul(row,_),U32.to_nat(coordinate)) == True{} : Bool}
          product_room) : {U32.to_nat(product_word) == Nat.mul(row,stride) : Nat}}
      +bottom_word = (product_word + kernel_word : U32)
      +bottom_room = Order.transitive(Nat.add(Nat.mul(row,stride),kernel),Nat.add(Nat.mul(out_height,stride),kernel),U32.to_nat(coordinate),
        Order.add_le(Nat.mul(row,stride),Nat.mul(out_height,stride),kernel,RowMajor.mul_le(row,out_height,stride,Bounds.strict_le(row,out_height,row_below))),vertical)
      +bottom_exact = {%kernel_exact : {U32.to_nat(bottom_word) == Nat.add(Nat.mul(row,stride),_) : Nat}
        %product_exact : {U32.to_nat(bottom_word) == Nat.add(_,U32.to_nat(kernel_word)) : Nat}
        Arithmetic.add(product_word,kernel_word,coordinate,Limits.coordinate_fits(limits),
          %Equal.sym(Nat,U32.to_nat(product_word),Nat.mul(row,stride),product_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(kernel_word)),U32.to_nat(coordinate)) == True{} : Bool}
          %Equal.sym(Nat,U32.to_nat(kernel_word),kernel,kernel_exact) : {Nat.is_le(Nat.add(Nat.mul(row,stride),_),U32.to_nat(coordinate)) == True{} : Bool}
          bottom_room) : {U32.to_nat(bottom_word) == Nat.add(Nat.mul(row,stride),kernel) : Nat}}
      +end_word = (first + rows : U32)
      +end = Nat.add(U32.to_nat(first),U32.to_nat(rows))
      +end_exact = Arithmetic.add(first,rows,capacity,Limits.capacity_fits(limits),
        Order.transitive(end,height,cap,fits,height_room(limits,problem,valid,supported)))
      +end_room = Order.transitive(Nat.add(end,padding),Nat.add(height,Nat.mul(2n,padding)),U32.to_nat(coordinate),
        Order.transitive(Nat.add(end,padding),Nat.add(height,padding),Nat.add(height,Nat.mul(2n,padding)),Order.add_le(end,height,padding,fits),
          %Equal.sym(Bool,Nat.is_le(Nat.add(height,padding),Nat.add(height,Nat.mul(2n,padding))),Nat.is_le(padding,Nat.mul(2n,padding)),
            Samples.le_offset(height,padding,Nat.mul(2n,padding))) : {_ == True{} : Bool}
          Order.le_add(padding,Nat.mul(1n,padding))),rows_bound)
      +limit_word = (end_word + padding_word : U32)
      +limit_exact = {%pad_exact : {U32.to_nat(limit_word) == Nat.add(end,_) : Nat}
        %end_exact : {U32.to_nat(limit_word) == Nat.add(_,U32.to_nat(padding_word)) : Nat}
        Arithmetic.add(end_word,padding_word,coordinate,Limits.coordinate_fits(limits),
          %Equal.sym(Nat,U32.to_nat(end_word),end,end_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(padding_word)),U32.to_nat(coordinate)) == True{} : Bool}
          %Equal.sym(Nat,U32.to_nat(padding_word),padding,pad_exact) : {Nat.is_le(Nat.add(end,_),U32.to_nat(coordinate)) == True{} : Bool}
          end_room) : {U32.to_nat(limit_word) == Nat.add(end,padding) : Nat}}
      Inr{%bottom_exact : {Nat.is_le(_,Nat.add(end,padding)) == True{} : Bool}
        %limit_exact : {Nat.is_le(U32.to_nat(bottom_word),_) == True{} : Bool}
        Bounds.as_nat_le(bottom_word,limit_word,below)}

# A band that passed its coverage check lies inside the parent and covers the span.
def copy_facts(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,
  +parent_first: U32,+parent_rows: U32,+first: U32,+rows: U32,
  +width_blocks: Nat,+width: {Nat.mul(width_blocks,8n) == ConvSpec.out_width(problem) : Nat},
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +parent_fits: {Nat.is_le(Nat.add(U32.to_nat(parent_first),U32.to_nat(parent_rows)),Guards.height(problem)) == True{} : Bool},
  +checked: {KernelWindow.covers(parent_first,parent_rows,U32.from_nat(Guards.height(problem)),U32.from_nat(Guards.kernel(problem)),U32.from_nat(Guards.stride(problem)),
    U32.from_nat(Guards.pad(problem)),U32.from_nat(ConvSpec.out_width(problem)),offset,span,KernelWindow.Band{first,rows}) == True{} : Bool}) ->
  CopyFacts<problem,parent_first,parent_rows,first,rows,offset,span>:
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +capacity = Machine.capacity(limits)
  +coordinate = Machine.coordinate(limits)
  +cap = U32.to_nat(capacity)
  +fits = Limits.capacity_fits(limits)
  +height = Guards.height(problem)
  +stride = Guards.stride(problem)
  +padding = Guards.pad(problem)
  +kernel = Guards.kernel(problem)
  +height_word = U32.from_nat(height)
  +kernel_word = U32.from_nat(kernel)
  +stride_word = U32.from_nat(stride)
  +padding_word = U32.from_nat(padding)
  +width_word = U32.from_nat(ConvSpec.out_width(problem))
  +start = U32.to_nat(first)
  +count = U32.to_nat(rows)
  +parent_start = U32.to_nat(parent_first)
  +parent_count = U32.to_nat(parent_rows)
  +c0 = condition(parent_first,parent_rows,height_word,kernel_word,stride_word,padding_word,width_word,offset,span,first,rows,0n,checked)
  +c1 = condition(parent_first,parent_rows,height_word,kernel_word,stride_word,padding_word,width_word,offset,span,first,rows,1n,checked)
  +c2 = condition(parent_first,parent_rows,height_word,kernel_word,stride_word,padding_word,width_word,offset,span,first,rows,2n,checked)
  +c3 = condition(parent_first,parent_rows,height_word,kernel_word,stride_word,padding_word,width_word,offset,span,first,rows,3n,checked)
  +c4 = condition(parent_first,parent_rows,height_word,kernel_word,stride_word,padding_word,width_word,offset,span,first,rows,4n,checked)
  +c5 = condition(parent_first,parent_rows,height_word,kernel_word,stride_word,padding_word,width_word,offset,span,first,rows,5n,checked)
  +c6 = condition(parent_first,parent_rows,height_word,kernel_word,stride_word,padding_word,width_word,offset,span,first,rows,6n,checked)
  +c7 = condition(parent_first,parent_rows,height_word,kernel_word,stride_word,padding_word,width_word,offset,span,first,rows,7n,checked)
  +c8 = condition(parent_first,parent_rows,height_word,kernel_word,stride_word,padding_word,width_word,offset,span,first,rows,8n,checked)
  +span_positive = Bounds.as_nat_lt(0,span,c8)
  +rows_positive = Bounds.as_nat_lt(0,rows,c7)
  +after = Bounds.as_nat_le(parent_first,first,c6)
  +height_bound = height_room(limits,problem,valid,supported)
  +parent_end = Nat.add(parent_start,parent_count)
  +parent_word = (parent_first + parent_rows : U32)
  +parent_exact = Arithmetic.add(parent_first,parent_rows,capacity,fits,Order.transitive(parent_end,height,cap,parent_fits,height_bound))
  +first_below = Bounds.as_nat_le(first,parent_word,c5)
  +first_le_end = {%parent_exact : {Nat.is_le(start,_) == True{} : Bool} first_below : {Nat.is_le(start,parent_end) == True{} : Bool}}
  +rest_word = (parent_word - first : U32)
  +rest_exact = {%parent_exact : {U32.to_nat(rest_word) == Nat.sub(_,start) : Nat}
    Arithmetic.subtract(parent_word,first,capacity,fits,first_below,
      %Equal.sym(Nat,U32.to_nat(parent_word),parent_end,parent_exact) : {Nat.is_le(Nat.sub(_,start),cap) == True{} : Bool}
      Order.transitive(Nat.sub(parent_end,start),parent_end,cap,Order.sub_le(parent_end,start),Order.transitive(parent_end,height,cap,parent_fits,height_bound))) :
    {U32.to_nat(rest_word) == Nat.sub(parent_end,start) : Nat}}
  +rows_le = {%rest_exact : {Nat.is_le(count,_) == True{} : Bool} Bounds.as_nat_le(rows,rest_word,c4) : {Nat.is_le(count,Nat.sub(parent_end,start)) == True{} : Bool}}
  +within = {%Positions.add_sub(start,parent_end,first_le_end) : {Nat.is_le(Nat.add(start,count),_) == True{} : Bool}
    %Equal.sym(Bool,Nat.is_le(Nat.add(start,count),Nat.add(start,Nat.sub(parent_end,start))),Nat.is_le(count,Nat.sub(parent_end,start)),
      Samples.le_offset(start,count,Nat.sub(parent_end,start))) : {_ == True{} : Bool}
    rows_le : {Nat.is_le(Nat.add(start,count),parent_end) == True{} : Bool}}
  +fits_height = Order.transitive(Nat.add(start,count),parent_end,height,within,parent_fits)
  +first_le = Order.transitive(start,Nat.add(start,count),height,Order.le_add(start,count),fits_height)
  +stride_positive = Contents.valid_condition(problem,valid,5n)
  +stride_exact = Start.stride_word(limits,problem,support)
  +remainder_word = (first % stride_word : U32)
  +remainder_exact = {Equal.trans(Nat,U32.to_nat(remainder_word),Nat.mod(start,U32.to_nat(stride_word)),Nat.mod(start,stride),
      U32Division.modulo(first,stride_word,coordinate,Limits.coordinate_fits(limits),Start.stride_divisor(limits,problem,support),
        %Equal.sym(Nat,U32.to_nat(stride_word),stride,stride_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
        stride_positive),
      %Equal.sym(Nat,U32.to_nat(stride_word),stride,stride_exact) : {Nat.mod(start,_) == Nat.mod(start,stride) : Nat}
      {==}) : {U32.to_nat(remainder_word) == Nat.mod(start,stride) : Nat}}
  +remainder_zero = Equal.trans(Nat,Nat.mod(start,stride),U32.to_nat(remainder_word),0n,
    Equal.sym(Nat,U32.to_nat(remainder_word),Nat.mod(start,stride),remainder_exact),word_zero(remainder_word,c3))
  +first_exact = {%Addition.zero_right(Nat.mul(Nat.div(start,stride),stride)) : {start == _ : Nat}
    %remainder_zero : {start == Nat.add(Nat.mul(Nat.div(start,stride),stride),_) : Nat}
    DivisionOrder.quotient_exact(start,stride,stride_positive) : {start == Nat.mul(Nat.div(start,stride),stride) : Nat}}
  +top = top_cover(limits,problem,valid,supported,offset,span,first,width_blocks,width,range,first_le,
    or_cases(U32.is_zero(first),U32.is_le((first + padding_word : U32),(offset / width_word * stride_word : U32)),c2))
  +bottom = bottom_cover(limits,problem,valid,supported,offset,span,first,rows,width_blocks,width,range,span_positive,fits_height,
    or_cases(U32.is_eq((first + rows : U32),height_word),
      U32.is_le(((offset + span - 1) / width_word * stride_word + kernel_word : U32),((first + rows : U32) + padding_word : U32)),c1))
  +pad_exact = SupportedCoordinateBounds.pad_exact(limits,problem,support)
  +kernel_exact = SupportedCoordinateBounds.kernel_exact(limits,problem,support)
  +rows_bound = padded_rows_bound(limits,problem,valid,support)
  +doubled = Nat.add(padding,padding)
  +doubled_exact = {%Equal.sym(Nat,Nat.add(padding,0n),padding,Addition.zero_right(padding)) : {doubled == Nat.add(padding,_) : Nat}
    {==} : {doubled == Nat.mul(2n,padding) : Nat}}
  +count_le = Order.transitive(count,Nat.add(start,count),height,Samples.le_after(start,count),fits_height)
  +outer_room = {%Equal.sym(Nat,Nat.add(Nat.add(count,padding),padding),Nat.add(count,doubled),Addition.associative(count,padding,padding)) :
      {Nat.is_le(_,U32.to_nat(coordinate)) == True{} : Bool}
    Order.transitive(Nat.add(count,doubled),Nat.add(height,Nat.mul(2n,padding)),U32.to_nat(coordinate),
      %doubled_exact : {Nat.is_le(Nat.add(count,doubled),Nat.add(height,_)) == True{} : Bool}
      Order.add_le(count,height,doubled,count_le),rows_bound) :
    {Nat.is_le(Nat.add(Nat.add(count,padding),padding),U32.to_nat(coordinate)) == True{} : Bool}}
  +inner_room = Order.transitive(Nat.add(count,padding),Nat.add(Nat.add(count,padding),padding),U32.to_nat(coordinate),
    Order.le_add(Nat.add(count,padding),padding),outer_room)
  +inner_word = (rows + padding_word : U32)
  +inner_exact = {%pad_exact : {U32.to_nat(inner_word) == Nat.add(count,_) : Nat}
    Arithmetic.add(rows,padding_word,coordinate,Limits.coordinate_fits(limits),
      %Equal.sym(Nat,U32.to_nat(padding_word),padding,pad_exact) : {Nat.is_le(Nat.add(count,_),U32.to_nat(coordinate)) == True{} : Bool}
      inner_room) : {U32.to_nat(inner_word) == Nat.add(count,padding) : Nat}}
  +outer_word = (inner_word + padding_word : U32)
  +outer_exact = {%pad_exact : {U32.to_nat(outer_word) == Nat.add(Nat.add(count,padding),_) : Nat}
    %inner_exact : {U32.to_nat(outer_word) == Nat.add(_,U32.to_nat(padding_word)) : Nat}
    Arithmetic.add(inner_word,padding_word,coordinate,Limits.coordinate_fits(limits),
      %Equal.sym(Nat,U32.to_nat(inner_word),Nat.add(count,padding),inner_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(padding_word)),U32.to_nat(coordinate)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(padding_word),padding,pad_exact) : {Nat.is_le(Nat.add(Nat.add(count,padding),_),U32.to_nat(coordinate)) == True{} : Bool}
      outer_room) : {U32.to_nat(outer_word) == Nat.add(Nat.add(count,padding),padding) : Nat}}
  +kernel_fits = {%doubled_exact : {Nat.is_le(kernel,Nat.add(count,_)) == True{} : Bool}
    %Addition.associative(count,padding,padding) : {Nat.is_le(kernel,_) == True{} : Bool}
    %kernel_exact : {Nat.is_le(_,Nat.add(Nat.add(count,padding),padding)) == True{} : Bool}
    %outer_exact : {Nat.is_le(U32.to_nat(kernel_word),_) == True{} : Bool}
    Bounds.as_nat_le(kernel_word,outer_word,c0) : {Nat.is_le(kernel,Nat.add(count,Nat.mul(2n,padding))) == True{} : Bool}}
  CopyFacts{Positions.Cover{Nat.div(start,stride),first_exact,fits_height,rows_positive,kernel_fits,top,bottom},after,within}
