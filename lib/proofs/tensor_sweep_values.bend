# One sweep over a logical input: the output holds, at every index inside the
# written count, the ordered fold of the samples of its line over the logical
# input, starting from pad; a sample outside the axis is pad. A valid sample
# reads an address inside the input count, so the fold over the storage tree
# is the same fold over the logical reader. No exchange, reassociation or law
# on the operation is used. The checked sweep supplies the bounds from its
# own guard; a caller of the unchecked sweep supplies them itself.
import Base
import ../tensor_access.bend as Access
import ../tensor_operations.bend as Operations
import ../tensor_shape.bend as Shapes
import ../storage_buffer.bend as Storage
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_certificate.bend as Certificate
import ./storage_shape.bend as Shape
import ./tensor_dense_contract.bend as Dense
import ./tensor_dense_normal.bend as DenseNormal
import ./tensor_gather_values.bend as GatherValues
import ./tensor_copy_values.bend as CopyValues
import ./tensor_sweep_refinement.bend as SweepLaws
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_access_refinement.bend as AccessLaws
import ./tensor_access_certificate.bend as AccessCertificate
import ./traversal_indexed_values.bend as IndexedValues
import ./ordered_fold_refinement.bend as Fold
import ./tensor_shape_refinement.bend as ShapeLaws
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_division.bend as U32Division
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./nat_division_order.bend as DivisionOrder
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./tensor_band_coordinates.bend as BandCoordinates
import ./machine_limit_bounds.bend as Limits
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges

def product_exact(+left: U32,+right: U32,+a: Nat,+b: Nat,+capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +left_exact: {U32.to_nat(left) == a : Nat},+right_exact: {U32.to_nat(right) == b : Nat},
  room: {Nat.is_le(Nat.mul(a,b),U32.to_nat(capacity)) == True{} : Bool}) -> {U32.to_nat((left * right : U32)) == Nat.mul(a,b) : Nat}:
  %left_exact : {U32.to_nat((left * right : U32)) == Nat.mul(_,b) : Nat}
  %right_exact : {U32.to_nat((left * right : U32)) == Nat.mul(U32.to_nat(left),_) : Nat}
  Arithmetic.multiply(left,right,capacity,capacity_ok,
    %Equal.sym(Nat,U32.to_nat(left),a,left_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(right)),U32.to_nat(capacity)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(right),b,right_exact) : {Nat.is_le(Nat.mul(a,_),U32.to_nat(capacity)) == True{} : Bool}
    room)

def sum_exact(+left: U32,+right: U32,+a: Nat,+b: Nat,+capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +left_exact: {U32.to_nat(left) == a : Nat},+right_exact: {U32.to_nat(right) == b : Nat},
  room: {Nat.is_le(Nat.add(a,b),U32.to_nat(capacity)) == True{} : Bool}) -> {U32.to_nat((left + right : U32)) == Nat.add(a,b) : Nat}:
  %left_exact : {U32.to_nat((left + right : U32)) == Nat.add(_,b) : Nat}
  %right_exact : {U32.to_nat((left + right : U32)) == Nat.add(U32.to_nat(left),_) : Nat}
  Arithmetic.add(left,right,capacity,capacity_ok,
    %Equal.sym(Nat,U32.to_nat(left),a,left_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(right)),U32.to_nat(capacity)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(right),b,right_exact) : {Nat.is_le(Nat.add(a,_),U32.to_nat(capacity)) == True{} : Bool}
    room)

def swap_last(+a: Nat,+b: Nat,+c: Nat) -> {Nat.add(Nat.add(a,b),c) == Nat.add(Nat.add(a,c),b) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(a,b),c),Nat.add(a,Nat.add(b,c)),Nat.add(Nat.add(a,c),b),Addition.associative(a,b,c),
    Equal.trans(Nat,Nat.add(a,Nat.add(b,c)),Nat.add(a,Nat.add(c,b)),Nat.add(Nat.add(a,c),b),
      Equal.cong(Nat,Nat,tail => Nat.add(a,tail),Nat.add(b,c),Nat.add(c,b),Order.commutative(b,c)),
      Equal.sym(Nat,Nat.add(Nat.add(a,c),b),Nat.add(a,Nat.add(c,b)),Addition.associative(a,c,b))))

# The address of a sample of a line. An output inside outer * out_extent * step
# lies in a block below outer and keeps a step offset below step; with a
# coordinate inside the axis its address, computed in native words, is inside
# outer * extent * step, and no partial result wraps.
def line_address(+position: U32,+extent: U32,+step: U32,+out_extent: U32,+coordinate: U32,+outer: Nat,+limits: Machine.Limits,
  +input_room: {Nat.is_le(Nat.mul(outer,Nat.mul(U32.to_nat(extent),U32.to_nat(step))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(outer,Nat.mul(U32.to_nat(out_extent),U32.to_nat(step))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(position),Nat.mul(outer,Nat.mul(U32.to_nat(out_extent),U32.to_nat(step)))) == True{} : Bool},
  +valid: {Nat.is_lt(U32.to_nat(coordinate),U32.to_nat(extent)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(((((position / step : U32) / out_extent : U32) * extent * step + (position % step : U32) : U32) + coordinate * step : U32)),
    Nat.mul(outer,Nat.mul(U32.to_nat(extent),U32.to_nat(step)))) == True{} : Bool}:
  +p = U32.to_nat(position)
  +s = U32.to_nat(step)
  +e = U32.to_nat(extent)
  +oe = U32.to_nat(out_extent)
  +c = U32.to_nat(coordinate)
  +capacity = Machine.capacity(limits)
  +capacity_ok = Limits.capacity_fits(limits)
  +half = Machine.half_word(limits)
  +half_ok = Limits.half_word_fits(limits)
  +limit = U32.to_nat(capacity)
  +line = Nat.mul(oe,s)
  +total_out = Nat.mul(outer,line)
  +total_in = Nat.mul(outer,Nat.mul(e,s))
  +total_positive = Order.le_lt_trans(0n,p,total_out,Order.zero_le(p),inside)
  +outer_positive = RowMajor.left_factor_positive(outer,line,total_positive)
  +line_positive = RowMajor.left_factor_positive(line,outer,
    %RowMajor.multiply_commutative(outer,line) : {Nat.is_lt(0n,_) == True{} : Bool}
    total_positive)
  +out_positive = RowMajor.left_factor_positive(oe,s,line_positive)
  +step_positive = RowMajor.left_factor_positive(s,oe,
    %RowMajor.multiply_commutative(oe,s) : {Nat.is_lt(0n,_) == True{} : Bool}
    line_positive)
  +line_room = Order.transitive(line,total_out,limit,
    %RowMajor.multiply_commutative(line,outer) : {Nat.is_le(line,_) == True{} : Bool}
    RowMajor.product_covers(line,outer,outer_positive),output_room)
  +out_room = Order.transitive(oe,line,limit,RowMajor.product_covers(oe,s,step_positive),line_room)
  +step_room = Order.transitive(s,line,limit,
    %RowMajor.multiply_commutative(s,oe) : {Nat.is_le(s,_) == True{} : Bool}
    RowMajor.product_covers(s,oe,out_positive),line_room)
  +along = (position / step : U32)
  +plane = (along / out_extent : U32)
  +offset = (position % step : U32)
  +a = Nat.div(p,s)
  +q = Nat.div(a,oe)
  +r = Nat.mod(p,s)
  +along_exact = U32Division.divide(position,step,half,half_ok,Limits.double_bound(limits,s,step_room),step_positive)
  +offset_exact = U32Division.modulo(position,step,half,half_ok,Limits.double_bound(limits,s,step_room),step_positive)
  +plane_exact = {
    %along_exact : {U32.to_nat(plane) == Nat.div(_,oe) : Nat}
    U32Division.divide(along,out_extent,half,half_ok,Limits.double_bound(limits,oe,out_room),out_positive) : {U32.to_nat(plane) == q : Nat}}
  +along_below = RowMajor.cancel_multiply_lt(a,Nat.mul(outer,oe),s,
    Order.le_lt_trans(Nat.mul(a,s),p,Nat.mul(Nat.mul(outer,oe),s),DivisionOrder.product_below(p,s,step_positive),
      %Equal.sym(Nat,Nat.mul(Nat.mul(outer,oe),s),total_out,RowMajor.assoc(outer,oe,s)) : {Nat.is_lt(p,_) == True{} : Bool}
      inside))
  +plane_below = RowMajor.cancel_multiply_lt(q,outer,oe,
    Order.le_lt_trans(Nat.mul(q,oe),a,Nat.mul(outer,oe),DivisionOrder.product_below(a,oe,out_positive),along_below))
  +offset_below = DivisionOrder.remainder_small(p,s,step_positive)
  +cells = Nat.mul(Nat.mul(q,e),s)
  +taps = Nat.mul(c,s)
  +address = RowMajor.index(q,e,s,c,r)
  +bounded = RowMajor.in_bounds(q,outer,e,s,c,r,plane_below,valid,offset_below)
  +reordered = Equal.trans(Nat,Nat.add(Nat.add(cells,r),taps),Nat.add(Nat.add(cells,taps),r),address,swap_last(cells,r,taps),RowMajor.expanded(q,e,s,c,r))
  +address_room = Bounds.strict_le(address,limit,Bounds.nat_trans(address,total_in,limit,bounded,input_room))
  +sum_room = {
    %Equal.sym(Nat,Nat.add(Nat.add(cells,r),taps),address,reordered) : {Nat.is_le(_,limit) == True{} : Bool}
    address_room : {Nat.is_le(Nat.add(Nat.add(cells,r),taps),limit) == True{} : Bool}}
  +base_room = Order.transitive(Nat.add(cells,r),Nat.add(Nat.add(cells,r),taps),limit,Order.le_add(Nat.add(cells,r),taps),sum_room)
  +cells_room = Order.transitive(cells,Nat.add(cells,r),limit,Order.le_add(cells,r),base_room)
  +taps_room = Order.transitive(taps,Nat.add(Nat.add(cells,r),taps),limit,
    %Order.commutative(taps,Nat.add(cells,r)) : {Nat.is_le(taps,_) == True{} : Bool}
    Order.le_add(taps,Nat.add(cells,r)),sum_room)
  +plane_rows = product_exact(plane,extent,q,e,capacity,capacity_ok,plane_exact,{==},RowMajor.row_product(q,outer,e,s,limit,plane_below,step_positive,input_room))
  +cells_exact = product_exact((plane * extent : U32),step,Nat.mul(q,e),s,capacity,capacity_ok,plane_rows,{==},cells_room)
  +base_exact = sum_exact((plane * extent * step : U32),offset,cells,r,capacity,capacity_ok,cells_exact,offset_exact,base_room)
  +taps_exact = product_exact(coordinate,step,c,s,capacity,capacity_ok,{==},{==},taps_room)
  +exact = sum_exact((plane * extent * step + offset : U32),(coordinate * step : U32),Nat.add(cells,r),taps,capacity,capacity_ok,base_exact,taps_exact,sum_room)
  %Equal.sym(Nat,U32.to_nat((plane * extent * step + offset + coordinate * step : U32)),address,
    Equal.trans(Nat,U32.to_nat((plane * extent * step + offset + coordinate * step : U32)),Nat.add(Nat.add(cells,r),taps),address,exact,reordered)) :
    {Nat.is_lt(_,total_in) == True{} : Bool}
  bounded

def input_cells(sweep: Access.Sweep,outer: Nat) -> Nat:
  Access.Sweep{extent,step,kernel,stride,padding,out_extent,origin} = sweep
  Nat.mul(outer,Nat.mul(U32.to_nat(extent),U32.to_nat(step)))

def output_cells(sweep: Access.Sweep,outer: Nat) -> Nat:
  Access.Sweep{extent,step,kernel,stride,padding,out_extent,origin} = sweep
  Nat.mul(outer,Nat.mul(U32.to_nat(out_extent),U32.to_nat(step)))

# What every sample of a sweep needs: both counts fit the machine capacity
# and the input holds its count.
type Guards<-sweep: Access.Sweep,-outer: Nat,-length: Nat,-limits: Machine.Limits> is Data:
  Guards{input_room: {Nat.is_le(input_cells(sweep,outer),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    output_room: {Nat.is_le(output_cells(sweep,outer),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    input_length: {Nat.is_le(input_cells(sweep,outer),length) == True{} : Bool}}

# The logical input read through the lines of a sweep.
type Swept<-Element: Data,-Logical: Data> is Data:
  Swept{logical: Logical,sweep: Access.Sweep,pad: Element}

def logical_when(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+pad: Element,valid: Bool,logical: Logical,index: U32) -> Element:
  match valid:
    case False{}: pad
    case True{}: original(logical,index)

def logical_line(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,logical: Logical,line: Access.Line<Element>,position: U32) -> Element:
  Access.Line{+base,+first,+extent,+step,+padding,pad} = line
  +coordinate = (first + position - padding : U32)
  logical_when(~Element,~Logical,~original,pad,U32.is_lt(coordinate,extent),logical,(base + coordinate * step : U32))

def value(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,context: Swept<Element,Logical>,position: U32) -> Element:
  Swept{+logical,+sweep,+pad} = context
  Access.Sweep{extent,step,kernel,stride,padding,out_extent,origin} = sweep
  Fold.values(~Logical,~Element,~Element,~Access.Line<Element>,~logical_line(~Element,~Logical,~original),~operation,U32.to_nat(kernel),0,logical,Access.line_at(~Element,sweep,pad,position),pad)

def tap_valid(~Element: Data,line: Access.Line<Element>,position: U32) -> Bool:
  Access.Line{base,first,extent,step,padding,pad} = line
  U32.is_lt((first + position - padding : U32),extent)

def tap_index(~Element: Data,line: Access.Line<Element>,position: U32) -> U32:
  Access.Line{base,first,extent,step,padding,pad} = line
  (base + (first + position - padding : U32) * step : U32)

# A valid sample of an output inside the written count reads inside the input.
def tap_inside(~Element: Data,+sweep: Access.Sweep,+pad: Element,+outer: Nat,+position: U32,+tap: U32,+length: Nat,+limits: Machine.Limits,
  guards: Guards<sweep,outer,length,limits>,inside: {Nat.is_lt(U32.to_nat(position),output_cells(sweep,outer)) == True{} : Bool},
  +valid: {tap_valid(~Element,Access.line_at(~Element,sweep,pad,position),tap) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(tap_index(~Element,Access.line_at(~Element,sweep,pad,position),tap)),length) == True{} : Bool}:
  Access.Sweep{+extent,+step,+kernel,+stride,+padding,+out_extent,+origin} = sweep
  Guards{input_room,output_room,input_length} = guards
  +along = (position / step : U32)
  +coordinate = (along % out_extent * stride + origin + tap - padding : U32)
  Bounds.nat_trans(U32.to_nat((along / out_extent * extent * step + position % step + coordinate * step : U32)),
    Nat.mul(outer,Nat.mul(U32.to_nat(extent),U32.to_nat(step))),length,
    line_address(position,extent,step,out_extent,coordinate,outer,limits,input_room,output_room,inside,Bounds.as_nat_lt(coordinate,extent,valid)),
    input_length)

def tap_located(~Element: Data,+sweep: Access.Sweep,+pad: Element,+outer: Nat,+position: U32,+tap: U32,+length: Nat,+limits: Machine.Limits,valid: Bool,
  guards: Guards<sweep,outer,length,limits>,inside: {Nat.is_lt(U32.to_nat(position),output_cells(sweep,outer)) == True{} : Bool},
  +exact: {tap_valid(~Element,Access.line_at(~Element,sweep,pad,position),tap) == valid : Bool}) ->
  Boolean.When(valid,{Nat.is_lt(U32.to_nat(tap_index(~Element,Access.line_at(~Element,sweep,pad,position),tap)),length) == True{} : Bool}):
  match valid:
    case False{}: Unit{}
    case True{}: tap_inside(~Element,sweep,pad,outer,position,tap,length,limits,guards,inside,exact)

# A valid sample inside the input holds the logical value of its address;
# an invalid sample is the same pad on both sides.
def sample(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+input: Tree.Tree<Element>,+length: Nat,+pad: Element,valid: Bool,+index: U32,
  values: Dense.Values(~Element,~Logical,~original,logical,Buffer.Content{input,length}),
  inside: Boolean.When(valid,{Nat.is_lt(U32.to_nat(index),length) == True{} : Bool})) ->
  {SweepLaws.sample_when(~Element,pad,valid,input,index) == logical_when(~Element,~Logical,~original,pad,valid,logical,index) : Element}:
  match valid:
    case False{}: {==}
    case True{}:
      %Arithmetic.native_roundtrip(index) : {Tree.read(Element,input,_) == original(logical,_) : Element}
      Dense.read(~Element,~Logical,~original,logical,Buffer.Content{input,length},U32.to_nat(index),values,inside)

def line_sample(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+input: Tree.Tree<Element>,+length: Nat,
  +sweep: Access.Sweep,+pad: Element,+outer: Nat,+position: U32,+tap: U32,+limits: Machine.Limits,
  values: Dense.Values(~Element,~Logical,~original,logical,Buffer.Content{input,length}),
  guards: Guards<sweep,outer,length,limits>,inside: {Nat.is_lt(U32.to_nat(position),output_cells(sweep,outer)) == True{} : Bool}) ->
  {SweepLaws.sample_line(~Element,input,Access.line_at(~Element,sweep,pad,position),tap) == logical_line(~Element,~Logical,~original,logical,Access.line_at(~Element,sweep,pad,position),tap) : Element}:
  Access.Sweep{+extent,+step,+kernel,+stride,+padding,+out_extent,+origin} = sweep
  +sweep = {Access.Sweep{extent,step,kernel,stride,padding,out_extent,origin} : Access.Sweep}
  +at = Access.line_at(~Element,sweep,pad,position)
  sample(~Element,~Logical,~original,logical,input,length,pad,tap_valid(~Element,at,tap),tap_index(~Element,at,tap),values,
    tap_located(~Element,sweep,pad,outer,position,tap,length,limits,tap_valid(~Element,at,tap),guards,inside,{==}))

def taps(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,remaining: Nat,+tap: U32,
  +logical: Logical,+input: Tree.Tree<Element>,+length: Nat,+sweep: Access.Sweep,+pad: Element,+outer: Nat,+position: U32,+limits: Machine.Limits,+initial: Element,
  +values: Dense.Values(~Element,~Logical,~original,logical,Buffer.Content{input,length}),
  +guards: Guards<sweep,outer,length,limits>,+inside: {Nat.is_lt(U32.to_nat(position),output_cells(sweep,outer)) == True{} : Bool}) ->
  {SweepLaws.line_values(~Element,~operation,remaining,tap,input,Access.line_at(~Element,sweep,pad,position),initial) ==
    Fold.values(~Logical,~Element,~Element,~Access.Line<Element>,~logical_line(~Element,~Logical,~original),~operation,remaining,tap,logical,Access.line_at(~Element,sweep,pad,position),initial) : Element}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +at = Access.line_at(~Element,sweep,pad,position)
      %Equal.sym(Element,SweepLaws.sample_line(~Element,input,at,tap),logical_line(~Element,~Logical,~original,logical,at,tap),
        line_sample(~Element,~Logical,~original,logical,input,length,sweep,pad,outer,position,tap,limits,values,guards,inside)) :
        {SweepLaws.line_values(~Element,~operation,rest,(tap + 1 : U32),input,at,operation(initial,_)) ==
          Fold.values(~Logical,~Element,~Element,~Access.Line<Element>,~logical_line(~Element,~Logical,~original),~operation,1n+rest,tap,logical,at,initial) : Element}
      taps(~Element,~Logical,~original,~operation,rest,(tap + 1 : U32),logical,input,length,sweep,pad,outer,position,limits,
        operation(initial,logical_line(~Element,~Logical,~original,logical,at,tap)),values,guards,inside)

# The ordered fold of the storage tree is the fold of the logical input at
# every output inside the written count.
def swept(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,
  +logical: Logical,+input: Tree.Tree<Element>,+length: Nat,+sweep: Access.Sweep,+pad: Element,+outer: Nat,+position: U32,+limits: Machine.Limits,
  +values: Dense.Values(~Element,~Logical,~original,logical,Buffer.Content{input,length}),
  +guards: Guards<sweep,outer,length,limits>,+inside: {Nat.is_lt(U32.to_nat(position),output_cells(sweep,outer)) == True{} : Bool}) ->
  {SweepLaws.sample_swept(~Element,~operation,input,Access.SweepContext{sweep,pad},position) ==
    value(~Element,~Logical,~original,~operation,Swept{logical,sweep,pad},position) : Element}:
  Access.Sweep{+extent,+step,+kernel,+stride,+padding,+out_extent,+origin} = sweep
  taps(~Element,~Logical,~original,~operation,U32.to_nat(kernel),0,logical,input,length,Access.Sweep{extent,step,kernel,stride,padding,out_extent,origin},pad,outer,position,limits,pad,values,guards,inside)

def identity(~Element: Data,+context: Access.SweepContext<Element>,+position: U32) -> {Access.swept_index(~Element,context,position) == position : U32}:
  {==}

# The output tree of the sweep model: an indexed store of the swept samples.
def run(~Element: Data,~operation: Element -> Element -> Element,count: Nat,context: Access.SweepContext<Element>,input: Tree.Tree<Element>,output: Tree.Tree<Element>) -> Tree.Tree<Element>:
  IndexedValues.run(~Tree.Tree<Element>,~Element,~Access.SweepContext<Element>,~SweepLaws.sample_swept(~Element,~operation),~Access.swept_index(~Element),count,0,context,input,output)

def model(~Element: Data,~operation: Element -> Element -> Element,count: Nat,context: Access.SweepContext<Element>,input: Buffer.Content<Element>,output: Buffer.Content<Element>) -> Buffer.Content<Element>:
  Buffer.Content{input,input_length} = input
  Buffer.Content{output,output_length} = output
  Buffer.Content{run(~Element,~operation,count,context,input,output),output_length}

# The run stores the fold of the logical input at every position it writes.
def cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,
  +logical: Logical,+sweep: Access.Sweep,+pad: Element,+outer: Nat,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+limits: Machine.Limits,+position: Nat,
  ready: Buffer.Sized<Element,output>,values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),guards: Guards<sweep,outer,Buffer.length(~Element,input),limits>,
  +cells: {output_cells(sweep,outer) == Buffer.length(~Element,output) : Nat},
  +selected: {Nat.is_lt(position,Buffer.length(~Element,output)) == True{} : Bool}) ->
  Dense.At(~Element,~Swept<Element,Logical>,~value(~Element,~Logical,~original,~operation),Swept{logical,sweep,pad},model(~Element,~operation,Buffer.length(~Element,output),Access.SweepContext{sweep,pad},Buffer.content(~Element,input),Buffer.content(~Element,output)),position):
  Buffer.Model{+input_tree,+input_length,input_certificate} = input
  Buffer.Model{+output_tree,+length,+certificate} = output
  Buffer.Sized{+depth_ok,room} = ready
  Guards{+input_room,+output_room,+input_length_room} = guards
  +output = {Buffer.Model{output_tree,length,certificate} : Buffer.Model<Element>}
  +context = {Access.SweepContext{sweep,pad} : Access.SweepContext<Element>}
  +limit = Machine.capacity(limits)
  +inside = {
    %Equal.sym(Nat,output_cells(sweep,outer),length,cells) : {Nat.is_lt(position,_) == True{} : Bool}
    selected : {Nat.is_lt(position,output_cells(sweep,outer)) == True{} : Bool}}
  +encoded = Bounds.roundtrip(position,limit,Limits.capacity_fits(limits),
    Bounds.strict_le(position,U32.to_nat(limit),Bounds.nat_trans(position,output_cells(sweep,outer),U32.to_nat(limit),inside,output_room)))
  Equal.trans(Element,Tree.read(Element,run(~Element,~operation,length,context,input_tree,output_tree),U32.from_nat(position)),
    SweepLaws.sample_swept(~Element,~operation,input_tree,context,U32.from_nat(position)),
    value(~Element,~Logical,~original,~operation,Swept{logical,sweep,pad},U32.from_nat(position)),
    IndexedValues.stored(~Tree.Tree<Element>,~Element,~Access.SweepContext<Element>,~SweepLaws.sample_swept(~Element,~operation),~Access.swept_index(~Element),~identity(~Element),
      length,context,input_tree,output_tree,Buffer.depth(~Element,output),Tree.capacity(Element,output_tree),position,
      {==},Buffer.balanced(~Element,output),depth_ok,Buffer.capacity_bound(~Element,output,depth_ok),room,selected),
    swept(~Element,~Logical,~original,~operation,logical,input_tree,input_length,sweep,pad,outer,U32.from_nat(position),limits,values,
      Guards{input_room,output_room,input_length_room},
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,encoded) : {Nat.is_lt(_,output_cells(sweep,outer)) == True{} : Bool}
      inside))

def cells_range(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,
  remaining: Nat,+position: Nat,+logical: Logical,+sweep: Access.Sweep,+pad: Element,+outer: Nat,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+limits: Machine.Limits,
  +ready: Buffer.Sized<Element,output>,+values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),+guards: Guards<sweep,outer,Buffer.length(~Element,input),limits>,
  +cells: {output_cells(sweep,outer) == Buffer.length(~Element,output) : Nat},
  +room: {Nat.is_le(Nat.add(position,remaining),Buffer.length(~Element,output)) == True{} : Bool}) ->
  Members.All(Nat,{index => Dense.At(~Element,~Swept<Element,Logical>,~value(~Element,~Logical,~original,~operation),Swept{logical,sweep,pad},model(~Element,~operation,Buffer.length(~Element,output),Access.SweepContext{sweep,pad},Buffer.content(~Element,input),Buffer.content(~Element,output)),index) : Nat -> Data},Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{cell(~Element,~Logical,~original,~operation,logical,sweep,pad,outer,input,output,limits,position,ready,values,guards,cells,
          Order.room_index(position,rest,Buffer.length(~Element,output),room)),
        cells_range(~Element,~Logical,~original,~operation,rest,1n+position,logical,sweep,pad,outer,input,output,limits,ready,values,guards,cells,
          Order.advance_room(position,rest,Buffer.length(~Element,output),room))}

def model_length(~Element: Data,~operation: Element -> Element -> Element,+count: Nat,+context: Access.SweepContext<Element>,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>) ->
  {Buffer.content_length(~Element,model(~Element,~operation,count,context,Buffer.content(~Element,input),Buffer.content(~Element,output))) ==
    Buffer.length(~Element,output) : Nat}:
  Buffer.Model{input_tree,input_length,input_certificate} = input
  Buffer.Model{output_tree,output_length,output_certificate} = output
  {==}

def model_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,
  +logical: Logical,+sweep: Access.Sweep,+pad: Element,+outer: Nat,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+limits: Machine.Limits,
  +ready: Buffer.Sized<Element,output>,+values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),+guards: Guards<sweep,outer,Buffer.length(~Element,input),limits>,
  +cells: {output_cells(sweep,outer) == Buffer.length(~Element,output) : Nat}) ->
  Dense.Values(~Element,~Swept<Element,Logical>,~value(~Element,~Logical,~original,~operation),Swept{logical,sweep,pad},model(~Element,~operation,Buffer.length(~Element,output),Access.SweepContext{sweep,pad},Buffer.content(~Element,input),Buffer.content(~Element,output))):
  %Equal.sym(Nat,Buffer.content_length(~Element,model(~Element,~operation,Buffer.length(~Element,output),Access.SweepContext{sweep,pad},Buffer.content(~Element,input),Buffer.content(~Element,output))),Buffer.length(~Element,output),
    model_length(~Element,~operation,Buffer.length(~Element,output),Access.SweepContext{sweep,pad},input,output)) :
    Members.All(Nat,{index => Dense.At(~Element,~Swept<Element,Logical>,~value(~Element,~Logical,~original,~operation),Swept{logical,sweep,pad},model(~Element,~operation,Buffer.length(~Element,output),Access.SweepContext{sweep,pad},Buffer.content(~Element,input),Buffer.content(~Element,output)),index) : Nat -> Data},Ranges.range(_,0n))
  cells_range(~Element,~Logical,~original,~operation,Buffer.length(~Element,output),0n,logical,sweep,pad,outer,input,output,limits,ready,values,guards,cells,Order.reflexive(Buffer.length(~Element,output)))

def model_ready(~Element: Data,~operation: Element -> Element -> Element,+count: Nat,+context: Access.SweepContext<Element>,input: Buffer.Content<Element>,output: Buffer.Content<Element>,
  evidence: Buffer.ContentReady<Element,output>) -> Buffer.ContentReady<Element,model(~Element,~operation,count,context,input,output)>:
  match input output:
    case Buffer.Content{+input,input_length} Buffer.Content{+output,+output_length}:
      IndexedValues.ready(~Tree.Tree<Element>,~Element,~Access.SweepContext<Element>,~SweepLaws.sample_swept(~Element,~operation),~Access.swept_index(~Element),
        count,context,input,output,output_length,evidence)

# The storage content of an output written whole by one sweep: well formed
# for its shape, with the fold of the logical input at every index.
def model_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,
  +logical: Logical,+sweep: Access.Sweep,+pad: Element,+outer: Nat,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+shape: List<&2,U32>,+limits: Machine.Limits,
  +ready: Buffer.Sized<Element,output>,+values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),+guards: Guards<sweep,outer,Buffer.length(~Element,input),limits>,
  +cells: {output_cells(sweep,outer) == Buffer.length(~Element,output) : Nat},
  length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(shape) : Nat}) ->
  GatherValues.Facts(~Element,~Swept<Element,Logical>,~value(~Element,~Logical,~original,~operation),Swept{logical,sweep,pad},shape,model(~Element,~operation,Buffer.length(~Element,output),Access.SweepContext{sweep,pad},Buffer.content(~Element,input),Buffer.content(~Element,output))):
  Shape.Both{
    Dense.WellFormed{
      Equal.trans(Nat,Buffer.content_length(~Element,model(~Element,~operation,Buffer.length(~Element,output),Access.SweepContext{sweep,pad},Buffer.content(~Element,input),Buffer.content(~Element,output))),Buffer.length(~Element,output),ShapeLaws.elements(shape),
        model_length(~Element,~operation,Buffer.length(~Element,output),Access.SweepContext{sweep,pad},input,output),length_exact),
      model_ready(~Element,~operation,Buffer.length(~Element,output),Access.SweepContext{sweep,pad},Buffer.content(~Element,input),Buffer.content(~Element,output),
        Buffer.content_ready(~Element,output,ready))},
    model_values(~Element,~Logical,~original,~operation,logical,sweep,pad,outer,input,output,limits,ready,values,guards,cells)}

# Checked products of two and three extents are their natural products.
def two(+first: U32,+second: U32,+count: U32,
  accepted: {Shapes.product([first,second],Some{1}) == Some{count} : Maybe<&2,U32>}) ->
  {Nat.mul(U32.to_nat(first),U32.to_nat(second)) == U32.to_nat(count) : Nat}:
  %ShapeLaws.times_one(U32.to_nat(second)) : {Nat.mul(U32.to_nat(first),_) == U32.to_nat(count) : Nat}
  %accepted : ShapeLaws.Count(_,ShapeLaws.elements([first,second]))
  ShapeLaws.nonempty_count([first,second])

def three(+planes: U32,+height: U32,+width: U32,+count: U32,
  accepted: {Shapes.product([planes,height,width],Some{1}) == Some{count} : Maybe<&2,U32>}) ->
  {Nat.mul(U32.to_nat(planes),Nat.mul(U32.to_nat(height),U32.to_nat(width))) == U32.to_nat(count) : Nat}:
  %ShapeLaws.times_one(U32.to_nat(width)) : {Nat.mul(U32.to_nat(planes),Nat.mul(U32.to_nat(height),_)) == U32.to_nat(count) : Nat}
  %accepted : ShapeLaws.Count(_,ShapeLaws.elements([planes,height,width]))
  ShapeLaws.nonempty_count([planes,height,width])

# What an accepted capacity guard says about a checked count.
type Held<-source: Maybe<&2,U32>,-length: Nat> is Data:
  Held{count: U32,exact: {source == Some{count} : Maybe<&2,U32>},within: {Nat.is_le(U32.to_nat(count),length) == True{} : Bool}}

def held(result: Maybe<&2,U32>,-source: Maybe<&2,U32>,+length: Nat,+capacity: U32,exact: {source == result : Maybe<&2,U32>},
  accepted: {Operations.holds(result,length,capacity) == True{} : Bool}) -> Held<source,length>:
  match result:
    case None{}: Empty.absurd(Held<source,length>,WordEncoding.false_true(accepted))
    case Some{+count}: Held{count,exact,Boolean.left(Nat.is_le(U32.to_nat(count),length),Nat.is_le(length,U32.to_nat(capacity)),accepted)}

def input_within(+sweep: Access.Sweep,+outer: U32,+length: Nat,held: Held<Operations.sweep_input(sweep,outer),length>) ->
  {Nat.is_le(input_cells(sweep,U32.to_nat(outer)),length) == True{} : Bool}:
  Access.Sweep{+extent,+step,kernel,stride,padding,out_extent,origin} = sweep
  Held{+count,exact,within} = held
  %Equal.sym(Nat,Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(step))),U32.to_nat(count),three(outer,extent,step,count,exact)) :
    {Nat.is_le(_,length) == True{} : Bool}
  within

def output_written(+sweep: Access.Sweep,+outer: U32,+written: U32,exact: {Operations.sweep_output(sweep,outer) == Some{written} : Maybe<&2,U32>}) ->
  {output_cells(sweep,U32.to_nat(outer)) == U32.to_nat(written) : Nat}:
  Access.Sweep{extent,+step,kernel,stride,padding,+out_extent,origin} = sweep
  three(outer,out_extent,step,written,exact)

def default_capacity(+limits: Machine.Limits,actual: {limits == Default.limits() : Machine.Limits}) -> {Machine.capacity(limits) == 1073741824 : U32}:
  %Equal.sym(Machine.Limits,limits,Default.limits(),actual) : {Machine.capacity(_) == 1073741824 : U32}
  {==}

# The guard of the checked sweep, over the models of its two owners.
def guard(~Element: Data,+sweep: Access.Sweep,+outer: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>) -> Bool:
  Operations.sweep_geometry(sweep) && Operations.holds(Operations.sweep_input(sweep,outer),Buffer.length(~Element,input),U32.shln(1,Buffer.depth(~Element,input))) &&
    Operations.holds(Operations.sweep_output(sweep,outer),Buffer.length(~Element,output),U32.shln(1,Buffer.depth(~Element,output)))

def guards_accepted(~Element: Data,+sweep: Access.Sweep,+outer: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+limits: Machine.Limits,
  +actual: {limits == Default.limits() : Machine.Limits},input_ready: Buffer.Sized<Element,input>,ready: Buffer.Sized<Element,output>,
  cells: {output_cells(sweep,U32.to_nat(outer)) == Buffer.length(~Element,output) : Nat},
  accepted: {guard(~Element,sweep,outer,input,output) == True{} : Bool}) -> Guards<sweep,U32.to_nat(outer),Buffer.length(~Element,input),limits>:
  +geometry = Operations.sweep_geometry(sweep)
  +input_guard = Operations.holds(Operations.sweep_input(sweep,outer),Buffer.length(~Element,input),U32.shln(1,Buffer.depth(~Element,input)))
  +output_guard = Operations.holds(Operations.sweep_output(sweep,outer),Buffer.length(~Element,output),U32.shln(1,Buffer.depth(~Element,output)))
  +limit = U32.to_nat(Machine.capacity(limits))
  +within = input_within(sweep,outer,Buffer.length(~Element,input),
    held(Operations.sweep_input(sweep,outer),Operations.sweep_input(sweep,outer),Buffer.length(~Element,input),U32.shln(1,Buffer.depth(~Element,input)),{==},
      Boolean.right(geometry,input_guard,Boolean.left((geometry && input_guard : Bool),output_guard,accepted))))
  Guards{
    Order.transitive(input_cells(sweep,U32.to_nat(outer)),Buffer.length(~Element,input),limit,within,
      %Buffer.content_length_of(~Element,input) : {Nat.is_le(_,limit) == True{} : Bool}
      DenseNormal.content_room(~Element,Buffer.content(~Element,input),Machine.capacity(limits),default_capacity(limits,actual),Buffer.content_ready(~Element,input,input_ready))),
    %Equal.sym(Nat,output_cells(sweep,U32.to_nat(outer)),Buffer.length(~Element,output),cells) : {Nat.is_le(_,limit) == True{} : Bool}
    %Buffer.content_length_of(~Element,output) : {Nat.is_le(_,limit) == True{} : Bool}
    DenseNormal.content_room(~Element,Buffer.content(~Element,output),Machine.capacity(limits),default_capacity(limits,actual),Buffer.content_ready(~Element,output,ready)),
    within}

def modeled_content(~Element: Data,+input_length: Nat,+output_length: Nat,+result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) ->
  {CopyValues.right_content(~Element,Operations.Accepted{OperationsLaws.modeled_pair(~Element,input_length,output_length,result)}) ==
    Buffer.Content{IndexedValues.output_tree(~Tree.Tree<Element>,~Element,result),output_length} : Buffer.Content<Element>}:
  AccessLaws.ModelResult{input,+output} = result
  Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,output_length},
    Tree.reflect(Element,Tree.pack(Element,output)),output,StorageLaws.reflect_packed(Element,output))

# The accepted branch of the checked model is the model of the whole output.
def accepted_model(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,
  +logical: Logical,+pad: Element,+sweep: Access.Sweep,+outer: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+shape: List<&2,U32>,+limits: Machine.Limits,+written: U32,
  +ready: Buffer.Sized<Element,output>,+values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),+guards: Guards<sweep,U32.to_nat(outer),Buffer.length(~Element,input),limits>,
  +cells: {output_cells(sweep,U32.to_nat(outer)) == Buffer.length(~Element,output) : Nat},
  length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(shape) : Nat},
  written_all: {U32.to_nat(written) == Buffer.length(~Element,output) : Nat}) ->
  GatherValues.Facts(~Element,~Swept<Element,Logical>,~value(~Element,~Logical,~original,~operation),Swept{logical,sweep,pad},shape,CopyValues.right_content(~Element,Operations.Accepted{OperationsLaws.modeled_pair(~Element,Buffer.length(~Element,input),Buffer.length(~Element,output),SweepLaws.sweep_model(~Element,~operation,U32.to_nat(written),Access.SweepContext{sweep,pad},Tree.reflect(Element,Tree.pack(Element,Buffer.tree(~Element,input))),Tree.reflect(Element,Tree.pack(Element,Buffer.tree(~Element,output)))))})):
  Buffer.Model{+input_tree,+input_length,+input_certificate} = input
  Buffer.Model{+output_tree,+output_length,+output_certificate} = output
  +input = {Buffer.Model{input_tree,input_length,input_certificate} : Buffer.Model<Element>}
  +output = {Buffer.Model{output_tree,output_length,output_certificate} : Buffer.Model<Element>}
  +context = {Access.SweepContext{sweep,pad} : Access.SweepContext<Element>}
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,input_tree)),input_tree,StorageLaws.reflect_packed(Element,input_tree)) :
    GatherValues.Facts(~Element,~Swept<Element,Logical>,~value(~Element,~Logical,~original,~operation),Swept{logical,sweep,pad},shape,CopyValues.right_content(~Element,Operations.Accepted{OperationsLaws.modeled_pair(~Element,input_length,output_length,SweepLaws.sweep_model(~Element,~operation,U32.to_nat(written),context,_,Tree.reflect(Element,Tree.pack(Element,output_tree))))}))
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,output_tree)),output_tree,StorageLaws.reflect_packed(Element,output_tree)) :
    GatherValues.Facts(~Element,~Swept<Element,Logical>,~value(~Element,~Logical,~original,~operation),Swept{logical,sweep,pad},shape,CopyValues.right_content(~Element,Operations.Accepted{OperationsLaws.modeled_pair(~Element,input_length,output_length,SweepLaws.sweep_model(~Element,~operation,U32.to_nat(written),context,input_tree,_))}))
  %Equal.sym(Buffer.Content<Element>,CopyValues.right_content(~Element,Operations.Accepted{OperationsLaws.modeled_pair(~Element,input_length,output_length,SweepLaws.sweep_model(~Element,~operation,U32.to_nat(written),context,input_tree,output_tree))}),
    Buffer.Content{IndexedValues.output_tree(~Tree.Tree<Element>,~Element,SweepLaws.sweep_model(~Element,~operation,U32.to_nat(written),context,input_tree,output_tree)),output_length},
    modeled_content(~Element,input_length,output_length,SweepLaws.sweep_model(~Element,~operation,U32.to_nat(written),context,input_tree,output_tree))) :
    GatherValues.Facts(~Element,~Swept<Element,Logical>,~value(~Element,~Logical,~original,~operation),Swept{logical,sweep,pad},shape,_)
  %Equal.sym(Nat,U32.to_nat(written),output_length,written_all) :
    GatherValues.Facts(~Element,~Swept<Element,Logical>,~value(~Element,~Logical,~original,~operation),Swept{logical,sweep,pad},shape,Buffer.Content{run(~Element,~operation,_,context,input_tree,output_tree),output_length})
  model_facts(~Element,~Logical,~original,~operation,logical,sweep,pad,U32.to_nat(outer),input,output,shape,limits,ready,values,guards,cells,length_exact)

def selected(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,valid: Bool,written: Maybe<&2,U32>,
  +logical: Logical,+pad: Element,+sweep: Access.Sweep,+outer: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+shape: List<&2,U32>,+limits: Machine.Limits,
  +actual: {limits == Default.limits() : Machine.Limits},+input_ready: Buffer.Sized<Element,input>,+ready: Buffer.Sized<Element,output>,+values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +cells: {output_cells(sweep,U32.to_nat(outer)) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(shape) : Nat},
  +valid_exact: {guard(~Element,sweep,outer,input,output) == valid : Bool},
  +written_exact: {Operations.sweep_output(sweep,outer) == written : Maybe<&2,U32>},
  accepted: {GatherValues.accepted_owners(~Element,SweepLaws.checked_model(~Element,~operation,SweepLaws.Call{valid,written,Access.SweepContext{sweep,pad},Buffer.length(~Element,input),Buffer.length(~Element,output)},Tree.reflect(Element,Tree.pack(Element,Buffer.tree(~Element,input))),Tree.reflect(Element,Tree.pack(Element,Buffer.tree(~Element,output))))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Swept<Element,Logical>,~value(~Element,~Logical,~original,~operation),Swept{logical,sweep,pad},shape,CopyValues.right_content(~Element,SweepLaws.checked_model(~Element,~operation,SweepLaws.Call{valid,written,Access.SweepContext{sweep,pad},Buffer.length(~Element,input),Buffer.length(~Element,output)},Tree.reflect(Element,Tree.pack(Element,Buffer.tree(~Element,input))),Tree.reflect(Element,Tree.pack(Element,Buffer.tree(~Element,output)))))):
  match valid written:
    case True{} Some{+written}:
      accepted_model(~Element,~Logical,~original,~operation,logical,pad,sweep,outer,input,output,shape,limits,written,ready,values,
        guards_accepted(~Element,sweep,outer,input,output,limits,actual,input_ready,ready,cells,valid_exact),cells,length_exact,
        Equal.trans(Nat,U32.to_nat(written),output_cells(sweep,U32.to_nat(outer)),Buffer.length(~Element,output),
          Equal.sym(Nat,output_cells(sweep,U32.to_nat(outer)),U32.to_nat(written),output_written(sweep,outer,written,written_exact)),cells))
    case True{} None{}: Empty.absurd(GatherValues.Facts(~Element,~Swept<Element,Logical>,~value(~Element,~Logical,~original,~operation),Swept{logical,sweep,pad},shape,CopyValues.right_content(~Element,SweepLaws.checked_model(~Element,~operation,SweepLaws.Call{True{},None{},Access.SweepContext{sweep,pad},Buffer.length(~Element,input),Buffer.length(~Element,output)},Tree.reflect(Element,Tree.pack(Element,Buffer.tree(~Element,input))),Tree.reflect(Element,Tree.pack(Element,Buffer.tree(~Element,output)))))),WordEncoding.false_true(accepted))
    case False{} Some{written}: Empty.absurd(GatherValues.Facts(~Element,~Swept<Element,Logical>,~value(~Element,~Logical,~original,~operation),Swept{logical,sweep,pad},shape,CopyValues.right_content(~Element,SweepLaws.checked_model(~Element,~operation,SweepLaws.Call{False{},Some{written},Access.SweepContext{sweep,pad},Buffer.length(~Element,input),Buffer.length(~Element,output)},Tree.reflect(Element,Tree.pack(Element,Buffer.tree(~Element,input))),Tree.reflect(Element,Tree.pack(Element,Buffer.tree(~Element,output)))))),WordEncoding.false_true(accepted))
    case False{} None{}: Empty.absurd(GatherValues.Facts(~Element,~Swept<Element,Logical>,~value(~Element,~Logical,~original,~operation),Swept{logical,sweep,pad},shape,CopyValues.right_content(~Element,SweepLaws.checked_model(~Element,~operation,SweepLaws.Call{False{},None{},Access.SweepContext{sweep,pad},Buffer.length(~Element,input),Buffer.length(~Element,output)},Tree.reflect(Element,Tree.pack(Element,Buffer.tree(~Element,input))),Tree.reflect(Element,Tree.pack(Element,Buffer.tree(~Element,output)))))),WordEncoding.false_true(accepted))

# One accepted checked sweep of a Ready input with complete values, into an
# output it writes whole: the output content is well formed for its shape and
# holds the fold of the logical input at every index. The accepted outcome is
# the only premise about the call; its guards are read from the call itself.
def modeled_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,
  +logical: Logical,+pad: Element,+sweep: Access.Sweep,+outer: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+shape: List<&2,U32>,+limits: Machine.Limits,
  +actual: {limits == Default.limits() : Machine.Limits},+input_ready: Buffer.Sized<Element,input>,+ready: Buffer.Sized<Element,output>,+values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +cells: {output_cells(sweep,U32.to_nat(outer)) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(shape) : Nat},
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,input),sweep,outer,Buffer.pack(~Element,output)))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Swept<Element,Logical>,~value(~Element,~Logical,~original,~operation),Swept{logical,sweep,pad},shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,input),sweep,outer,Buffer.pack(~Element,output))))):
  Buffer.Model{+input_tree,+input_length,Certificate.Certificate{+input_depth,+input_equation}} = input
  Buffer.Model{+output_tree,+output_length,Certificate.Certificate{+output_depth,+output_equation}} = output
  +input = {Buffer.Model{input_tree,input_length,Certificate.Certificate{input_depth,input_equation}} : Buffer.Model<Element>}
  +output = {Buffer.Model{output_tree,output_length,Certificate.Certificate{output_depth,output_equation}} : Buffer.Model<Element>}
  %Equal.sym(Operations.Outcome<OperationsLaws.PairOwners(Element)>,OperationsLaws.observe_pair_outcome(~Element,Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,input),sweep,outer,Buffer.pack(~Element,output))),SweepLaws.denotation(~Element,~operation,pad,Buffer.pack(~Element,input),sweep,outer,Buffer.pack(~Element,output)),
    SweepLaws.checked(~Element,~operation,pad,Buffer.pack(~Element,input),sweep,outer,Buffer.pack(~Element,output))) :
    GatherValues.Facts(~Element,~Swept<Element,Logical>,~value(~Element,~Logical,~original,~operation),Swept{logical,sweep,pad},shape,CopyValues.right_content(~Element,_))
  selected(~Element,~Logical,~original,~operation,guard(~Element,sweep,outer,input,output),Operations.sweep_output(sweep,outer),logical,pad,sweep,outer,input,output,shape,limits,
    actual,input_ready,ready,values,cells,length_exact,{==},{==},
    %SweepLaws.checked(~Element,~operation,pad,Buffer.pack(~Element,input),sweep,outer,Buffer.pack(~Element,output)) : {GatherValues.accepted_owners(~Element,_) == True{} : Bool}
    accepted)

# The digits of a position (block * out_extent + along) * step + offset, as
# the native divisions of Access.line_at read them.
type Decoded<-position: U32,-step: U32,-out_extent: U32,-block: Nat,-along: Nat,-offset: Nat> is Data:
  Decoded{block_exact: {U32.to_nat(((position / step : U32) / out_extent : U32)) == block : Nat},
    along_exact: {U32.to_nat(((position / step : U32) % out_extent : U32)) == along : Nat},
    offset_exact: {U32.to_nat((position % step : U32)) == offset : Nat}}

def decoded(+position: U32,+step: U32,+out_extent: U32,+block: Nat,+along: Nat,+offset: Nat,+limits: Machine.Limits,
  +exact: {U32.to_nat(position) == Nat.add(Nat.mul(Nat.add(Nat.mul(block,U32.to_nat(out_extent)),along),U32.to_nat(step)),offset) : Nat},
  +along_inside: {Nat.is_lt(along,U32.to_nat(out_extent)) == True{} : Bool},
  +offset_inside: {Nat.is_lt(offset,U32.to_nat(step)) == True{} : Bool},
  +step_room: {Nat.is_le(U32.to_nat(step),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_room: {Nat.is_le(U32.to_nat(out_extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> Decoded<position,step,out_extent,block,along,offset>:
  +p = U32.to_nat(position)
  +s = U32.to_nat(step)
  +oe = U32.to_nat(out_extent)
  +line = Nat.add(Nat.mul(block,oe),along)
  +half = Machine.half_word(limits)
  +half_ok = Limits.half_word_fits(limits)
  +step_positive = Order.le_lt_trans(0n,offset,s,Order.zero_le(offset),offset_inside)
  +out_positive = Order.le_lt_trans(0n,along,oe,Order.zero_le(along),along_inside)
  +whole = (position / step : U32)
  +line_exact = Equal.trans(Nat,U32.to_nat(whole),Nat.div(p,s),line,
    U32Division.divide(position,step,half,half_ok,Limits.double_bound(limits,s,step_room),step_positive),
    %Equal.sym(Nat,p,Nat.add(Nat.mul(line,s),offset),exact) : {Nat.div(_,s) == line : Nat}
    BandCoordinates.quotient(line,offset,s,step_positive,offset_inside))
  Decoded{
    Equal.trans(Nat,U32.to_nat((whole / out_extent : U32)),Nat.div(U32.to_nat(whole),oe),block,
      U32Division.divide(whole,out_extent,half,half_ok,Limits.double_bound(limits,oe,out_room),out_positive),
      %Equal.sym(Nat,U32.to_nat(whole),line,line_exact) : {Nat.div(_,oe) == block : Nat}
      BandCoordinates.quotient(block,along,oe,out_positive,along_inside)),
    Equal.trans(Nat,U32.to_nat((whole % out_extent : U32)),Nat.mod(U32.to_nat(whole),oe),along,
      U32Division.modulo(whole,out_extent,half,half_ok,Limits.double_bound(limits,oe,out_room),out_positive),
      %Equal.sym(Nat,U32.to_nat(whole),line,line_exact) : {Nat.mod(_,oe) == along : Nat}
      BandCoordinates.remainder(block,along,oe,out_positive,along_inside)),
    Equal.trans(Nat,U32.to_nat((position % step : U32)),Nat.mod(p,s),offset,
      U32Division.modulo(position,step,half,half_ok,Limits.double_bound(limits,s,step_room),step_positive),
      %Equal.sym(Nat,p,Nat.add(Nat.mul(line,s),offset),exact) : {Nat.mod(_,s) == offset : Nat}
      BandCoordinates.remainder(line,offset,s,step_positive,offset_inside))}

# The address of a sample whose block, step offset and coordinate are known:
# computed in native words, it is the row-major index of that cell.
def address_exact(+plane: U32,+extent: U32,+step: U32,+offset: U32,+coordinate: U32,+q: Nat,+r: Nat,+outer: Nat,+limits: Machine.Limits,
  +plane_exact: {U32.to_nat(plane) == q : Nat},+offset_exact: {U32.to_nat(offset) == r : Nat},
  +plane_below: {Nat.is_lt(q,outer) == True{} : Bool},+offset_below: {Nat.is_lt(r,U32.to_nat(step)) == True{} : Bool},
  +valid: {Nat.is_lt(U32.to_nat(coordinate),U32.to_nat(extent)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(outer,Nat.mul(U32.to_nat(extent),U32.to_nat(step))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat((plane * extent * step + offset + coordinate * step : U32)) == RowMajor.index(q,U32.to_nat(extent),U32.to_nat(step),U32.to_nat(coordinate),r) : Nat}:
  +s = U32.to_nat(step)
  +e = U32.to_nat(extent)
  +c = U32.to_nat(coordinate)
  +capacity = Machine.capacity(limits)
  +capacity_ok = Limits.capacity_fits(limits)
  +limit = U32.to_nat(capacity)
  +total = Nat.mul(outer,Nat.mul(e,s))
  +step_positive = Order.le_lt_trans(0n,r,s,Order.zero_le(r),offset_below)
  +cells = Nat.mul(Nat.mul(q,e),s)
  +taps = Nat.mul(c,s)
  +address = RowMajor.index(q,e,s,c,r)
  +bounded = RowMajor.in_bounds(q,outer,e,s,c,r,plane_below,valid,offset_below)
  +reordered = Equal.trans(Nat,Nat.add(Nat.add(cells,r),taps),Nat.add(Nat.add(cells,taps),r),address,swap_last(cells,r,taps),RowMajor.expanded(q,e,s,c,r))
  +address_room = Bounds.strict_le(address,limit,Bounds.nat_trans(address,total,limit,bounded,room))
  +sum_room = {
    %Equal.sym(Nat,Nat.add(Nat.add(cells,r),taps),address,reordered) : {Nat.is_le(_,limit) == True{} : Bool}
    address_room : {Nat.is_le(Nat.add(Nat.add(cells,r),taps),limit) == True{} : Bool}}
  +base_room = Order.transitive(Nat.add(cells,r),Nat.add(Nat.add(cells,r),taps),limit,Order.le_add(Nat.add(cells,r),taps),sum_room)
  +cells_room = Order.transitive(cells,Nat.add(cells,r),limit,Order.le_add(cells,r),base_room)
  +taps_room = Order.transitive(taps,Nat.add(Nat.add(cells,r),taps),limit,
    %Order.commutative(taps,Nat.add(cells,r)) : {Nat.is_le(taps,_) == True{} : Bool}
    Order.le_add(taps,Nat.add(cells,r)),sum_room)
  +plane_rows = product_exact(plane,extent,q,e,capacity,capacity_ok,plane_exact,{==},RowMajor.row_product(q,outer,e,s,limit,plane_below,step_positive,room))
  +cells_exact = product_exact((plane * extent : U32),step,Nat.mul(q,e),s,capacity,capacity_ok,plane_rows,{==},cells_room)
  +base_exact = sum_exact((plane * extent * step : U32),offset,cells,r,capacity,capacity_ok,cells_exact,offset_exact,base_room)
  +taps_exact = product_exact(coordinate,step,c,s,capacity,capacity_ok,{==},{==},taps_room)
  Equal.trans(Nat,U32.to_nat((plane * extent * step + offset + coordinate * step : U32)),Nat.add(Nat.add(cells,r),taps),address,
    sum_exact((plane * extent * step + offset : U32),(coordinate * step : U32),Nat.add(cells,r),taps,capacity,capacity_ok,base_exact,taps_exact,sum_room),reordered)

# Two lines whose samples agree tap by tap fold to the same value: the line
# of a stored leaf against the line of the logical input it holds.
def TapFacts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+input: Tree.Tree<Element>,+local: Access.Line<Element>,+global: Access.Line<Element>,remaining: Nat,+tap: U32) -> Data:
  match remaining:
    case 0n: Unit
    case 1n+rest:
      Shape.Both<{SweepLaws.sample_line(~Element,input,local,tap) == logical_line(~Element,~Logical,~original,logical,global,tap) : Element},
        TapFacts(~Element,~Logical,~original,logical,input,local,global,rest,(tap + 1 : U32))>

def taps_match(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,remaining: Nat,+tap: U32,
  +logical: Logical,+input: Tree.Tree<Element>,+local: Access.Line<Element>,+global: Access.Line<Element>,+initial: Element,
  facts: TapFacts(~Element,~Logical,~original,logical,input,local,global,remaining,tap)) ->
  {SweepLaws.line_values(~Element,~operation,remaining,tap,input,local,initial) == Fold.values(~Logical,~Element,~Element,~Access.Line<Element>,~logical_line(~Element,~Logical,~original),~operation,remaining,tap,logical,global,initial) : Element}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      Shape.Both{head,tail} = facts
      %Equal.sym(Element,SweepLaws.sample_line(~Element,input,local,tap),logical_line(~Element,~Logical,~original,logical,global,tap),head) :
        {SweepLaws.line_values(~Element,~operation,rest,(tap + 1 : U32),input,local,operation(initial,_)) == Fold.values(~Logical,~Element,~Element,~Access.Line<Element>,~logical_line(~Element,~Logical,~original),~operation,1n+rest,tap,logical,global,initial) : Element}
      taps_match(~Element,~Logical,~original,~operation,rest,(tap + 1 : U32),logical,input,local,global,operation(initial,logical_line(~Element,~Logical,~original,logical,global,tap)),tail)

# The unchecked sweep of two owners is the pair of its model: the callers
# that establish the bounds themselves (a band leaf) start from here.
def swept_packed(~Element: Data,~operation: Element -> Element -> Element,+count: Nat,+context: Access.SweepContext<Element>,
  +input: Buffer.Model<Element>,+output: Buffer.Model<Element>) ->
  {OperationsLaws.observe_pair(~Element,Operations.swept(~Element,~operation,count,context,Buffer.pack(~Element,input),Buffer.pack(~Element,output))) ==
    OperationsLaws.modeled_pair(~Element,Buffer.length(~Element,input),Buffer.length(~Element,output),
      SweepLaws.sweep_model(~Element,~operation,count,context,Buffer.tree(~Element,input),Buffer.tree(~Element,output))) : OperationsLaws.PairOwners(Element)}:
  Buffer.Model{+input,+input_length,+input_certificate} = input
  Buffer.Model{+output,+output_length,+output_certificate} = output
  OperationsLaws.pair_finished(~Element,input_length,output_length,
    Access.sweep_run(~Element,~operation,count,context,Tree.pack(Element,input),Tree.pack(Element,output)),
    AccessCertificate.sweep(~Element,~operation,count,context,Tree.pack(Element,input),Tree.pack(Element,output),input_certificate,output_certificate),
    SweepLaws.sweep_model(~Element,~operation,count,context,input,output),SweepLaws.sweep(~Element,~operation,count,context,input,output))
