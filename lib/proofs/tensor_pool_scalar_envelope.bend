# Pool scalar arithmetic with formal word bounds and machine limits.
import Base
import ../machine_limits.bend as Machine
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./checked_word_refinement.bend as Checked
import ./u32_division.bend as U32Division
import ./nat_division_order.bend as DivisionOrder
import ./machine_limit_bounds.bend as Limits
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition

type Constraints<-limits: Machine.Limits,-small: U32,-twice: U32,-padded: U32,-output: U32,-padded_stride: U32,-scalar: U32> is Data:
  Constraints{
    small_room: {U32.is_le(small,Machine.capacity(limits)) == True{} : Bool},
    twice_room: {U32.is_le(twice,Machine.capacity(limits)) == True{} : Bool},
    padded_room: {U32.is_le(padded,Machine.capacity(limits)) == True{} : Bool},
    output_room: {U32.is_le(output,Machine.capacity(limits)) == True{} : Bool},
    padded_stride_room: {U32.is_le(padded_stride,Machine.capacity(limits)) == True{} : Bool},
    scalar_room: {U32.is_le(scalar,Machine.half_word(limits)) == True{} : Bool},
    unit_room: {U32.is_le(1,Machine.capacity(limits)) == True{} : Bool},
    twice_exact: {twice == (small + small : U32) : U32},
    padded_exact: {padded == (small + twice : U32) : U32},
    output_exact: {output == (padded + 1 : U32) : U32},
    padded_stride_exact: {padded_stride == (padded + small : U32) : U32},
    scalar_exact: {scalar == (padded_stride + small : U32) : U32}}

type Envelope is Data:
  Envelope{limits: Machine.Limits,small: U32,twice: U32,padded: U32,output: U32,padded_stride: U32,scalar: U32,
    constraints: Constraints<limits,small,twice,padded,output,padded_stride,scalar>}

def limits(envelope: Envelope) -> Machine.Limits:
  Envelope{limits,small,twice,padded,output,padded_stride,scalar,constraints} = envelope
  limits
def small(envelope: Envelope) -> U32:
  Envelope{limits,small,twice,padded,output,padded_stride,scalar,constraints} = envelope
  small
def padded(envelope: Envelope) -> U32:
  Envelope{limits,small,twice,padded,output,padded_stride,scalar,constraints} = envelope
  padded

def small_room(+envelope: Envelope,+value: U32,within: {U32.is_le(value,small(envelope)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(value),U32.to_nat(Machine.capacity(limits(envelope)))) == True{} : Bool}:
  Envelope{+limits,+small,twice,padded,output,padded_stride,scalar,Constraints{room,tr,pr,orr,psr,sr,ur,te,pe,oe,pse,se}} = envelope
  Order.transitive(U32.to_nat(value),U32.to_nat(small),U32.to_nat(Machine.capacity(limits)),Bounds.as_nat_le(value,small,within),Bounds.as_nat_le(small,Machine.capacity(limits),room))

def half_add(+left: U32,+right: U32,+limits: Machine.Limits,
  left_room: {U32.is_le(left,Machine.capacity(limits)) == True{} : Bool},right_room: {U32.is_le(right,Machine.capacity(limits)) == True{} : Bool}) ->
  {U32.to_nat((left + right : U32)) == Nat.add(U32.to_nat(left),U32.to_nat(right)) : Nat}:
  +capacity = U32.to_nat(Machine.capacity(limits))
  +left_bound = Bounds.as_nat_le(left,Machine.capacity(limits),left_room)
  +right_bound = Bounds.as_nat_le(right,Machine.capacity(limits),right_room)
  +sum_bound = Order.transitive(Nat.add(U32.to_nat(left),U32.to_nat(right)),Nat.add(capacity,U32.to_nat(right)),Nat.add(capacity,capacity),
    Order.add_le(U32.to_nat(left),capacity,U32.to_nat(right),left_bound),Order.offset_le(capacity,U32.to_nat(right),capacity,right_bound))
  Arithmetic.add(left,right,Machine.half_word(limits),Limits.half_word_fits(limits),
    %Equal.sym(Nat,U32.to_nat(Machine.half_word(limits)),Nat.double(capacity),Limits.doubled_capacity(limits)) :
      {Nat.is_le(Nat.add(U32.to_nat(left),U32.to_nat(right)),_) == True{} : Bool}
    %Addition.double(capacity) : {Nat.is_le(Nat.add(U32.to_nat(left),U32.to_nat(right)),_) == True{} : Bool}
    sum_bound)

def sum_bound(+left: Nat,+right: Nat,+left_upper: U32,+right_upper: U32,+sum_upper: U32,+limits: Machine.Limits,
  left_inside: {Nat.is_le(left,U32.to_nat(left_upper)) == True{} : Bool},right_inside: {Nat.is_le(right,U32.to_nat(right_upper)) == True{} : Bool},
  left_room: {U32.is_le(left_upper,Machine.capacity(limits)) == True{} : Bool},right_room: {U32.is_le(right_upper,Machine.capacity(limits)) == True{} : Bool},
  native_sum: {sum_upper == (left_upper + right_upper : U32) : U32}) ->
  {Nat.is_le(Nat.add(left,right),U32.to_nat(sum_upper)) == True{} : Bool}:
  +exact = Equal.trans(Nat,U32.to_nat(sum_upper),U32.to_nat((left_upper + right_upper : U32)),Nat.add(U32.to_nat(left_upper),U32.to_nat(right_upper)),
    Equal.cong(U32,Nat,U32.to_nat,sum_upper,(left_upper + right_upper : U32),native_sum),half_add(left_upper,right_upper,limits,left_room,right_room))
  %Equal.sym(Nat,U32.to_nat(sum_upper),Nat.add(U32.to_nat(left_upper),U32.to_nat(right_upper)),exact) : {Nat.is_le(Nat.add(left,right),_) == True{} : Bool}
  Order.transitive(Nat.add(left,right),Nat.add(U32.to_nat(left_upper),right),Nat.add(U32.to_nat(left_upper),U32.to_nat(right_upper)),
    Order.add_le(left,U32.to_nat(left_upper),right,left_inside),Order.offset_le(U32.to_nat(left_upper),right,U32.to_nat(right_upper),right_inside))

def twice_natural(+value: Nat) -> {Nat.mul(2n,value) == Nat.add(value,value) : Nat}:
  Equal.cong(Nat,Nat,tail => Nat.add(value,tail),Nat.add(value,0n),value,Addition.zero_right(value))

def twice_bound(+value: Nat,+upper: U32,+twice_upper: U32,+limits: Machine.Limits,
  inside: {Nat.is_le(value,U32.to_nat(upper)) == True{} : Bool},+upper_room: {U32.is_le(upper,Machine.capacity(limits)) == True{} : Bool},
  native_twice: {twice_upper == (upper + upper : U32) : U32}) ->
  {Nat.is_le(Nat.mul(2n,value),U32.to_nat(twice_upper)) == True{} : Bool}:
  +exact = Equal.trans(Nat,U32.to_nat(twice_upper),U32.to_nat((upper + upper : U32)),Nat.mul(2n,U32.to_nat(upper)),
    Equal.cong(U32,Nat,U32.to_nat,twice_upper,(upper + upper : U32),native_twice),
    Equal.trans(Nat,U32.to_nat((upper + upper : U32)),Nat.add(U32.to_nat(upper),U32.to_nat(upper)),Nat.mul(2n,U32.to_nat(upper)),
      half_add(upper,upper,limits,upper_room,upper_room),Equal.sym(Nat,Nat.mul(2n,U32.to_nat(upper)),Nat.add(U32.to_nat(upper),U32.to_nat(upper)),twice_natural(U32.to_nat(upper)))))
  %Equal.sym(Nat,U32.to_nat(twice_upper),Nat.mul(2n,U32.to_nat(upper)),exact) : {Nat.is_le(Nat.mul(2n,value),_) == True{} : Bool}
  %RowMajor.multiply_commutative(value,2n) : {Nat.is_le(_,Nat.mul(2n,U32.to_nat(upper))) == True{} : Bool}
  %RowMajor.multiply_commutative(U32.to_nat(upper),2n) : {Nat.is_le(Nat.mul(value,2n),_) == True{} : Bool}
  RowMajor.mul_le(value,U32.to_nat(upper),2n,inside)

def padded_small(+envelope: Envelope,+extent: U32,+padding: U32,
  extent_small: {U32.is_le(extent,small(envelope)) == True{} : Bool},padding_small: {U32.is_le(padding,small(envelope)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(extent),Nat.mul(2n,U32.to_nat(padding))),U32.to_nat(padded(envelope))) == True{} : Bool}:
  Envelope{+limits,+small,+twice,+padded,output,padded_stride,scalar,Constraints{+sr,tr,pr,orr,psr,scr,ur,te,pe,oe,pse,se}} = envelope
  +ep = Bounds.as_nat_le(extent,small,extent_small)
  +pp = Bounds.as_nat_le(padding,small,padding_small)
  sum_bound(U32.to_nat(extent),Nat.mul(2n,U32.to_nat(padding)),small,twice,padded,limits,ep,
    twice_bound(U32.to_nat(padding),small,twice,limits,pp,sr,te),sr,tr,pe)

def padded_exact(+envelope: Envelope,+extent: U32,+padding: U32,
  +extent_small: {U32.is_le(extent,small(envelope)) == True{} : Bool},+padding_small: {U32.is_le(padding,small(envelope)) == True{} : Bool}) ->
  {U32.to_nat((extent + 2 * padding : U32)) == Nat.add(U32.to_nat(extent),Nat.mul(2n,U32.to_nat(padding))) : Nat}:
  Envelope{+limits,+small,+twice,+padded,output,padded_stride,scalar,+constraints} = envelope
  Constraints{sr,tr,pr,orr,psr,scr,ur,te,pe,oe,pse,se} = constraints
  +rebuilt = {Envelope{limits,small,twice,padded,output,padded_stride,scalar,constraints} : Envelope}
  +total = Nat.add(U32.to_nat(extent),Nat.mul(2n,U32.to_nat(padding)))
  +room = Order.transitive(total,U32.to_nat(padded),U32.to_nat(Machine.capacity(limits)),padded_small(rebuilt,extent,padding,extent_small,padding_small),Bounds.as_nat_le(padded,Machine.capacity(limits),pr))
  +double_room = Order.transitive(Nat.mul(2n,U32.to_nat(padding)),total,U32.to_nat(Machine.capacity(limits)),
    %Order.commutative(Nat.mul(2n,U32.to_nat(padding)),U32.to_nat(extent)) : {Nat.is_le(Nat.mul(2n,U32.to_nat(padding)),_) == True{} : Bool}
    Order.le_add(Nat.mul(2n,U32.to_nat(padding)),U32.to_nat(extent)),room)
  +double_exact = Arithmetic.multiply(2,padding,Machine.capacity(limits),Limits.capacity_fits(limits),double_room)
  Equal.trans(Nat,U32.to_nat((extent + 2 * padding : U32)),Nat.add(U32.to_nat(extent),U32.to_nat((2 * padding : U32))),total,
    Arithmetic.add(extent,(2 * padding : U32),Machine.capacity(limits),Limits.capacity_fits(limits),
      %Equal.sym(Nat,U32.to_nat((2 * padding : U32)),Nat.mul(2n,U32.to_nat(padding)),double_exact) : {Nat.is_le(Nat.add(U32.to_nat(extent),_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room),
    Equal.cong(Nat,Nat,value => Nat.add(U32.to_nat(extent),value),U32.to_nat((2 * padding : U32)),Nat.mul(2n,U32.to_nat(padding)),double_exact))

def axis_output_exact(+envelope: Envelope,+extent: U32,+kernel: U32,+stride: U32,+padding: U32,+output_extent: U32,
  +extent_small: {U32.is_le(extent,small(envelope)) == True{} : Bool},+padding_small: {U32.is_le(padding,small(envelope)) == True{} : Bool},
  stride_small: {U32.is_le(stride,small(envelope)) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  kernel_inside: {U32.is_le(kernel,(extent + 2 * padding : U32)) == True{} : Bool},
  output_equal: {U32.is_eq(output_extent,((extent + 2 * padding - kernel : U32) / stride + 1 : U32)) == True{} : Bool}) ->
  {U32.to_nat(output_extent) == Nat.add(Nat.div(Nat.sub(Nat.add(U32.to_nat(extent),Nat.mul(2n,U32.to_nat(padding))),U32.to_nat(kernel)),U32.to_nat(stride)),1n) : Nat}:
  Envelope{+limits,+small,twice,+padded,+output,padded_stride,scalar,+constraints} = envelope
  Constraints{sr,tr,+pr,orr,psr,scr,ur,te,padded_certificate,oe,pse,se} = constraints
  +rebuilt = {Envelope{limits,small,twice,padded,output,padded_stride,scalar,constraints} : Envelope}
  +capacity = Machine.capacity(limits)
  +padded_native = (extent + 2 * padding : U32)
  +difference_native = (padded_native - kernel : U32)
  +quotient_native = (difference_native / stride : U32)
  +total = Nat.add(U32.to_nat(extent),Nat.mul(2n,U32.to_nat(padding)))
  +difference = Nat.sub(total,U32.to_nat(kernel))
  +quotient = Nat.div(difference,U32.to_nat(stride))
  +pe = padded_exact(rebuilt,extent,padding,extent_small,padding_small)
  +padded_bound = padded_small(rebuilt,extent,padding,extent_small,padding_small)
  +kernel_below = Bounds.as_nat_le(kernel,padded_native,kernel_inside)
  +difference_room = Order.transitive(Nat.sub(U32.to_nat(padded_native),U32.to_nat(kernel)),U32.to_nat(padded_native),U32.to_nat(capacity),Order.sub_le(U32.to_nat(padded_native),U32.to_nat(kernel)),
    %Equal.sym(Nat,U32.to_nat(padded_native),total,pe) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    Order.transitive(total,U32.to_nat(padded),U32.to_nat(capacity),padded_bound,Bounds.as_nat_le(padded,capacity,pr)))
  +de = Equal.trans(Nat,U32.to_nat(difference_native),Nat.sub(U32.to_nat(padded_native),U32.to_nat(kernel)),difference,
    Arithmetic.subtract(padded_native,kernel,capacity,Limits.capacity_fits(limits),kernel_below,difference_room),
    Equal.cong(Nat,Nat,value => Nat.sub(value,U32.to_nat(kernel)),U32.to_nat(padded_native),total,pe))
  +qe = Equal.trans(Nat,U32.to_nat(quotient_native),Nat.div(U32.to_nat(difference_native),U32.to_nat(stride)),quotient,
    U32Division.divide(difference_native,stride,Machine.half_word(limits),Limits.half_word_fits(limits),Limits.double_bound(limits,U32.to_nat(stride),small_room(rebuilt,stride,stride_small)),stride_positive),
    Equal.cong(Nat,Nat,value => Nat.div(value,U32.to_nat(stride)),U32.to_nat(difference_native),difference,de))
  +quotient_below = Order.transitive(quotient,Nat.mul(quotient,U32.to_nat(stride)),difference,
    RowMajor.product_covers(quotient,U32.to_nat(stride),stride_positive),DivisionOrder.product_below(difference,U32.to_nat(stride),stride_positive))
  +quotient_small = Order.transitive(quotient,difference,U32.to_nat(padded),quotient_below,Order.transitive(difference,total,U32.to_nat(padded),Order.sub_le(total,U32.to_nat(kernel)),padded_bound))
  +sum_room = Order.transitive(Nat.add(quotient,1n),U32.to_nat(output),U32.to_nat(capacity),
    sum_bound(quotient,1n,padded,1,output,limits,quotient_small,Order.reflexive(1n),pr,ur,oe),
    Bounds.as_nat_le(output,capacity,orr))
  Equal.trans(Nat,U32.to_nat(output_extent),U32.to_nat((quotient_native + 1 : U32)),Nat.add(quotient,1n),
    Equal.cong(U32,Nat,U32.to_nat,output_extent,(quotient_native + 1 : U32),Checked.classified_equal(output_extent,(quotient_native + 1 : U32),output_equal)),
    Equal.trans(Nat,U32.to_nat((quotient_native + 1 : U32)),Nat.add(U32.to_nat(quotient_native),1n),Nat.add(quotient,1n),
      Arithmetic.add(quotient_native,1,capacity,Limits.capacity_fits(limits),
        %Equal.sym(Nat,U32.to_nat(quotient_native),quotient,qe) : {Nat.is_le(Nat.add(_,1n),U32.to_nat(capacity)) == True{} : Bool} sum_room),
      Equal.cong(Nat,Nat,value => Nat.add(value,1n),U32.to_nat(quotient_native),quotient,qe)))

def axis_scalar_room(+envelope: Envelope,+extent: U32,+kernel: U32,+stride: U32,+padding: U32,+output_extent: U32,+coordinate: Nat,+tap: Nat,
  extent_small: {U32.is_le(extent,small(envelope)) == True{} : Bool},+padding_small: {U32.is_le(padding,small(envelope)) == True{} : Bool},
  kernel_small: {U32.is_le(kernel,small(envelope)) == True{} : Bool},stride_small: {U32.is_le(stride,small(envelope)) == True{} : Bool},
  +stride_positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +output_exact: {U32.to_nat(output_extent) == Nat.add(Nat.div(Nat.sub(Nat.add(U32.to_nat(extent),Nat.mul(2n,U32.to_nat(padding))),U32.to_nat(kernel)),U32.to_nat(stride)),1n) : Nat},
  coordinate_inside: {Nat.is_le(coordinate,U32.to_nat(output_extent)) == True{} : Bool},tap_inside: {Nat.is_le(tap,U32.to_nat(kernel)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(coordinate,U32.to_nat(stride)),tap),U32.to_nat(Machine.half_word(limits(envelope)))) == True{} : Bool}:
  Envelope{+limits,+small,twice,+padded,output,+padded_stride,+scalar,+constraints} = envelope
  Constraints{+sr,tr,+pr,orr,psr,scr,ur,te,pe,oe,pse,se} = constraints
  +rebuilt = {Envelope{limits,small,twice,padded,output,padded_stride,scalar,constraints} : Envelope}
  +total = Nat.add(U32.to_nat(extent),Nat.mul(2n,U32.to_nat(padding)))
  +difference = Nat.sub(total,U32.to_nat(kernel))
  +quotient = Nat.div(difference,U32.to_nat(stride))
  +padded_bound = padded_small(rebuilt,extent,padding,extent_small,padding_small)
  +sp = Bounds.as_nat_le(stride,small,stride_small)
  +kp = Bounds.as_nat_le(kernel,small,kernel_small)
  +maximum = Nat.mul(Nat.add(quotient,1n),U32.to_nat(stride))
  +maximum_exact = Equal.trans(Nat,maximum,Nat.add(Nat.mul(quotient,U32.to_nat(stride)),Nat.mul(1n,U32.to_nat(stride))),Nat.add(Nat.mul(quotient,U32.to_nat(stride)),U32.to_nat(stride)),
    RowMajor.distrib(quotient,1n,U32.to_nat(stride)),
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(quotient,U32.to_nat(stride)),value),Nat.mul(1n,U32.to_nat(stride)),U32.to_nat(stride),Addition.zero_right(U32.to_nat(stride))))
  +origin_below = Order.transitive(Nat.mul(coordinate,U32.to_nat(stride)),maximum,Nat.add(difference,U32.to_nat(stride)),
    RowMajor.mul_le(coordinate,Nat.add(quotient,1n),U32.to_nat(stride),%output_exact : {Nat.is_le(coordinate,_) == True{} : Bool} coordinate_inside),
    %Equal.sym(Nat,maximum,Nat.add(Nat.mul(quotient,U32.to_nat(stride)),U32.to_nat(stride)),maximum_exact) : {Nat.is_le(_,Nat.add(difference,U32.to_nat(stride))) == True{} : Bool}
    Order.add_le(Nat.mul(quotient,U32.to_nat(stride)),difference,U32.to_nat(stride),DivisionOrder.product_below(difference,U32.to_nat(stride),stride_positive)))
  +coordinate_below = Order.transitive(Nat.add(Nat.mul(coordinate,U32.to_nat(stride)),tap),Nat.add(Nat.add(difference,U32.to_nat(stride)),tap),Nat.add(Nat.add(difference,U32.to_nat(stride)),U32.to_nat(kernel)),
    Order.add_le(Nat.mul(coordinate,U32.to_nat(stride)),Nat.add(difference,U32.to_nat(stride)),tap,origin_below),Order.offset_le(Nat.add(difference,U32.to_nat(stride)),tap,U32.to_nat(kernel),tap_inside))
  +padded_stride_bound = sum_bound(total,U32.to_nat(stride),padded,small,padded_stride,limits,padded_bound,sp,pr,sr,pse)
  +maximum_bound = Order.transitive(Nat.add(Nat.add(difference,U32.to_nat(stride)),U32.to_nat(kernel)),Nat.add(Nat.add(total,U32.to_nat(stride)),U32.to_nat(kernel)),U32.to_nat(scalar),
    Order.add_le(Nat.add(difference,U32.to_nat(stride)),Nat.add(total,U32.to_nat(stride)),U32.to_nat(kernel),Order.add_le(difference,total,U32.to_nat(stride),Order.sub_le(total,U32.to_nat(kernel)))),
    sum_bound(Nat.add(total,U32.to_nat(stride)),U32.to_nat(kernel),padded_stride,small,scalar,limits,padded_stride_bound,kp,psr,sr,se))
  Order.transitive(Nat.add(Nat.mul(coordinate,U32.to_nat(stride)),tap),U32.to_nat(scalar),U32.to_nat(Machine.half_word(limits)),
    Order.transitive(Nat.add(Nat.mul(coordinate,U32.to_nat(stride)),tap),Nat.add(Nat.add(difference,U32.to_nat(stride)),U32.to_nat(kernel)),U32.to_nat(scalar),coordinate_below,maximum_bound),Bounds.as_nat_le(scalar,Machine.half_word(limits),scr))
