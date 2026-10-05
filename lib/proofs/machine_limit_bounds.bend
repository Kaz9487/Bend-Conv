# Project the checked binary evidence; these are not extra assumptions.
import Base
import ../machine_limits.bend as Machine
import ./u32_shift.bend as U32Shift
import ./nat_order.bend as Order

def dimension_fits(+limits: Machine.Limits) -> {U32.is_le(Machine.dimension(limits),2147483648) == True{} : Bool}:
  Machine.Limits{d,c,k,h,Machine.Facts{df,cf,kf,hf,da,ca,db,cb,dc,dk,ddk,ch,sc,sh,chb}} = limits
  df
def capacity_fits(+limits: Machine.Limits) -> {U32.is_le(Machine.capacity(limits),2147483648) == True{} : Bool}:
  Machine.Limits{d,c,k,h,Machine.Facts{df,cf,kf,hf,da,ca,db,cb,dc,dk,ddk,ch,sc,sh,chb}} = limits
  cf
def coordinate_fits(+limits: Machine.Limits) -> {U32.is_le(Machine.coordinate(limits),2147483648) == True{} : Bool}:
  Machine.Limits{d,c,k,h,Machine.Facts{df,cf,kf,hf,da,ca,db,cb,dc,dk,ddk,ch,sc,sh,chb}} = limits
  kf
def half_word_fits(+limits: Machine.Limits) -> {U32.is_le(Machine.half_word(limits),2147483648) == True{} : Bool}:
  Machine.Limits{d,c,k,h,Machine.Facts{df,cf,kf,hf,da,ca,db,cb,dc,dk,ddk,ch,sc,sh,chb}} = limits
  hf
def dimension_allocatable(+limits: Machine.Limits) -> {U32.is_le(Machine.dimension(limits),1073741824) == True{} : Bool}:
  Machine.Limits{d,c,k,h,Machine.Facts{df,cf,kf,hf,da,ca,db,cb,dc,dk,ddk,ch,sc,sh,chb}} = limits
  da
def capacity_allocatable(+limits: Machine.Limits) -> {U32.is_le(Machine.capacity(limits),1073741824) == True{} : Bool}:
  Machine.Limits{d,c,k,h,Machine.Facts{df,cf,kf,hf,da,ca,db,cb,dc,dk,ddk,ch,sc,sh,chb}} = limits
  ca
def dimension_below_half(+limits: Machine.Limits) -> {U32.is_lt(Machine.dimension(limits),2147483648) == True{} : Bool}:
  Machine.Limits{d,c,k,h,Machine.Facts{df,cf,kf,hf,da,ca,db,cb,dc,dk,ddk,ch,sc,sh,chb}} = limits
  db
def capacity_below_half(+limits: Machine.Limits) -> {U32.is_lt(Machine.capacity(limits),2147483648) == True{} : Bool}:
  Machine.Limits{d,c,k,h,Machine.Facts{df,cf,kf,hf,da,ca,db,cb,dc,dk,ddk,ch,sc,sh,chb}} = limits
  cb
def dimension_within_capacity(+limits: Machine.Limits) -> {U32.is_le(Machine.dimension(limits),
    Machine.capacity(limits)) == True{} : Bool}:
  Machine.Limits{d,c,k,h,Machine.Facts{df,cf,kf,hf,da,ca,db,cb,dc,dk,ddk,ch,sc,sh,chb}} = limits
  dc
def dimension_within_coordinate(+limits: Machine.Limits) -> {U32.is_le(Machine.dimension(limits),
    Machine.coordinate(limits)) == True{} : Bool}:
  Machine.Limits{d,c,k,h,Machine.Facts{df,cf,kf,hf,da,ca,db,cb,dc,dk,ddk,ch,sc,sh,chb}} = limits
  dk
def doubled_dimension_within_coordinate(+limits: Machine.Limits) -> {U32.is_le(U32.shl(Machine.dimension(limits)),
    Machine.coordinate(limits)) == True{} : Bool}:
  Machine.Limits{d,c,k,h,Machine.Facts{df,cf,kf,hf,da,ca,db,cb,dc,dk,ddk,ch,sc,sh,chb}} = limits
  ddk
def capacity_doubles_to_half(+limits: Machine.Limits) -> {U32.shl(Machine.capacity(limits)) == Machine.half_word(limits) : U32}:
  Machine.Limits{d,c,k,h,Machine.Facts{df,cf,kf,hf,da,ca,db,cb,dc,dk,ddk,ch,sc,sh,chb}} = limits
  ch
def seven_within_capacity(+limits: Machine.Limits) -> {U32.is_le(7,Machine.capacity(limits)) == True{} : Bool}:
  Machine.Limits{d,c,k,h,Machine.Facts{df,cf,kf,hf,da,ca,db,cb,dc,dk,ddk,ch,sc,sh,chb}} = limits
  sc
def sixteen_within_half(+limits: Machine.Limits) -> {U32.is_le(16,Machine.half_word(limits)) == True{} : Bool}:
  Machine.Limits{d,c,k,h,Machine.Facts{df,cf,kf,hf,da,ca,db,cb,dc,dk,ddk,ch,sc,sh,chb}} = limits
  sh
def capacity_within_half(+limits: Machine.Limits) -> {U32.is_le(Machine.capacity(limits),
    Machine.half_word(limits)) == True{} : Bool}:
  Machine.Limits{d,c,k,h,Machine.Facts{df,cf,kf,hf,da,ca,db,cb,dc,dk,ddk,ch,sc,sh,chb}} = limits
  chb

def doubled_capacity(+limits: Machine.Limits) -> {U32.to_nat(Machine.half_word(limits)) == Nat.double(U32.to_nat(Machine.capacity(limits))) : Nat}:
  %capacity_doubles_to_half(limits) : {U32.to_nat(_) == Nat.double(U32.to_nat(Machine.capacity(limits))) : Nat}
  U32Shift.double(Machine.capacity(limits),capacity_below_half(limits))

def double_bound(+limits: Machine.Limits,+value: Nat,bounded: {Nat.is_le(value,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.double(value),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(Machine.half_word(limits)),Nat.double(U32.to_nat(Machine.capacity(limits))),doubled_capacity(limits)) :
    {Nat.is_le(Nat.double(value),_) == True{} : Bool}
  Order.double_le(value,U32.to_nat(Machine.capacity(limits)),bounded)
