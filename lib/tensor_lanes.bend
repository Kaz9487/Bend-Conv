# Owned FP32 input bundles use the same Cells/Lanes rounds as unary maps.
import Base
import ./math_compare.bend as Compare
import ./storage_buffer.bend as Storage
import ./traversal_lanes.bend as Lanes
import ./traversal_cells.bend as Cells
import ./kernel_tile.bend as Tile
import ./tensor_operations.bend as Operations
import ./tensor_view.bend as Views
import ./tensor_access.bend as Access
import ./proofs/storage_certificate.bend as Certificate
import ./proofs/traversal_lanes_certificate.bend as LanesCertificate
import ./proofs/traversal_witness.bend as Witness

def identity(context: Unit,value: F32) -> F32:
  value

def read_result(length: Nat,result: Array<F32> & Tile.Tile8,
  certificate: Certificate.Certificate<F32,Witness.first(~Array<F32>,~Tile.Tile8,result)>) -> Storage.Buffer<F32> & Tile.Tile8:
  (values,tile) = result
  (Storage.Buffer{values,length,certificate},tile)

def read(buffer: Storage.Buffer<F32>,+position: U32) -> Storage.Buffer<F32> & Tile.Tile8:
  Storage.Buffer{values,length,certificate} = buffer
  read_result(length,Lanes.gather(~Unit,~identity,values,position,Unit{}),LanesCertificate.gather(~Unit,~identity,values,position,Unit{},certificate))

def write(buffer: Storage.Buffer<F32>,+position: U32,+tile: Tile.Tile8) -> Storage.Buffer<F32>:
  Storage.Buffer{values,length,certificate} = buffer
  Storage.Buffer{Lanes.scatter(values,position,tile),length,LanesCertificate.scatter(values,position,tile,certificate)}

def combine_tile(~operation: F32 -> F32 -> F32,left: Tile.Tile8,right: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{a,b,c,d,e,f,g,h} = left
  Tile.Tile{i,j,k,l,m,n,o,p} = right
  Tile.Tile{operation(a,i),operation(b,j),operation(c,k),operation(d,l),operation(e,m),operation(f,n),operation(g,o),operation(h,p)}

type Binary is Type:
  Binary{left: Storage.Buffer<F32>,right: Storage.Buffer<F32>}

def pair_reads(~operation: F32 -> F32 -> F32,left: Storage.Buffer<F32> & Tile.Tile8,right: Storage.Buffer<F32> & Tile.Tile8) -> Binary & Tile.Tile8:
  (left,a) = left
  (right,b) = right
  (Binary{left,right},combine_tile(~operation,a,b))

def pair_gather(~operation: F32 -> F32 -> F32,owners: Binary,position: U32,context: Unit) -> Binary & Tile.Tile8:
  Binary{left,right} = owners
  +position = position
  pair_reads(~operation,read(left,position),read(right,position))

def pair_scatter(owners: Binary,position: U32,tile: Tile.Tile8) -> Binary:
  Binary{left,right} = owners
  Binary{write(left,position,tile),right}

def pair_result(result: Storage.Buffer<F32> & Storage.Buffer<F32>) -> Binary:
  (left,right) = result
  Binary{left,right}

# The checked caller has already validated both views and owners.
def pair_finish(~operation: F32 -> F32 -> F32,count: Nat,state: Cells.State<Binary,Tile.Tile8,Unit>) -> Binary:
  Cells.State{Binary{left,right},scratch,+position,slot,context} = state
  +count = count
  pair_result(Operations.combine_kernel(~F32,~operation,count,Views.View{[Views.Axis{U32.from_nat(count),1}],position},position,left,right))

def pair_map(~operation: F32 -> F32 -> F32,count: Nat,position: U32,left: Storage.Buffer<F32>,right: Storage.Buffer<F32>) -> Binary:
  Lanes.rounds_with(~Binary,~Unit,~pair_gather(~operation),~pair_scatter,~pair_finish(~operation),count,
    Cells.State{Binary{left,right},Lanes.scratch(),position,0,Unit{}})

def pair_buffers(owners: Binary) -> Storage.Buffer<F32> & Storage.Buffer<F32>:
  Binary{left,right} = owners
  (left,right)

def pair_when(~operation: F32 -> F32 -> F32,fast: Bool,count: Nat,right_view: Views.View,offset: U32,
  left: Storage.Buffer<F32>,right: Storage.Buffer<F32>) -> Storage.Buffer<F32> & Storage.Buffer<F32>:
  match fast:
    case True{}: pair_buffers(pair_map(~operation,count,0,left,right))
    case False{}: Operations.combine_kernel(~F32,~operation,count,right_view,offset,left,right)

def combine(~operation: F32 -> F32 -> F32,+count: Nat,+right_view: Views.View,+offset: U32,
  left: Storage.Buffer<F32>,right: Storage.Buffer<F32>) -> Storage.Buffer<F32> & Storage.Buffer<F32>:
  pair_when(~operation,Access.zero_contiguous(right_view) && U32.is_zero(offset),count,right_view,offset,left,right)

type Selected is Type:
  Selected{mask: Storage.Buffer<F32>,yes: Storage.Buffer<F32>,no: Storage.Buffer<F32>}

def select_tile(mask: Tile.Tile8,yes: Tile.Tile8,no: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{a,b,c,d,e,f,g,h} = mask
  Tile.Tile{i,j,k,l,m,n,o,p} = yes
  Tile.Tile{q,r,s,t,u,v,w,x} = no
  Tile.Tile{Compare.select(a,i,q),Compare.select(b,j,r),Compare.select(c,k,s),Compare.select(d,l,t),
    Compare.select(e,m,u),Compare.select(f,n,v),Compare.select(g,o,w),Compare.select(h,p,x)}

def select_reads(mask: Storage.Buffer<F32> & Tile.Tile8,yes: Storage.Buffer<F32> & Tile.Tile8,no: Storage.Buffer<F32> & Tile.Tile8) -> Selected & Tile.Tile8:
  (mask,a) = mask
  (yes,b) = yes
  (no,c) = no
  (Selected{mask,yes,no},select_tile(a,b,c))

def select_gather(owners: Selected,position: U32,context: Unit) -> Selected & Tile.Tile8:
  Selected{mask,yes,no} = owners
  +position = position
  select_reads(read(mask,position),read(yes,position),read(no,position))

def select_scatter(owners: Selected,position: U32,tile: Tile.Tile8) -> Selected:
  Selected{mask,yes,no} = owners
  Selected{mask,write(yes,position,tile),no}

def select_joined(mask: Storage.Buffer<F32>,joined: Storage.Buffer<F32> & Storage.Buffer<F32>) -> Selected:
  (yes,no) = joined
  Selected{mask,yes,no}

def select_rejected(+count: Nat,+position: U32,yes: Storage.Buffer<F32>,result: Storage.Buffer<F32> & Storage.Buffer<F32>) -> Selected:
  (no,mask) = result
  select_joined(mask,Operations.combine_kernel(~F32,~Compare.union,count,Views.View{[Views.Axis{U32.from_nat(count),1}],position},position,yes,no))

def select_chosen(+count: Nat,+position: U32,no: Storage.Buffer<F32>,result: Storage.Buffer<F32> & Storage.Buffer<F32>) -> Selected:
  (yes,mask) = result
  select_rejected(count,position,yes,
    Operations.combine_kernel(~F32,~Compare.rejected,count,Views.View{[Views.Axis{U32.from_nat(count),1}],position},position,no,mask))

# The final 0..7 values use the existing certified binary kernels. Only the
# tail makes three passes; full cells read all three inputs in one round.
def select_finish(count: Nat,state: Cells.State<Selected,Tile.Tile8,Unit>) -> Selected:
  Cells.State{Selected{mask,yes,no},scratch,+position,slot,context} = state
  +count = count
  select_chosen(count,position,no,
    Operations.combine_kernel(~F32,~Compare.selected,count,Views.View{[Views.Axis{U32.from_nat(count),1}],position},position,yes,mask))

def select_result(owners: Selected) -> Storage.Buffer<F32>:
  Selected{mask,yes,no} = owners
  yes

def select(count: Nat,mask: Storage.Buffer<F32>,yes: Storage.Buffer<F32>,no: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  select_result(Lanes.rounds_with(~Selected,~Unit,~select_gather,~select_scatter,~select_finish,count,
    Cells.State{Selected{mask,yes,no},Lanes.scratch(),0,0,Unit{}}))
