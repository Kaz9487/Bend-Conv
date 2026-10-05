import ../../lib/proofs/storage_shape.bend as Shape
# Every key in the actual frozen K-axis enumeration is in its rectangular bounds.
import Base
import ../specification.bend as ConvSpec
import ../matrix_specification.bend as MatrixSpec
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ./convolution_guard_facts.bend as Guards
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/sequence_membership.bend as Members

def IndexBounds(+p: ConvSpec.Problem,index: MatrixSpec.Reduction) -> Data:
  MatrixSpec.KIndex{input_channel_index,kernel_row_index,kernel_column_index} = index
  Shape.Both<{Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool},Shape.Both<{Nat.is_lt(kernel_row_index,Guards.kernel(p)) == True{} : Bool},{Nat.is_lt(kernel_column_index,
      Guards.kernel(p)) == True{} : Bool}>>

def AllBounds(+p: ConvSpec.Problem,indices: List<&2,MatrixSpec.Reduction>) -> Data:
  Members.All(MatrixSpec.Reduction,{index => IndexBounds(p,index) : MatrixSpec.Reduction -> Data},indices)

def concatenate_bounds(+p: ConvSpec.Problem,+left: List<&2,MatrixSpec.Reduction>,+right: List<&2,MatrixSpec.Reduction>,left_proof: AllBounds(p,
    left),right_proof: AllBounds(p,right)) -> AllBounds(p,OrderedMatrix.cat(MatrixSpec.Reduction,left,right)):
  Members.all_concatenated(MatrixSpec.Reduction,{index => IndexBounds(p,index) : MatrixSpec.Reduction -> Data},left,right,left_proof,right_proof)

def column_bounds(+p: ConvSpec.Problem,+count: Nat,+kernel_column_index: Nat,+kernel_row_index: Nat,+input_channel_index: Nat,+input_channel_in_bounds: {Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool},
  +kernel_row_in_bounds: {Nat.is_lt(kernel_row_index,Guards.kernel(p)) == True{} : Bool},+room: {Nat.is_le(Nat.add(kernel_column_index,count),
      Guards.kernel(p)) == True{} : Bool}) -> AllBounds(p,MatrixSpec.kcols(count,kernel_column_index,kernel_row_index,input_channel_index)):
  match count:
    case 0n:
      Unit{}
    case 1n+ +rest:
      Shape.Both{Shape.Both{input_channel_in_bounds,Shape.Both{kernel_row_in_bounds,Order.room_index(kernel_column_index,rest,Guards.kernel(p),room)}},column_bounds(p,rest,1n+kernel_column_index,kernel_row_index,input_channel_index,
          input_channel_in_bounds,kernel_row_in_bounds,Order.advance_room(kernel_column_index,rest,Guards.kernel(p),room))}

def row_bounds(+p: ConvSpec.Problem,+count: Nat,+kernel_row_index: Nat,+input_channel_index: Nat,+input_channel_in_bounds: {Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(kernel_row_index,count),Guards.kernel(p)) == True{} : Bool}) -> AllBounds(p,MatrixSpec.krows(count,kernel_row_index,input_channel_index,Guards.kernel(p))):
  match count:
    case 0n:
      Unit{}
    case 1n+ +rest:
      concatenate_bounds(p,MatrixSpec.kcols(Guards.kernel(p),0n,kernel_row_index,input_channel_index),MatrixSpec.krows(rest,1n+kernel_row_index,input_channel_index,Guards.kernel(p)),
        column_bounds(p,Guards.kernel(p),0n,kernel_row_index,input_channel_index,input_channel_in_bounds,Order.room_index(kernel_row_index,rest,Guards.kernel(p),room),Order.reflexive(Guards.kernel(p))),
        row_bounds(p,rest,1n+kernel_row_index,input_channel_index,input_channel_in_bounds,Order.advance_room(kernel_row_index,rest,Guards.kernel(p),room)))

def channel_bounds(+p: ConvSpec.Problem,+count: Nat,+input_channel_index: Nat,+room: {Nat.is_le(Nat.add(input_channel_index,count),Guards.input_channel_count(p)) == True{} : Bool}) -> AllBounds(p,
  MatrixSpec.kchannels(count,input_channel_index,Guards.kernel(p))):
  match count:
    case 0n:
      Unit{}
    case 1n+ +rest:
      concatenate_bounds(p,MatrixSpec.krows(Guards.kernel(p),0n,input_channel_index,Guards.kernel(p)),MatrixSpec.kchannels(rest,1n+input_channel_index,Guards.kernel(p)),
        row_bounds(p,Guards.kernel(p),0n,input_channel_index,Order.room_index(input_channel_index,rest,Guards.input_channel_count(p),room),Order.reflexive(Guards.kernel(p))),
        channel_bounds(p,rest,1n+input_channel_index,Order.advance_room(input_channel_index,rest,Guards.input_channel_count(p),room)))

def matrix_reduction_in_bounds(+p: ConvSpec.Problem) -> AllBounds(p,MatrixSpec.kchannels(Guards.input_channel_count(p),0n,Guards.kernel(p))):
  channel_bounds(p,Guards.input_channel_count(p),0n,Order.reflexive(Guards.input_channel_count(p)))
