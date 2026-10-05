import ../../lib/proofs/ordered_reduction_split.bend as Ordered
# Flattening preserves the exact matrix reduction sequence, not just its set.
import Base
import ../../lib/proofs/traversal_reduction_refinement.bend as ReductionLaws
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/natural_addition.bend as Addition
import ../matrix_specification.bend as MatrixSpec
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ./reduction_coordinate_bounds.bend as Coordinates
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_order.bend as Order
import ./sequence_congruence.bend as Sequences

def flatten(+kernel: Nat,indices: List<&2,MatrixSpec.Reduction>) -> List<&2,Nat>:
  match indices:
    case Nil{}:
      []
    case MatrixSpec.KIndex{input_channel_index,kernel_row_index,kernel_column_index} <> tail:
      Coordinates.reduction(input_channel_index,kernel_row_index,kernel_column_index,kernel) <> flatten(kernel,tail)

def flatten_cat(+kernel: Nat,+left: List<&2,MatrixSpec.Reduction>,+right: List<&2,MatrixSpec.Reduction>) -> {flatten(kernel,
    OrderedMatrix.cat(MatrixSpec.Reduction,left,right)) == OrderedMatrix.cat(Nat,flatten(kernel,left),flatten(kernel,right)) : List<&2,Nat>}:
  match left:
    case Nil{}:
      {==}
    case MatrixSpec.KIndex{input_channel_index,kernel_row_index,kernel_column_index} <> +tail:
      %flatten_cat(kernel,tail,right) : {Coordinates.reduction(input_channel_index,kernel_row_index,kernel_column_index,kernel) <> flatten(kernel,OrderedMatrix.cat(MatrixSpec.Reduction,tail,
            right)) == Coordinates.reduction(input_channel_index,kernel_row_index,kernel_column_index,kernel) <> _ : List<&2,Nat>}
      {==}

def columns_in_order(+count: Nat,+kernel_column_index: Nat,+kernel_row_index: Nat,+input_channel_index: Nat,+kernel: Nat) -> {flatten(kernel,MatrixSpec.kcols(count,kernel_column_index,kernel_row_index,
      input_channel_index)) == Ranges.range(count,Coordinates.reduction(input_channel_index,kernel_row_index,kernel_column_index,kernel)) : List<&2,Nat>}:
  match count:
    case 0n:
      {==}
    case 1n+ +rest:
      %Addition.successor_right(Nat.mul(Nat.add(Nat.mul(input_channel_index,kernel),kernel_row_index),kernel),kernel_column_index) : {flatten(kernel,MatrixSpec.kcols(1n+rest,kernel_column_index,kernel_row_index,
            input_channel_index)) == Coordinates.reduction(input_channel_index,kernel_row_index,kernel_column_index,kernel) <> Ranges.range(rest,_) : List<&2,Nat>}
      %columns_in_order(rest,1n+kernel_column_index,kernel_row_index,input_channel_index,kernel) : {Coordinates.reduction(input_channel_index,kernel_row_index,kernel_column_index,kernel) <> flatten(kernel,MatrixSpec.kcols(rest,1n+kernel_column_index,kernel_row_index,
            input_channel_index)) == Coordinates.reduction(input_channel_index,kernel_row_index,kernel_column_index,kernel) <> _ : List<&2,Nat>}
      {==}

def row_start(+input_channel_index: Nat,+kernel_row_index: Nat,+kernel: Nat) -> Nat:
  Nat.mul(Nat.add(Nat.mul(input_channel_index,kernel),kernel_row_index),kernel)

def next_row(+input_channel_index: Nat,+kernel_row_index: Nat,+kernel: Nat) -> {row_start(input_channel_index,1n+kernel_row_index,kernel) == Nat.add(row_start(input_channel_index,kernel_row_index,kernel),kernel) : Nat}:
  %Equal.sym(Nat,Nat.add(Nat.mul(input_channel_index,kernel),1n+kernel_row_index),1n+Nat.add(Nat.mul(input_channel_index,kernel),kernel_row_index),Addition.successor_right(Nat.mul(input_channel_index,kernel),
      kernel_row_index)) : {Nat.mul(_,kernel) == Nat.add(row_start(input_channel_index,kernel_row_index,kernel),kernel) : Nat}
  Order.commutative(kernel,row_start(input_channel_index,kernel_row_index,kernel))

def rows_in_order(+count: Nat,+kernel_row_index: Nat,+input_channel_index: Nat,+kernel: Nat) -> {flatten(kernel,MatrixSpec.krows(count,kernel_row_index,input_channel_index,
      kernel)) == Ranges.range(Nat.mul(count,kernel),row_start(input_channel_index,kernel_row_index,kernel)) : List<&2,Nat>}:
  match count:
    case 0n:
      {==}
    case 1n+ +rest:
      +left = MatrixSpec.kcols(kernel,0n,kernel_row_index,input_channel_index)
      +right = MatrixSpec.krows(rest,1n+kernel_row_index,input_channel_index,kernel)
      %Equal.sym(List<&2,Nat>,flatten(kernel,OrderedMatrix.cat(MatrixSpec.Reduction,left,right)),OrderedMatrix.cat(Nat,flatten(kernel,left),
          flatten(kernel,right)),flatten_cat(kernel,left,right)) : {_ == Ranges.range(Nat.mul(1n+rest,kernel),row_start(input_channel_index,kernel_row_index,kernel)) : List<&2,Nat>}
      %Equal.sym(List<&2,Nat>,OrderedMatrix.cat(Nat,flatten(kernel,left),flatten(kernel,right)),OrderedMatrix.cat(Nat,Ranges.range(kernel,
            Coordinates.reduction(input_channel_index,kernel_row_index,0n,kernel)),Ranges.range(Nat.mul(rest,kernel),row_start(input_channel_index,1n+kernel_row_index,kernel))),Sequences.cat(Nat,
          flatten(kernel,left),flatten(kernel,right),Ranges.range(kernel,Coordinates.reduction(input_channel_index,kernel_row_index,0n,kernel)),Ranges.range(Nat.mul(rest,
              kernel),row_start(input_channel_index,1n+kernel_row_index,kernel)),columns_in_order(kernel,0n,kernel_row_index,input_channel_index,kernel),rows_in_order(rest,1n+kernel_row_index,input_channel_index,
            kernel))) : {_ == Ranges.range(Nat.mul(1n+rest,kernel),row_start(input_channel_index,kernel_row_index,kernel)) : List<&2,Nat>}
      %Equal.sym(Nat,Coordinates.reduction(input_channel_index,kernel_row_index,0n,kernel),row_start(input_channel_index,kernel_row_index,kernel),Addition.zero_right(row_start(input_channel_index,kernel_row_index,
            kernel))) : {OrderedMatrix.cat(Nat,Ranges.range(kernel,_),Ranges.range(Nat.mul(rest,kernel),row_start(input_channel_index,1n+kernel_row_index,
              kernel))) == Ranges.range(Nat.mul(1n+rest,kernel),row_start(input_channel_index,kernel_row_index,kernel)) : List<&2,Nat>}
      Ranges.join(row_start(input_channel_index,kernel_row_index,kernel),kernel,Nat.mul(rest,kernel),row_start(input_channel_index,1n+kernel_row_index,kernel),Nat.mul(1n+rest,kernel),next_row(input_channel_index,kernel_row_index,kernel),{==})

def channel_start(+input_channel_index: Nat,+kernel: Nat) -> {row_start(input_channel_index,0n,kernel) == Nat.mul(input_channel_index,Nat.mul(kernel,kernel)) : Nat}:
  %Equal.sym(Nat,Nat.add(Nat.mul(input_channel_index,kernel),0n),Nat.mul(input_channel_index,kernel),Addition.zero_right(Nat.mul(input_channel_index,kernel))) : {Nat.mul(_,
      kernel) == Nat.mul(input_channel_index,Nat.mul(kernel,kernel)) : Nat}
  RowMajor.assoc(input_channel_index,kernel,kernel)

def channels_in_order(+count: Nat,+input_channel_index: Nat,+kernel: Nat) -> {flatten(kernel,MatrixSpec.kchannels(count,input_channel_index,kernel)) == Ranges.range(Nat.mul(count,
      Nat.mul(kernel,kernel)),Nat.mul(input_channel_index,Nat.mul(kernel,kernel))) : List<&2,Nat>}:
  match count:
    case 0n:
      {==}
    case 1n+ +rest:
      +width = Nat.mul(kernel,kernel)
      +left = MatrixSpec.krows(kernel,0n,input_channel_index,kernel)
      +right = MatrixSpec.kchannels(rest,1n+input_channel_index,kernel)
      %Equal.sym(List<&2,Nat>,flatten(kernel,OrderedMatrix.cat(MatrixSpec.Reduction,left,right)),OrderedMatrix.cat(Nat,flatten(kernel,left),
          flatten(kernel,right)),flatten_cat(kernel,left,right)) : {_ == Ranges.range(Nat.mul(1n+rest,width),Nat.mul(input_channel_index,width)) : List<&2,Nat>}
      %Equal.sym(List<&2,Nat>,OrderedMatrix.cat(Nat,flatten(kernel,left),flatten(kernel,right)),OrderedMatrix.cat(Nat,Ranges.range(width,
            row_start(input_channel_index,0n,kernel)),Ranges.range(Nat.mul(rest,width),Nat.mul(1n+input_channel_index,width))),Sequences.cat(Nat,flatten(kernel,left),
          flatten(kernel,right),Ranges.range(width,row_start(input_channel_index,0n,kernel)),Ranges.range(Nat.mul(rest,width),Nat.mul(1n+input_channel_index,width)),
          rows_in_order(kernel,0n,input_channel_index,kernel),channels_in_order(rest,1n+input_channel_index,kernel))) : {_ == Ranges.range(Nat.mul(1n+rest,width),Nat.mul(input_channel_index,
            width)) : List<&2,Nat>}
      %Equal.sym(Nat,row_start(input_channel_index,0n,kernel),Nat.mul(input_channel_index,width),channel_start(input_channel_index,kernel)) : {OrderedMatrix.cat(Nat,Ranges.range(width,_),
          Ranges.range(Nat.mul(rest,width),Nat.mul(1n+input_channel_index,width))) == Ranges.range(Nat.mul(1n+rest,width),Nat.mul(input_channel_index,width)) : List<&2,Nat>}
      Ranges.join(Nat.mul(input_channel_index,width),width,Nat.mul(rest,width),Nat.mul(1n+input_channel_index,width),Nat.mul(1n+rest,width),Order.commutative(width,Nat.mul(input_channel_index,
            width)),{==})

law matrix_reduction_is_consecutive:
  for +channels: Nat
  for +kernel: Nat
  {flatten(kernel,MatrixSpec.kchannels(channels,0n,kernel)) == Ranges.range(Nat.mul(channels,Nat.mul(kernel,kernel)),0n) : List<&2,Nat>}

def matrix_reduction_is_consecutive(channels,kernel):
  channels_in_order(channels,0n,kernel)

def address_taps(indices: List<&2,Nat>,+input_base: Nat,+weight_base: Nat) -> List<&2,ReductionLaws.AddressPair>:
  match indices:
    case Nil{}:
      []
    case +index <> tail:
      ReductionLaws.AddressPair{U32.from_nat(Nat.add(input_base,index)),U32.from_nat(Nat.add(weight_base,index))} <> address_taps(tail,input_base,weight_base)

def matrix_tap(index: MatrixSpec.Reduction,+kernel: Nat,input_base: Nat,weight_base: Nat) -> ReductionLaws.AddressPair:
  MatrixSpec.KIndex{input_channel_index,kernel_row_index,kernel_column_index} = index
  +reduction = Coordinates.reduction(input_channel_index,kernel_row_index,kernel_column_index,kernel)
  ReductionLaws.AddressPair{U32.from_nat(Nat.add(input_base,reduction)),U32.from_nat(Nat.add(weight_base,reduction))}

type TapLayout is Data:
  TapLayout{kernel: Nat,input_base: Nat,weight_base: Nat}

def mapped_tap(layout: TapLayout,index: MatrixSpec.Reduction) -> ReductionLaws.AddressPair:
  TapLayout{kernel,input_base,weight_base} = layout
  matrix_tap(index,kernel,input_base,weight_base)

def matrix_taps(indices: List<&2,MatrixSpec.Reduction>,kernel: Nat,input_base: Nat,weight_base: Nat) -> List<&2,ReductionLaws.AddressPair>:
  Ordered.map_keys(~MatrixSpec.Reduction,~ReductionLaws.AddressPair,~TapLayout,~mapped_tap,indices,TapLayout{kernel,input_base,weight_base})

def matrix_taps_flatten(+indices: List<&2,MatrixSpec.Reduction>,+kernel: Nat,+input_base: Nat,+weight_base: Nat) -> {matrix_taps(indices,kernel,
    input_base,weight_base) == address_taps(flatten(kernel,indices),input_base,weight_base) : List<&2,ReductionLaws.AddressPair>}:
  match indices:
    case Nil{}:
      {==}
    case MatrixSpec.KIndex{+input_channel_index,+kernel_row_index,+kernel_column_index} <> +tail:
      +index = Coordinates.reduction(input_channel_index,kernel_row_index,kernel_column_index,kernel)
      %matrix_taps_flatten(tail,kernel,input_base,weight_base) : {ReductionLaws.AddressPair{U32.from_nat(Nat.add(input_base,index)),
          U32.from_nat(Nat.add(weight_base,index))} <> matrix_taps(tail,kernel,input_base,
          weight_base) == ReductionLaws.AddressPair{U32.from_nat(Nat.add(input_base,index)),U32.from_nat(Nat.add(weight_base,index))} <> _ : List<&2,ReductionLaws.AddressPair>}
      {==}

def address_successor(+base: Nat,+index: Nat) -> {(U32.from_nat(Nat.add(base,index)) + 1 : U32) == U32.from_nat(Nat.add(base,1n+index)) : U32}:
  %Equal.sym(Nat,Nat.add(base,1n+index),1n+Nat.add(base,index),Addition.successor_right(base,index)) : {(U32.from_nat(Nat.add(base,
          index)) + 1 : U32) == U32.from_nat(_) : U32}
  Bounds.add_one(U32.from_nat(Nat.add(base,index)))

def consecutive_taps(+count: Nat,+start: Nat,+input_base: Nat,+weight_base: Nat) -> {ReductionLaws.addresses(count,U32.from_nat(Nat.add(input_base,start)),
    U32.from_nat(Nat.add(weight_base,start))) == address_taps(Ranges.range(count,start),input_base,weight_base) : List<&2,ReductionLaws.AddressPair>}:
  match count:
    case 0n:
      {==}
    case 1n+ +rest:
      +first = {ReductionLaws.AddressPair{U32.from_nat(Nat.add(input_base,start)),U32.from_nat(Nat.add(weight_base,start))} : ReductionLaws.AddressPair}
      %Equal.sym(U32,(U32.from_nat(Nat.add(input_base,start)) + 1 : U32),U32.from_nat(Nat.add(input_base,1n+start)),address_successor(input_base,
          start)) : {first <> ReductionLaws.addresses(rest,_,(U32.from_nat(Nat.add(weight_base,
              start)) + 1 : U32)) == first <> address_taps(Ranges.range(rest,1n+start),input_base,weight_base) : List<&2,ReductionLaws.AddressPair>}
      %Equal.sym(U32,(U32.from_nat(Nat.add(weight_base,start)) + 1 : U32),U32.from_nat(Nat.add(weight_base,1n+start)),
        address_successor(weight_base,start)) : {first <> ReductionLaws.addresses(rest,U32.from_nat(Nat.add(input_base,1n+start)),
          _) == first <> address_taps(Ranges.range(rest,1n+start),input_base,weight_base) : List<&2,ReductionLaws.AddressPair>}
      %consecutive_taps(rest,1n+start,input_base,weight_base) : {first <> ReductionLaws.addresses(rest,U32.from_nat(Nat.add(input_base,1n+start)),
          U32.from_nat(Nat.add(weight_base,1n+start))) == first <> _ : List<&2,ReductionLaws.AddressPair>}
      {==}

law kernel_taps_follow_matrix_reduction:
  for +channels: Nat
  for +kernel: Nat
  for +input_base: Nat
  for +weight_base: Nat
  {ReductionLaws.addresses(Nat.mul(channels,Nat.mul(kernel,kernel)),U32.from_nat(input_base),
      U32.from_nat(weight_base)) == matrix_taps(MatrixSpec.kchannels(channels,0n,kernel),kernel,input_base,weight_base) : List<&2,ReductionLaws.AddressPair>}

def kernel_taps_follow_matrix_reduction(channels,kernel,input_base,weight_base):
  +indices = MatrixSpec.kchannels(channels,0n,kernel)
  +count = Nat.mul(channels,Nat.mul(kernel,kernel))
  %Equal.sym(List<&2,ReductionLaws.AddressPair>,matrix_taps(indices,kernel,input_base,weight_base),address_taps(flatten(kernel,indices),input_base,
      weight_base),matrix_taps_flatten(indices,kernel,input_base,weight_base)) : {ReductionLaws.addresses(count,U32.from_nat(input_base),
      U32.from_nat(weight_base)) == _ : List<&2,ReductionLaws.AddressPair>}
  %Equal.sym(List<&2,Nat>,flatten(kernel,indices),Ranges.range(count,0n),matrix_reduction_is_consecutive(channels,
      kernel)) : {ReductionLaws.addresses(count,U32.from_nat(input_base),U32.from_nat(weight_base)) == address_taps(_,input_base,weight_base) : List<&2,
    ReductionLaws.AddressPair>}
  %Addition.zero_right(input_base) : {ReductionLaws.addresses(count,U32.from_nat(_),U32.from_nat(weight_base)) == address_taps(Ranges.range(count,0n),
      input_base,weight_base) : List<&2,ReductionLaws.AddressPair>}
  %Addition.zero_right(weight_base) : {ReductionLaws.addresses(count,U32.from_nat(Nat.add(input_base,0n)),
      U32.from_nat(_)) == address_taps(Ranges.range(count,0n),input_base,weight_base) : List<&2,ReductionLaws.AddressPair>}
  consecutive_taps(count,0n,input_base,weight_base)

