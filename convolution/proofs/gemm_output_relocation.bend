# Relocate an aligned local GEMM write list into a clipped output window.
# Values and write order are preserved; only destination indices change.
import Base
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_rows.bend as Rows
import ../../lib/proofs/kernel_gemm_refinement.bend as GemmLaws
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/u32_extrema.bend as U32Extrema
import ../../lib/proofs/nat_extrema.bend as NatExtrema
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/word_arithmetic.bend as WordArithmetic
import ../../lib/proofs/word_multiplication.bend as Multiplication
import ../../lib/proofs/kernel_output_refinement.bend as Output
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/storage_address_bounds.bend as Address
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ./bounded_write_readback.bend as Readback
import ./storage_capacity.bend as StorageCapacity
import ./gather_write_program.bend as GatherWrites
import ./conditional_write_coverage.bend as Coverage
import ./gemm_output_readback.bend as GemmReadback
import ../../lib/proofs/kernel_gemm_writes.bend as GemmWrites

type Placement is Data:
  Placement{span: Nat,columns: Nat,start: Nat}

def position(placement: Placement,index: U32) -> Nat:
  Placement{span,columns,start} = placement
  Nat.add(start,Nat.mod(U32.to_nat(index),span))

def destination(placement: Placement,+index: U32) -> Nat:
  Placement{+span,columns,start} = placement
  Nat.add(Nat.mul(Nat.div(U32.to_nat(index),span),columns),Nat.add(start,Nat.mod(U32.to_nat(index),span)))

def selected(+placement: Placement,index: U32) -> Bool:
  Placement{span,columns,start} = placement
  Nat.is_lt(position(placement,index),columns)

def relocate(+placement: Placement,writes: List<&2,Writes.Assignment<F32>>) -> List<&2,Writes.Assignment<F32>>:
  match writes:
    case Nil{}: []
    case Writes.Put{+index,value} <> rest:
      GatherWrites.prepend(selected(placement,index),U32.from_nat(destination(placement,index)),value,relocate(placement,rest))

# Distinct retained local addresses remain distinct in the full output.
def injective(+span: Nat,+columns: Nat,+start: Nat,+first: U32,+second: U32,
  +span_positive: {Nat.is_lt(0n,span) == True{} : Bool},
  +first_inside: {Nat.is_lt(Nat.add(start,Nat.mod(U32.to_nat(first),span)),columns) == True{} : Bool},
  +second_inside: {Nat.is_lt(Nat.add(start,Nat.mod(U32.to_nat(second),span)),columns) == True{} : Bool},
  +same: {destination(Placement{span,columns,start},first) == destination(Placement{span,columns,start},second) : Nat}) ->
  {first == second : U32}:
  +first_row = Nat.div(U32.to_nat(first),span)
  +second_row = Nat.div(U32.to_nat(second),span)
  +first_column = Nat.mod(U32.to_nat(first),span)
  +second_column = Nat.mod(U32.to_nat(second),span)
  +positive = Order.le_lt_trans(0n,Nat.add(start,first_column),columns,Order.zero_le(Nat.add(start,first_column)),first_inside)
  +rows_same = Radix.row_unique(first_row,second_row,columns,Nat.add(start,first_column),Nat.add(start,second_column),positive,first_inside,second_inside,same)
  +columns_same = Order.cancel_sum(start,first_column,second_column,
    Radix.col_unique(first_row,second_row,columns,Nat.add(start,first_column),Nat.add(start,second_column),positive,first_inside,second_inside,same))
  Address.equal_values(first,second,
    %Equal.sym(Nat,U32.to_nat(first),Nat.add(Nat.mul(first_row,span),first_column),Radix.division_identity(first,span,span_positive)) : {_ == U32.to_nat(second) : Nat}
    %Equal.sym(Nat,U32.to_nat(second),Nat.add(Nat.mul(second_row,span),second_column),Radix.division_identity(second,span,span_positive)) : {Nat.add(Nat.mul(first_row,span),first_column) == _ : Nat}
    %Equal.sym(Nat,first_row,second_row,rows_same) : {Nat.add(Nat.mul(_,span),first_column) == Nat.add(Nat.mul(second_row,span),second_column) : Nat}
    %Equal.sym(Nat,first_column,second_column,columns_same) : {Nat.add(Nat.mul(second_row,span),_) == Nat.add(Nat.mul(second_row,span),second_column) : Nat}
    {==})

# The bound is needed only for writes that survive clipping.
def Within(+placement: Placement,+capacity: U32,writes: List<&2,Writes.Assignment<F32>>) -> Data:
  match writes:
    case Nil{}: Unit
    case Writes.Put{index,value} <> rest:
      Shape.Both<Boolean.When(selected(placement,index),{Nat.is_lt(destination(placement,index),U32.to_nat(capacity)) == True{} : Bool}),Within(placement,capacity,rest)>

def retained_distinct(+span: Nat,+columns: Nat,+start: Nat,+query: U32,+index: U32,
  +different: {Nat.is_eq(destination(Placement{span,columns,start},query),destination(Placement{span,columns,start},index)) == False{} : Bool},
  +same: Bool,+same_exact: {Nat.is_eq(U32.to_nat(query),U32.to_nat(index)) == same : Bool}) ->
  {Nat.is_eq(destination(Placement{span,columns,start},query),destination(Placement{span,columns,start},index)) == Nat.is_eq(U32.to_nat(query),U32.to_nat(index)) : Bool}:
  match same:
    case False{}: Equal.trans(Bool,Nat.is_eq(destination(Placement{span,columns,start},query),destination(Placement{span,columns,start},index)),False{},Nat.is_eq(U32.to_nat(query),U32.to_nat(index)),different,Equal.sym(Bool,Nat.is_eq(U32.to_nat(query),U32.to_nat(index)),False{},same_exact))
    case True{}:
      +equal = Address.equal_values(query,index,Order.equal_of_is_eq(U32.to_nat(query),U32.to_nat(index),same_exact))
      +impossible =
        {%equal : {Nat.is_eq(destination(Placement{span,columns,start},_),destination(Placement{span,columns,start},index)) == False{} : Bool}
          different : {Nat.is_eq(destination(Placement{span,columns,start},index),destination(Placement{span,columns,start},index)) == False{} : Bool}}
      Empty.absurd({Nat.is_eq(destination(Placement{span,columns,start},query),destination(Placement{span,columns,start},index)) == Nat.is_eq(U32.to_nat(query),U32.to_nat(index)) : Bool},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(destination(Placement{span,columns,start},index),destination(Placement{span,columns,start},index)),True{},
          Equal.sym(Bool,Nat.is_eq(destination(Placement{span,columns,start},index),destination(Placement{span,columns,start},index)),False{},impossible),Order.equal_reflexive(destination(Placement{span,columns,start},index)))))

def natural_comparison(+span: Nat,+columns: Nat,+start: Nat,+query: U32,+index: U32,
  +span_positive: {Nat.is_lt(0n,span) == True{} : Bool},
  +query_selected: {selected(Placement{span,columns,start},query) == True{} : Bool},
  +index_selected: {selected(Placement{span,columns,start},index) == True{} : Bool},
  +same: Bool,+same_exact: {Nat.is_eq(destination(Placement{span,columns,start},query),destination(Placement{span,columns,start},index)) == same : Bool}) ->
  {Nat.is_eq(destination(Placement{span,columns,start},query),destination(Placement{span,columns,start},index)) == Nat.is_eq(U32.to_nat(query),U32.to_nat(index)) : Bool}:
  match same:
    case True{}:
      +equal = injective(span,columns,start,query,index,span_positive,query_selected,index_selected,
        Order.equal_of_is_eq(destination(Placement{span,columns,start},query),destination(Placement{span,columns,start},index),same_exact))
      %Equal.sym(U32,query,index,equal) : {Nat.is_eq(destination(Placement{span,columns,start},_),destination(Placement{span,columns,start},index)) == Nat.is_eq(U32.to_nat(_),U32.to_nat(index)) : Bool}
      Equal.trans(Bool,Nat.is_eq(destination(Placement{span,columns,start},index),destination(Placement{span,columns,start},index)),True{},Nat.is_eq(U32.to_nat(index),U32.to_nat(index)),
        Order.equal_reflexive(destination(Placement{span,columns,start},index)),Equal.sym(Bool,Nat.is_eq(U32.to_nat(index),U32.to_nat(index)),True{},Order.equal_reflexive(U32.to_nat(index))))
    case False{}:
      retained_distinct(span,columns,start,query,index,same_exact,Nat.is_eq(U32.to_nat(query),U32.to_nat(index)),{==})

def retained_comparison(+span: Nat,+columns: Nat,+start: Nat,+query: U32,+index: U32,+capacity: U32,
  +span_positive: {Nat.is_lt(0n,span) == True{} : Bool},+capacity_fits: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +query_selected: {selected(Placement{span,columns,start},query) == True{} : Bool},
  +index_selected: {selected(Placement{span,columns,start},index) == True{} : Bool},
  +index_inside: {Nat.is_lt(destination(Placement{span,columns,start},index),U32.to_nat(capacity)) == True{} : Bool},
  +same: Bool,+same_exact: {Nat.is_eq(destination(Placement{span,columns,start},query),destination(Placement{span,columns,start},index)) == same : Bool}) ->
  {Nat.is_eq(destination(Placement{span,columns,start},query),U32.to_nat(U32.from_nat(destination(Placement{span,columns,start},index)))) == Nat.is_eq(U32.to_nat(query),U32.to_nat(index)) : Bool}:
  +placement = {Placement{span,columns,start} : Placement}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(destination(placement,index))),destination(placement,index),
    Bounds.roundtrip(destination(placement,index),capacity,capacity_fits,Bounds.strict_le(destination(placement,index),U32.to_nat(capacity),index_inside))) :
    {Nat.is_eq(destination(placement,query),_) == Nat.is_eq(U32.to_nat(query),U32.to_nat(index)) : Bool}
  natural_comparison(span,columns,start,query,index,span_positive,query_selected,index_selected,same,same_exact)


def dropped_distinct(+placement: Placement,+query: U32,+index: U32,
  +query_selected: {selected(placement,query) == True{} : Bool},+index_dropped: {selected(placement,index) == False{} : Bool},
  +same: Bool,+same_exact: {Nat.is_eq(U32.to_nat(query),U32.to_nat(index)) == same : Bool}) ->
  {Nat.is_eq(U32.to_nat(query),U32.to_nat(index)) == False{} : Bool}:
  match same:
    case False{}: same_exact
    case True{}:
      +equal = Address.equal_values(query,index,Order.equal_of_is_eq(U32.to_nat(query),U32.to_nat(index),same_exact))
      Empty.absurd({Nat.is_eq(U32.to_nat(query),U32.to_nat(index)) == False{} : Bool},
        WordEncoding.false_true(Equal.trans(Bool,False{},selected(placement,index),True{},Equal.sym(Bool,selected(placement,index),False{},index_dropped),
          %equal : {selected(placement,_) == True{} : Bool} query_selected)))

def prepend_comparison(+span: Nat,+columns: Nat,+start: Nat,+query: U32,+index: U32,+value: F32,+rest: List<&2,Writes.Assignment<F32>>,+capacity: U32,
  +span_positive: {Nat.is_lt(0n,span) == True{} : Bool},+capacity_fits: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +query_selected: {selected(Placement{span,columns,start},query) == True{} : Bool},
  +keep: Bool,+keep_exact: {selected(Placement{span,columns,start},index) == keep : Bool},
  index_inside: Boolean.When(keep,{Nat.is_lt(destination(Placement{span,columns,start},index),U32.to_nat(capacity)) == True{} : Bool})) ->
  {GatherWrites.contains(destination(Placement{span,columns,start},query),GatherWrites.prepend(keep,U32.from_nat(destination(Placement{span,columns,start},index)),value,rest)) ==
    Nat.is_eq(U32.to_nat(query),U32.to_nat(index)) || GatherWrites.contains(destination(Placement{span,columns,start},query),rest) : Bool}:
  match keep:
    case False{}:
      %Equal.sym(Bool,Nat.is_eq(U32.to_nat(query),U32.to_nat(index)),False{},
        dropped_distinct(Placement{span,columns,start},query,index,query_selected,keep_exact,Nat.is_eq(U32.to_nat(query),U32.to_nat(index)),{==})) :
        {GatherWrites.contains(destination(Placement{span,columns,start},query),rest) == _ || GatherWrites.contains(destination(Placement{span,columns,start},query),rest) : Bool}
      {==}
    case True{}:
      Equal.cong(Bool,Bool,head => head || GatherWrites.contains(destination(Placement{span,columns,start},query),rest),
        Nat.is_eq(destination(Placement{span,columns,start},query),U32.to_nat(U32.from_nat(destination(Placement{span,columns,start},index)))),Nat.is_eq(U32.to_nat(query),U32.to_nat(index)),
        retained_comparison(span,columns,start,query,index,capacity,span_positive,capacity_fits,query_selected,keep_exact,index_inside,
          Nat.is_eq(destination(Placement{span,columns,start},query),destination(Placement{span,columns,start},index)),{==}))

def coverage(+span: Nat,+columns: Nat,+start: Nat,+query: U32,+writes: List<&2,Writes.Assignment<F32>>,+capacity: U32,
  +span_positive: {Nat.is_lt(0n,span) == True{} : Bool},+capacity_fits: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +query_selected: {selected(Placement{span,columns,start},query) == True{} : Bool},
  bounds: Within(Placement{span,columns,start},capacity,writes)) ->
  {GatherWrites.contains(destination(Placement{span,columns,start},query),relocate(Placement{span,columns,start},writes)) == GatherWrites.contains(U32.to_nat(query),writes) : Bool}:
  match writes:
    case Nil{}: {==}
    case Writes.Put{+index,+value} <> +rest:
      Shape.Both{here,tail} = bounds
      +placement = {Placement{span,columns,start} : Placement}
      %Equal.sym(Bool,GatherWrites.contains(destination(placement,query),GatherWrites.prepend(selected(placement,index),U32.from_nat(destination(placement,index)),value,relocate(placement,rest))),
        Nat.is_eq(U32.to_nat(query),U32.to_nat(index)) || GatherWrites.contains(destination(placement,query),relocate(placement,rest)),
        prepend_comparison(span,columns,start,query,index,value,relocate(placement,rest),capacity,span_positive,capacity_fits,query_selected,selected(placement,index),{==},here)) :
        {_ == GatherWrites.contains(U32.to_nat(query),Writes.Put{index,value} <> rest) : Bool}
      Equal.cong(Bool,Bool,tail => Nat.is_eq(U32.to_nat(query),U32.to_nat(index)) || tail,
        GatherWrites.contains(destination(placement,query),relocate(placement,rest)),GatherWrites.contains(U32.to_nat(query),rest),
        coverage(span,columns,start,query,rest,capacity,span_positive,capacity_fits,query_selected,tail))

def prepend_bounds(+placement: Placement,+capacity: U32,+index: U32,+value: F32,+rest: List<&2,Writes.Assignment<F32>>,
  +capacity_fits: {U32.is_le(capacity,2147483648) == True{} : Bool},+keep: Bool,
  +inside: Boolean.When(keep,{Nat.is_lt(destination(placement,index),U32.to_nat(capacity)) == True{} : Bool}),
  tail: Readback.WritesWithinCapacity(rest,capacity)) ->
  Readback.WritesWithinCapacity(GatherWrites.prepend(keep,U32.from_nat(destination(placement,index)),value,rest),capacity):
  match keep:
    case False{}: tail
    case True{}:
      Shape.Both{Bounds.native_bound(destination(placement,index),capacity,
        Bounds.roundtrip(destination(placement,index),capacity,capacity_fits,Bounds.strict_le(destination(placement,index),U32.to_nat(capacity),inside)),inside),tail}

def bounds(+placement: Placement,+capacity: U32,+writes: List<&2,Writes.Assignment<F32>>,
  +capacity_fits: {U32.is_le(capacity,2147483648) == True{} : Bool},within: Within(placement,capacity,writes)) ->
  Readback.WritesWithinCapacity(relocate(placement,writes),capacity):
  match writes:
    case Nil{}: Unit{}
    case Writes.Put{+index,+value} <> +rest:
      Shape.Both{+here,tail} = within
      prepend_bounds(placement,capacity,index,value,relocate(placement,rest),capacity_fits,selected(placement,index),here,bounds(placement,capacity,rest,capacity_fits,tail))

def prepend_values(+span: Nat,+columns: Nat,+start: Nat,+query: U32,+index: U32,+value: F32,+wanted: F32,+rest: List<&2,Writes.Assignment<F32>>,+capacity: U32,
  +span_positive: {Nat.is_lt(0n,span) == True{} : Bool},+capacity_fits: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +query_selected: {selected(Placement{span,columns,start},query) == True{} : Bool},
  +keep: Bool,+keep_exact: {selected(Placement{span,columns,start},index) == keep : Bool},
  +index_inside: Boolean.When(keep,{Nat.is_lt(destination(Placement{span,columns,start},index),U32.to_nat(capacity)) == True{} : Bool}),
  +correct: Coverage.Matches(Nat.is_eq(U32.to_nat(query),U32.to_nat(index)),value,wanted),
  tail: Coverage.MatchingWritesHaveValue(destination(Placement{span,columns,start},query),wanted,Readback.natural_write_addresses(rest))) ->
  Coverage.MatchingWritesHaveValue(destination(Placement{span,columns,start},query),wanted,Readback.natural_write_addresses(GatherWrites.prepend(keep,U32.from_nat(destination(Placement{span,columns,start},index)),value,rest))):
  match keep:
    case False{}: tail
    case True{}:
      +placement = {Placement{span,columns,start} : Placement}
      Shape.Both{
        %Equal.sym(Bool,Nat.is_eq(destination(placement,query),U32.to_nat(U32.from_nat(destination(placement,index)))),Nat.is_eq(U32.to_nat(query),U32.to_nat(index)),
          retained_comparison(span,columns,start,query,index,capacity,span_positive,capacity_fits,query_selected,keep_exact,index_inside,
            Nat.is_eq(destination(placement,query),destination(placement,index)),{==})) : Coverage.Matches(_,value,wanted)
        correct,tail}

def values(+span: Nat,+columns: Nat,+start: Nat,+query: U32,+wanted: F32,+writes: List<&2,Writes.Assignment<F32>>,+capacity: U32,
  +span_positive: {Nat.is_lt(0n,span) == True{} : Bool},+capacity_fits: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +query_selected: {selected(Placement{span,columns,start},query) == True{} : Bool},
  bounds: Within(Placement{span,columns,start},capacity,writes),
  correct: Coverage.MatchingWritesHaveValue(U32.to_nat(query),wanted,Readback.natural_write_addresses(writes))) ->
  Coverage.MatchingWritesHaveValue(destination(Placement{span,columns,start},query),wanted,Readback.natural_write_addresses(relocate(Placement{span,columns,start},writes))):
  match writes:
    case Nil{}: Unit{}
    case Writes.Put{+index,+value} <> +rest:
      Shape.Both{+inside,tail_bounds} = bounds
      Shape.Both{+same,tail_values} = correct
      +placement = {Placement{span,columns,start} : Placement}
      prepend_values(span,columns,start,query,index,value,wanted,relocate(placement,rest),capacity,span_positive,capacity_fits,query_selected,selected(placement,index),{==},inside,same,
        values(span,columns,start,query,wanted,rest,capacity,span_positive,capacity_fits,query_selected,tail_bounds,tail_values))
def evidence(+span: Nat,+columns: Nat,+start: Nat,+query: U32,+wanted: F32,+writes: List<&2,Writes.Assignment<F32>>,+capacity: U32,
  +span_positive: {Nat.is_lt(0n,span) == True{} : Bool},+capacity_fits: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +query_selected: {selected(Placement{span,columns,start},query) == True{} : Bool},
  +within: Within(Placement{span,columns,start},capacity,writes),
  correct: Coverage.MatchingWritesHaveValue(U32.to_nat(query),wanted,Readback.natural_write_addresses(writes))) ->
  Readback.WriteEvidence<destination(Placement{span,columns,start},query),wanted,capacity,relocate(Placement{span,columns,start},writes)>:
  Readback.WriteEvidence{
    bounds(Placement{span,columns,start},capacity,writes,capacity_fits,within),
    values(span,columns,start,query,wanted,writes,capacity,span_positive,capacity_fits,query_selected,within,correct)}

def retained_bound(+rows: Nat,+span: Nat,+columns: Nat,+start: Nat,+index: U32,+capacity: U32,
  +span_positive: {Nat.is_lt(0n,span) == True{} : Bool},
  +local_inside: {Nat.is_lt(U32.to_nat(index),Nat.mul(rows,span)) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(rows,columns),U32.to_nat(capacity)) == True{} : Bool},
  +keep: Bool,+keep_exact: {selected(Placement{span,columns,start},index) == keep : Bool}) ->
  Boolean.When(keep,{Nat.is_lt(destination(Placement{span,columns,start},index),U32.to_nat(capacity)) == True{} : Bool}):
  match keep:
    case False{}: Unit{}
    case True{}:
      +row = Nat.div(U32.to_nat(index),span)
      +column = Nat.mod(U32.to_nat(index),span)
      Bounds.nat_trans(destination(Placement{span,columns,start},index),Nat.mul(rows,columns),U32.to_nat(capacity),
        RowMajor.radix(row,rows,Nat.add(start,column),columns,
          RowMajor.row_inside(row,rows,span,column,
            %Radix.division_identity(index,span,span_positive) : {Nat.is_lt(_,Nat.mul(rows,span)) == True{} : Bool} local_inside),keep_exact),output_room)

def within_local(+rows: Nat,+span: Nat,+columns: Nat,+start: Nat,+capacity: U32,+local_capacity: U32,+writes: List<&2,Writes.Assignment<F32>>,
  +span_positive: {Nat.is_lt(0n,span) == True{} : Bool},
  +local_exact: {U32.to_nat(local_capacity) == Nat.mul(rows,span) : Nat},
  +output_room: {Nat.is_le(Nat.mul(rows,columns),U32.to_nat(capacity)) == True{} : Bool},
  bounds: Readback.WritesWithinCapacity(writes,local_capacity)) -> Within(Placement{span,columns,start},capacity,writes):
  match writes:
    case Nil{}: Unit{}
    case Writes.Put{+index,value} <> +rest:
      Shape.Both{here,tail} = bounds
      Shape.Both{
        retained_bound(rows,span,columns,start,index,capacity,span_positive,
          %local_exact : {Nat.is_lt(U32.to_nat(index),_) == True{} : Bool}
          Bounds.as_nat_lt(index,local_capacity,here),
          output_room,selected(Placement{span,columns,start},index),{==}),
        within_local(rows,span,columns,start,capacity,local_capacity,rest,span_positive,local_exact,output_room,tail)}

def relocate_prepend(+placement: Placement,+keep: Bool,+index: U32,+value: F32,+left: List<&2,Writes.Assignment<F32>>,+right: List<&2,Writes.Assignment<F32>>) ->
  {GatherWrites.prepend(keep,index,value,List.append(&2,Writes.Assignment<F32>,left,right)) ==
    List.append(&2,Writes.Assignment<F32>,GatherWrites.prepend(keep,index,value,left),right) : List<&2,Writes.Assignment<F32>>}:
  match keep:
    case False{}: {==}
    case True{}: {==}

def concatenate(+placement: Placement,+left: List<&2,Writes.Assignment<F32>>,+right: List<&2,Writes.Assignment<F32>>) ->
  {relocate(placement,List.append(&2,Writes.Assignment<F32>,left,right)) == List.append(&2,Writes.Assignment<F32>,relocate(placement,left),relocate(placement,right)) : List<&2,Writes.Assignment<F32>>}:
  match left:
    case Nil{}: {==}
    case Writes.Put{+index,+value} <> +rest:
      %Equal.sym(List<&2,Writes.Assignment<F32>>,relocate(placement,List.append(&2,Writes.Assignment<F32>,rest,right)),
        List.append(&2,Writes.Assignment<F32>,relocate(placement,rest),relocate(placement,right)),concatenate(placement,rest,right)) :
        {GatherWrites.prepend(selected(placement,index),U32.from_nat(destination(placement,index)),value,_) == List.append(&2,Writes.Assignment<F32>,relocate(placement,left),relocate(placement,right)) : List<&2,Writes.Assignment<F32>>}
      relocate_prepend(placement,selected(placement,index),U32.from_nat(destination(placement,index)),value,relocate(placement,rest),relocate(placement,right))

def coordinate_destination(+span: Nat,+columns: Nat,+start: Nat,+row: Nat,+column: Nat,+index: U32,
  +positive: {Nat.is_lt(0n,span) == True{} : Bool},+inside: {Nat.is_lt(column,span) == True{} : Bool},
  +exact: {U32.to_nat(index) == Nat.add(Nat.mul(row,span),column) : Nat}) ->
  {destination(Placement{span,columns,start},index) == Nat.add(Nat.mul(row,columns),Nat.add(start,column)) : Nat}:
  %Equal.sym(Nat,U32.to_nat(index),Nat.add(Nat.mul(row,span),column),exact) :
    {Nat.add(Nat.mul(Nat.div(_,span),columns),Nat.add(start,Nat.mod(_,span))) == Nat.add(Nat.mul(row,columns),Nat.add(start,column)) : Nat}
  %Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(row,span),column),span),row,Radix.quotient(row,column,span,positive,inside)) :
    {Nat.add(Nat.mul(_,columns),Nat.add(start,Nat.mod(Nat.add(Nat.mul(row,span),column),span))) == Nat.add(Nat.mul(row,columns),Nat.add(start,column)) : Nat}
  %Equal.sym(Nat,Nat.mod(Nat.add(Nat.mul(row,span),column),span),column,Radix.remainder(row,column,span,positive,inside)) :
    {Nat.add(Nat.mul(row,columns),Nat.add(start,_)) == Nat.add(Nat.mul(row,columns),Nat.add(start,column)) : Nat}
  {==}

def coordinate_selected(+span: Nat,+columns: Nat,+start: Nat,+row: Nat,+column: Nat,+index: U32,
  +positive: {Nat.is_lt(0n,span) == True{} : Bool},+inside: {Nat.is_lt(column,span) == True{} : Bool},
  +exact: {U32.to_nat(index) == Nat.add(Nat.mul(row,span),column) : Nat}) ->
  {selected(Placement{span,columns,start},index) == Nat.is_lt(Nat.add(start,column),columns) : Bool}:
  %Equal.sym(Nat,U32.to_nat(index),Nat.add(Nat.mul(row,span),column),exact) :
    {Nat.is_lt(Nat.add(start,Nat.mod(_,span)),columns) == Nat.is_lt(Nat.add(start,column),columns) : Bool}
  %Equal.sym(Nat,Nat.mod(Nat.add(Nat.mul(row,span),column),span),column,Radix.remainder(row,column,span,positive,inside)) :
    {Nat.is_lt(Nat.add(start,_),columns) == Nat.is_lt(Nat.add(start,column),columns) : Bool}
  {==}

# The final row store clips its eight lanes by the remaining global columns.
def clipped_lane(+lane: U32,+position: U32,+columns: U32,+limit: U32,
  +lane_small: {Nat.is_lt(U32.to_nat(lane),8n) == True{} : Bool},
  +before: {Nat.is_le(U32.to_nat(position),U32.to_nat(columns)) == True{} : Bool},
  +columns_room: {Nat.is_le(U32.to_nat(columns),U32.to_nat(limit)) == True{} : Bool},
  +limit_fits: {U32.is_le(limit,2147483648) == True{} : Bool}) ->
  {U32.is_lt(lane,U32.min(8,(columns - position : U32))) == Nat.is_lt(Nat.add(U32.to_nat(position),U32.to_nat(lane)),U32.to_nat(columns)) : Bool}:
  %Equal.sym(Bool,U32.is_lt(lane,U32.min(8,(columns - position : U32))),Nat.is_lt(U32.to_nat(lane),U32.to_nat(U32.min(8,(columns - position : U32)))),Comparison.lt(lane,U32.min(8,(columns - position : U32)))) :
    {_ == Nat.is_lt(Nat.add(U32.to_nat(position),U32.to_nat(lane)),U32.to_nat(columns)) : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.min(8,(columns - position : U32))),Nat.min(8n,U32.to_nat((columns - position : U32))),U32Extrema.minimum(8,(columns - position : U32))) :
    {Nat.is_lt(U32.to_nat(lane),_) == Nat.is_lt(Nat.add(U32.to_nat(position),U32.to_nat(lane)),U32.to_nat(columns)) : Bool}
  %Equal.sym(Nat,U32.to_nat((columns - position : U32)),Nat.sub(U32.to_nat(columns),U32.to_nat(position)),
    Arithmetic.subtract(columns,position,limit,limit_fits,before,Order.transitive(Nat.sub(U32.to_nat(columns),U32.to_nat(position)),U32.to_nat(columns),U32.to_nat(limit),Order.sub_le(U32.to_nat(columns),U32.to_nat(position)),columns_room))) :
    {Nat.is_lt(U32.to_nat(lane),Nat.min(8n,_)) == Nat.is_lt(Nat.add(U32.to_nat(position),U32.to_nat(lane)),U32.to_nat(columns)) : Bool}
  %Equal.sym(Bool,Nat.is_lt(U32.to_nat(lane),Nat.min(8n,Nat.sub(U32.to_nat(columns),U32.to_nat(position)))),Nat.is_lt(U32.to_nat(lane),8n) && Nat.is_lt(U32.to_nat(lane),Nat.sub(U32.to_nat(columns),U32.to_nat(position))),NatExtrema.below_minimum(U32.to_nat(lane),8n,Nat.sub(U32.to_nat(columns),U32.to_nat(position)))) :
    {_ == Nat.is_lt(Nat.add(U32.to_nat(position),U32.to_nat(lane)),U32.to_nat(columns)) : Bool}
  %Equal.sym(Bool,Nat.is_lt(U32.to_nat(lane),8n),True{},lane_small) :
    {_ && Nat.is_lt(U32.to_nat(lane),Nat.sub(U32.to_nat(columns),U32.to_nat(position))) == Nat.is_lt(Nat.add(U32.to_nat(position),U32.to_nat(lane)),U32.to_nat(columns)) : Bool}
  Order.below_difference(U32.to_nat(position),U32.to_nat(lane),U32.to_nat(columns),before)

# Encoded address arithmetic is modular; bounds are proved separately for
# retained writes before any storage readback theorem is applied.
def encoded_row_lane(+row: U32,+columns: U32,+position: U32,+lane: U32) ->
  {(row * columns + position + lane : U32) == U32.from_nat(Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(columns)),Nat.add(U32.to_nat(position),U32.to_nat(lane)))) : U32}:
  %Equal.sym(U32,(row * columns : U32),U32.from_nat(Nat.mul(U32.to_nat(row),U32.to_nat(columns))),Multiplication.native_multiply(row,columns)) :
    {(_ + position + lane : U32) == U32.from_nat(Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(columns)),Nat.add(U32.to_nat(position),U32.to_nat(lane)))) : U32}
  %Arithmetic.native_roundtrip(position) :
    {(U32.from_nat(Nat.mul(U32.to_nat(row),U32.to_nat(columns))) + _ + lane : U32) == U32.from_nat(Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(columns)),Nat.add(U32.to_nat(position),U32.to_nat(lane)))) : U32}
  %Arithmetic.native_roundtrip(lane) :
    {(U32.from_nat(Nat.mul(U32.to_nat(row),U32.to_nat(columns))) + U32.from_nat(U32.to_nat(position)) + _ : U32) == U32.from_nat(Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(columns)),Nat.add(U32.to_nat(position),U32.to_nat(lane)))) : U32}
  %Equal.sym(U32,(U32.from_nat(Nat.mul(U32.to_nat(row),U32.to_nat(columns))) + U32.from_nat(U32.to_nat(position)) : U32),U32.from_nat(Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(columns)),U32.to_nat(position))),WordArithmetic.native_add(Nat.mul(U32.to_nat(row),U32.to_nat(columns)),U32.to_nat(position))) :
    {(_ + U32.from_nat(U32.to_nat(lane)) : U32) == U32.from_nat(Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(columns)),Nat.add(U32.to_nat(position),U32.to_nat(lane)))) : U32}
  %Addition.associative(Nat.mul(U32.to_nat(row),U32.to_nat(columns)),U32.to_nat(position),U32.to_nat(lane)) :
    {(U32.from_nat(Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(columns)),U32.to_nat(position))) + U32.from_nat(U32.to_nat(lane)) : U32) == U32.from_nat(_) : U32}
  WordArithmetic.native_add(Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(columns)),U32.to_nat(position)),U32.to_nat(lane))

def single_write(+placement: Placement,+source: U32,+target: U32,+value: F32,+keep: Bool,
  +same_address: {U32.from_nat(destination(placement,source)) == target : U32},
  +same_keep: {selected(placement,source) == keep : Bool}) ->
  {relocate(placement,[Writes.Put{source,value}]) == Output.lane_writes(keep,target,value) : List<&2,Writes.Assignment<F32>>}:
  match keep:
    case False{}:
      %Equal.sym(Bool,selected(placement,source),False{},same_keep) : {GatherWrites.prepend(_,U32.from_nat(destination(placement,source)),value,[]) == [] : List<&2,Writes.Assignment<F32>>}
      {==}
    case True{}:
      %Equal.sym(Bool,selected(placement,source),True{},same_keep) : {GatherWrites.prepend(_,U32.from_nat(destination(placement,source)),value,[]) == [Writes.Put{target,value}] : List<&2,Writes.Assignment<F32>>}
      %Equal.sym(U32,U32.from_nat(destination(placement,source)),target,same_address) : {[Writes.Put{_,value}] == [Writes.Put{target,value}] : List<&2,Writes.Assignment<F32>>}
      {==}

# Connect a concrete local GEMM store to the corresponding clipped store.
def shifted_lane(+query: GemmReadback.Query,+unit: Nat,+row: Nat,+group: Nat,+lane: Nat,
  +row_word: U32,+local_position: U32,+global_position: U32,+start: U32,+columns: U32,+value: F32,
  +row_exact: {U32.to_nat(row_word) == GemmReadback.row_number(unit,row) : Nat},
  +local_exact: {U32.to_nat(local_position) == Nat.mul(group,8n) : Nat},
  +global_exact: {U32.to_nat(global_position) == Nat.add(U32.to_nat(start),Nat.mul(group,8n)) : Nat},
  +row_inside: {Nat.is_lt(GemmReadback.row_number(unit,row),U32.to_nat(GemmReadback.rows(query))) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,GemmReadback.groups(query)) == True{} : Bool},
  +lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +before: {Nat.is_le(U32.to_nat(global_position),U32.to_nat(columns)) == True{} : Bool},
  +columns_room: {Nat.is_le(U32.to_nat(columns),U32.to_nat(GemmReadback.limit(query))) == True{} : Bool}) ->
  {relocate(Placement{U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start)},
      [Writes.Put{(row_word * GemmReadback.columns(query) + local_position + U32.from_nat(lane) : U32),value}]) ==
    Output.lane_writes(U32.is_lt(U32.from_nat(lane),U32.min(8,(columns - global_position : U32))),(row_word * columns + global_position + U32.from_nat(lane) : U32),value) : List<&2,Writes.Assignment<F32>>}:
  +span = U32.to_nat(GemmReadback.columns(query))
  +placement = {Placement{span,U32.to_nat(columns),U32.to_nat(start)} : Placement}
  +local_index = (row_word * GemmReadback.columns(query) + local_position + U32.from_nat(lane) : U32)
  +lane_exact = Bounds.roundtrip(lane,8,{==},Bounds.strict_le(lane,8n,lane_small))
  +local_column = GemmReadback.lane_number(group,lane)
  +column_inside = GemmReadback.lane_inside(query,group,lane,group_inside,lane_small)
  +local_index_exact = GemmReadback.lane_offset(query,unit,row,group,lane,(row_word * GemmReadback.columns(query) + local_position : U32),
    GemmReadback.offset_exact(query,unit,row,group,row_word,local_position,row_exact,local_exact,row_inside,group_inside),row_inside,group_inside,lane_small)
  +shift_exact =
    {%Equal.sym(Nat,U32.to_nat(global_position),Nat.add(U32.to_nat(start),Nat.mul(group,8n)),global_exact) :
      {Nat.add(_,U32.to_nat(U32.from_nat(lane))) == Nat.add(U32.to_nat(start),local_column) : Nat}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(lane)),lane,lane_exact) :
      {Nat.add(Nat.add(U32.to_nat(start),Nat.mul(group,8n)),_) == Nat.add(U32.to_nat(start),local_column) : Nat}
    Addition.associative(U32.to_nat(start),Nat.mul(group,8n),lane) :
      {Nat.add(U32.to_nat(global_position),U32.to_nat(U32.from_nat(lane))) == Nat.add(U32.to_nat(start),local_column) : Nat}}
  single_write(placement,local_index,(row_word * columns + global_position + U32.from_nat(lane) : U32),value,
    U32.is_lt(U32.from_nat(lane),U32.min(8,(columns - global_position : U32))),
    %Equal.sym(Nat,destination(placement,local_index),Nat.add(Nat.mul(GemmReadback.row_number(unit,row),U32.to_nat(columns)),Nat.add(U32.to_nat(start),local_column)),
      coordinate_destination(span,U32.to_nat(columns),U32.to_nat(start),GemmReadback.row_number(unit,row),local_column,local_index,GemmReadback.columns_positive(query),column_inside,local_index_exact)) :
      {U32.from_nat(_) == (row_word * columns + global_position + U32.from_nat(lane) : U32) : U32}
    %row_exact : {U32.from_nat(Nat.add(Nat.mul(_,U32.to_nat(columns)),Nat.add(U32.to_nat(start),local_column))) == (row_word * columns + global_position + U32.from_nat(lane) : U32) : U32}
    %shift_exact : {U32.from_nat(Nat.add(Nat.mul(U32.to_nat(row_word),U32.to_nat(columns)),_)) == (row_word * columns + global_position + U32.from_nat(lane) : U32) : U32}
    Equal.sym(U32,(row_word * columns + global_position + U32.from_nat(lane) : U32),
      U32.from_nat(Nat.add(Nat.mul(U32.to_nat(row_word),U32.to_nat(columns)),Nat.add(U32.to_nat(global_position),U32.to_nat(U32.from_nat(lane))))),
      encoded_row_lane(row_word,columns,global_position,U32.from_nat(lane))),
    %Equal.sym(Bool,selected(placement,local_index),Nat.is_lt(Nat.add(U32.to_nat(start),local_column),U32.to_nat(columns)),
      coordinate_selected(span,U32.to_nat(columns),U32.to_nat(start),GemmReadback.row_number(unit,row),local_column,local_index,GemmReadback.columns_positive(query),column_inside,local_index_exact)) :
      {_ == U32.is_lt(U32.from_nat(lane),U32.min(8,(columns - global_position : U32))) : Bool}
    %shift_exact : {Nat.is_lt(_,U32.to_nat(columns)) == U32.is_lt(U32.from_nat(lane),U32.min(8,(columns - global_position : U32))) : Bool}
    Equal.sym(Bool,U32.is_lt(U32.from_nat(lane),U32.min(8,(columns - global_position : U32))),Nat.is_lt(Nat.add(U32.to_nat(global_position),U32.to_nat(U32.from_nat(lane))),U32.to_nat(columns)),
      clipped_lane(U32.from_nat(lane),global_position,columns,GemmReadback.limit(query),
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(lane)),lane,lane_exact) : {Nat.is_lt(_,8n) == True{} : Bool} lane_small,
        before,columns_room,GemmReadback.limit_fits(query))))

def initial_relocated(+placement: Placement,+source: U32,+target: U32,+value: F32,+keep: Bool,
  here: {relocate(placement,[Writes.Put{(source + 0 : U32),value}]) == Output.lane_writes(keep,(target + 0 : U32),value) : List<&2,Writes.Assignment<F32>>}) ->
  {relocate(placement,[Writes.Put{source,value}]) == Output.lane_writes(keep,target,value) : List<&2,Writes.Assignment<F32>>}:
  %Arithmetic.add_zero(source) : {relocate(placement,[Writes.Put{_,value}]) == Output.lane_writes(keep,target,value) : List<&2,Writes.Assignment<F32>>}
  %Arithmetic.add_zero(target) : {relocate(placement,[Writes.Put{(source + 0 : U32),value}]) == Output.lane_writes(keep,_,value) : List<&2,Writes.Assignment<F32>>}
  here

def next_relocated(+placement: Placement,+previous: List<&2,Writes.Assignment<F32>>,+mapped: List<&2,Writes.Assignment<F32>>,
  +source: U32,+target: U32,+value: F32,+keep: Bool,
  +prefix: {relocate(placement,previous) == mapped : List<&2,Writes.Assignment<F32>>},
  +here: {relocate(placement,[Writes.Put{source,value}]) == Output.lane_writes(keep,target,value) : List<&2,Writes.Assignment<F32>>}) ->
  {relocate(placement,List.append(&2,Writes.Assignment<F32>,previous,[Writes.Put{source,value}])) == Output.append_lane(keep,mapped,target,value) : List<&2,Writes.Assignment<F32>>}:
  %Equal.sym(List<&2,Writes.Assignment<F32>>,relocate(placement,List.append(&2,Writes.Assignment<F32>,previous,[Writes.Put{source,value}])),
    List.append(&2,Writes.Assignment<F32>,relocate(placement,previous),relocate(placement,[Writes.Put{source,value}])),concatenate(placement,previous,[Writes.Put{source,value}])) :
    {_ == Output.append_lane(keep,mapped,target,value) : List<&2,Writes.Assignment<F32>>}
  %Equal.sym(List<&2,Writes.Assignment<F32>>,relocate(placement,previous),mapped,prefix) :
    {List.append(&2,Writes.Assignment<F32>,_,relocate(placement,[Writes.Put{source,value}])) == Output.append_lane(keep,mapped,target,value) : List<&2,Writes.Assignment<F32>>}
  %Equal.sym(List<&2,Writes.Assignment<F32>>,relocate(placement,[Writes.Put{source,value}]),Output.lane_writes(keep,target,value),here) :
    {List.append(&2,Writes.Assignment<F32>,mapped,_) == Output.append_lane(keep,mapped,target,value) : List<&2,Writes.Assignment<F32>>}
  {==}

def tail_relocated(+placement: Placement,+source: U32,+target: U32,+count: U32,+bias: F32,+accumulator: Tile.Tile8,
  +lane0: {relocate(placement,[Writes.Put{(source + 0 : U32),(Tile.lane0(accumulator) + bias : F32)}]) == Output.lane_writes(U32.is_lt(0,count),(target + 0 : U32),(Tile.lane0(accumulator) + bias : F32)) : List<&2,Writes.Assignment<F32>>},
  +lane1: {relocate(placement,[Writes.Put{(source + 1 : U32),(Tile.lane1(accumulator) + bias : F32)}]) == Output.lane_writes(U32.is_lt(1,count),(target + 1 : U32),(Tile.lane1(accumulator) + bias : F32)) : List<&2,Writes.Assignment<F32>>},
  +lane2: {relocate(placement,[Writes.Put{(source + 2 : U32),(Tile.lane2(accumulator) + bias : F32)}]) == Output.lane_writes(U32.is_lt(2,count),(target + 2 : U32),(Tile.lane2(accumulator) + bias : F32)) : List<&2,Writes.Assignment<F32>>},
  +lane3: {relocate(placement,[Writes.Put{(source + 3 : U32),(Tile.lane3(accumulator) + bias : F32)}]) == Output.lane_writes(U32.is_lt(3,count),(target + 3 : U32),(Tile.lane3(accumulator) + bias : F32)) : List<&2,Writes.Assignment<F32>>},
  +lane4: {relocate(placement,[Writes.Put{(source + 4 : U32),(Tile.lane4(accumulator) + bias : F32)}]) == Output.lane_writes(U32.is_lt(4,count),(target + 4 : U32),(Tile.lane4(accumulator) + bias : F32)) : List<&2,Writes.Assignment<F32>>},
  +lane5: {relocate(placement,[Writes.Put{(source + 5 : U32),(Tile.lane5(accumulator) + bias : F32)}]) == Output.lane_writes(U32.is_lt(5,count),(target + 5 : U32),(Tile.lane5(accumulator) + bias : F32)) : List<&2,Writes.Assignment<F32>>},
  +lane6: {relocate(placement,[Writes.Put{(source + 6 : U32),(Tile.lane6(accumulator) + bias : F32)}]) == Output.lane_writes(U32.is_lt(6,count),(target + 6 : U32),(Tile.lane6(accumulator) + bias : F32)) : List<&2,Writes.Assignment<F32>>},
  +lane7: {relocate(placement,[Writes.Put{(source + 7 : U32),(Tile.lane7(accumulator) + bias : F32)}]) == Output.lane_writes(U32.is_lt(7,count),(target + 7 : U32),(Tile.lane7(accumulator) + bias : F32)) : List<&2,Writes.Assignment<F32>>}) ->
  {relocate(placement,Output.tail_assignments(8,source,bias,accumulator)) == Output.tail_assignments(count,target,bias,accumulator) : List<&2,Writes.Assignment<F32>>}:
  +v0 = (Tile.lane0(accumulator) + bias : F32)
  +v1 = (Tile.lane1(accumulator) + bias : F32)
  +v2 = (Tile.lane2(accumulator) + bias : F32)
  +v3 = (Tile.lane3(accumulator) + bias : F32)
  +v4 = (Tile.lane4(accumulator) + bias : F32)
  +v5 = (Tile.lane5(accumulator) + bias : F32)
  +v6 = (Tile.lane6(accumulator) + bias : F32)
  +v7 = (Tile.lane7(accumulator) + bias : F32)
  +a0 = {[Writes.Put{source,v0}] : List<&2,Writes.Assignment<F32>>}
  +t0 = Output.lane_writes(U32.is_lt(0,count),target,v0)
  +a1 = List.append(&2,Writes.Assignment<F32>,a0,[Writes.Put{(source + 1 : U32),v1}])
  +t1 = Output.append_lane(U32.is_lt(1,count),t0,(target + 1 : U32),v1)
  +a2 = List.append(&2,Writes.Assignment<F32>,a1,[Writes.Put{(source + 2 : U32),v2}])
  +t2 = Output.append_lane(U32.is_lt(2,count),t1,(target + 2 : U32),v2)
  +a3 = List.append(&2,Writes.Assignment<F32>,a2,[Writes.Put{(source + 3 : U32),v3}])
  +t3 = Output.append_lane(U32.is_lt(3,count),t2,(target + 3 : U32),v3)
  +a4 = List.append(&2,Writes.Assignment<F32>,a3,[Writes.Put{(source + 4 : U32),v4}])
  +t4 = Output.append_lane(U32.is_lt(4,count),t3,(target + 4 : U32),v4)
  +a5 = List.append(&2,Writes.Assignment<F32>,a4,[Writes.Put{(source + 5 : U32),v5}])
  +t5 = Output.append_lane(U32.is_lt(5,count),t4,(target + 5 : U32),v5)
  +a6 = List.append(&2,Writes.Assignment<F32>,a5,[Writes.Put{(source + 6 : U32),v6}])
  +t6 = Output.append_lane(U32.is_lt(6,count),t5,(target + 6 : U32),v6)
  next_relocated(placement,a6,t6,(source + 7 : U32),(target + 7 : U32),v7,U32.is_lt(7,count),
    next_relocated(placement,a5,t5,(source + 6 : U32),(target + 6 : U32),v6,U32.is_lt(6,count),
    next_relocated(placement,a4,t4,(source + 5 : U32),(target + 5 : U32),v5,U32.is_lt(5,count),
    next_relocated(placement,a3,t3,(source + 4 : U32),(target + 4 : U32),v4,U32.is_lt(4,count),
    next_relocated(placement,a2,t2,(source + 3 : U32),(target + 3 : U32),v3,U32.is_lt(3,count),
    next_relocated(placement,a1,t1,(source + 2 : U32),(target + 2 : U32),v2,U32.is_lt(2,count),
    next_relocated(placement,a0,t0,(source + 1 : U32),(target + 1 : U32),v1,U32.is_lt(1,count),
    initial_relocated(placement,source,target,v0,U32.is_lt(0,count),lane0),lane1),lane2),lane3),lane4),lane5),lane6),lane7)

def row_relocated(+query: GemmReadback.Query,+unit: Nat,+row: Nat,+group: Nat,
  +row_word: U32,+local_position: U32,+global_position: U32,+start: U32,+columns: U32,+accumulator: Tile.Tile8,+bias: Tree.Tree<F32>,
  +row_exact: {U32.to_nat(row_word) == GemmReadback.row_number(unit,row) : Nat},
  +local_exact: {U32.to_nat(local_position) == Nat.mul(group,8n) : Nat},
  +global_exact: {U32.to_nat(global_position) == Nat.add(U32.to_nat(start),Nat.mul(group,8n)) : Nat},
  +group_inside: {Nat.is_lt(group,GemmReadback.groups(query)) == True{} : Bool},
  +before: {Nat.is_le(U32.to_nat(global_position),U32.to_nat(columns)) == True{} : Bool},
  +columns_room: {Nat.is_le(U32.to_nat(columns),U32.to_nat(GemmReadback.limit(query))) == True{} : Bool},
  +valid: Bool,+valid_exact: {U32.is_lt(row_word,GemmReadback.rows(query)) == valid : Bool}) ->
  {relocate(Placement{U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start)},
      GemmWrites.row_writes(valid,row_word,local_position,GemmReadback.columns(query),accumulator,bias)) ==
    GemmWrites.row_writes(valid,row_word,global_position,columns,accumulator,bias) : List<&2,Writes.Assignment<F32>>}:
  match valid:
    case False{}: {==}
    case True{}:
      +inside = GemmReadback.row_inside(query,unit,row,row_word,row_exact,valid_exact)
      +bias_value = Tree.read(F32,bias,row_word)
      %Equal.sym(U32,U32.min(8,(GemmReadback.columns(query) - local_position : U32)),8,GemmReadback.complete_group_count(query,group,local_position,local_exact,group_inside)) :
        {relocate(Placement{U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start)},
          Output.tail_assignments(_,(row_word * GemmReadback.columns(query) + local_position : U32),bias_value,accumulator)) ==
          Output.tail_assignments(U32.min(8,(columns - global_position : U32)),(row_word * columns + global_position : U32),bias_value,accumulator) : List<&2,Writes.Assignment<F32>>}
      tail_relocated(Placement{U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start)},
        (row_word * GemmReadback.columns(query) + local_position : U32),(row_word * columns + global_position : U32),U32.min(8,(columns - global_position : U32)),bias_value,accumulator,
        shifted_lane(query,unit,row,group,0n,row_word,local_position,global_position,start,columns,(Tile.lane0(accumulator) + bias_value : F32),row_exact,local_exact,global_exact,inside,group_inside,{==},before,columns_room),
        shifted_lane(query,unit,row,group,1n,row_word,local_position,global_position,start,columns,(Tile.lane1(accumulator) + bias_value : F32),row_exact,local_exact,global_exact,inside,group_inside,{==},before,columns_room),
        shifted_lane(query,unit,row,group,2n,row_word,local_position,global_position,start,columns,(Tile.lane2(accumulator) + bias_value : F32),row_exact,local_exact,global_exact,inside,group_inside,{==},before,columns_room),
        shifted_lane(query,unit,row,group,3n,row_word,local_position,global_position,start,columns,(Tile.lane3(accumulator) + bias_value : F32),row_exact,local_exact,global_exact,inside,group_inside,{==},before,columns_room),
        shifted_lane(query,unit,row,group,4n,row_word,local_position,global_position,start,columns,(Tile.lane4(accumulator) + bias_value : F32),row_exact,local_exact,global_exact,inside,group_inside,{==},before,columns_room),
        shifted_lane(query,unit,row,group,5n,row_word,local_position,global_position,start,columns,(Tile.lane5(accumulator) + bias_value : F32),row_exact,local_exact,global_exact,inside,group_inside,{==},before,columns_room),
        shifted_lane(query,unit,row,group,6n,row_word,local_position,global_position,start,columns,(Tile.lane6(accumulator) + bias_value : F32),row_exact,local_exact,global_exact,inside,group_inside,{==},before,columns_room),
        shifted_lane(query,unit,row,group,7n,row_word,local_position,global_position,start,columns,(Tile.lane7(accumulator) + bias_value : F32),row_exact,local_exact,global_exact,inside,group_inside,{==},before,columns_room))

# The last aligned group begins before the real output end, even when its
# final lanes are padding.
def group_before_output(+query: GemmReadback.Query,+start: U32,+columns: U32,+group: Nat,
  +group_inside: {Nat.is_lt(group,GemmReadback.groups(query)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(GemmReadback.columns(query))),Nat.add(U32.to_nat(columns),7n)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat(start),Nat.mul(group,8n)),U32.to_nat(columns)) == True{} : Bool}:
  +position = Nat.add(U32.to_nat(start),Nat.mul(group,8n))
  +next_room = Order.offset_le(U32.to_nat(start),Nat.mul(1n+group,8n),Nat.mul(GemmReadback.groups(query),8n),
    RowMajor.mul_le(1n+group,GemmReadback.groups(query),8n,Order.strict_follow(group,GemmReadback.groups(query),group_inside)))
  +room = Order.transitive(Nat.add(U32.to_nat(start),Nat.mul(1n+group,8n)),Nat.add(U32.to_nat(start),Nat.mul(GemmReadback.groups(query),8n)),Nat.add(U32.to_nat(columns),7n),next_room,
    %GemmReadback.columns_exact(query) : {Nat.is_le(Nat.add(U32.to_nat(start),_),Nat.add(U32.to_nat(columns),7n)) == True{} : Bool} end_room)
  Bounds.predecessor_strict(position,U32.to_nat(columns),
    Order.cancel_offset_le(7n,1n+position,U32.to_nat(columns),
      %RowMajor.add_swap(U32.to_nat(start),8n,Nat.mul(group,8n)) : {Nat.is_le(_,Nat.add(7n,U32.to_nat(columns))) == True{} : Bool}
      %Order.commutative(U32.to_nat(columns),7n) : {Nat.is_le(Nat.add(U32.to_nat(start),Nat.mul(1n+group,8n)),_) == True{} : Bool}
      room))

def shifted_position(+query: GemmReadback.Query,+start: U32,+columns: U32,+group: Nat,+local_position: U32,
  +local_exact: {U32.to_nat(local_position) == Nat.mul(group,8n) : Nat},
  +group_inside: {Nat.is_lt(group,GemmReadback.groups(query)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(GemmReadback.columns(query))),Nat.add(U32.to_nat(columns),7n)) == True{} : Bool},
  +columns_room: {Nat.is_le(U32.to_nat(columns),U32.to_nat(GemmReadback.limit(query))) == True{} : Bool}) ->
  {U32.to_nat((start + local_position : U32)) == Nat.add(U32.to_nat(start),Nat.mul(group,8n)) : Nat}:
  %local_exact : {U32.to_nat((start + local_position : U32)) == Nat.add(U32.to_nat(start),_) : Nat}
  Arithmetic.add(start,local_position,GemmReadback.limit(query),GemmReadback.limit_fits(query),
    %Equal.sym(Nat,U32.to_nat(local_position),Nat.mul(group,8n),local_exact) : {Nat.is_le(Nat.add(U32.to_nat(start),_),U32.to_nat(GemmReadback.limit(query))) == True{} : Bool}
    Order.transitive(Nat.add(U32.to_nat(start),Nat.mul(group,8n)),U32.to_nat(columns),U32.to_nat(GemmReadback.limit(query)),
      Bounds.strict_le(Nat.add(U32.to_nat(start),Nat.mul(group,8n)),U32.to_nat(columns),group_before_output(query,start,columns,group,group_inside,end_room)),columns_room))

def concatenate_relocated(+placement: Placement,+left: List<&2,Writes.Assignment<F32>>,+right: List<&2,Writes.Assignment<F32>>,
  +mapped_left: List<&2,Writes.Assignment<F32>>,+mapped_right: List<&2,Writes.Assignment<F32>>,
  +left_equal: {relocate(placement,left) == mapped_left : List<&2,Writes.Assignment<F32>>},
  +right_equal: {relocate(placement,right) == mapped_right : List<&2,Writes.Assignment<F32>>}) ->
  {relocate(placement,List.append(&2,Writes.Assignment<F32>,left,right)) == List.append(&2,Writes.Assignment<F32>,mapped_left,mapped_right) : List<&2,Writes.Assignment<F32>>}:
  %Equal.sym(List<&2,Writes.Assignment<F32>>,relocate(placement,List.append(&2,Writes.Assignment<F32>,left,right)),List.append(&2,Writes.Assignment<F32>,relocate(placement,left),relocate(placement,right)),concatenate(placement,left,right)) :
    {_ == List.append(&2,Writes.Assignment<F32>,mapped_left,mapped_right) : List<&2,Writes.Assignment<F32>>}
  %Equal.sym(List<&2,Writes.Assignment<F32>>,relocate(placement,left),mapped_left,left_equal) :
    {List.append(&2,Writes.Assignment<F32>,_,relocate(placement,right)) == List.append(&2,Writes.Assignment<F32>,mapped_left,mapped_right) : List<&2,Writes.Assignment<F32>>}
  %Equal.sym(List<&2,Writes.Assignment<F32>>,relocate(placement,right),mapped_right,right_equal) :
    {List.append(&2,Writes.Assignment<F32>,mapped_left,_) == List.append(&2,Writes.Assignment<F32>,mapped_left,mapped_right) : List<&2,Writes.Assignment<F32>>}
  {==}

def job_context(+query: GemmReadback.Query,+unit: Nat,+group: Nat,+start: U32,+columns: U32) ->
  {GemmWrites.job_writes(Gemm.job(Gemm.Layout{GemmReadback.rows(query),GemmReadback.columns(query),GemmReadback.inner(query),start,columns},U32.from_nat(GemmReadback.job_index(unit,group,GemmReadback.units(query)))),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)) ==
    GemmWrites.context_writes(Gemm.Context{(GemmReadback.unit_word(query,unit,group) * 4 : U32),(start + GemmReadback.group_word(query,unit,group) * 8 : U32),GemmReadback.rows(query),columns,
      (GemmReadback.group_word(query,unit,group) * GemmReadback.inner(query) : U32),(GemmReadback.unit_word(query,unit,group) / 2 * GemmReadback.inner(query) : U32)},GemmReadback.unit_accumulators(query,unit,group),GemmReadback.bias(query)) : List<&2,Writes.Assignment<F32>>}:
  GemmReadback.Query{rows,span,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  {==}

def local_job_context(+query: GemmReadback.Query,+unit: Nat,+group: Nat) ->
  {GemmReadback.job_writes(query,unit,group) ==
    GemmWrites.context_writes(Gemm.Context{(GemmReadback.unit_word(query,unit,group) * 4 : U32),(GemmReadback.group_word(query,unit,group) * 8 : U32),GemmReadback.rows(query),GemmReadback.columns(query),
      (GemmReadback.group_word(query,unit,group) * GemmReadback.inner(query) : U32),(GemmReadback.unit_word(query,unit,group) / 2 * GemmReadback.inner(query) : U32)},GemmReadback.unit_accumulators(query,unit,group),GemmReadback.bias(query)) : List<&2,Writes.Assignment<F32>>}:
  %Equal.sym(Gemm.Layout,GemmReadback.layout(query),Gemm.Layout{GemmReadback.rows(query),GemmReadback.columns(query),GemmReadback.inner(query),0,GemmReadback.columns(query)},GemmReadback.layout_exact(query)) :
    {GemmWrites.job_writes(Gemm.job(_,U32.from_nat(GemmReadback.job_index(unit,group,GemmReadback.units(query)))),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)) ==
      GemmWrites.context_writes(Gemm.Context{(GemmReadback.unit_word(query,unit,group) * 4 : U32),(GemmReadback.group_word(query,unit,group) * 8 : U32),GemmReadback.rows(query),GemmReadback.columns(query),
        (GemmReadback.group_word(query,unit,group) * GemmReadback.inner(query) : U32),(GemmReadback.unit_word(query,unit,group) / 2 * GemmReadback.inner(query) : U32)},GemmReadback.unit_accumulators(query,unit,group),GemmReadback.bias(query)) : List<&2,Writes.Assignment<F32>>}
  %Arithmetic.add_zero_left((GemmReadback.group_word(query,unit,group) * 8 : U32)) :
    {GemmWrites.job_writes(Gemm.job(Gemm.Layout{GemmReadback.rows(query),GemmReadback.columns(query),GemmReadback.inner(query),0,GemmReadback.columns(query)},U32.from_nat(GemmReadback.job_index(unit,group,GemmReadback.units(query)))),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)) ==
      GemmWrites.context_writes(Gemm.Context{(GemmReadback.unit_word(query,unit,group) * 4 : U32),_,GemmReadback.rows(query),GemmReadback.columns(query),
        (GemmReadback.group_word(query,unit,group) * GemmReadback.inner(query) : U32),(GemmReadback.unit_word(query,unit,group) / 2 * GemmReadback.inner(query) : U32)},GemmReadback.unit_accumulators(query,unit,group),GemmReadback.bias(query)) : List<&2,Writes.Assignment<F32>>}
  job_context(query,unit,group,0,GemmReadback.columns(query))

def job_relocated(+query: GemmReadback.Query,+unit: Nat,+group: Nat,+start: U32,+columns: U32,
  +unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(GemmReadback.rows(query))) == True{} : Bool},
  +unit_inside: {Nat.is_lt(unit,GemmReadback.units(query)) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,GemmReadback.groups(query)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(GemmReadback.columns(query))),Nat.add(U32.to_nat(columns),7n)) == True{} : Bool},
  +columns_room: {Nat.is_le(U32.to_nat(columns),U32.to_nat(GemmReadback.limit(query))) == True{} : Bool}) ->
  {relocate(Placement{U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start)},GemmReadback.job_writes(query,unit,group)) ==
    GemmWrites.job_writes(Gemm.job(Gemm.Layout{GemmReadback.rows(query),GemmReadback.columns(query),GemmReadback.inner(query),start,columns},U32.from_nat(GemmReadback.job_index(unit,group,GemmReadback.units(query)))),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)) : List<&2,Writes.Assignment<F32>>}:
  +placement = {Placement{U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start)} : Placement}
  +rows = GemmReadback.rows(query)
  +span = GemmReadback.columns(query)
  +inner = GemmReadback.inner(query)
  +word = GemmReadback.unit_word(query,unit,group)
  +group_word = GemmReadback.group_word(query,unit,group)
  +word_exact = GemmReadback.unit_exact(query,unit,group,unit_inside,group_inside)
  +local_position = (group_word * 8 : U32)
  +global_position = (start + local_position : U32)
  +local_exact = GemmReadback.position_exact(query,group,group_word,GemmReadback.group_exact(query,unit,group,unit_inside,group_inside),group_inside)
  +global_exact = shifted_position(query,start,columns,group,local_position,local_exact,group_inside,end_room,columns_room)
  +before =
    {%Equal.sym(Nat,U32.to_nat(global_position),Nat.add(U32.to_nat(start),Nat.mul(group,8n)),global_exact) : {Nat.is_le(_,U32.to_nat(columns)) == True{} : Bool}
      Bounds.strict_le(Nat.add(U32.to_nat(start),Nat.mul(group,8n)),U32.to_nat(columns),group_before_output(query,start,columns,group,group_inside,end_room)) : {Nat.is_le(U32.to_nat(global_position),U32.to_nat(columns)) == True{} : Bool}}
  +index = (word * 4 : U32)
  +accumulators = GemmReadback.unit_accumulators(query,unit,group)
  +bias = GemmReadback.bias(query)
  +i0 = index
  +a0 = Rows.first(~Tile.Tile8,accumulators)
  +w0 = GemmWrites.row_writes(U32.is_lt(i0,rows),i0,local_position,span,a0,bias)
  +t0 = GemmWrites.row_writes(U32.is_lt(i0,rows),i0,global_position,columns,a0,bias)
  +e0 = row_relocated(query,unit,0n,group,i0,local_position,global_position,start,columns,a0,bias,GemmReadback.row_zero_exact(query,unit,word,word_exact,unit_rows),local_exact,global_exact,group_inside,before,columns_room,U32.is_lt(i0,rows),{==})
  +i1 = (index + 1 : U32)
  +a1 = Rows.second(~Tile.Tile8,accumulators)
  +w1 = GemmWrites.row_writes(U32.is_lt(i1,rows),i1,local_position,span,a1,bias)
  +t1 = GemmWrites.row_writes(U32.is_lt(i1,rows),i1,global_position,columns,a1,bias)
  +e1 = row_relocated(query,unit,1n,group,i1,local_position,global_position,start,columns,a1,bias,GemmReadback.row_offset_exact(query,unit,word,1n,word_exact,unit_rows,{==}),local_exact,global_exact,group_inside,before,columns_room,U32.is_lt(i1,rows),{==})
  +i2 = (index + 2 : U32)
  +a2 = Rows.third(~Tile.Tile8,accumulators)
  +w2 = GemmWrites.row_writes(U32.is_lt(i2,rows),i2,local_position,span,a2,bias)
  +t2 = GemmWrites.row_writes(U32.is_lt(i2,rows),i2,global_position,columns,a2,bias)
  +e2 = row_relocated(query,unit,2n,group,i2,local_position,global_position,start,columns,a2,bias,GemmReadback.row_offset_exact(query,unit,word,2n,word_exact,unit_rows,{==}),local_exact,global_exact,group_inside,before,columns_room,U32.is_lt(i2,rows),{==})
  +i3 = (index + 3 : U32)
  +a3 = Rows.fourth(~Tile.Tile8,accumulators)
  +w3 = GemmWrites.row_writes(U32.is_lt(i3,rows),i3,local_position,span,a3,bias)
  +t3 = GemmWrites.row_writes(U32.is_lt(i3,rows),i3,global_position,columns,a3,bias)
  +e3 = row_relocated(query,unit,3n,group,i3,local_position,global_position,start,columns,a3,bias,GemmReadback.row_offset_exact(query,unit,word,3n,word_exact,unit_rows,{==}),local_exact,global_exact,group_inside,before,columns_room,U32.is_lt(i3,rows),{==})
  +w01 = List.append(&2,Writes.Assignment<F32>,w0,w1)
  +w02 = List.append(&2,Writes.Assignment<F32>,w01,w2)
  +t01 = List.append(&2,Writes.Assignment<F32>,t0,t1)
  +t02 = List.append(&2,Writes.Assignment<F32>,t01,t2)
  %Equal.sym(List<&2,Writes.Assignment<F32>>,GemmReadback.job_writes(query,unit,group),
    GemmWrites.context_writes(Gemm.Context{index,local_position,rows,span,(group_word * inner : U32),(word / 2 * inner : U32)},accumulators,bias),local_job_context(query,unit,group)) :
    {relocate(placement,_) == GemmWrites.job_writes(Gemm.job(Gemm.Layout{rows,span,inner,start,columns},U32.from_nat(GemmReadback.job_index(unit,group,GemmReadback.units(query)))),GemmReadback.input(query),GemmReadback.weights(query),bias) : List<&2,Writes.Assignment<F32>>}
  %Equal.sym(List<&2,Writes.Assignment<F32>>,GemmWrites.job_writes(Gemm.job(Gemm.Layout{rows,span,inner,start,columns},U32.from_nat(GemmReadback.job_index(unit,group,GemmReadback.units(query)))),GemmReadback.input(query),GemmReadback.weights(query),bias),
    GemmWrites.context_writes(Gemm.Context{index,global_position,rows,columns,(group_word * inner : U32),(word / 2 * inner : U32)},accumulators,bias),job_context(query,unit,group,start,columns)) :
    {relocate(placement,List.append(&2,Writes.Assignment<F32>,w02,w3)) == _ : List<&2,Writes.Assignment<F32>>}
  concatenate_relocated(placement,w02,w3,t02,t3,
    concatenate_relocated(placement,w01,w2,t01,t2,concatenate_relocated(placement,w0,w1,t0,t1,e0,e1),e2),e3)

def indexed_job_relocated(+query: GemmReadback.Query,+index: Nat,+start: U32,+columns: U32,
  +decoded: GemmReadback.IndexedJob<query,index>,
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(GemmReadback.columns(query))),Nat.add(U32.to_nat(columns),7n)) == True{} : Bool},
  +columns_room: {Nat.is_le(U32.to_nat(columns),U32.to_nat(GemmReadback.limit(query))) == True{} : Bool}) ->
  {relocate(Placement{U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start)},
      GemmWrites.job_writes(Gemm.job(GemmReadback.layout(query),U32.from_nat(index)),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query))) ==
    GemmWrites.job_writes(Gemm.job(Gemm.Layout{GemmReadback.rows(query),GemmReadback.columns(query),GemmReadback.inner(query),start,columns},U32.from_nat(index)),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)) : List<&2,Writes.Assignment<F32>>}:
  GemmReadback.IndexedJob{+unit,+group,+unit_rows,+unit_inside,+group_inside,+exact} = decoded
  %Equal.sym(Nat,index,GemmReadback.job_index(unit,group,GemmReadback.units(query)),exact) :
    {relocate(Placement{U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start)},
        GemmWrites.job_writes(Gemm.job(GemmReadback.layout(query),U32.from_nat(_)),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query))) ==
      GemmWrites.job_writes(Gemm.job(Gemm.Layout{GemmReadback.rows(query),GemmReadback.columns(query),GemmReadback.inner(query),start,columns},U32.from_nat(_)),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)) : List<&2,Writes.Assignment<F32>>}
  job_relocated(query,unit,group,start,columns,unit_rows,unit_inside,group_inside,end_room,columns_room)

def job_at_relocated(+query: GemmReadback.Query,+index: Nat,+start: U32,+columns: U32,
  +inside: {Nat.is_lt(index,Nat.mul(GemmReadback.groups(query),GemmReadback.units(query))) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(GemmReadback.columns(query))),Nat.add(U32.to_nat(columns),7n)) == True{} : Bool},
  +columns_room: {Nat.is_le(U32.to_nat(columns),U32.to_nat(GemmReadback.limit(query))) == True{} : Bool}) ->
  {relocate(Placement{U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start)},
      GemmWrites.job_writes(Gemm.job(GemmReadback.layout(query),U32.from_nat(index)),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query))) ==
    GemmWrites.job_writes(Gemm.job(Gemm.Layout{GemmReadback.rows(query),GemmReadback.columns(query),GemmReadback.inner(query),start,columns},U32.from_nat(index)),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)) : List<&2,Writes.Assignment<F32>>}:
  indexed_job_relocated(query,index,start,columns,GemmReadback.indexed_job(query,index,inside),end_room,columns_room)

def all_relocated(+query: GemmReadback.Query,+count: Nat,+first: Nat,+start: U32,+columns: U32,
  +room: {Nat.is_le(Nat.add(first,count),Nat.mul(GemmReadback.groups(query),GemmReadback.units(query))) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(GemmReadback.columns(query))),Nat.add(U32.to_nat(columns),7n)) == True{} : Bool},
  +columns_room: {Nat.is_le(U32.to_nat(columns),U32.to_nat(GemmReadback.limit(query))) == True{} : Bool}) ->
  {relocate(Placement{U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start)},
      GemmWrites.all_writes(count,U32.from_nat(first),GemmReadback.layout(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query))) ==
    GemmWrites.all_writes(count,U32.from_nat(first),Gemm.Layout{GemmReadback.rows(query),GemmReadback.columns(query),GemmReadback.inner(query),start,columns},GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)) : List<&2,Writes.Assignment<F32>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +placement = {Placement{U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start)} : Placement}
      +local_layout = GemmReadback.layout(query)
      +global_layout = {Gemm.Layout{GemmReadback.rows(query),GemmReadback.columns(query),GemmReadback.inner(query),start,columns} : Gemm.Layout}
      +input = GemmReadback.input(query)
      +weights = GemmReadback.weights(query)
      +bias = GemmReadback.bias(query)
      +jobs = Nat.mul(GemmReadback.groups(query),GemmReadback.units(query))
      +local_job = GemmWrites.job_writes(Gemm.job(local_layout,U32.from_nat(first)),input,weights,bias)
      +global_job = GemmWrites.job_writes(Gemm.job(global_layout,U32.from_nat(first)),input,weights,bias)
      %Equal.sym(U32,(U32.from_nat(first) + 1 : U32),U32.from_nat(1n+first),Bounds.add_one(U32.from_nat(first))) :
        {relocate(placement,List.append(&2,Writes.Assignment<F32>,local_job,GemmWrites.all_writes(rest,_,local_layout,input,weights,bias))) ==
          List.append(&2,Writes.Assignment<F32>,global_job,GemmWrites.all_writes(rest,_,global_layout,input,weights,bias)) : List<&2,Writes.Assignment<F32>>}
      concatenate_relocated(placement,local_job,GemmWrites.all_writes(rest,U32.from_nat(1n+first),local_layout,input,weights,bias),
        global_job,GemmWrites.all_writes(rest,U32.from_nat(1n+first),global_layout,input,weights,bias),
        job_at_relocated(query,first,start,columns,Order.room_index(first,rest,jobs,room),end_room,columns_room),
        all_relocated(query,rest,1n+first,start,columns,Order.advance_room(first,rest,jobs,room),end_room,columns_room))

def local_writes(+query: GemmReadback.Query) -> List<&2,Writes.Assignment<F32>>:
  GemmWrites.all_writes(Nat.mul(GemmReadback.groups(query),GemmReadback.units(query)),0,GemmReadback.layout(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query))

def local_capacity(+query: GemmReadback.Query) -> U32:
  U32.from_nat(Nat.mul(U32.to_nat(GemmReadback.rows(query)),U32.to_nat(GemmReadback.columns(query))))

def local_capacity_exact(+query: GemmReadback.Query) ->
  {U32.to_nat(local_capacity(query)) == Nat.mul(U32.to_nat(GemmReadback.rows(query)),U32.to_nat(GemmReadback.columns(query))) : Nat}:
  Bounds.roundtrip(Nat.mul(U32.to_nat(GemmReadback.rows(query)),U32.to_nat(GemmReadback.columns(query))),GemmReadback.limit(query),GemmReadback.limit_fits(query),GemmReadback.area_room(query))

def complete_evidence(+query: GemmReadback.Query) ->
  Readback.WriteEvidence<GemmReadback.address(query),GemmReadback.wanted(query),local_capacity(query),local_writes(query)>:
  GemmReadback.all_evidence(query,local_capacity(query),Nat.mul(GemmReadback.groups(query),GemmReadback.units(query)),0n,
    %Equal.sym(Nat,U32.to_nat(local_capacity(query)),Nat.mul(U32.to_nat(GemmReadback.rows(query)),U32.to_nat(GemmReadback.columns(query))),local_capacity_exact(query)) :
      {Nat.is_le(Nat.mul(U32.to_nat(GemmReadback.rows(query)),U32.to_nat(GemmReadback.columns(query))),_) == True{} : Bool}
    Order.reflexive(Nat.mul(U32.to_nat(GemmReadback.rows(query)),U32.to_nat(GemmReadback.columns(query)))),Order.reflexive(Nat.mul(GemmReadback.groups(query),GemmReadback.units(query))))

def all_within(+query: GemmReadback.Query,+start: U32,+columns: U32,+capacity: U32,
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat(GemmReadback.rows(query)),U32.to_nat(columns)),U32.to_nat(capacity)) == True{} : Bool}) ->
  Within(Placement{U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start)},capacity,local_writes(query)):
  within_local(U32.to_nat(GemmReadback.rows(query)),U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start),capacity,local_capacity(query),local_writes(query),
    GemmReadback.columns_positive(query),local_capacity_exact(query),output_room,
    Readback.evidence_bounds(GemmReadback.address(query),GemmReadback.wanted(query),local_capacity(query),local_writes(query),complete_evidence(query)))

def run_relocated(+query: GemmReadback.Query,+start: U32,+columns: U32,+output: Tree.Tree<F32>,
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(GemmReadback.columns(query))),Nat.add(U32.to_nat(columns),7n)) == True{} : Bool},
  +columns_room: {Nat.is_le(U32.to_nat(columns),U32.to_nat(GemmReadback.limit(query))) == True{} : Bool}) ->
  {GemmLaws.run_at(GemmReadback.rows(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query),output,GemmReadback.columns(query),GemmReadback.inner(query),start,columns) ==
    Writes.pure(F32,relocate(Placement{U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start)},local_writes(query)),output) : Tree.Tree<F32>}:
  +jobs = Nat.mul(GemmReadback.groups(query),GemmReadback.units(query))
  %Equal.sym(Tree.Tree<F32>,GemmLaws.run_at(GemmReadback.rows(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query),output,GemmReadback.columns(query),GemmReadback.inner(query),start,columns),
    Writes.pure(F32,GemmWrites.all_writes(GemmWrites.job_count(GemmReadback.rows(query),GemmReadback.columns(query)),0,Gemm.Layout{GemmReadback.rows(query),GemmReadback.columns(query),GemmReadback.inner(query),start,columns},GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)),output),
    GemmWrites.run_at_writes(GemmReadback.rows(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query),output,GemmReadback.columns(query),GemmReadback.inner(query),start,columns)) :
    {_ == Writes.pure(F32,relocate(Placement{U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start)},local_writes(query)),output) : Tree.Tree<F32>}
  %Equal.sym(Nat,GemmWrites.job_count(GemmReadback.rows(query),GemmReadback.columns(query)),jobs,GemmReadback.count_exact(query)) :
    {Writes.pure(F32,GemmWrites.all_writes(_,0,Gemm.Layout{GemmReadback.rows(query),GemmReadback.columns(query),GemmReadback.inner(query),start,columns},GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)),output) ==
      Writes.pure(F32,relocate(Placement{U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start)},local_writes(query)),output) : Tree.Tree<F32>}
  Equal.cong(List<&2,Writes.Assignment<F32>>,Tree.Tree<F32>,writes => Writes.pure(F32,writes,output),
    GemmWrites.all_writes(jobs,0,Gemm.Layout{GemmReadback.rows(query),GemmReadback.columns(query),GemmReadback.inner(query),start,columns},GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)),
    relocate(Placement{U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start)},local_writes(query)),
    Equal.sym(List<&2,Writes.Assignment<F32>>,relocate(Placement{U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start)},local_writes(query)),
      GemmWrites.all_writes(jobs,0,Gemm.Layout{GemmReadback.rows(query),GemmReadback.columns(query),GemmReadback.inner(query),start,columns},GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query)),
      all_relocated(query,jobs,0n,start,columns,Order.reflexive(jobs),end_room,columns_room)))

# Read an in-window result directly from the full output allocation.
def run_readback(+query: GemmReadback.Query,+start: U32,+columns: U32,+output: Tree.Tree<F32>,+depth: Nat,
  +balanced: Shape.Balanced(F32,depth,output),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat(GemmReadback.rows(query)),U32.to_nat(columns)),U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(GemmReadback.columns(query))),Nat.add(U32.to_nat(columns),7n)) == True{} : Bool},
  +columns_room: {Nat.is_le(U32.to_nat(columns),U32.to_nat(GemmReadback.limit(query))) == True{} : Bool},
  +query_inside: {Nat.is_lt(Nat.add(U32.to_nat(start),GemmReadback.column(query)),U32.to_nat(columns)) == True{} : Bool}) ->
  {Tree.read(F32,GemmLaws.run_at(GemmReadback.rows(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query),output,GemmReadback.columns(query),GemmReadback.inner(query),start,columns),
    U32.from_nat(Nat.add(Nat.mul(GemmReadback.channel(query),U32.to_nat(columns)),Nat.add(U32.to_nat(start),GemmReadback.column(query))))) == GemmReadback.wanted(query) : F32}:
  +span = U32.to_nat(GemmReadback.columns(query))
  +rows = U32.to_nat(GemmReadback.rows(query))
  +width = U32.to_nat(columns)
  +placement = {Placement{span,width,U32.to_nat(start)} : Placement}
  +area = Nat.mul(rows,span)
  +local_capacity = U32.from_nat(area)
  +local_exact = Bounds.roundtrip(area,GemmReadback.limit(query),GemmReadback.limit_fits(query),GemmReadback.area_room(query))
  +jobs = Nat.mul(GemmReadback.groups(query),GemmReadback.units(query))
  +local_writes = GemmWrites.all_writes(jobs,0,GemmReadback.layout(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query))
  +source_query = U32.from_nat(GemmReadback.address(query))
  +source_inside = RowMajor.radix(GemmReadback.channel(query),rows,GemmReadback.column(query),span,GemmReadback.channel_inside(query),GemmReadback.column_inside(query))
  +source_exact = Bounds.roundtrip(GemmReadback.address(query),GemmReadback.limit(query),GemmReadback.limit_fits(query),
    Order.transitive(GemmReadback.address(query),area,U32.to_nat(GemmReadback.limit(query)),Bounds.strict_le(GemmReadback.address(query),area,source_inside),GemmReadback.area_room(query)))
  +capacity = Tree.capacity(F32,output)
  +capacity_fits = StorageCapacity.small(depth,output,balanced,depth_ok)
  +local_evidence = complete_evidence(query)
  +within = all_within(query,start,columns,capacity,output_room)
  +keep_query =
    {%Equal.sym(Bool,selected(placement,source_query),Nat.is_lt(Nat.add(U32.to_nat(start),GemmReadback.column(query)),width),
        coordinate_selected(span,width,U32.to_nat(start),GemmReadback.channel(query),GemmReadback.column(query),source_query,GemmReadback.columns_positive(query),GemmReadback.column_inside(query),source_exact)) : {_ == True{} : Bool}
      query_inside : {selected(placement,source_query) == True{} : Bool}}
  +target = Nat.add(Nat.mul(GemmReadback.channel(query),width),Nat.add(U32.to_nat(start),GemmReadback.column(query)))
  +target_exact = coordinate_destination(span,width,U32.to_nat(start),GemmReadback.channel(query),GemmReadback.column(query),source_query,GemmReadback.columns_positive(query),GemmReadback.column_inside(query),source_exact)
  %Equal.sym(Tree.Tree<F32>,GemmLaws.run_at(GemmReadback.rows(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query),output,GemmReadback.columns(query),GemmReadback.inner(query),start,columns),
    Writes.pure(F32,relocate(placement,local_writes),output),run_relocated(query,start,columns,output,end_room,columns_room)) :
    {Tree.read(F32,_,U32.from_nat(target)) == GemmReadback.wanted(query) : F32}
  %target_exact : {Tree.read(F32,Writes.pure(F32,relocate(placement,local_writes),output),U32.from_nat(_)) == GemmReadback.wanted(query) : F32}
  GemmReadback.evidence_readback(relocate(placement,local_writes),output,depth,balanced,depth_ok,destination(placement,source_query),GemmReadback.wanted(query),
    %Equal.sym(Nat,destination(placement,source_query),target,target_exact) : {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
    Bounds.nat_trans(target,Nat.mul(rows,width),U32.to_nat(capacity),RowMajor.radix(GemmReadback.channel(query),rows,Nat.add(U32.to_nat(start),GemmReadback.column(query)),width,GemmReadback.channel_inside(query),query_inside),output_room),
    evidence(span,width,U32.to_nat(start),source_query,GemmReadback.wanted(query),local_writes,capacity,GemmReadback.columns_positive(query),capacity_fits,keep_query,within,
      %Equal.sym(Nat,U32.to_nat(source_query),GemmReadback.address(query),source_exact) : Coverage.MatchingWritesHaveValue(_,GemmReadback.wanted(query),Readback.natural_write_addresses(local_writes))
      Readback.evidence_values(GemmReadback.address(query),GemmReadback.wanted(query),local_capacity,local_writes,local_evidence)),
    %Equal.sym(Bool,GatherWrites.contains(destination(placement,source_query),relocate(placement,local_writes)),GatherWrites.contains(U32.to_nat(source_query),local_writes),
      coverage(span,width,U32.to_nat(start),source_query,local_writes,capacity,GemmReadback.columns_positive(query),capacity_fits,keep_query,within)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(source_query),GemmReadback.address(query),source_exact) : {GatherWrites.contains(_,local_writes) == True{} : Bool}
    GemmReadback.complete_coverage(query))

def before_distinct(+span: Nat,+columns: Nat,+start: Nat,+channel: Nat,+column: Nat,+index: U32,+query: U32,+capacity: U32,
  +column_inside: {Nat.is_lt(column,columns) == True{} : Bool},+before: {Nat.is_lt(column,start) == True{} : Bool},
  +query_exact: {U32.to_nat(query) == Nat.add(Nat.mul(channel,columns),column) : Nat},
  +index_inside: {Nat.is_lt(destination(Placement{span,columns,start},index),U32.to_nat(capacity)) == True{} : Bool},
  +capacity_fits: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +selected: {Nat.is_lt(position(Placement{span,columns,start},index),columns) == True{} : Bool},
  +same: Bool,+same_exact: {U32.is_eq(U32.from_nat(destination(Placement{span,columns,start},index)),query) == same : Bool}) ->
  {U32.is_eq(U32.from_nat(destination(Placement{span,columns,start},index)),query) == False{} : Bool}:
  match same:
    case False{}: same_exact
    case True{}:
      +placement = {Placement{span,columns,start} : Placement}
      +address = destination(placement,index)
      +pos = position(placement,index)
      +roundtrip = Bounds.roundtrip(address,capacity,capacity_fits,Bounds.strict_le(address,U32.to_nat(capacity),index_inside))
      +equal = Address.equal_words(U32.from_nat(address),query,same_exact)
      +natural_equal =
        {%Equal.sym(Nat,address,U32.to_nat(U32.from_nat(address)),Equal.sym(Nat,U32.to_nat(U32.from_nat(address)),address,roundtrip)) : {_ == Nat.add(Nat.mul(channel,columns),column) : Nat}
        %Equal.sym(U32,U32.from_nat(address),query,equal) : {U32.to_nat(_) == Nat.add(Nat.mul(channel,columns),column) : Nat}
        query_exact : {address == Nat.add(Nat.mul(channel,columns),column) : Nat}}
      +column_equal = Radix.col_unique(Nat.div(U32.to_nat(index),span),channel,columns,pos,column,
        Order.le_lt_trans(0n,column,columns,Order.zero_le(column),column_inside),selected,column_inside,natural_equal)
      +impossible = Equal.trans(Bool,False{},Nat.is_lt(pos,start),True{},
        Equal.sym(Bool,Nat.is_lt(pos,start),False{},Order.not_below(start,pos,Order.le_add(start,Nat.mod(U32.to_nat(index),span)))),
        %Equal.sym(Nat,pos,column,column_equal) : {Nat.is_lt(_,start) == True{} : Bool} before)
      Empty.absurd({U32.is_eq(U32.from_nat(address),query) == False{} : Bool},WordEncoding.false_true(impossible))

def prepend_excludes_before(+span: Nat,+columns: Nat,+start: Nat,+channel: Nat,+column: Nat,+index: U32,+value: F32,+query: U32,+capacity: U32,+rest: List<&2,Writes.Assignment<F32>>,
  +column_inside: {Nat.is_lt(column,columns) == True{} : Bool},+before: {Nat.is_lt(column,start) == True{} : Bool},
  +query_exact: {U32.to_nat(query) == Nat.add(Nat.mul(channel,columns),column) : Nat},
  +capacity_fits: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +keep: Bool,+keep_exact: {selected(Placement{span,columns,start},index) == keep : Bool},
  +index_inside: Boolean.When(keep,{Nat.is_lt(destination(Placement{span,columns,start},index),U32.to_nat(capacity)) == True{} : Bool}),
  tail: Readback.Excludes(query,rest)) ->
  Readback.Excludes(query,GatherWrites.prepend(keep,U32.from_nat(destination(Placement{span,columns,start},index)),value,rest)):
  match keep:
    case False{}: tail
    case True{}:
      Shape.Both{before_distinct(span,columns,start,channel,column,index,query,capacity,column_inside,before,query_exact,index_inside,capacity_fits,keep_exact,
        U32.is_eq(U32.from_nat(destination(Placement{span,columns,start},index)),query),{==}),tail}

def excludes_before(+span: Nat,+columns: Nat,+start: Nat,+channel: Nat,+column: Nat,+query: U32,+capacity: U32,+writes: List<&2,Writes.Assignment<F32>>,
  +column_inside: {Nat.is_lt(column,columns) == True{} : Bool},+before: {Nat.is_lt(column,start) == True{} : Bool},
  +query_exact: {U32.to_nat(query) == Nat.add(Nat.mul(channel,columns),column) : Nat},
  +capacity_fits: {U32.is_le(capacity,2147483648) == True{} : Bool},within: Within(Placement{span,columns,start},capacity,writes)) ->
  Readback.Excludes(query,relocate(Placement{span,columns,start},writes)):
  match writes:
    case Nil{}: Unit{}
    case Writes.Put{+index,+value} <> +rest:
      Shape.Both{+here,tail} = within
      prepend_excludes_before(span,columns,start,channel,column,index,value,query,capacity,relocate(Placement{span,columns,start},rest),column_inside,before,query_exact,capacity_fits,
        selected(Placement{span,columns,start},index),{==},here,
        excludes_before(span,columns,start,channel,column,query,capacity,rest,column_inside,before,query_exact,capacity_fits,tail))

# A later block preserves every completed spatial position in every channel.
def run_preserves_before(+query: GemmReadback.Query,+start: U32,+columns: U32,+output: Tree.Tree<F32>,+depth: Nat,+column: Nat,
  +balanced: Shape.Balanced(F32,depth,output),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat(GemmReadback.rows(query)),U32.to_nat(columns)),U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(GemmReadback.columns(query))),Nat.add(U32.to_nat(columns),7n)) == True{} : Bool},
  +columns_room: {Nat.is_le(U32.to_nat(columns),U32.to_nat(GemmReadback.limit(query))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(columns)) == True{} : Bool},+before: {Nat.is_lt(column,U32.to_nat(start)) == True{} : Bool}) ->
  {Tree.read(F32,GemmLaws.run_at(GemmReadback.rows(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query),output,GemmReadback.columns(query),GemmReadback.inner(query),start,columns),
      U32.from_nat(Nat.add(Nat.mul(GemmReadback.channel(query),U32.to_nat(columns)),column))) ==
    Tree.read(F32,output,U32.from_nat(Nat.add(Nat.mul(GemmReadback.channel(query),U32.to_nat(columns)),column))) : F32}:
  +placement = {Placement{U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start)} : Placement}
  +capacity = Tree.capacity(F32,output)
  +capacity_fits = StorageCapacity.small(depth,output,balanced,depth_ok)
  +target = Nat.add(Nat.mul(GemmReadback.channel(query),U32.to_nat(columns)),column)
  +target_inside = Bounds.nat_trans(target,Nat.mul(U32.to_nat(GemmReadback.rows(query)),U32.to_nat(columns)),U32.to_nat(capacity),
    RowMajor.radix(GemmReadback.channel(query),U32.to_nat(GemmReadback.rows(query)),column,U32.to_nat(columns),GemmReadback.channel_inside(query),column_inside),output_room)
  +target_exact = Bounds.roundtrip(target,capacity,capacity_fits,Bounds.strict_le(target,U32.to_nat(capacity),target_inside))
  +within = all_within(query,start,columns,capacity,output_room)
  %Equal.sym(Tree.Tree<F32>,GemmLaws.run_at(GemmReadback.rows(query),GemmReadback.input(query),GemmReadback.weights(query),GemmReadback.bias(query),output,GemmReadback.columns(query),GemmReadback.inner(query),start,columns),
    Writes.pure(F32,relocate(placement,local_writes(query)),output),run_relocated(query,start,columns,output,end_room,columns_room)) :
    {Tree.read(F32,_,U32.from_nat(target)) == Tree.read(F32,output,U32.from_nat(target)) : F32}
  Readback.excluded_readback(relocate(placement,local_writes(query)),output,depth,U32.from_nat(target),balanced,depth_ok,
    bounds(placement,capacity,local_writes(query),capacity_fits,within),Bounds.native_bound(target,capacity,target_exact,target_inside),
    excludes_before(U32.to_nat(GemmReadback.columns(query)),U32.to_nat(columns),U32.to_nat(start),GemmReadback.channel(query),column,U32.from_nat(target),capacity,local_writes(query),
      column_inside,before,target_exact,capacity_fits,within))
