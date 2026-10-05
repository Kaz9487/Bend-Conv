# A sample function over the frozen K-axis retains its coordinate at every index.
import Base
import ../specification.bend as ConvSpec
import ../matrix_specification.bend as MatrixSpec
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ./matrix_axis_lengths.bend as AxisLengths
import ./sequence_congruence.bend as Sequences
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/word_encoding.bend as WordEncoding

def values(~Context: Data,~sample: Context -> MatrixSpec.Reduction -> F32,taps: List<&2,MatrixSpec.Reduction>,+context: Context) -> List<&2,F32>:
  match taps:
    case Nil{}: []
    case tap <> rest: sample(context,tap) <> values(~Context,~sample,rest,context)

def values_length(~Context: Data,~sample: Context -> MatrixSpec.Reduction -> F32,+taps: List<&2,MatrixSpec.Reduction>,+context: Context) ->
  {ConvSpec.length(values(~Context,~sample,taps,context)) == AxisLengths.length(MatrixSpec.Reduction,taps) : Nat}:
  match taps:
    case Nil{}: {==}
    case tap <> +rest:
      %values_length(~Context,~sample,rest,context) : {1n+ConvSpec.length(values(~Context,~sample,rest,context)) == 1n+_ : Nat}
      {==}

def values_concatenated(~Context: Data,~sample: Context -> MatrixSpec.Reduction -> F32,+left: List<&2,MatrixSpec.Reduction>,+right: List<&2,MatrixSpec.Reduction>,+context: Context) ->
  {values(~Context,~sample,OrderedMatrix.cat(MatrixSpec.Reduction,left,right),context) == OrderedMatrix.cat(F32,values(~Context,~sample,left,context),values(~Context,~sample,right,context)) : List<&2,F32>}:
  match left:
    case Nil{}: {==}
    case +tap <> +rest:
      %values_concatenated(~Context,~sample,rest,right,context) : {sample(context,tap) <> values(~Context,~sample,OrderedMatrix.cat(MatrixSpec.Reduction,rest,right),context) == sample(context,tap) <> _ : List<&2,F32>}
      {==}

def columns(~Context: Data,~sample: Context -> MatrixSpec.Reduction -> F32,+context: Context,+count: Nat,+start: Nat,+kernel_row: Nat,+channel: Nat,+column: Nat,
  inside: {Nat.is_lt(column,count) == True{} : Bool}) ->
  {ConvSpec.at(column,values(~Context,~sample,MatrixSpec.kcols(count,start,kernel_row,channel),context)) == sample(context,MatrixSpec.KIndex{channel,kernel_row,Nat.add(start,column)}) : F32}:
  match count:
    case 0n:
      Empty.absurd({ConvSpec.at(column,[]) == sample(context,MatrixSpec.KIndex{channel,kernel_row,Nat.add(start,column)}) : F32},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(column,0n),True{},Equal.sym(Bool,Nat.is_lt(column,0n),False{},Division.lt_zero(column)),inside)))
    case 1n+ +rest:
      match column:
        case 0n:
          %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) : {sample(context,MatrixSpec.KIndex{channel,kernel_row,start}) == sample(context,MatrixSpec.KIndex{channel,kernel_row,_}) : F32}
          {==}
        case 1n+ +previous:
          %Equal.sym(Nat,Nat.add(start,1n+previous),1n+Nat.add(start,previous),Addition.successor_right(start,previous)) :
            {ConvSpec.at(previous,values(~Context,~sample,MatrixSpec.kcols(rest,1n+start,kernel_row,channel),context)) == sample(context,MatrixSpec.KIndex{channel,kernel_row,_}) : F32}
          columns(~Context,~sample,context,rest,1n+start,kernel_row,channel,previous,inside)

def rows(~Context: Data,~sample: Context -> MatrixSpec.Reduction -> F32,+context: Context,+count: Nat,+start: Nat,+channel: Nat,+kernel: Nat,+row: Nat,+column: Nat,
  row_inside: {Nat.is_lt(row,count) == True{} : Bool},+column_inside: {Nat.is_lt(column,kernel) == True{} : Bool}) ->
  {ConvSpec.at(Nat.add(Nat.mul(row,kernel),column),values(~Context,~sample,MatrixSpec.krows(count,start,channel,kernel),context)) == sample(context,MatrixSpec.KIndex{channel,Nat.add(start,row),column}) : F32}:
  match count:
    case 0n:
      Empty.absurd({ConvSpec.at(Nat.add(Nat.mul(row,kernel),column),[]) == sample(context,MatrixSpec.KIndex{channel,Nat.add(start,row),column}) : F32},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(row,0n),True{},Equal.sym(Bool,Nat.is_lt(row,0n),False{},Division.lt_zero(row)),row_inside)))
    case 1n+ +rest:
      match row:
        case 0n:
          +left = MatrixSpec.kcols(kernel,0n,start,channel)
          +right = MatrixSpec.krows(rest,1n+start,channel,kernel)
          %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) :
            {ConvSpec.at(column,values(~Context,~sample,OrderedMatrix.cat(MatrixSpec.Reduction,left,right),context)) == sample(context,MatrixSpec.KIndex{channel,_,column}) : F32}
          %Equal.sym(List<&2,F32>,values(~Context,~sample,OrderedMatrix.cat(MatrixSpec.Reduction,left,right),context),OrderedMatrix.cat(F32,values(~Context,~sample,left,context),values(~Context,~sample,right,context)),values_concatenated(~Context,~sample,left,right,context)) :
            {ConvSpec.at(column,_) == sample(context,MatrixSpec.KIndex{channel,start,column}) : F32}
          %Equal.sym(F32,ConvSpec.at(column,OrderedMatrix.cat(F32,values(~Context,~sample,left,context),values(~Context,~sample,right,context))),ConvSpec.at(column,values(~Context,~sample,left,context)),
            Sequences.at_concatenated_left(values(~Context,~sample,left,context),values(~Context,~sample,right,context),column,
              %Equal.sym(Nat,ConvSpec.length(values(~Context,~sample,left,context)),AxisLengths.length(MatrixSpec.Reduction,left),values_length(~Context,~sample,left,context)) : {Nat.is_lt(column,_) == True{} : Bool}
              %Equal.sym(Nat,AxisLengths.length(MatrixSpec.Reduction,left),kernel,AxisLengths.kcols(kernel,0n,start,channel)) : {Nat.is_lt(column,_) == True{} : Bool}
              column_inside)) : {_ == sample(context,MatrixSpec.KIndex{channel,start,column}) : F32}
          columns(~Context,~sample,context,kernel,0n,start,channel,column,column_inside)
        case 1n+ +previous:
          +left = MatrixSpec.kcols(kernel,0n,start,channel)
          +right = MatrixSpec.krows(rest,1n+start,channel,kernel)
          +offset = Nat.add(Nat.mul(previous,kernel),column)
          %Equal.sym(Nat,Nat.add(Nat.add(kernel,Nat.mul(previous,kernel)),column),Nat.add(kernel,offset),Addition.associative(kernel,Nat.mul(previous,kernel),column)) :
            {ConvSpec.at(_,values(~Context,~sample,OrderedMatrix.cat(MatrixSpec.Reduction,left,right),context)) == sample(context,MatrixSpec.KIndex{channel,Nat.add(start,1n+previous),column}) : F32}
          %Equal.sym(List<&2,F32>,values(~Context,~sample,OrderedMatrix.cat(MatrixSpec.Reduction,left,right),context),OrderedMatrix.cat(F32,values(~Context,~sample,left,context),values(~Context,~sample,right,context)),values_concatenated(~Context,~sample,left,right,context)) :
            {ConvSpec.at(Nat.add(kernel,offset),_) == sample(context,MatrixSpec.KIndex{channel,Nat.add(start,1n+previous),column}) : F32}
          %AxisLengths.kcols(kernel,0n,start,channel) :
            {ConvSpec.at(Nat.add(_,offset),OrderedMatrix.cat(F32,values(~Context,~sample,left,context),values(~Context,~sample,right,context))) == sample(context,MatrixSpec.KIndex{channel,Nat.add(start,1n+previous),column}) : F32}
          %values_length(~Context,~sample,left,context) :
            {ConvSpec.at(Nat.add(_,offset),OrderedMatrix.cat(F32,values(~Context,~sample,left,context),values(~Context,~sample,right,context))) == sample(context,MatrixSpec.KIndex{channel,Nat.add(start,1n+previous),column}) : F32}
          %Equal.sym(F32,ConvSpec.at(Nat.add(ConvSpec.length(values(~Context,~sample,left,context)),offset),OrderedMatrix.cat(F32,values(~Context,~sample,left,context),values(~Context,~sample,right,context))),
            ConvSpec.at(offset,values(~Context,~sample,right,context)),Sequences.at_concatenated_right(values(~Context,~sample,left,context),values(~Context,~sample,right,context),offset)) :
            {_ == sample(context,MatrixSpec.KIndex{channel,Nat.add(start,1n+previous),column}) : F32}
          %Equal.sym(Nat,Nat.add(start,1n+previous),1n+Nat.add(start,previous),Addition.successor_right(start,previous)) :
            {ConvSpec.at(offset,values(~Context,~sample,right,context)) == sample(context,MatrixSpec.KIndex{channel,_,column}) : F32}
          rows(~Context,~sample,context,rest,1n+start,channel,kernel,previous,column,row_inside,column_inside)

def lookup_first(~Context: Data,~sample: Context -> MatrixSpec.Reduction -> F32,+context: Context,+left: List<&2,MatrixSpec.Reduction>,+right: List<&2,MatrixSpec.Reduction>,+index: Nat,+length: Nat,
  exact_length: {AxisLengths.length(MatrixSpec.Reduction,left) == length : Nat},inside: {Nat.is_lt(index,length) == True{} : Bool}) ->
  {ConvSpec.at(index,values(~Context,~sample,OrderedMatrix.cat(MatrixSpec.Reduction,left,right),context)) == ConvSpec.at(index,values(~Context,~sample,left,context)) : F32}:
  %Equal.sym(List<&2,F32>,values(~Context,~sample,OrderedMatrix.cat(MatrixSpec.Reduction,left,right),context),OrderedMatrix.cat(F32,values(~Context,~sample,left,context),values(~Context,~sample,right,context)),values_concatenated(~Context,~sample,left,right,context)) :
    {ConvSpec.at(index,_) == ConvSpec.at(index,values(~Context,~sample,left,context)) : F32}
  Sequences.at_concatenated_left(values(~Context,~sample,left,context),values(~Context,~sample,right,context),index,
    %Equal.sym(Nat,ConvSpec.length(values(~Context,~sample,left,context)),AxisLengths.length(MatrixSpec.Reduction,left),values_length(~Context,~sample,left,context)) : {Nat.is_lt(index,_) == True{} : Bool}
    %Equal.sym(Nat,AxisLengths.length(MatrixSpec.Reduction,left),length,exact_length) : {Nat.is_lt(index,_) == True{} : Bool}
    inside)

def lookup_after(~Context: Data,~sample: Context -> MatrixSpec.Reduction -> F32,+context: Context,+left: List<&2,MatrixSpec.Reduction>,+right: List<&2,MatrixSpec.Reduction>,+index: Nat,+length: Nat,
  exact_length: {AxisLengths.length(MatrixSpec.Reduction,left) == length : Nat}) ->
  {ConvSpec.at(Nat.add(length,index),values(~Context,~sample,OrderedMatrix.cat(MatrixSpec.Reduction,left,right),context)) == ConvSpec.at(index,values(~Context,~sample,right,context)) : F32}:
  %Equal.sym(List<&2,F32>,values(~Context,~sample,OrderedMatrix.cat(MatrixSpec.Reduction,left,right),context),OrderedMatrix.cat(F32,values(~Context,~sample,left,context),values(~Context,~sample,right,context)),values_concatenated(~Context,~sample,left,right,context)) :
    {ConvSpec.at(Nat.add(length,index),_) == ConvSpec.at(index,values(~Context,~sample,right,context)) : F32}
  %exact_length : {ConvSpec.at(Nat.add(_,index),OrderedMatrix.cat(F32,values(~Context,~sample,left,context),values(~Context,~sample,right,context))) == ConvSpec.at(index,values(~Context,~sample,right,context)) : F32}
  %values_length(~Context,~sample,left,context) : {ConvSpec.at(Nat.add(_,index),OrderedMatrix.cat(F32,values(~Context,~sample,left,context),values(~Context,~sample,right,context))) == ConvSpec.at(index,values(~Context,~sample,right,context)) : F32}
  Sequences.at_concatenated_right(values(~Context,~sample,left,context),values(~Context,~sample,right,context),index)

def channels(~Context: Data,~sample: Context -> MatrixSpec.Reduction -> F32,+context: Context,+count: Nat,+start: Nat,+kernel: Nat,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,count) == True{} : Bool},+row_inside: {Nat.is_lt(row,kernel) == True{} : Bool},+column_inside: {Nat.is_lt(column,kernel) == True{} : Bool}) ->
  {ConvSpec.at(RowMajor.index(channel,kernel,kernel,row,column),values(~Context,~sample,MatrixSpec.kchannels(count,start,kernel),context)) == sample(context,MatrixSpec.KIndex{Nat.add(start,channel),row,column}) : F32}:
  match count:
    case 0n:
      Empty.absurd({ConvSpec.at(RowMajor.index(channel,kernel,kernel,row,column),[]) == sample(context,MatrixSpec.KIndex{Nat.add(start,channel),row,column}) : F32},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(channel,0n),True{},Equal.sym(Bool,Nat.is_lt(channel,0n),False{},Division.lt_zero(channel)),channel_inside)))
    case 1n+ +rest:
      match channel:
        case 0n:
          +left = MatrixSpec.krows(kernel,0n,start,kernel)
          +right = MatrixSpec.kchannels(rest,1n+start,kernel)
          +offset = Nat.add(Nat.mul(row,kernel),column)
          %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) :
            {ConvSpec.at(offset,values(~Context,~sample,OrderedMatrix.cat(MatrixSpec.Reduction,left,right),context)) == sample(context,MatrixSpec.KIndex{_,row,column}) : F32}
          %Equal.sym(F32,ConvSpec.at(offset,values(~Context,~sample,OrderedMatrix.cat(MatrixSpec.Reduction,left,right),context)),ConvSpec.at(offset,values(~Context,~sample,left,context)),
            lookup_first(~Context,~sample,context,left,right,offset,Nat.mul(kernel,kernel),AxisLengths.krows(kernel,0n,start,kernel),RowMajor.radix(row,kernel,column,kernel,row_inside,column_inside))) :
            {_ == sample(context,MatrixSpec.KIndex{start,row,column}) : F32}
          rows(~Context,~sample,context,kernel,0n,start,kernel,row,column,row_inside,column_inside)
        case 1n+ +previous:
          +left = MatrixSpec.krows(kernel,0n,start,kernel)
          +right = MatrixSpec.kchannels(rest,1n+start,kernel)
          +square = Nat.mul(kernel,kernel)
          +offset = RowMajor.index(previous,kernel,kernel,row,column)
          %Equal.sym(Nat,RowMajor.index(1n+previous,kernel,kernel,row,column),Nat.add(square,offset),Addition.associative(square,Nat.mul(previous,square),Nat.add(Nat.mul(row,kernel),column))) :
            {ConvSpec.at(_,values(~Context,~sample,OrderedMatrix.cat(MatrixSpec.Reduction,left,right),context)) == sample(context,MatrixSpec.KIndex{Nat.add(start,1n+previous),row,column}) : F32}
          %Equal.sym(F32,ConvSpec.at(Nat.add(square,offset),values(~Context,~sample,OrderedMatrix.cat(MatrixSpec.Reduction,left,right),context)),ConvSpec.at(offset,values(~Context,~sample,right,context)),
            lookup_after(~Context,~sample,context,left,right,offset,square,AxisLengths.krows(kernel,0n,start,kernel))) : {_ == sample(context,MatrixSpec.KIndex{Nat.add(start,1n+previous),row,column}) : F32}
          %Equal.sym(Nat,Nat.add(start,1n+previous),1n+Nat.add(start,previous),Addition.successor_right(start,previous)) :
            {ConvSpec.at(offset,values(~Context,~sample,right,context)) == sample(context,MatrixSpec.KIndex{_,row,column}) : F32}
          channels(~Context,~sample,context,rest,1n+start,kernel,previous,row,column,channel_inside,row_inside,column_inside)
