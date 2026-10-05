import Base
import ./specification.bend as ConvSpec

law kernel_columns:
  for +n: Nat
  for +kernel_column_index: Nat
  for +kernel_row_index: Nat
  for +input_channel_index: Nat
  for +output_channel_index: Nat
  for +output_row_index: Nat
  for +output_column_index: Nat
  for +p: ConvSpec.Problem
  for +acc: F32
  F32

law kernel_rows:
  for +n: Nat
  for +kernel_row_index: Nat
  for +input_channel_index: Nat
  for +output_channel_index: Nat
  for +output_row_index: Nat
  for +output_column_index: Nat
  for +p: ConvSpec.Problem
  for +acc: F32
  F32

law input_channels:
  for +n: Nat
  for +input_channel_index: Nat
  for +output_channel_index: Nat
  for +output_row_index: Nat
  for +output_column_index: Nat
  for +p: ConvSpec.Problem
  for +acc: F32
  F32

law output_columns:
  for +n: Nat
  for +output_column_index: Nat
  for +output_row_index: Nat
  for +output_channel_index: Nat
  for +p: ConvSpec.Problem
  List<&2,F32>

law output_rows:
  for +n: Nat
  for +output_row_index: Nat
  for +output_channel_index: Nat
  for +p: ConvSpec.Problem
  List<&2,F32>

law output_channels:
  for +n: Nat
  for +output_channel_index: Nat
  for +p: ConvSpec.Problem
  List<&2,F32>

def cat(xs: List<&2,F32>, ys: List<&2,F32>) -> List<&2,F32>:
  List.append(&2,F32,xs,ys)

def kernel_columns(n,kernel_column_index,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc):
  match n:
    case 0n:
      acc
    case 1n+rest:
      kernel_columns(rest,1n+kernel_column_index,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,(acc + ConvSpec.sample(p,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index) * ConvSpec.weight(p,output_channel_index,input_channel_index,kernel_row_index,kernel_column_index) : F32))

def kernel_rows(n,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc):
  match n:
    case 0n:
      acc
    case 1n+rest:
      kernel_rows(rest,1n+kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,kernel_columns(ConvSpec.kernel_size(p),0n,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc))

def input_channels(n,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc):
  match n:
    case 0n:
      acc
    case 1n+rest:
      input_channels(rest,1n+input_channel_index,output_channel_index,output_row_index,output_column_index,p,kernel_rows(ConvSpec.kernel_size(p),0n,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc))

def output_columns(n,output_column_index,output_row_index,output_channel_index,p):
  match n:
    case 0n:
      Nil{}
    case 1n+rest:
      (input_channels(ConvSpec.channels(p),0n,output_channel_index,output_row_index,output_column_index,p,0.0) + ConvSpec.bias_at(p,output_channel_index) : F32) <> output_columns(rest,1n+output_column_index,output_row_index,output_channel_index,p)

def output_rows(n,output_row_index,output_channel_index,p):
  match n:
    case 0n:
      Nil{}
    case 1n+rest:
      cat(output_columns(ConvSpec.out_width(p),0n,output_row_index,output_channel_index,p),output_rows(rest,1n+output_row_index,output_channel_index,p))

def output_channels(n,output_channel_index,p):
  match n:
    case 0n:
      Nil{}
    case 1n+rest:
      cat(output_rows(ConvSpec.out_height(p),0n,output_channel_index,p),output_channels(rest,1n+output_channel_index,p))

def values(+p: ConvSpec.Problem) -> List<&2,F32>:
  output_channels(ConvSpec.outputs(p),0n,p)
