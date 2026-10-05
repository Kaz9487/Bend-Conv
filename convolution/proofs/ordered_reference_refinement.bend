import Base
import ../specification.bend as ConvSpec
import ../reference.bend as Reference
import ../ordered_reference.bend as OrderedReference

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
  {OrderedReference.kernel_columns(n,kernel_column_index,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc) == Reference.sum(ConvSpec.kernel_columns(n,kernel_column_index,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p),acc) : F32}

law kernel_rows:
  for +n: Nat
  for +kernel_row_index: Nat
  for +input_channel_index: Nat
  for +output_channel_index: Nat
  for +output_row_index: Nat
  for +output_column_index: Nat
  for +p: ConvSpec.Problem
  for +acc: F32
  {OrderedReference.kernel_rows(n,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc) == Reference.sum(ConvSpec.kernel_rows(n,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p),acc) : F32}

law input_channels:
  for +n: Nat
  for +input_channel_index: Nat
  for +output_channel_index: Nat
  for +output_row_index: Nat
  for +output_column_index: Nat
  for +p: ConvSpec.Problem
  for +acc: F32
  {OrderedReference.input_channels(n,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc) == Reference.sum(ConvSpec.input_channels(n,input_channel_index,output_channel_index,output_row_index,output_column_index,p),acc) : F32}

law output_columns:
  for +n: Nat
  for +output_column_index: Nat
  for +output_row_index: Nat
  for +output_channel_index: Nat
  for +p: ConvSpec.Problem
  {OrderedReference.output_columns(n,output_column_index,output_row_index,output_channel_index,p) == Reference.evaluate(ConvSpec.output_columns(n,output_column_index,output_row_index,output_channel_index,p)) : List<&2,F32>}

law output_rows:
  for +n: Nat
  for +output_row_index: Nat
  for +output_channel_index: Nat
  for +p: ConvSpec.Problem
  {OrderedReference.output_rows(n,output_row_index,output_channel_index,p) == Reference.evaluate(ConvSpec.output_rows(n,output_row_index,output_channel_index,p)) : List<&2,F32>}

law output_channels:
  for +n: Nat
  for +output_channel_index: Nat
  for +p: ConvSpec.Problem
  {OrderedReference.output_channels(n,output_channel_index,p) == Reference.evaluate(ConvSpec.output_channels(n,output_channel_index,p)) : List<&2,F32>}


law sum_cat:
  for +xs: List<&2,ConvSpec.Product>
  for +ys: List<&2,ConvSpec.Product>
  for +acc: F32
  {Reference.sum(ys,Reference.sum(xs,acc)) == Reference.sum(ConvSpec.cat_products(xs,ys),acc) : F32}

def sum_cat(xs,ys,acc):
  match xs:
    case Nil{}:
      {==}
    case ConvSpec.Product{x,w} <> tail:
      sum_cat(tail,ys,(acc + x * w : F32))

law evaluate_cat:
  for +xs: List<&2,ConvSpec.DotJob>
  for +ys: List<&2,ConvSpec.DotJob>
  {OrderedReference.cat(Reference.evaluate(xs),
      Reference.evaluate(ys)) == Reference.evaluate(ConvSpec.cat_jobs(xs,ys)) : List<&2,F32>}

def evaluate_cat(xs,ys):
  match xs:
    case Nil{}:
      {==}
    case ConvSpec.DotJob{products,bias} <> tail:
      %evaluate_cat(tail,ys) : {(Reference.sum(products,
            0.0) + bias : F32) <> OrderedReference.cat(Reference.evaluate(tail),
          Reference.evaluate(ys)) == (Reference.sum(products,0.0) + bias : F32) <> _ : List<&2,F32>}
      {==}

def kernel_columns(n,kernel_column_index,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc):
  match n:
    case 0n:
      {==}
    case 1n+rest:
      kernel_columns(rest,1n+kernel_column_index,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,(acc + ConvSpec.sample(p,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index) * ConvSpec.weight(p,output_channel_index,input_channel_index,kernel_row_index,
            kernel_column_index) : F32))

def kernel_rows(n,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc):
  match n:
    case 0n:
      {==}
    case 1n+rest:
      %sum_cat(ConvSpec.kernel_columns(ConvSpec.kernel_size(p),0n,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p),ConvSpec.kernel_rows(rest,1n+kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p),
        acc) : {OrderedReference.kernel_rows(rest,1n+kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,OrderedReference.kernel_columns(ConvSpec.kernel_size(p),0n,kernel_row_index,input_channel_index,output_channel_index,output_row_index,
            output_column_index,p,acc)) == _ : F32}
      %kernel_columns(ConvSpec.kernel_size(p),0n,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc) : {OrderedReference.kernel_rows(rest,1n+kernel_row_index,input_channel_index,output_channel_index,
          output_row_index,output_column_index,p,OrderedReference.kernel_columns(ConvSpec.kernel_size(p),0n,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,
            acc)) == Reference.sum(ConvSpec.kernel_rows(rest,1n+kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p),_) : F32}
      kernel_rows(rest,1n+kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,OrderedReference.kernel_columns(ConvSpec.kernel_size(p),0n,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc))

def input_channels(n,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc):
  match n:
    case 0n:
      {==}
    case 1n+rest:
      %sum_cat(ConvSpec.kernel_rows(ConvSpec.kernel_size(p),0n,input_channel_index,output_channel_index,output_row_index,output_column_index,p),ConvSpec.input_channels(rest,1n+input_channel_index,output_channel_index,output_row_index,output_column_index,p),
        acc) : {OrderedReference.input_channels(rest,1n+input_channel_index,output_channel_index,output_row_index,output_column_index,p,OrderedReference.kernel_rows(ConvSpec.kernel_size(p),0n,input_channel_index,output_channel_index,output_row_index,output_column_index,p,
            acc)) == _ : F32}
      %kernel_rows(ConvSpec.kernel_size(p),0n,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc) : {OrderedReference.input_channels(rest,1n+input_channel_index,output_channel_index,output_row_index,output_column_index,p,
          OrderedReference.kernel_rows(ConvSpec.kernel_size(p),0n,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc)) == Reference.sum(ConvSpec.input_channels(rest,
            1n+input_channel_index,output_channel_index,output_row_index,output_column_index,p),_) : F32}
      input_channels(rest,1n+input_channel_index,output_channel_index,output_row_index,output_column_index,p,OrderedReference.kernel_rows(ConvSpec.kernel_size(p),0n,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc))

def output_columns(n,output_column_index,output_row_index,output_channel_index,p):
  match n:
    case 0n:
      {==}
    case 1n+rest:
      %input_channels(ConvSpec.channels(p),0n,output_channel_index,output_row_index,output_column_index,p,0.0) : {(OrderedReference.input_channels(ConvSpec.channels(p),0n,
            output_channel_index,output_row_index,output_column_index,p,0.0) + ConvSpec.bias_at(p,output_channel_index) : F32) <> OrderedReference.output_columns(rest,1n+output_column_index,output_row_index,output_channel_index,p) == (_ + ConvSpec.bias_at(p,
            output_channel_index) : F32) <> Reference.evaluate(ConvSpec.output_columns(rest,1n+output_column_index,output_row_index,output_channel_index,p)) : List<&2,F32>}
      %output_columns(rest,1n+output_column_index,output_row_index,output_channel_index,p) : {(OrderedReference.input_channels(ConvSpec.channels(p),0n,output_channel_index,output_row_index,output_column_index,p,
            0.0) + ConvSpec.bias_at(p,output_channel_index) : F32) <> OrderedReference.output_columns(rest,1n+output_column_index,output_row_index,output_channel_index,
          p) == (OrderedReference.input_channels(ConvSpec.channels(p),0n,output_channel_index,output_row_index,output_column_index,p,0.0) + ConvSpec.bias_at(p,output_channel_index) : F32) <> _ : List<&2,F32>}
      {==}

def output_rows(n,output_row_index,output_channel_index,p):
  match n:
    case 0n:
      {==}
    case 1n+rest:
      %evaluate_cat(ConvSpec.output_columns(ConvSpec.out_width(p),0n,output_row_index,output_channel_index,p),ConvSpec.output_rows(rest,1n+output_row_index,output_channel_index,
          p)) : {OrderedReference.cat(OrderedReference.output_columns(ConvSpec.out_width(p),0n,output_row_index,output_channel_index,p),OrderedReference.output_rows(rest,
            1n+output_row_index,output_channel_index,p)) == _ : List<&2,F32>}
      %output_columns(ConvSpec.out_width(p),0n,output_row_index,output_channel_index,
        p) : {OrderedReference.cat(OrderedReference.output_columns(ConvSpec.out_width(p),0n,output_row_index,output_channel_index,p),OrderedReference.output_rows(rest,1n+output_row_index,
            output_channel_index,p)) == OrderedReference.cat(_,Reference.evaluate(ConvSpec.output_rows(rest,1n+output_row_index,output_channel_index,p))) : List<&2,F32>}
      %output_rows(rest,1n+output_row_index,output_channel_index,p) : {OrderedReference.cat(OrderedReference.output_columns(ConvSpec.out_width(p),0n,output_row_index,
            output_channel_index,p),OrderedReference.output_rows(rest,1n+output_row_index,output_channel_index,
            p)) == OrderedReference.cat(OrderedReference.output_columns(ConvSpec.out_width(p),0n,output_row_index,output_channel_index,p),_) : List<&2,F32>}
      {==}

def output_channels(n,output_channel_index,p):
  match n:
    case 0n:
      {==}
    case 1n+rest:
      %evaluate_cat(ConvSpec.output_rows(ConvSpec.out_height(p),0n,output_channel_index,p),ConvSpec.output_channels(rest,1n+output_channel_index,
          p)) : {OrderedReference.cat(OrderedReference.output_rows(ConvSpec.out_height(p),0n,output_channel_index,p),OrderedReference.output_channels(rest,
            1n+output_channel_index,p)) == _ : List<&2,F32>}
      %output_rows(ConvSpec.out_height(p),0n,output_channel_index,
        p) : {OrderedReference.cat(OrderedReference.output_rows(ConvSpec.out_height(p),0n,output_channel_index,p),OrderedReference.output_channels(rest,1n+output_channel_index,
            p)) == OrderedReference.cat(_,Reference.evaluate(ConvSpec.output_channels(rest,1n+output_channel_index,p))) : List<&2,F32>}
      %output_channels(rest,1n+output_channel_index,p) : {OrderedReference.cat(OrderedReference.output_rows(ConvSpec.out_height(p),0n,output_channel_index,
            p),OrderedReference.output_channels(rest,1n+output_channel_index,p)) == OrderedReference.cat(OrderedReference.output_rows(ConvSpec.out_height(p),0n,
            output_channel_index,p),_) : List<&2,F32>}
      {==}

