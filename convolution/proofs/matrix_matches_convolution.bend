import Base
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ../matrix_specification.bend as MatrixSpec
import ../specification.bend as ConvSpec
import ../ordered_reference.bend as OrderedReference

import ../../lib/proofs/ordered_reduction_split.bend as Ordered
import ./sequence_congruence.bend as Sequences
import ./matrix_convolution_law.bend as MatrixLaws
import ../specification_law.bend as SpecificationLaw
import ../reference.bend as Reference
import ./reference_refinement.bend as ProofsReferenceLaws

def kcols(+n: Nat,+kernel_column_index: Nat,+kernel_row_index: Nat,+input_channel_index: Nat,+output_channel_index: Nat,+output_row_index: Nat,+output_column_index: Nat,+p: ConvSpec.Problem,+acc: F32) -> {OrderedMatrix.dot(~Nat,
    ~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,MatrixSpec.kcols(n,kernel_column_index,kernel_row_index,input_channel_index),output_channel_index,
    MatrixSpec.Position{output_row_index,output_column_index},p,acc) == OrderedReference.kernel_columns(n,kernel_column_index,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc) : F32}:
  match n:
    case 0n:
      {==}
    case 1n+ +rest:
      kcols(rest,1n+kernel_column_index,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,(acc + ConvSpec.sample(p,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index) * ConvSpec.weight(p,output_channel_index,input_channel_index,kernel_row_index,kernel_column_index) : F32))

def krows(+n: Nat,+kernel_row_index: Nat,+input_channel_index: Nat,+output_channel_index: Nat,+output_row_index: Nat,+output_column_index: Nat,+p: ConvSpec.Problem,+acc: F32) -> {OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,
    ~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,MatrixSpec.krows(n,kernel_row_index,input_channel_index,ConvSpec.kernel_size(p)),output_channel_index,
    MatrixSpec.Position{output_row_index,output_column_index},p,acc) == OrderedReference.kernel_rows(n,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc) : F32}:
  match n:
    case 0n:
      {==}
    case 1n+ +rest:
      %Equal.sym(F32,OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,
          OrderedMatrix.cat(MatrixSpec.Reduction,MatrixSpec.kcols(ConvSpec.kernel_size(p),0n,kernel_row_index,input_channel_index),MatrixSpec.krows(rest,1n+kernel_row_index,input_channel_index,
              ConvSpec.kernel_size(p))),output_channel_index,MatrixSpec.Position{output_row_index,output_column_index},p,acc),OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,
          ~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,MatrixSpec.krows(rest,1n+kernel_row_index,input_channel_index,ConvSpec.kernel_size(p)),output_channel_index,
          MatrixSpec.Position{output_row_index,output_column_index},p,OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,
            ~MatrixSpec.weight,MatrixSpec.kcols(ConvSpec.kernel_size(p),0n,kernel_row_index,input_channel_index),output_channel_index,MatrixSpec.Position{output_row_index,output_column_index},p,acc)),
        Ordered.reduction_split(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,
          ~MatrixSpec.weight,MatrixSpec.kcols(ConvSpec.kernel_size(p),0n,kernel_row_index,input_channel_index),MatrixSpec.krows(rest,1n+kernel_row_index,input_channel_index,ConvSpec.kernel_size(p)),output_channel_index,
          MatrixSpec.Position{output_row_index,output_column_index},p,acc)) : {_ == OrderedReference.kernel_rows(rest,1n+kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,
          OrderedReference.kernel_columns(ConvSpec.kernel_size(p),0n,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc)) : F32}
      %kcols(ConvSpec.kernel_size(p),0n,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc) : {OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,
          ~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,MatrixSpec.krows(rest,1n+kernel_row_index,input_channel_index,ConvSpec.kernel_size(p)),output_channel_index,
          MatrixSpec.Position{output_row_index,output_column_index},p,OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,
            ~MatrixSpec.weight,MatrixSpec.kcols(ConvSpec.kernel_size(p),0n,kernel_row_index,input_channel_index),output_channel_index,MatrixSpec.Position{output_row_index,output_column_index},p,
            acc)) == OrderedReference.kernel_rows(rest,1n+kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,_) : F32}
      krows(rest,1n+kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index,p,OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,
          ~MatrixSpec.weight,MatrixSpec.kcols(ConvSpec.kernel_size(p),0n,kernel_row_index,input_channel_index),output_channel_index,MatrixSpec.Position{output_row_index,output_column_index},p,acc))

def kchannels(+n: Nat,+input_channel_index: Nat,+output_channel_index: Nat,+output_row_index: Nat,+output_column_index: Nat,+p: ConvSpec.Problem,+acc: F32) -> {OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,
    ~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,MatrixSpec.kchannels(n,input_channel_index,ConvSpec.kernel_size(p)),output_channel_index,
    MatrixSpec.Position{output_row_index,output_column_index},p,acc) == OrderedReference.input_channels(n,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc) : F32}:
  match n:
    case 0n:
      {==}
    case 1n+ +rest:
      %Equal.sym(F32,OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,
          OrderedMatrix.cat(MatrixSpec.Reduction,MatrixSpec.krows(ConvSpec.kernel_size(p),0n,input_channel_index,ConvSpec.kernel_size(p)),
            MatrixSpec.kchannels(rest,1n+input_channel_index,ConvSpec.kernel_size(p))),output_channel_index,MatrixSpec.Position{output_row_index,output_column_index},p,acc),OrderedMatrix.dot(~Nat,
          ~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,MatrixSpec.kchannels(rest,1n+input_channel_index,
            ConvSpec.kernel_size(p)),output_channel_index,MatrixSpec.Position{output_row_index,output_column_index},p,OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,
            ~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,MatrixSpec.krows(ConvSpec.kernel_size(p),0n,input_channel_index,ConvSpec.kernel_size(p)),output_channel_index,
            MatrixSpec.Position{output_row_index,output_column_index},p,acc)),Ordered.reduction_split(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,
          ~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,MatrixSpec.krows(ConvSpec.kernel_size(p),0n,input_channel_index,ConvSpec.kernel_size(p)),
          MatrixSpec.kchannels(rest,1n+input_channel_index,ConvSpec.kernel_size(p)),output_channel_index,MatrixSpec.Position{output_row_index,output_column_index},p,
          acc)) : {_ == OrderedReference.input_channels(rest,1n+input_channel_index,output_channel_index,output_row_index,output_column_index,p,OrderedReference.kernel_rows(ConvSpec.kernel_size(p),0n,input_channel_index,output_channel_index,output_row_index,
            output_column_index,p,acc)) : F32}
      %krows(ConvSpec.kernel_size(p),0n,input_channel_index,output_channel_index,output_row_index,output_column_index,p,acc) : {OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,
          ~MatrixSpec.patch,~MatrixSpec.weight,MatrixSpec.kchannels(rest,1n+input_channel_index,ConvSpec.kernel_size(p)),output_channel_index,MatrixSpec.Position{output_row_index,output_column_index},p,
          OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,
            MatrixSpec.krows(ConvSpec.kernel_size(p),0n,input_channel_index,ConvSpec.kernel_size(p)),output_channel_index,MatrixSpec.Position{output_row_index,output_column_index},p,
            acc)) == OrderedReference.input_channels(rest,1n+input_channel_index,output_channel_index,output_row_index,output_column_index,p,_) : F32}
      kchannels(rest,1n+input_channel_index,output_channel_index,output_row_index,output_column_index,p,OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,
          ~MatrixSpec.weight,MatrixSpec.krows(ConvSpec.kernel_size(p),0n,input_channel_index,ConvSpec.kernel_size(p)),output_channel_index,MatrixSpec.Position{output_row_index,output_column_index},p,acc))

def columns(+n: Nat,+output_column_index: Nat,+output_row_index: Nat,+output_channel_index: Nat,+p: ConvSpec.Problem) -> {OrderedMatrix.row(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,
    ~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.columns(n,output_column_index,output_row_index),
    MatrixSpec.kchannels(ConvSpec.channels(p),0n,ConvSpec.kernel_size(p)),output_channel_index,p) == OrderedReference.output_columns(n,output_column_index,output_row_index,output_channel_index,p) : List<&2,F32>}:
  match n:
    case 0n:
      {==}
    case 1n+ +rest:
      %kchannels(ConvSpec.channels(p),0n,output_channel_index,output_row_index,output_column_index,p,0.0) : {(OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,
            ~MatrixSpec.patch,~MatrixSpec.weight,MatrixSpec.kchannels(ConvSpec.channels(p),0n,ConvSpec.kernel_size(p)),output_channel_index,MatrixSpec.Position{output_row_index,
              output_column_index},p,0.0) + ConvSpec.bias_at(p,output_channel_index) : F32) <> OrderedMatrix.row(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,
          ~MatrixSpec.patch,~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.columns(rest,1n+output_column_index,output_row_index),MatrixSpec.kchannels(ConvSpec.channels(p),0n,
            ConvSpec.kernel_size(p)),output_channel_index,p) == (_ + ConvSpec.bias_at(p,output_channel_index) : F32) <> OrderedReference.output_columns(rest,1n+output_column_index,output_row_index,output_channel_index,
          p) : List<&2,F32>}
      %columns(rest,1n+output_column_index,output_row_index,output_channel_index,p) : {(OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,
            ~MatrixSpec.weight,MatrixSpec.kchannels(ConvSpec.channels(p),0n,ConvSpec.kernel_size(p)),output_channel_index,MatrixSpec.Position{output_row_index,output_column_index},p,
            0.0) + ConvSpec.bias_at(p,output_channel_index) : F32) <> OrderedMatrix.row(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,
          ~MatrixSpec.patch,~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.columns(rest,1n+output_column_index,output_row_index),MatrixSpec.kchannels(ConvSpec.channels(p),0n,
            ConvSpec.kernel_size(p)),output_channel_index,p) == (OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,
            ~MatrixSpec.patch,~MatrixSpec.weight,MatrixSpec.kchannels(ConvSpec.channels(p),0n,ConvSpec.kernel_size(p)),output_channel_index,MatrixSpec.Position{output_row_index,
              output_column_index},p,0.0) + ConvSpec.bias_at(p,output_channel_index) : F32) <> _ : List<&2,F32>}
      {==}

law cat_stream:
  for +a: List<&2,F32>
  for +b: List<&2,F32>
  {OrderedMatrix.cat(F32,a,b) == OrderedReference.cat(a,b) : List<&2,F32>}
def cat_stream(a,b):
  Sequences.stream_cat(a,b)

def rows(+n: Nat,+output_row_index: Nat,+output_channel_index: Nat,+p: ConvSpec.Problem) -> {OrderedMatrix.row(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,
    ~MatrixSpec.patch,~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.rows(n,output_row_index,ConvSpec.out_width(p)),MatrixSpec.kchannels(ConvSpec.channels(p),
      0n,ConvSpec.kernel_size(p)),output_channel_index,p) == OrderedReference.output_rows(n,output_row_index,output_channel_index,p) : List<&2,F32>}:
  match n:
    case 0n:
      {==}
    case 1n+ +rest:
      %Equal.sym(List<&2,F32>,OrderedMatrix.row(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,
          ~MatrixSpec.weight,~ConvSpec.bias_at,OrderedMatrix.cat(MatrixSpec.Pixel,MatrixSpec.columns(ConvSpec.out_width(p),0n,output_row_index),
            MatrixSpec.rows(rest,1n+output_row_index,ConvSpec.out_width(p))),MatrixSpec.kchannels(ConvSpec.channels(p),0n,ConvSpec.kernel_size(p)),output_channel_index,p),
        OrderedMatrix.cat(F32,OrderedMatrix.row(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,
            ~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.columns(ConvSpec.out_width(p),0n,output_row_index),MatrixSpec.kchannels(ConvSpec.channels(p),0n,
              ConvSpec.kernel_size(p)),output_channel_index,p),OrderedMatrix.row(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,
            ~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.rows(rest,1n+output_row_index,ConvSpec.out_width(p)),MatrixSpec.kchannels(ConvSpec.channels(p),0n,
              ConvSpec.kernel_size(p)),output_channel_index,p)),Ordered.column_split(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,
          ~MatrixSpec.patch,~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.columns(ConvSpec.out_width(p),0n,output_row_index),MatrixSpec.rows(rest,1n+output_row_index,
            ConvSpec.out_width(p)),MatrixSpec.kchannels(ConvSpec.channels(p),0n,ConvSpec.kernel_size(p)),output_channel_index,
          p)) : {_ == OrderedReference.cat(OrderedReference.output_columns(ConvSpec.out_width(p),0n,output_row_index,output_channel_index,p),
          OrderedReference.output_rows(rest,1n+output_row_index,output_channel_index,p)) : List<&2,F32>}
      %columns(ConvSpec.out_width(p),0n,output_row_index,output_channel_index,p) : {OrderedMatrix.cat(F32,OrderedMatrix.row(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,
            ~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.columns(ConvSpec.out_width(p),0n,output_row_index),
            MatrixSpec.kchannels(ConvSpec.channels(p),0n,ConvSpec.kernel_size(p)),output_channel_index,p),OrderedMatrix.row(~Nat,~MatrixSpec.Reduction,
            ~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.rows(rest,1n+output_row_index,
              ConvSpec.out_width(p)),MatrixSpec.kchannels(ConvSpec.channels(p),0n,ConvSpec.kernel_size(p)),output_channel_index,p)) == OrderedReference.cat(_,
          OrderedReference.output_rows(rest,1n+output_row_index,output_channel_index,p)) : List<&2,F32>}
      %rows(rest,1n+output_row_index,output_channel_index,p) : {OrderedMatrix.cat(F32,OrderedMatrix.row(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,
            ~MatrixSpec.patch,~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.columns(ConvSpec.out_width(p),0n,output_row_index),
            MatrixSpec.kchannels(ConvSpec.channels(p),0n,ConvSpec.kernel_size(p)),output_channel_index,p),OrderedMatrix.row(~Nat,~MatrixSpec.Reduction,
            ~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.rows(rest,1n+output_row_index,
              ConvSpec.out_width(p)),MatrixSpec.kchannels(ConvSpec.channels(p),0n,ConvSpec.kernel_size(p)),output_channel_index,
            p)) == OrderedReference.cat(OrderedMatrix.row(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,
            ~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.columns(ConvSpec.out_width(p),0n,output_row_index),MatrixSpec.kchannels(ConvSpec.channels(p),0n,
              ConvSpec.kernel_size(p)),output_channel_index,p),_) : List<&2,F32>}
      cat_stream(OrderedMatrix.row(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,
          ~ConvSpec.bias_at,MatrixSpec.columns(ConvSpec.out_width(p),0n,output_row_index),MatrixSpec.kchannels(ConvSpec.channels(p),0n,
            ConvSpec.kernel_size(p)),output_channel_index,p),OrderedMatrix.row(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,
          ~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.rows(rest,1n+output_row_index,ConvSpec.out_width(p)),MatrixSpec.kchannels(ConvSpec.channels(p),0n,
            ConvSpec.kernel_size(p)),output_channel_index,p))

def channels(+n: Nat,+output_channel_index: Nat,+p: ConvSpec.Problem) -> {OrderedMatrix.matmul(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,
    ~MatrixSpec.patch,~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.channels(n,output_channel_index),MatrixSpec.kchannels(ConvSpec.channels(p),0n,
      ConvSpec.kernel_size(p)),MatrixSpec.rows(ConvSpec.out_height(p),0n,ConvSpec.out_width(p)),p) == OrderedReference.output_channels(n,output_channel_index,
    p) : List<&2,F32>}:
  match n:
    case 0n:
      {==}
    case 1n+ +rest:
      %rows(ConvSpec.out_height(p),0n,output_channel_index,p) : {OrderedMatrix.cat(F32,OrderedMatrix.row(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,
            ~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.rows(ConvSpec.out_height(p),0n,
              ConvSpec.out_width(p)),MatrixSpec.kchannels(ConvSpec.channels(p),0n,ConvSpec.kernel_size(p)),output_channel_index,p),OrderedMatrix.matmul(~Nat,
            ~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,~ConvSpec.bias_at,
            MatrixSpec.channels(rest,1n+output_channel_index),MatrixSpec.kchannels(ConvSpec.channels(p),0n,ConvSpec.kernel_size(p)),
            MatrixSpec.rows(ConvSpec.out_height(p),0n,ConvSpec.out_width(p)),p)) == OrderedReference.cat(_,
          OrderedReference.output_channels(rest,1n+output_channel_index,p)) : List<&2,F32>}
      %channels(rest,1n+output_channel_index,p) : {OrderedMatrix.cat(F32,OrderedMatrix.row(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,
            ~MatrixSpec.patch,~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.rows(ConvSpec.out_height(p),0n,ConvSpec.out_width(p)),
            MatrixSpec.kchannels(ConvSpec.channels(p),0n,ConvSpec.kernel_size(p)),output_channel_index,p),OrderedMatrix.matmul(~Nat,~MatrixSpec.Reduction,
            ~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.channels(rest,1n+output_channel_index),
            MatrixSpec.kchannels(ConvSpec.channels(p),0n,ConvSpec.kernel_size(p)),MatrixSpec.rows(ConvSpec.out_height(p),0n,
              ConvSpec.out_width(p)),p)) == OrderedReference.cat(OrderedMatrix.row(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,
            ~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.rows(ConvSpec.out_height(p),0n,
              ConvSpec.out_width(p)),MatrixSpec.kchannels(ConvSpec.channels(p),0n,ConvSpec.kernel_size(p)),output_channel_index,p),_) : List<&2,F32>}
      cat_stream(OrderedMatrix.row(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,
          ~ConvSpec.bias_at,MatrixSpec.rows(ConvSpec.out_height(p),0n,ConvSpec.out_width(p)),MatrixSpec.kchannels(ConvSpec.channels(p),0n,
            ConvSpec.kernel_size(p)),output_channel_index,p),OrderedMatrix.matmul(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,
          ~MatrixSpec.weight,~ConvSpec.bias_at,MatrixSpec.channels(rest,1n+output_channel_index),MatrixSpec.kchannels(ConvSpec.channels(p),0n,
            ConvSpec.kernel_size(p)),MatrixSpec.rows(ConvSpec.out_height(p),0n,ConvSpec.out_width(p)),p))

def validated(+ok: Bool,+p: ConvSpec.Problem) -> {MatrixSpec.finish(ok,p) == Reference.finish(ok,p) : ConvSpec.ConvOutcome}:
  match ok:
    case False{}:
      {==}
    case True{}:
      %channels(ConvSpec.outputs(p),0n,p) : {ConvSpec.Accepted{ConvSpec.out_height(p),ConvSpec.out_width(p),
          MatrixSpec.values(p)} == ConvSpec.Accepted{ConvSpec.out_height(p),ConvSpec.out_width(p),_} : ConvSpec.ConvOutcome}
      {==}

def MatrixLaws.convolution(p):
  %SpecificationLaw.convolution(p) : {MatrixSpec.infer(p) == _ : ConvSpec.ConvOutcome}
  validated(ConvSpec.valid(p),p)
