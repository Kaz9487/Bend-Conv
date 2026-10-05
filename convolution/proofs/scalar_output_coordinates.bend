# Decode the exact native output-loop index, with bounds from runtime guards.
import Base
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../scalar_kernel.bend as ScalarConv
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./reduction_coordinate_bounds.bend as Coordinates
import ./supported_padded_coordinates.bend as Padded
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/machine_limit_bounds.bend as Limits

type OutputPixel<-problem: ConvSpec.Problem> is Data:
  OutputPixel{channel: Nat,row: Nat,column: Nat,
    channel_bound: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
    row_bound: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},
    column_bound: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}}

def output_channel(-problem: ConvSpec.Problem,pixel: OutputPixel<problem>) -> Nat:
  OutputPixel{channel,row,column,hc,hr,hx} = pixel
  channel

def output_row(-problem: ConvSpec.Problem,pixel: OutputPixel<problem>) -> Nat:
  OutputPixel{channel,row,column,hc,hr,hx} = pixel
  row

def output_column(-problem: ConvSpec.Problem,pixel: OutputPixel<problem>) -> Nat:
  OutputPixel{channel,row,column,hc,hr,hx} = pixel
  column

def index(+problem: ConvSpec.Problem,pixel: OutputPixel<problem>) -> Nat:
  OutputPixel{channel,row,column,hc,hr,hx} = pixel
  RowMajor.index(channel,ConvSpec.out_height(problem),ConvSpec.out_width(problem),row,column)

type Decoded<-problem: ConvSpec.Problem,-pixel: OutputPixel<problem>> is Data:
  Decoded{
    channel: {(U32.from_nat(index(problem,pixel)) / Padded.spatial(problem) : U32) == U32.from_nat(output_channel(problem,pixel)) : U32},
    row: {(U32.from_nat(index(problem,pixel)) % Padded.spatial(problem) / U32.from_nat(ConvSpec.out_width(problem)) : U32) == U32.from_nat(output_row(problem,pixel)) : U32},
    column: {(U32.from_nat(index(problem,pixel)) % U32.from_nat(ConvSpec.out_width(problem)) : U32) == U32.from_nat(output_column(problem,pixel)) : U32}}

def decode(+limits: Machine.Limits,+problem: ConvSpec.Problem,+pixel: OutputPixel<problem>,
  valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {ScalarConv.supported_with_limits(limits,problem) == True{} : Bool}) -> Decoded<problem,pixel>:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+input_channel_count,+output_channel_count,+height,+width,+kernel,+stride,+padding} = shape
  OutputPixel{+channel,+row,+column,+channel_bound,+row_bound,+column_bound} = pixel
  +facts = Guards.valid(problem,valid)
  +support = Support.scalar(limits,problem,supported)
  +output_height = ConvSpec.out_height(problem)
  +output_width = ConvSpec.out_width(problem)
  +spatial = Nat.mul(output_height,output_width)
  +position = Nat.add(Nat.mul(row,output_width),column)
  +address = U32.from_nat(index(problem,pixel))
  +half = Machine.half_word(limits)
  +half_fits = Limits.half_word_fits(limits)
  +spatial_exact = Padded.scalar_spatial_exact(limits,problem,facts,support)
  +width_exact = Coordinates.width_exact(limits,problem,facts,support)
  +spatial_positive = Support.positive_mul(output_height,output_width,{==},{==})
  +position_bound = RowMajor.radix(row,output_height,column,output_width,row_bound,column_bound)
  +spatial_double = Limits.double_bound(limits,spatial,Padded.spatial_bound_facts(limits,problem,facts,support))
  +width_double = Limits.double_bound(limits,output_width,Coordinates.width_bound(limits,problem,facts,support))
  +index_exact = Bounds.roundtrip(index(problem,pixel),Machine.capacity(limits),Limits.capacity_fits(limits),
    Bounds.strict_le(index(problem,pixel),U32.to_nat(Machine.capacity(limits)),
      Bounds.nat_trans(index(problem,pixel),Nat.mul(output_channel_count,spatial),U32.to_nat(Machine.capacity(limits)),
        RowMajor.in_bounds(channel,output_channel_count,output_height,output_width,row,column,channel_bound,row_bound,column_bound),Padded.scalar_output_bound(limits,problem,support))))
  +channel_exact = Radix.quotient_word(address,Padded.spatial(problem),channel,position,spatial,half,half_fits,index_exact,spatial_exact,spatial_positive,position_bound,spatial_double)
  +position_exact = Radix.remainder_word(address,Padded.spatial(problem),channel,position,spatial,half,half_fits,index_exact,spatial_exact,spatial_positive,position_bound,spatial_double)
  +row_exact = Radix.quotient_word((address % Padded.spatial(problem) : U32),U32.from_nat(output_width),row,column,output_width,half,half_fits,
    position_exact,width_exact,{==},column_bound,width_double)
  +column_exact = Radix.remainder_word(address,U32.from_nat(output_width),Nat.add(Nat.mul(channel,output_height),row),column,output_width,half,half_fits,
    Equal.trans(Nat,U32.to_nat(address),index(problem,pixel),Nat.add(Nat.mul(Nat.add(Nat.mul(channel,output_height),row),output_width),column),index_exact,
      Equal.sym(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(channel,output_height),row),output_width),column),index(problem,pixel),
        RowMajor.mixed(channel,output_height,output_width,row,column))),width_exact,{==},column_bound,width_double)
  Decoded{Encoded.encoding((address / Padded.spatial(problem) : U32),channel,channel_exact),
    Encoded.encoding((address % Padded.spatial(problem) / U32.from_nat(output_width) : U32),row,row_exact),
    Encoded.encoding((address % U32.from_nat(output_width) : U32),column,column_exact)}
