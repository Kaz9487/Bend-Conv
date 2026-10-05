import Base
import ../math_sequence.bend as Sequence
import ../math_fp32.bend as Fp32
import ./indexed_sequence_laws.bend as SequenceLaws

law zip_mac_at_index:
  for +length: Nat
  for +index: Sequence.Index(length)
  for +accumulator: Sequence.Values(F32,length)
  for +input: Sequence.Values(F32,length)
  for +weight: F32
  {Sequence.get(F32,length,index,Fp32.zip_mac(length,accumulator,input,weight)) == Fp32.multiply_add(Sequence.get(F32,length,index,accumulator),
      Sequence.get(F32,length,index,input),weight) : F32}
def zip_mac_at_index(length,index,accumulator,input,weight):
  SequenceLaws.zip_map_at_index(~F32,~F32,~Fp32.weighted_add,length,index,weight,accumulator,input)