import Base
import ./kernel_gather.bend as Gather
import ./storage_buffer.bend as Storage
import ./proofs/kernel_gather_certificate.bend as GatherCertificate

def run(+channels: U32,+offset: U32,+positions: U32,+span: U32,input: Storage.Buffer<F32>,output: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Storage.Buffer{input,input_length,input_certificate} = input
  Storage.Buffer{output,length,output_certificate} = output
  Storage.Buffer{Gather.gather_block(channels,offset,positions,span,input,output),length,
    GatherCertificate.certificate(channels,offset,positions,span,input,output,input_certificate,output_certificate)}
