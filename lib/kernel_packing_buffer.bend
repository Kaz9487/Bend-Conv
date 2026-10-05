import Base
import ./kernel_spatial_packing.bend as Packing
import ./kernel_tile.bend as Tile
import ./storage_buffer.bend as Storage
import ./proofs/kernel_packing_certificate.bend as PackingCertificate
import ./proofs/storage_certificate.bend as Certificate
import ./proofs/traversal_certificate.bend as TraversalCertificate

def buffers(input_length: Nat,output_length: Nat,result: Array<F32> & Array<Tile.Tile8>,
  certificates: TraversalCertificate.Certificates<Array<F32>,Tile.Tile8,Certificate.family(~F32),result>) ->
  Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>:
  (input,output) = result
  TraversalCertificate.Certificates{source_certificate,output_certificate} = certificates
  (Storage.Buffer{input,input_length,source_certificate},Storage.Buffer{output,output_length,output_certificate})

def run(+call: Packing.Call,input: Storage.Buffer<F32>,output: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>:
  Storage.Buffer{input,input_length,input_certificate} = input
  Storage.Buffer{output,output_length,output_certificate} = output
  buffers(input_length,output_length,Packing.run(call,input,output),PackingCertificate.certificates(call,input,output,input_certificate,output_certificate))
