import Base
import ./kernel_tile.bend as Tile
import ./kernel_gemm.bend as Gemm
import ./storage_buffer.bend as Storage
import ./proofs/kernel_gemm_certificate.bend as GemmCertificate
import ./kernel_weights.bend as Weights
import ./proofs/kernel_weights_certificate.bend as WeightCertificate

# The live certificate is updated once, outside every output and MAC loop.
def run(+rows: U32,input: Storage.Buffer<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,
  output: Storage.Buffer<F32>,+columns: U32,+inner: U32) -> Storage.Buffer<F32>:
  Storage.Buffer{input,input_length,input_certificate} = input
  Storage.Buffer{weights,weight_length,weight_certificate} = weights
  Storage.Buffer{bias,bias_length,bias_certificate} = bias
  Storage.Buffer{output,length,certificate} = output
  Storage.Buffer{Gemm.run(rows,input,weights,bias,output,columns,inner),length,
    GemmCertificate.certificate(rows,input,weights,bias,output,columns,inner,input_certificate,weight_certificate,bias_certificate,certificate)}

def reorder(+rows: U32,+inner: U32,weights: Storage.Buffer<F32>) -> Storage.Buffer<Tile.Tile8>:
  Storage.Buffer{weights,length,certificate} = weights
  Storage.Buffer{Weights.run(rows,inner,weights),U32.to_nat((((rows + 7) / 8) * inner : U32)),
    WeightCertificate.certificate(rows,inner,weights,certificate)}
