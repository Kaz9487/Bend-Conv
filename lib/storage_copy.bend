# Internal contiguous ranges. Callers establish bounds; the shared copy proof
# and storage certificates cover the actual loop without a per-element check.
import Base
import ./storage_buffer.bend as Storage
import ./tensor_view.bend as Views
import ./tensor_access.bend as Access
import ./tensor_operations.bend as Operations
import ./proofs/tensor_access_certificate.bend as AccessCertificate

def range(~Element: Data,+count: U32,+source: U32,+destination: U32,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) ->
  Storage.Buffer<Element> & Storage.Buffer<Element>:
  Storage.Buffer{input,input_length,input_certificate} = input
  Storage.Buffer{output,output_length,output_certificate} = output
  +view = {Views.View{[Views.Axis{count,1}],source} : Views.View}
  Operations.pair_buffers(~Element,input_length,output_length,
    Access.copy_run(~Element,True{},U32.to_nat(count),view,destination,input,output),
    AccessCertificate.copy(~Element,True{},U32.to_nat(count),view,destination,input,output,input_certificate,output_certificate))
