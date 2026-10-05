import Base
import ./storage_buffer.bend as Storage
import ./proofs/storage_certificate.bend as Certificate
import ./proofs/storage_clone_certificate.bend as CertificateClone

def buffers(~Element: Data,+length: Nat,pair: Array<Element> & Array<Element>,certificates: CertificateClone.Copies<Element,pair>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  (left,right) = pair
  CertificateClone.Copies{left_certificate,right_certificate} = certificates
  (Storage.Buffer{left,length,left_certificate},Storage.Buffer{right,length,right_certificate})

def clone(~Element: Data,input: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  Storage.Buffer{input,length,certificate} = input
  Certificate.Certificate{+depth,+equation} = certificate
  buffers(~Element,length,Array.clone(Element,input),CertificateClone.clone(~Element,input,
    Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation}))
