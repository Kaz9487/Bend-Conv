import Base
import ../kernel_spatial_packing.bend as Packing
import ../kernel_tile.bend as Tile
import ./kernel_packing_witness.bend as PackingWitness
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./traversal_witness.bend as TraversalWitness
import ./traversal_certificate.bend as TraversalCertificate

def preserves(+depth: Nat,call: Packing.Call,-input: Array<F32>,-output: Array<Tile.Tile8>,input_certificate: Certificate.Certificate<F32,input>,
  equation: {output == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,depth,Tree.reflect(Tile.Tile8,output))) : Array<Tile.Tile8>}) ->
  TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,Packing.run(call,input,output)>:
  Packing.Call{+count,+start,+h,+w,+k,+s,+p,+ow,+inner} = call
  PackingWitness.pack(depth,count,start,input,output,h,w,k,s,p,ow,inner,Certificate.witness(F32,input,input_certificate),Certificate.witness_at(Tile.Tile8,depth,output,equation))

def certificates(call: Packing.Call,-input: Array<F32>,-output: Array<Tile.Tile8>,input_certificate: Certificate.Certificate<F32,input>,output_certificate: Certificate.Certificate<Tile.Tile8,output>) ->
  TraversalCertificate.Certificates<Array<F32>,Tile.Tile8,Certificate.family(~F32),Packing.run(call,input,output)>:
  Certificate.Certificate{+depth,+equation} = input_certificate
  TraversalCertificate.operation(~Array<F32>,~Tile.Tile8,~Packing.Call,~Certificate.family(~F32),~Packing.run,~preserves,
    call,input,output,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},output_certificate)
