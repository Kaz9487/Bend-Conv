# Save a tensor as a raw FP32 file and load it again.
import Base
import stelliferous@0.0.2.0/tensor_f32.bend as F

def text(value: F32) -> String:
  +value = value
  F32.show(value)

def main() -> IO(Unit):
  do IO<Unit>:
    # 1.0 .. 6.0 as two rows, written as 24 bytes
    saved  : F.Tensor() <- F.save_raw("out/values.bin",F.reshape(F.arange(1.0,6),[2,3]))
    # the same file, read with its shape
    loaded : F.Tensor() <- F.load_raw("out/values.bin",[2,3])
    # the sums, or [] if a step above failed
    IO.print(List.show(~&2,~F32,~text,F.to_list_or(F.add(saved,loaded),[])))
