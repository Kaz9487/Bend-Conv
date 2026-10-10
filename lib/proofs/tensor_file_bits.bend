# Reading a value's bits and rebuilding it are inverse: a saved value loads
# with every bit, NaN payloads included.
import Base
import ../tensor_file.bend as Files

def word_round_trip(word: U32) -> {Files.to_bits(Files.from_bits(word)) == word : U32}:
  match word:
    case U32{bits}: {==}

def value_round_trip(value: F32) -> {Files.from_bits(Files.to_bits(value)) == value : F32}:
  match value:
    case F32{bits}: {==}
