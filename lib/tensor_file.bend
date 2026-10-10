# Raw little-endian FP32 files and NumPy .npy files through the official File effects.
import Base
import ./storage_buffer.bend as Buffers
import ./tensor.bend as Tensors
import ./tensor_layout.bend as Layout

def Tensor() -> Type:
  Tensors.Tensor<F32>

def Read() -> Type:
  File & Result<&1,&1,U32 & String,List<&2,U32>>

def Written() -> Type:
  File & Result<&1,&1,U32 & String,Unit>

# Values per File effect.
def chunk() -> U32:
  16384

# An FP32 value and its 32 bits are the same word.

def from_bits(word: U32) -> F32:
  U32{bits} = word
  F32{bits}

def to_bits(value: F32) -> U32:
  F32{bits} = value
  U32{bits}

# Reading: bytes gather into words, least significant first. The values of one
# read are written into the buffer before the next read, so no list outgrows a read.

type Words is Data:
  Words{values: List<&2,F32>,count: U32,word: U32,weight: U32}

# The buffer of the tensor, the values found so far, and a word in progress.
type Target is Type:
  Target{buffer: Buffers.Buffer<F32>,found: U32,word: U32,weight: U32}

type Reading is Type:
  More{file: File,target: Target}
  End{file: File,target: Target,complete: Bool}

type Outcome is Type:
  Outcome{target: Target,complete: Bool}

def gather(full: Bool,values: List<&2,F32>,+count: U32,+word: U32,+weight: U32) -> Words:
  match full:
    case False{}: Words{values,count,word,(weight * 256 : U32)}
    case True{}: Words{from_bits(word) <> values,(count + 1 : U32),0,1}

def push(+byte: U32,words: Words) -> Words:
  Words{values,+count,+word,+weight} = words
  gather(U32.is_eq(weight,16777216),values,count,(word + byte * weight : U32),weight)

def trickle(bytes: List<&2,U32>,words: Words) -> Words:
  match bytes:
    case []: words
    case byte <> rest: trickle(rest,push(byte,words))

# Four bytes at a time while the bytes are whole words; the values are newest first.
def quads(bytes: List<&2,U32>,values: List<&2,F32>,+count: U32) -> Words:
  match bytes:
    case Con{first,Con{second,Con{third,Con{fourth,rest}}}}:
      quads(rest,from_bits((first + second * 256 + third * 65536 + fourth * 16777216 : U32)) <> values,(count + 1 : U32))
    case rest: trickle(rest,Words{values,count,0,1})

def words(whole: Bool,bytes: List<&2,U32>,+word: U32,+weight: U32) -> Words:
  match whole:
    case True{}: quads(bytes,[],0)
    case False{}: trickle(bytes,Words{[],0,word,weight})

# Values beyond the count of the shape are counted and not written.
def placed(read: Words,buffer: Buffers.Buffer<F32>,+found: U32,+count: U32) -> Target:
  Words{values,+added,+word,+weight} = read
  Target{Buffers.write_at(~F32,List.take(&2,F32,List.reverse(&2,F32,values),U32.to_nat((count - U32.min(found,count) : U32))),U32.min(found,count),buffer),
    (found + added : U32),word,weight}

def fill(bytes: List<&2,U32>,target: Target,+count: U32) -> Target:
  Target{buffer,+found,+word,+weight} = target
  placed(words(U32.is_eq(weight,1),bytes,word,weight),buffer,found,count)

def absorbed(bytes: List<&2,U32>,file: File,target: Target,+count: U32) -> Reading:
  match bytes:
    case []: End{file,target,True{}}
    case byte <> rest: More{file,fill(byte <> rest,target,count)}

def received(result: Result<&1,&1,U32 & String,List<&2,U32>>,file: File,target: Target,+count: U32) -> Reading:
  match result:
    case Fail{error}: End{file,target,False{}}
    case Done{bytes}: absorbed(bytes,file,target,count)

def taken(read: Read(),target: Target,+count: U32) -> Reading:
  (file,result) = read
  received(result,file,target,count)

def closed(file: File,target: Target,complete: Bool) -> IO(Outcome):
  do IO<Outcome>:
    File.close(file)
    IO.pure(Outcome,Outcome{target,complete})

# A file longer than its shape stops at the fuel and fails the count.
def read_all(fuel: Nat,reading: Reading,+count: U32) -> IO(Outcome):
  match fuel reading:
    case 0n More{file,target}: closed(file,target,True{})
    case 0n End{file,target,complete}: closed(file,target,complete)
    case 1n+rest End{file,target,complete}: closed(file,target,complete)
    case 1n+rest More{file,target}:
      do IO<Outcome>:
        read : Read() <- File.read_bytes(file,(chunk() * 4 : U32))
        read_all(rest,taken(read,target,count),count)

def fresh(+count: U32) -> Target:
  Target{Tensors.buffer(~F32,0.0,count),0,0,1}

def fuel(+count: U32) -> Nat:
  U32.to_nat((count / chunk() + 2 : U32))

def fault(operation: String,path: String,detail: String,shape: List<&2,U32>) -> Tensor():
  Tensors.failed(~F32,operation,Tensors.FileFault{path,detail},shape,0.0,1)

def sized(matches: Bool,operation: String,path: String,+shape: List<&2,U32>,buffer: Buffers.Buffer<F32>) -> Tensor():
  match matches:
    case True{}: Tensors.from_storage(~F32,shape,buffer,0.0)
    case False{}: fault(operation,path,"does not hold the values of the shape",shape)

def whole(aligned: Bool,operation: String,path: String,+shape: List<&2,U32>,buffer: Buffers.Buffer<F32>,+found: U32,+count: U32) -> Tensor():
  match aligned:
    case False{}: fault(operation,path,"is not a whole number of FP32 values",shape)
    case True{}: sized(U32.is_eq(found,count),operation,path,shape,buffer)

def aligned(operation: String,path: String,+shape: List<&2,U32>,target: Target,+count: U32) -> Tensor():
  Target{buffer,+found,word,+weight} = target
  whole(U32.is_eq(weight,1),operation,path,shape,buffer,found,count)

def checked(complete: Bool,operation: String,path: String,+shape: List<&2,U32>,target: Target,+count: U32) -> Tensor():
  match complete:
    case False{}: fault(operation,path,"cannot be read",shape)
    case True{}: aligned(operation,path,shape,target,count)

def loaded(operation: String,path: String,+shape: List<&2,U32>,outcome: Outcome,+count: U32) -> Tensor():
  Outcome{target,complete} = outcome
  checked(complete,operation,path,shape,target,count)

def opened(result: Result<&1,&1,U32 & String,File>,path: String,+shape: List<&2,U32>,+count: U32) -> IO(Tensor()):
  match result:
    case Fail{error}: IO.pure(Tensor(),fault("load",path,"cannot be opened",shape))
    case Done{file}:
      do IO<Tensor()>:
        outcome : Outcome <- read_all(fuel(count),More{file,fresh(count)},count)
        IO.pure(Tensor(),loaded("load",path,shape,outcome,count))

def counted(count: Maybe<&2,U32>,+path: String,+shape: List<&2,U32>) -> IO(Tensor()):
  match count:
    case None{}: IO.pure(Tensor(),Tensors.failed(~F32,"load",Tensors.InvalidShape{shape},shape,0.0,1))
    case Some{+count}:
      do IO<Tensor()>:
        result : Result<&1,&1,U32 & String,File> <- File.open(path,"r")
        opened(result,path,shape,count)

# The file holds the values of the shape in row-major order and nothing else.
def load(+path: String,+shape: List<&2,U32>) -> IO(Tensor()):
  counted(Tensors.count_of(shape),path,shape)

# The NumPy .npy header: magic, version, header length, then a text such as
# {'descr': '<f4', 'fortran_order': False, 'shape': (16, 3, 6, 6), }

type Header is Data:
  Header{shape: List<&2,U32>,data: List<&2,U32>}

def codes(text: String) -> List<&2,U32>:
  match text:
    case SNil{}: []
    case SCon{head,tail}: Char.to_u32(head) <> codes(tail)

def magic() -> List<&2,U32>:
  [147,78,85,77,80,89]

def starts(pattern: List<&2,U32>,bytes: List<&2,U32>) -> Bool:
  match pattern bytes:
    case Nil{} bytes: True{}
    case Con{first,rest} Nil{}: False{}
    case Con{first,rest} Con{byte,more}: U32.is_eq(first,byte) && starts(rest,more)

def at(bytes: List<&2,U32>,index: Nat) -> U32:
  match bytes index:
    case Nil{} index: 0
    case Con{byte,rest} 0n: byte
    case Con{byte,rest} 1n+before: at(rest,before)

def contains(text: List<&2,U32>,+pattern: List<&2,U32>) -> Bool:
  match text:
    case Nil{}: False{}
    case Con{+byte,+rest}: starts(pattern,byte <> rest) || contains(rest,pattern)

# The text after the first occurrence of the pattern.
def after(text: List<&2,U32>,+pattern: List<&2,U32>) -> Maybe<&2,List<&2,U32>>:
  match text:
    case Nil{}: None{}
    case Con{+byte,+rest}:
      Bool.pick(Maybe<&2,List<&2,U32>>,starts(pattern,byte <> rest),Some{List.drop(&2,U32,byte <> rest,List.length(&2,U32,pattern))},after(rest,pattern))

def pushed(digits: Bool,current: U32,shape: List<&2,U32>) -> List<&2,U32>:
  match digits:
    case True{}: current <> shape
    case False{}: shape

def closed_shape(valid: Bool,shape: List<&2,U32>) -> Maybe<&2,List<&2,U32>>:
  match valid:
    case True{}: Some{List.reverse(&2,U32,shape)}
    case False{}: None{}

# Extents up to the closing parenthesis: digits, commas and spaces.
def extents(text: List<&2,U32>,+current: U32,digits: Bool,valid: Bool,shape: List<&2,U32>) -> Maybe<&2,List<&2,U32>>:
  match text:
    case Nil{}: None{}
    case Con{+byte,rest}:
      match byte:
        case 41: closed_shape(valid,pushed(digits,current,shape))
        case 44: extents(rest,0,False{},valid,pushed(digits,current,shape))
        case 32: extents(rest,current,digits,valid,shape)
        case _: extents(rest,(current * 10 + byte - 48 : U32),True{},valid && U32.is_le(48,byte) && U32.is_le(byte,57),shape)

def shaped(shape: Maybe<&2,List<&2,U32>>,data: List<&2,U32>) -> Maybe<&2,Header>:
  match shape:
    case None{}: None{}
    case Some{shape}: Some{Header{shape,data}}

def typed(supported: Bool,text: Maybe<&2,List<&2,U32>>,data: List<&2,U32>) -> Maybe<&2,Header>:
  match supported text:
    case False{} text: None{}
    case True{} None{}: None{}
    case True{} Some{text}: shaped(extents(text,0,False{},True{},[]),data)

def described(+text: List<&2,U32>,+length: Nat,data: List<&2,U32>) -> Maybe<&2,Header>:
  typed(Nat.is_eq(List.length(&2,U32,text),length) && contains(text,codes("'descr': '<f4'")) && contains(text,codes("'fortran_order': False")),
    after(text,codes("'shape': (")),data)

def region(+bytes: List<&2,U32>,+length: Nat) -> Maybe<&2,Header>:
  described(List.take(&2,U32,bytes,length),length,List.drop(&2,U32,bytes,length))

# Version 1 has a two-byte header length; later versions have four bytes.
def versioned(major: U32,+bytes: List<&2,U32>) -> Maybe<&2,Header>:
  match major:
    case 1: region(List.drop(&2,U32,bytes,10n),U32.to_nat((at(bytes,8n) + at(bytes,9n) * 256 : U32)))
    case _: region(List.drop(&2,U32,bytes,12n),U32.to_nat((at(bytes,8n) + at(bytes,9n) * 256 + at(bytes,10n) * 65536 + at(bytes,11n) * 16777216 : U32)))

def marked(valid: Bool,+bytes: List<&2,U32>) -> Maybe<&2,Header>:
  match valid:
    case False{}: None{}
    case True{}: versioned(at(bytes,6n),bytes)

def header(+bytes: List<&2,U32>) -> Maybe<&2,Header>:
  marked(starts(magic(),bytes),bytes)

def refused(file: File,tensor: Tensor()) -> IO(Tensor()):
  do IO<Tensor()>:
    File.close(file)
    IO.pure(Tensor(),tensor)

def npy_counted(count: Maybe<&2,U32>,+shape: List<&2,U32>,data: List<&2,U32>,file: File,+path: String) -> IO(Tensor()):
  match count:
    case None{}: refused(file,Tensors.failed(~F32,"load_npy",Tensors.InvalidShape{shape},shape,0.0,1))
    case Some{+count}:
      do IO<Tensor()>:
        outcome : Outcome <- read_all(fuel(count),More{file,fill(data,fresh(count),count)},count)
        IO.pure(Tensor(),loaded("load_npy",path,shape,outcome,count))

def npy_parsed(parsed: Maybe<&2,Header>,file: File,+path: String) -> IO(Tensor()):
  match parsed:
    case None{}: refused(file,fault("load_npy",path,"is not a little-endian FP32 .npy array in C order",[]))
    case Some{Header{+shape,data}}: npy_counted(Tensors.count_of(shape),shape,data,file,path)

def npy_received(result: Result<&1,&1,U32 & String,List<&2,U32>>,file: File,+path: String) -> IO(Tensor()):
  match result:
    case Fail{error}: refused(file,fault("load_npy",path,"cannot be read",[]))
    case Done{bytes}: npy_parsed(header(bytes),file,path)

def npy_headed(read: Read(),+path: String) -> IO(Tensor()):
  (file,result) = read
  npy_received(result,file,path)

def npy_opened(result: Result<&1,&1,U32 & String,File>,+path: String) -> IO(Tensor()):
  match result:
    case Fail{error}: IO.pure(Tensor(),fault("load_npy",path,"cannot be opened",[]))
    case Done{file}:
      do IO<Tensor()>:
        read : Read() <- File.read_bytes(file,(chunk() * 4 : U32))
        npy_headed(read,path)

# An FP32 array saved by NumPy, with its shape.
def load_npy(+path: String) -> IO(Tensor()):
  do IO<Tensor()>:
    result : Result<&1,&1,U32 & String,File> <- File.open(path,"r")
    npy_opened(result,path)

# Writing

type Writing is Type:
  Pending{file: File,storage: Array<F32>,offset: U32,remaining: Nat}
  Finished{file: File,complete: Bool}

def bytes(+word: U32,rest: List<&2,U32>) -> List<&2,U32>:
  (word % 256 : U32) <> (word / 256 % 256 : U32) <> (word / 65536 % 256 : U32) <> (word / 16777216 : U32) <> rest

def encode(count: Nat,values: List<&2,F32>) -> List<&2,U32>:
  match count values:
    case 0n values: []
    case 1n+rest Nil{}: []
    case 1n+rest Con{value,more}: bytes(to_bits(value),encode(rest,more))

# The cursor retains the array; only the current chunk becomes a list.
def read_chunk(count: Nat,+offset: U32,storage: Array<F32>,values: List<&2,F32>) -> Array<F32> & List<&2,F32>:
  match count:
    case 0n: (storage,List.reverse(&2,F32,values))
    case 1n+rest:
      Buffers.bind_pair(Array<F32>,F32,Array<F32> & List<&2,F32>,Array.get(F32,storage,offset),storage => value =>
        read_chunk(rest,(offset + 1 : U32),storage,value <> values))

def pending(remaining: Nat,file: File,storage: Array<F32>,offset: U32) -> Writing:
  match remaining:
    case 0n: Finished{file,True{}}
    case 1n+rest: Pending{file,storage,offset,1n+rest}

def continued(result: Result<&1,&1,U32 & String,Unit>,file: File,storage: Array<F32>,remaining: Nat,offset: U32) -> Writing:
  match result:
    case Fail{error}: Finished{file,False{}}
    case Done{unit}: pending(remaining,file,storage,offset)

def after_write(written: Written(),storage: Array<F32>,remaining: Nat,offset: U32) -> Writing:
  (file,result) = written
  continued(result,file,storage,remaining,offset)

def write_chunk(file: File,read: Array<F32> & List<&2,F32>,+count: Nat,remaining: Nat,offset: U32) -> IO(Writing):
  (storage,values) = read
  do IO<Writing>:
    written : Written() <- File.write_bytes(file,encode(count,values))
    IO.pure(Writing,after_write(written,storage,Nat.sub(remaining,count),(offset + U32.from_nat(count) : U32)))

def finished(file: File,complete: Bool) -> IO(Bool):
  do IO<Bool>:
    File.close(file)
    IO.pure(Bool,complete)

def write_all(fuel: Nat,writing: Writing) -> IO(Bool):
  match fuel writing:
    case 0n Pending{file,storage,offset,remaining}: finished(file,False{})
    case 0n Finished{file,complete}: finished(file,complete)
    case 1n+rest Finished{file,complete}: finished(file,complete)
    case 1n+rest Pending{file,storage,+offset,+remaining}:
      +count = Nat.min(U32.to_nat(chunk()),remaining)
      do IO<Bool>:
        next : Writing <- write_chunk(file,read_chunk(count,offset,storage,[]),count,remaining,offset)
        write_all(rest,next)

def unsaved(pair: Tensor() & List<&2,U32>,operation: String,path: String,detail: String) -> Tensor():
  (tensor,shape) = pair
  fault(operation,path,detail,shape)

def written(complete: Bool,operation: String,path: String,tensor: Tensor()) -> Tensor():
  match complete:
    case True{}: tensor
    case False{}: unsaved(Tensors.shape(~F32,tensor),operation,path,"cannot be written")

# The prefix is written first; a raw file has none.
def created(result: Result<&1,&1,U32 & String,File>,operation: String,+path: String,tensor: Tensor(),prefix: List<&2,U32>,storage: Array<F32>,+length: Nat) -> IO(Tensor()):
  match result:
    case Fail{error}: IO.pure(Tensor(),unsaved(Tensors.shape(~F32,tensor),operation,path,"cannot be created"))
    case Done{file}:
      do IO<Tensor()>:
        head : Written() <- File.write_bytes(file,prefix)
        complete : Bool <- write_all(Nat.add(length,2n),after_write(head,storage,length,0))
        IO.pure(Tensor(),written(complete,operation,path,tensor))

def saving(operation: String,+path: String,tensor: Tensor(),prefix: List<&2,U32>,buffer: Result<&2,&1,Tensors.Problem,Buffers.Buffer<F32>>) -> IO(Tensor()):
  match buffer:
    case Fail{problem}: IO.pure(Tensor(),tensor)
    case Done{Buffers.Buffer{storage,length,certificate}}:
      do IO<Tensor()>:
        result : Result<&1,&1,U32 & String,File> <- File.open(path,"w")
        created(result,operation,path,tensor,prefix,storage,length)

# Writes the values in row-major order and returns the tensor. A failed tensor
# writes nothing and keeps its problem.
def save(+path: String,tensor: Tensor()) -> IO(Tensor()):
  Tensors.then(~Tensor(),~Tensor(),~IO(Tensor()),Layout.copy(~F32,tensor),kept => copy => saving("save",path,kept,[],Tensors.into_storage(~F32,copy)))

def joined(shape: List<&2,U32>) -> String:
  match shape:
    case []: ""
    case [extent]: U32.show(extent)
    case extent <> rest: U32.show(extent) ++ ", " ++ joined(rest)

# NumPy writes (), (3,) and (2, 3).
def tuple(shape: List<&2,U32>) -> String:
  match shape:
    case []: "()"
    case [extent]: "(" ++ U32.show(extent) ++ ",)"
    case extent <> rest: "(" ++ joined(extent <> rest) ++ ")"

def spaces(count: Nat) -> List<&2,U32>:
  match count:
    case 0n: [10]
    case 1n+rest: 32 <> spaces(rest)

# Version 1: the header ends in a newline and the data starts at a multiple of 64 bytes.
def framed(+text: List<&2,U32>,+length: U32) -> List<&2,U32>:
  +padding = ((64 - (11 + length) % 64) % 64 : U32)
  +total = (length + padding + 1 : U32)
  List.append(&2,U32,magic(),1 <> 0 <> (total % 256 : U32) <> (total / 256 : U32) <> List.append(&2,U32,text,spaces(U32.to_nat(padding))))

def npy_header(shape: List<&2,U32>) -> List<&2,U32>:
  +text = codes("{'descr': '<f4', 'fortran_order': False, 'shape': " ++ tuple(shape) ++ ", }")
  framed(text,U32.from_nat(List.length(&2,U32,text)))

def npy_saving(+path: String,pair: Tensor() & Tensor(),+shape: List<&2,U32>) -> IO(Tensor()):
  (kept,copy) = pair
  saving("save_npy",path,kept,npy_header(shape),Tensors.into_storage(~F32,copy))

# Writes a .npy file that NumPy loads with the tensor's shape.
def save_npy(+path: String,tensor: Tensor()) -> IO(Tensor()):
  Tensors.then(~Tensor(),~List<&2,U32>,~IO(Tensor()),Tensors.shape(~F32,tensor),tensor => shape => npy_saving(path,Layout.copy(~F32,tensor),shape))
