# Models: a step computes with tensors it takes by name from a store, so a
# network reads as one step after another and its weights are loaded once.
import Base
import ./tensor.bend as Tensors
import ./tensor_layout.bend as Layout
import ./tensor_file.bend as Files

def Tensor() -> Type:
  Tensors.Tensor<F32>

type Store is Type:
  Empty{}
  Entry{name: String,tensor: Tensor(),rest: Store}

# A step: given the store, a value and the store that remains.
def Model(A: Type) -> Type:
  Store -> A & Store

def Model.pure(-A: Type,value: A) -> Model(A):
  store => (value,store)

def Model.next(-A: Type,-B: Type,pair: A & Store,next: A -> Model(B)) -> B & Store:
  (value,store) = pair
  next(value)(store)

def Model.bind(-A: Type,-B: Type,step: Model(A),next: A -> Model(B)) -> Model(B):
  store => Model.next(A,B,step(store),next)

def missing(+name: String) -> Tensor():
  Tensors.failed(~F32,"take",Tensors.MissingName{name},[],0.0,1)

def passed(name: String,tensor: Tensor(),pair: Tensor() & Store) -> Tensor() & Store:
  (result,store) = pair
  (result,Entry{name,tensor,store})

def scan(rest: Store,same: Bool,+name: String,tensor: Tensor(),+wanted: String) -> Tensor() & Store:
  match rest same:
    case Empty{} True{}: (tensor,Empty{})
    case Entry{next,other,more} True{}: (tensor,Entry{next,other,more})
    case Empty{} False{}: (missing(wanted),Entry{name,tensor,Empty{}})
    case Entry{+next,other,more} False{}: passed(name,tensor,scan(more,String.eq(next,wanted),next,other,wanted))

def taken(store: Store,+wanted: String) -> Tensor() & Store:
  match store:
    case Empty{}: (missing(wanted),Empty{})
    case Entry{+name,tensor,rest}: scan(rest,String.eq(name,wanted),name,tensor,wanted)

# The tensor of that name leaves the store. A name that is not there gives a
# failed tensor.
def take(+name: String) -> Model(Tensor()):
  store => taken(store,name)

def kept(+name: String,pair: Tensor() & Tensor(),store: Store) -> Tensor() & Store:
  (tensor,copy) = pair
  (tensor,Entry{name,copy,store})

# Puts a copy under the name, for a later take, and gives the tensor back.
def keep(+name: String,tensor: Tensor()) -> Model(Tensor()):
  store => kept(name,Layout.copy(~F32,tensor),store)

def run(-A: Type,step: Model(A),store: Store) -> A & Store:
  step(store)

def value(-A: Type,pair: A & Store) -> A:
  (result,store) = pair
  result

# Runs the steps on the store and gives the result; what remains in the store is dropped.
def result(-A: Type,step: Model(A),store: Store) -> A:
  value(A,step(store))

def joined(+name: String,first: Tensor(),second: Tensor(),rest: Store & Store) -> Store & Store:
  (left,right) = rest
  (Entry{name,first,left},Entry{name,second,right})

def copied_entry(+name: String,pair: Tensor() & Tensor(),rest: Store & Store) -> Store & Store:
  (first,second) = pair
  joined(name,first,second,rest)

# Two stores with the same tensors, to run a model more than once.
def copy(store: Store) -> Store & Store:
  match store:
    case Empty{}: (Empty{},Empty{})
    case Entry{+name,tensor,rest}: copied_entry(name,Layout.copy(~F32,tensor),copy(rest))

# Loading: folder/names.txt lists one name per line; folder/<name>.npy holds its tensor.

def entry(+name: String,tensor: Tensor(),store: Store) -> Store:
  Entry{name,tensor,store}

def load_each(names: List<&2,String>,+folder: String) -> IO(Store):
  match names:
    case Nil{}: IO.pure(Store,Empty{})
    case Con{+name,rest}:
      match name:
        case SNil{}: load_each(rest,folder)
        case SCon{head,tail}:
          do IO<Store>:
            tensor : Tensor() <- Files.load_npy(folder ++ "/" ++ name ++ ".npy")
            store : Store <- load_each(rest,folder)
            IO.pure(Store,entry(name,tensor,store))

def named(text: Result<&1,&1,U32 & String,String>,+folder: String) -> IO(Store):
  match text:
    case Fail{error}: IO.pure(Store,Empty{})
    case Done{text}: load_each(String.lines(text),folder)

def listed(read: File & Result<&1,&1,U32 & String,String>,+folder: String) -> IO(Store):
  (file,text) = read
  do IO<Store>:
    File.close(file)
    named(text,folder)

def opened(result: Result<&1,&1,U32 & String,File>,+folder: String) -> IO(Store):
  match result:
    case Fail{error}: IO.pure(Store,Empty{})
    case Done{file}:
      do IO<Store>:
        read : File & Result<&1,&1,U32 & String,String> <- File.read(file,1048576)
        listed(read,folder)

# A folder without names.txt gives an empty store; every take then fails by name.
def load(+folder: String) -> IO(Store):
  do IO<Store>:
    result : Result<&1,&1,U32 & String,File> <- File.open(folder ++ "/names.txt","r")
    opened(result,folder)

# Saving: the tensors of a store as folder/<name>.npy, and their names in folder/names.txt.

def named_rest(+name: String,tensor: Tensor(),pair: Store & String) -> Store & String:
  (rest,text) = pair
  (Entry{name,tensor,rest},name ++ "
" ++ text)

def first(written: File & Result<&1,&1,U32 & String,Unit>) -> File:
  (file,result) = written
  file

def names(store: Store) -> Store & String:
  match store:
    case Empty{}: (Empty{},"")
    case Entry{+name,tensor,rest}: named_rest(name,tensor,names(rest))

def save_each(store: Store,+folder: String) -> IO(Store):
  match store:
    case Empty{}: IO.pure(Store,Empty{})
    case Entry{+name,tensor,rest}:
      do IO<Store>:
        saved : Tensor() <- Files.save_npy(folder ++ "/" ++ name ++ ".npy",tensor)
        others : Store <- save_each(rest,folder)
        IO.pure(Store,entry(name,saved,others))

def listing(result: Result<&1,&1,U32 & String,File>,text: String) -> IO(Unit):
  match result:
    case Fail{error}: IO.pure(Unit,Unit{})
    case Done{file}:
      do IO<Unit>:
        written : File & Result<&1,&1,U32 & String,Unit> <- File.write(file,text)
        File.close(first(written))

def save_listed(pair: Store & String,+folder: String) -> IO(Store):
  (store,text) = pair
  do IO<Store>:
    result : Result<&1,&1,U32 & String,File> <- File.open(folder ++ "/names.txt","w")
    listing(result,text)
    save_each(store,folder)

# Gives the store back; a tensor that could not be written comes back failed.
def save(+folder: String,store: Store) -> IO(Store):
  save_listed(names(store),folder)
