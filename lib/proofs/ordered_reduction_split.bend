import Base
import ../ordered_matrix.bend as OrderedMatrix
import ./sequence_membership.bend as Members
import ./storage_shape.bend as Shape

# Reindexing changes keys, never the ordered sequence of FP32 operations.
def map_keys(~Key: Data,~Mapped: Data,~Context: Data,~map: Context -> Key -> Mapped,
  keys: List<&2,Key>,+context: Context) -> List<&2,Mapped>:
  match keys:
    case Nil{}: []
    case key <> rest: map(context,key) <> map_keys(~Key,~Mapped,~Context,~map,rest,context)

def mapped_sample(~Key: Data,~Mapped: Data,~Column: Data,~Source: Data,~Context: Data,
  ~map: Context -> Key -> Mapped,~input: Source -> Column -> Mapped -> F32,
  combined: Shape.Both<Context,Source>,column: Column,key: Key) -> F32:
  Shape.Both{context,source} = combined
  input(source,column,map(context,key))

def dot_reindexed(~Key: Data,~Mapped: Data,~Row: Data,~Column: Data,~Source: Data,~Context: Data,
  ~map: Context -> Key -> Mapped,~input: Source -> Column -> Mapped -> F32,~weight: Source -> Row -> Mapped -> F32,
  +keys: List<&2,Key>,+context: Context,+row: Row,+column: Column,+source: Source,+initial: F32) ->
  {OrderedMatrix.dot(~Row,~Mapped,~Column,~Source,~input,~weight,map_keys(~Key,~Mapped,~Context,~map,keys,context),row,column,source,initial) ==
    OrderedMatrix.dot(~Row,~Key,~Column,~Shape.Both<Context,Source>,
      ~mapped_sample(~Key,~Mapped,~Column,~Source,~Context,~map,~input),
      ~mapped_sample(~Key,~Mapped,~Row,~Source,~Context,~map,~weight),keys,row,column,Shape.Both{context,source},initial) : F32}:
  match keys:
    case Nil{}: {==}
    case +key <> +rest:
      dot_reindexed(~Key,~Mapped,~Row,~Column,~Source,~Context,~map,~input,~weight,
        rest,context,row,column,source,(initial + input(source,column,map(context,key)) * weight(source,row,map(context,key)) : F32))

# Identical ordered samples imply identical FP32 evaluation. The two sides may
# have different row/column keys and storage representations. No reassociation
# or real-number algebra is used, and obligations apply only to visited keys.
def MatchingSamples(~Key: Data,~LeftRow: Data,~LeftColumn: Data,~LeftSource: Data,~RightRow: Data,~RightColumn: Data,~RightSource: Data,
  ~left_input: LeftSource -> LeftColumn -> Key -> F32,~left_weight: LeftSource -> LeftRow -> Key -> F32,
  ~right_input: RightSource -> RightColumn -> Key -> F32,~right_weight: RightSource -> RightRow -> Key -> F32,
  keys: List<&2,Key>,+left_row: LeftRow,+left_column: LeftColumn,+left_source: LeftSource,
  +right_row: RightRow,+right_column: RightColumn,+right_source: RightSource) -> Data:
  Members.All(Key,{key => Shape.Both<
    {left_input(left_source,left_column,key) == right_input(right_source,right_column,key) : F32},
    {left_weight(left_source,left_row,key) == right_weight(right_source,right_row,key) : F32}> : Key -> Data},keys)

def dot_congruence(~Key: Data,~LeftRow: Data,~LeftColumn: Data,~LeftSource: Data,~RightRow: Data,~RightColumn: Data,~RightSource: Data,
  ~left_input: LeftSource -> LeftColumn -> Key -> F32,~left_weight: LeftSource -> LeftRow -> Key -> F32,
  ~right_input: RightSource -> RightColumn -> Key -> F32,~right_weight: RightSource -> RightRow -> Key -> F32,
  +keys: List<&2,Key>,+left_row: LeftRow,+left_column: LeftColumn,+left_source: LeftSource,
  +right_row: RightRow,+right_column: RightColumn,+right_source: RightSource,+initial: F32,
  agreements: MatchingSamples(~Key,~LeftRow,~LeftColumn,~LeftSource,~RightRow,~RightColumn,~RightSource,~left_input,~left_weight,~right_input,~right_weight,
    keys,left_row,left_column,left_source,right_row,right_column,right_source)) ->
  {OrderedMatrix.dot(~LeftRow,~Key,~LeftColumn,~LeftSource,~left_input,~left_weight,keys,left_row,left_column,left_source,initial) ==
    OrderedMatrix.dot(~RightRow,~Key,~RightColumn,~RightSource,~right_input,~right_weight,keys,right_row,right_column,right_source,initial) : F32}:
  match keys:
    case Nil{}: {==}
    case +key <> +rest:
      Shape.Both{head,tail} = agreements
      Shape.Both{same_input,same_weight} = head
      %Equal.sym(F32,left_input(left_source,left_column,key),right_input(right_source,right_column,key),same_input) :
        {OrderedMatrix.dot(~LeftRow,~Key,~LeftColumn,~LeftSource,~left_input,~left_weight,rest,left_row,left_column,left_source,
          (initial + _ * left_weight(left_source,left_row,key) : F32)) ==
          OrderedMatrix.dot(~RightRow,~Key,~RightColumn,~RightSource,~right_input,~right_weight,rest,right_row,right_column,right_source,
            (initial + right_input(right_source,right_column,key) * right_weight(right_source,right_row,key) : F32)) : F32}
      %Equal.sym(F32,left_weight(left_source,left_row,key),right_weight(right_source,right_row,key),same_weight) :
        {OrderedMatrix.dot(~LeftRow,~Key,~LeftColumn,~LeftSource,~left_input,~left_weight,rest,left_row,left_column,left_source,
          (initial + right_input(right_source,right_column,key) * _ : F32)) ==
          OrderedMatrix.dot(~RightRow,~Key,~RightColumn,~RightSource,~right_input,~right_weight,rest,right_row,right_column,right_source,
            (initial + right_input(right_source,right_column,key) * right_weight(right_source,right_row,key) : F32)) : F32}
      dot_congruence(~Key,~LeftRow,~LeftColumn,~LeftSource,~RightRow,~RightColumn,~RightSource,~left_input,~left_weight,~right_input,~right_weight,
        rest,left_row,left_column,left_source,right_row,right_column,right_source,
        (initial + right_input(right_source,right_column,key) * right_weight(right_source,right_row,key) : F32),tail)

law cat_assoc:
  for -A: Data
  for +a: List<&2,A>
  for +b: List<&2,A>
  for +c: List<&2,A>
  {OrderedMatrix.cat(A,OrderedMatrix.cat(A,a,b),c) == OrderedMatrix.cat(A,a,OrderedMatrix.cat(A,b,c)) : List<&2,A>}
def cat_assoc(A,a,b,c):
  match a:
    case Nil{}:
      {==}
    case +h <> +tail:
      %cat_assoc(A,tail,b,c) : {h <> OrderedMatrix.cat(A,OrderedMatrix.cat(A,tail,b),c) == h <> _ : List<&2,A>}
      {==}

def reduction_split(~R: Data,~K: Data,~C: Data,~P: Data,~x: P -> C -> K -> F32,~w: P -> R -> K -> F32,+a: List<&2,K>,+b: List<&2,K>,+r: R,+c: C,
  +p: P,+acc: F32) -> {OrderedMatrix.dot(~R,~K,~C,~P,~x,~w,OrderedMatrix.cat(K,a,b),r,c,p,acc) == OrderedMatrix.dot(~R,~K,~C,~P,~x,~w,b,r,c,p,
    OrderedMatrix.dot(~R,~K,~C,~P,~x,~w,a,r,c,p,acc)) : F32}:
  match a:
    case Nil{}:
      {==}
    case +k <> +tail:
      reduction_split(~R,~K,~C,~P,~x,~w,tail,b,r,c,p,(acc + x(p,c,k) * w(p,r,k) : F32))

def column_split(~R: Data,~K: Data,~C: Data,~P: Data,~x: P -> C -> K -> F32,~w: P -> R -> K -> F32,~bias: P -> R -> F32,+a: List<&2,C>,
  +b: List<&2,C>,+ks: List<&2,K>,+r: R,+p: P) -> {OrderedMatrix.row(~R,~K,~C,~P,~x,~w,~bias,OrderedMatrix.cat(C,a,b),ks,r,
    p) == OrderedMatrix.cat(F32,OrderedMatrix.row(~R,~K,~C,~P,~x,~w,~bias,a,ks,r,p),OrderedMatrix.row(~R,~K,~C,~P,~x,~w,~bias,b,ks,r,
      p)) : List<&2,F32>}:
  match a:
    case Nil{}:
      {==}
    case +c <> +tail:
      %column_split(~R,~K,~C,~P,~x,~w,~bias,tail,b,ks,r,p) : {(OrderedMatrix.dot(~R,~K,~C,~P,~x,~w,ks,r,c,p,0.0) + bias(p,
            r) : F32) <> OrderedMatrix.row(~R,~K,~C,~P,~x,~w,~bias,OrderedMatrix.cat(C,tail,b),ks,r,p) == (OrderedMatrix.dot(~R,~K,~C,~P,~x,~w,
            ks,r,c,p,0.0) + bias(p,r) : F32) <> _ : List<&2,F32>}
      {==}

def row_split(~R: Data,~K: Data,~C: Data,~P: Data,~x: P -> C -> K -> F32,~w: P -> R -> K -> F32,~bias: P -> R -> F32,+a: List<&2,R>,+b: List<&2,
  R>,+ks: List<&2,K>,+cs: List<&2,C>,+p: P) -> {OrderedMatrix.matmul(~R,~K,~C,~P,~x,~w,~bias,OrderedMatrix.cat(R,a,b),ks,cs,
    p) == OrderedMatrix.cat(F32,OrderedMatrix.matmul(~R,~K,~C,~P,~x,~w,~bias,a,ks,cs,p),OrderedMatrix.matmul(~R,~K,~C,~P,~x,~w,~bias,b,ks,cs,
      p)) : List<&2,F32>}:
  match a:
    case Nil{}:
      {==}
    case +r <> +tail:
      %Equal.sym(List<&2,F32>,OrderedMatrix.cat(F32,OrderedMatrix.cat(F32,OrderedMatrix.row(~R,~K,~C,~P,~x,~w,~bias,cs,ks,r,p),
            OrderedMatrix.matmul(~R,~K,~C,~P,~x,~w,~bias,tail,ks,cs,p)),OrderedMatrix.matmul(~R,~K,~C,~P,~x,~w,~bias,b,ks,cs,p)),
        OrderedMatrix.cat(F32,OrderedMatrix.row(~R,~K,~C,~P,~x,~w,~bias,cs,ks,r,p),OrderedMatrix.cat(F32,OrderedMatrix.matmul(~R,~K,~C,~P,~x,~w,
              ~bias,tail,ks,cs,p),OrderedMatrix.matmul(~R,~K,~C,~P,~x,~w,~bias,b,ks,cs,p))),cat_assoc(F32,OrderedMatrix.row(~R,~K,~C,~P,~x,~w,
            ~bias,cs,ks,r,p),OrderedMatrix.matmul(~R,~K,~C,~P,~x,~w,~bias,tail,ks,cs,p),OrderedMatrix.matmul(~R,~K,~C,~P,~x,~w,~bias,b,ks,cs,
            p))) : {OrderedMatrix.cat(F32,OrderedMatrix.row(~R,~K,~C,~P,~x,~w,~bias,cs,ks,r,p),OrderedMatrix.matmul(~R,~K,~C,~P,~x,~w,~bias,
            OrderedMatrix.cat(R,tail,b),ks,cs,p)) == _ : List<&2,F32>}
      %row_split(~R,~K,~C,~P,~x,~w,~bias,tail,b,ks,cs,p) : {OrderedMatrix.cat(F32,OrderedMatrix.row(~R,~K,~C,~P,~x,~w,~bias,cs,ks,r,p),
          OrderedMatrix.matmul(~R,~K,~C,~P,~x,~w,~bias,OrderedMatrix.cat(R,tail,b),ks,cs,p)) == OrderedMatrix.cat(F32,OrderedMatrix.row(~R,~K,~C,
            ~P,~x,~w,~bias,cs,ks,r,p),_) : List<&2,F32>}
      {==}
