# Structural cost of the 4x8 pipeline. Only guarded dimensions enter here.
# Packing and GEMM run in each leaf; reordering and final gather are serial.
import Base
import ./kernel_cost.bend as Cost
import ./kernel_window.bend as KernelWindow
import ./math_activation.bend as Activations
import ./kernel_pipeline.bend as Pipeline

type Owners is Data:
  Owners{input_capacity: U32,bias_capacity: U32}

# Image and stencil extents that decide which input rows a fork copies.
type Geometry is Data:
  Geometry{channels: U32,height: U32,width: U32,kernel: U32,stride: U32,padding: U32,output_width: U32}

type Work is Data:
  Work{positions: U32,inner: U32,rows: U32,owners: Owners,geometry: Geometry,activation: Activations.Activation}

# The input rows a partition node holds and the capacity of their owner.
type InputRows is Data:
  InputRows{first: U32,rows: U32,capacity: Nat}

# Allocation and moved/initialized payload are distinct for a copied band.
type ForkInput is Data:
  ForkInput{bytes: Nat,transfer: Nat,left: InputRows}

# critical_ns is the span (the costliest root-to-leaf path) and work_ns the sum
# over every leaf and fork; together they bound time on a given worker count.
type Movement is Data:
  Movement{packed: Nat,leaves: Nat,copies: Nat,critical_ns: F32,work_ns: F32,minimum_leaf_ns: F32}

type Plan is Data:
  Plan{packed: Bool,split_depth: U32,workspace_bytes: Nat,estimated_ns: F32}

def capacity_loop(remaining: Nat,+value: U32,+capacity: Nat,zero: Bool) -> Nat:
  match remaining zero:
    case 0n zero: capacity
    case 1n+rest True{}: capacity
    case 1n+rest False{}:
      +next = U32.shr(value)
      capacity_loop(rest,next,Nat.double(capacity),U32.is_zero(next))

def capacity(+elements: U32) -> Nat:
  +value = (U32.max(1,elements) - 1 : U32)
  capacity_loop(32n,value,1n,U32.is_zero(value))

def float_parts(parts: Nat & Nat) -> F32:
  (high,low) = parts
  (F32.from_nat(high) * 4294967296.0 + F32.from_nat(low) : F32)

def payload_float(bytes: Nat) -> F32:
  float_parts(Nat.divmod(bytes,Nat.double(2147483648n)))

def weight_bytes(work: Work) -> Nat:
  Work{positions,inner,rows,owners,geometry,activation} = work
  Nat.mul(32n,capacity(((rows + 7) / 8 * inner : U32)))

# The weight reorder writes complete eight-channel tiles, including zero lanes.
def reorder_ns(+work: Work,config: Cost.Configuration) -> F32:
  Work{positions,+inner,+rows,owners,geometry,activation} = work
  Cost.Configuration{mac,byte_ns,operation_ns,grain,wake} = config
  ((U32.to_f32(((rows + 7) / 8 * 8 : U32)) * U32.to_f32(inner) * 4.0 + payload_float(weight_bytes(work))) * byte_ns : F32)

# Each fork clones the reordered weights and bias, and splits the input rows.
def fork_bytes(+work: Work,input: Nat) -> Nat:
  Work{positions,inner,rows,Owners{input_capacity,bias},geometry,activation} = work
  Nat.add(weight_bytes(work),Nat.add(input,Nat.mul(4n,U32.to_nat(U32.max(2,bias)))))

def clone_bytes(+window: InputRows) -> Nat:
  InputRows{first,rows,+capacity} = window
  Nat.mul(4n,Nat.max(2n,capacity))

def band_rows(+band: KernelWindow.Band,+channels: U32,+width: U32) -> InputRows:
  KernelWindow.Band{+first,+rows} = band
  InputRows{first,rows,capacity((channels * rows * width : U32))}

def band_bytes(+band: KernelWindow.Band,+channels: U32,+width: U32) -> Nat:
  KernelWindow.Band{first,+rows} = band
  Nat.mul(4n,Nat.max(2n,capacity((channels * rows * width : U32))))

# copied initializes the new owner and copies every logical band value.
# Array.clone instead copies the complete allocation, without initialization.
def input_transfer(copy: Bool,band: KernelWindow.Band,window: InputRows,channels: U32,width: U32) -> Nat:
  match copy band:
    case False{} band: clone_bytes(window)
    case True{} KernelWindow.Band{first,rows}:
      +count = (channels * rows * width : U32)
      Nat.mul(4n,Nat.add(U32.to_nat(count),capacity(count)))

# As in the pipelines: a checked band over aligned rows is copied and the
# second half keeps the parent; otherwise the parent owner is cloned.
def input_bytes(copy: Bool,+band: KernelWindow.Band,+window: InputRows,+channels: U32,+width: U32) -> Nat:
  match copy:
    case False{}: clone_bytes(window)
    case True{}: band_bytes(band,channels,width)

def left_rows(copy: Bool,+band: KernelWindow.Band,+window: InputRows,+channels: U32,+width: U32) -> InputRows:
  match copy:
    case False{}: window
    case True{}: band_rows(band,channels,width)

def fork_window(+offset: U32,+half: U32,+window: InputRows,+geometry: Geometry) -> ForkInput:
  InputRows{+first,+count,+capacity} = window
  Geometry{+channels,+height,+width,+kernel,+stride,+padding,+output_width} = geometry
  +band = KernelWindow.band(first,count,height,kernel,stride,padding,output_width,offset,half)
  +copy = KernelWindow.copies_band(first,count,height,kernel,stride,padding,output_width,offset,half,band)
  +parent = {InputRows{first,count,capacity} : InputRows}
  ForkInput{input_bytes(copy,band,parent,channels,width),input_transfer(copy,band,parent,channels,width),left_rows(copy,band,parent,channels,width)}

def fork_input(+offset: U32,+half: U32,+window: InputRows,+work: Work) -> ForkInput:
  Work{positions,inner,rows,owners,+geometry,activation} = work
  fork_window(offset,half,window,geometry)

def forked_bytes(+fork: ForkInput) -> Nat:
  ForkInput{+bytes,transfer,left} = fork
  bytes

def forked_transfer(+fork: ForkInput) -> Nat:
  ForkInput{bytes,+transfer,left} = fork
  transfer

def forked_left(+fork: ForkInput) -> InputRows:
  ForkInput{bytes,transfer,+left} = fork
  left

def root_rows(+work: Work) -> InputRows:
  Work{positions,inner,rows,Owners{+input,bias},Geometry{channels,+height,width,kernel,stride,padding,output_width},activation} = work
  InputRows{0,height,U32.to_nat(input)}

# The activation pass reads and writes each logical output in place; the
# identity skips the pass.
def activation_ns(+activation: Activations.Activation,+config: Cost.Configuration) -> F32:
  Bool.pick(F32,Activations.is_identity(activation),0.0,Cost.element_ns(4.0,Activations.operations(activation),config))

def leaf_cost(span: U32,+count: U32,work: Work,+config: Cost.Configuration) -> Movement:
  Work{positions,+inner,+rows,owners,geometry,activation} = work
  Cost.Configuration{mac_ns,+byte_ns,operation_ns,grain,wake} = config
  +span = span
  +packed = Nat.mul(32n,capacity(U32.max(Pipeline.block_tiles(),inner)))
  +output_capacity = capacity((rows * count : U32))
  +output = Nat.mul(4n,Nat.max(2n,output_capacity))
  # Padded lanes still execute every MAC, including absent rows of the last unit.
  +core = (U32.to_f32(((rows + 3) / 4 * 4 : U32)) * U32.to_f32(span) * U32.to_f32(inner) * mac_ns : F32)
  # Initializing each allocation and transferring logical data are separate work.
  +packing = ((U32.to_f32(span) * U32.to_f32(inner) * 4.0 + payload_float(packed)) * byte_ns : F32)
  +stores = ((U32.to_f32(rows) * U32.to_f32(count) * 4.0 + payload_float(Nat.mul(4n,output_capacity))) * byte_ns : F32)
  +finish = (U32.to_f32(rows) * U32.to_f32(count) * activation_ns(activation,config) : F32)
  +time = (core + packing + stores + finish : F32)
  Movement{packed,output,0n,time,time,time}

def merge(bytes: Nat,transfer: Nat,byte_ns: F32,left: Movement,right: Movement) -> Movement:
  Movement{lp,lo,lc,lt,lw,lmin} = left
  Movement{rp,ro,rc,rt,rw,rmin} = right
  +copy = (payload_float(transfer) * byte_ns : F32)
  Movement{Nat.add(lp,rp),Nat.add(lo,ro),Nat.add(bytes,Nat.add(lc,rc)),
    (copy + F32.max(lt,rt) : F32),(copy + lw + rw : F32),F32.min(lmin,rmin)}

def movement(depth: Nat,+offset: U32,+span: U32,+window: InputRows,+work: Work,+config: Cost.Configuration) -> Movement:
  match depth:
    case 0n:
      Work{+positions,inner,rows,owners,geometry,activation} = work
      leaf_cost(span,U32.min(span,(positions - offset : U32)),work,config)
    case 1n+ +rest:
      Cost.Configuration{mac,byte_ns,operation_ns,grain,wake} = config
      +half = (span / 16 * 8 : U32)
      +fork = fork_input(offset,half,window,work)
      merge(fork_bytes(work,forked_bytes(fork)),fork_bytes(work,forked_transfer(fork)),byte_ns,movement(rest,offset,half,forked_left(fork),work,config),
        movement(rest,(offset + half : U32),(span - half : U32),window,work,config))

def workspace(+depth: U32,+work: Work,cost: Movement) -> Nat:
  Work{+positions,inner,rows,owners,geometry,activation} = work
  Movement{packed,leaves,copies,span,total,minimum} = cost
  Nat.add(weight_bytes(work),Nat.add(packed,Nat.add(copies,
    Bool.pick(Nat,U32.is_zero(depth),0n,leaves))))

# Estimate the partition on workers by (work - span) / workers + span, Graham's
# bound for greedy list scheduling. BendRT does not steal work: it deals forked
# tasks across rings that workers claim each turn, so this is an estimate, not
# a guarantee. One worker gives the total work, so a split only adds its
# copies; more workers shorten the span but pay copies.
def scheduled(+workers: U32,+span: F32,+total: F32) -> F32:
  ((total - span) / U32.to_f32(workers) + span : F32)

def estimate(+depth: U32,+work: Work,config: Cost.Configuration,cost: Movement,+workers: U32) -> F32:
  Work{+positions,+inner,+rows,owners,geometry,activation} = work
  Cost.Configuration{mac,+byte_ns,operation_ns,grain,wake} = config
  Movement{packed,leaves,copies,+span,+total,minimum} = cost
  +reorder = reorder_ns(work,config)
  # A depth-zero blocked leaf already owns the exact logical output.
  +gather = Bool.pick(F32,U32.is_zero(depth),0.0,
    ((U32.to_f32(rows) * U32.to_f32(positions) * 4.0 + payload_float(Nat.mul(4n,capacity((rows * positions : U32))))) * byte_ns : F32))
  (reorder + scheduled(workers,span,total) + gather : F32)

def fits(budget: Maybe<&2,Nat>,bytes: Nat) -> Bool:
  match budget:
    case None{}: True{}
    case Some{limit}: Nat.is_le(bytes,limit)

def choose(fits: Bool,depth: U32,bytes: Nat,+time: F32,best: Plan) -> Plan:
  Plan{+packed,old_depth,old_bytes,+old_time} = best
  Bool.pick(Plan,fits && (Bool.not(packed) || F32.is_lt(time,old_time)),Plan{True{},depth,bytes,time},Plan{packed,old_depth,old_bytes,old_time})

def candidate(+depth: U32,+work: Work,+config: Cost.Configuration,+budget: Maybe<&2,Nat>,+workers: U32,+cost: Movement,best: Plan) -> Plan:
  Cost.Configuration{mac,byte_ns,operation_ns,grain,wake} = config
  Movement{packed,leaves,copies,span,total,minimum} = cost
  choose((U32.is_zero(depth) || F32.is_le(grain,minimum)) && fits(budget,workspace(depth,work,cost)),
    depth,workspace(depth,work,cost),estimate(depth,work,config,cost,workers),best)

def candidates(remaining: Nat,allowed: Bool,+depth: U32,+tasks: U32,+ceiling: U32,+workers: U32,+span: U32,+work: Work,+config: Cost.Configuration,
  +budget: Maybe<&2,Nat>,best: Plan) -> Plan:
  match remaining allowed:
    case 0n allowed: best
    case 1n+rest False{}: best
    case 1n+rest True{}:
      +next_tasks = (tasks * 2 : U32)
      candidates(rest,U32.is_le(tasks,(ceiling / 2 : U32)) && U32.is_le(1,(span / 8 / next_tasks : U32)),
        (depth + 1 : U32),next_tasks,ceiling,workers,span,work,config,budget,
        candidate(depth,work,config,budget,workers,movement(U32.to_nat(depth),0,span,root_rows(work),work,config),best))

# workers is the runtime's pool size. As in XNNPACK, several workers take up to
# five tasks each, so idle workers pick up the remaining ones; one worker never
# splits. Every leaf keeps at least one block and one grain of work, and the
# scheduling bound must strictly improve.
def select(+work: Work,workers: U32,+budget: Maybe<&2,Nat>,+config: Cost.Configuration) -> Plan:
  Work{positions,inner,rows,owners,geometry,activation} = work
  +workers = U32.max(1,U32.min(workers,858993459))
  candidates(32n,True{},0,1,Bool.pick(U32,U32.is_eq(workers,1),1,(workers * 5 : U32)),workers,KernelWindow.padded(positions),work,config,budget,
    Plan{False{},0,0n,0.0})
