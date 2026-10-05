# Prediction parameters, not machine bounds or correctness certificates. The
# worker count is not a profile value: callers pass the runtime's pool size.
import Base

type Configuration is Data:
  Configuration{mac_ns: F32,copy_byte_ns: F32,operation_ns: F32,grain_ns: F32,wake_round_ns: F32}

# One element's cost. Bytes price loading, storing and the loop that moves
# them, as the copies that measured copy_byte_ns do; operations are the other
# scalar work, counted from source.
def element_ns(+bytes: F32,+operations: F32,config: Configuration) -> F32:
  Configuration{mac_ns,byte_ns,operation_ns,grain,wake} = config
  (bytes * byte_ns + operations * operation_ns : F32)

# The official runtime grows fork tasks and queues fork-free tasks for its
# work round (monk_step, row_grow, cube_run). No fixed number of leaves can
# be subtracted as already completed by growth. Price complete waves of at
# most workers leaves, using the largest path (including clones) per wave.
# This is a conservative scheduling estimate, not a bound on the runtime's
# static ring distribution. Callers charge the two wake rounds separately.
def region_ns(+workers: U32,+leaves: U32,+span: F32) -> F32:
  +pool = U32.max(workers,1)
  +waves = (leaves / pool + Bool.pick(U32,U32.is_zero((leaves % pool : U32)),0,1) : U32)
  (span * U32.to_f32(waves) : F32)
