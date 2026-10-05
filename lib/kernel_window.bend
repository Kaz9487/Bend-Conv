# Input rows a spatial range reads. A window stores rows [first, first + rows)
# of every channel contiguously. The first row is a stride multiple, so packing
# a window is packing a shorter image with output rows shifted by first / stride.
import Base
import ./traversal_array.bend as Traversal

type Band is Data:
  Band{first: U32,rows: U32}

def first_row(+stride: U32,+padding: U32,+output_row: U32) -> U32:
  +top = (output_row * stride : U32)
  Bool.pick(U32,U32.is_le(padding,top),((top - padding) / stride * stride : U32),0)

def end_row(+height: U32,+kernel: U32,+stride: U32,+padding: U32,+output_row: U32) -> U32:
  +bottom = (output_row * stride + kernel : U32)
  Bool.pick(U32,U32.is_le(bottom,padding),0,U32.min(height,(bottom - padding : U32)))

# Rows read by output positions [offset, offset + span), kept inside the parent
# window [parent_first, parent_first + parent_rows). A band has at least one row,
# and its first row stays a stride multiple.
def band(+parent_first: U32,+parent_rows: U32,+height: U32,+kernel: U32,+stride: U32,+padding: U32,+output_width: U32,+offset: U32,+span: U32) -> Band:
  +parent_end = (parent_first + parent_rows : U32)
  +top = ((parent_end - 1) / stride * stride : U32)
  +first = U32.max(parent_first,U32.min(first_row(stride,padding,(offset / output_width : U32)),top))
  +last = end_row(height,kernel,stride,padding,((offset + span - 1) / output_width : U32))
  Band{first,(U32.min(parent_end,U32.max(last,(first + 1 : U32))) - first : U32)}

# The checked facts that let a band stand in for its parent over a nonempty
# span: rows inside the parent window, a stride-aligned first row that no read
# needs above, an end no read needs below (or the image end), and room for the
# kernel. The comparisons avoid wrapped sums: rows are bounded by the parent end
# minus the first row. Pipelines copy only a band that passes; any other split
# keeps a clone of the parent.
def covers(+parent_first: U32,+parent_rows: U32,+height: U32,+kernel: U32,+stride: U32,+padding: U32,+output_width: U32,+offset: U32,+span: U32,
  +band: Band) -> Bool:
  Band{+first,+rows} = band
  +parent_end = (parent_first + parent_rows : U32)
  +top = (offset / output_width * stride : U32)
  +bottom = ((offset + span - 1) / output_width * stride + kernel : U32)
  +end = (first + rows : U32)
  U32.is_lt(0,span) && U32.is_lt(0,rows) && U32.is_le(parent_first,first) && U32.is_le(first,parent_end) &&
    U32.is_le(rows,(parent_end - first : U32)) && U32.is_zero((first % stride : U32)) &&
    (U32.is_zero(first) || U32.is_le((first + padding : U32),top)) &&
    (U32.is_eq(end,height) || U32.is_le(bottom,(end + padding : U32))) &&
    U32.is_le(kernel,(rows + padding + padding : U32))

# A fork copies the left child's band only when rows are whole eight-position
# blocks and the band passes its coverage check; otherwise it clones the parent.
def copies_band(+parent_first: U32,+parent_rows: U32,+height: U32,+kernel: U32,+stride: U32,+padding: U32,+output_width: U32,+offset: U32,+span: U32,
  +band: Band) -> Bool:
  U32.is_zero((output_width % 8 : U32)) && covers(parent_first,parent_rows,height,kernel,stride,padding,output_width,offset,span,band)

# Output positions rounded up to whole eight-position blocks.
def padded(positions: U32) -> U32:
  ((positions + 7) / 8 * 8 : U32)

# Packing start of a window: output rows before its first row move the start
# back by first / stride rows. A window that starts at row 0 keeps the offset.
def start(+offset: U32,+stride: U32,+output_width: U32,+first: U32) -> U32:
  Bool.pick(U32,U32.is_zero(first),offset,(offset - first / stride * output_width : U32))

# Channel c copies count values from c * source_stride + source_start to
# c * target_stride. Each plane is one contiguous copy.
def copy_planes(remaining: Nat,+channel: U32,+count: U32,+source_stride: U32,+source_start: U32,+target_stride: U32,
  owners: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  match remaining:
    case 0n: owners
    case 1n+rest:
      (source,target) = owners
      copy_planes(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,
        Traversal.copy(~F32,U32.to_nat(count),(channel * source_stride + source_start : U32),(channel * target_stride : U32),source,target))
