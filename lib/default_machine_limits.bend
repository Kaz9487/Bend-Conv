# The original execution envelope, with every binary constraint checked here.
import Base
import ./machine_limits.bend as Machine

# The capacity as a word, for run-time guards that need no other limit.
def capacity() -> U32:
  1073741824

def limits() -> Machine.Limits:
  Machine.Limits{65535,capacity(),2147483647,2147483648,
    Machine.Facts{{==},{==},{==},{==},{==},{==},{==},{==},{==},{==},{==},{==},{==},{==},{==}}}
