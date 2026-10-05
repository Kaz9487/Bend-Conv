"""Reject generated C in which a guarded constructor became shared.

The official compiler marks every constructor of a program as shared when a
live definition duplicates a value whose type is computed by a type function
from a value index, such as a certificate family applied to an owner. Matches
then take the reference-counting path and the unchanged convolution kernel
slows down. The library never duplicates partition tree constructors, so a
shared flag on them identifies the program-wide mode without depending on
constructor numbering.

A View is guarded for a second reason. The library builds a view for each use;
a View value that one definition hands to two others, or that a loop carries
through its iterations, makes the compiler share every View constructor, and
each match on a view then takes the reference-counting path. The launcher
checks every compilation with --any-program, which leaves the View out: a
program's own definitions may share a view, as the internal view checks do.
"""

import re
import sys
from pathlib import Path

PROGRAM_WIDE = 'a duplicated value with a computed, value-indexed type made every constructor shared'
ANY_PROGRAM = {
    'LIB_TRAVERSAL_PARTITION_LEAF': PROGRAM_WIDE,
    'LIB_TRAVERSAL_PARTITION_FORK': PROGRAM_WIDE,
}
GUARDED = dict(ANY_PROGRAM, LIB_TENSOR_VIEW_VIEW='a View value is used more than once; build the view again for each use')


def check(path, guarded=GUARDED):
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    ids = {name: int(value) for name, value in re.findall(r'#define CID__*(\w+) (\d+)\n', text)}
    # Each constructor row is { arity, shared }.
    table = re.search(r'CID_T\[\]\[2\] = \{(.*)\};', text)
    if not table:
        return [f'{path}: no constructor sharing table']
    shared = [int(flag) for flag in re.findall(r'\{\s*\d+\s*,\s*(\d+)\s*\}', table.group(1))]
    present = [name for name in guarded if name in ids]
    errors = [f'{path}: {name} is shared; {guarded[name]}' for name in present if shared[ids[name]]]
    print(f'{path}: {sum(shared)} of {len(shared)} constructors shared; guarded {len(present)}')
    return errors


def main():
    paths = [argument for argument in sys.argv[1:] if argument != '--any-program']
    if not paths:
        sys.exit('Usage: check_generated_c.py [--any-program] GENERATED.c ...')
    guarded = ANY_PROGRAM if len(paths) < len(sys.argv) - 1 else GUARDED
    errors = [error for path in paths for error in check(path, guarded)]
    if errors:
        sys.exit('\n'.join(errors))


if __name__ == '__main__':
    main()
