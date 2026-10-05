"""Shared host-only counters for diagnostic copies of official generated C."""

import re


def replace_once(source, needle, replacement):
    """Match a unique C anchor without depending on its spaces or line breaks.

    Anchors are declarations or statements without string literals or comments.
    Identifier and punctuation changes still fail; this is not a C parser.
    """
    tokens = re.findall(r'\w+|[^\w\s]', needle)
    pattern = r'\s*'.join(
        rf'(?<!\w){re.escape(token)}(?!\w)' if re.fullmatch(r'\w+', token) else re.escape(token)
        for token in tokens
    )
    matches = list(re.finditer(pattern, source))
    assert len(matches) == 1, f'Compiler instrumentation anchor changed: {needle}'
    match = matches[0]
    return source[: match.start()] + replacement + source[match.end() :]


def instrument(source, phases):
    assert isinstance(phases, int) and phases > 0
    source = replace_once(
        source,
        'INLINE u64 heap_alloc(Env e, u32 cls) {',
        f"""
#if !DEVICE
static unsigned profile_phase = {phases};
static unsigned long long profile_requests[{phases}], profile_bytes[{phases}], profile_clones[{phases}], profile_copy_bytes[{phases}];
#endif
INLINE u64 heap_alloc(Env e, u32 cls) {{
#if !DEVICE
  if (profile_phase < {phases}) {{
    __atomic_fetch_add(&profile_requests[profile_phase], 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&profile_bytes[profile_phase], (1ull << cls) * 8, __ATOMIC_RELAXED);
  }}
#endif
""",
    )
    return replace_once(
        source,
        'OUTLINE Term blk_copy(Env e, Term a) {',
        f"""OUTLINE Term blk_copy(Env e, Term a) {{
#if !DEVICE
  if (profile_phase < {phases}) {{
    __atomic_fetch_add(&profile_clones[profile_phase], 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&profile_copy_bytes[profile_phase], (1ull << blk_span(a)) * 8, __ATOMIC_RELAXED);
  }}
#endif
""",
    )


def write_counts(filename, phases):
    """C fragment used only at a stopped/joined phase boundary."""
    return f"""
FILE*profile=fopen({filename},"w");if(!profile)exit(2);
fprintf(profile,"[");for(unsigned j=0;j<{phases};j++)fprintf(profile,
"%s{{\\"allocation_requests\\":%llu,\\"requested_bytes\\":%llu,\\"clone_calls\\":%llu,\\"clone_bytes\\":%llu}}",
j?",":"",profile_requests[j],profile_bytes[j],profile_clones[j],profile_copy_bytes[j]);
fprintf(profile,"]\\n");fclose(profile);
"""
