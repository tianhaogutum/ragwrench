# Quoted fence opener cut

The exact adjacent records split after two backticks in a single-level `> `
quoted opener. With source and min_chars=0, recover one left-ID a record equal
to source.md byte for byte, one rejoin_fence edit, ordered metadata provenance,
no findings, coverage 1.0 and zero second-fix edits.

Require one consistent explicit raw prefix (0-3 spaces then `> `), one/two
marker characters on the left, unique aligned ordered adjacent nonoverlapping
spans, exact raw concatenation, actual full-source opener identity and a complete
matching closer in the pair. Both marker types and any length >=3 are supported.
No source, changed/missing bytes, invalid/incomplete closer, ambiguous matches
even with hints, lists, nested/mixed prefixes, literal-code/source-closer,
overlap/reversal and wrong cuts must not activate this fallback. Existing
complete-opener repair retains its separate contract.
