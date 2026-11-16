# Quoted first-body-cell cut

Original synthetic fixture. Input h ends with `> | al` after one complete header
and delimiter; r starts `pha | b |\n`. The exact source is a complete quoted table.
With source and min_chars=0, emit one h record with the exact source bytes, one
rejoin_table edit, ordered provenance, no CL002, coverage 1.0, conserved characters
and zero second-fix edits. Flatten prior input provenance in its existing order.

This fixture covers first-body token recovery. Every line must have the identical
raw single-level prefix: 0..3 ASCII spaces, `>`, then one ASCII space. The left
record contains exactly a whole header, whole delimiter and partial first body
row. Both unique aligned ordered nonoverlapping spans must touch that same source
row after its prefix. Require exact concatenated raw source and one contiguous
valid quoted table outside fences; invent no payload, schema or prefix bytes.
Accept Unicode, escaped pipes, short/colon delimiter cells and LF/CRLF/CR.

Refuse no source, nested/mixed/missing/changed prefixes, prefix/list/fence
boundaries, preambles, blank-separated tables, invalid/count-mismatched
delimiters, missing characters, overlapping/reversed/fragmented alignments and
ambiguous occurrences even with verified hints. Later body rows are covered separately by the quoted_later_body_cut fixture. Existing unquoted/header/delimiter and whole-row repair
keep their behavior.

Executable contracts: tests/lint/test_quoted_body_cuts.py and
 tests/lint/installed_quoted_body_smoke.py.
