# Quoted later-body cell cut

Original synthetic fixture. Input a ends with `> | al` after a complete header,
delimiter and preceding body row; b starts `pha | b |\n`. With exact source and
min_chars=0, emit one a record equal to the complete raw source table, one
rejoin_table edit, ordered prior provenance, no findings, coverage 1.0 and zero
second-fix edits. Preserve input records and all payload/prefix/whitespace bytes.

Generalize to any number of complete preceding body rows. Require one contiguous
valid table and one identical explicit raw single-level prefix on every row:
0..3 ASCII spaces, `>`, then one ASCII space. The cut must lie in a payload row,
with both unique ordered nonoverlapping aligned spans touching that same row;
raw concatenation must exactly equal adjacent source content. Verify the entire
local table is the same complete table run in full source, outside fenced code.
Conservatively refuse prior outside-code source lists for later-body recovery.

Refuse absent source, changed/missing bytes or whitespace, ambiguity even with
verified hints, overlap/reversal/fragmented alignments, partial source tables,
nested/mixed/missing prefixes, prefix-only/row-ending cuts, invalid delimiter or
non-pipe rows, blanks, lists, containers and multiple tables. No preambles.
Retain the existing pipe-table dialect (body cell counts may vary), header,
delimiter, first-body, whole-row, fence and conservation contracts. Preserve raw
Unicode, escaped pipes and LF/CRLF/CR. No fixed row-count allowlist.

Executable contracts: tests/lint/test_quoted_later_body_cuts.py and
 tests/lint/installed_quoted_later_body_smoke.py.
