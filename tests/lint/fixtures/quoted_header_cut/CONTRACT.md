# Quoted header-token cut

Original synthetic fixture; no splitter or third-party generator is required.
`chunks.jsonl` divides `source.md` between a backslash and its escaped pipe in
`café \| 雪`. The left chunk already contains ordered provenance for h0/h1;
right metadata must append after those inputs.

With `--source source.md --min-chars 0 --fail-on warning`, check exits 1 before
fix, fix/check exit 0 afterward. Output is one h record containing the exact
UTF-8 source, one rejoin_table edit, provenance [h0, h1, r], full lexical source
coverage and identical bytes/zero changes on a second fix. Without source,
retain the original records and report CL002.

Only one table with an identical raw prefix on every line is supported: zero
to three ASCII spaces, then `> ` (one ASCII space). The left chunk contains
only the partial header; the cut must follow this prefix and precede the header
line end. Require a unique aligned ordered nonoverlapping span per chunk and
an exact raw contiguous source slice outside fenced code. Reject prefix cuts,
nested/mixed depths, missing/changed prefixes, preambles, list/fence boundaries,
blank-separated tables, invalid delimiters and ambiguous source occurrences,
even when supplied hints select one occurrence. No header or delimiter is made
up; prefix and whitespace bytes are retained.

`tests/lint/test_quoted_header_cuts.py` covers these contracts and LF/CRLF/CR,
Unicode, escaped pipes, optional outer pipes, offset hints, source context,
conservation and idempotence. `tests/lint/installed_quoted_header_smoke.py` repeats
check/fix/check/second-fix against an installed wheel outside the repository.
