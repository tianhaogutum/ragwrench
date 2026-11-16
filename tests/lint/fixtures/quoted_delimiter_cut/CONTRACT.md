# Quoted delimiter-cell cut

Original synthetic fixture; no splitter dependency. The two records touch inside
`--` on the source delimiter row. With source and min_chars=0, output must be one
h record with text exactly equal to source.md, one rejoin_table edit, ordered
left/right metadata provenance, no issues, coverage 1.0, preserved characters
and zero changes/identical output on a second fix. Existing ordered provenance
must flatten in input order. Without source, retain both records and CL002.

Bound this recovery to one complete header plus a partial delimiter in the left
record. Both unique aligned ordered nonoverlapping spans must touch the same
source delimiter row after its prefix. Every raw line has the identical prefix:
0..3 ASCII spaces, `>`, then one ASCII space. Require an exact concatenation of
input text and one contiguous valid source table outside fences. Preserve every
prefix, whitespace byte, line ending and metadata value; synthesize no text.

Accept short/colon delimiters, optional outer pipes, Unicode, escaped pipes and
LF/CRLF/CR. Refuse nested/mixed/missing/changed prefixes, prefix cuts, list/fence
boundaries, preambles, blank-separated tables, invalid/count-mismatched
delimiters, missing characters, overlapping/reversed/fragmented alignments and
ambiguous source occurrences even with verified hints. Existing quoted header,
unquoted and whole-row repairs keep their contracts. Quoted body/prefix token
cuts and delimiter cuts after a preamble remain outside this round.

See tests/lint/test_quoted_delimiter_cuts.py and
 tests/lint/installed_quoted_delimiter_smoke.py for executable contracts.
