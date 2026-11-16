# Exact four-character opener cut

Synthetic input, no splitter dependency. With min_chars=0 and source.md,
chunks.jsonl must become one ID a record containing every exact source byte,
ordered a/b metadata provenance, one rejoin_fence edit, no findings, coverage
1.0 and zero second-fix changes. Before implementation this pair remains
unchanged with CL001 on b despite unique aligned ordered adjacent spans.

Support only unquoted four-backtick/four-tilde openers cut after one/two marker
characters. Require exact raw source concatenation and unique ordered adjacent
nonoverlapping spans even with hints. Full-source scanning must prove the cut
is inside an actual opener and a complete matching closer is within the pair.
A shorter three-character marker stays literal code, including apparent tables,
headings and lists before and after it. Preserve preambles, info strings,
Unicode, all payload/whitespace/line-ending bytes, left ID and prior provenance.

Refuse missing bytes/closers, wrong/short/invalid or truncated closers, changed
whitespace, ambiguous matches even with hints, reversed/overlapping spans,
quoted/list containers, literal code and full-source closers. No source or
CL001 disabled leaves this cut unchanged. Retain three-character recovery,
complete-opener joins and table repairs. Five-plus-character opener token cuts
remain unsupported; cuts after three characters use the existing complete-opener
path rather than this reconstruction path.
