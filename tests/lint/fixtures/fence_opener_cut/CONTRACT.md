# Exact source-backed fence opener token cut

Synthetic input; no external splitter or parser dependency. With min_chars=0,
chunks.jsonl plus source.md must produce exactly one chunk with ID a, text equal
to every byte of source.md, left metadata and ordered provenance for a then b.
There is exactly one rejoin_fence edit, no remaining findings and no second-fix
changes. Prior implementation leaves both records and CL001 on b's final line.

Cuts after one or two characters of an unquoted backtick/tilde source opener of length >=3 are supported. The adjacent four_fence_opener_cut
fixture covers the four-character extension; long_fence_opener_cut covers five
and tests cover longer lengths. Both records must have unique aligned,
ordered nonoverlapping spans. Their raw concatenation must equal the contiguous
source slice, including all whitespace. The complete matching closer must be in
the two records. Source scanning must prove this is an actual opener, rather than
a closer or literal code inside another fence. Preserve info strings and opaque
payload, including apparent headings, tables, lists and shorter fence markers.

Refuse this new path without source, when CL001 is disabled, on
missing bytes/closers, mismatched or invalid closers, ambiguous source matches
even with verified hints, changed whitespace, quote/list containers, and cuts
in info strings/payload/indentation or between lines. List parsing is outside the
scanner: conservatively refuse if any preceding outside source line starts a
list marker, or any line before the opener in the records is quoted/indented.
Existing complete-opener joins (including payload/info cuts already handled),
no-source behavior and table repairs keep their existing contracts.
