# Exact long fence opener-token reconstruction

Synthetic input splits the five-backtick opener after two marker characters.
With source and min_chars=0, expect one left-ID `a` record whose text is byte-for-byte
`source.md`, one `rejoin_fence` change, ordered left/right metadata provenance,
no findings, coverage 1.0 and a byte-identical second fix with no edits.
The initial check may succeed: the right suffix looks like a complete shorter fence.

Generalize to any length >=3 with cut after one/two marker characters only.
Test both backticks and tildes at lengths 5/6/8/37, raw LF/CRLF/CR, indentation,
Unicode info/payload, every shorter interior marker, and table/heading-like code
on both sides. Preserve left ID and flatten prior provenance in order.

Refuse missing/changed bytes, missing/short/wrong/invalid/truncated closers,
ambiguous left/right/both even with verified hints, quote/list containers,
literal code in the full source, source closers, overlap/reversal, absent source
and disabled CL001. The helper refuses cuts in indentation, info, payload and
after >=3 markers; existing complete-opener behavior remains in place.
