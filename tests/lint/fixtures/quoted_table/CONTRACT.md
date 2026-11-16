# Explicit quoted table contract

This original synthetic table has five raw lines at quote depth two, with
different spellings of that prefix, Unicode and an escaped pipe. The pinned
LangChain generator cuts it at line boundaries, separating header, delimiter
and body rows. No framework package is needed to consume the committed fixture.

With `min_chars=0` and the default maximum, `fix` must emit one chunk whose text
equals `source.md` exactly, with every input metadata object in ordered merge
provenance, 100% source coverage, no CL002 and no changes on a second fix.
The same repair works without a source when all raw line endings survive.
`check` flags each incomplete input run with CL002 at its original chunk line;
the repaired fixture passes `--fail-on warning`.

Parsing removes explicit quote prefixes only for recognition. Consecutive rows
must have the same depth; prefix spelling can vary. Header and delimiter need
equal cell counts. A missing header, mismatched count, quote-depth transition,
bare/quoted blank, intervening prose or fence prevents rejoining. A leading
blank on the right is a boundary, even if later lines look like table rows.
An already complete right-hand table stays separate. Fenced content is opaque.
If verified source whitespace restores a blank between fragments, no join is
made. CL002 disable suppresses table joining. No headers or prefixes are guessed.

The existing CL005 row splitter may split a recognized oversized complete table;
it must repeat the exact raw header/delimiter and retain every raw body row in
order. An indivisible row remains CL005. Lazy quote continuation, cuts inside
prefixes/rows, inline-code pipes and other Markdown containers are outside this
contract; this is the existing simple pipe-table dialect inside explicit quotes.
