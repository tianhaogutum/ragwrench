# ragwrench cite-check

RAG answers often cite sources that do not exist, cite the wrong passage, or quote text
that never appears in the source. `ragwrench cite-check` checks exported answers from
Dify, RAGFlow, LlamaIndex or generic JSONL against their retrieved sources and reports
these problems with stable diagnostic codes. Use it as a fast, deterministic regression
gate in CI; it complements semantic or human evaluation rather than replacing it.

_Formerly the standalone `rag-cite-check` tool (`citecheck`)._

Deterministic, offline citation diagnostics for exported RAG answers. Catch
missing/wrong references, weak lexical support, literal quantity mismatches,
misattribution and invented quotations in regression suites. No model, API key,
network request or framework installation is needed at runtime.

This is a lexical CI gate: a clean report **does not prove that an answer is
true or entailed by its sources**. Negation, argument roles, paraphrases and
cross-language entailment are outside its scope.

## Quick start

The command uses only the Python standard library.

```sh
pip install ragwrench
ragwrench cite-check check --format dify examples/cite-check/dify_response.json
ragwrench cite-check check --format llamaindex examples/cite-check/llamaindex_response.json
ragwrench cite-check check --format ragflow examples/cite-check/ragflow_response.json
ragwrench cite-check check --jsonl examples/cite-check/generic.jsonl --report json
ragwrench cite-check bench
```

`bench` writes `citecheck-bench.json` in the current directory and prints its measured comparison. To
replace a previous report, run `ragwrench cite-check bench --force`.
Tests and benchmark are offline.

The wheel includes the replay corpus, so `bench` also works without a clone of the
repository. Upstream licenses are documented in [THIRD_PARTY.md](../THIRD_PARTY.md).

## Input formats and numbering

Each JSON input is one complete response. `--jsonl` accepts one complete
response per nonblank line for **any** adapter; all records must validate before
a report is written. Use `-` to read UTF-8 from stdin. There is no autodetection.

| `--format` | Answer | Source fields | Unified `idx` |
| --- | --- | --- | --- |
| `generic` (default) | `answer` | `sources[]: {idx, id?, title?, text}` | Explicit positive integer; sparse indices allowed |
| `llamaindex` | `response` | `source_nodes[].node: {id_?, text, metadata?: {file_name?, title?}}` | List order + 1, independent of document IDs and rank metadata |
| `dify` | `answer` | `metadata.retriever_resources[]: {position, segment_id?, document_id?, document_name?, content}` | Explicit `position`, never reassigned by list order |
| `ragflow` | `answer` or successful `{code:0,data:{answer,...}}` | `reference.chunks[]: {chunk_id?/id?, docnm_kwd?/document_name?, content_with_weight?/content}` | List order + 1; `##0$$` and `[ID:0]` map to source 1 |

Sources normalize to `{idx,id,title,text}`. IDs/titles are strings; fallback IDs
use position. LlamaIndex's supplied citation nodes must be the actual
CitationQueryEngine nodes, not unsplit original retrieval documents. Dify keeps
segment IDs where available. RAGFlow HTML source tables are reduced to text
without fetching or rendering anything. Empty source arrays are valid;
missing source arrays, empty source text, ambiguous duplicate indices,
multimodal/streaming values and wrong formats fail explicitly (exit 2).

Only the RAGFlow **chunks array** completion export is supported. OpenAI
`choices[].message` envelopes, keyed chunk dictionaries and SSE streams must be
converted/aggregated explicitly; they are rejected rather than guessed. This project is
independent of LlamaIndex, Dify and RAGFlow; it copies no upstream code and does not
change their licenses.

Minimal generic JSON:

```json
{
  "answer": "The warranty lasts 24 months [1]. Returns are free [3].",
  "sources": [
    {"idx": 1, "id": "w", "title": "warranty.md", "text": "The warranty lasts 12 months."},
    {"idx": 2, "id": "r", "title": "returns.md", "text": "Returns are free within 30 days."}
  ]
}
```

The equivalent Dify fixture is runnable:

```sh
ragwrench cite-check check --format dify examples/cite-check/dify_errors.json --report json
# Exit 1: CC004 (24 vs 12 months), CC001 ([3] missing),
# CC005 (source 2 supports returns), CC007 (source 2 unused).
```

## Diagnostics and CI behavior

| Code | Default severity | Deterministic rule |
| --- | --- | --- |
| CC001 | error | Dangling index or malformed/non-numeric citation candidate, with character offsets |
| CC002 | warning | Claim-like declarative sentence has no syntactically valid citation |
| CC003 | warning | Every cited source covers less than `--support-threshold` of claim tokens (default 0.5) |
| CC004 | error | A literal normalized number/date/unit is absent from the union of cited sources |
| CC005 | hint | An uncited source covers at least `--best-source-threshold` (0.65) and exceeds best cited coverage by `--misattribution-margin` (0.3) |
| CC006 | error | A quoted span of at least two characters is absent from every cited source even after the documented normalization |
| CC007 | info | A retrieved source is never cited |
| CC008 | info (`error` with `--strict-quotes`) | A quoted span matches a cited source only after Unicode/whitespace normalization; no exact match exists in any cited source |

Supported markers: `[1]`, `[1,2]`, `[1][2]`, `【1】`, `【1,2】`, `Source 1`,
`source 1`, `##0$$`, `[ID:0]`. Bracket markers use ASCII numeric indices starting
at 1; RAGFlow markers start at 0. Groups allow up to 64 indices, each at most
nine digits. Bracketed non-link text is treated as a citation candidate;
`[abc]`, `[1,]`, unmatched/stray brackets and out-of-range indices produce CC001.
Malformed markers do not satisfy CC002. A well-formed dangling citation is
reported by CC001 rather than also being called uncited. Ranges such as `[1-3]`
are rejected, not expanded.

The scanner handles EN/CJK sentence endings, decimals, some abbreviations,
quoted text, line boundaries and citations immediately **after** punctuation.
A citation after terminal punctuation belongs to the preceding sentence on
that same line. Code fences, indented code, Markdown headings, inline code,
inline link destinations, URLs and HTML tags are masked with offsets preserved.
This is a small Markdown dialect, not a complete CommonMark parser; reference
links, backslash escapes other than inline backtick openers and destination
parentheses, container blocks, tables-as-prose and exotic abbreviations can produce
incorrect diagnostics. Code/heading-only answers explicitly report zero
analyzed prose sentences; they do not fail by default.

Inline code pairs complete backtick runs of **equal length**, following the
delimiter rule in [CommonMark 0.31.2](https://spec.commonmark.org/0.31.2/#code-spans).
Shorter or longer runs inside the span are literal code. Single line endings
inside a matched span are masked too, so they cannot split a claim; each input
code point still occupies exactly one position. Spans cannot cross blank lines
or recognized heading/code blocks. Unmatched runs remain prose. A backtick
fence's info string cannot contain backticks, so line-leading triple-backtick
inline code does not hide the prose following it.

An odd run of backslashes before an inline backtick opener escapes its first
backtick; an even run allows the full delimiter to open code. Remaining
backticks after an escaped one may open a shorter span. Backslashes inside an
open code span stay literal and cannot escape its closer. For example,
`` \`literal [9]\` `` remains prose, so a missing source 9 produces CC001 and
exit 1. Destination parentheses use the same odd/even backslash convention described
below. Other Markdown backslash escapes remain outside this small dialect.

Single-line HTML tags recognized by this dialect are opaque outside code:
backticks in tag attributes cannot open a code span and hide following prose.
For example, `` <span title="`">literal [9]` `` with no sources reports CC001 for
`[9]`. HTML-looking text inside an already-open code span stays literal code;
its backticks can still close that span. Single-line `<…>` tags end at the first
unquoted `>`; balanced single- and double-quoted attribute values can contain
`>` and backticks. HTML quotes are not backslash-escaped. An unclosed quote or
a tag crossing a newline does not form a recognized tag and remains prose,
subject to the other code/link/URL rules. Whitespace immediately after `<`
(including Unicode whitespace) also leaves the angle-bracketed text as prose:
`说明🙂 < limit [9] >` reports dangling `[9]` at original offsets `12..15`,
with sentence offsets `0..17` and default exit 1. A decimal digit immediately
after `<` also starts prose: `说明🙂 <3 [9] >` reports only dangling `[9]`
at original offsets `7..10`, with sentence offsets `0..12` and default exit 1.
This includes Unicode decimal digits, such as `٣` and `３`. Code, links and URLs
within this prose still follow their usual rules. Ordinary opening, closing and
self-closing tags (including names with later digits, such as `<h3>`) and
http(s) autolinks retain their masking. Other punctuation/symbol-led angles also
remain prose: `说明🙂 <+ [9] >` reports only dangling `[9]` at `7..10`,
sentence `0..12`, and default exit 1. The supported punctuation openers are `/`
(closing tags), `!` (declarations/comments), and `?` (processing instructions);
their existing single-line masking is retained. For other openers the dialect
accepts Unicode word characters except decimal digits and underscore. This
keeps Unicode tag names while excluding punctuation and symbols such as `+`,
`-`, `=`, `_`, and `＋`. Code, links and URLs inside rejected angles still follow
their usual rules.

Whitespace immediately after the closing-tag opener `</` (including Unicode
whitespace) also leaves the angles as prose. `说明🙂 </ [9] >` with no sources
reports only dangling `[9]` at original offsets `7..10`, sentence `0..12`,
and default exit 1. A decimal digit immediately after `</` also starts prose:
`说明🙂 </3 [9] >` reports only dangling `[9]` at `8..11`, sentence `0..13`,
and default exit 1. This includes Unicode decimal digits such as `٣` and `３`.
Punctuation/symbols immediately after `</` also start prose:
`说明🙂 </+ [9] >` reports only dangling `[9]` at original offsets `8..11`,
sentence `0..13`, and default exit 1. Closing names require the same initial
Unicode word character as ordinary opening names, excluding decimal digits and
underscore. Thus `</_`, `</!`, `</?`, `</-`, `</＋` and `</🙂` start prose too;
`</>` has no name and remains prose. Ordinary `</em>`, `</h3>` and Unicode
closing names retain their masking; code, links, tags and URLs within rejected angles still follow
their usual rules.

This is only an immediate-opener boundary, not a full HTML parser or tag-name
validator. After a supported opening or closing prefix, tag bodies stay
permissive: for example, `</em+ [9]>` still masks its contents, as do closing
tags with attributes. Text after opening `<!` or `<?` also stays permissive;
non-decimal Unicode word numerals such as `²` and `Ⅲ` can still start opening
and closing names. Combining marks cannot start names; no tag-name normalization
is performed. General `!` declarations still end at the first unquoted `>` under the tag
rule; full declaration grammar and multiline HTML remain unsupported.

Complete same-line `<!-- ... -->` comments are opaque through the **first `-->`**.
Their bodies can contain `>`, balanced or unbalanced quotes and backticks;
these cannot expose citations or open code. For example, `说明🙂 <!-- > [9] -->`
with no sources has no findings or citation indices, sentence `0..18` with the
original text, and default exit 0. Empty `<!---->` is supported. Nested comment
openers are literal text, not nesting; text after the first terminator is scanned
normally. Outside `[9]` still reports CC001, including after a comment containing
backticks. An already-open code span retains precedence and can close inside
comment-looking text. Comments are recognized in source order alongside tags,
links and URLs, not masked before code pairing.

The special comment token cannot cross CR or LF. Incomplete or newline-crossing
candidates use the existing generic `!`-tag fallback and ordinary code/link/URL
rules; they are not guaranteed to stay entirely visible. For example,
`<!-- [8] > [9]` masks only through the first `>` and keeps `[9]` visible;
`<!-- [9]` has no tag closer and remains prose. `<!-- >\n[9] -->` keeps the
second-line citation visible. The legacy generic tag rule excludes LF only,
so a CR-only candidate such as `<!--\r[9] -->` can still be masked as a generic
tag. This is a bounded comment-token rule, not HTML body validation, nested
comment support or a full HTML parser.

Complete same-line `<? ... ?>` processing instructions are opaque through the
**first `?>`**, even when the body contains `>`, quotes or backticks. For example,
`说明🙂 <?note > [9] ?>` with no sources has no findings or citation indices,
one original sentence `0..19`, and default exit 0. Empty `<??>` is supported;
target names and bodies are not validated. Quotes do not delay the terminator,
and nested `<?` openers do not nest. Text after the first `?>` is scanned normally,
including outside citations. Code already open before `<?` retains precedence;
PIs inside recognized link destinations, tags or URLs follow source order too.

The special PI token cannot cross CR/LF. Incomplete/newline-crossing candidates
retain the generic `?`-tag and ordinary code/link/URL fallback, just as unsupported
comments retain the `!`-tag fallback. Thus `<?note [8] > [9]` hides only through
`>` and leaves `[9]` visible; `<?note [9]` stays prose. `<?note >\n[9] ?>`
leaves the second-line citation visible. The legacy generic tag rule excludes
LF only, so `<?note\r[9] ?>` can still mask as a generic tag. These rules do not
add multiline PI support, target/body validation or full XML/HTML parsing.

Complete literal same-line `<![CDATA[ ... ]]>` sections are opaque through the
**first `]]>`**, including bodies with `>`, quotes or backticks. For example,
`说明🙂 <![CDATA[ > [9] ]]>` with no sources has no findings or citation indices,
one original Unicode sentence `0..23`, and default exit 0. Empty `<![CDATA[]]>`
is supported. Quotes do not delay the terminator; nested openers are literal
body text. Outside citations are scanned normally after the first terminator.
Already-open code and recognized tags, comments, PIs, destinations and URLs
retain their source-order precedence.

Only the exact case-sensitive `<![CDATA[` opener gets this special rule.
Incomplete or CR/LF-crossing candidates retain the generic `!`-tag and ordinary
code/link/URL fallback: `<![CDATA[ [8] > [9]` masks only through `>` and leaves
`[9]` visible, while `<![CDATA[ [9]` stays prose. `<![CDATA[ >\n[9] ]]>`
leaves the second-line citation visible. Generic tags exclude LF only, so
`<![CDATA[\r[9] ]]>` can still mask as a generic tag. Lowercase/misspelled openers
keep that same generic fallback. No CDATA body validation, nesting, multiline
support, general declaration grammar or full XML/HTML parsing is added.
Comment/PI/CDATA scans reuse the next terminator, newline or end for repeated
candidates.

```sh
ragwrench cite-check check --jsonl examples/cite-check/html_quotes.jsonl --report json
# Exit 1 at the default error gate: records 1/2 have only CC001 for [9], 29..32.
# Original Unicode sentences span 0..33; record 3 is code with no findings.
ragwrench cite-check check --jsonl examples/cite-check/html_openers.jsonl --report json
# Exit 1: angle-bracketed prose and quoted-tag controls retain real [9].
# Code and URL-autolink controls remain clean.
ragwrench cite-check check --jsonl examples/cite-check/html_digits.jsonl --report json
# Exit 1: <3 / <３ prose retains [9] at 7..10, sentence 0..12.
# Quoted tags and link destinations hide their contents; code/autolinks are clean.
ragwrench cite-check check --jsonl examples/cite-check/html_punctuation.jsonl --report json
# Exit 1: <+ / <- prose retains [9] at 7..10, sentence 0..12.
# Quoted tags/comments and nested destinations hide their contents.
# Code/autolinks and the supported source-4 quotation are clean.
ragwrench cite-check check --jsonl examples/cite-check/html_closing_whitespace.jsonl --report json
# Exit 1: </ followed by space or EM SPACE retains [9] at 7..10, sentence 0..12.
# Quoted tags and nested destinations hide their contents; code/autolinks and
# the supported source-4 quotation are clean.
ragwrench cite-check check --jsonl examples/cite-check/html_closing_punctuation.jsonl --report json
# Exit 1: </+ / </_ / </＋ / </🙂 / </= prose retains [9] at 8..11, sentence 0..13.
# Ordinary/Unicode/quoted closing tags, !/? openers and the source-4 quote are clean.
ragwrench cite-check check --jsonl examples/cite-check/html_closing_digits.jsonl --report json
# Exit 1: </3, </٣ and </３ prose retains [9] at 8..11, sentence 0..13.
# Quoted tags and nested destinations hide their contents; code/autolinks and
# the supported source-4 quotation are clean.
ragwrench cite-check check --jsonl examples/cite-check/html_comments.jsonl --report json
# Exit 1: records 2–6 retain outside/fallback [9]; record 1 (the exact example),
# comment-looking code and the supported source-4 quotation are clean.
ragwrench cite-check check --jsonl examples/cite-check/processing_instructions.jsonl --report json
# Exit 1: records 2–5 retain outside/fallback [9]; the exact PI, code and
# supported source-4 quotation are clean.
ragwrench cite-check check --jsonl examples/cite-check/cdata_sections.jsonl --report json
# Exit 1: records 2–5 retain outside/fallback [9]; the exact CDATA, code and
# supported source-4 quotation are clean.
```

Recognized inline links keep their labels as prose and hide their delimiters
and destinations. Outside code, destination backticks cannot open code spans.
For example, `` [archive](https://example.invalid/`)literal [9]` `` keeps
`archive` and `literal` visible and reports only dangling `[9]` when no source 9
exists. Code and tags within labels retain their usual masking. An already-open
code span takes precedence over link-looking contents, even when its closing
backticks appear in a link-looking destination.

Destinations start with `](` immediately after a recognized label and end when
the outer parentheses balance on the same line. Nested ASCII parentheses are
supported: `说明🙂 [archive](part(one)[8]) [9]` masks the complete destination,
keeps `archive` visible, and reports only dangling `[9]` at original offsets
`28..31` (sentence `0..31`) when no sources exist. Brackets, backticks, quotes
and tag-like text inside the destination are opaque. An odd backslash run before
a parenthesis makes it literal; an even run leaves it structural. Backslashes
otherwise remain literal; this does not add general Markdown escape support.

Each candidate scans at most 10,000 code points and permits depth 64, counting
both outer parentheses in the scan budget and the outer pair as depth 1.
A missing outer closer, CR/LF, or exceeded scan/depth bound leaves the candidate
unrecognized: no partial link mask is applied, and the existing prose, code,
tag and URL rules still run. Such syntax may produce citation diagnostics;
it does not itself raise an input-limit error. Empty/ordinary destinations and
image-style `![label](destination)` retain the same label/masking behavior.
Reference links, multiline destinations, nested labels, title/angle-destination
parsing and full CommonMark syntax remain outside this bounded dialect.

```sh
ragwrench cite-check check --jsonl examples/cite-check/link_boundaries.jsonl --report json
# Exit 1 at the default error gate: record 1 has only CC001 for [9], 48..51.
# Record 2 contains link-looking code and has no findings.
ragwrench cite-check check --jsonl examples/cite-check/nested_destinations.jsonl --report json
# Exit 1: record 1 has only CC001 for outside [9], 28..31, sentence 0..31.
# Record 2 cites existing source 4 with full lexical coverage and no findings;
# record 3 contains link-looking code and has no findings.
```

These offsets count original Unicode code points, including the CJK/emoji
prefix. Sentence text and offsets still refer to the original answer.

Bare `http://` and `https://` URLs outside code are also opaque: URL backticks
cannot open a span and hide following prose. The scheme letters are ASCII
case-insensitive (`HTTPS://`, `hTtP://`). Other schemes (for example `ftp://`)
and Unicode lookalikes (for example fullwidth `ＨＴＴＰＳ://`) are not URLs. The
existing URL boundary is unchanged: URLs end at whitespace or `<`/`>`, and
trailing punctuation belongs to the URL. An already-open code span keeps
precedence over URL-looking text, including a closing backtick within it. URLs
are never fetched.

```sh
ragwrench cite-check check --jsonl examples/cite-check/url_boundaries.jsonl --report json
# Exit 1 at the default error gate: record 1 has only CC001 for [9], 38..41.
# Its original Unicode sentence spans 0..42; record 2 is code with no findings.
ragwrench cite-check check --jsonl examples/cite-check/url_schemes.jsonl --report json
# Exit 1: records 3/4/5 have only CC001 for [9] at 30..33, 15..18, 17..20: the
# backtick-after-URL record, the ftp:// record and the fullwidth-scheme record.
# Records 1/2 (HTTPS and hTtP URLs), 6 (code) and 7 (source-4 citation) are clean.
```

The local synthetic fixture checks escaped prose, literal backslashes inside
code, a backtick in an HTML attribute, and HTML-looking code:

```sh
ragwrench cite-check check --jsonl examples/cite-check/inline_boundaries.jsonl --report json
# Exit 1: records 1/3 have CC001 at original code-point offsets 14..17/28..31;
# records 2/4 have no findings.
```

For example, the answer below with source 1 text `Supported syntax is documented.`
has only citation `[1]`, no findings, and exit 0:

````text
Supported syntax is ``use `x` [9]`` [1].
````

The local synthetic integration fixture also covers multiline code containing
quantities/quotes and line-leading triple-backtick code:

```sh
ragwrench cite-check check --jsonl examples/cite-check/backtick_code.jsonl --report json --fail-on info
```

A `[9]` outside code, including after an unmatched delimiter, still produces
CC001 and exit 1 when source 9 is missing. Original answer text and Unicode
code-point offsets remain available in JSON reports.

A claim-like sentence has at least `--min-claim-tokens` distinct non-stopword
tokens (default 2), or a numeric literal. Questions are skipped for CC002.
Coverage is `|claim_tokens ∩ source_tokens| / |claim_tokens|`, using NFKC,
case-folded word tokens, common English stopwords and CJK character bigrams.
Thousands separators and supported unit aliases normalize. There is no learned
model, stemming, synonym expansion or translation. Multi-source coverage uses
the **best individual** source for CC003/CC005; it does not infer logical support
from combinations of passages. Ties choose the lowest source index.

Quantities normalize signed base-10 decimals (`1,000` = `1000`, `1.50` = `1.5`),
fullwidth digits, percentages, numeric dates (`YYYY-MM-DD`, `YYYY/M/D`,
`YYYY年M月D日`) and a fixed vocabulary: percent, month/year/day/hour/minute/second,
kg/g/km/m/cm, MB/GB, USD/EUR/CNY and °C, with listed English singular/plural and
common Chinese aliases in `lexical.py`. A specified unit must match; an untyped
number can match that value with any unit. No calendar validation, arithmetic,
scale conversion, spelled-out numbers, locale decimal commas, scientific
notation or ratios are interpreted. Unknown unit words are treated as unitless;
these limitations appear in each report. Merely finding a number somewhere in
a source does not prove that it describes the right entity.

Quote contents are checked against each cited source's extracted prose, first
as an exact case-sensitive substring, then using NFKC, case-folding and collapsed
whitespace. NFKC covers fullwidth letters/digits/punctuation. Additional explicit
equivalences are `。` → `.`, `、` → `,`, `“”「」『』` → `"`, and `‘’` → `'`.
Punctuation is mapped, never deleted; `信号：稳定` cannot match `信号稳定`.
Recognized outer delimiters are `"…"`, `“…”`, `「…」`, `『…』`, `＂…＂`, and
`｢…｣`; apostrophes/single quotes are not outer delimiters. Quote extraction is
single-line, non-nesting, and limited to spans of at least two original code
points. These are deterministic text equivalences, not translation or semantic
support.

An exact match in **any cited source** suppresses quote findings. A match only
after normalization produces **CC008**, which is informational by default.
`--strict-quotes` promotes CC008 to an error so the default error gate rejects
altered literal quotations. **CC006** remains the error for an unsupported quote
in both modes; normalization never turns an unsupported quote into a strict-mode
normalization finding. Other diagnostics and `--fail-on` still apply normally:
`--fail-on info` gates informational CC008 too, and `--fail-on none` disables all
gating, including strict quote errors.

For example, `手册说「信号：稳定」【1】。` citing source 1 `手册说信号:稳定。`
produces CC008 (exit 0 by default, exit 1 with `--strict-quotes`). Its JSON evidence
retains `quote: "信号：稳定"`, the exclusive original answer offsets `start: 4,
end: 9`, `normalized_quote: "信号:稳定"`, and `match_kind: "normalized"`.
CC008 lists only cited source indices that matched after normalization. Changing
the quote to `信号：中断` produces CC006 with `match_kind: "unsupported"` and
exit 1 in either mode. Neither mode rewrites the answer or source text. Source
matching uses the existing HTML-to-prose extraction, not raw HTML markup; quote
evidence and sentence text always retain the original answer characters.

CC008 also includes `source_match_count` (all matching cited sources) and
`source_matches` (the first 50, in citation order, without duplicate indices).
Each entry contains `source_idx`, `excerpt`, `start`, `end`, and
`offset_basis: "source_prose"`. For the example above this is
`{source_idx: 1, excerpt: "信号:稳定", start: 3, end: 8,
offset_basis: "source_prose"}`. Each source contributes its **first normalized
occurrence**; later occurrences are not enumerated. CC006 has no source matches,
and an exact match in any cited source still suppresses CC008 entirely.

Source offsets are exclusive Python Unicode code-point positions in
`ragwrench.citecheck.lexical.source_text(source.text)`, **not raw HTML or normalized-text
positions**. Extraction decodes HTML character references and joins HTMLParser
text-data chunks with one space; it is not browser rendering. The excerpt is
exactly that prose slice, preserving original case, combining marks and internal
whitespace. NFKC composition, character expansion and casefolding preserve the
contributing source positions. A match inside an expansion uses the smallest
contiguous slice covering its contributing original characters: e.g. `"FI"`
matching `ﬃ` reports excerpt `ﬃ`, whose normalized form also includes the first
`f`. Reordered combining marks can likewise require a covering slice containing
intervening marks. The existing normalized substring rule is unchanged.

The local synthetic RAGFlow HTML example decodes `cafe&#x301;`, inserts a space
between text nodes, and preserves a tab. It reports excerpt `"cafe\u0301 &\tSTABLE"`
at source-prose offsets `26..40` for answer quote `"CAFÉ & stable"`:

```sh
ragwrench cite-check check --format ragflow examples/cite-check/source_quote_evidence.json --report json
# Exit 0: CC008 info, source_match_count 1, source_matches[0] as above.
ragwrench cite-check check --format ragflow examples/cite-check/source_quote_evidence.json --strict-quotes --report json
# Exit 1: same evidence, CC008 error.
```

The local synthetic JSONL fixture contains an exact fullwidth-delimited quote,
two normalization-only quotes, and one unsupported quote:

```sh
ragwrench cite-check check --jsonl examples/cite-check/unicode_quotes.jsonl --report json
# Exit 1: records 2/3 have informational CC008; record 4 has error CC006.
ragwrench cite-check check --jsonl examples/cite-check/unicode_quotes.jsonl --strict-quotes --report json
# Exit 1: records 2/3 have error CC008; record 4 still has CC006.
```

Default output is Markdown; `--report json` emits a versioned report with each
record's sentences, offsets, source indices, coverage, findings, evidence and
scope notes. Answer offsets are Python Unicode code-point offsets into the
original answer, with exclusive `end`; nested CC008 source offsets use the
explicit `source_prose` basis above. Reports include answer sentences and selected
literal/quote evidence, including matched source excerpts (possibly an entire
short source); treat reports with the same confidentiality as inputs. Source
IDs, titles and arbitrary metadata are not echoed.

```sh
ragwrench cite-check check --format dify examples/cite-check/dify_errors.json \
  --fail-on warning --report json -o regression.json
```

Exit codes: **0** = no finding at/above the chosen level; **1** = gate failed;
**2** = invalid input/configuration, unsupported shape, I/O or output error.
`--fail-on` accepts `error` (default), `warning`, `hint`, `info`, `none`.
Ordering is `info < hint < warning < error`; `none` never gates findings.
Use `--fail-on warning` to gate uncited and weakly supported claims as well as
structural errors. Thresholds must be finite and in `[0,1]`; token minimum is
an integer in `[1,100]`.

Output files must be under the current working directory, in an existing parent
directory, and cannot overwrite input files or traverse symlinks. Existing
reports need `--force`; writes are atomic. These checks assume the caller
controls the directory (they are not protection against concurrent malicious
filesystem changes). Inputs are regular local files or stdin; exported IDs,
paths, HTML and URLs are never executed, followed or fetched. No shell commands,
workflow steps, remote services or dry-run side effects exist in the checker.

Limits fail explicitly with exit 2: 8 MiB per input file, JSON depth 64, 1,000
JSONL records, 100,000 answer characters, 256 sources, 100,000 characters per
source field, 2,000,000 combined source-field characters per record, 1,000
sentences/record, 10,000 characters/sentence, 20,000 source quantities/record,
256 quantities/sentence, 128 characters/numeric literal and 10,000 findings per
record. Evidence lists are capped at 50 items (with a quantity count provided).
Inputs are not truncated to make a gate pass. Duplicate JSON keys/non-finite
numbers are rejected. Input errors are terse and do not quote payload values.

## Reproducible benchmark

The MIT-licensed corpus contains **78 labelled synthetic cases**, including clean
controls, injected dangling/malformed markers, citation deletion, unrelated
sources, numeric/date/unit substitution, wrong-source attribution, altered
quotes, normalization-only quotes, unused sources, mixed language and known
semantic blind spots.
[benchmarks/cite-check/build_corpus.py](../benchmarks/cite-check/build_corpus.py) is the human-authored
case/label recipe; it never calls the checker. Its generated JSONL is packaged
at `src/ragwrench/citecheck/data/benchmark.jsonl`.

```sh
python3 benchmarks/cite-check/build_corpus.py
ragwrench cite-check bench --force
ragwrench cite-check bench --cases src/ragwrench/citecheck/data/benchmark.jsonl \
  --support-threshold 0.7 -o experiment.json
```

Every run computes per-check TP/FP/FN/TN, precision, recall and F1, plus micro
metrics, dataset SHA-256, actual single-process timings and each case's label,
prediction, false positives/negatives and baseline prediction. Evaluation counts
**presence of each code per case**, including informational CC007/CC008; it does not
measure finding-level localization. Undefined precision/recall is `null`, not
an invented perfect score. `bench` exits 0 after a successful measurement even
if metrics regress; it is a measurement command rather than an accuracy gate.

The marker-exists baseline shares exactly the same marker/Markdown parser and
predicts only CC001 when a reference cannot resolve. It has no lexical, numeric
or quote checks. On the bundled regression corpus, the recorded replay gives:

| Method | Micro TP / FP / FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| ragwrench cite-check | 75 / 0 / 0 | 1.000 | 1.000 | 1.000 |
| Marker-exists baseline | 12 / 0 / 63 | 1.000 | 0.160 | 0.275862 |

The complete run-written report is [citecheck-bench.json](../benchmarks/cite-check/citecheck-bench.json).
This deliberately simple corpus exercises the implemented checks; it is not
held-out data and these scores must not be generalized to production accuracy.
The negation/argument-swap cases are explicitly labelled as known blind spots
of the **diagnostic codes**, not semantically correct answers. Their lack of
semantic findings is not evidence of faithfulness. Add reviewed deployment
cases with the same `{id, expected:[codes], record, format?}` schema before
choosing thresholds for a real gate. Model-judge accuracy/cost claims are not
measured here. Runtime/timings vary; no speedup over the simpler baseline is
claimed.

## Scope

The test suite in `tests/citecheck/` covers each diagnostic on real framework export
fixtures and synthetic edge cases.

No LLM/NLI entailment, fetching live sources, rewriting answers, server, telemetry, paid
APIs, publishing or production-data access is included. Original code and synthetic
fixtures are MIT licensed.
