# Filter semantics in v0.1

These are explicit bridge semantics for the Dify operator names. The public contract does not fully specify type coercion, array behavior or date encodings. No expression evaluation, SQL, regex or template interpolation is used. Names must be allowlisted literal metadata keys.

| Operator | Accepted comparison value | Match behavior |
| --- | --- | --- |
| `contains`, `not contains` | string | Case-sensitive substring of a string, or exact membership of an array |
| `start with`, `end with` | string | Case-sensitive string prefix/suffix |
| `is`, `is not` | string | Exact string equality/inequality |
| `in`, `not in` | array of strings | Scalar string belongs/does not belong to the list |
| `empty`, `not empty` | omitted/ignored | Missing, null, empty string, empty array or empty object counts as empty |
| `=`, `≠`, `>`, `<`, `≥`, `≤` | finite number | Numeric comparison; booleans and numeric strings are excluded |
| `before`, `after` | ISO-8601 string | Strict chronological comparison; dates/naive datetimes mean UTC; offsets respected |

Outside `empty`/`not empty`, missing fields and mismatched types never match, including negative operators. `0` and `false` are not empty. Array membership is supported by contains/not contains only; array-valued `is`/`in` do not match. No recursive groups or coercion. Comparison strings are capped at 4096 characters and in/not in lists at 256 entries; oversized values produce HTTP 400. Invalid comparison dates are HTTP 400; an invalid date stored in a chunk simply cannot satisfy a date predicate. Only `and` and `or` are supported. Empty condition lists mean no restriction for either logical operator.

For Qdrant, string `is` becomes MatchValue; non-empty `in` becomes MatchAny; numeric inequalities below absolute 2^53 become Range. Numeric equality/inequality, empty tests, date comparisons, substring/prefix/suffix tests, negative tests and large numeric bounds are post-filters. The integer native equality rule is deliberately avoided because Qdrant server integer keyword matching need not equal floating-point payload matching. Large numeric ranges stay local to avoid losing integer precision. Metadata paths use the configured metadata object field plus the allowlisted simple key.

Every native filter is a **superset** of bridge semantics: the Python predicate still runs on returned payloads to enforce scalar/array and missing-value rules. AND groups may push down their supported subset; OR groups push only when every condition has a native translation. This prevents an unsupported OR branch from being accidentally excluded. Empty `in` lists return no matches under the predicate; an empty `not in` list matches any string.

Post-filters retrieve additional ranked pages up to max_candidates. The service returns eligible results only after checking their payloads and normalized score. If more qualifying-score candidates remain beyond the budget before top_k is filled, it raises HTTP 422 rather than claiming an exhaustive empty/short result. Raise max_candidates or narrow the query/filter. An extra one-point probe distinguishes actual source exhaustion from reaching the budget. At most max_candidates + 1 points are fetched per query, in addition to the collection-schema check. The budget can be 1..100,000, overfetch 1..100. JSONL evaluates the full bounded in-memory corpus and does not use the Qdrant candidate budget.
