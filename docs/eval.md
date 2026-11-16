# ragwrench eval

When you change a chunker, embedding model or reranker you need to know whether retrieval
got better or worse, and whether the difference is real or noise. `ragwrench eval` scores
saved retrieval runs (TREC or JSONL) against relevance judgments and compares a candidate
with a baseline using paired significance tests. Use `run` to score one run and
`compare` in CI to fail the build when a metric drops by more than a threshold you choose.

_Formerly the standalone `retrieval-regress-gate` tool (`retgate`)._

A deterministic retrieval evaluator and CI regression gate for changes to chunkers,
embedders and rerankers. It uses only the Python standard library: no models,
network calls, compiled extensions or services. Reads existing retrieval results;
it does not run a retriever.

## Quick start

```sh
pip install ragwrench
ragwrench eval run --qrels examples/eval/qrels.tsv --run examples/eval/base.trec \
  --metrics ndcg@10,mrr@10,recall@100
ragwrench eval compare --qrels examples/eval/qrels.tsv \
  --baseline examples/eval/base.trec --candidate examples/eval/cand.trec \
  --metrics ndcg@10,recall@100 --test randomization --seed 42 \
  --max-drop ndcg@10=0.01 --max-drop recall@100=0.02
```

The last command intentionally exits **1**: this synthetic candidate regresses.
Its nDCG@10 drops from 1.000000 to 0.703518, and recall@100 from 1 to 0.833333.
Without `--max-drop`, comparison reports results and exits 0 even for a regression.
Choose thresholds explicitly for CI.

`run` was called `eval` before (`ragwrench eval eval`); that spelling still works but
is no longer listed in `--help`.

## Metric semantics

Metrics are comma-separated `name@k`; `1 <= k <= 1000000`.
Judgments are integer grades 0–100. Every grade above zero is binary relevant.
Unjudged retrieved documents count as nonrelevant.

| Metric | Per-query definition |
|---|---|
| `ndcg@k` | DCG / ideal DCG; discount at rank r is `1/log2(r+1)` |
| `mrr@k` | Reciprocal first relevant rank in the top k, or 0 |
| `recall@k` | Relevant hits in top k / **all** relevant judgments |
| `precision@k` | Relevant hits in top k / **k**, including unfilled positions |
| `map@k` | Sum of precision at relevant ranks <= k / **all** relevant judgments |
| `hit@k` | 1 if the top k contains any relevant document, otherwise 0 |

`--gain linear` (default) uses grade as nDCG gain. `--gain exponential` uses
`2**grade - 1`. Queries with zero relevant judgments score 0 for all metrics.
Cutoffs beyond the returned list are supported; they do not fabricate hits.

**Ranking:** descending score, then **descending lexicographic document ID**
(`doc2` before `doc10`), matching
[trec_eval's comparator](https://github.com/usnistgov/trec_eval/blob/master/form_res_rels.c).
Supplied run ranks are syntax-checked but ignored for ranking. IDs are UTF-8,
case-sensitive strings; no numeric ID ordering or Unicode normalization is applied.
Default nDCG is the [linear-gain trec_eval definition](https://github.com/usnistgov/trec_eval/blob/master/m_ndcg_cut.c).
`map@k` uses the [trec_eval map_cut denominator](https://github.com/usnistgov/trec_eval/blob/master/m_map_cut.c),
which can differ from libraries dividing by `min(k, relevant_count)`.

**Query universe:** all qrels query IDs, paired by ID in sorted order. Missing run
queries score 0, including when a query is missing from both compared runs. Extra
run queries are ignored and listed in the report. Each run must be nonempty and
share at least one qrels query; otherwise evaluation errors instead of silently
scoring unrelated inputs. This is analogous to trec_eval's complete-query (`-c`)
evaluation policy, not its default intersection-only aggregate.

## Compare, statistics and CI reports

```sh
ragwrench eval compare --qrels examples/eval/qrels.tsv \
  --baseline examples/eval/base.trec --candidate examples/eval/cand.trec \
  --metrics ndcg@10,mrr@10 --test bootstrap --samples 10000 --seed 42 \
  --confidence 0.95 --top 5 --format json

# Write a JUnit report and fail CI on any mean drop beyond 0.01:
ragwrench eval compare --qrels examples/eval/qrels.tsv \
  --baseline examples/eval/base.trec --candidate examples/eval/cand.trec \
  --metrics ndcg@10 --max-drop ndcg@10=0.01 \
  --format junit --output gate.xml
```

`--format text|json|markdown|junit` works with both `run` and `compare`. Eval
defaults to JSON, including means, per-query scores and a saved evaluation manifest. Compare defaults to text;
its JSON includes per-query baseline/candidate/delta/outcome, missing and ignored queries,
wins/ties/losses, the worst query regressions, inference settings and violations.
JUnit has one testcase per metric and one failure per violated gate.
`schema_version` is currently 1. Numbers in JSON are full precision; text rounds
for display.

For each metric the paired observations are `candidate(query) - baseline(query)`.
All p-values are **two-sided** and comparisons require at least two qrels queries.

- `--test randomization` (default): Monte Carlo sign flips of paired differences.
  Count `abs(permuted mean) >= abs(observed mean)`. Independent seeded bootstrap
  draws provide a paired percentile confidence interval for the mean delta.
- `--test bootstrap`: resample paired differences with replacement. The CI is a
  percentile interval with linearly interpolated endpoints. For the p-value,
  center replicate means at the observed mean to approximate the null of zero.
- `--test ttest`: paired Student t-test on differences, sample standard deviation,
  n−1 degrees of freedom, and a Student t confidence interval. A stdlib numerical
  incomplete-beta CDF and inverse-tail bisection avoid SciPy. Zero variance returns
  a point CI and p=1 for zero effect or p=0 for nonzero effect, with a warning.

Monte Carlo p-values use `(extreme + 1)/(samples + 1)`, so they never report zero.
The comparison tolerance is `1e-15` for resampling tail inclusion. Samples default
to 10000, allowed range 100–1000000. Each metric uses the same seed and sorted query
order, independent of metric argument order. Reproducibility targets the same
Python version; pin the interpreter in CI if byte-identical output is required.

Wins/ties/losses use an absolute `1e-12` tolerance. A gate fails when mean delta is
below `-max_drop - 1e-12`. Thresholds are absolute score units, must name evaluated
metrics, and are repeatable. **Statistical significance never suppresses a gate
failure.** No thresholds means no gate failures. Exit codes: **0** success or pass,
**1** threshold violation, **2** invalid input/options or I/O failure.

## Save evaluations and compare later

Every `run --format json` report is a self-contained evaluation artifact with
an embedded `manifest`. Save baseline and candidate evaluations separately, then
compare them without needing the original input files:

```sh
ragwrench eval run --qrels examples/eval/qrels.tsv --run examples/eval/base.trec \
  --metrics ndcg@10,recall@100 --output baseline.eval.json
ragwrench eval run --qrels examples/eval/qrels.tsv --run examples/eval/cand.trec \
  --metrics ndcg@10,recall@100 --output candidate.eval.json
ragwrench eval compare-saved --baseline baseline.eval.json \
  --candidate candidate.eval.json --test randomization --samples 1000 --seed 42 \
  --max-drop ndcg@10=0.01 --max-drop recall@100=0.02 --format json
```

Expected: the last command exits **1**, with the same means and drop failures as
direct `compare`. Metrics and gain are inherited from the saved evaluations;
they cannot be overridden. Statistical settings and drop gates are chosen at
comparison time. Text, JSON, Markdown and JUnit reports are supported. JSON and
JUnit retain both manifests and missing/ignored query IDs.

Manifest schema 1 records the ragwrench eval version, dataset and run SHA-256 hashes,
the complete sorted judged query IDs, metrics, gain, ranking rule and query
policy. Hashes cover **canonical parsed evaluation inputs**, not file bytes:
sorted-key compact ASCII JSON of the query-to-document judgments and scores
(run scores normalized to floats). Empty judgment maps are included. Equivalent
TREC/BEIR/JSONL conversions and reordered records preserve hashes; changing
judgments, document IDs or scores changes them. LlamaIndex query/corpus text,
source paths, input ranks and run tags are outside the scoring inputs and are
not hashed. The artifact checksum binds the whole evaluation report and manifest,
excluding the checksum field itself. Artifacts contain no paths or timestamps.

`compare-saved` refuses different judged query sets and names IDs present on
only one side. It also refuses different dataset hashes (even with identical
query IDs), scoring parameters, unsupported manifest/evaluator versions, legacy
reports without manifests, and corrupt/incomplete scores. These errors exit
**2**, emit no report and leave existing output untouched, including with
`--force`. Regenerate legacy artifacts using `run --format json`.

Missing **run results** remain explicit zeros over every judged query, including
queries absent from both runs. For example, with two judged queries and a perfect
baseline, a candidate retrieving the relevant document for only one query scores
`hit@1=0.5`, lists the other query as missing and fails `--max-drop hit@1=0.49`.
Missing **saved score rows or metric values** are invalid input; comparison never
intersects available rows or drops queries to improve a mean. Saved means are
checked against the complete per-query data before pairing.

Checksums detect accidental edits, not a malicious artifact author who can
recompute them; use trusted artifacts. Comparison checks the saved bindings and
scores without re-reading or authenticating original files. Saved comparisons
require the exact same metric set and ragwrench eval version across all artifacts.

### Validate one saved evaluation

`validate-evaluation` checks one complete current schema-1 JSON eval artifact,
including its manifest and **recomputed whole-artifact SHA-256 checksum**. It
reuses `read_evaluation` / `validate_evaluation`; no second artifact, original
qrels/runs, scoring, comparisons or statistical inference are needed:

```sh
ragwrench eval validate-evaluation --input baseline.eval.json
ragwrench eval validate-evaluation --input baseline.eval.json --output receipt.json
```

Both commands exit **0**. Stdout is exactly the deterministic JSON below with a
final newline; `--output` writes those bytes atomically and leaves stdout empty:

```json
{
  "schema_version": 1,
  "command": "validate-evaluation",
  "valid": true
}
```

The existing complete saved-evaluation contract applies: supported report,
manifest and evaluator versions; exact fields; canonical judged/metric IDs;
complete per-query scores in range; exact means over all judged queries;
consistent missing/ignored IDs and scoring parameters; valid hash syntax and a
matching artifact checksum. Legacy artifacts without a manifest must be
regenerated. Strict UTF-8 JSON rejects duplicate keys (also nested), nonfinite
values and trailing content. Invalid/incomplete/corrupt artifacts and read/write
errors exit **2**, print a diagnostic to stderr and emit no receipt. There is no
exit **1**, comparison, or gate-pass claim.

Existing output requires `--force`; without `--output`, force does not change
stdout. The parent directory must exist. Symlink outputs (including dangling
links), direct/resolved-path input aliases and hardlinks to the input are
refused even with force. Input/output path loops are controlled errors. Invalid
input or a failed write preserves existing output, including with force; new
output appears only after the receipt is fully written and synced, and temporary
files are cleaned up. No filesystem snapshot or race guarantee is provided.

**Artifact consistency, not dataset/run provenance:** the full artifact checksum
can be recomputed from this one file. Original dataset/run hash contents and the
correctness of scores against those unavailable inputs cannot be verified.
Someone who edits a consistent payload and recomputes its checksum can produce
a valid artifact; this does not authenticate an author or prove ragwrench eval produced
it. The receipt contains no artifact identity, paths or timestamps. Retain it
alongside the trusted artifact if needed. The artifact stays in memory; callers
must bound resources for large or adversarial files.

### Compare a saved candidate family

`compare-many-saved` accepts one fixed saved baseline and repeatable named
`--candidate NAME=PATH` evaluations. Using the artifacts saved above:

```sh
ragwrench eval compare-many-saved --baseline baseline.eval.json \
  --candidate unchanged=baseline.eval.json --candidate proposed=candidate.eval.json \
  --test randomization --samples 1000 --seed 42 --correction holm \
  --max-drop ndcg@10=0.01 --max-drop recall@100=0.02 \
  --candidate-max-drop proposed:recall@100=0.2 --format json
```

Expected: **exit 1**; four candidate–metric hypotheses. `unchanged` passes;
`proposed` fails nDCG@10 but passes recall@100 under its 0.2 override. All saved
artifacts are validated before inference or output. Every candidate must match
the baseline's complete judged dataset (IDs and judgment hash), scoring
parameters and supported evaluator version. Candidate names, correction,
shared thresholds and overrides follow the `compare-many` rules below.
Metrics and gain come from the artifacts; `--metrics` and `--gain` are refused.
Original qrels and run files are unnecessary.

Text, JSON, Markdown, JUnit and per-query CSV reuse the existing family formats.
JSON `command` is `compare-many-saved`; `candidates.NAME` contains the paired
`compare-saved` report, including `evaluation_manifests.baseline` and `.candidate`,
missing/ignored query IDs and complete per-query outcomes. Each JUnit testcase's
JSON `system-out` includes those manifests and query metadata alongside the
family policy and effective threshold. CSV contains outcomes only.

Any invalid/incomplete artifact or incompatible member rejects the whole family
with **exit 2**, identifies the baseline or candidate name, emits no report and
preserves existing output even with `--force`. All saved inputs are protected
from output overwrite. Valid families exit **0** when gates pass or **1** when
any gate fails, and always emit the full family report. Holm/Bonferroni correct
all candidate–metric p-values; CIs remain unadjusted and significance does not
affect gates. No candidate is silently skipped or compared to another candidate.

Python API: `compare_many_saved(baseline_report, {"proposed": candidate_report},
correction="holm", thresholds={"hit@1": 0.49},
candidate_thresholds={"proposed": {"hit@1": 0.5}})` from `ragwrench.evalgate.compare`.
Inputs are the parsed eval JSON objects; all CLI compatibility rules apply.

## Multiple candidates against a fixed baseline

`compare-many` evaluates the baseline once and compares every named candidate
against it on the complete qrels query universe. It does not compare candidates
to each other or select a new baseline. Existing `compare` behavior is unchanged.

```sh
ragwrench eval compare-many --qrels examples/eval/qrels.tsv \
  --baseline examples/eval/base.trec \
  --candidate unchanged=examples/eval/base.trec --candidate proposed=examples/eval/cand.trec \
  --metrics ndcg@10,recall@100 --test randomization --samples 1000 --seed 42 \
  --correction holm --alpha 0.05 \
  --max-drop ndcg@10=0.01 --max-drop recall@100=0.02 --format json
```

Expected: **exit 1**; `unchanged` passes with three ties per metric, while
`proposed` fails both drop thresholds. The JSON `policy` records four hypotheses
(two candidates × two metrics), the correction, alpha, shared thresholds and
the gate rule. `candidates.NAME` contains the full paired report, including
`per_query.QUERY.METRIC` baseline/candidate scores, delta and `win|tie|loss`.
The top-level violations identify both candidate and metric.

The repeatable `--candidate NAME=PATH` accepts one or more runs; names must match
`[A-Za-z0-9][A-Za-z0-9_.-]*` and be unique. Paths may contain `=`. Auto-detection
allows TREC and JSONL candidates in the same command. Invalid names, duplicate
names, malformed or unrelated runs exit **2** before any report is emitted or
existing output replaced. All qrels, baseline and candidate inputs are protected
from output overwrite, including with `--force`.

**Correction is required explicitly:** `--correction none|bonferroni|holm`.
The family contains **all candidate–metric pairs**, including unchanged runs and
metrics without drop thresholds. Bonferroni uses `min(1, family_size * p)`;
Holm sorts raw p-values, multiplies by the number of remaining hypotheses and
takes cumulative maxima, capped at 1. Every metric row retains the raw two-sided
`p_value` and adds `adjusted_p_value` and `significant` (`adjusted_p_value <= alpha`).
`--alpha` defaults to 0.05 and must be strictly between 0 and 1. Significance is
two-sided, so it can describe improvements or regressions. With `none`, labels
are unadjusted and have no family-wise error control.

**Threshold policy:** repeatable `--max-drop METRIC=VALUE` sets shared metric
defaults. Optional `--candidate-max-drop NAME:METRIC=VALUE` overrides that metric
for one named candidate, whether the override is stricter or more permissive.
Precedence is **candidate override > shared metric default > no gate**. Zero is
a valid override; an override may gate a metric without a shared default. Pairs
with neither threshold remain ungated. Any candidate–metric violation fails the
whole command. Significance never changes a gate decision; no thresholds means
exit 0 for valid inputs.

For both `compare-many` and `compare-many-saved`, an explicit candidate override
`--candidate-max-drop NAME:METRIC=off` disables that pair's gate, including an
inherited shared default. Only the exact lowercase token `off` is accepted, and
only for candidate overrides; shared `--max-drop` remains numeric. The pair still
contributes complete per-query outcomes, inference and a hypothesis to the
correction family. An opt-out without a shared default is allowed and recorded.
Using the saved artifacts above:

```sh
ragwrench eval compare-many-saved --baseline baseline.eval.json \
  --candidate strict=candidate.eval.json --candidate relaxed=candidate.eval.json \
  --correction holm --test ttest --max-drop ndcg@10=0.01 \
  --candidate-max-drop relaxed:ndcg@10=off --format json
```

Expected: **exit 1**; only `strict/ndcg@10` violates. Both candidates remain in
the four-pair correction family; `relaxed` has no effective gates. To disable
every inherited gate for a candidate, opt out of each shared metric explicitly.

```sh
ragwrench eval compare-many --qrels examples/eval/qrels.tsv \
  --baseline examples/eval/base.trec \
  --candidate strict=examples/eval/cand.trec --candidate relaxed=examples/eval/cand.trec \
  --metrics ndcg@10 --test ttest --correction holm \
  --max-drop ndcg@10=0.01 --candidate-max-drop relaxed:ndcg@10=0.3 --format json
```

Expected: **exit 1**; both candidates drop by about 0.296482, but only `strict`
fails. `relaxed` uses 0.3 instead of the shared 0.01. Candidate names are
case-sensitive and must name a supplied `--candidate`; metrics must appear in
`--metrics` (or the saved metrics). Each candidate–metric override may occur
once, including numeric/off and off/off assignments. Unknown names or
metrics, duplicate overrides, malformed expressions, negative or nonfinite
values exit **2** without producing a report or replacing output, even with
`--force`. Shared defaults and an override for the same pair are allowed.

JSON `policy.thresholds` retains shared defaults; `candidate_thresholds` records
explicit overrides. `threshold_precedence` states the rule, and
`effective_thresholds.NAME` contains the resolved gates (ungated metrics are
omitted). Explicit opt-outs are retained as `null` in `candidate_thresholds`;
omitting an override still inherits the shared default. Each metric row's
`max_drop` is its effective threshold, or `null` if
ungated. JUnit includes this same complete policy and effective `max_drop` in
each testcase's JSON `system-out`. The Python API accepts
`compare_many(..., thresholds={"hit@1": 0.49},
candidate_thresholds={"relaxed": {"hit@1": 0.5}})` with the same validation
and precedence.

The Python API represents a candidate opt-out with `None`, for example
`candidate_thresholds={"relaxed": {"hit@1": None}}`. Strings such as `"off"`
are CLI syntax only; shared API `thresholds` cannot contain `None`. Policy
validation still checks every candidate/metric name and every saved artifact,
even when all of a candidate's gates are disabled.

Confidence intervals are **marginal and unadjusted**, even when p-values are
corrected. The same seed is used independently for each pair, so adding or
reordering candidates/metrics does not change existing raw inference; changing
the family may change adjusted p-values. Reports sort candidate names, query IDs
and metric names for reproducible exports on the same Python version.

Text, JSON, Markdown and JUnit are supported. JUnit has one testcase per
candidate–metric pair and includes the policy in testcase output. For a flat
per-query export, use `--format csv`:

```sh
ragwrench eval compare-many --qrels examples/eval/qrels.tsv \
  --baseline examples/eval/base.trec \
  --candidate unchanged=examples/eval/base.trec --candidate proposed=examples/eval/cand.trec \
  --metrics ndcg@10 --samples 1000 --seed 42 --correction holm \
  --max-drop ndcg@10=0.01 --format csv --output outcomes.csv
```

This also exits 1 and writes six rows with columns
`candidate_name,query_id,metric,baseline,candidate,delta,outcome`. CSV contains
per-query outcomes only; use JSON for inference, policy and gate summaries.
CSV quoting preserves punctuation in query IDs. Gate failures still produce
complete reports; invalid input produces none.

### Limit total family resampling work

Both family commands accept an optional `--max-family-resamples LIMIT`. The
limit must be a positive integer. For `bootstrap` and `randomization`, required
work units are **judged query count × samples × candidate count × metric count**.
Every pair counts, including unchanged candidates, ungated metrics and explicit
`off` overrides. `ttest` uses zero resampling work. These are normalized work
units, not a count of RNG calls or a timing estimate: randomization uses separate
bootstrap CI and sign-flip streams, each with the requested sample count.

Using the saved artifacts above (three queries, two candidates, two metrics):

```sh
ragwrench eval compare-many-saved --baseline baseline.eval.json \
  --candidate strict=candidate.eval.json --candidate relaxed=candidate.eval.json \
  --correction holm --test randomization --samples 1000 --seed 42 \
  --max-drop ndcg@10=0.01 --candidate-max-drop relaxed:ndcg@10=off \
  --max-family-resamples 12000 --format json
```

Expected: **exit 1**; only `strict/ndcg@10` violates. Exactly 12000 work units
are permitted. Changing the limit to 11999 gives **exit 2**, with
`required=12000, allowed=11999`, before any paired inference. Invalid limits and
over-budget families emit no report and preserve output even with `--force`.
All saved artifacts must still pass compatibility validation first.

When a limit is supplied, JSON `policy.estimated_family_resamples` and
`policy.max_family_resamples` record the estimate and limit; JUnit includes them
in every testcase's policy. Allowed families retain identical inference, gates,
manifests and ordering. The API keyword is `max_family_resamples=12000`; use a
positive Python `int` (booleans and floats are refused), or `None` to omit it.
Omitting the option preserves existing policy exports and per-pair behavior.
The independent limit of 50000000 query draws per resampled pair still applies;
a larger family limit cannot relax it. This option bounds resampling work only;
input files and evaluated scores are still held in memory.

### Limit family pair count

Both family commands also accept optional `--max-family-pairs LIMIT` (API:
`max_family_pairs=4`). This independently bounds **candidate count × metric
count** for every statistical method, including `ttest`. Unchanged candidates,
ungated metrics and explicit gate opt-outs all count. The limit is inclusive
and must be a positive integer; API booleans, floats and strings are refused.
Use `None` or omit the option for no pair limit and unchanged policy exports.

Using the saved evaluations above (two metrics):

```sh
ragwrench eval compare-many-saved --baseline baseline.eval.json \
  --candidate strict=candidate.eval.json --candidate relaxed=candidate.eval.json \
  --correction holm --test ttest --max-drop ndcg@10=0.01 \
  --candidate-max-drop relaxed:ndcg@10=off \
  --max-family-pairs 4 --max-family-resamples 1 --format json
```

Expected: **exit 1**; only `strict/ndcg@10` violates. The family has four pairs
and zero resampling work. A pair limit of 3 gives **exit 2**, with
`family pair limit exceeded: required=4, allowed=3`, before any paired inference.
Invalid limits and excess pair counts emit no report and preserve output even
with `--force`. Saved artifact compatibility and named candidate diagnostics
still take precedence. When both limits are exceeded, the pair limit is checked
first. Neither limit relaxes the other or the existing per-pair resampling cap.

JSON `policy.max_family_pairs` records an explicit limit alongside the existing
`policy.family_size`; each JUnit testcase includes the same policy. Allowed
families retain identical inference, gates, manifests and ordering. Pair counts
bound the number of hypotheses, not bytes of memory or time; inputs, evaluated
scores and reports are still held in memory.

Python API metric sequences must be nonempty and contain unique metric names,
just like CLI `--metrics`. For example, appending another `hit@1` to
`parse_metrics("hit@1,recall@1")` raises `InputError: duplicate metric: hit@1`
when passed to `evaluate`, `saved_evaluation`, `compare` or `compare_many`.
Validation happens before scoring or paired inference, regardless of family
limits. Duplicates are rejected rather than silently combined: each metric
must identify exactly one hypothesis per candidate, so `policy.family_size`
matches both the report rows and JUnit testcase count. Saved artifacts also
require unique metrics; an invalid saved member retains its named diagnostic.

Output defaults to stdout. `--output PATH` writes atomically to an explicitly named
file; its parent must already exist. Existing files require `--force`; input files
and symlink outputs are refused. Data IDs are never interpreted as paths, shell
commands or templates. Commands read inputs and emit a report or conversion;
family preflight can validate an intended comparison without paired inference.

### Audit a family without paired inference

Add `--preflight-only` to `compare-many` or `compare-many-saved` to validate
normal inputs, options, saved bindings and resource limits without running
`paired_test`, resampling, p-value correction or gate decisions. For example,
using the saved artifacts above:

```sh
ragwrench eval compare-many-saved --baseline baseline.eval.json \
  --candidate strict=candidate.eval.json --candidate relaxed=candidate.eval.json \
  --correction holm --test ttest --max-drop ndcg@10=0.01 \
  --candidate-max-drop relaxed:ndcg@10=off \
  --max-family-pairs 4 --max-family-resamples 1 --preflight-only
```

Expected: **exit 0**, `mode: "preflight-only"`, four candidate–metric pairs,
zero estimated resamples, and the resolved policy with only `strict/ndcg@10`
gated. This validates the family; it makes no gate-pass claim, even though a
normal comparison of these artifacts would exit 1. Direct families use the same
flag with their usual qrels/run/metric arguments and required correction.

**JSON-only contract:** omitted `--format` selects JSON in this mode; explicit
`--format json` is equivalent. Explicit text, Markdown, JUnit or CSV exits **2**.
Schema 1 contains `command` (`compare-many` or `compare-many-saved`),
`mode: "preflight-only"`, `query_count`, sorted `candidate_names` and
`metric_names`, `family_size`, `estimated_family_resamples`, `limits`, `settings`
and `policy`. `limits` always records `max_family_pairs` and
`max_family_resamples` (JSON `null` when absent). `settings` records `test`,
`seed`, `samples` and `confidence`. `policy` exactly matches the normal family
policy for the same options, including shared, explicit and effective gates and
any enabled limit annotations. There are no numerical inference results,
`passed`, `violations`, per-query outcomes or candidate comparison reports.
Summary JSON contains no paths or timestamps. Both direct and saved preflight
include `input_bindings`. Saved bindings project fully validated manifests after
all members pass compatibility checks. Direct bindings use the same canonical
artifact construction as `run`, reusing the already-evaluated baseline and
candidate payloads without scoring again or reading input files again. The
binding schema is:

```json
{
  "schema_version": 1,
  "evaluator": {"name": "retgate", "version": "0.1.0"},
  "dataset_sha256": "<shared dataset SHA-256>",
  "parameters": {
    "metrics": ["ndcg@10", "recall@100"],
    "gain": "linear",
    "tie_rule": "<validated ranking rule>",
    "query_policy": "<validated query policy>"
  },
  "baseline": {"artifact_sha256": "<baseline checksum>", "run_sha256": "<baseline run SHA-256>"},
  "candidates": {
    "relaxed": {"artifact_sha256": "<candidate checksum>", "run_sha256": "<candidate run SHA-256>"},
    "strict": {"artifact_sha256": "<candidate checksum>", "run_sha256": "<candidate run SHA-256>"}
  }
}
```

The evaluator, dataset hash and scoring parameters are shared by every artifact
under the existing exact compatibility rules. Direct bindings equal saved
preflight bindings over equivalent `run` artifacts; the summaries differ only
in `command` for the same settings, limits and policy. Candidate names are sorted;
digests are lowercase SHA-256 hex strings. Full manifests, query IDs and scores
are omitted. Different valid run inputs change the artifact/run bindings even
when their evaluation scores match; equivalent source record order, JSON object
order, metric order or whitespace does not change bindings or summary bytes.
Changed judgments change the dataset and artifact bindings even with equal
scores. Run hashes include ignored queries and normalize scores to floats,
matching `run`; paths, original ranks and tags are excluded. Saved preflight
needs no original qrels/run files. Neither mode performs inference. These
checksums detect accidental edits; they do not authenticate an artifact author.

This is an additive field in direct and saved preflight summary schema 1, with its own
binding `schema_version: 1`; consumers should accept the additional field.
Normal eval and saved/direct comparison outputs and the other summary fields
retain their existing contracts. API summaries
copy nested binding objects/lists, so editing them does not change input artifacts.

Every pair counts, including ungated and opted-out pairs. Estimated work is
`query_count * samples * family_size` for bootstrap/randomization, zero for
ttest. Preflight also enforces at least two judged queries, valid statistical
settings (including the sample range for ttest), and the independent 50000000
per-pair resampling cap. Pair/work excess uses the same required/allowed errors
as comparison. Invalid/incompatible/over-budget input exits **2**, emits no
summary and preserves existing output even with `--force`. Successful summaries
use the usual atomic output and input-overwrite protections.

Python API: pass `preflight_only=True` to `compare_many` or
`compare_many_saved` with the same family options. The flag must be a boolean;
preflight seeds must be integers, matching CLI `--seed`. The returned dict uses
the schema above; `render(summary, "json")` exports it, while other formats raise
`InputError`. `preflight_only=False` is the default and keeps normal comparison
behavior and formats. Preflight still reads/evaluates direct inputs or validates
complete saved payloads in memory; it is not a memory or timing guarantee.

### Require a prior audit before inference

Supply `--expected-preflight PATH` to either family command to require a prior
`--preflight-only` JSON audit to match the current family. For example:

```sh
ragwrench eval compare-many --qrels examples/eval/qrels.tsv \
  --baseline examples/eval/base.trec --candidate proposed=examples/eval/cand.trec \
  --metrics ndcg@10 --test ttest --correction holm --max-drop ndcg@10=0.01 \
  --preflight-only --output family.audit.json
ragwrench eval compare-many --qrels examples/eval/qrels.tsv \
  --baseline examples/eval/base.trec --candidate proposed=examples/eval/cand.trec \
  --metrics ndcg@10 --test ttest --correction holm --max-drop ndcg@10=0.01 \
  --expected-preflight family.audit.json --format json
```

Expected: audit **exit 0**, then comparison **exit 1** with the usual gate report.
Changing only the second command's `--max-drop` to `ndcg@10=0.3` instead gives
**exit 2** with an `expected preflight mismatch at ...` diagnostic and no report.
An unchanged family retains normal exit **0/1** and all normal report formats.
Omitting the option preserves all existing behavior and exports.

The guard compares the **entire** current schema-1 bound audit: all
`input_bindings`, statistical `settings`, explicit/null `limits`, counts, sorted
names, estimated work and complete `policy`, including explicit overrides,
opt-outs and effective gates. Changed valid judgments or run scores reject even
when aggregate evaluation scores are equal; ignored run queries are also bound.
The only command normalization allows `compare-many` and `compare-many-saved`
to match, so equivalent direct/saved artifacts can share an audit. Object key
order, source record order and CLI metric/candidate argument order do not matter.
Audit arrays must retain their canonical order. Numeric threshold/alpha/confidence
values compare by value (`1` and `1.0` are equal); schema versions, counts, work,
seed, samples and non-null limits require actual integers, excluding booleans.
Other JSON types remain distinct, and nonfinite values are refused.

Audits must contain **all and only** the current fields at every object level.
Missing bindings, unknown fields, unsupported versions, duplicate JSON keys,
malformed JSON and mismatched values reject before paired inference or correction.
This deliberately rejects older unbound summaries and future extensions until
supported; regenerate an audit with the installed version. No fields are silently
ignored. The existing audit schema excludes `--top`, output format and paths;
these presentation choices may differ. Checksums identify canonical parsed
content, not raw source bytes, and do not authenticate an author.

All actual family members are evaluated/validated before comparing the audit,
including late candidates. The current audit reuses those payloads and bindings:
no second scoring pass or original-file reads, and saved families still need no
original qrels/runs. Actual invalid settings, limits or members retain their
existing errors. Mismatch and parse errors emit no stdout and preserve an
existing output even with `--force`. The expected-audit path is an input;
overwriting it or a symlink alias is refused before inference. Inputs remain in
memory; this feature makes no timing, memory or filesystem race guarantee.

API: pass `expected_preflight=audit_mapping` to `compare_many` or
`compare_many_saved`. Root/nested objects may be `Mapping` instances; arrays
must be lists. The audit is not mutated. Omitted/`None` disables the guard; CLI
JSON `null` is an error. Combining the guard with `preflight_only=True` (CLI
`--preflight-only`) checks drift and returns the current audit without inference.

### Save a guarded comparison's drift diagnostic

Add `--preflight-diff-output PATH` with `--expected-preflight PATH` to either
family command. On **valid audit drift**, the comparison still exits **2**,
prints the usual error to stderr, and emits no stdout or normal report. It
atomically writes the same standalone schema-1 JSON as `diff-preflight`, including
**all** sorted JSON Pointer changes. For example, after creating
`family.audit.json` above:

```sh
ragwrench eval compare-many --qrels examples/eval/qrels.tsv \
  --baseline examples/eval/base.trec --candidate proposed=examples/eval/cand.trec \
  --metrics ndcg@10 --test ttest --correction holm --max-drop ndcg@10=0.3 \
  --expected-preflight family.audit.json --preflight-diff-output family.drift.json \
  --format json --output comparison.json
```

Expected: **exit 2**, `family.drift.json` has `matches: false` and changes at
`/policy/effective_thresholds/proposed/ndcg@10` and `/policy/thresholds/ndcg@10`;
`comparison.json` is not created. The diff uses already-built current/expected
audit mappings and independently validates both. It performs no extra scoring,
original-input reads, paired tests or p-value correction. Saved comparisons
continue to work without original qrels/runs. The option also works with
`--preflight-only`.

A matching guard retains its normal report and exit **0/1** (or preflight audit
and exit **0**); the diagnostic path is **neither created nor modified**. An
existing diagnostic is retained even with `--force`; callers should use a fresh
path or inspect the command's exit code rather than treat file existence as a
drift signal. Malformed/unbound/unsupported audits, invalid actual candidates,
options or budgets produce no diff and preserve both existing outputs. A
diagnostic write failure exits **2**, preserves existing files and emits no normal
report. Omitting the diagnostic option keeps existing CLI behavior.

The diagnostic requires the expected guard. Before inference, both diagnostic
and normal report destinations must have existing parent directories, must be
absent or regular files, and must not be symlinks or aliases of each other or any
qrels/run/saved-artifact/expected-audit input (including hard links and resolved
path aliases). Existing destinations require `--force`, which applies to both,
even when the guard would match. No permission probe or filesystem snapshot/race
guarantee is provided. A refused destination leaves both outputs untouched.

API: valid drift from either family function raises
`ragwrench.evalgate.compare.PreflightMismatch`, a subclass of `InputError`, regardless of
whether a CLI destination was requested. Its `.diff` is an owned JSON-compatible
standalone schema-1 diagnostic; changing it does not mutate the audits. Invalid
audits/families raise ordinary `InputError` with no `.diff`. API callers control
their own persistence:

```python
from ragwrench.evalgate.compare import PreflightMismatch, compare_many_saved

try:
    report = compare_many_saved(baseline, candidates, correction="holm",
                                expected_preflight=expected_audit)
except PreflightMismatch as error:
    diagnostic = error.diff  # all changes, no inference results
```

### Diff two bound audits without retrieval inputs

`diff-preflight` diagnoses drift using only two complete bound audit JSON files.
It needs no qrels, runs or saved eval artifacts and performs no scoring, paired
inference or p-value correction:

```sh
ragwrench eval compare-many --qrels examples/eval/qrels.tsv \
  --baseline examples/eval/base.trec --candidate proposed=examples/eval/cand.trec \
  --metrics ndcg@10 --test ttest --correction holm \
  --preflight-only --output expected.audit.json
ragwrench eval compare-many --qrels examples/eval/qrels.tsv \
  --baseline examples/eval/base.trec --candidate proposed=examples/eval/cand.trec \
  --metrics ndcg@10 --test ttest --correction holm --seed 43 \
  --preflight-only --output actual.audit.json
ragwrench eval diff-preflight --expected expected.audit.json --actual expected.audit.json
ragwrench eval diff-preflight --expected expected.audit.json --actual actual.audit.json
```

Expected: both audit commands exit **0**. The first diff exits **0** with
`matches: true` and `changes: []`; the second exits **1** with:

```json
{
  "schema_version": 1,
  "command": "diff-preflight",
  "matches": false,
  "changes": [
    {"path": "/settings/seed", "expected_present": true, "actual_present": true,
     "expected": 42, "actual": 43}
  ]
}
```

Diff output is JSON only, to stdout or atomic `--output PATH`; `--force` permits
replacing an existing output while protecting both audit inputs and symlink
aliases. Invalid audits, read/write errors and late structural failures exit
**2**, emit no partial diff and preserve an existing output. Duplicate JSON keys,
invalid encoding and nonfinite numbers are errors.

Each audit is validated **independently**, so two identical malformed audits
cannot match. Require the complete current schema 1 with no missing/unknown
fields at any object level, current evaluator version, lowercase SHA-256 digest
syntax, canonical sorted unique candidate/metric arrays, matching binding names
and scoring parameters, valid statistical settings, counts and estimated work,
allowed limits (including the independent per-pair cap), and consistent policy.
Shared/override metric keys must belong to the audit, overrides must name audited
candidates, and effective thresholds must follow the documented precedence.
Policy limit annotations must exactly reflect the explicit limits. Older unbound
and unknown-schema audits must be regenerated or explicitly supported first.
Full artifacts and original inputs are absent: digest contents cannot be
recomputed or authenticated, and structurally consistent edits are valid drift.

Only root `compare-many` versus `compare-many-saved` is normalized. Object key
order is irrelevant; arrays must already be canonical. Threshold, alpha and
confidence numbers compare exactly by value (`1` equals `1.0`, no tolerance);
schema/count/work/seed/sample/limit integers exclude floats and booleans. Changes
are sorted lexicographically by RFC 6901 JSON Pointer `path`. Shared objects are
compared recursively; arrays are reported as whole values. An added/removed map
key produces one entry containing its entire subtree, including candidate and
metric additions/removals. A missing side has value `null` and presence `false`,
distinct from a present `null` gate opt-out. Values in returned subtrees are
independent copies with sorted object keys. No timestamps or paths are embedded
apart from field pointers. Existing expected-audit guards and reports are unchanged.

Python API: `from ragwrench.evalgate.preflight import diff_preflight, validate_preflight`.
`diff_preflight(expected_mapping, actual_mapping)` returns the JSON-compatible
dict above; `validate_preflight(audit_mapping)` returns `None` on success. Root
and nested objects accept `Mapping`; arrays must be lists. Neither API mutates
inputs. Invalid inputs raise `InputError`; diff errors identify the expected or
actual audit. Use `render(diff, "json")` to export. Audits remain in memory;
this provides no memory, timing or filesystem snapshot guarantee.

### Validate one archived bound audit

`validate-preflight` checks one complete current schema-1 bound family audit,
using the same `validate_preflight` API described above. It needs no second
audit, original qrels/runs or saved evaluation artifacts, and performs no
scoring, paired inference or correction. For an audit created above:

```sh
ragwrench eval validate-preflight --input family.audit.json
ragwrench eval validate-preflight --input family.audit.json --output receipt.json
```

Both commands exit **0**. The first emits exactly this deterministic JSON
receipt (including a final newline); the second writes it atomically with empty
stdout:

```json
{
  "schema_version": 1,
  "command": "validate-preflight",
  "valid": true
}
```

The entire audit must pass validation: supported schema/evaluator versions,
exact required fields with no unknown fields at any depth, canonical names,
settings, counts, work/limits, resolved policy and every binding/digest format.
Older unbound audits must be regenerated. UTF-8 JSON parsing rejects duplicate
keys (including nested duplicates), nonfinite constants and trailing content.
Malformed, inconsistent or unreadable audits exit **2**, print a diagnostic to
stderr and emit no receipt. There is no exit **1** or gate-pass claim.

An existing output requires `--force`; without `--output`, `--force` has no
effect on stdout. Output parents must exist. Symlink outputs (including dangling
links) and direct, resolved-path or hardlink aliases of the input are refused
even with `--force`. Validation or file-write failure creates no receipt and
preserves existing output; temporary files are cleaned up. Existing files are
replaced only after the receipt is fully written and synced. These protections
provide no filesystem snapshot or race guarantee.

**Consistency, not authenticity:** a successful receipt validates the archived
audit's structure, derived values and lowercase SHA-256 syntax. It cannot
recompute digests or verify them against unavailable original inputs/full
artifacts, authenticate an author, or prove that an audit was actually produced
by retgate. The receipt contains no input identity, paths or timestamps; retain
it alongside the trusted audit if needed.

## Formats and conversions

Qrels auto-detection examines content, not filename. `--qrels-format` overrides it;
`--run-format` similarly selects `trec` or `jsonl` (default `auto`). Baseline and
candidate can use different formats under auto-detection.

- **TREC qrels:** exactly four whitespace-separated fields: `query iteration doc grade`.
  Iteration is ignored. Example: `q1 0 d1 2`.
- **BEIR qrels:** header `query-id<TAB>corpus-id<TAB>score`, then exactly three
  tab-separated fields. Pass an existing split such as `dataset/qrels/test.tsv`.
  Evaluation needs only the qrels; `corpus.jsonl` and `queries.jsonl` are not loaded.
- **LlamaIndex QA JSON:** `queries` and `corpus` are ID-to-text objects;
  `relevant_docs` maps every query ID to an array of corpus IDs. `mode`, if supplied,
  must be `text`. All references must resolve, repeated relevant IDs error, and
  listed documents become grade 1. Empty relevance arrays evaluate as zero but
  cannot be converted to TREC/BEIR, which would lose the query. New/unknown root
  fields are rejected to avoid silently accepting another dataset schema.
- **TREC run:** exactly six fields: `query Q0 doc positive-rank finite-score tag`.
  Score is authoritative. Example: `q1 Q0 d1 1 12.5 bm25`.
- **JSONL run:** one object per line, exactly
  `{"query_id":"q1","doc_id":"d1","score":12.5}`. IDs must be strings, score a
  number or numeric string; bools and nonfinite values error. Nested framework
  result dumps need adaptation to this documented flat interchange schema.

Blank lines are skipped; comments and headerless BEIR TSV are not accepted. IDs
must be nonempty and free of whitespace, control characters and Unicode
surrogates. Duplicate query/document pairs, duplicate JSON object keys, fractional
or negative grades, malformed ranks, and NaN/Infinity fail with `path:line` errors.
JSON syntax has exact line numbers; JSON dataset semantic errors point to root
line 1 and describe the invalid mapping. All inputs are validated before output.

```sh
ragwrench eval convert --input examples/eval/qrels.tsv \
  --from beir-qrels --to trec-qrels --output judgments.trec
ragwrench eval convert --input judgments.trec \
  --from trec-qrels --to beir-qrels
ragwrench eval convert --input tests/evalgate/fixtures/llamaindex.json \
  --from llamaindex --to trec-qrels
ragwrench eval convert --input tests/evalgate/fixtures/results.jsonl \
  --from jsonl-run --to trec-run
ragwrench eval convert --input examples/eval/base.trec \
  --from trec-run --to jsonl-run
```

Writers sort queries and documents deterministically; run writers regenerate ranks
using the ranking rule and preserve scores. TREC/BEIR qrels round trips preserve
grades. Format conversion cannot infer a run from qrels, generate missing corpus
text, or preserve run tags and original rank fields. Unsupported cross-kind
conversions explicitly error.

## Offline validation and reproducible experiments

```sh
pip install -e '.[test]'   # from a clone of the repository
python -m pytest tests/evalgate -q
ragwrench eval run --qrels examples/eval/qrels.tsv --run examples/eval/base.trec --metrics ndcg@10,mrr@10,recall@100
ragwrench eval compare --qrels examples/eval/qrels.tsv --baseline examples/eval/base.trec --candidate examples/eval/cand.trec --metrics ndcg@10 --seed 42
python examples/eval/tie_invariance_demo.py
python examples/eval/paired_experiment.py
```

All fixtures are small, synthetic and redistributable under this repository's MIT
license. Optional tests cross-check pytrec_eval and ranx when they are already
installed; the test suite never downloads them.

`tie_invariance_demo.py` compares ragwrench eval with an explicitly modeled naive sort
that preserves input order for ties: 100 seeded shuffles yield **1** ragwrench eval metric
vector and **6** distinct naive nDCG@2 values. This measures score stability, not
speed, and does not assert a particular ranx version is broken.
`paired_experiment.py` moves the relevant result from rank 2 to rank 1 for 40
synthetic queries: measured nDCG@10 0.630930 → 1.000000, MRR@10 0.500000 → 1.000000,
40 wins and seeded randomization p=0.00019996 (5000 draws). These are controlled
correctness experiments, not claims of real-world retrieval improvements.

## Scope and upstream relationship

This is an independent implementation, not a wrapper or fork of ranx, trec_eval, BEIR or
LlamaIndex. No upstream source code is copied or vendored.

Limitations: no retrieval/model execution, LLM judgments, dataset downloading,
custom relevance thresholds, bpref, negative/unjudged-sentinel grades or trec_eval's
full option set. Exp-gain nDCG is explicitly different from trec_eval's default.
Statistical tests assume independent query units; bootstrap intervals can be poor
for small or degenerate samples, and t-tests assume approximately normal mean
differences. Single-candidate `compare` does not adjust multiple metrics; use
`compare-many` with Holm or Bonferroni for an explicit family correction. No exact
permutation enumeration, BCa intervals, stratification or power guarantees.

Files are held in memory; use modest offline evaluation sets. Bootstrap and
randomization reject `query_count * samples > 50000000` rather than perform an
unexpectedly large job (reduce samples or select ttest). Very large or adversarial
files require caller resource limits. There is no parallel worker pool or daemon.
