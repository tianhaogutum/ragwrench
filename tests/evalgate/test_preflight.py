"""Preflight validates a complete family without inference or gate decisions."""
import copy
import json

import pytest

from ragwrench.evalgate.cli import main
from ragwrench.evalgate.compare import compare_many, compare_many_saved
from ragwrench.evalgate.formats import InputError
from ragwrench.evalgate.metrics import parse_metrics
from ragwrench.evalgate.report import render
from ragwrench.evalgate.saved import saved_evaluation


@pytest.fixture
def family(tmp_path):
    qrels = {"q1": {"d": 1}, "q2": {"d": 1}}
    base = {q: {"d": 1.0} for q in qrels}
    cand = {"q2": {"d": 1.0}}
    metrics = parse_metrics("recall@1,hit@1")
    artifacts = [saved_evaluation(qrels, run, metrics) for run in (base, cand)]
    paths = [tmp_path / name for name in ("qrels", "base", "cand", "base.json", "cand.json")]
    for path, text in zip(paths, ["q1 0 d 1\nq2 0 d 1\n", "q1 Q0 d 1 1 b\nq2 Q0 d 1 1 b\n",
                                  "q2 Q0 d 1 1 c\n", *(json.dumps(a) for a in artifacts)]):
        path.write_text(text)
    return qrels, base, cand, metrics, artifacts, paths


def api(family, saved, **extra):
    qrels, base, cand, metrics, artifacts, _ = family
    options = dict(correction="holm", method="bootstrap", samples=100,
                   thresholds={"hit@1": .49}, candidate_thresholds={"relaxed": {"hit@1": None}})
    options.update(extra)
    if saved:
        return compare_many_saved(artifacts[0], {"strict": artifacts[1], "relaxed": artifacts[1]}, **options)
    return compare_many(qrels, base, {"strict": cand, "relaxed": cand}, metrics, **options)


def argv(family, saved, *extra):
    qrels, base, cand, bs, cs = family[-1]
    args = ["compare-many-saved" if saved else "compare-many", "--baseline", str(bs if saved else base),
            "--candidate", f"strict={cs if saved else cand}", "--candidate", f"relaxed={cs if saved else cand}",
            "--test", "bootstrap", "--samples", "100", "--correction", "holm", "--max-drop", "hit@1=.49",
            "--candidate-max-drop", "relaxed:hit@1=off"]
    if not saved:
        args += ["--qrels", str(qrels), "--metrics", "recall@1,hit@1"]
    return args + list(map(str, extra))


def forbid_inference(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("preflight must never invoke paired inference")
    monkeypatch.setattr("ragwrench.evalgate.compare.paired_test", forbidden)
    monkeypatch.setattr("ragwrench.evalgate.statistics.paired_test", forbidden)
    monkeypatch.setattr("ragwrench.evalgate.compare.adjust_p_values", forbidden)


def keys_recursive(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from keys_recursive(child)
    elif isinstance(value, list):
        for child in value:
            yield from keys_recursive(child)


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("method", ["bootstrap", "randomization", "ttest"])
@pytest.mark.parametrize("limits", [False, True])
def test_preflight_summary_without_inference(family, saved, method, limits, monkeypatch):
    options = dict(method=method)
    work = 0 if method == "ttest" else 800
    if limits:
        options.update(max_family_pairs=4, max_family_resamples=max(1, work))
    actual = api(family, saved, **options)
    assert actual["passed"] is False
    before = copy.deepcopy(family[:-1])
    forbid_inference(monkeypatch)
    summary = api(family, saved, preflight_only=True, **options)
    assert family[:-1] == before
    assert set(summary) == {"schema_version", "command", "mode", "query_count", "candidate_names", "metric_names",
                            "family_size", "estimated_family_resamples", "limits", "settings", "policy", "input_bindings"}
    assert summary["schema_version"] == 1
    assert summary["command"] == ("compare-many-saved" if saved else "compare-many")
    assert summary["mode"] == "preflight-only"
    assert summary["query_count"] == 2
    assert summary["candidate_names"] == ["relaxed", "strict"]
    assert summary["metric_names"] == ["hit@1", "recall@1"]
    assert summary["family_size"] == 4
    assert summary["estimated_family_resamples"] == work
    assert summary["settings"] == {"test": method, "seed": 42, "samples": 100, "confidence": .95}
    assert summary["limits"] == {"max_family_pairs": 4 if limits else None,
                                 "max_family_resamples": max(1, work) if limits else None}
    assert summary["policy"] == actual["policy"]
    assert summary["policy"]["effective_thresholds"] == {"relaxed": {}, "strict": {"hit@1": .49}}
    assert not set(keys_recursive({k: v for k, v in summary.items() if k != "input_bindings"})) & {
                                               "passed", "violations", "candidates", "per_query", "means", "delta",
                                               "p_value", "adjusted_p_value", "ci", "significant", "wins", "losses"}
    assert json.loads(render(summary, "json")) == summary


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("options,error", [
    ({"max_family_pairs": 3}, "family pair limit exceeded: required=4, allowed=3"),
    ({"max_family_resamples": 799}, "family resampling budget exceeded: required=800, allowed=799"),
    ({"max_family_pairs": True}, "positive integer"),
    ({"max_family_resamples": 0}, "positive integer"),
    ({"method": "unknown"}, "unknown statistical test"),
    ({"samples": 99}, "samples must be an integer"),
    ({"samples": 1_000_001}, "samples must be an integer"),
    ({"samples": 100.5}, "samples must be an integer"),
    ({"confidence": 0}, "confidence must be finite"),
    ({"confidence": float("nan")}, "confidence must be finite"),
    ({"seed": "42"}, "preflight seed must be an integer"),
    ({"seed": True}, "preflight seed must be an integer"),
    ({"samples": 99, "method": "ttest"}, "samples must be an integer"),
    ({"confidence": "bad"}, "confidence must be finite"),
    ({"top": -1}, "top must be nonnegative"),
    ({"correction": "unknown"}, "correction must be"),
    ({"alpha": 1}, "alpha must be finite"),
    ({"thresholds": {"hit@1": None}}, "thresholds must be finite"),
    ({"candidate_thresholds": {"missing": {"hit@1": None}}}, "unknown max-drop candidate"),
    ({"candidate_thresholds": {"relaxed": {"hit@2": None}}}, "threshold metric must be evaluated"),
])
def test_invalid_preflight_never_infers(family, saved, options, error, monkeypatch):
    forbid_inference(monkeypatch)
    with pytest.raises(InputError, match=error):
        api(family, saved, preflight_only=True, **options)


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("count,method,error", [
    (1, "ttest", "at least two qrels queries"),
    (51, "bootstrap", "resampling budget exceeds 50000000"),
    (51, "randomization", "resampling budget exceeds 50000000"),
])
def test_preflight_enforces_implicit_inference_limits(saved, count, method, error, monkeypatch):
    qrels = {f"q{i}": {"d": 1} for i in range(count)}
    base = {q: {"d": 1.0} for q in qrels}
    metrics = parse_metrics("hit@1")
    artifact = saved_evaluation(qrels, base, metrics)
    forbid_inference(monkeypatch)
    options = dict(correction="none", preflight_only=True, method=method, samples=1_000_000)
    with pytest.raises(InputError, match=error):
        if saved:
            compare_many_saved(artifact, {"same": artifact}, **options)
        else:
            compare_many(qrels, base, {"same": base}, metrics, **options)


@pytest.mark.parametrize("kind,error", [
    ("queries", "judged query sets differ"), ("judgments", "dataset hashes differ"),
    ("metrics", "evaluation parameters differ"), ("gain", "evaluation parameters differ"),
    ("checksum", "checksum mismatch"), ("row", "per-query data"),
    ("version", "unsupported evaluator/version"), ("legacy", "manifest required"),
])
def test_saved_preflight_validates_all_members_before_limits(family, kind, error, monkeypatch):
    qrels, base, _, metrics, artifacts, paths = family
    bad = copy.deepcopy(artifacts[1])
    if kind == "queries":
        bad = saved_evaluation({"other": {"d": 1}, "q2": {"d": 1}}, {"q2": {"d": 1}}, metrics)
    elif kind == "judgments":
        bad = saved_evaluation({q: {"d": 2} for q in qrels}, base, metrics)
    elif kind == "metrics":
        bad = saved_evaluation(qrels, base, parse_metrics("hit@1"))
    elif kind == "gain":
        bad = saved_evaluation(qrels, base, metrics, "exponential")
    elif kind == "checksum":
        bad["manifest"]["run"]["sha256"] = "0" * 64
    elif kind == "row":
        del bad["per_query"]["q1"]
    elif kind == "version":
        bad["manifest"]["evaluator"]["version"] = "unknown"
    else:
        del bad["manifest"]
    artifacts[1] = bad
    paths[-1].write_text(json.dumps(bad))
    forbid_inference(monkeypatch)
    with pytest.raises(InputError, match=f"candidate 'relaxed': .*{error}"):
        api(family, True, preflight_only=True, max_family_pairs=1)


@pytest.mark.parametrize("saved", [False, True])
def test_preflight_cli_default_json_exit_zero_even_for_gate_failure(family, saved, capsys, monkeypatch):
    forbid_inference(monkeypatch)
    args = argv(family, saved, "--preflight-only", "--max-family-pairs", 4, "--max-family-resamples", 800)
    assert main(args) == 0
    first = capsys.readouterr()
    assert first.err == ""
    summary = json.loads(first.out)
    assert summary["mode"] == "preflight-only" and "passed" not in summary
    assert main(args + ["--format", "json"]) == 0
    assert capsys.readouterr().out == first.out
    out = family[-1][0].parent / "summary.json"
    assert main(args + ["--output", str(out)]) == 0
    assert out.read_text() == first.out


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("extra,error", [
    (["--max-family-pairs", "3"], "required=4, allowed=3"),
    (["--max-family-resamples", "799"], "required=800, allowed=799"),
    (["--confidence", "nan"], "confidence must be finite"),
    (["--samples", "99"], "samples must be an integer"),
    (["--candidate-max-drop", "relaxed:hit@1=0"], "duplicate candidate max-drop"),
    *[(["--format", fmt], "preflight-only supports only JSON") for fmt in ("text", "markdown", "junit", "csv")],
])
def test_preflight_cli_errors_preserve_output(family, saved, extra, error, capsys, monkeypatch):
    forbid_inference(monkeypatch)
    output = family[-1][0].parent / "protected"
    output.write_text("sentinel\n")
    assert main(argv(family, saved, "--preflight-only", "--output", output, "--force", *extra)) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and error in captured.err
    assert output.read_text() == "sentinel\n"


@pytest.mark.parametrize("fmt", ["text", "markdown", "junit", "csv"])
def test_summary_render_rejects_non_json(family, fmt):
    summary = api(family, False, preflight_only=True)
    with pytest.raises(InputError, match="preflight-only supports only JSON"):
        render(summary, fmt)


@pytest.mark.parametrize("saved", [False, True])
def test_preflight_deterministic_order_and_no_flag_compatibility(family, saved, capsys):
    actual = api(family, saved)
    assert api(family, saved, preflight_only=False) == actual
    assert main(argv(family, saved)) == 1
    assert capsys.readouterr().out.startswith("FAIL:")
    first = render(api(family, saved, preflight_only=True), "json")
    qrels, base, cand, metrics, artifacts, _ = family
    metrics.reverse()
    for a in artifacts:
        a["per_query"] = dict(reversed(list(a["per_query"].items())))
    assert render(api(family, saved, preflight_only=True), "json") == first
    if not saved:
        summary = compare_many(qrels, base, {"relaxed": cand, "strict": cand}, metrics, correction="holm",
                               method="bootstrap", samples=100, thresholds={"hit@1": .49},
                               candidate_thresholds={"relaxed": {"hit@1": None}}, preflight_only=True)
        assert render(summary, "json") == first


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("limit", ["max_family_pairs", "max_family_resamples"])
@pytest.mark.parametrize("value", [False, -1, 1.5, float("inf"), "4"])
def test_preflight_limit_types(family, saved, limit, value, monkeypatch):
    forbid_inference(monkeypatch)
    with pytest.raises(InputError, match="positive integer"):
        api(family, saved, preflight_only=True, **{limit: value})


@pytest.mark.parametrize("saved", [False, True])
def test_ttest_preflight_zero_work_still_counts_opt_outs(family, saved, monkeypatch):
    forbid_inference(monkeypatch)
    summary = api(family, saved, method="ttest", samples=1_000_000, preflight_only=True,
                  candidate_thresholds={"relaxed": {"hit@1": None}, "strict": {"hit@1": None}},
                  max_family_pairs=4, max_family_resamples=1)
    assert summary["estimated_family_resamples"] == 0
    assert summary["policy"]["effective_thresholds"] == {"relaxed": {}, "strict": {}}
    with pytest.raises(InputError, match="required=4, allowed=3"):
        api(family, saved, method="ttest", preflight_only=True, max_family_pairs=3, max_family_resamples=1)


def test_saved_preflight_named_baseline_and_late_candidate_errors(family, capsys, monkeypatch):
    artifacts, paths = family[-2:]
    bad = copy.deepcopy(artifacts[0])
    bad["per_query"].pop("q1")
    forbid_inference(monkeypatch)
    with pytest.raises(InputError, match="baseline: .*per-query"):
        compare_many_saved(bad, {"valid": artifacts[1]}, correction="none", preflight_only=True)
    with pytest.raises(InputError, match="candidate 'z_bad': .*per-query"):
        compare_many_saved(artifacts[0], {"a_valid": artifacts[0], "z_bad": bad},
                           correction="none", preflight_only=True, max_family_pairs=1)
    bad_path = paths[0].parent / "bad.json"
    bad_path.write_text(json.dumps(bad))
    output = paths[0].parent / "protected"
    for extra, error in [(["--candidate", f"z_bad={bad_path}"], "candidate 'z_bad'"),
                         (["--baseline", bad_path], "baseline:")]:
        output.write_text("sentinel")
        assert main(argv(family, True, "--preflight-only", "--output", output, "--force", *extra)) == 2
        captured = capsys.readouterr()
        assert error in captured.err and captured.out == ""
        assert output.read_text() == "sentinel"


@pytest.mark.parametrize("saved", [False, True])
def test_preflight_cannot_overwrite_inputs(family, saved, capsys, monkeypatch):
    forbid_inference(monkeypatch)
    for path in family[-1][3:] if saved else family[-1][:3]:
        before = path.read_bytes()
        assert main(argv(family, saved, "--preflight-only", "--output", path, "--force")) == 2
        captured = capsys.readouterr()
        assert "overwrite an input file" in captured.err and captured.out == ""
        assert path.read_bytes() == before


@pytest.mark.parametrize("saved", [False, True])
def test_preflight_mode_requires_boolean(family, saved, monkeypatch):
    forbid_inference(monkeypatch)
    with pytest.raises(InputError, match="preflight_only must be a boolean"):
        api(family, saved, preflight_only="false")


def expected_bindings(baseline, candidates):
    manifest = baseline["manifest"]
    return {"schema_version": 1, "evaluator": copy.deepcopy(manifest["evaluator"]),
            "dataset_sha256": manifest["dataset"]["sha256"],
            "parameters": copy.deepcopy(manifest["parameters"]),
            "baseline": {"artifact_sha256": manifest["artifact_sha256"],
                         "run_sha256": manifest["run"]["sha256"]},
            "candidates": {name: {"artifact_sha256": artifact["manifest"]["artifact_sha256"],
                                  "run_sha256": artifact["manifest"]["run"]["sha256"]}
                           for name, artifact in sorted(candidates.items())}}


@pytest.mark.parametrize("method", ["bootstrap", "randomization", "ttest"])
def test_saved_preflight_input_bindings(family, method, monkeypatch):
    before = copy.deepcopy(family[-2])
    baseline, candidate = family[-2]
    forbid_inference(monkeypatch)
    summary = api(family, True, method=method, preflight_only=True)
    bindings = summary["input_bindings"]
    assert bindings == expected_bindings(baseline, {"strict": candidate, "relaxed": candidate})
    assert list(bindings["candidates"]) == ["relaxed", "strict"]
    assert family[-2] == before
    bindings["parameters"]["metrics"].append("hit@2")
    bindings["evaluator"]["version"] = "changed"
    bindings["baseline"]["artifact_sha256"] = "changed"
    bindings["candidates"]["strict"]["run_sha256"] = "changed"
    assert family[-2] == before
    assert api(family, True, method=method, preflight_only=True)["input_bindings"] == expected_bindings(
        baseline, {"strict": candidate, "relaxed": candidate})


def test_saved_preflight_binding_identity_and_order(family, monkeypatch):
    qrels, base, cand, metrics, artifacts, _ = family
    forbid_inference(monkeypatch)
    options = dict(correction="none", preflight_only=True, method="ttest")
    summary = compare_many_saved(artifacts[0], {"z": artifacts[1], "a": artifacts[0]}, **options)
    # Changed run inputs with exactly the same scores must remain distinguishable.
    changed = saved_evaluation(qrels, {q: {d: s + 10 for d, s in docs.items()} for q, docs in base.items()}, metrics)
    assert changed["means"] == artifacts[0]["means"]
    replacement = compare_many_saved(changed, {"a": artifacts[0], "z": artifacts[1]}, **options)
    assert replacement["input_bindings"]["baseline"] != summary["input_bindings"]["baseline"]
    replacement = compare_many_saved(artifacts[0], {"a": changed, "z": artifacts[1]}, **options)
    assert replacement["input_bindings"]["candidates"]["a"] != summary["input_bindings"]["candidates"]["a"]
    assert {k: v for k, v in replacement.items() if k != "input_bindings"} == {
        k: v for k, v in summary.items() if k != "input_bindings"}

    def reordered(value):
        if isinstance(value, dict):
            return {k: reordered(v) for k, v in reversed(list(value.items()))}
        if isinstance(value, list):
            return [reordered(v) for v in value]
        return value

    equivalent = saved_evaluation(dict(reversed(list(qrels.items()))), dict(reversed(list(base.items()))),
                                  list(reversed(metrics)))
    assert render(compare_many_saved(reordered(equivalent),
                                     {"a": reordered(artifacts[0]), "z": reordered(artifacts[1])}, **options),
                  "json") == render(summary, "json")


@pytest.mark.parametrize("position", ["baseline", "late_candidate"])
@pytest.mark.parametrize("kind,error", [("checksum", "checksum mismatch"), ("dataset", "dataset hashes differ"),
                                       ("parameters", "evaluation parameters differ"),
                                       ("queries", "judged query sets differ"),
                                       ("version", "unsupported evaluator/version")])
def test_saved_binding_rejection_preserves_forced_output(family, position, kind, error, capsys, monkeypatch):
    qrels, base, _, metrics, artifacts, paths = family
    bad = copy.deepcopy(artifacts[0])
    if kind == "checksum":
        bad["manifest"]["artifact_sha256"] = "0" * 64
    elif kind == "dataset":
        bad = saved_evaluation({q: {"d": 2} for q in qrels}, base, metrics)
    elif kind == "parameters":
        bad = saved_evaluation(qrels, base, metrics, gain="exponential")
    elif kind == "queries":
        bad = saved_evaluation({"q1": {"d": 1}, "other": {"d": 1}}, {"q1": {"d": 1}}, metrics)
    else:
        bad["manifest"]["evaluator"]["version"] = "unknown"
    baseline = bad if position == "baseline" else artifacts[0]
    candidates = {"a_valid": artifacts[1], "z_bad": bad if position == "late_candidate" else artifacts[1]}
    context = "baseline:" if position == "baseline" and kind in ("checksum", "version") else (
        "candidate 'a_valid':" if position == "baseline" else "candidate 'z_bad':")
    forbid_inference(monkeypatch)
    with pytest.raises(InputError, match=f"{context} .*{error}"):
        compare_many_saved(baseline, candidates, correction="holm", preflight_only=True, max_family_pairs=1)
    bad_path = paths[0].parent / "bad.json"
    bad_path.write_text(json.dumps(bad))
    output = paths[0].parent / "protected.json"
    output.write_bytes(b"sentinel\n")
    args = ["compare-many-saved", "--baseline", str(bad_path if position == "baseline" else paths[3]),
            "--candidate", f"a_valid={paths[4]}", "--candidate",
            f"z_bad={bad_path if position == 'late_candidate' else paths[4]}", "--correction", "holm",
            "--preflight-only", "--max-family-pairs", "1", "--force", "--output", str(output)]
    assert main(args) == 2
    captured = capsys.readouterr()
    assert context in captured.err and error in captured.err and captured.out == ""
    assert output.read_bytes() == b"sentinel\n"


def test_saved_binding_cli_standalone_deterministic(family, capsys, monkeypatch):
    forbid_inference(monkeypatch)
    for path in family[-1][:3]:
        path.unlink()
    assert main(argv(family, True, "--preflight-only")) == 0
    first = capsys.readouterr()
    assert first.err == ""
    assert json.loads(first.out)["input_bindings"] == expected_bindings(
        family[-2][0], {"strict": family[-2][1], "relaxed": family[-2][1]})
    # Reformat/reorder all artifact objects, and reverse named CLI candidates.
    for path in family[-1][3:]:
        artifact = json.loads(path.read_text())
        path.write_text(json.dumps(dict(reversed(list(artifact.items()))), indent=4))
    args = argv(family, True, "--preflight-only", "--format", "json")
    args[3:7] = args[5:7] + args[3:5]
    assert main(args) == 0
    assert capsys.readouterr().out == first.out


@pytest.mark.parametrize("method", ["bootstrap", "randomization", "ttest"])
@pytest.mark.parametrize("gain", ["linear", "exponential"])
@pytest.mark.parametrize("limited", [False, True])
def test_direct_preflight_input_bindings_match_saved(family, method, gain, limited, monkeypatch):
    qrels, base, cand, metrics, _, _ = family
    before = copy.deepcopy((qrels, base, cand, metrics))
    baseline, candidate = [saved_evaluation(qrels, run, metrics, gain) for run in (base, cand)]
    options = dict(method=method, gain=gain)
    if limited:
        options.update(max_family_pairs=4, max_family_resamples=800)
    forbid_inference(monkeypatch)
    summary = api(family, False, preflight_only=True, **options)
    assert summary["input_bindings"] == expected_bindings(baseline, {"strict": candidate, "relaxed": candidate})
    saved = compare_many_saved(baseline, {"strict": candidate, "relaxed": candidate}, correction="holm",
                               method=method, samples=100, thresholds={"hit@1": .49},
                               candidate_thresholds={"relaxed": {"hit@1": None}}, preflight_only=True,
                               **({"max_family_pairs": 4, "max_family_resamples": 800} if limited else {}))
    assert render({**summary, "command": "compare-many-saved"}, "json") == render(saved, "json")
    assert (qrels, base, cand, metrics) == before
    summary["input_bindings"]["parameters"]["metrics"].append("hit@2")
    summary["input_bindings"]["evaluator"]["version"] = "changed"
    summary["input_bindings"]["baseline"]["run_sha256"] = "changed"
    summary["input_bindings"]["candidates"]["strict"]["artifact_sha256"] = "changed"
    assert api(family, False, preflight_only=True, **options)["input_bindings"] == saved["input_bindings"]
    assert (qrels, base, cand, metrics) == before


def test_direct_preflight_binding_order_and_single_scoring(monkeypatch):
    import ragwrench.evalgate.compare as comparison
    from ragwrench.evalgate.metrics import evaluate
    qrels = {"q2": {"z": 2, "a": 1}, "q1": {"a": 1, "z": 0}, "missing": {"a": 1}}
    base = {"q1": {"a": 4, "z": 4.0}, "q2": {"z": 4, "a": 3}, "ignored": {"z": 1}}
    cand = {"q2": {"a": 4.0, "z": 4}}
    candidates = {"z": cand, "a": base}
    metrics = parse_metrics("ndcg@2,hit@1,recall@2")
    baseline = saved_evaluation(qrels, base, metrics)
    artifacts = {name: saved_evaluation(qrels, run, metrics) for name, run in candidates.items()}
    forbid_inference(monkeypatch)
    expected = compare_many_saved(baseline, artifacts, correction="holm", method="ttest", preflight_only=True)
    def forbidden(*args, **kwargs):
        pytest.fail("direct preflight must reuse existing scores without reading inputs")
    monkeypatch.setattr("ragwrench.evalgate.saved.evaluate", forbidden)
    monkeypatch.setattr("ragwrench.evalgate.formats.read_qrels", forbidden)
    monkeypatch.setattr("ragwrench.evalgate.formats.read_run", forbidden)
    calls = []
    def counted(qrels, run, metrics, gain):
        calls.append(run)
        return evaluate(qrels, run, metrics, gain)
    monkeypatch.setattr(comparison, "evaluate", counted)
    def audit(qrels, base, candidates, metrics):
        calls.clear()
        summary = compare_many(qrels, base, candidates, metrics, correction="holm", method="ttest", preflight_only=True)
        assert len(calls) == 3
        return summary
    first = audit(qrels, base, candidates, metrics)
    assert render({**first, "command": "compare-many-saved"}, "json") == render(expected, "json")
    def reverse(value):
        return {k: reverse(v) if isinstance(v, dict) else v for k, v in reversed(list(value.items()))}
    reordered = audit(reverse(qrels), reverse(base), reverse(candidates), list(reversed(metrics)))
    assert render(reordered, "json") == render(first, "json")


@pytest.mark.parametrize("changed", ["baseline", "candidate", "ignored_run", "judgments"])
def test_direct_preflight_binding_identity_equal_scores(family, changed, monkeypatch):
    qrels, base, cand, metrics, artifacts, _ = family
    forbid_inference(monkeypatch)
    original = api(family, False, preflight_only=True)
    qrels, base, cand = copy.deepcopy((qrels, base, cand))
    if changed == "judgments":
        qrels["q1"]["d"] = 2
    elif changed == "ignored_run":
        cand["ignored"] = {"d": 1.0}
    else:
        run = base if changed == "baseline" else cand
        for docs in run.values():
            for doc in docs:
                docs[doc] += 10
    baseline, candidate = [saved_evaluation(qrels, run, metrics) for run in (base, cand)]
    assert [baseline["means"], candidate["means"]] == [a["means"] for a in artifacts]
    replacement = api((qrels, base, cand, metrics, artifacts, []), False, preflight_only=True)
    assert replacement["input_bindings"] == expected_bindings(baseline, {"strict": candidate, "relaxed": candidate})
    assert replacement["input_bindings"] != original["input_bindings"]
    if changed == "judgments":
        assert replacement["input_bindings"]["dataset_sha256"] != original["input_bindings"]["dataset_sha256"]
    else:
        target = "baseline" if changed == "baseline" else "candidates"
        assert replacement["input_bindings"][target] != original["input_bindings"][target]
        assert replacement["input_bindings"]["dataset_sha256"] == original["input_bindings"]["dataset_sha256"]
    assert {k: v for k, v in replacement.items() if k != "input_bindings"} == {
        k: v for k, v in original.items() if k != "input_bindings"}


@pytest.mark.parametrize("bad", [{}, {"unrelated": {"d": 1.0}}])
def test_direct_preflight_late_unrelated_candidate(family, bad, monkeypatch):
    qrels, base, _, metrics, _, _ = family
    forbid_inference(monkeypatch)
    with pytest.raises(InputError, match="nonempty and share at least one query ID"):
        compare_many(qrels, base, {"a_valid": base, "z_bad": bad}, metrics, correction="none",
                     preflight_only=True, max_family_pairs=1)


@pytest.mark.parametrize("text,error", [("broken\n", "expected 6 TREC run fields"),
                                         ("q1 Q0 d 1 NaN tag\n", "score must be a finite number"),
                                         ("q1 Q0 d 1 1 tag\nq1 Q0 d 2 1 tag\n", "duplicate"),
                                         ("unrelated Q0 d 1 1 tag\n", "share at least one query ID")])
def test_direct_binding_cli_late_candidate_preserves_forced_output(family, text, error, capsys, monkeypatch):
    forbid_inference(monkeypatch)
    bad = family[-1][0].parent / "bad.run"
    bad.write_text(text)
    output = bad.with_name("protected.json")
    output.write_bytes(b"sentinel\n")
    assert main(argv(family, False, "--candidate", f"z_bad={bad}", "--preflight-only",
                     "--max-family-pairs", 1, "--output", output, "--force")) == 2
    captured = capsys.readouterr()
    assert error in captured.err and captured.out == ""
    assert output.read_bytes() == b"sentinel\n"


def test_direct_binding_cli_reads_once_and_matches_saved(family, capsys, monkeypatch):
    import ragwrench.evalgate.cli as cli
    from collections import Counter
    from pathlib import Path
    forbid_inference(monkeypatch)
    calls = []
    for name in ("read_qrels", "read_run"):
        original = getattr(cli, name)
        def counted(path, *args, _original=original, **kwargs):
            calls.append(Path(path))
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(cli, name, counted)
    assert main(argv(family, False, "--preflight-only")) == 0
    first = capsys.readouterr()
    # Same candidate path occurs twice in the existing interface; no extra reads for bindings.
    qrels, base, cand = family[-1][:3]
    assert Counter(calls) == Counter({qrels: 1, base: 1, cand: 2})
    assert main(argv(family, True, "--preflight-only")) == 0
    saved = capsys.readouterr()
    direct = json.loads(first.out)
    assert "input_bindings" in direct
    assert first.err == saved.err == ""
    assert render({**direct, "command": "compare-many-saved"}, "json") == saved.out
    for path in family[-1][:3]:
        path.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n")
    args = argv(family, False, "--preflight-only", "--metrics", "hit@1,recall@1")
    args[3:7] = args[5:7] + args[3:5]
    assert main(args) == 0
    assert capsys.readouterr().out == first.out
