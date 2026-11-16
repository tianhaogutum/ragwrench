import json
import subprocess
import sys

import pytest

from ragwrench.highlight.cli import main
from ragwrench.highlight.core import CiteError, Region, read_regions, regions_jsonl, write_text


def test_cli_pipeline_docling(examples, tmp_path, capsys):
    regions, merged, output = tmp_path / "regions.jsonl", tmp_path / "merged.jsonl", tmp_path / "out.pdf"
    assert main(["normalize", "--format", "docling", str(examples / "sample.docling.json"),
                 "--pdf", str(examples / "sample.pdf"), "-o", str(regions)]) == 0
    assert main(["consolidate", str(regions), "--chunks", str(examples / "chunks.docling.json"),
                 "--per-page", "-o", str(merged)]) == 0
    assert main(["annotate", str(examples / "sample.pdf"), str(merged), "-o", str(output)]) == 0
    assert main(["verify", str(output), str(merged), "--text-check", "required"]) == 0
    assert "OK 2/2" in capsys.readouterr().out
    citations = json.loads((tmp_path / "citations.json").read_text())
    assert citations["chunks"]["quarterly-summary"]["links"] == ["out.pdf#page=1", "out.pdf#page=2"]
    assert main(["manifest", str(output), str(merged), "-o", str(tmp_path / "standalone.json")]) == 0


def test_cli_pipeline_unstructured(examples, tmp_path):
    regions, merged = tmp_path / "regions.jsonl", tmp_path / "merged.jsonl"
    assert main(["normalize", "--format", "unstructured", str(examples / "chunks.unstructured.json"),
                 "--pdf", str(examples / "sample.pdf"), "-o", str(regions)]) == 0
    assert main(["consolidate", str(regions), "-o", str(merged)]) == 0
    assert main(["annotate", str(examples / "sample.pdf"), str(merged), "-o", str(tmp_path / "out.pdf")]) == 0
    assert main(["verify", str(tmp_path / "out.pdf"), str(merged), "--text-check", "required"]) == 0


@pytest.mark.parametrize("content,match", [
    ("{bad json}", "invalid JSON"), ("{\"texts\": []}", "no supported items"),
    ('{"texts":[{"self_ref":"a","text":"x"}]}', "prov"),
    ('{"texts": NaN}', "nonfinite"), ("[", "invalid JSON"),
])
def test_bad_input_no_output(tmp_path, capsys, content, match):
    source, output = tmp_path / "input.json", tmp_path / "output.jsonl"
    source.write_text(content)
    assert main(["normalize", "--format", "docling", str(source), "-o", str(output)]) == 1
    assert match in capsys.readouterr().err
    assert not output.exists()


def test_protect_inputs_existing_and_conflicting_outputs(examples, tmp_path, capsys):
    original = examples / "sample.pdf"
    before = original.read_bytes()
    assert main(["annotate", str(original), str(examples / "regions.jsonl"), "-o", str(original), "--force"]) == 1
    assert original.read_bytes() == before
    out = tmp_path / "out.pdf"
    out.write_bytes(b"keep")
    base = ["annotate", str(original), str(examples / "regions.jsonl"), "-o", str(out)]
    assert main(base) == 1
    assert out.read_bytes() == b"keep"
    assert main(base + ["--force", "--manifest", str(out)]) == 1
    assert out.read_bytes() == b"keep"
    assert main(base + ["--force"]) == 0
    assert out.read_bytes().startswith(b"%PDF")


def test_symlink_to_input_protected(examples, tmp_path):
    alias = tmp_path / "alias.pdf"
    alias.symlink_to(examples / "sample.pdf")
    assert main(["annotate", str(examples / "sample.pdf"), str(examples / "regions.jsonl"), "-o", str(alias), "--force"]) == 1
    assert alias.is_symlink()


def test_manifest_location_error_before_pdf_write(examples, tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    output = tmp_path / "out.pdf"
    assert main(["annotate", str(examples / "sample.pdf"), str(examples / "regions.jsonl"), "-o", str(output),
                 "--manifest", str(nested / "citations.json")]) == 1
    assert not output.exists()


def test_real_console_entrypoint(examples, tmp_path):
    # Real subprocess exercises installed console script/module exit code propagation.
    result = subprocess.run([sys.executable, "-m", "ragwrench.highlight", "annotate", str(examples / "sample.pdf"),
                             str(examples / "regions.jsonl"), "-o", str(tmp_path / "out.pdf")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, "-m", "ragwrench.highlight", "verify", str(tmp_path / "out.pdf"),
                             str(examples / "regions.jsonl")], capture_output=True, text=True)
    assert result.returncode == 0 and "OK 2/2" in result.stdout
    result = subprocess.run([sys.executable, "-m", "ragwrench.highlight", "verify", str(tmp_path / "absent.pdf"),
                             str(examples / "regions.jsonl")], capture_output=True, text=True)
    assert result.returncode == 1 and "Traceback" not in result.stderr


@pytest.mark.parametrize("arguments", [[], ["normalize"], ["verify", "a", "b", "--tolerance", "nan"],
                                      ["verify", "a", "b", "--tolerance", "-1"]])
def test_usage_exit_code(arguments):
    with pytest.raises(SystemExit) as exc:
        main(arguments)
    assert exc.value.code == 2


@pytest.mark.parametrize("value", [
    {}, {"chunk_id": "c", "source_ref": "s", "page": 0, "rect": [0, 0, 1, 1]},
    {"chunk_id": "c", "source_ref": "s", "page": True, "rect": [0, 0, 1, 1]},
    {"chunk_id": "c", "source_ref": "s", "page": 1, "rect": [2, 0, 1, 1]},
    {"chunk_id": "c", "source_ref": "s", "page": 1, "rect": [0, 0, 1]},
])
def test_malformed_regions(value, tmp_path):
    path = tmp_path / "regions.jsonl"
    path.write_text(json.dumps(value))
    with pytest.raises(CiteError, match="regions line 1"):
        read_regions(path)


def test_duplicate_and_empty_regions(tmp_path):
    path = tmp_path / "regions.jsonl"
    path.write_text("")
    with pytest.raises(CiteError, match="expected"):
        read_regions(path)
    r = Region("c", "s", 1, (0, 0, 1, 1))
    path.write_text(regions_jsonl([r]) * 2)
    with pytest.raises(CiteError, match="duplicate"):
        read_regions(path)


def test_metadata_refs_are_never_paths(tmp_path):
    payload = "../../outside/$(touch secret)"
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"texts": [{"self_ref": payload, "text": "safe", "prov": [{"page_no": 1,
        "bbox": {"l": 0, "t": 0, "r": 10, "b": 10}}]}],
        "pages": {"1": {"size": {"width": 100, "height": 100}}}}))
    output = tmp_path / "regions.jsonl"
    assert main(["normalize", "--format", "docling", str(source), "-o", str(output)]) == 0
    assert read_regions(output)[0].source_ref == payload
    assert sorted(p.name for p in tmp_path.iterdir()) == ["regions.jsonl", "source.json"]


def test_file_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("ragwrench.highlight.core.MAX_JSON_BYTES", 10)
    source = tmp_path / "large.jsonl"
    source.write_bytes(b" " * 11)
    with pytest.raises(CiteError, match="at most 10 bytes"):
        read_regions(source)


def test_atomic_failure_preserves_output(tmp_path):
    from ragwrench.highlight.core import atomic_write
    path = tmp_path / "keep"
    path.write_bytes(b"original")
    def fail(stream):
        stream.write(b"partial")
        raise OSError("simulated full filesystem")
    with pytest.raises(OSError, match="simulated"):
        atomic_write(path, fail)
    assert path.read_bytes() == b"original"
    assert [p.name for p in tmp_path.iterdir()] == ["keep"]


def test_invalid_source_refs():
    with pytest.raises(CiteError, match="include source_ref"):
        Region.from_dict({"chunk_id": "c", "source_ref": "s", "source_refs": ["x"], "page": 1, "rect": [0, 0, 1, 1]})
