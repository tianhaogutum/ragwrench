"""One bounded literal UTF-8 text input for offline query/document encoding."""

import json

import pytest

from ragwrench.bm25 import cli, store
from ragwrench.bm25.encoder import Encoder

TEXT = "  ＡＴＬＡＳ 北京 北京\r\n東京\t서울 é\r𠮷野家\n  "


def forbidden(*args, **kwargs):
    raise AssertionError("unexpected file/storage/encoding access")


@pytest.fixture(autouse=True)
def no_storage(monkeypatch):
    monkeypatch.setattr(cli, "open_local", forbidden)
    monkeypatch.setattr(store, "QdrantClient", forbidden)
    monkeypatch.setattr(cli, "read_corpus", forbidden)


def test_multilingual_file(tmp_path, capsys):
    source = tmp_path / "multilingual.txt"
    source.write_bytes(TEXT.encode("utf-8"))
    # Establish ordinary text as a positive control before the new parser option.
    assert cli.main(["encode", TEXT]) == 0
    expected = json.loads(capsys.readouterr().out)
    assert cli.main(["encode", "--text-file", str(source)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == expected


@pytest.mark.parametrize("document", [False, True])
@pytest.mark.parametrize("tokenizer", ["whitespace", "unicode", "cjk-bigram"])
@pytest.mark.parametrize("saved", [False, True])
def test_encoder_parity(tmp_path, capsys, monkeypatch, document, tokenizer, saved):
    source = tmp_path / "text.txt"
    source.write_bytes(TEXT.encode())
    encoder = Encoder(tokenizer, k1=1.8, b=0.3, avgdl=19.5, stopwords=("atlas",))
    if saved:
        config = tmp_path / "config.json"
        config.write_text(json.dumps(encoder.config), encoding="utf-8")
        flags = ["--encoder-config", str(config)]
    else:
        words = tmp_path / "stopwords.txt"
        words.write_text("ＡＴＬＡＳ\n", encoding="utf-8")
        flags = [
            "--tokenizer",
            tokenizer,
            "--k1",
            "1.8",
            "--b",
            "0.3",
            "--avgdl",
            "19.5",
            "--stopwords",
            str(words),
        ]
    vector = encoder.encode_documents([TEXT])[0] if document else encoder.encode_query(TEXT)
    original = Encoder._encode
    seen = []

    def capture(self, text, query):
        seen.append(text)
        return original(self, text, query)

    monkeypatch.setattr(Encoder, "_encode", capture)
    assert (
        cli.main(
            ["encode", "--text-file", str(source), *flags, *(["--document"] if document else [])]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    assert seen == [TEXT]  # CRLF, lone CR, tabs and trailing whitespace survive.
    assert json.loads(captured.out) == {
        "tokens": encoder.tokenizer(TEXT),
        **vector.as_dict(),
        "config": encoder.config,
        "collision_stats": encoder.collision_stats,
    }


@pytest.mark.parametrize(
    "argv", [[], ["text", "--text-file", "missing"], ["", "--text-file", "missing"]]
)
def test_exactly_one_before_any_files(capsys, monkeypatch, argv):
    monkeypatch.setattr(cli, "read_encoder_config", forbidden)
    monkeypatch.setattr(cli, "read_stopwords", forbidden)
    monkeypatch.setattr(cli.Path, "open", forbidden)
    with pytest.raises(SystemExit) as exc:
        cli.main(["encode", *argv, "--encoder-config", "missing"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "error:" in captured.err


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--tokenizer", "cjk-bigram"),
        ("--stopwords", "missing"),
        ("--k1", "1.2"),
        ("--b", "0.75"),
        ("--avgdl", "100"),
    ],
)
def test_config_conflicts_before_any_files(capsys, monkeypatch, flag, value):
    monkeypatch.setattr(cli.Path, "open", forbidden)
    monkeypatch.setattr(Encoder, "_encode", forbidden)
    assert (
        cli.main(["encode", "--text-file", "missing", "--encoder-config", "missing", flag, value])
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == "" and "cannot be combined" in captured.err


@pytest.mark.parametrize("document", [False, True])
@pytest.mark.parametrize("text", ["", " \r\n\t ", "ＡＴＬＡＳ\natlas\r\n"])
def test_empty_warnings(tmp_path, capsys, document, text):
    source = tmp_path / "empty.txt"
    source.write_bytes(text.encode())
    config = tmp_path / "config.json"
    encoder = Encoder(stopwords=("atlas",))
    config.write_text(json.dumps(encoder.config), encoding="utf-8")
    with pytest.warns(UserWarning, match="empty or stopword-only"):
        assert (
            cli.main(
                [
                    "encode",
                    "--text-file",
                    str(source),
                    "--encoder-config",
                    str(config),
                    *(["--document"] if document else []),
                ]
            )
            == 0
        )
    assert json.loads(capsys.readouterr().out) == {
        "tokens": [],
        "indices": [],
        "values": [],
        "config": encoder.config,
        "collision_stats": encoder.collision_stats,
    }


def test_bom_is_literal_and_dash_is_filename(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "-"
    source.write_bytes(b"\xef\xbb\xbfAtlas\r\n")
    assert cli.main(["encode", "--text-file", "-", "--tokenizer", "whitespace"]) == 0
    assert json.loads(capsys.readouterr().out)["tokens"] == ["\ufeffatlas"]
    # An explicitly empty positional text still counts as a supplied text source.
    with pytest.warns(UserWarning, match="empty or stopword-only"):
        assert cli.main(["encode", ""]) == 0
    assert json.loads(capsys.readouterr().out)["tokens"] == []


@pytest.mark.parametrize("raw", [b"\xff", b"a\xc0\xaf", b"\xed\xa0\x80", b"\xf0\x9f\x92"])
def test_strict_utf8_before_encoding(tmp_path, capsys, monkeypatch, raw):
    source = tmp_path / "invalid.txt"
    source.write_bytes(raw)
    monkeypatch.setattr(Encoder, "_encode", forbidden)
    assert cli.main(["encode", "--text-file", str(source)]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "codec can't decode" in captured.err


@pytest.mark.parametrize("kind", ["missing", "directory", "denied"])
def test_unreadable_before_encoding(tmp_path, capsys, monkeypatch, kind):
    source = tmp_path / "text.txt"
    if kind == "directory":
        source.mkdir()
    if kind == "denied":

        def denied(*args, **kwargs):
            raise PermissionError("forced unreadable file")

        monkeypatch.setattr(cli.Path, "open", denied)
    monkeypatch.setattr(Encoder, "_encode", forbidden)
    assert cli.main(["encode", "--text-file", str(source)]) == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("document", [False, True])
def test_real_character_limit(tmp_path, capsys, monkeypatch, document):
    from ragwrench.bm25.tokenizers import MAX_TEXT_CHARS

    source = tmp_path / "large.txt"
    flags = ["--document"] if document else []
    source.write_bytes((" " * (MAX_TEXT_CHARS - 1) + "x").encode())
    assert cli.main(["encode", "--text-file", str(source), *flags]) == 0
    assert json.loads(capsys.readouterr().out)["tokens"] == ["x"]
    source.write_bytes(b" " * MAX_TEXT_CHARS + b"x")
    monkeypatch.setattr(Encoder, "_encode", forbidden)
    assert cli.main(["encode", "--text-file", str(source), *flags]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "text exceeds 1000000 characters" in captured.err


def test_real_byte_boundary(tmp_path, capsys, monkeypatch):
    source = tmp_path / "large.txt"
    # Exact byte cap decodes successfully then fails the independent character cap.
    source.write_bytes("😀".encode() * (cli.MAX_ENCODE_TEXT_BYTES // 4))
    monkeypatch.setattr(Encoder, "_encode", forbidden)
    assert cli.main(["encode", "--text-file", str(source)]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "text exceeds 1000000 characters" in captured.err
    with source.open("ab") as stream:
        stream.write(b"\xff")
    assert cli.main(["encode", "--text-file", str(source)]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "exceeds 4 MiB" in captured.err


def test_read_is_bounded(monkeypatch):
    from io import BytesIO

    sizes = []

    class Stream(BytesIO):
        def read(self, size=-1):
            sizes.append(size)
            assert 0 < size <= 4 * 1024 * 1024 + 1
            return super().read(size)

    monkeypatch.setattr(cli.Path, "open", lambda *a, **kw: Stream(b"x\r\n"))
    assert cli._read_encode_text(cli.Path("ignored")) == "x\r\n"
    assert sizes == [4 * 1024 * 1024 + 1]


@pytest.mark.parametrize("document", [False, True])
def test_collision_diagnostics(tmp_path, capsys, document):
    source = tmp_path / "collision.txt"
    source.write_bytes(b"collision402\ncollision169677\r\n")
    with pytest.warns(UserWarning, match="hash collision"):
        assert (
            cli.main(
                [
                    "encode",
                    "--text-file",
                    str(source),
                    "--tokenizer",
                    "whitespace",
                    "--b",
                    "0",
                    *(["--document"] if document else []),
                ]
            )
            == 0
        )
    result = json.loads(capsys.readouterr().out)
    assert result["indices"] == [3793890753]
    assert result["values"] == ([2.0] if document else [1.0])
    assert result["collision_stats"]["collision_count"] == 1


@pytest.mark.parametrize("invalid", ["config", "runtime", "numeric"])
def test_existing_validation(tmp_path, capsys, monkeypatch, invalid):
    source = tmp_path / "text.txt"
    source.write_bytes(b"a a a")
    config = tmp_path / "config.json"
    saved = Encoder("whitespace", k1=1e308 if invalid == "numeric" else 1.2).config
    if invalid == "runtime":
        saved["unicode_version"] = "incompatible"
    config.write_text("{" if invalid == "config" else json.dumps(saved), encoding="utf-8")
    if invalid != "numeric":
        monkeypatch.setattr(Encoder, "_encode", forbidden)
    assert (
        cli.main(
            ["encode", "--text-file", str(source), "--encoder-config", str(config), "--document"]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == "" and "error:" in captured.err
