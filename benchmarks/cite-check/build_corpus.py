"""Rebuild the original synthetic corpus. Labels are authored here, not by the checker.

Cases intentionally isolate mechanisms. This corpus is a regression benchmark,
not a sampled distribution of production RAG errors or a semantic gold standard.
"""
import json
from pathlib import Path

cases = []


def case(name, answer, source_texts, expected, *, note="", indices=None):
    cases.append({"id": name, "note": note, "expected": expected,
                  "record": {"answer": answer,
                             "sources": [{"idx": idx, "id": f"s{idx}", "title": "synthetic.txt", "text": text}
                                         for idx, text in zip(indices or range(1, len(source_texts) + 1), source_texts)]}})


clean = [
    ("square", "The rover weighs 10 kg [1].", "The rover weighs 10 kg."),
    ("cjk", "保修期为12个月【1】。", "保修期为12个月。"),
    ("source", "The archive stores local snapshots Source 1.", "The archive stores local snapshots."),
    ("ragflow", "The battery lasts 8 hours. ##0$$", "The battery lasts 8 hours."),
    ("id", "The battery lasts 8 hours [ID:0].", "The battery lasts 8 hours."),
    ("thousands", "The store holds 1,000 records [1].", "The store holds 1000 records."),
    ("percent", "The completion rate is 25% [1].", "The completion rate is 25 percent."),
    ("date", "The launch date is 2026/11/01 [1].", "The launch date is 2026-11-01."),
    ("fullwidth", "The rover weighs １０ kg [1].", "The rover weighs 10 kilograms."),
    ("quote", 'The manual says “keep the signal stable” [1].', "The manual says keep the signal stable."),
    ("markdown", "# Reference [90]\n```sh\nrm -rf / [99]\n```\nThe warranty lasts 12 months [1].", "The warranty lasts 12 months."),
    ("postfix", "The warranty lasts 12 months. [1]", "The warranty lasts 12 months."),
]
for name, answer, source in clean:
    case("clean-" + name, answer, [source], [])
case("clean-group", "The rover uses solar panels [1,2].", ["The rover uses solar panels."] * 2, [])
case("clean-adjacent", "The rover uses solar panels [1][2].", ["The rover uses solar panels."] * 2, [])
case("clean-sparse", "The rover uses solar panels [7].", ["The rover uses solar panels."], [], indices=[7])
case("clean-mixed", "The battery lasts 8 hours [1]. 保修期为12个月【2】。", ["The battery lasts 8 hours.", "保修期为12个月。"], [])

for i, marker in enumerate(("[9]", "【5】", "[1,2]", "[1][2]", "##8$$", "Source 6")):
    case(f"dangling-{i}", "The rover uses solar panels " + marker + ".", [], ["CC001"], note="No source can resolve the marker(s).")
for i, marker in enumerate(("[abc]", "[1,]", "[1", "[0]", "##x$$", "Source x")):
    case(f"malformed-{i}", "The rover uses solar panels " + marker + ".", [], ["CC001", "CC002"], note="Malformed markers cannot satisfy citation coverage.")

claims = ["The rover uses solar panels.", "The warranty lasts 12 months.", "The archive stores local snapshots.",
          "保修期为12个月。", "The completion rate is 25%.", "Returns are free."]
for i, claim in enumerate(claims):
    case(f"uncited-{i}", claim, [], ["CC002"])

for i, claim in enumerate(("The rover uses solar panels", "The warranty covers accidental damage", "The archive stores local snapshots",
                            "The engine burns liquid hydrogen", "The library supports incremental updates", "火星探测器使用太阳能")):
    case(f"unsupported-{i}", claim + " [1].", ["Bananas grow in warm tropical forests."], ["CC003"])

numbers = [
    ("The warranty lasts 24 months [1].", "The warranty lasts 12 months."),
    ("The launch date is 2026-11-02 [1].", "The launch date is 2026-11-01."),
    ("The completion rate is 15% [1].", "The completion rate is 25%."),
    ("The rover weighs 10 g [1].", "The rover weighs 10 kg."),
    ("保修期为24个月【1】。", "保修期为12个月。"),
    ("The warranty lasts 12 years [1].", "The warranty lasts 12 months."),
    ("The budget is $100 [1].", "The budget is 100 EUR."),
    ("The target temperature is -10 °C [1].", "The target temperature is 10 °C."),
]
for i, (answer, source) in enumerate(numbers):
    case(f"quantity-{i}", answer, [source], ["CC004"], note="A literal or its attached unit was changed, not inferred by an LLM.")
for i, claim in enumerate(("The rover uses solar panels", "The warranty covers accidental damage", "The archive stores local snapshots",
                            "The engine burns liquid hydrogen", "The library supports incremental updates", "火星探测器使用太阳能")):
    case(f"misattributed-{i}", claim + " [1].", ["Bananas grow in warm tropical forests.", claim + "."],
         ["CC003", "CC005", "CC007"], note="Source 2 exactly supports the claim; citation points to unrelated source 1.")
for i, (quote, original) in enumerate((("signal stable", "signal steady"), ("archive frozen", "archive closed"),
                                      ("system ready", "system waiting"), ("engine available", "engine unavailable"),
                                      ("battery healthy", "battery damaged"), ("return approved", "return rejected"))):
    case(f"quote-{i}", f'The manual says "{quote}" [1].', [f"The manual says {original}."], ["CC006"])
for i, claim in enumerate(claims):
    case(f"unused-{i}", claim.rstrip("。.") + " [1].", [claim, "The ocean contains salt water."], ["CC007"])
case("question", "What powers the rover?", [], [], note="Questions are not classified as factual assertions.")
case("code-only", "```python\nprint('[99]')\n```", [], [], note="No prose analyzed; result contains an explicit note.")
case("semantic-limit-negation", "The warranty covers accidental damage [1].", ["The warranty does not cover accidental damage. The warranty covers manufacturing faults."], [],
     note="Known blind spot: high lexical overlap does not establish entailment; NOT a semantic positive label.")
case("semantic-limit-relation", "Mira defeated Kato [1].", ["Kato defeated Mira."], [],
     note="Known blind spot: identical word sets cannot detect swapped argument roles.")

for i, (quote, original) in enumerate((
    ("信号：稳定", "信号:稳定"), ("信号:稳定", "信号：稳定"),
    ("信号。稳定、继续", "信号.稳定,继续"), ("ＡＢＣ（稳定）！", "ABC(稳定)!"),
    ("Signal   stable", "signal stable"), ("他说‘稳定’", "他说'稳定'"),
)):
    case(f"quote-normalized-{i}", f'The manual says 「{quote}」 [1].',
         [f"The manual says {original}."], ["CC008"],
         note="Normalization-only evidence is informational by default and an error with --strict-quotes; never CC006.")
case("quote-fullwidth-exact", "手册说＂信号：稳定＂【1】。", ["手册说信号：稳定。"], [],
     note="Original quote contents are exact even with fullwidth delimiters.")
case("quote-punctuation-not-deleted", "手册说「信号：稳定」【1】。", ["手册说信号稳定。"], ["CC006"],
     note="Punctuation equivalents are mapped, never removed to force a match.")

if __name__ == "__main__":
    path = Path(__file__).resolve().parents[2] / "src/ragwrench/citecheck/data/benchmark.jsonl"
    path.write_text("".join(json.dumps(c, ensure_ascii=False, separators=(",", ":")) + "\n" for c in cases), encoding="utf-8")
    print(f"Wrote {len(cases)} independently labelled synthetic cases to {path}")
