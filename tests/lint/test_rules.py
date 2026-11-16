import pytest

from ragwrench.lint.lint import lint, suffix_overlap
from ragwrench.lint.model import Chunk, Config, InputError


def chunks(*texts):
    return [Chunk(f'c{i}', text) for i, text in enumerate(texts)]


def rules(*texts, source=None, **kwargs):
    report = lint(chunks(*texts), source, Config(min_chars=0, **kwargs))
    return {i.rule for i in report.issues}


@pytest.mark.parametrize('rule,bad,good', [
    ('CL001', ['```py\nx = 1'], ['```py\nx = 1\n```']),
    ('CL002', ['| row | value |\n| a | b |'], ['| key | value |\n| --- | --- |\n| a | b |']),
    ('CL003', ['Body.\n## Next'], ['## Next\nBody.']),
    ('CL004', ['The sentence is', 'continued here.'], ['The sentence ends.', 'Another begins.']),
    ('CL004', ['中文内容尚未', '结束。'], ['中文内容结束。', '下一句。']),
    ('CL006', ['Identical text.', 'Identical text.'], ['Different A.', 'Distinct B.']),
    ('CL008', ['- first item', '- second item'], ['- complete item', 'Ordinary paragraph.']),
])
def test_each_structural_rule_positive_negative(rule, bad, good):
    assert rule in rules(*bad)
    assert rule not in rules(*good)


def test_size_positive_negative():
    config = Config(min_chars=10, max_chars=20)
    assert any(i.rule == 'CL005' for i in lint(chunks('tiny'), config=config).issues)
    assert not any(i.rule == 'CL005' for i in lint(chunks('Exactly enough.'), config=config).issues)
    assert 'CL005' in rules('x' * 101, max_chars=100)


def test_coverage_positive_negative():
    assert 'CL007' in rules('Hello', source='Hello missing.')
    assert 'CL007' not in rules('Hello world.', source='Hello world.')


@pytest.mark.parametrize('text,unclosed', [
    ('````python\na\n```\nb\n````', False),
    ('````python\na\n```', True),
    ('~~~python\na\n~~~~', False),
    ('~~~~python\na\n~~~', True),
    ('```python\na\n~~~', True),
    ('   ```python\na\n   ```', False),
    ('    ```python\na', False),
    ('```bad`info\na', False),
    ('~~~`valid info\na\n~~~', False),
    ('```\na\n``` trailing', True),
    ('```\na\n````\n', False),
    ('```py\r\na\r\n```\r\n', False),
])
def test_commonmark_fences(text, unclosed):
    assert ('CL001' in rules(text)) == unclosed


def test_code_suppresses_other_structures():
    assert rules('```md\n| no header | x |\n## Heading\n- list\n```') == set()
    assert 'CL002' not in rules('~~~\n| row | value |\n~~~')


def test_escaped_pipes_are_prose():
    assert 'CL002' not in rules(r'An escaped \| pipe.')


def test_heading_syntax():
    assert 'CL003' not in rules('#hashtag')
    assert 'CL003' not in rules('####### too deep')
    assert 'CL003' in rules('###')


def test_table_mismatched_header():
    assert 'CL002' in rules('| a | b |\n| --- |\n| 1 | 2 |')


def test_near_duplicates():
    text = 'This is a deliberately long repeated paragraph about local retrieval systems. ' * 2
    report = lint(chunks(text, text + '!'), config=Config(min_chars=0))
    assert any(i.details.get('kind') == 'near' for i in report.issues)


def test_overlap():
    common = 'a sizeable common boundary text'
    report = lint(chunks('prefix ' + common, common + ' suffix'), config=Config(min_chars=0))
    assert any(i.details.get('kind') == 'overlap' for i in report.issues)
    assert suffix_overlap('abcabc', 'abcx') == 3
    assert suffix_overlap('xxxxx', 'zzzzz') == 0


def test_rule_disable():
    config = Config(disable=['CL001', 'CL003', 'CL005'])
    assert lint(chunks('```'), config=config).issues == []


def test_list_continuation_and_code_boundary():
    assert 'CL008' in rules('- item first line', '  continuation')
    assert 'CL008' not in rules('```\n- item\n```', '- next')
    assert 'CL004' not in rules('```\nplain unfinished\n```', 'continued')


def test_duplicate_pair_budget():
    with pytest.raises(InputError, match='pairs'):
        lint(chunks('a', 'b', 'c'), config=Config(max_pair_checks=2))
    assert lint(chunks('a', 'b', 'c'), config=Config(max_pair_checks=2, disable=['CL006']))


def test_large_near_duplicate_budget():
    with pytest.raises(InputError, match='200000'):
        lint(chunks('a' * 200001))


@pytest.mark.parametrize('kwargs', [
    {'min_chars': -1}, {'max_chars': 0}, {'max_chars': 10, 'min_chars': 11},
    {'near_duplicate': 0}, {'near_duplicate': float('nan')}, {'max_overlap': 2},
    {'disable': ['CL999']}, {'disable': 'CL001'}, {'max_pair_checks': True},
    {'fail_on': 'all'}, {'min_chars': True},
])
def test_invalid_config(kwargs):
    with pytest.raises(InputError):
        Config(**kwargs).validate()


def test_nbsp_is_not_valid_fence_closer_padding():
    assert 'CL001' in rules('```\ncode\n```\u00a0')
    assert 'CL001' not in rules('```\ncode\n```\t')


def test_total_duplicate_analysis_memory_budget():
    with pytest.raises(InputError, match='total characters'):
        lint(chunks(*['x' * 180000 for _ in range(6)]))


def test_unicode_separator_is_not_a_markdown_line_break():
    assert 'CL001' not in rules('Prose\u2028```')
    assert 'CL001' not in rules('Prose\u0085```')
