"""Python retrieval cutoff contract before any migration or verification I/O."""
from __future__ import annotations

import json

import pytest

from ragwrench.migrate import checkpoints, engine
from ragwrench.migrate.model import ParityError
from ragwrench.migrate.stores import Store
from .test_checkpoints import source, target_for
from .test_paths import tree_state


class Cutoff(int):
    pass


class NumericTrap:
    def __lt__(self, other):
        pytest.fail("Unsupported top_k was compared")

    def __index__(self):
        pytest.fail("Unsupported top_k invoked __index__")

    def __int__(self):
        pytest.fail("Unsupported top_k invoked __int__")


INVALID = [1.5, 1.0, float('nan'), float('inf'), float('-inf'), '2', None,
           True, False, 0, -1, Cutoff(0), Cutoff(-1), [], {}, NumericTrap()]


@pytest.mark.parametrize('mode', ['checkpoint', 'resume', 'copy', 'dry-run', 'verify', 'verify-skipped'])
def test_top_k_refused_before_io(source, tmp_path, monkeypatch, mode):
    target = target_for(source, tmp_path)
    checkpoint = tmp_path / 'progress.json'
    if mode in {'resume', 'verify', 'verify-skipped'}:
        assert engine.copy(source, target, checkpoint=checkpoint, top_k=2)['parity']['ok']
    sentinel = tmp_path / 'sentinel'
    sentinel.write_bytes(b'preserve me')
    (tmp_path / 'empty').mkdir()
    before = tree_state(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail('top_k reached path/checkpoint/snapshot/backend I/O')

    with monkeypatch.context() as guard:
        for module, names in [(engine, ('checked_store_path', 'disjoint', 'read_store', 'Store', 'check_directory')),
                              (checkpoints, ('checkpoint_path', 'load', 'save'))]:
            for name in names:
                guard.setattr(module, name, forbidden)
        for value in INVALID:
            with pytest.raises(ParityError) as caught:
                if mode.startswith('verify'):
                    engine.verify(source, target, top_k=value,
                                  sample_queries=0 if mode == 'verify-skipped' else 10)
                else:
                    kwargs = {'top_k': value}
                    if mode in {'checkpoint', 'resume'}:
                        kwargs.update(checkpoint=checkpoint, resume=mode == 'resume')
                    if mode == 'dry-run':
                        kwargs['dry_run'] = True
                    engine.copy(source, target, **kwargs)
            assert str(caught.value) == 'top_k must be a positive integer'
            assert caught.value.__cause__ is None
    assert tree_state(tmp_path) == before
    if mode in {'checkpoint', 'copy', 'dry-run'}:
        assert not target.path.exists() and not checkpoint.exists()


@pytest.mark.parametrize('value', [1, 2, 4097, Cutoff(2), None],
                         ids=['one', 'two', 'above-cli-bound', 'int-subclass', 'default'])
def test_top_k_supported_create_resume_verify(source, tmp_path, value):
    target = target_for(source, tmp_path)
    checkpoint = tmp_path / 'progress.json'
    before = tree_state(source.path)
    kwargs = {} if value is None else {'top_k': value}
    result = engine.copy(source, target, checkpoint=checkpoint, batch_size=2, **kwargs)
    assert result['copied'] == 5 and result['parity']['ok']
    assert result['parity']['top_k']['k'] == (10 if value is None else value)
    again = engine.copy(source, target, checkpoint=checkpoint, resume=True, **kwargs)
    assert again['copied'] == 0 and again['skipped'] == 5 and again['parity']['ok']
    assert engine.verify(source, target, **kwargs)['ok']
    assert json.loads(checkpoint.read_text())['parity']['ok']
    assert tree_state(source.path) == before


@pytest.mark.parametrize('operation', ['copy', 'verify'])
@pytest.mark.parametrize('error_type', [ValueError, TypeError, OSError, RuntimeError])
def test_top_k_preserves_downstream_error_identity(source, tmp_path, monkeypatch, operation, error_type):
    target = target_for(source, tmp_path)
    error = error_type('unrelated snapshot error')

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(engine, 'read_store', fail)
    with pytest.raises(error_type) as caught:
        getattr(engine, operation)(source, target, top_k=2)
    assert caught.value is error


def test_top_k_preserves_query_error_identity(source, tmp_path, monkeypatch):
    target = target_for(source, tmp_path)
    engine.copy(source, target)
    error = ValueError('unrelated retrieval failure')

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(Store, 'query', fail)
    with pytest.raises(ValueError) as caught:
        engine.verify(source, target, top_k=2)
    assert caught.value is error
