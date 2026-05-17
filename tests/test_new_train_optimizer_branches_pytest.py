"""Branch-focused tests for optimizer instantiation in new_train.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.new_train import _instantiate_optimizer


@pytest.mark.unit
def test_instantiate_optimizer_rejects_non_callable():
    with pytest.raises(ValueError):
        _instantiate_optimizer(123, {'learning_rate': 1e-3})


@pytest.mark.unit
def test_instantiate_optimizer_signature_introspection_failure(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr('src.training.new_train.inspect.signature', lambda *_: (_ for _ in ()).throw(TypeError('x')))

    def factory(**kwargs):
        return kwargs

    out = _instantiate_optimizer(factory, {'learning_rate': 1e-3, 'beta_1': 0.9})
    assert out['learning_rate'] == 1e-3
    assert out['beta_1'] == 0.9


@pytest.mark.unit
def test_instantiate_optimizer_typeerror_fallback_constructor():
    import src.training.new_train as new_train_mod

    class Factory:
        def __init__(self):
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            if kwargs:
                raise TypeError('kwargs not accepted')
            return {'ok': True, 'calls': self.calls}

    original_signature = new_train_mod.inspect.signature
    new_train_mod.inspect.signature = lambda *_: (_ for _ in ()).throw(ValueError('no signature'))
    factory = Factory()
    try:
        out = _instantiate_optimizer(factory, {'foo': 1})
        assert out['ok'] is True
        assert out['calls'] == 2
    finally:
        new_train_mod.inspect.signature = original_signature
