"""Snapshot-free plan validation using the packaged, local-only JSON Schema."""
from functools import lru_cache
from importlib.resources import files
import json

from jsonschema import Draft202012Validator, validators

from . import __version__
from .model import InputError


class PlanValidationError(InputError):
    def __init__(self, path, message):
        self.path = path
        self.message = message
        super().__init__(f"invalid fix plan at {path}: {message}")


def schema_text():
    return files('ragwrench.lint').joinpath('schemas', 'fix-plan-v1.schema.json').read_text(encoding='utf-8')


@lru_cache(maxsize=1)
def _validator():
    # JSON Schema accepts 1.0 as an integer; saved plans require exact JSON integer
    # types because deterministic apply compares canonical JSON representations.
    checker = Draft202012Validator.TYPE_CHECKER.redefine('integer', lambda _, value: type(value) is int)
    validator = validators.extend(Draft202012Validator, type_checker=checker)
    return validator(json.loads(schema_text()))


def _path(parts):
    return '$' + ''.join(f'[{part}]' if isinstance(part, int) else
                         f'.{part}' if part.isidentifier() else f'[{json.dumps(part)}]'
                         for part in parts)


def _schema_error(error):
    parts = list(error.absolute_path)
    if error.validator == 'required':
        missing = next(key for key in error.validator_value if key not in error.instance)
        parts.append(missing)
        message = 'required field is missing'
    elif error.validator == 'additionalProperties':
        parts.append(sorted(set(error.instance) - set(error.schema.get('properties', {})))[0])
        message = 'unknown field'
    elif error.validator == 'dependentRequired':
        missing = next(other for key, others in error.validator_value.items() if key in error.instance
                       for other in others if other not in error.instance)
        parts.append(missing)
        message = 'character hints must be supplied together'
    else:
        # Avoid jsonschema's default messages, which can echo entire sensitive
        # replacement records or metadata. Field paths plus constraints suffice.
        messages = {
            'type': f"expected {error.validator_value}",
            'const': f"must equal {error.validator_value!r}",
            'enum': 'must be one of the supported values',
            'minimum': f"must be >= {error.validator_value}",
            'exclusiveMinimum': f"must be > {error.validator_value}",
            'maximum': f"must be <= {error.validator_value}",
            'minLength': f'must contain at least {error.validator_value} characters',
            'maxLength': f'must contain at most {error.validator_value} characters',
            'pattern': 'must be a lowercase 64-digit SHA256' if error.validator_value == '^[0-9a-f]{64}$'
                       else 'must contain non-whitespace text',
        }
        message = messages.get(error.validator, 'does not satisfy the plan schema')
    return PlanValidationError(_path(parts), message)


def validate_structure(plan):
    """Check schema and self-contained semantics; never read chunks or run repair.

    Report the first error in schema/array order. Hash contents, snapshot freshness,
    coordinate agreement with the original stream and conservation need preview/apply.
    """
    error = next(_validator().iter_errors(plan), None)
    if error is not None:
        raise _schema_error(error)
    if plan['tool_version'] != __version__:
        raise PlanValidationError('$.tool_version', f'unsupported tool version; expected {__version__}')
    if plan['config']['max_chars'] < plan['config']['min_chars']:
        raise PlanValidationError('$.config.max_chars', 'must be >= min_chars')
    previous_chunk = previous_byte = 0
    seen_ids = set()
    for index, edit in enumerate(plan['edits']):
        path = f'$.edits[{index}]'
        for unit in ('chunk', 'byte'):
            if edit[f'end_{unit}'] < edit[f'start_{unit}']:
                raise PlanValidationError(f'{path}.end_{unit}', f'must be >= start_{unit}')
        if edit['start_chunk'] < previous_chunk:
            raise PlanValidationError(f'{path}.start_chunk', 'edits must be ordered and nonoverlapping')
        if edit['start_byte'] < previous_byte:
            raise PlanValidationError(f'{path}.start_byte', 'edits must be ordered and nonoverlapping')
        previous_chunk, previous_byte = edit['end_chunk'], edit['end_byte']
        for item, record in enumerate(edit['replacement']):
            record_path = f'{path}.replacement[{item}]'
            if record['id'] in seen_ids:
                raise PlanValidationError(f'{record_path}.id', 'duplicate replacement chunk id')
            seen_ids.add(record['id'])
            if 'start_char_idx' in record and record['end_char_idx'] <= record['start_char_idx']:
                raise PlanValidationError(f'{record_path}.end_char_idx', 'must be > start_char_idx')
    return plan
