"""Type-preserving source IDs and deterministic Qdrant UUID mapping."""

import json
import uuid

ID_NAMESPACE = uuid.UUID("5a9694fb-f85b-5828-922c-e34aa9d5757a")


def id_key(value: str | int) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def point_id(value: str | int) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, id_key(value)))
