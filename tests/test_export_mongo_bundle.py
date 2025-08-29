from datetime import datetime
from typing import Any, Dict

import pytest

from scripts.export_mongo_bundle import _serialize

try:
    from bson import ObjectId  # type: ignore
except Exception:  # pragma: no cover
    ObjectId = None  # type: ignore


@pytest.mark.skipif(ObjectId is None, reason="bson.ObjectId not available")
def test_serialize_handles_objectid_and_datetime():
    now = datetime(2024, 1, 2, 3, 4, 5)
    sample: Dict[str, Any] = {
        "_id": ObjectId("65a1e6b5e1382378f10f9a1a"),
        "nested": {
            "created_at": now,
            "items": [ObjectId("65a1e6b5e1382378f10f9a1b"), {"ts": now}],
        },
    }

    out = _serialize(sample)

    assert isinstance(out["_id"], str)
    assert out["nested"]["created_at"] == now.isoformat()
    assert isinstance(out["nested"]["items"][0], str)
    assert out["nested"]["items"][1]["ts"] == now.isoformat()
