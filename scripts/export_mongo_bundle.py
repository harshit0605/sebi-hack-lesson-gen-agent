import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on the path to import db.get_database
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.graphs.content_generation.db.db import get_database  # noqa: E402

try:
    from bson import ObjectId  # type: ignore
except Exception:  # pragma: no cover
    ObjectId = None  # type: ignore


def _serialize(obj: Any) -> Any:
    """Recursively convert Mongo types (ObjectId, datetime) to JSON-serializable ones."""
    # ObjectId
    if ObjectId is not None and isinstance(obj, ObjectId):  # type: ignore[arg-type]
        return str(obj)
    # datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    # dict
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


def export_bundle() -> Dict[str, Any]:
    db = get_database()

    collections = {
        "learning_journeys": db.learning_journeys,
        "lessons": db.lessons,
        "content_blocks": db.content_blocks,
        "anchors": db.anchors,
        "processing_states": db.processing_states,
    }

    bundle: Dict[str, Any] = {}
    for name, coll in collections.items():
        docs = list(coll.find({}))
        bundle[name] = _serialize(docs)

    # Include a tiny manifest
    bundle["_export_meta"] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "collections": list(collections.keys()),
        "counts": {k: len(v) for k, v in bundle.items() if isinstance(v, list)},
    }

    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MongoDB collections into a JSON bundle.")
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default="",
        help="Output filepath for the export. If omitted, prints to stdout.",
    )

    args = parser.parse_args()

    data = export_bundle()
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Export written to: {out_path.as_posix()}")
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
