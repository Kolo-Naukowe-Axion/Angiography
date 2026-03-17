from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from cadica_selected_utils import (  # type: ignore
        DEFAULT_CADICA_ROOT,
        DEFAULT_SPLIT_MANIFEST,
        SPLITS,
        load_split_manifest,
        normalize_split_manifest,
    )
else:
    from .cadica_selected_utils import (
        DEFAULT_CADICA_ROOT,
        DEFAULT_SPLIT_MANIFEST,
        SPLITS,
        load_split_manifest,
        normalize_split_manifest,
    )


def default_output_path(cadica_root: Path, split_manifest: Path) -> Path:
    return cadica_root / "splits" / split_manifest.stem / "manifest.json"


def export_split_manifest(cadica_root: Path, split_manifest: Path, output_path: Path) -> dict:
    cadica_root = cadica_root.resolve()
    payload = normalize_split_manifest(load_split_manifest(split_manifest), cadica_root=cadica_root)
    payload["dataset_root"] = str(cadica_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the tracked CADICA seed-42 split as a full selected-video manifest."
    )
    parser.add_argument("--cadica-root", type=Path, default=DEFAULT_CADICA_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output or default_output_path(args.cadica_root, args.split_manifest)
    payload = export_split_manifest(
        cadica_root=args.cadica_root,
        split_manifest=args.split_manifest,
        output_path=output_path,
    )

    print(f"Wrote full split manifest to {output_path.resolve()}")
    for split in SPLITS:
        split_payload = payload["splits"][split]
        print(
            f"{split:5s}: "
            f"{len(split_payload['patients']):2d} patients, "
            f"{len(split_payload['selected_videos']):3d} selected videos"
        )


if __name__ == "__main__":
    main()
