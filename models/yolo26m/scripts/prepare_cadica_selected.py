from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from cadica_selected_utils import (  # type: ignore
        DEFAULT_CADICA_ROOT,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_SPLIT_MANIFEST,
        SPLITS,
        prepare_dataset,
    )
else:
    from .cadica_selected_utils import (
        DEFAULT_CADICA_ROOT,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_SPLIT_MANIFEST,
        SPLITS,
        prepare_dataset,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a YOLO-format CADICA dataset using selected keyframes only."
    )
    parser.add_argument("--cadica-root", type=Path, default=DEFAULT_CADICA_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating symlinks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing non-empty output directory before preparing the dataset.",
    )
    args = parser.parse_args()

    summary = prepare_dataset(
        cadica_root=args.cadica_root,
        split_manifest=args.split_manifest,
        output_root=args.output_root,
        copy_images=args.copy_images,
        force=args.force,
    )

    print(f"Prepared dataset at {args.output_root.resolve()}")
    print(f"Materialization mode: {summary['materialization']}")
    for split in SPLITS:
        split_summary = summary["splits"][split]
        print(
            f"{split:5s}: "
            f"{split_summary['image_count']:4d} images, "
            f"{split_summary['positive_image_count']:4d} positive, "
            f"{split_summary['negative_image_count']:4d} negative, "
            f"{split_summary['bbox_count']:4d} boxes"
        )
    print(f"data.yaml: {args.output_root.resolve() / 'data.yaml'}")
    print(f"summary.json: {args.output_root.resolve() / 'summary.json'}")


if __name__ == "__main__":
    main()
