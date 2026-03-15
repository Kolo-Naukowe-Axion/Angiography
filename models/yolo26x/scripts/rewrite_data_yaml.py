from __future__ import annotations

import argparse
from pathlib import Path


def rewrite_data_yaml(dataset_root: Path) -> Path:
    dataset_root = dataset_root.resolve()
    data_yaml = dataset_root / "data.yaml"
    lines = data_yaml.read_text(encoding="utf-8").splitlines()

    rewritten = False
    updated_lines: list[str] = []
    for line in lines:
        if line.startswith("path:"):
            updated_lines.append(f"path: {dataset_root}")
            rewritten = True
        else:
            updated_lines.append(line)

    if not rewritten:
        updated_lines.insert(0, f"path: {dataset_root}")

    data_yaml.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite a prepared CADICA data.yaml to the current dataset root.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()

    data_yaml = rewrite_data_yaml(args.dataset_root)
    print(f"Rewrote dataset path in {data_yaml}")


if __name__ == "__main__":
    main()
