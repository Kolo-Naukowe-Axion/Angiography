# Angiography uv Workspace Design

## Goal

Set up `uv` at the Angiography repo root so the Python parts of the repo can be bootstrapped consistently from one place instead of relying on ad hoc virtualenv creation in each folder.

## Decision

Use a repo-level `uv` workspace rooted at `Angiography/` with script-style member projects:

- `demo-app/backend`
- `models/yolov8-stenosis`
- `models/yolo26x`
- `models/yolo26m_cadica`
- `models/sam_vmnet`
- `output/jupyter-notebook`

Each member gets its own `pyproject.toml` so `uv` can resolve dependencies from the repo root with `uv sync --package <member>`.

## Why this design

- The repo already contains several independent Python surfaces.
- Most of them are application or script workflows, not publishable libraries, so `tool.uv.package = false` is the right default.
- `sam_vmnet` has materially different runtime expectations, so its `uv` definition is intentionally conservative and focused on a lockable local base environment.
- The vendored `med_sam` code now has a `pyproject.toml` so it can participate in `uv`-managed dependency resolution as a local path dependency.

## Trade-offs

- `uv` workspaces share a lockfile, so this is a repo-level coordination layer rather than a promise that every project has identical runtime needs.
- `sam_vmnet` remains the most specialized member; its local `uv` setup is intentionally aimed at a usable macOS/base environment first.
- Existing shell scripts still work as-is; they were not rewritten to require `uv`.

## Expected usage

From `Angiography/`:

```bash
uv lock
uv sync --package angio-demo-backend --group dev
uv run --package angio-demo-backend pytest demo-app/backend/tests

uv sync --package angio-jupyter-notebook
uv sync --package angio-yolo26m-cadica --group dev
uv sync --package angio-yolo26x --group dev
uv sync --package angio-yolov8-stenosis

# SAM-VMNet base environment
uv sync --package angio-sam-vmnet

# SAM-VMNet notebook helpers
uv sync --package angio-sam-vmnet --extra notebooks
```
