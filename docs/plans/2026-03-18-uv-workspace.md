# Angiography uv Workspace Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a repo-level `uv` workspace for the Angiography Python projects and verify that the main members can be locked and synced from the repository root.

**Architecture:** Define a root workspace `pyproject.toml`, add member-level script-project `pyproject.toml` files, represent vendored `med_sam` as a local dependency, and validate the setup with `uv lock` plus package-specific sync and smoke-test commands.

**Tech Stack:** `uv`, PEP 621 `pyproject.toml`, FastAPI, Ultralytics, Jupyter, PyTorch

---

### Task 1: Add the workspace root

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Modify: `.gitignore`

**Step 1: Define the workspace root metadata**

Add a root `pyproject.toml` that:

- names the workspace
- sets `requires-python`
- sets `tool.uv.package = false`
- lists each workspace member
- declares conflicts involving `angio-sam-vmnet`

**Step 2: Add a repo Python pin**

Create `.python-version` with the preferred local interpreter.

**Step 3: Ignore the shared workspace environment**

Add `.venv/` to `.gitignore`.

**Step 4: Verify the workspace file parses**

Run: `uv lock`

Expected: a `uv.lock` file is created successfully.

### Task 2: Convert each active Python area into a uv member

**Files:**
- Create: `demo-app/backend/pyproject.toml`
- Create: `models/yolov8-stenosis/pyproject.toml`
- Create: `models/yolo26x/pyproject.toml`
- Create: `models/yolo26m_cadica/pyproject.toml`
- Create: `output/jupyter-notebook/pyproject.toml`

**Step 1: Translate existing dependency files into `pyproject.toml`**

Keep these members as non-packaged applications via `tool.uv.package = false`.

**Step 2: Preserve test dependencies where needed**

Use `dependency-groups.dev` for pytest/httpx style dependencies.

**Step 3: Verify package-specific sync**

Run:

```bash
uv sync --package angio-demo-backend --group dev
uv sync --package angio-jupyter-notebook
uv sync --package angio-yolo26m-cadica --group dev
uv sync --package angio-yolo26x --group dev
uv sync --package angio-yolov8-stenosis
```

Expected: each command resolves without lockfile or packaging errors.

### Task 3: Make SAM-VMNet uv-aware without forcing all members to match it

**Files:**
- Create: `models/sam_vmnet/pyproject.toml`
- Create: `models/sam_vmnet/med_sam/pyproject.toml`

**Step 1: Add a buildable `med_sam` project file**

Represent the vendored package with a modern `pyproject.toml`.

**Step 2: Add the `sam_vmnet` member**

Use a path dependency on `medsam`, keep the member non-packaged, and push specialized notebook support into optional dependencies while keeping the default environment locally lockable.

**Step 3: Verify the base sync path**

Run:

```bash
uv sync --package angio-sam-vmnet
```

Expected: base dependency resolution succeeds on the current machine, even if the legacy GPU stack remains optional.

### Task 4: Smoke-test the migrated members

**Files:**
- Test: `demo-app/backend/tests/test_api.py`
- Test: `models/yolo26x/tests/test_train_defaults.py`
- Test: `models/yolo26m_cadica/tests/test_cadica_selected.py`

**Step 1: Backend tests**

Run: `uv run --package angio-demo-backend --group dev pytest demo-app/backend/tests -q`

Expected: backend tests pass.

**Step 2: YOLO26x tests**

Run: `uv run --package angio-yolo26x --group dev pytest models/yolo26x/tests -q`

Expected: unit tests pass.

**Step 3: YOLO26m CADICA tests**

Run: `uv run --package angio-yolo26m-cadica --group dev pytest models/yolo26m_cadica/tests -q`

Expected: unit tests pass.
