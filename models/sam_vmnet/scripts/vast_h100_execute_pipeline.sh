#!/usr/bin/env bash
set -euo pipefail

# Run the pipeline on an existing Vast instance.
# Usage: ./scripts/vast_h100_execute_pipeline.sh <instance_id>

INSTANCE_ID="${1:-}"
REMOTE_CMD="${REMOTE_CMD:-cd /workspace/Angiography/models/sam_vmnet && bash run_vast_h100_pipeline.sh}"

if [[ -z "${INSTANCE_ID}" ]]; then
  echo "Usage: $0 <instance_id>"
  exit 1
fi

if ! command -v vastai >/dev/null 2>&1; then
  echo "vastai CLI not found. Install first: pip install vastai"
  exit 2
fi

echo "Executing pipeline on instance ${INSTANCE_ID}"
vastai execute "${INSTANCE_ID}" "${REMOTE_CMD}"

echo "Follow logs with: vastai logs ${INSTANCE_ID}"
echo "Get SSH URL with: vastai ssh-url ${INSTANCE_ID}"
