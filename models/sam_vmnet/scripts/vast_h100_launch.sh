#!/usr/bin/env bash
set -euo pipefail

# Host-side helper for Vast CLI.
# Requires: vastai CLI configured with API key.

QUERY="${QUERY:-gpu_name~'H100 SXM'}"
IMAGE="${IMAGE:-nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04}"
DISK_GB="${DISK_GB:-120}"
ONSTART_CMD="${ONSTART_CMD:-cd /workspace/Angiography/models/sam_vmnet && bash run_vast_h100_pipeline.sh}"

if ! command -v vastai >/dev/null 2>&1; then
  echo "vastai CLI not found. Install first: pip install vastai"
  exit 1
fi

echo "Searching offers with query: ${QUERY}"
OFFERS_JSON="$(vastai search offers "${QUERY}" --raw)"

OFFER_ID="$(python3 - <<'PY' <<<"${OFFERS_JSON}"
import json, sys
raw = sys.stdin.read().strip()
if not raw:
    print("")
    raise SystemExit(0)
items = json.loads(raw)
if isinstance(items, dict):
    items = items.get("offers", [])
def price(item):
    for key in ("dph_total", "dph", "price", "cost"):
        if key in item:
            try:
                return float(item[key])
            except Exception:
                pass
    return float("inf")
items = sorted(items, key=price)
print(items[0].get("id", "") if items else "")
PY
)"

if [[ -z "${OFFER_ID}" ]]; then
  echo "No offers matched query."
  exit 2
fi

echo "Creating instance from offer: ${OFFER_ID}"
CREATE_OUTPUT="$(vastai create instance "${OFFER_ID}" --image "${IMAGE}" --disk "${DISK_GB}" --ssh --onstart-cmd "${ONSTART_CMD}" --raw || true)"
echo "${CREATE_OUTPUT}"

echo "Tip: list instances with: vastai show instances"
