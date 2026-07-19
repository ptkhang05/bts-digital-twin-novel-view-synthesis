#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This worker supports Linux only." >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"

compose_file='infra/gpu/docker-compose.yml'
processed='processed/smoke/chair'
training_output='outputs/.staging/chair-smoke-training'
render_root='outputs/.staging/chair-smoke-rendered'

cleanup() {
  rm -rf -- "$training_output" "$render_root"
}
trap cleanup EXIT

bash infra/gpu/verify_gpu.sh

docker compose -f "$compose_file" run --rm nvs \
  python -m bts_nvs.prepare \
  --scene VAI_NVS_DATA_ROUND2/chair \
  --out "$processed" \
  --copy-mode copy \
  --overwrite

docker compose -f "$compose_file" run --rm nvs \
  python -m bts_nvs.train \
  --scene "$processed" \
  --preset quality \
  --output-dir "$training_output" \
  --experiment-name chair-smoke \
  -- --max-num-iterations 500 --viewer.quit-on-train-completion True

config_path="$(find "$training_output" -type f -name config.yml -print -quit)"
if [[ -z "$config_path" ]]; then
  echo "Smoke training produced no config.yml" >&2
  exit 1
fi

docker compose -f "$compose_file" run --rm nvs \
  python -m bts_nvs.render \
  --checkpoint "$config_path" \
  --targets "$processed/target_cameras.json" \
  --out "$render_root/chair" \
  --distortion auto

docker compose -f "$compose_file" run --rm nvs \
  python -m bts_nvs.validate_submission \
  --data-root VAI_NVS_DATA_ROUND2/chair \
  --submission "$render_root"

echo "GPU smoke passed: CUDA, ns-train 500 iterations, target render and strict dimensions/JPEG validation."
