#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This worker supports Linux only." >&2
  exit 1
fi
if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "This worker requires x86-64." >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"

compose_file='infra/gpu/docker-compose.yml'

nvidia-smi
docker version
docker compose version
docker compose -f "$compose_file" build nvs
docker compose -f "$compose_file" run --rm nvs python -c \
  'import os, pwd, shutil, cv2, lpips, skimage, torch, bts_nvs; runtime_user=pwd.getpwuid(os.getuid()).pw_name; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.version.cuda == "11.8", f"Unexpected CUDA runtime: {torch.version.cuda}"; assert torch.__version__.startswith("2.1.2"), f"Unexpected torch: {torch.__version__}"; total=torch.cuda.get_device_properties(0).total_memory; assert total >= 23 * 1024**3, f"GPU has only {total / 1024**3:.1f} GiB"; assert shutil.which("ns-train"), "ns-train missing"; assert shutil.which("ns-render"), "ns-render missing"; print(runtime_user, bts_nvs.__version__, torch.__version__, torch.version.cuda, cv2.__version__, skimage.__version__, torch.cuda.get_device_name(0), f"{total / 1024**3:.1f} GiB")'

docker image inspect bts-nvs-gpu:nerfstudio-1.1.5 --format 'derived_image_id={{.Id}}'
