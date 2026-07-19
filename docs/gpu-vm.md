# Ubuntu GPU worker

Dùng host Ubuntu 22.04 hoặc 24.04 x86-64. Vòng đầu đã chạy thành công trên
Ubuntu 24.04.3; môi trường train bên trong image vẫn được pin ở Ubuntu
22.04/CUDA 11.8. Không dùng hoặc hỗ trợ macOS.

## Cấu hình và điều kiện dừng

- NVIDIA GPU tối thiểu 24 GB VRAM; 8 vCPU, 32 GB RAM và 100 GB NVMe persistent.
- Đây là ước tính kỹ thuật, chưa phải benchmark đã xác nhận. `smoke_chair.sh` là
  gate bắt buộc.
- Nếu 24 GB OOM, giảm tải rasterization/batch trước; nếu vẫn OOM thì dùng GPU 48 GB.
- Không mở viewer port ra Internet. Chỉ chạy headless hoặc SSH tunnel.

Image nền được pin tuyệt đối:

```text
ghcr.io/nerfstudio-project/nerfstudio:1.1.5@sha256:b59b8e1012d7a43679d3234b3de9c8416a4b8435fcbf21b9d8c4494b8563f19e
```

Image này dùng Ubuntu 22.04, Python 3.10, CUDA 11.8 và Torch 2.1.2+cu118.
Layer metric bổ sung được khóa tại `infra/gpu/constraints-metrics.txt`. Ledger phải
ghi `derived_image_id` thực tế do `verify_gpu.sh` in ra; tag local không được xem
là digest tái lập. Docker build tải sẵn AlexNet/LPIPS weight chỉ để tính metric;
weight này không được truyền vào train hoặc render và phải được khai báo minh bạch
vì quy định BTC về pretrained metric weight chưa hoàn toàn rõ.

## Bootstrap máy sạch

Chọn VM có NVIDIA driver do nhà cung cấp cài sẵn và tương thích CUDA 11.8. Sau
khi SSH vào VM:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg git python3-venv
nvidia-smi
```

Cài Docker Engine + Compose plugin theo
[hướng dẫn Ubuntu chính thức](https://docs.docker.com/engine/install/ubuntu/), sau
đó cài NVIDIA Container Toolkit và cấu hình Docker theo
[hướng dẫn NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker version
docker compose version
```

Không cài CUDA toolkit, Nerfstudio hoặc COLMAP trực tiếp lên host; chúng nằm trong
container. Không chạy COLMAP lại.

## Clone public repo

Repo hiện hành là public. Clone read-only qua HTTPS rồi checkout đúng commit:

```bash
git clone https://github.com/ptkhang05/bts-digital-twin-novel-view-synthesis.git
cd bts-digital-twin-novel-view-synthesis
git checkout '<exact-commit-sha>'
mkdir -p processed outputs/candidate outputs/best outputs/.staging
```

Không cấu hình PAT, phiên `gh auth`, deploy key hoặc private key máy local trên
VM. Raw input/output tiếp tục chỉ tồn tại local và bị Git ignore.

## Tải, extract an toàn và xác minh input

Tạo bootstrap venv CPU trên host:

```bash
python3 -m venv .bootstrap-venv
.bootstrap-venv/bin/python -m pip install pip==26.0.1
.bootstrap-venv/bin/python -m pip install \
  -c requirements/constraints-cpu.txt -e '.[metrics]'
.bootstrap-venv/bin/python -m pip install -r infra/gpu/requirements-bootstrap.txt
```

Tải đúng archive mới, sau đó dùng ingestor của repo. Ingestor chỉ extract những
member khớp manifest, chặn traversal/link/duplicate/ZIP bomb, audit toàn bộ staging,
atomic-promote và **chỉ xóa archive sau khi audit thành công**:

```bash
.bootstrap-venv/bin/gdown 1b9F4B1tDVX8bIX4fZxsP9bduRynDUN_a \
  -O VAI_NVS_DATA_ROUND2.zip
.bootstrap-venv/bin/python -m bts_nvs.ingest \
  --archive VAI_NVS_DATA_ROUND2.zip \
  --data-root VAI_NVS_DATA_ROUND2 \
  --manifest manifests/vai_nvs_round2.json
```

Nếu manifest không khớp, lệnh dừng, giữ archive và không thay data root. Không tải
dữ liệu cũ.

## Build và GPU smoke bắt buộc

```bash
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
bash infra/gpu/smoke_chair.sh 2>&1 | tee processed/chair-smoke.log
```

Script này chạy chính derived Compose image với `--gpus all --ipc=host`, kiểm tra
`nvidia-smi`, PyTorch CUDA, `ns-train`, `ns-render`, VRAM; train `chair` 500
iteration; render target; rồi strict-validate tên, RGB JPEG và kích thước. Bất kỳ
CUDA/render/dimension/OOM nào đều chặn thí nghiệm tiếp theo. `outputs/.staging`
được dọn bằng trap cả khi thành công lẫn lỗi. Giữ `HOST_UID`/`HOST_GID` trong
shell hiện tại cho các lệnh Compose tiếp theo để artifact thuộc user host.

## Holdout classic so với antialiased

Tạo split cố định bằng filename sort lexicographic; vị trí 10, 20, 30, ... làm
holdout. Vì raw data và `processed` là hai bind mount khác nhau, dùng `copy` trên VM:

```bash
docker compose -f infra/gpu/docker-compose.yml run --rm nvs \
  python -m bts_nvs.holdout \
  --data-root VAI_NVS_DATA_ROUND2 \
  --processed-root processed/holdout \
  --ground-truth-root processed/holdout-ground-truth \
  --interval 10 --copy-mode copy --overwrite
```

Với từng scene, chạy tuần tự hai preset chỉ khác rasterization:

```bash
scenes=(bonsai chair HCM0421 HCM0539 HCM0540 HCM0644 HCM0674)
for preset in quality quality-aa; do
  for scene in "${scenes[@]}"; do
    docker compose -f infra/gpu/docker-compose.yml run --rm nvs \
      python -m bts_nvs.train \
      --scene "processed/holdout/$scene" \
      --preset "$preset" \
      --output-dir "outputs/.staging/holdout-$preset/training" \
      --experiment-name "$scene" \
      -- --machine.seed 42 --max-num-iterations 10000 \
      --viewer.quit-on-train-completion True
  done
done
```

Với mỗi config đã train, render `processed/holdout/<scene>/holdout_cameras.json`
vào `outputs/.staging/holdout-<preset>/rendered/<scene>` bằng
`python -m bts_nvs.render --distortion auto`. Sau đủ 7 scene, lưu hai kết quả
so sánh ra `processed` trước khi dọn staging:

```bash
mkdir -p processed/holdout-comparison

docker compose -f infra/gpu/docker-compose.yml run --rm nvs \
  python -m bts_nvs.score_submission \
  --data-root processed/holdout-ground-truth \
  --submission outputs/.staging/holdout-quality/rendered \
  --psnr-max 50 \
  --out processed/holdout-comparison/classic.json

docker compose -f infra/gpu/docker-compose.yml run --rm nvs \
  python -m bts_nvs.score_submission \
  --data-root processed/holdout-ground-truth \
  --submission outputs/.staging/holdout-quality-aa/rendered \
  --psnr-max 50 \
  --out processed/holdout-comparison/antialiased.json
```

Chọn equal-scene proxy cao hơn; hòa thì giữ `quality` (`classic`). Đây là proxy
local, không phải điểm BTC.

### Hiệu chuẩn holdout 30k sau feedback đầu tiên

Submission đầu tiên dùng 30.000 iteration nhưng A/B ban đầu chỉ dùng holdout
10.000 iteration. Trước khi tune model, hiệu chuẩn lại baseline bằng cách chỉ đổi
`max-num-iterations` từ 10.000 thành 30.000; giữ nguyên split, seed, preset
`quality`, pose normalization, distortion và JPEG quality. Kết quả 10k incumbent
là `0.6955825215814241` theo equal-scene proxy.

Không xóa checkpoint 30k sau khi chấm vì đây là baseline cho A/B tiếp theo:

```bash
set -euo pipefail
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"

test -f processed/holdout-comparison/classic.json
test "$(find processed/holdout-ground-truth -type f -iname '*.jpg' | wc -l)" -eq 164
if [[ -d outputs/.staging/holdout-quality-30k ]] \
  && find outputs/.staging/holdout-quality-30k -mindepth 1 -print -quit | grep -q .; then
  echo "30k holdout staging is not empty; stop to avoid mixing runs" >&2
  exit 1
fi

scenes=(bonsai chair HCM0421 HCM0539 HCM0540 HCM0644 HCM0674)
for scene in "${scenes[@]}"; do
  docker compose -f infra/gpu/docker-compose.yml run --rm nvs \
    python -m bts_nvs.train \
    --scene "processed/holdout/$scene" \
    --preset quality \
    --output-dir outputs/.staging/holdout-quality-30k/training \
    --experiment-name "$scene" \
    --log-file "processed/holdout-quality-30k-$scene-train.log" \
    -- --machine.seed 42 --max-num-iterations 30000 \
    --viewer.quit-on-train-completion True
done

test "$(find outputs/.staging/holdout-quality-30k/training \
  -type f -name 'step-000029999.ckpt' | wc -l)" -eq 7

if grep -nE 'Traceback|ExternalCommandError|CUDA out of memory|PermissionError' \
  processed/holdout-quality-30k-*-train.log; then
  echo "30k holdout logs contain errors; stop before rendering" >&2
  exit 1
fi

for scene in "${scenes[@]}"; do
  config="$(find "outputs/.staging/holdout-quality-30k/training/$scene" \
    -type f -name config.yml -print | sort | tail -1)"
  test -n "$config"
  docker compose -f infra/gpu/docker-compose.yml run --rm nvs \
    python -m bts_nvs.render \
    --checkpoint "$config" \
    --targets "processed/holdout/$scene/holdout_cameras.json" \
    --out "outputs/.staging/holdout-quality-30k/rendered/$scene" \
    --distortion auto
done

docker compose -f infra/gpu/docker-compose.yml run --rm nvs \
  python -m bts_nvs.score_submission \
  --data-root processed/holdout-ground-truth \
  --submission outputs/.staging/holdout-quality-30k/rendered \
  --psnr-max 50 \
  --out processed/holdout-comparison/classic-30k.json
```

So sánh đúng hai JSON cùng split và cùng công thức:

```bash
.bootstrap-venv/bin/python - <<'PY'
import json
from pathlib import Path

baseline = json.loads(Path("processed/holdout-comparison/classic.json").read_text())
challenger = json.loads(Path("processed/holdout-comparison/classic-30k.json").read_text())
expected_counts = {
    "bonsai": 24,
    "chair": 20,
    "HCM0421": 24,
    "HCM0539": 24,
    "HCM0540": 24,
    "HCM0644": 24,
    "HCM0674": 24,
}
def scene_counts(payload):
    return {row["scene"]: row["count"] for row in payload["scenes"]}


assert baseline["aggregate"]["count"] == challenger["aggregate"]["count"] == 164
assert scene_counts(baseline) == scene_counts(challenger) == expected_counts
assert baseline["scoring"] == challenger["scoring"]
delta = challenger["aggregate"]["score"] - baseline["aggregate"]["score"]
print(f"classic_10k={baseline['aggregate']['score']:.12f}")
print(f"classic_30k={challenger['aggregate']['score']:.12f}")
print(f"delta={delta:+.12f} ({delta * 100:+.6f} leaderboard points)")
PY
```

Đây chỉ là calibration, không tạo submission mới. Nếu 30k thắng, dùng JSON 30k
làm incumbent local cho thí nghiệm kế tiếp: giữ sáu scene classic và A/B riêng
`bonsai` với rasterization `classic`/`antialiased` ở 30k, cùng seed. Nếu 30k
không thắng, ghi kết quả rồi dừng: không tự động full-train 10k, không render
target và không thay ZIP. Khi đó phải lập một protocol một-biến mới trước khi
chạy tiếp. Dữ liệu 10k hiện có dự báo hybrid bonsai chỉ tăng khoảng `0.0363`
điểm toàn bài, nên phải xác nhận lại ở 30k trước khi retrain full hoặc thay
`submission.zip`.

## Train full và render candidate

Đây là quy trình tạo candidate sau khi một challenger đã thắng protocol holdout
được ghi trong ledger; calibration 30k ở trên tự nó không mở gate này. Baseline
đầu tiên đã dùng preset `quality` 30.000 iteration. Mọi candidate sau phải dùng
đúng cấu hình/iterations thắng thí nghiệm một-biến đã định trước cho cả 7 scene.
Pose normalization vẫn tắt; distortion là `auto`; JPEG quality 95:

```bash
docker compose -f infra/gpu/docker-compose.yml run --rm nvs \
  python -m bts_nvs.prepare_dataset \
  --root VAI_NVS_DATA_ROUND2 \
  --out processed/full \
  --copy-mode copy \
  --dataset-id vai_nvs_round2 \
  --manifest manifests/vai_nvs_round2.json \
  --overwrite
```

Train output đặt dưới `outputs/candidate/training/<scene>`; render đúng
`processed/full/<scene>/target_cameras.json` vào
`outputs/candidate/rendered/<scene>`. Không ghi candidate khác hoặc ZIP timestamp.

Container chỉ mount source/manifest/data read-only; quyền ghi chỉ cấp cho
`processed/` và `outputs/`. `.git`, raw yêu cầu BTC và ledger không được đưa vào
container. Sau render, dùng bootstrap venv trên host để tạo ZIP root duy nhất:

```bash
.bootstrap-venv/bin/python -m bts_nvs.submit \
  --data-root VAI_NVS_DATA_ROUND2 \
  --submission outputs/candidate/rendered \
  --out submission.zip
```

Ghi exact Git commit, base image digest, `derived_image_id`, config/command, seed,
iterations, distortion, JPEG quality, GPU/runtime và metric từng scene vào feedback
ledger. Mỗi vòng sau chỉ đổi một biến; chỉ nộp candidate vượt best trên holdout.
