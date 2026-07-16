# Ubuntu GPU worker

Chỉ dùng Ubuntu 22.04 x86-64. Không dùng hoặc hỗ trợ macOS.

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

## Clone private repo

Tạo key riêng ngay trên VM:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/bts_nvs_deploy -C bts-nvs-gpu-vm -N ''
cat ~/.ssh/bts_nvs_deploy.pub
```

Thêm public key vào repository dưới dạng deploy key **read-only**. Xác nhận host
key GitHub theo [SSH key fingerprints chính thức](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints), rồi clone đúng commit:

```bash
export GIT_SSH_COMMAND='ssh -i ~/.ssh/bts_nvs_deploy -o IdentitiesOnly=yes'
git clone git@github.com:ptkhang05/bts-digital-twin-novel-view-synthesis.git
cd bts-digital-twin-novel-view-synthesis
git checkout '<exact-commit-sha>'
mkdir -p processed outputs/candidate outputs/best outputs/.staging
```

Không copy PAT, phiên `gh auth` hoặc private key máy local lên VM. Thu hồi deploy
key trên GitHub khi hủy VM.

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
bash infra/gpu/smoke_chair.sh 2>&1 | tee processed/chair-smoke.log
```

Script này chạy chính derived Compose image với `--gpus all --ipc=host`, kiểm tra
`nvidia-smi`, PyTorch CUDA, `ns-train`, `ns-render`, VRAM; train `chair` 500
iteration; render target; rồi strict-validate tên, RGB JPEG và kích thước. Bất kỳ
CUDA/render/dimension/OOM nào đều chặn thí nghiệm tiếp theo. `outputs/.staging`
được dọn bằng trap cả khi thành công lẫn lỗi.

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
      -- --max-num-iterations 10000 --viewer.quit-on-train-completion True
  done
done
```

Với mỗi config đã train, render `processed/holdout/<scene>/holdout_cameras.json`
vào `outputs/.staging/holdout-<preset>/rendered/<scene>` bằng
`python -m bts_nvs.render --distortion auto`. Sau đủ 7 scene:

```bash
docker compose -f infra/gpu/docker-compose.yml run --rm nvs \
  python -m bts_nvs.score_submission \
  --data-root processed/holdout-ground-truth \
  --submission outputs/.staging/holdout-quality/rendered \
  --psnr-max 50 \
  --out outputs/.staging/holdout-quality/metrics.json
```

Lặp cho `quality-aa`. Chọn equal-scene proxy cao hơn; hòa thì giữ `quality`
(`classic`). Đây là proxy local, không phải điểm BTC.

## Train full và render candidate

Chuẩn bị 100% train với manifest provenance, rồi train preset thắng 30.000 iteration
cho cả 7 scene. Pose normalization vẫn tắt; distortion là `auto`; JPEG quality 95:

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
