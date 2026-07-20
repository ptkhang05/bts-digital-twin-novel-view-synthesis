# BTS Digital Twin — Novel View Synthesis

Pipeline tái lập cho bộ dữ liệu hiện hành `VAI_NVS_DATA_ROUND2`: audit input,
chuẩn bị camera COLMAP cho Nerfstudio, train/render có xử lý distortion, chấm
holdout theo từng scene, tạo đúng một `submission.zip` và lưu phản hồi BTC trong
ledger append-only.

Nguồn vận hành chuẩn là [đặc tả BTC hiện hành](docs/current-btc-spec.md). Chính
sách dữ liệu, Git và artifact được ghi tại
[ADR-001](docs/decisions/001-round2-artifact-policy.md) và quyết định visibility
hiện hành tại [ADR-002](docs/decisions/002-public-repository.md).
Trạng thái phục hồi sau sự cố ổ D được ghi tại
[báo cáo recovery 2026-07-20](docs/recovery-2026-07-20.md).

## Nền tảng hỗ trợ

- Local: Windows, Python 3.13 trong `.venv`, CPU-only.
- Train/render: host Ubuntu 22.04/24.04 x86-64 với NVIDIA GPU; container được
  pin trên Ubuntu 22.04/CUDA 11.8.
- Không hỗ trợ macOS.

Máy Windows không cần CUDA, Nerfstudio GPU, COLMAP hoặc Node. Sparse
reconstruction đã có trong input BTC và không được chạy COLMAP lại.

## Cài đặt local Windows

Dependency CPU được khóa phiên bản. `TORCH_HOME` được đặt trong workspace để
LPIPS/AlexNet không ghi cache ra ngoài folder dự án:

```powershell
$env:TORCH_HOME = (Join-Path $PWD ".cache\torch")
.\.venv\Scripts\python.exe -m pip install pip==26.0.1
.\.venv\Scripts\python.exe -m pip install `
  --extra-index-url https://download.pytorch.org/whl/cpu `
  -c requirements\constraints-cpu.txt `
  -c requirements\constraints-perceptual-cpu.txt `
  -e ".[dev,metrics,perceptual]"
.\.venv\Scripts\python.exe -m pip check
```

Weight AlexNet pretrained chỉ được dùng để tính LPIPS, không được dùng để train
hoặc sinh ảnh. Lần tính metric đầu tiên có thể tải weight vào `.cache\torch`; thư
mục này bị Git ignore. Không dùng editable install global.

## Dataset contract đã xác minh

`manifests/vai_nvs_round2.json` khóa các bất biến sau:

- 7 scene, 1.653 ảnh train và 386 target.
- 5 scene HCM dùng `SIMPLE_RADIAL`; distortion lấy từ `cameras.bin`.
- Tổng 298 registered pose dư bị loại bằng tập tên ảnh train chính xác.
- Tên, case và kích thước target lấy từ `test/test_poses.csv`.

Mỗi scene có cấu trúc:

```text
scene_id/
  train/images/*.[jJ][pP][gG]
  train/sparse/0/cameras.bin
  train/sparse/0/images.bin
  train/sparse/0/points3D.bin
  test/test_poses.csv
```

CSV và manifest là nguồn chuẩn khi ví dụ trong văn bản BTC mâu thuẫn với input.

## Cài input mới và audit

Nếu input được tải dưới dạng ZIP, dùng ingestor an toàn. Lệnh chỉ nhận đúng member
trong manifest, audit staging, atomic-promote và chỉ xóa archive sau khi thành
công:

```powershell
.\.venv\Scripts\python.exe -m bts_nvs.ingest `
  --archive VAI_NVS_DATA_ROUND2.zip `
  --data-root VAI_NVS_DATA_ROUND2 `
  --manifest manifests\vai_nvs_round2.json
```

Kiểm tra lại input hiện có mà không sửa manifest:

```powershell
.\.venv\Scripts\python.exe -m bts_nvs.audit `
  --data-root VAI_NVS_DATA_ROUND2 `
  --check-manifest manifests\vai_nvs_round2.json
```

Chỉ dùng `--manifest manifests\vai_nvs_round2.json` khi chủ ý sinh lại manifest
từ một dataset đã được xác minh độc lập.

## Holdout local proxy

Holdout cố định được chọn sau khi sort filename lexicographic; các vị trí có
`(index + 1) mod 10 = 0` làm ground truth:

```powershell
.\.venv\Scripts\python.exe -m bts_nvs.holdout `
  --data-root VAI_NVS_DATA_ROUND2 `
  --processed-root processed\holdout `
  --ground-truth-root processed\holdout-ground-truth `
  --interval 10 `
  --copy-mode hardlink `
  --overwrite
```

Train 7 scene với `quality` và `quality-aa`, render từng
`processed/holdout/<scene>/holdout_cameras.json`, rồi chấm toàn submission:

```powershell
$env:TORCH_HOME = (Join-Path $PWD ".cache\torch")
.\.venv\Scripts\python.exe -m bts_nvs.score_submission `
  --data-root processed\holdout-ground-truth `
  --submission outputs\.staging\holdout-quality\rendered `
  --psnr-max 50 `
  --out processed\holdout-comparison\classic.json
```

Validator chạy trước metric; thiếu/thừa/sai ảnh sẽ bị từ chối. Score tổng là
trung bình đều score của 7 scene. Văn bản BTC không nêu `PSNR_max`, nhưng kết
quả chính thức đầu tiên khớp `PSNR_max=50` tới độ chính xác hiển thị. Đây là xác
nhận thực nghiệm cho vòng hiện tại, không phải một công bố bằng văn bản. Metric
holdout vẫn là proxy local, không được gọi là điểm BTC.

## Chuẩn bị full train, train và render

Chuẩn bị dữ liệu với provenance đã xác minh:

```powershell
.\.venv\Scripts\python.exe -m bts_nvs.prepare_dataset `
  --root VAI_NVS_DATA_ROUND2 `
  --out processed\full `
  --copy-mode hardlink `
  --dataset-id vai_nvs_round2 `
  --manifest manifests\vai_nvs_round2.json `
  --overwrite
```

Train/render thực tế chạy trên máy GPU Linux; xem
[hướng dẫn Ubuntu GPU](docs/gpu-vm.md). Các CLI lõi:

```text
python -m bts_nvs.train --scene <processed-scene> --preset quality ...
python -m bts_nvs.render --checkpoint <config.yml> --targets <target_cameras.json> --out <scene-output> --distortion auto
```

- `quality`: `splatfacto-big`, rasterization `classic`.
- `quality-aa`: cùng cấu hình nhưng `antialiased`.
- `fast`: smoke kỹ thuật 500 iteration, không dùng làm candidate cuối.

Nerfstudio 1.1.5 `splatfacto-big` được tune cho 30.000 iteration và override
model base defaults thành `cull_alpha_thresh=0.005`,
`densify_grad_thresh=0.0005`. Scheduler của means cũng kết thúc ở 30.000 step;
không chỉ tăng `max-num-iterations` vượt 30k mà không thiết kế lại scheduler.

Pose normalization mặc định bị tắt để giữ COLMAP frame. Chỉ dùng
`--debug-allow-pose-normalization` cho debug có chủ ý. `--distortion auto` dùng
exact/redistort cho `SIMPLE_RADIAL` và render thường cho pinhole. Render là
transactional; lỗi giữa chừng không thay output hợp lệ trước đó.

## Tạo ZIP duy nhất

Candidate phải nằm tại `outputs/candidate/rendered/<scene>`:

```powershell
.\.venv\Scripts\python.exe -m bts_nvs.submit `
  --data-root VAI_NVS_DATA_ROUND2 `
  --submission outputs\candidate\rendered `
  --out submission.zip
```

`submit` kiểm tra đủ 7 scene/386 ảnh, exact filename/case/kích thước, RGB JPEG,
duplicate member và CRC. ZIP chỉ được atomic-replace sau khi validation và
SHA-256 thành công. Không tạo ZIP timestamp hoặc đổi đuôi PNG thành `.JPG`.

## Ghi phản hồi BTC và chọn best

Sau khi người dùng trả điểm chính thức, ghi feedback đầy đủ. Các tham số JSON là
JSON inline, không phải đường dẫn file:

```powershell
$metrics = Get-Content -Raw processed\holdout-comparison\classic.json
.\.venv\Scripts\python.exe -m bts_nvs.feedback `
  --submission-id round2-001 `
  --dataset-id vai_nvs_round2 `
  --score <official-score> `
  --data-root VAI_NVS_DATA_ROUND2 `
  --dataset-manifest-sha256 4839983968385ec56a418909bd70c77a310233b328762db1a2f4bde1c7bcadb8 `
  --git-commit "$(git rev-parse HEAD)" `
  --container-image-digest <derived-image-id> `
  --command "<exact-train-and-render-command>" `
  --config-json '{"preset":"quality","rasterization":"classic"}' `
  --metrics-json $metrics `
  --seed 42 `
  --iterations 30000 `
  --distortion auto `
  --jpeg-quality 95 `
  --hardware-json '{"gpu":"<model>","vram_gib":<number>}' `
  --gpu-time-seconds <seconds> `
  --hypothesis "classic baseline" `
  --next-action "change exactly one variable"

.\.venv\Scripts\python.exe -m bts_nvs.ledger best `
  --dataset-id vai_nvs_round2
```

`feedback` xác minh `submission.zip` khớp candidate trước khi thay đổi trạng
thái. Điểm cao hơn được promote; hòa điểm ưu tiên GPU-time thấp hơn, sau đó ZIP
nhỏ hơn; thiếu dữ liệu tie-break thì giữ incumbent. Candidate thua bị xóa và ZIP
được tái tạo từ best. Chỉ so sánh record cùng dataset. Ledger duy nhất là
`experiments/submission_history.jsonl`; không chạy nhiều writer đồng thời.

`python -m bts_nvs.ledger add-feedback` chỉ dùng để nhập feedback lịch sử khi
không còn artifact candidate/best cần chuyển trạng thái.

## Artifact và Git

```text
outputs/
  candidate/  # lần cải thiện đang chờ phản hồi
  best/       # tốt nhất đã được BTC xác nhận
  .staging/   # tạm thời, tự dọn
submission.zip
```

Raw input, raw yêu cầu BTC, output, checkpoint/log và ZIP chỉ giữ local và bị
Git ignore. Code, config, manifest, ledger, tài liệu và CI đi qua branch/PR trên
repo public. VM clone read-only qua HTTPS rồi checkout đúng commit; không cần
deploy key, PAT hay phiên đăng nhập GitHub.

## Quality gates

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m build --no-isolation
.\.venv\Scripts\python.exe -m pip check
```

GitHub Actions chạy `ruff`, `pytest`, build wheel/sdist và `pip check` trên Python
3.10 và 3.13. GPU smoke vẫn phải chạy trên VM vì CI không có CUDA hoặc raw input
BTC.
