# Đặc tả vận hành BTC hiện hành

Ngày đối chiếu: 2026-07-16 (Asia/Ho_Chi_Minh).

Tài liệu này là bản dẫn xuất có thể commit. Hai file yêu cầu gốc của BTC và dữ
liệu ảnh được giữ local, không đưa lên Git. Khi văn bản mô tả và dữ liệu thực tế
mâu thuẫn, pipeline dùng thứ tự ưu tiên sau:

1. `test/test_poses.csv` của scene hiện hành cho tên ảnh, kích thước và số target.
2. COLMAP `cameras.bin`/`images.bin` cho camera model, distortion và pose.
3. Văn bản BTC cho công thức chấm, format ZIP và quy định cuộc thi.

## Dataset hiện hành

`VAI_NVS_DATA_ROUND2` gồm 7 scene, 1.653 ảnh train và 386 target:

| Scene | Train | Target | Kích thước | Camera |
| --- | ---: | ---: | --- | --- |
| bonsai | 248 | 28 | 1920×1080 | SIMPLE_PINHOLE |
| chair | 205 | 58 | 720×1280 | SIMPLE_PINHOLE |
| HCM0421 | 240 | 60 | 1320×989 | SIMPLE_RADIAL |
| HCM0539 | 240 | 60 | 1320×989 | SIMPLE_RADIAL |
| HCM0540 | 240 | 60 | 1320×989 | SIMPLE_RADIAL |
| HCM0644 | 240 | 60 | 1320×989 | SIMPLE_RADIAL |
| HCM0674 | 240 | 60 | 1320×989 | SIMPLE_RADIAL |

Năm scene HCM có tổng cộng 298 pose COLMAP đã đăng ký nhưng không thuộc tập
train hoặc target. Chỉ ảnh có tên hiện diện chính xác trong `train/images` được
đưa vào training. Scene `chair` không có `points3D.ply`; `points3D.bin` là nguồn
fallback hợp lệ.

CSV không chứa distortion. Pipeline phải lấy hệ số radial từ `cameras.bin` và
giữ nguyên hoa/thường của tên target (`.jpg` cho bonsai/chair, `.JPG` cho HCM).
Các giá trị `tx,ty,tz` là COLMAP world-to-camera translation, không phải camera
center trong world space.

## Submission và metric

- Nộp đúng một `submission.zip`, chứa trực tiếp `scene_id/image_name`.
- Phải đủ và chỉ đủ 7 scene/386 ảnh RGB JPEG, đúng tên, case và kích thước.
- BTC công bố `0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm`, trong đó
  `PSNR_norm=clamp(PSNR/PSNR_max,0,1)` và điểm cuối là trung bình đều theo scene.
- BTC chưa công bố giá trị `PSNR_max`. Giá trị 50 trong repo chỉ là proxy local
  và mọi báo cáo phải gắn nhãn rõ ràng.
- Bộ target hiện hành không có ground truth. Metric local chỉ được tính trên
  holdout xác định từ ảnh train và không phải điểm BTC.

## Quy định dữ liệu và tái lập

- Không dùng ảnh/video/3D bên ngoài chứa cùng scene; không dò hoặc suy luận
  ground truth test; không chỉnh ảnh thủ công theo từng pose.
- Mã nguồn, config, dependency, checkpoint và log phải đủ để tái lập. LPIPS
  pretrained chỉ được dùng ở bước đo metric, không làm đầu vào train/render.

## Đối chiếu liên kết và mâu thuẫn

- Dataset hiện hành: [Google Drive của BTC](https://drive.google.com/file/d/1b9F4B1tDVX8bIX4fZxsP9bduRynDUN_a/view?usp=sharing).
- [LPIPS](https://arxiv.org/abs/1801.03924), [SSIM DOI](https://doi.org/10.1109/TIP.2003.819861)
  và [3D Gaussian Splatting baseline](https://github.com/graphdeco-inria/gaussian-splatting)
  truy cập được khi kiểm tra.
- Hai logo BTC trả về PNG; `https://competition.viettel.vn/var-2026.jpg` trả về
  HTML nên không được xem là asset ảnh hợp lệ.
- Trang cuộc thi là ứng dụng web và không cung cấp đầy đủ nội dung khi truy cập
  ẩn danh; các endpoint submission/leaderboard cần phiên đăng nhập của người dùng.
- Văn bản vòng ghi `test_pose.csv`, nhưng file thật là `test_poses.csv`; pipeline
  chỉ chấp nhận tên file thật.
- Văn bản ghi 40–70 target nhưng `bonsai` có 28; CSV/manifest thực tế là chuẩn.
- Cụm “RTX A4000 (20 GB)” mâu thuẫn với datasheet: RTX A4000 là 16 GB, còn RTX
  4000 Ada là 20 GB. Cấu hình VM mặc định dùng tối thiểu 24 GB VRAM và xác nhận
  bằng smoke test.

Nguồn kỹ thuật chính thức dùng cho quyết định vận hành:

- [NVIDIA RTX A4000 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-a4000-datasheet.pdf)
  và [RTX 4000 Ada datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-4000-datasheet.pdf).
- [CUDA 11.8 release notes](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/).
- [Nerfstudio installation guide](https://docs.nerf.studio/quickstart/installation.html)
  và [release v1.1.5](https://github.com/nerfstudio-project/nerfstudio/releases/tag/v1.1.5).
- [GitHub repository visibility](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility)
  và [deploy keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys).
