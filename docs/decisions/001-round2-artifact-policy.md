# ADR-001: Dataset hiện hành và chính sách artifact

## Status

Accepted

## Date

2026-07-16

## Context

BTC thay bộ input trong khi workspace còn dữ liệu, ZIP và output của lần nộp
trước. Các artifact GPU lớn không phù hợp với Git, nhưng code/config và lịch sử
thử nghiệm phải tái lập được. Máy local không có NVIDIA GPU.

## Decision

- `VAI_NVS_DATA_ROUND2` là dataset hiện hành; CSV và COLMAP là nguồn dữ liệu chuẩn.
- Repo GitHub chuyển sang private. Raw input, yêu cầu gốc, output, checkpoint và
  ZIP không commit; manifest/hash, code, config, tài liệu và ledger phải commit.
- Chỉ giữ `outputs/candidate`, `outputs/best`, vùng staging tự dọn và một
  `submission.zip` ở root.
- `experiments/submission_history.jsonl` là ledger append-only duy nhất. Chỉ
  phản hồi BTC hoàn chỉnh mới tham gia xếp hạng best trong cùng dataset.
- Máy Windows làm audit/test/package; máy Ubuntu GPU chạy image Nerfstudio được
  pin bằng digest và clone bằng deploy key read-only.
- Không hỗ trợ macOS.

## Alternatives considered

- Giữ repo public: bị loại vì source thử nghiệm và metadata cuộc thi không cần
  công khai; private repo vẫn clone tự động bằng deploy key.
- Commit dữ liệu bằng Git LFS: bị loại vì dung lượng, chi phí clone và vì dữ liệu
  BTC không cần xuất hiện trong lịch sử Git.
- Giữ ZIP theo timestamp: bị loại vì gây tràn artifact và khó xác định bản nộp
  hiện hành.

## Consequences

- Mỗi VM phải có key riêng và key phải bị thu hồi khi VM bị hủy.
- Việc chuyển private không thu hồi các clone được tạo khi repo từng public.
- Manifest/hash và commit SHA trở thành bằng chứng nối output local với source.
- Repo private trên GitHub plan hiện tại trả về HTTP 403 khi cấu hình branch
  protection/rulesets. Vì vậy maintainer phải kiểm tra CI xanh thủ công trước merge;
  muốn GitHub enforce required checks thì cần nâng GitHub plan, không chuyển repo public.
