# ADR-002: Dùng repository public cho GPU worker

## Status

Accepted

## Date

2026-07-19

## Context

ADR-001 chọn repository private và deploy key read-only cho mỗi GPU VM. Trong
quá trình vận hành thực tế, người dùng quyết định chuyển repository sang public
để máy mới có thể clone source đã commit qua HTTPS mà không phải tạo, cài và thu
hồi key cho từng VM. GitHub xác nhận repository đang ở visibility `PUBLIC` ngày
2026-07-19.

Raw input BTC, yêu cầu gốc, output, checkpoint, log và `submission.zip` đã được
Git ignore và không cần xuất hiện trong repository. Chuyển visibility chỉ áp
dụng cho code, config, manifest/hash, ledger dẫn xuất, tài liệu và CI.

## Decision

- Giữ repository GitHub ở chế độ public.
- GPU VM clone read-only qua HTTPS và checkout đúng commit SHA.
- Không cài PAT, phiên `gh auth`, deploy key hoặc private key của máy local lên
  GPU VM.
- Tiếp tục chặn raw input, yêu cầu gốc, output, checkpoint, log, ZIP và secret
  khỏi Git; chỉ stage đường dẫn rõ ràng và kiểm tra `git ls-files` trước merge.

ADR này chỉ thay thế phần visibility/deploy-key của ADR-001. Dataset hiện hành,
cấu trúc artifact, ledger append-only, phân vai Windows/Linux và việc không hỗ
trợ macOS trong ADR-001 vẫn giữ nguyên.

## Alternatives considered

### Private repository với deploy key

- Ưu điểm: source chỉ hiện với người được cấp quyền.
- Nhược điểm: mỗi VM cần vòng đời key riêng; thao tác thêm dễ sai và không cần
  thiết cho quy trình clone read-only mà người dùng đã chọn.
- Không chọn: chi phí vận hành lớn hơn lợi ích trong phạm vi source hiện tại.

### Public repository kèm raw artifact

- Ưu điểm: VM có thể lấy mọi thứ từ một nơi.
- Nhược điểm: làm lộ hoặc phình lịch sử Git bằng dữ liệu, output và submission.
- Không chọn: raw artifact vẫn phải giữ ngoài Git.

## Consequences

- VM mới có thể clone source qua HTTPS mà không cần secret GitHub.
- Code, manifest/hash, ledger dẫn xuất và tài liệu trở thành nội dung công khai.
- Việc chuyển public không làm an toàn cho secret đã từng commit; kiểm tra file
  tracked và secret scan vẫn là quality gate bắt buộc.
- Nếu sau này đổi lại private, phải tạo một ADR mới và khôi phục cơ chế clone có
  xác thực phù hợp thay vì sửa hoặc xóa lịch sử quyết định này.
