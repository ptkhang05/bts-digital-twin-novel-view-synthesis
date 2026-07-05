[Trang chủ](https://competition.viettel.vn/) | [Đề bài](https://competition.viettel.vn/contests) | [Bảng xếp hạng](https://competition.viettel.vn/leaderboards) | [Diễn đàn](https://competition.viettel.vn/forum) | [Thể lệ](https://competition.viettel.vn/rules)

[Quay lại đề bài](https://competition.viettel.vn/contests/var-2026)

Vòng 1 Đang mở

# Vòng 1 - Sơ loại

02/07/2026 - 30/07/2026

[Lịch sử nộp bài](https://competition.viettel.vn/contests/var-2026/phases/019e649f-4e43-75ab-9a7b-bd1459cd7c06/submissions) | [Nộp bài](https://competition.viettel.vn/contests/var-2026/phases/019e649f-4e43-75ab-9a7b-bd1459cd7c06/submit)

## Đề bài & Quy định

## 1. Mô tả vòng thi

Đây là vòng thi đầu tiên của bài thi **VAR 2026 - Digital Twin cho trạm BTS**.

Ở vòng này, ban tổ chức công bố tập public set và private test #1 gồm các scenes khác nhau.
Thí sinh xây dựng pipeline và đánh giá trên tập public set.
Sau khi công bố tập private test #1, thí sinh sử dụng các ảnh training của mỗi scene để thực hiện sinh ảnh RGB tại các pose mục tiêu được yêu cầu trong file `test_pose.csv`.

---

## 2. Dữ liệu vòng 1

| Hạng mục | Thông tin |
| --- | --- |
| Số ảnh/scene | 150 - 300 ảnh RGB |
| Số poses mục tiêu/scene | 40 - 70 |
| Dung lượng | 200 - 300 MB |

Cấu trúc dữ liệu giống như đã mô tả trong đề bài chính (xem mục **2.3 Cấu trúc dữ liệu**).

---

## 3. Yêu cầu submission

Thí sinh nộp **một file nén** chứa toàn bộ ảnh sinh, theo cấu trúc:

```
submission_round1.zip
├── scene_001/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── scene_002/
│   ├── 0001.png
│   └── ...
└── ...
```

Yêu cầu:

- **Kích thước ảnh:** đúng theo width, height trong `test_pose.csv`
- **Tên file:** theo image_name trong `test_pose.csv`
- **Đầy đủ:** thiếu ảnh tại bất kỳ pose nào của bất kỳ scene nào sẽ ảnh hưởng đến kết quả

---

## 4. Timeline vòng 1

| Mốc thời gian | Sự kiện |
| --- | --- |
| `02/07/2026` | Công bố private test #1 - thí sinh tải dữ liệu |
| `30/07/2026` | **Deadline submission** |

> Thí sinh có thể submit nhiều lần trong thời gian mở. Hệ thống ghi nhận **bản submit cuối cùng** trước deadline.

---

## 5. Lưu ý riêng cho vòng 1

- Đây là vòng làm quen với dữ liệu thực tế - hãy kiểm tra kỹ pipeline trên dữ liệu training public trước khi chạy trên private test
- Hạ tầng huấn luyện do thí sinh tự chuẩn bị. Hãy ước lượng thời gian chạy để đảm bảo kịp deadline
- Cấu hình tham khảo cho mỗi job inference: 1 × RTX A4000 (20 GB VRAM), 4–8 CPU cores, 16–32 GB RAM
- Mọi thắc mắc về dữ liệu hoặc submission liên hệ kênh hỗ trợ chính thức của ban tổ chức

**Chúc thí sinh thi tốt!**

## Chi tiết vòng thi

| Hạng mục | Thông tin |
| --- | --- |
| Loại bài nộp | Tệp ZIP |
| Hạ tầng chấm | GPU |
| Giới hạn nộp bài | 5 lần/ngày |
| Thời gian chờ | 600 giây |

## Dữ liệu công khai

[Tải dữ liệu](https://drive.google.com/file/d/15muyDAfU1SqxpVgLWD4tkMa2wKJGjqiv/view?usp=sharing)

## Tập đoàn Công nghiệp - Viễn thông Quân đội

Lô D26, Khu đô thị mới Cầu Giấy, Phường Cầu Giấy, Hà Nội, Việt Nam

### Theo dõi Viettel

- [Facebook](https://www.facebook.com/ViettelCareers)
- [YouTube](https://www.youtube.com/@viettelcareers2728)
- [LinkedIn](https://www.linkedin.com/company/viettel-group/)
- [Community Group](https://www.facebook.com/groups/viettelairace)

### Về cuộc thi

- [Thể lệ](https://competition.viettel.vn/rules)
- [Đề bài](https://competition.viettel.vn/contests)
- [Bảng xếp hạng](https://competition.viettel.vn/leaderboards)

### Hỗ trợ

- [Diễn đàn](https://competition.viettel.vn/forum)

© 2026 Tập đoàn Công nghiệp - Viễn thông Quân đội. Bảo lưu mọi quyền.
