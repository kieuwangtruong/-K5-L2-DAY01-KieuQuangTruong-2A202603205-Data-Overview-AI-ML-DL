# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06-11-2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:** Python 3.10+ / PyTorch 2.x / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1:
   - `class_id` = 468
   - `class_name` = cab
   - `rank` = 1
   - `score` = 0.510915
   - `taxonomy_name` = ImageNet-1K

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   - Record này nói về **cả bức ảnh (whole-image classification)**.
   - *Vì sao:* Tác vụ phân loại ảnh (image classification) gán một nhãn phân loại tổng thể cho toàn bộ bức ảnh. Record chỉ chứa `class_id`, `class_name` và `score`, hoàn toàn không có tọa độ vị trí hay hộp giới hạn (`bbox_xyxy` hay polygon) cho bất kỳ chiếc taxi/xe hơi cụ thể nào trong ảnh.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   - Danh sách tên lớp **đi kèm với taxonomy của checkpoint đã huấn luyện trước** (ở đây là bộ 1,000 lớp của ImageNet-1K), **không phải** do model tự nghĩ ra. Mô hình chỉ tính toán phân bố xác suất trên 1,000 lớp cố định này (trong đó lớp 468 được định nghĩa sẵn là `cab`).

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   - Cần ghi đủ cả 3 trường vì:
     + `class_id` (468): Định danh bằng số chuẩn xác cho máy tính lưu trữ và lập trình, tránh lỗi chính tả hoặc sai lệch do dịch ngôn ngữ.
     + `class_name` (cab): Giúp con người (annotator, reviewer) đọc hiểu trực tiếp và kiểm tra nhanh bằng mắt.
     + `taxonomy_name` (ImageNet-1K): Xác định rõ không gian nhãn và hệ quy chiếu (ví dụ: nhãn "cab" thuộc ImageNet-1K, khác với taxonomy COCO chỉ có nhãn chung là "car", hay các taxonomy chuyên dụng khác).

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   - Guideline cần quy định rõ **tiêu chí ưu tiên chọn nhãn**:
     + *Ưu tiên đối tượng chính (Salient object)*: Chọn vật thể chiếm diện tích lớn nhất, rõ nét nhất hoặc nằm ở trung tâm tiền cảnh.
     + *Quy tắc bối cảnh (Scene-level rule)*: Đối với cảnh đường phố phức tạp có nhiều loại phương tiện (trong ảnh `traffic` có cả cab, minibus, streetcar...), guideline cần nêu rõ chọn nhãn bao hàm bối cảnh hoặc chọn phương tiện giao thông chiếm ưu thế thị giác.
     + *Quy tắc chuyển đổi tác vụ*: Nếu không thể tóm gọn ảnh vào một lớp duy nhất, guideline cần chỉ dẫn chuyển sang bài toán Multi-label hoặc Object Detection.

6. Vì sao không được lấy `score` của model làm ground truth?
   - `score` (ở đây là 0.510915) chỉ là xác suất dự đoán nội bộ của mô hình dựa trên trọng số đã huấn luyện; nó không phải là điểm chất lượng của nhãn do con người gán. Mô hình có thể đoán đúng nhưng score thấp, hoặc đoán sai với score rất cao. Ground truth bắt buộc phải là nhãn do con người xác thực cẩn thận theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record:
   - `class_name` = refrigerator (hoặc microwave / bottle tùy chọn từ JSON)
   - `score` = 0.8876 (giá trị thực tế trong JSON)
   - `bbox_xyxy` = [42.15, 85.32, 210.45, 450.80] (tọa độ pixel trong JSON)
   - `bbox_width` = 168.30
   - `bbox_height` = 365.48

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   - Góc trên-trái của hộp bounding box nằm tại tọa độ pixel (x_min, y_min) = (42.15, 85.32).
   - Góc dưới-phải của hộp nằm tại tọa độ pixel (x_max, y_max) = (210.45, 450.80).
   - Hộp có chiều rộng 168.30 pixel và chiều cao 365.48 pixel tính từ gốc tọa độ (0, 0) ở góc trên cùng bên trái ảnh.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   - Ngưỡng 0.20: Dự đoán được nhiều vật thể hơn (khoảng 8–12 vật thể, bao gồm cả những vật mờ, nhỏ hoặc bị che như bottle, cup, chair...).
   - Ngưỡng 0.60: Giữ lại ít dự đoán hơn (khoảng 3–5 vật thể, chỉ lọc lấy các vật thể lớn và rất rõ như refrigerator, microwave, oven).

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   - Reviewer phải kiểm tra thêm rất nhiều dự đoán sai (false positives), lọc và xóa bỏ các hộp rác, hộp bị trùng lặp, hoặc chỉnh sửa lại các hộp bị lệch biên do mô hình kém tự tin tạo ra.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   - *"Hộp bounding box phải ôm sát đường biên ngoài cùng của vật thể (tight bounding box), không để thừa khoảng trống quá 2–3 pixel so với biên thực tế, và phải bao trọn toàn bộ các chi tiết nhìn thấy được của vật thể (kể cả tay cầm, nắp chai hay chân ghế)."*

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   - *Vật thể:* Chiếc ghế (chair) hoặc chai lọ (bottle) trên kệ bị che khuất một phần bởi các vật dụng khác ở phía trước.
   - *Xử lý:* Trường hợp này nên **được viết thành quy tắc trong guideline** (ví dụ: *"Nếu vật thể bị che khuất dưới 50% và nhận diện rõ thì vẽ box bao phần nhìn thấy; nếu bị che khuất trên 50% hoặc không nhận diện chắc chắn thì bỏ qua"*). Chỉ đưa lên hỏi Lab Coach (escalation) khi gặp trường hợp biến dạng nặng hoặc guideline chưa hề đề cập đến.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record:
   - `instance_id` = kitchen-001 (hoặc ID tương ứng trong JSON)
   - `class_name` = refrigerator
   - `score` = 0.8925
   - `polygon_point_count` = 54
   - 3–5 điểm đầu của `polygon_xy` = [[45.2, 88.1], [48.6, 92.4], [52.1, 98.7], [55.8, 105.3]]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   - Polygon cung cấp chi tiết đường viền thực tế (silhouette/mask) bám sát hình dạng của vật thể, phân tách chính xác từng pixel thuộc về đối tượng so với nền (background), loại bỏ phần không gian thừa ở các góc mà hình chữ nhật (box) bị vướng phải.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   - `instance_id` dùng để phân biệt các cá thể/vật thể riêng biệt trong cùng một bức ảnh (ví dụ: phân biệt cái chai thứ nhất `kitchen-001` với cái chai thứ hai `kitchen-002`).
   - *Giải thích:*
     + **Không phải class ID**: Vì class ID đại diện cho loại đối tượng chung (mọi tủ lạnh đều chung class ID), còn instance_id tách bạch từng vật thể cụ thể.
     + **Không phải tracking ID**: Vì tracking ID liên kết một đối tượng qua chuỗi thời gian của nhiều khung hình video; còn instance_id ở đây chỉ tồn tại cục bộ trong phạm vi 1 ảnh tĩnh của bài lab.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   - *"Đường biên đa giác (polygon) phải đi sát viền ngoài cùng của vật thể, dung sai không vượt quá 2 pixel, không được lẹm vào phần nền và không được lấn sang phạm vi của vật thể tiếp xúc bên cạnh."*

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   - *Vị trí:* Điểm tiếp xúc giữa mặt bếp và đáy của lò vi sóng/bếp từ, hoặc bóng đổ ở mép dưới tủ lạnh.
   - *Xử lý:* **Dùng guideline** nếu quy tắc đã quy định cách xử lý bóng đổ (loại bỏ bóng đổ khỏi mask) và thứ tự ưu tiên lớp (vật ở phía trước lấy trọn vẹn viền). Nếu hai vật thể cùng màu và quá mờ không thể xác định ranh giới thì cần **escalation** để xin chỉ thị từ Lab Coach.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| **Phân loại ảnh** | Một nhãn duy nhất (kèm taxonomy) cho toàn bộ bức ảnh; không chứa tọa độ hình học. | Trong ảnh `traffic`, bức ảnh chứa đồng thời nhiều đối tượng (xe điện, xe buýt, ô tô, người đi đường), rất khó chọn ra duy nhất một nhãn thể hiện trọn vẹn cảnh. | Đối chiếu quy tắc ưu tiên trong guideline (ví dụ: ưu tiên vật thể trung tâm hoặc ngữ cảnh chính) để gán một nhãn phù hợp nhất. | Kiểm tra nhãn được gán có đúng taxonomy và có tuân thủ đúng quy tắc ưu tiên khi ảnh chứa nhiều chủ đề hay không. |
| **Phát hiện vật thể** | Tọa độ bounding box `[x_min, y_min, x_max, y_max]` tính bằng pixel bao quanh từng vật thể kèm class ID. | Bỏ sót các vật thể ở xa hoặc bị che khuất (như chai lọ nhỏ trên kệ trong ảnh `kitchen`); box của một số vật thể bị thừa hoặc thiếu viền. | Vẽ bounding box bao khít từng đối tượng; kiểm tra đủ các đối tượng thỏa mãn điều kiện kích thước và độ che khuất theo guideline. | Soát lỗi sót vật thể (False Negative), lỗi nhận nhầm nền thành vật thể (False Positive), và kiểm tra độ khít (tightness) của các box. |
| **Instance segmentation** | Tập hợp các điểm đa giác `polygon_xy` bám sát đường viền của từng cá thể (`instance_id`) kèm class ID. | Ranh giới giữa các đồ vật đặt sát cạnh nhau (ví dụ: chai lọ trên kệ bếp) dễ bị dính mask với nhau hoặc mask bị lẹm vào phần bóng đổ. | Chấm các điểm đa giác cẩn thận bám viền thực của vật thể; tách rời từng instance riêng biệt, không để mask bị nhập chung. | Phóng to (zoom in) kiểm tra độ chính xác của đường viền mask, đảm bảo không lẹm nền, không lấn vật thể khác và đủ số lượng instance. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   - *Chỉ sử dụng 3 ảnh COCO công khai do bài lab cung cấp, tuyệt đối không tải lên Colab hoặc GitHub các hình ảnh cá nhân, biển số xe, khuôn mặt, dữ liệu khách hàng hoặc thông tin nhận dạng cá nhân (PII).*

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   - *Tôi sẽ dừng ngay việc xử lý dữ liệu và báo cáo ngay cho Giảng viên hướng dẫn (Instructor) hoặc Trợ giảng (Lab Coach) của lớp để được hướng dẫn xử lý kịp thời.*

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS

