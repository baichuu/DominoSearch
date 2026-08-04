# Báo cáo thực nghiệm các hướng pruning ResNet-18 trên Tesla T4

Ngày tổng hợp: 2026-08-04.

## 1. Mục tiêu và phạm vi

Báo cáo trả lời ba câu hỏi bằng số liệu thực tế:

1. Mỗi hướng pruning giữ được accuracy đến đâu?
2. Parameter và MAC hiệu dụng giảm bao nhiêu?
3. Implementation PyTorch hiện tại có giảm latency hoặc memory trên Tesla T4
   hay không?

Phạm vi gồm dense baseline và bốn hướng pruning của repository: Uniform N:M,
Domino mixed N:M, structured channel L1 và unstructured global magnitude. Mỗi
hướng có hai mốc: ngay sau pruning và sau fine-tune giới hạn. Báo cáo không trộn
quantization, distillation hoặc thay kiến trúc.

## 2. Thuật ngữ dùng trong bảng

- **Top-1**: tỷ lệ ảnh mà lớp có xác suất cao nhất là đáp án đúng.
- **Top-5**: tỷ lệ ảnh mà đáp án đúng nằm trong năm lớp có xác suất cao nhất.
- **Δ dense**: Top-1 của model pruning trừ Top-1 dense, đơn vị điểm phần trăm.
- **Parameter hiệu dụng**: số parameter còn được dùng theo N:M hoặc số non-zero;
  không phải kích thước file checkpoint.
- **MAC hiệu dụng**: số multiply-accumulate còn lại theo mask; đây là mức giảm
  tính toán lý thuyết, chưa phải runtime speedup.
- **Median/P95 latency**: trung vị/phân vị 95 của 100 lần inference sau 30 lần
  warm-up, batch size 1.
- **Throughput**: số sample xử lý trong một giây từ cùng phép đo latency.
- **Peak memory**: đỉnh CUDA memory quan sát được trong benchmark.

## 3. Giao thức và kiểm tra tính hợp lệ

Tất cả chín JSON trong bảng đã được audit với cùng các điều kiện có thể kiểm
chứng từ artifact:

- `resnet18_sparse`, input `1 × 3 × 224 × 224`;
- ImageNet validation đủ 50.000 ảnh;
- cùng 14 shard Parquet, tổng 6.693.093.726 byte;
- Tesla T4, CUDA 12.8, PyTorch 2.11.0+cu128;
- seed 42, latency batch size 1, warm-up 30, measured iterations 100;
- checkpoint load không có missing hoặc unexpected key;
- source worktree sạch, branch và commit được ghi trong JSON;
- mỗi phương pháp có cả kết quả trước và sau fine-tune.

Fine-tune dùng 3 epoch, batch size 256, base learning rate 0.01, 50.000 training
sample mỗi epoch và validation nội bộ 1.000 sample. Checkpoint cuối được benchmark
lại trên đủ 50.000 validation sample. Đây là budget giới hạn, chỉ dùng khoảng
3,9% tập train ImageNet mỗi epoch, không phải full-training protocol.

JSON hiện chưa lưu accuracy batch size. Lệnh chạy sử dụng batch 64, nhưng trường
này không thể audit độc lập từ JSON; đây là một hạn chế provenance cần bổ sung
cho benchmark schema sau này.

## 4. Dense baseline

Dense baseline đạt:

- Top-1: **69,754%**;
- Top-5: **89,078%**;
- parameter: **11.689.512**;
- MAC/sample: **1.814.073.344**;
- median/P95 latency: **3,725/5,281 ms**;
- throughput: **268,44 sample/s**;
- peak memory: **93,19 MB**.

Mọi `Δ dense` trong báo cáo dùng đúng baseline này, không dùng số từ lần chạy
debug hoặc một thiết bị khác.

## 5. Kết quả đầy đủ

`Δ trước FT` chỉ có ở checkpoint sau fine-tune và cho biết fine-tune phục hồi hay
làm giảm Top-1. Peak memory dùng MB thập phân.

| Phương pháp               | Giai đoạn | Top-1 % | Top-5 % | Δ dense | Δ trước FT | Param giảm % | MAC giảm % | Median ms | P95 ms | sample/s | Peak MB |
| ------------------------- | --------- | ------: | ------: | ------: | ---------: | -----------: | ---------: | --------: | -----: | -------: | ------: |
| Dense                     | Baseline  |  69,754 |  89,078 |  +0,000 |          — |         0,00 |       0,00 |     3,725 |  5,281 |   268,44 |   93,19 |
| Domino mixed N:M          | Trước FT  |  68,328 |  88,356 |  -1,426 |          — |        30,27 |       9,56 |     8,577 |  9,370 |   116,59 |  114,06 |
| Domino mixed N:M          | Sau FT    |  67,996 |  88,168 |  -1,758 |     -0,332 |        30,27 |       9,56 |     8,197 |  8,705 |   122,00 |  114,06 |
| Uniform 3:4               | Trước FT  |  66,094 |  86,856 |  -3,660 |          — |        23,49 |      23,10 |    22,230 | 23,531 |    44,98 |  114,06 |
| Uniform 3:4               | Sau FT    |  68,388 |  88,288 |  -1,366 |     +2,294 |        23,49 |      23,10 |    22,761 | 24,265 |    43,94 |  114,06 |
| Structured channel 10% L1 | Trước FT  |  51,066 |  75,698 | -18,688 |          — |         8,99 |      10,16 |     3,742 |  4,209 |   267,25 |   93,19 |
| Structured channel 10% L1 | Sau FT    |  66,672 |  87,362 |  -3,082 |    +15,606 |         8,99 |      10,16 |     3,710 |  4,069 |   269,55 |   93,19 |
| Unstructured global 30%   | Trước FT  |  69,218 |  88,878 |  -0,536 |          — |        28,63 |      21,89 |     3,775 |  4,049 |   264,88 |   93,19 |
| Unstructured global 30%   | Sau FT    |  68,572 |  88,512 |  -1,182 |     -0,646 |        28,63 |      21,89 |     3,712 |  4,754 |   269,40 |   93,19 |

## 6. Phân tích từng hướng bằng dữ liệu thực tế

### 6.1 Uniform 3:4 N:M

**Cách tối ưu.** Hầu hết sparse convolution dùng cùng quy tắc giữ ba trong mỗi
bốn weight. Layer đầu, linear cuối và convolution 1×1 được giữ dense để bảo vệ
boundary nhạy cảm.

**Kết quả.** Pruning trực tiếp làm Top-1 giảm 3,660 điểm. Fine-tune phục hồi
2,294 điểm, đưa model lên 68,388%, còn kém dense 1,366 điểm. Parameter/MAC hiệu
dụng giảm lần lượt 23,49% và 23,10%.

**Runtime.** Median latency sau fine-tune là 22,761 ms, cao gấp khoảng 6,1 lần
dense; peak memory tăng từ 93,19 lên 114,06 MB. Nguyên nhân là layer hiện tạo mask
rồi vẫn gọi dense PyTorch operator, không phải sparse kernel chuyên dụng.

**Kết luận.** Uniform 3:4 chứng minh fine-tune có thể phục hồi accuracy và tạo
được N:M đều, nhưng chưa tối ưu runtime trên stack hiện tại. Trong hai hướng N:M
đã thử, checkpoint sau fine-tune của Uniform có Top-1 cao hơn Domino và MAC giảm
nhiều hơn.

### 6.2 DominoSearch mixed N:M

**Cách tối ưu.** Scheme phân bổ N:M theo layer thay vì ép cùng một tỷ lệ. Run này
giữ phần lớn layer ở 16:16 và dùng 8:16 cho ba convolution sâu, nhắm giảm parameter
theo objective search.

**Kết quả.** Trước fine-tune đạt 68,328% Top-1, cao hơn Uniform trước fine-tune
2,234 điểm. Tuy nhiên, fine-tune làm giảm 0,332 điểm xuống 67,996%. Parameter hiệu
dụng giảm 30,27%, nhưng MAC chỉ giảm 9,56% vì các layer được prune không chiếm tỷ
trọng compute tương ứng.

**Runtime.** Median latency sau fine-tune 8,197 ms vẫn cao gấp khoảng 2,2 lần
dense và peak memory là 114,06 MB.

**Kết luận.** Mixed N:M bảo toàn accuracy ngay sau pruning tốt hơn Uniform trong
hai cấu hình đã chạy, nhưng objective theo parameter chưa tạo điểm tốt về MAC và
fine-tune hiện tại không cải thiện checkpoint. Chưa thể kết luận Domino kém hơn
Uniform nói chung vì hai run không có cùng parameter/MAC budget; chỉ có thể nói
Uniform 3:4 tốt hơn ở checkpoint sau fine-tune của thí nghiệm hiện tại.

### 6.3 Structured channel 10% theo L1

**Cách tối ưu.** Hidden channel được xếp hạng bằng L1 norm rồi materialize thành
zero; mask cố định ngăn weight mọc lại trong fine-tune. Residual output shape được
giữ an toàn. Artifact vẫn là tensor dense-shape, chưa compact channel vật lý.

**Kết quả.** Đây là hướng nhạy nhất với one-shot pruning: Top-1 giảm còn 51,066%.
Fine-tune phục hồi tới 15,606 điểm nhưng kết quả cuối 66,672% vẫn thấp hơn dense
3,082 điểm. Parameter giảm 8,99% và MAC giảm 10,16%.

**Runtime.** Latency và memory gần dense vì tensor chưa được compact. Sai khác
3,710 ms so với dense 3,725 ms quá nhỏ để gọi là speedup từ một lần đo.

**Kết luận.** Mask channel L1 hiện chứng minh được pipeline persistent mask và khả
năng phục hồi bằng fine-tune, nhưng chưa phải model structured compact và chưa là
điểm accuracy–complexity cạnh tranh. Cần compact export và sweep tỷ lệ nhỏ hơn
hoặc ranking tốt hơn trước khi đánh giá lại runtime.

### 6.4 Unstructured global magnitude 30%

**Cách tối ưu.** Toàn bộ weight đủ điều kiện được xếp hạng chung theo trị tuyệt
đối; 30% weight nhỏ nhất bị zero hóa. Global threshold cho phép layer nhạy giữ
nhiều weight hơn layer dư thừa; mask được giữ cố định khi fine-tune.

**Kết quả.** Ngay sau pruning đạt 69,218% Top-1, chỉ kém dense 0,536 điểm, trong
khi parameter hiệu dụng giảm 28,63% và MAC giảm 21,89%. Đây là điểm cân bằng
accuracy–độ phức tạp tốt nhất trong các run hiện có. Fine-tune với LR 0.01 làm
Top-1 giảm 0,646 điểm xuống 68,572%, nên checkpoint trước fine-tune tốt hơn.

**Runtime.** Median latency 3,775 ms trước fine-tune và 3,712 ms sau fine-tune đều
gần dense. Dense operator không bỏ qua hiệu quả các zero không có cấu trúc, do đó
không có bằng chứng tăng tốc. Peak memory cũng không giảm.

**Kết luận.** Đây là hướng tốt nhất hiện tại nếu tiêu chí là giữ accuracy trong
khi giảm số non-zero/MAC lý thuyết. Nó chưa phải hướng tốt nhất theo runtime. Cần
tune lại fine-tune với LR thấp hơn trước khi dùng checkpoint sau fine-tune.

## 7. So sánh và quyết định hiện tại

### 7.1 Nếu chọn model ngay bây giờ

Chọn **Unstructured global magnitude 30% trước fine-tune**:

- Top-1 cao nhất trong tất cả model pruning: 69,218%;
- chỉ mất 0,536 điểm so với dense;
- parameter giảm 28,63%, MAC giảm 21,89%;
- không cần dùng checkpoint fine-tune đang làm accuracy xấu đi.

Từ “tốt nhất” ở đây chỉ có nghĩa là điểm accuracy–độ phức tạp tốt nhất trong
chín run đã đo, không có nghĩa là runtime nhanh nhất hay tốt nhất cho mọi thiết
bị.

### 7.2 Nếu bắt buộc chọn checkpoint sau fine-tune

| Hạng | Phương pháp               | Top-1 % | Δ dense | Param giảm % | MAC giảm % |
| ---: | ------------------------- | ------: | ------: | -----------: | ---------: |
|    1 | Unstructured global 30%   |  68,572 |  -1,182 |        28,63 |      21,89 |
|    2 | Uniform 3:4               |  68,388 |  -1,366 |        23,49 |      23,10 |
|    3 | Domino mixed N:M          |  67,996 |  -1,758 |        30,27 |       9,56 |
|    4 | Structured channel 10% L1 |  66,672 |  -3,082 |         8,99 |      10,16 |

### 7.3 Những điều dữ liệu chưa chứng minh

- Không hướng nào chứng minh runtime nhanh hơn dense trên stack PyTorch hiện tại.
- Chưa có sweep nhiều sparsity cho từng hướng, nên chưa xây được Pareto frontier
  đầy đủ.
- Chưa có kết quả full fine-tune ImageNet; kết luận sau fine-tune chỉ áp dụng cho
  budget 3 epoch × 50.000 sample.
- Checkpoint có nhiều zero không tự nhỏ hơn vì tensor vẫn được lưu dense; file
  training còn chứa optimizer state.

## 8. Thí nghiệm tiếp theo dựa trên kết quả

1. Giữ nguyên unstructured global 30% và mask, thử LR `0.001` rồi `0.0001` với
   cùng budget. Chỉ nhận checkpoint nếu Top-1 vượt 69,218% mà sparsity không đổi.
2. Sweep unstructured 20%, 30%, 40%, 50% để xây đường cong accuracy–complexity;
   so global với local tại cùng tỷ lệ.
3. Với N:M, dùng Uniform 3:4 làm mốc và chạy Domino ở cùng MAC budget khoảng 23%
   để so công bằng khả năng phân bổ sparsity theo layer.
4. Với structured, thử tỷ lệ thấp hơn và BN-gamma/Taylor ranking; sau đó tạo model
   compact thật trước khi đo runtime.
5. Lặp latency benchmark nhiều lần, báo median giữa các run và chỉ đánh giá
   runtime khi implementation thực sự khai thác pattern sparse/compact.

## 9. Truy vết artifact, checkpoint và commit

Các JSON nguồn được dùng trực tiếp để đối chiếu bảng:

```text
dense-resnet18-imagenet-t4-20260804.json
domino-params23-before-20260804.json
domino-params23-after-train50k-20260804.json
uniform-3of4-conservative-before-20260804.json
uniform-3of4-after-train50k-20260804.json
structured-channel10-l1-before-20260804.json
structured-channel10-l1-after-train50k-20260804.json
unstructured-global30-before-20260804.json
unstructured-global30-after-train50k-20260804.json
```

Checkpoint sau fine-tune được truy vết bằng SHA-256:

| Hướng                     | SHA-256 checkpoint cuối                                            |
| ------------------------- | ------------------------------------------------------------------ |
| Domino mixed N:M          | `863eca5a477bba9450cb2f1c63bf85db0b27c9ee7e7a18d97af1542455ac6d59` |
| Uniform 3:4               | `00d5f5b31fdb828e7623b45ec3b96a85cc298bc14a29866ab12814cbb3eff864` |
| Structured channel 10% L1 | `4c6a4d131516b687fdc3dd94fcb417d92a3e3c9fccdb1ad1524f2452cb388913` |
| Unstructured global 30%   | `044acb2472cab120b63082bf46b6c73701a848a9c8590db6e6b73337324d2d50` |

Uniform được resume sau epoch 1; structured và unstructured được resume từ
checkpoint epoch 2 do session/runtime bị gián đoạn. Manifest cuối đánh dấu hoàn
tất, lưu SHA-256 checkpoint, và chỉ kết quả benchmark đầy đủ sau resume được đưa
vào bảng. Việc resume không thay đổi scheme, mask, seed hoặc tổng số epoch.

### Vị trí lưu trữ

Artifact gốc nằm ngoài Git:

```text
Google Drive/MyDrive/DominoSearch-artifacts/
├── results/
├── runs/
├── checkpoints/
├── schemes/
└── reports/
```

### Commit nguồn

| Hướng                  | Branch                           | Implementation commit | Commit của kết quả sau FT      |
| ---------------------- | -------------------------------- | --------------------- | ------------------------------ |
| Dense/shared           | `master`                         | `d8254f6`             | `c3d022f` (benchmark baseline) |
| Uniform N:M            | `pruning-uniform-nm`             | `59ed0b1`             | `78f79e7`                      |
| Domino mixed N:M       | `pruning-domino-mixed-nm`        | `956077e`, `77f4aac`  | `025de97`                      |
| Structured channel     | `pruning-structured-channel`     | `f12b328`             | `9cce258`                      |
| Unstructured magnitude | `pruning-unstructured-magnitude` | `d47df96`             | `142aa66`                      |

CSV máy đọc và validity audit được lưu trong thư mục `reports/` trên Drive. Chỉ
chín candidate run vượt kiểm tra tính hợp lệ mới được đưa vào bảng; debug run bị
loại khỏi kết luận.
