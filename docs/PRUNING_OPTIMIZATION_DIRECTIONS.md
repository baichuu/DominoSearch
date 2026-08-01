# Các hướng tối ưu model bằng pruning

## 1. Mục tiêu

Mục tiêu của dự án là giảm chi phí của model để model có thể triển khai trên phần
cứng có tài nguyên hạn chế như CPU yếu, FPGA hoặc các board nhúng, trong khi độ
chính xác giảm ít nhất có thể.

Các chỉ số cần tối ưu đồng thời:

- giảm số parameter hiệu dụng;
- giảm MACs/FLOPs hiệu dụng;
- giữ Top-1 và Top-5 accuracy gần model dense;
- giảm latency và tăng throughput trên phần cứng mục tiêu;
- giảm bộ nhớ, băng thông bộ nhớ và tài nguyên FPGA như BRAM, DSP, LUT.

Không được kết luận model nhanh hơn chỉ dựa vào MACs. Tốc độ thực tế còn phụ thuộc
vào kiểu pruning, kernel inference và khả năng hỗ trợ sparsity của phần cứng.

## 2. Baseline chung

Nhánh gốc: `master`.

Baseline phải dùng model chưa pruning, tức `N=M`, cùng checkpoint, dataset và điều
kiện benchmark với các model tối ưu.

Ví dụ:

```bash
python benchmark/benchmark_model.py \
  --run-name dense \
  --model resnet18_sparse \
  --checkpoint /path/to/dense_checkpoint.pth \
  --n 1 --m 1 \
  --dataset-format imagefolder \
  --data-root /path/to/imagenet/val \
  --output results/dense.json
```

## 3. Hướng 1: Uniform N:M pruning

Nhánh Git: `pruning-uniform-nm`.

### Ý tưởng

Áp dụng cùng một cấu hình N:M cho hầu hết hoặc toàn bộ layer. Ví dụ, với 2:4,
trong mỗi nhóm 4 weight chỉ giữ lại 2 weight khác 0.

Các cấu hình nên thử:

| Cấu hình | Tỷ lệ weight giữ lại | Tỷ lệ bị pruning |
| -------- | -------------------: | ---------------: |
| 4:4      |                 100% |               0% |
| 3:4      |                  75% |              25% |
| 2:4      |                  50% |              50% |
| 1:4      |                  25% |              75% |
| 8:16     |                  50% |              50% |
| 4:16     |                  25% |              75% |

### Cách triển khai

1. Load dense checkpoint.
2. Áp dụng cùng N:M cho các `SparseConv` và `SparseLinear`.
3. Đo accuracy ngay sau pruning.
4. Fine-tune bằng dynamic sparse training.
5. Đo lại accuracy và toàn bộ benchmark.

Nên thử thêm biến thể giữ layer đầu và layer cuối ở dạng dense vì hai layer này
thường nhạy cảm với pruning.

### Ưu điểm

- đơn giản và dễ kiểm tra;
- không cần chạy thuật toán search;
- cấu trúc đều, dễ thiết kế accelerator;
- là baseline cần thiết để chứng minh mixed N:M có thực sự tốt hơn.

### Hạn chế

- mọi layer bị ép dùng cùng sparsity dù độ nhạy khác nhau;
- layer nhạy có thể làm accuracy giảm mạnh;
- layer ít nhạy chưa chắc đã được pruning đủ nhiều.

### Giả thuyết cần kiểm tra

Uniform 2:4 có thể là điểm cân bằng đầu tiên giữa accuracy và độ thưa. Nếu phần
cứng không hỗ trợ trực tiếp N:M thì MACs giảm theo lý thuyết nhưng latency thực tế
có thể không giảm.

## 4. Hướng 2: DominoSearch mixed N:M pruning

Nhánh Git: `pruning-domino-mixed-nm`.

### Ý tưởng

Không dùng một tỷ lệ cho toàn model. DominoSearch tìm N:M riêng cho từng layer:
layer nhạy được giữ nhiều weight hơn, layer ít nhạy bị pruning mạnh hơn.

Ví dụ:

```text
Layer 1:  16:16  (dense)
Layer 2:   8:16  (giữ 50%)
Layer 3:   4:16  (giữ 25%)
Layer 4:   2:16  (giữ 12.5%)
```

### Cách triển khai

1. Chạy dense baseline.
2. Chọn tập N ứng viên, ví dụ `N ∈ {2, 4, 8, 16}` với `M=16`.
3. Chạy DominoSearch để tìm scheme cho từng layer.
4. Lưu scheme vào `searched_scheme.txt`.
5. Đo model ngay sau khi áp dụng scheme.
6. Fine-tune bằng dynamic sparse training.
7. Benchmark checkpoint sau fine-tune.

Các target sparsity nên thử: 25%, 50%, 60%, 70% và 80%. Không nên chỉ thử một
target vì cần xây dựng đường cong accuracy–resource.

### Ưu điểm

- phân bổ sparsity theo độ nhạy của từng layer;
- thường giữ accuracy tốt hơn uniform pruning tại cùng budget;
- đây là hướng chính của bài báo DominoSearch.

### Hạn chế

- search và fine-tune tốn thời gian;
- mixed N:M làm phần cứng phức tạp hơn uniform N:M;
- objective gốc dựa nhiều vào FLOPs, chưa phản ánh hoàn toàn latency trên board.

### Cải tiến quan trọng nên nghiên cứu

Thay cost theo FLOPs bằng cost đo hoặc ước lượng từ phần cứng mục tiêu:

```text
Cost = α × latency + β × energy + γ × bandwidth + δ × memory
```

Mỗi cấu hình N:M của từng loại layer được benchmark trên CPU/FPGA để xây dựng
lookup table hoặc huấn luyện cost predictor. DominoSearch sau đó tìm scheme dựa
trên cost thực tế thay vì chỉ dựa vào FLOPs.

### Giả thuyết cần kiểm tra

Tại cùng effective MACs, mixed N:M phải có Top-1 cao hơn uniform N:M. Nếu không,
chi phí search chưa tạo ra lợi ích đủ lớn.

## 5. Hướng 3: Structured channel/filter pruning

Nhánh Git: `pruning-structured-channel`.

### Ý tưởng

Loại bỏ toàn bộ output channel hoặc filter thay vì loại các weight riêng lẻ. Khi
channel bị xóa, kích thước tensor và layer kế tiếp cũng được thu nhỏ thật sự.

Các tiêu chí xếp hạng channel nên thử:

- L1 norm hoặc L2 norm của filter;
- độ lớn hệ số gamma của BatchNorm;
- Taylor importance, dựa trên weight và gradient;
- sensitivity đo bằng mức giảm accuracy khi loại channel.

### Cách triển khai

1. Tính importance score cho từng output channel.
2. Giữ nguyên layer đầu, layer cuối và các layer rất nhạy trong thử nghiệm đầu.
3. Prune dần 10%, 20%, 30%, 40% và 50% channel.
4. Sửa input channel của layer tiếp theo cho khớp.
5. Xử lý đúng các nhánh residual của ResNet.
6. Fine-tune sau mỗi mức pruning.
7. Export model compact thật sự rồi benchmark.

### Ưu điểm

- model và tensor nhỏ thật, không chỉ có nhiều giá trị 0;
- dense library thông thường cũng có thể tăng tốc;
- dễ khai thác hơn trên CPU, FPGA và board không có sparse kernel;
- giảm cả compute và activation memory.

### Hạn chế

- thay đổi kiến trúc và shape của model;
- residual connection làm việc xóa channel phức tạp;
- pruning mạnh có thể làm mất accuracy nhanh hơn fine-grained N:M.

### Giả thuyết cần kiểm tra

Structured pruning có thể giảm latency thực tế tốt hơn N:M trên phần cứng không hỗ
trợ sparse operator, kể cả khi accuracy tại cùng tỷ lệ parameter thấp hơn một ít.

## 6. Hướng 4: Unstructured magnitude pruning

Nhánh Git: `pruning-unstructured-magnitude`.

### Ý tưởng

Xóa riêng từng weight có độ lớn nhỏ nhất. Có thể pruning theo từng layer hoặc
global pruning trên toàn model.

Các biến thể cần thử:

- local magnitude pruning: mỗi layer có cùng tỷ lệ;
- global magnitude pruning: chọn weight nhỏ nhất trên toàn model;
- one-shot pruning: prune một lần rồi fine-tune;
- gradual pruning: tăng sparsity từ từ trong quá trình fine-tune.

### Cách triển khai

1. Load dense checkpoint.
2. Tính threshold theo độ lớn tuyệt đối của weight.
3. Tạo binary mask và giữ mask trong quá trình fine-tune.
4. Thử sparsity 25%, 50%, 70%, 80%, 90%.
5. Đo cả accuracy và số non-zero thực tế.

### Ưu điểm

- linh hoạt, thường giữ accuracy tốt tại cùng số non-zero;
- dễ triển khai làm baseline nghiên cứu;
- global pruning tự phân bổ sparsity giữa các layer ở mức đơn giản.

### Hạn chế

- pattern không đều, khó tăng tốc trên CPU/FPGA;
- cần lưu index/mask;
- model có nhiều số 0 nhưng file và phép toán vẫn dense nếu không có sparse format;
- không phải ứng viên triển khai tốt nhất nếu board không hỗ trợ unstructured sparsity.

### Giả thuyết cần kiểm tra

Hướng này có thể đạt accuracy tốt nhất theo số non-zero nhưng không nhất thiết có
latency tốt nhất. Nó chủ yếu là đối chứng để phân biệt “model thưa” và “model chạy
nhanh thật”.

## 7. Ma trận thí nghiệm tối thiểu

| Nhóm               | Cấu hình tối thiểu        | Fine-tune | Vai trò            |
| ------------------ | ------------------------- | --------- | ------------------ |
| Dense              | N=M                       | Không     | Baseline           |
| Uniform N:M        | 3:4, 2:4, 1:4             | Có        | Baseline N:M       |
| Domino mixed N:M   | target 50%, 60%, 70%, 80% | Có        | Hướng chính        |
| Structured channel | 10%, 20%, 30%, 40%, 50%   | Có        | Hướng phần cứng    |
| Unstructured       | 50%, 70%, 80%, 90%        | Có        | Đối chứng accuracy |

Mỗi cấu hình cần lưu:

- commit và tên nhánh;
- checkpoint đầu vào và checkpoint sau fine-tune;
- pruning scheme hoặc mask;
- seed và hyperparameter;
- JSON benchmark;
- log huấn luyện;
- kết quả export/implementation trên phần cứng nếu có.

## 8. Tiêu chí chọn hướng tốt nhất

Không chọn model chỉ vì có sparsity cao nhất. Nên chọn theo Pareto frontier: không
có model khác vừa chính xác hơn, vừa nhanh hơn, vừa nhỏ hơn.

Thứ tự đánh giá đề xuất:

1. Loại cấu hình làm Top-1 giảm quá ngưỡng cho phép, ví dụ hơn 1–2 điểm phần trăm.
2. Trong các cấu hình còn lại, so latency trên đúng phần cứng mục tiêu.
3. Nếu latency gần nhau, ưu tiên model dùng ít memory/năng lượng hơn.
4. Nếu vẫn gần nhau, ưu tiên scheme đơn giản và dễ triển khai hơn.

Nên báo cáo ít nhất:

```text
ΔTop-1          = Top-1(pruned) - Top-1(dense)
MAC reduction   = 1 - effective_MACs / dense_MACs
Speedup         = latency_dense / latency_pruned
Memory reduction= 1 - memory_pruned / memory_dense
```

## 9. Thứ tự triển khai đề xuất

1. Hoàn thành `pruning-uniform-nm` để xác nhận pipeline và tạo baseline nhanh.
2. Hoàn thành `pruning-domino-mixed-nm` và so với uniform tại cùng MAC budget.
3. Triển khai `pruning-structured-channel` để kiểm tra speedup thật trên CPU/board.
4. Dùng `pruning-unstructured-magnitude` làm đối chứng về accuracy và sparsity.
5. Chọn 2–3 điểm Pareto tốt nhất để đo trên FPGA hoặc phần cứng mục tiêu.

Kỳ vọng thực tế: mixed N:M có thể tốt nhất về cân bằng accuracy–sparsity, trong khi
structured channel pruning có khả năng tốt hơn về latency trên phần cứng không có
sparse kernel. Kết quả benchmark thực tế mới quyết định hướng cuối cùng.

## 10. Công cụ benchmark

Hướng dẫn chạy benchmark nằm tại [`benchmark/README.md`](../benchmark/README.md).
Mọi nhánh phải dùng cùng công cụ và cùng điều kiện đo để kết quả có thể so sánh.
