# Báo cáo hardware-aware cost theo từng layer trên Tesla T4

Ngày thực nghiệm: 2026-08-05  
Nhánh: `pruning-domino-mixed-nm`

## 1. Yêu cầu nghiên cứu

Mục tiêu là thay cách đánh giá cấu hình pruning chỉ dựa trên FLOPs bằng cost đo
hoặc ước lượng từ phần cứng. Với mỗi layer và mỗi cấu hình N:M, hệ thống cần:

1. thu thập cost trên Tesla T4;
2. xây dựng lookup table;
3. huấn luyện cost predictor cho từng layer;
4. dùng cost để chọn mixed N:M scheme;
5. benchmark model trước và sau fine-tune.

Phạm vi báo cáo chỉ gồm pruning và kết quả trên Tesla T4. Không trộn quantization,
distillation hoặc thay đổi kiến trúc model.

## 2. Hàm cost

Với layer `i` và cấu hình `N:M`, cost chuẩn hóa được định nghĩa:

```text
C_i(N:M) = α × L_i/L_dense
         + β × E_i/E_dense
         + γ × B_i/B_dense
         + δ × W_i/W_dense
```

Trong đó:

- `L_i`: latency của layer trên T4;
- `E_i`: năng lượng tiêu thụ;
- `B_i`: bandwidth ước lượng từ input, output, weight và bias;
- `W_i`: memory ước lượng từ effective weight và bias;
- `α`, `β`, `γ`, `δ`: trọng số của từng thành phần, tổng bằng 1.

Energy không được đo trong thí nghiệm này nên `β = 0`. Không suy đoán energy từ
TDP của GPU vì cách đó không tạo ra số đo năng lượng hợp lệ.

Objective được dùng để tạo scheme thử nghiệm:

```text
α = 0,20   # latency
β = 0,00   # energy
γ = 0,40   # bandwidth
δ = 0,40   # memory
```

Cost tổng hợp này là một proxy phục vụ search. Giảm composite cost không đồng
nghĩa latency end-to-end chắc chắn giảm.

## 3. Cách thu thập dữ liệu từng layer

Model được dùng là `resnet18_sparse` với dense checkpoint có SHA-256:

```text
a96f895435d204f75bc5980ac0c2f580cf21d078ad3d66f1a13a664274fe4e81
```

Profiler chạy trên 21 sparse layer. Mỗi layer được thử với năm cấu hình:

```text
1:16, 2:16, 4:16, 8:16, 16:16
```

Tổng cộng có `21 × 5 = 105` điểm profile. Giao thức latency:

- thiết bị: Tesla T4;
- PyTorch: `2.11.0+cu128`;
- CUDA runtime: `12.8`;
- input: `1 × 3 × 224 × 224`;
- batch size: 1;
- warm-up: 30 iteration;
- đo: 100 iteration;
- repeat: 7;
- seed: 42.

Checkpoint được load không có missing hoặc unexpected key. Profile ghi lại
branch, commit, môi trường, shape của layer và toàn bộ kết quả đo.

## 4. Lookup table và cost predictor

### 4.1 Lookup table

Lookup lấy trực tiếp cost đã profile của đúng cặp `layer × N:M`. Vì toàn bộ 105
ứng viên đã được đo, lookup là nguồn cost chính xác hơn predictor trong thí nghiệm
này.

Kết quả latency từng layer cho thấy:

- 21 layer được kiểm tra;
- 84 cấu hình sparse được so với dense cùng layer;
- không có cấu hình sparse nào có measured latency thấp hơn dense cùng layer.

Nguyên nhân là sparse layer hiện tạo mask nhưng cuối cùng vẫn gọi dense PyTorch
operator. Operator không bỏ qua phép tính của các weight bằng 0.

### 4.2 Cost predictor

Predictor sử dụng ridge regression trên `log(1 + metric)`. Feature gồm:

- loại layer;
- channel input/output;
- kích thước input/output;
- kernel và group;
- parameter và MAC dense;
- mật độ `N/M`;
- tương tác giữa mật độ với parameter và MAC.

Predictor được đánh giá bằng leave-one-layer-out trên 105 điểm:

| Metric | MAE | MAPE | R² | Đánh giá |
| --- | ---: | ---: | ---: | --- |
| Latency | 0,255 ms | 69,85% | 0,577 | Chưa đủ tin cậy |
| Bandwidth | 782.320 byte | 43,97% | 0,446 | Sai số còn lớn |
| Memory | 357.821 byte | 47,28% | 0,689 | Chỉ nên tham khảo |

Vì sai số predictor cao và lookup đã bao phủ đầy đủ ứng viên, scheme cuối sử
dụng lookup. Predictor đã được triển khai nhưng chưa được dùng làm bằng chứng
chính cho kết luận tối ưu.

## 5. Scheme được chọn

Selector giải bài toán multiple-choice Pareto với:

- target composite cost reduction: 3%;
- loss cần tối thiểu hóa: số parameter bị loại;
- cost mode: lookup.

Kết quả đạt composite cost reduction dự đoán `3,102%`. Scheme giữ 16 layer ở
`16:16` và đặt `1:16` cho năm layer:

```text
SparseConv11_256-256-(3, 3)
SparseConv13_256-256-(3, 3)
SparseConv14_256-256-(3, 3)
SparseConv16_512-512-(3, 3)
Linear0_512-1000
```

Scheme dự đoán giảm:

- parameter: 37,25% trong manifest; benchmark thực tế ghi 37,22%;
- MAC: 23,92%;
- composite hardware cost: 3,102%.

Việc dùng `1:16` tương đương chỉ giữ một weight trong mỗi nhóm 16 weight. Đây là
mức pruning rất mạnh đối với các layer được chọn.

## 6. Giao thức huấn luyện và benchmark

Dense và model pruning sử dụng cùng:

- kiến trúc ResNet-18;
- dense checkpoint ban đầu;
- ImageNet validation đủ 50.000 ảnh;
- preprocessing và input size;
- Tesla T4 và phiên bản phần mềm;
- seed 42;
- batch latency 1, warm-up 30, đo 100 iteration.

Fine-tune sử dụng:

- 3 epoch;
- 50.000 training sample mỗi epoch;
- validation nội bộ 1.000 sample;
- batch size 256;
- base learning rate 0,01;
- weight decay 0,002.

Kết quả trước và sau fine-tune đều được benchmark lại trên đủ 50.000 ảnh, không
dùng validation 1.000 ảnh làm kết quả cuối.

## 7. Kết quả thực tế

| Model | Top-1 % | Top-5 % | Param giảm % | MAC giảm % | Median ms | P95 ms | sample/s | Peak MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense baseline | 69,754 | 89,078 | 0,00 | 0,00 | 3,725 | 5,281 | 268,44 | 93,19 |
| Pruning trước fine-tune | 0,250 | 0,900 | 37,22 | 23,92 | 6,709 | 7,272 | 149,05 | 113,96 |
| Pruning sau fine-tune | 51,962 | 77,920 | 37,22 | 23,92 | 6,923 | 7,954 | 144,45 | 113,96 |

Fine-tune phục hồi `51,712` điểm Top-1 so với model ngay sau pruning. Tuy nhiên,
kết quả cuối vẫn thấp hơn dense `17,792` điểm Top-1.

Parameter và MAC hiệu dụng giảm đáng kể, nhưng latency không giảm:

- median latency tăng từ 3,725 ms lên 6,923 ms;
- throughput giảm từ 268,44 xuống 144,45 sample/s;
- peak device memory tăng từ 93,19 MB lên 113,96 MB.

Do đó model này không được gọi là nhanh hơn hoặc tối ưu hơn dense trên T4.

## 8. Kết luận đối với yêu cầu

Yêu cầu triển khai đã hoàn thành trên T4:

- có hàm cost phần cứng;
- có profile và lookup theo từng `layer × N:M`;
- có cost predictor và chỉ số đánh giá sai số;
- có cơ chế chọn mixed N:M theo cost;
- có benchmark đầy đủ trước/sau fine-tune.

Kết quả thực nghiệm là một kết quả âm nhưng hợp lệ. Pipeline hoạt động, tuy nhiên
objective hiện tại chọn pruning quá mạnh ở các layer nhạy và dense operator không
khai thác sparsity. Vì vậy scheme thử nghiệm chưa tạo ra điểm accuracy–complexity
tốt và không tạo runtime speedup trên T4.

Hướng cải tiến trực tiếp là thêm sensitivity/accuracy loss vào selector, giới hạn
N tối thiểu cho các layer nhạy và so sánh với Uniform N:M tại cùng effective MAC.

## 9. File nguồn và khả năng truy vết

Code chính:

```text
search/profile_layer_hardware.py
search/hardware_cost.py
search/select_scheme_from_hardware_profile.py
search/find_mix_from_dense_imagenet.py
```

Artifact trên Google Drive:

```text
MyDrive/DominoSearch-artifacts/results/
├── t4-resnet18-layer-cost-latency-20260805.json
├── hardware-profile-selector-target3-before-20260805.json
└── hardware-profile-selector-target3-after-20260805.json

MyDrive/DominoSearch-artifacts/reports/
└── hardware-profile-selector-target3-comparison-20260805.md
```

Checkpoint cuối:

```text
SHA-256: d0489224d1c67f184869ce11d05f579a81cca3b61590a8f4a24ab7749520c02a
```

Các artifact benchmark không được commit vào Git; chỉ tài liệu và code được lưu
trong repository.
