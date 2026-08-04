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

## 9. Hướng nghiên cứu tiếp theo

### 9.1 Vấn đề cần giải quyết

Cost hiện tại trả lời cấu hình nào giảm latency, bandwidth hoặc memory, nhưng chưa
đánh giá pruning layer đó làm accuracy giảm bao nhiêu. Vì vậy selector đã chọn
`1:16` cho một số layer nhạy. Cấu hình này xóa 15 trong mỗi 16 weight, làm Top-1
giảm từ 69,754% xuống 0,250% trước fine-tune.

Phiên bản tiếp theo cần tối ưu đồng thời hai mục tiêu:

1. giảm hardware cost;
2. giới hạn tổn thất accuracy.

### 9.2 Đo sensitivity cho từng layer

Sensitivity biểu diễn mức độ accuracy hoặc loss bị ảnh hưởng khi chỉ prune một
layer. Với mỗi cặp `layer × N:M`:

1. bắt đầu từ cùng dense checkpoint;
2. chỉ áp dụng N:M cho layer đang đánh giá;
3. giữ tất cả layer còn lại dense;
4. đánh giá trên cùng validation subset 1.000–5.000 ảnh;
5. ghi lại mức tăng loss và mức giảm Top-1;
6. khôi phục dense checkpoint trước khi đo cấu hình kế tiếp.

Hai định nghĩa có thể lưu đồng thời:

```text
SensitivityLoss_i(N:M) = Loss_pruned_i(N:M) - Loss_dense

SensitivityTop1_i(N:M) = Top1_dense - Top1_pruned_i(N:M)
```

Giá trị càng lớn nghĩa là layer càng nhạy và cần được giữ dense hoặc chỉ prune
nhẹ. Artifact sensitivity phải ghi checkpoint, dataset subset, seed, preprocessing,
scheme tạm thời và kết quả của mọi layer.

### 9.3 Kết hợp sensitivity với hardware cost

Objective mới có thể dùng tổng có trọng số:

```text
Score_i(N:M) = HardwareCost_i(N:M) + λ × Sensitivity_i(N:M)
```

Trong đó `λ` điều khiển mức ưu tiên bảo vệ accuracy. `λ` lớn tạo scheme an toàn
hơn; `λ` nhỏ ưu tiên giảm tài nguyên mạnh hơn.

Một lựa chọn khác là xếp hạng theo lợi ích trên tổn thất:

```text
Benefit_i(N:M)
    = HardwareCostReduction_i(N:M) / (Sensitivity_i(N:M) + ε)
```

Cấu hình tốt phải giảm được nhiều cost nhưng chỉ gây sensitivity nhỏ. Cả hai cách
đều phải được benchmark end-to-end; score không thay thế accuracy thực tế.

### 9.4 Giới hạn an toàn cho search

Không nên cho selector dùng `1:16` tự do trong lần thử đầu. Tập ứng viên an toàn:

```text
8:16, 12:16, 16:16
```

Hoặc với pattern nhỏ hơn:

```text
2:4, 3:4, 4:4
```

Các ràng buộc ban đầu:

- giữ convolution đầu tiên dense;
- giữ linear cuối dense;
- không dùng cấu hình thấp hơn `8:16` cho layer nhạy;
- chỉ thay đổi một số ít layer trong mỗi bước search;
- loại scheme nếu Top-1 trên validation subset giảm quá ngưỡng;
- fail rõ ràng nếu target cost không khả thi, không tự chọn scheme phá accuracy.

### 9.5 Search với accuracy budget

Bài toán nên được viết thành tối ưu có ràng buộc:

```text
minimize HardwareCost(scheme)

subject to:
    EstimatedTop1(scheme) >= Top1_dense - AccuracyBudget
    MACReduction(scheme)  >= TargetMACReduction
```

Thử nghiệm đầu có thể dùng:

```text
AccuracyBudget     = 1,0 điểm Top-1
TargetMACReduction = khoảng 23%
```

Với dense Top-1 69,754%, scheme chỉ được chấp nhận ở bước search nhanh nếu Top-1
ước lượng hoặc subset không thấp hơn 68,754%. Kết luận cuối vẫn phải dựa trên
benchmark đủ 50.000 validation ảnh.

### 9.6 So sánh công bằng với Uniform N:M

Uniform 3:4 hiện là mốc phù hợp:

| Scheme | Top-1 sau FT % | MAC giảm % |
| --- | ---: | ---: |
| Uniform 3:4 | 68,388 | 23,10 |
| Hardware-aware hiện tại | 51,962 | 23,92 |

Hai scheme có MAC reduction gần nhau, nhưng hardware-aware hiện mất accuracy lớn
hơn. Phiên bản mới phải được tìm ở cùng budget khoảng 23% MAC để kiểm tra mixed
N:M có phân bổ sparsity tốt hơn Uniform hay không.

### 9.7 Fine-tune và tiêu chí nhận kết quả

Sau khi tìm được scheme an toàn hơn:

- benchmark đủ 50.000 ảnh trước fine-tune;
- fine-tune với learning rate ban đầu `0,001` thay vì `0,01`;
- giữ mask cố định trong thử nghiệm đầu;
- dùng cùng 3 epoch × 50.000 training sample;
- lưu và đánh giá checkpoint từng epoch;
- xác nhận sparsity không đổi sau fine-tune;
- chỉ nhận checkpoint nếu accuracy tăng so với trước fine-tune.

Nếu cần, gradual pruning có thể là thí nghiệm riêng sau khi fixed-mask fine-tune
đã có baseline. Không trộn hai cơ chế vào cùng một kết quả.

### 9.8 Tiêu chí thành công đề xuất

Mục tiêu cho vòng thí nghiệm tiếp theo:

```text
MAC reduction:              khoảng 23%
Top-1 trước fine-tune:      >= 67,0%
Top-1 sau fine-tune:        >= 68,4%
Sai lệch sparsity yêu cầu:  0%
```

Mốc Top-1 sau fine-tune 68,4% được chọn để ít nhất cạnh tranh với Uniform 3:4
68,388% ở cùng MAC budget. Runtime chỉ được gọi là cải thiện nếu median/P95
latency và throughput được đo tốt hơn dense qua nhiều lần lặp.

### 9.9 Thứ tự triển khai

1. Viết profiler sensitivity cho từng `layer × N:M`.
2. Sinh sensitivity lookup table có provenance đầy đủ.
3. Thêm sensitivity term và accuracy budget vào selector.
4. Áp dụng boundary protection và giới hạn N tối thiểu.
5. Search scheme ở khoảng 23% MAC reduction.
6. Chạy validation subset để loại scheme không đạt accuracy budget.
7. Benchmark scheme còn lại trên đủ 50.000 ảnh trước fine-tune.
8. Fine-tune với learning rate thấp và mask cố định.
9. Benchmark đủ 50.000 ảnh sau fine-tune.
10. So sánh trực tiếp với dense và Uniform 3:4 bằng cùng protocol.

## 10. File nguồn và khả năng truy vết

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
