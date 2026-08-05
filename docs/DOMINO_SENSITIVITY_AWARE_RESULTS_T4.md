# Kết quả sensitivity-aware Domino mixed N:M trên Tesla T4

> Ngày chạy: 2026-08-05
>
> Nhánh thí nghiệm: `pruning-domino-mixed-nm`
>
> Commit được benchmark: `28d7cd730dbf17955d28ca83083a54b656874bd3`

## 1. Kết luận ngắn

Scheme sensitivity-aware MAC15 sau ba epoch fine-tune đạt **69,648% Top-1** trên
đủ 50.000 ảnh ImageNet validation. So với dense 69,754%, model chỉ mất **0,106
điểm phần trăm Top-1**, trong khi giảm **15,135% effective MAC** và **13,876%
effective parameter**.

Đây là điểm accuracy–MAC tốt nhất hiện có của **Domino mixed N:M** trong các run
đã kiểm tra. Nó không chứng minh model chạy nhanh hơn trên T4: median latency
13,396 ms vẫn chậm hơn dense 3,778 ms khoảng 3,55 lần. Nguyên nhân là sparse
layer hiện tạo mask rồi gọi dense PyTorch operator.

Không được diễn giải kết quả này thành speedup trên Jetson, CPU hoặc FPGA. Các
phần cứng đó phải được profile và benchmark trực tiếp.

## 2. Mục tiêu và thay đổi so với selector cũ

Selector Domino cũ có thể ưu tiên giảm parameter, MAC hoặc hardware cost mà
không biết layer nào nhạy với pruning. Run hardware-profile-selector trước đây
đã giảm mạnh một số layer nhạy và làm Top-1 rất thấp.

Pipeline mới tách hai tín hiệu:

1. hardware profile đo cost của từng cặp `layer × N:M`;
2. sensitivity profile đo mức giảm accuracy khi chỉ prune một layer;
3. selector tìm scheme đạt budget resource với sensitivity ước lượng thấp nhất;
4. first convolution và classifier được bảo vệ;
5. scheme được kiểm tra end-to-end trước và sau fine-tune.

Không có quantization, distillation hoặc thay kiến trúc trong thí nghiệm này.

## 3. Dữ liệu dùng để chọn scheme

Sensitivity profile dùng 1.000 ảnh validation với ứng viên `2:4`, `3:4`, `4:4`
cho 21 sparse layer. Đây là **profile debug dùng để xếp hạng layer**, không phải
kết quả accuracy cuối. Một lần profile 5.000 ảnh đã vượt thời lượng runtime và
không tạo artifact hoàn chỉnh, nên không được dùng làm bằng chứng.

Hardware profile đo 21 layer × 3 candidate trên cùng T4, batch 1, 30 warm-up,
100 iteration và 7 repeat. Không có candidate sparse nào nhanh hơn dense tại
cùng layer. Predictor latency có leave-one-layer-out MAPE 94,950% và R² 0,617,
nên không đủ tin cậy để chọn scheme. Vì vậy vòng này dùng target MAC; hardware
lookup chỉ ghi nhận giới hạn của implementation PyTorch hiện tại.

Sweep end-to-end 5.000 ảnh sau đó cho kết quả:

| Target scheme | Top-1 5k | Giảm parameter | Giảm MAC | Quyết định |
| --- | ---: | ---: | ---: | --- |
| MAC10 | 68,86% | 11,04% | 10,36% | Giữ để tham khảo |
| MAC15 | 68,74% | 13,88% | 15,14% | Chọn fine-tune |
| MAC20 | 67,28% | 16,78% | 20,00% | Không chọn |
| MAC23 | 66,26% | 18,05% | 23,10% | Loại |

MAC15 là điểm cân bằng: so với MAC10, nó giảm thêm khoảng 4,78 điểm phần trăm
MAC nhưng chỉ mất 0,12 điểm Top-1 trên subset. MAC23 không đạt cổng Top-1 67,0%
và kết quả full trước fine-tune chỉ đạt 65,116%, nên không được fine-tune trong
vòng này.

## 4. Scheme MAC15

Scheme giữ dense 11/21 sparse layer và dùng 3:4 cho 10/21 layer:

```text
4:4: conv0, conv1, conv4, conv7, conv10, conv12, conv14,
     conv15, conv16, conv17, linear0
3:4: conv2, conv3, conv5, conv6, conv8, conv9, conv11,
     conv13, conv18, conv19
```

Không có layer 2:4 trong scheme được chọn. Benchmark xác nhận scheme bao phủ đủ
21 layer và checkpoint load với `missing_keys=[]`, `unexpected_keys=[]`.

## 5. Protocol fine-tune

| Thuộc tính | Giá trị |
| --- | --- |
| Dense checkpoint | `resnet18-dense.pth` |
| Scheme | `domino-sensitive-mac15-20260805.txt` |
| Train sample mỗi epoch | 50.000 |
| Validation nội bộ mỗi epoch | 1.000 |
| Epoch | 3 |
| Base learning rate | 0,001 |
| Seed | 42 |
| Batch size từ config | 256 |
| Train shards nhìn thấy | 294/294 |
| Validation shards nhìn thấy | 14/14 |
| Data transport khi train | rclone mount, VFS full cache |

Validation nội bộ 1.000 ảnh lần lượt đạt 72,2%, 72,1% và 72,2%. Sau khi train,
cả ba checkpoint được screen lại trên cùng prefix 5.000 ảnh:

| Checkpoint | Top-1 5k | Top-5 5k |
| --- | ---: | ---: |
| Epoch 1 | 71,22% | 89,58% |
| Epoch 2 | 71,10% | 89,94% |
| Epoch 3 | **71,30%** | 89,88% |

Epoch 3 được chọn theo Top-1. Các số 1.000/5.000 ảnh chỉ dùng để chọn checkpoint;
kết luận dưới đây dùng đủ 50.000 ảnh.

## 6. Kết quả hợp lệ trên 50.000 ảnh

Mọi run trong bảng dùng cùng ResNet-18, preprocessing, ImageNet validation,
Tesla T4, PyTorch 2.11.0+cu128, CUDA 12.8, seed 42, performance batch 1, warm-up
30 và 100 iteration đo latency.

| Run | Top-1 | Top-5 | Giảm parameter | Giảm MAC | Median | P95 | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 69,754% | 89,080% | 0% | 0% | 3,778 ms | 4,734 ms | 264,69 mẫu/s |
| MAC15 trước fine-tune | 67,930% | 87,916% | 13,876% | 15,135% | 14,017 ms | 14,991 ms | 71,34 mẫu/s |
| MAC15 epoch 3 sau fine-tune | **69,648%** | **89,190%** | **13,876%** | **15,135%** | 13,396 ms | 14,193 ms | 74,65 mẫu/s |

Fine-tune phục hồi **1,718 điểm Top-1** và **1,274 điểm Top-5** so với checkpoint
pruned ban đầu. So với dense, kết quả cuối mất 0,106 điểm Top-1 nhưng Top-5 cao
hơn 0,110 điểm. Mức giảm effective parameter/MAC không đổi, đúng với scheme.

## 7. So với các Domino/Uniform run trước

| Phương án sau fine-tune | Top-1 | Giảm parameter | Giảm MAC | Nhận xét |
| --- | ---: | ---: | ---: | --- |
| Sensitivity-aware MAC15 | **69,648%** | 13,876% | 15,135% | Accuracy cao nhất trong Domino mixed hiện có |
| Uniform 3:4 conservative | 68,388% | 23,494% | 23,101% | Giảm resource nhiều hơn nhưng mất thêm 1,260 điểm Top-1 |
| Domino params-23 | 67,996% | 30,275% | 9,559% | Giảm parameter mạnh, nhưng giảm MAC ít và accuracy thấp hơn |

MAC15 nằm trên Pareto frontier accuracy–MAC: nó ưu tiên giữ accuracy, còn
Uniform 3:4 ưu tiên giảm MAC nhiều hơn. Hai model không cùng MAC budget, vì vậy
không được kết luận MAC15 thắng Uniform ở cùng budget. Scheme sensitivity-aware
MAC23 cùng khoảng 23,1% MAC đã thất bại cổng accuracy trước fine-tune; cần cải
thiện cách ước lượng tương tác giữa layer trước khi thử lại budget này.

So với Domino params-23, MAC15 vừa có Top-1 cao hơn 1,652 điểm vừa giảm MAC nhiều
hơn 5,576 điểm phần trăm, nhưng params-23 giảm parameter nhiều hơn. Việc chọn
model phụ thuộc constraint thực tế là compute hay storage.

## 8. Diễn giải runtime và memory

Host speedup của MAC15 sau fine-tune là `3,778 / 13,396 = 0,282×`, tức không có
speedup. Peak device memory đo được là 114,056 MB so với dense 93,187 MB. Đây là
overhead của implementation mask+dense hiện tại, không phải memory cần thiết của
một sparse kernel hoặc FPGA implementation.

Checkpoint fine-tune khoảng 93,6 MB vì chứa state phục vụ resume/optimizer. Nó
không phải sparse model đã nén; tensor weight vẫn được lưu dense. Effective MAC
và parameter là số lý thuyết theo N:M.

## 9. Artifact

Các artifact không được commit vào Git. Chúng nằm trên Google Drive dưới
`DominoSearch-artifacts/`:

```text
profiles/resnet18-m4-sensitivity-1k-debug-20260805.json
profiles/resnet18-m4-t4-hardware-20260805.json
schemes/domino-sensitive-mac15-20260805.txt
schemes/domino-sensitive-mac15-20260805.txt.json
results/dense-full-opt3-20260805.json
results/domino-sensitive-mac15-full-before-opt3-20260805.json
results/domino-sensitive-mac15-epoch3-full-after-20260805.json
results/domino-mac15-selected-full-summary-20260805.json
results/debug/domino-mac15-checkpoint-screen-20260805.json
logs/domino-sensitive-mac15-lr001-3epoch-train50k-20260805.log
runs/domino-sensitive-mac15-lr001-3epoch-train50k-20260805/model.pth-1
runs/domino-sensitive-mac15-lr001-3epoch-train50k-20260805/model.pth-2-local-backup
runs/domino-sensitive-mac15-lr001-3epoch-train50k-20260805/model.pth-3
```

Checkpoint SHA-256:

```text
epoch 1: 55aa8ae25eaac5199dc9128a7a25f0dd0b1c8ba9ab466a5249706e9ded0a85b3
epoch 2: 2f14fbc8846b10c18e852e8aa54530f784ea39b81711e0645ba44b0f08be3ce0
epoch 3: 556fc1af171f3fc12c865233d31d7697d1474c77ba52a3859aa624eba3c9b36f
```

## 10. Hướng tiếp theo

1. Dùng staged/conditioned re-profiling từ checkpoint MAC15 để giảm sai số do
   cộng độc lập sensitivity giữa các layer. Code và protocol nằm tại
   [`DOMINO_STAGED_MIXED_NM.md`](DOMINO_STAGED_MIXED_NM.md).
2. Tạo thêm target MAC17–18 để tìm điểm nằm giữa MAC15 và MAC20, chỉ fine-tune
   nếu full pre-fine-tune đạt cổng accuracy.
3. Muốn so trực tiếp Uniform 3:4, phải cải thiện scheme ở đúng khoảng 23,1% MAC
   và dùng cùng fine-tune budget.
4. Khi có Jetson Nano, đo lại lookup `layer × N:M` trên chính Jetson và chỉ cho
   selector dùng pattern/kernel mà runtime mục tiêu khai thác.
5. Nếu Jetson không có sparse kernel phù hợp, structured compact pruning là
   hướng triển khai khác; không trộn kết quả đó vào báo cáo mixed N:M này.

## 11. Kết luận

Thí nghiệm chứng minh sensitivity-aware selection kết hợp fine-tune LR thấp có
thể tạo ResNet-18 mixed N:M giảm 15,135% effective MAC với gần như giữ nguyên
Top-1. Nó là **ứng viên sẵn sàng để chuyển sang đo trên phần cứng mục tiêu**, chưa
phải bằng chứng model nhanh hơn hay nhỏ hơn trên T4/Jetson/FPGA.
