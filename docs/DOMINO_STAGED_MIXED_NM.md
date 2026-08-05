# Domino staged mixed N:M: tối ưu mạnh hơn nhưng có kiểm soát

## 1. Kết luận thiết kế

Điểm MAC15 hiện là mốc an toàn nhất đã đo: ResNet-18 đạt 69,648% Top-1 sau ba
epoch fine-tune, giảm 15,135% effective MAC và 13,876% effective parameter.
Không nên nhảy trực tiếp từ dense lên MAC23 vì thử nghiệm trước chỉ còn 65,116%
Top-1 trước fine-tune.

Hướng mới dùng **staged/conditioned pruning**:

```text
dense -> MAC15 đã fine-tune -> profile lại -> MAC18 -> fine-tune
                              -> profile lại -> MAC20 -> fine-tune
                                              -> MAC23 nếu còn đạt cổng accuracy
```

Ở mỗi stage, độ nhạy của một layer được đo trong khi mọi layer khác giữ scheme
của stage trước. Selector chỉ được giữ nguyên hoặc tăng sparsity của từng layer;
nó không được làm một layer trở lại dense để bù cho layer khác. Cách này giảm sai
số tương tác giữa các layer so với profile từng layer từ model dense rồi cộng mọi
penalty lại. Profiler cũng bỏ qua candidate ít sparse hơn base để giảm thời gian
GPU mà không làm mất lựa chọn hợp lệ.

Đây vẫn hoàn toàn là mixed N:M pruning. Default profile từ dense và thuật toán
DominoSearch gốc không bị thay đổi.

## 2. Vì sao hướng này phù hợp hơn với board yếu

- Prune theo bước nhỏ cho phép dừng ở Pareto point cuối cùng còn giữ accuracy.
- Profile conditioned đo đúng trạng thái model đang có, thay vì giả định mọi
  layer khác dense.
- Target MAC được tăng dần nên không phải tốn fine-tune cho scheme đã fail rõ
  ngay trước fine-tune.
- Scheme và checkpoint của mỗi stage độc lập, nên có thể quay về MAC15 an toàn.

Tuy nhiên, effective MAC thấp hơn **không tự động làm Jetson Nano nhanh hơn**.
Nano dùng GPU Maxwell 128 CUDA cores; TensorRT chỉ có structured sparse
acceleration 2:4 trên Ampere trở lên. Implementation PyTorch hiện tại tạo mask
rồi gọi dense operator, do đó mục tiêu trước khi có board là tìm accuracy–MAC
Pareto tốt hơn, không tuyên bố latency speedup.

## 3. Stage 1: đo lại từ checkpoint MAC15

Các biến dưới đây chỉ minh họa path chuẩn trên Drive. Không ghi đè artifact cũ.

```bash
cd /content/DominoSearch

ART=/content/DominoSearch-artifacts
DATA=/content/drive/MyDrive/DominoSearch-data/imagenet-1k
BASE_SCHEME=$ART/schemes/domino-sensitive-mac15-20260805.txt
BASE_CKPT=$ART/runs/domino-sensitive-mac15-lr001-3epoch-train50k-20260805/model.pth-3
```

Smoke profile 1.000 ảnh trước:

```bash
python search/profile_layer_sensitivity.py \
  --model resnet18_sparse \
  --checkpoint "$BASE_CKPT" \
  --base-scheme-file "$BASE_SCHEME" \
  --m 4 --candidate-n 2 3 4 \
  --layout NHWC --device cuda \
  --dataset-format parquet \
  --parquet-root "$DATA" \
  --parquet-pattern 'data/validation-*.parquet' \
  --dataset-num-samples 50000 \
  --accuracy-batch-size 64 --workers 2 \
  --max-eval-samples 1000 --seed 42 \
  --output "$ART/profiles/resnet18-m4-conditioned-mac15-1k-debug.json"
```

Nếu baseline trong profile gần kết quả MAC15 đã đo, chạy profile 5.000 ảnh với
tên output mới. Profile 1.000 ảnh chỉ là debug; selector dùng cho candidate nên
lấy profile 5.000 ảnh.

## 4. Chọn ba target trung gian

Hardware profile phải có đúng cùng 21 layer và candidates `2:4`, `3:4`, `4:4`.
Chạy selector ba lần; mỗi output là file mới:

```bash
for TARGET in 0.18 0.20 0.23; do
  LABEL=${TARGET/0./mac}
  python search/select_sensitivity_aware_scheme.py \
    --hardware-profile "$ART/profiles/resnet18-m4-t4-hardware-20260805.json" \
    --sensitivity-profile "$ART/profiles/resnet18-m4-conditioned-mac15-5k.json" \
    --base-scheme-file "$BASE_SCHEME" \
    --target-metric macs \
    --target-reduction "$TARGET" \
    --sensitivity-metric top1_drop_percent \
    --minimum-n 2 \
    --protect-first-conv --protect-linear \
    --output "$ART/schemes/domino-staged-${LABEL}-from-mac15.txt"
done
```

Sidecar JSON ghi cả base scheme, SHA-256, mức giảm từ dense và phần giảm thêm so
với MAC15. Selector fail nếu profile không được conditioned từ đúng base scheme,
thiếu layer/candidate, hoặc một candidate làm layer bớt sparse.

## 5. Cổng đánh giá để tiết kiệm GPU

Đánh giá theo thứ tự MAC18, MAC20 rồi MAC23:

1. benchmark 1.000 ảnh, trạng thái `debug`;
2. chỉ khi Top-1 hợp lý mới benchmark đủ 50.000 ảnh trước fine-tune;
3. chỉ fine-tune nếu full pre-fine-tune đạt cổng;
4. fine-tune 3 epoch, LR `0.001`, fixed mask và cùng 50.000 train sample/epoch;
5. benchmark đủ 50.000 ảnh sau fine-tune;
6. stage tiếp theo bắt đầu từ checkpoint/scheme tốt nhất vừa được chấp nhận.

Cổng đề xuất:

| Target | Full Top-1 trước FT tối thiểu | Full Top-1 sau FT tối thiểu |
| --- | ---: | ---: |
| MAC18 | 67,5% | 69,3% |
| MAC20 | 67,0% | 69,0% |
| MAC23 | 66,5% | 68,4% |

Các ngưỡng là tiêu chí thí nghiệm, không phải kết quả đã đo. Không được ghi một
stage là thành công nếu chỉ đạt effective MAC mà chưa qua accuracy full-val.

## 6. Các hướng pruning tiếp theo sau staged sensitivity

Thứ tự ưu tiên nghiên cứu:

1. **Conditioned direct sensitivity**: hướng đã implement, dễ kiểm chứng nhất.
2. **Taylor/Fisher sensitivity**: dùng gradient để xếp hạng nhanh hơn, nhưng phải
   đo tương quan với direct validation delta trước khi dùng selector.
3. **Dynamic mask/regrowth trong fine-tune**: cho phép weight bị prune có cơ hội
   mọc lại, chạy thành thí nghiệm riêng so với fixed-mask hiện tại.
4. **Pareto sweep theo cost thật**: khi có board, thay target MAC bằng latency hoặc
   energy lookup đo trên board và không dùng phép quy đổi từ T4.
5. **Candidate runtime-aware**: loại N:M mà runtime mục tiêu không có kernel khai
   thác; việc giảm search space phải dựa trên profile thực tế.

Không đưa quantization, distillation hoặc đổi kiến trúc vào bảng pruning-only.
Nếu thử về sau, chúng phải là thí nghiệm kết hợp riêng.

## 7. Bằng chứng cần lưu cho mỗi stage

- scheme và sidecar manifest;
- conditioned sensitivity profile;
- checkpoint đầu vào và checkpoint sau từng epoch;
- JSON 1.000 ảnh debug;
- JSON 50.000 ảnh trước và sau fine-tune;
- cùng seed, preprocessing, batch size, warm-up và iterations;
- bảng so sánh với dense, MAC15 và Uniform 3:4;
- khi có board: latency median/P95, throughput, peak memory, power/energy, power
  mode, clocks, nhiệt độ và runtime version.

## 8. Tiêu chí kết luận

Một stage chỉ được gọi là **tốt hơn về accuracy–complexity** khi giảm MAC/parameter
nhiều hơn MAC15 và vẫn đạt cổng accuracy. Chỉ được gọi là **nhanh hơn trên board**
khi benchmark end-to-end trên chính board chứng minh latency/throughput tốt hơn
dense trong cùng điều kiện.

## 9. Trạng thái chạy ngày 2026-08-05

Pipeline đã được chạy trên Colab Tesla T4 từ commit `706a37f`; benchmark cuối
được ghi nhận ở commit `5e999fe`. Input được
kiểm tra trước khi chạy:

- 14 validation shard, tổng `6.693.093.726` byte;
- checkpoint MAC15 epoch 3 có SHA-256
  `556fc1af171f3fc12c865233d31d7697d1474c77ba52a3859aa624eba3c9b36f`;
- checkpoint load exact, không có missing/unexpected key;
- base scheme, hardware profile và conditioned profile cùng đủ 21 layer;
- branch `pruning-domino-mixed-nm`, worktree Colab sạch.

Conditioned profile 1.000 ảnh hoàn tất với baseline 72,600% Top-1, 89,500%
Top-5. Nó chứa 53 candidate row cho 21 layer và được lưu với trạng thái debug.
Profile 5.000 ảnh đạt baseline 71,340% Top-1 và đã đo đến layer 19/21 thì Colab
runtime bị mất (404/401). Không có JSON hoàn chỉnh, do đó partial profile **không
được dùng** để tạo scheme hoặc làm bằng chứng cuối.

Selector dùng profile 1.000 ảnh đã tạo ba debug scheme monotonic:

| Scheme debug | Giảm MAC | Giảm parameter | Giảm MAC thêm so với MAC15 | Sensitivity cộng ước lượng |
| --- | ---: | ---: | ---: | ---: |
| MAC18 | 18,322% | 15,230% | 3,186 điểm % | 0,0 điểm Top-1 |
| MAC20 | 20,711% | 16,177% | 5,576 điểm % | 0,5 điểm Top-1 |
| MAC23 | 23,101% | 23,752% | 7,966 điểm % | 1,5 điểm Top-1 |

MAC18 chỉ thay đổi hai layer so với MAC15:

```text
SparseConv4_64-64-(3, 3):     4:4 -> 3:4
SparseConv14_256-256-(3, 3):  4:4 -> 3:4
```

Nó được chọn vì đạt target với sensitivity proxy thấp nhất. Profile dùng để sinh
scheme vẫn mang nhãn debug do chỉ dùng 1.000 ảnh, nhưng MAC18 sau đó đã vượt qua
benchmark end-to-end đủ 50.000 ảnh trước và sau fine-tune.

### 9.1. Kết quả MAC18 đủ 50.000 ảnh

Mọi run dưới đây dùng cùng ResNet-18, ImageNet validation, preprocessing, seed
42, T4, performance batch 1, 30 warm-up và 100 iteration. Checkpoint đều load
exact với `missing_keys=[]`, `unexpected_keys=[]`.

| Run | Top-1 | Top-5 | Giảm parameter | Giảm MAC | Median | P95 | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 69,754% | 89,080% | 0% | 0% | 3,778 ms | 4,734 ms | 264,69 mẫu/s |
| MAC15 epoch 3 | 69,648% | 89,190% | 13,876% | 15,135% | 13,396 ms | 14,193 ms | 74,65 mẫu/s |
| MAC18 trước fine-tune | 69,536% | 89,078% | 15,216% | 18,322% | 15,714 ms | 17,039 ms | 63,64 mẫu/s |
| MAC18 epoch 1 sau fine-tune | **69,654%** | **89,124%** | **15,216%** | **18,322%** | 15,552 ms | 17,250 ms | 64,30 mẫu/s |

Fine-tune phục hồi 0,118 điểm Top-1. So với dense, MAC18 mất 0,100 điểm Top-1
nhưng giảm 18,322% effective MAC. So với MAC15, nó giảm thêm 3,186 điểm phần
trăm MAC và Top-1 cao hơn 0,006 điểm. Chênh lệch accuracy này quá nhỏ để tuyên
bố MAC18 chính xác hơn; hai model được xem là gần như ngang accuracy.

### 9.2. Fine-tune và sự cố epoch 3

MAC18 được khởi tạo đúng từ checkpoint MAC15 epoch 3 rồi fine-tune với LR
0,001, 50.000 training sample mỗi epoch và validation nội bộ 1.000 ảnh. Epoch 1
đạt 72,600% Top-1, epoch 2 đạt 72,800%. Epoch 3 đã bắt đầu nhưng rclone FUSE trả
`OSError: [Errno 5] Input/output error` trước validation và trước khi lưu
checkpoint. Vì vậy **không có kết quả MAC18 epoch 3 hợp lệ**.

Hai checkpoint hợp lệ được screen trên cùng 5.000 ảnh:

| Checkpoint | Top-1 5k | Top-5 5k | Quyết định |
| --- | ---: | ---: | --- |
| Epoch 1 | **71,560%** | 89,860% | Chọn theo Top-1 |
| Epoch 2 | 71,500% | **90,040%** | Không chọn |

Benchmark đủ 50.000 ảnh dùng checkpoint epoch 1 đúng theo tiêu chí Top-1 đã
định trước. Kết quả này hợp lệ cho protocol fine-tune một epoch được chọn, nhưng
chưa phải so sánh cùng budget ba epoch với MAC15.

### 9.3. Diễn giải đúng

MAC18 hiện là ứng viên accuracy–complexity tốt nhất trong các staged run đã đo:
nó gần giữ nguyên dense accuracy và giảm MAC nhiều hơn MAC15. Nó **không nhanh
hơn trên T4**. Median 15,552 ms chậm hơn dense khoảng 4,12 lần vì implementation
hiện tạo mask rồi gọi dense PyTorch operator. Effective MAC/parameter là mức
giảm lý thuyết; không phải bằng chứng speedup, file nhỏ hơn hoặc Jetson nhanh hơn.

Artifact đã lưu trên Drive:

```text
profiles/resnet18-m4-conditioned-mac15-1k-debug-20260805.json
logs/resnet18-m4-conditioned-mac15-1k-debug-20260805.log
logs/resnet18-m4-conditioned-mac15-5k-interrupted-partial-20260805.log
schemes/domino-staged-mac18-from-mac15-debug-20260805.txt[.json]
schemes/domino-staged-mac20-from-mac15-debug-20260805.txt[.json]
schemes/domino-staged-mac23-from-mac15-debug-20260805.txt[.json]
results/domino-staged-mac18-from-mac15-full-before-20260805.json
results/domino-staged-mac18-epoch1-full-after-20260805.json
results/domino-staged-mac18-comparison-20260805.md
results/domino-staged-mac18-comparison-20260805.csv
results/debug/domino-staged-mac18-epoch1-5k-screen-20260805.json
results/debug/domino-staged-mac18-epoch2-5k-screen-20260805.json
logs/domino-staged-mac18-lr001-3epoch-train50k-20260805.log
runs/domino-staged-mac18-lr001-3epoch-train50k-20260805/model.pth-1
runs/domino-staged-mac18-lr001-3epoch-train50k-20260805/model.pth-2
```

Checkpoint SHA-256:

```text
epoch 1: 3f4a81f0f0eec1a1f4c3ce2d16b4b2884f02d01730ea449d95c5785932058ca0
epoch 2: 9f9ede1e03b5b87699127891c82c4156d1cc0d717b04e0e204b0ce29d60e01d1
```

Bước tiếp theo là profile conditioned lại từ checkpoint MAC18 epoch 1, ưu tiên
5.000 ảnh nếu runtime cho phép, rồi mới sinh ứng viên khoảng MAC20. Không dùng
thẳng scheme MAC20 debug hiện có làm kết luận cuối vì nó được suy ra từ profile
1.000 ảnh của checkpoint MAC15 và chưa qua full-val gate.

Nguồn tham khảo:

- DominoSearch: `assets/DominoSearch.pdf`, đặc biệt mục tiêu complexity có thể là
  model size, FLOPs, latency hoặc energy và phần layer-wise penalty;
- [HALP: Hardware-Aware Latency Pruning](https://arxiv.org/abs/2110.10811);
- [GraNet: gradual pruning with neuroregeneration](https://arxiv.org/abs/2106.10404);
- [NVIDIA Jetson Nano specifications](https://developer.nvidia.com/embedded/jetson-nano);
- [TensorRT structured sparsity requirements](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/data-formats-tensors.html).
