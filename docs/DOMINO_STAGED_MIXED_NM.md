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

Profiler ghi tiến độ nguyên tử vào `<output>.partial.json` sau từng candidate.
Nếu Colab bị ngắt, chạy lại đúng lệnh và thêm `--resume`; chương trình chỉ tái sử
dụng các row khi checkpoint SHA-256, base scheme, dataset manifest, candidate,
seed và toàn bộ cấu hình profile khớp. Khi profile hoàn tất, JSON cuối được ghi
vào `--output` và partial file được xóa. Không dùng partial JSON cho selector hay
báo cáo kết quả.

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
| ------ | ----------------------------: | --------------------------: |
| MAC18  |                         67,5% |                       69,3% |
| MAC20  |                         67,0% |                       69,0% |
| MAC23  |                         66,5% |                       68,4% |

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
| ------------ | -------: | -------------: | -------------------------: | -------------------------: |
| MAC18        |  18,322% |        15,230% |               3,186 điểm % |             0,0 điểm Top-1 |
| MAC20        |  20,711% |        16,177% |               5,576 điểm % |             0,5 điểm Top-1 |
| MAC23        |  23,101% |        23,752% |               7,966 điểm % |             1,5 điểm Top-1 |

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

| Run                         |       Top-1 |       Top-5 | Giảm parameter |    Giảm MAC |    Median |       P95 |   Throughput |
| --------------------------- | ----------: | ----------: | -------------: | ----------: | --------: | --------: | -----------: |
| Dense                       |     69,754% |     89,080% |             0% |          0% |  3,778 ms |  4,734 ms | 264,69 mẫu/s |
| MAC15 epoch 3               |     69,648% |     89,190% |        13,876% |     15,135% | 13,396 ms | 14,193 ms |  74,65 mẫu/s |
| MAC18 trước fine-tune       |     69,536% |     89,078% |        15,216% |     18,322% | 15,714 ms | 17,039 ms |  63,64 mẫu/s |
| MAC18 epoch 1 sau fine-tune | **69,654%** | **89,124%** |    **15,216%** | **18,322%** | 15,552 ms | 17,250 ms |  64,30 mẫu/s |

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

| Checkpoint |    Top-1 5k |    Top-5 5k | Quyết định      |
| ---------- | ----------: | ----------: | --------------- |
| Epoch 1    | **71,560%** |     89,860% | Chọn theo Top-1 |
| Epoch 2    |     71,500% | **90,040%** | Không chọn      |

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

## 10. Kế hoạch thí nghiệm tiếp theo

### 10.1. Mốc xuất phát

Đóng băng MAC18 epoch 1 làm checkpoint đầu vào của vòng tiếp theo:

```text
Top-1:                       69,654%
Top-5:                       89,124%
Effective MAC reduction:    18,322%
Effective param reduction:  15,216%
```

Không quay lại dense hoặc MAC15 để sinh stage mới. Dense, MAC15 và MAC18 vẫn
được giữ làm các mốc so sánh.

### 10.2. Conditioned profile từ MAC18

1. Load exact checkpoint MAC18 epoch 1 và scheme MAC18.
2. Giữ nguyên MAC18 làm cấu hình nền.
3. Lần lượt thử prune thêm từng layer bằng các candidate N:M hợp lệ.
4. Đo trên cùng 5.000 validation image, cùng preprocessing và seed 42.
5. Với mỗi candidate, ghi lại layer, cấu hình nền/thử nghiệm, Top-1/Top-5 giảm,
   MAC/parameter giảm thêm, checkpoint, scheme, branch và commit.
6. Xác nhận profile bao phủ chính xác 21 sparse layer và không có candidate bị
   bỏ qua âm thầm.

Profile 1.000 ảnh chỉ được dùng để debug pipeline. Scheme dùng cho kết luận phải
được sinh từ profile 5.000 ảnh hoàn chỉnh hoặc được ghi rõ là debug nếu runtime
không cho phép hoàn tất.

### 10.3. Sinh ứng viên MAC20

Selector tìm scheme monotonic đạt khoảng 20–21% effective MAC reduction với
sensitivity cộng ước lượng nhỏ nhất:

- không layer nào được chuyển về cấu hình dense hơn MAC18;
- chỉ prune thêm các layer ít nhạy;
- first convolution và classifier tiếp tục được bảo vệ nếu chưa có bằng chứng
  đủ mạnh để prune;
- scheme phải khớp chính xác toàn bộ layer mà model mong đợi;
- scheme MAC20 debug cũ từ profile MAC15 1.000 ảnh không được dùng làm kết quả
  cuối.

### 10.4. Ba cổng đánh giá trước fine-tune

**Cổng A — 1.000 ảnh debug:** kiểm tra checkpoint load exact, scheme/mask đúng,
sparsity đo được khớp yêu cầu và accuracy không sụp đổ. Kết quả này không được
đưa vào bảng kết luận.

**Cổng B — 5.000 ảnh screening:** so MAC18 và MAC20 trên cùng prefix dataset.
Nếu MAC20 giảm accuracy quá mạnh, quay lại selector và tạo scheme bảo thủ hơn.

**Cổng C — đủ 50.000 ảnh:** chỉ cho phép fine-tune khi MAC20 đạt cả hai ngưỡng
đề xuất:

```text
Top-1 trước fine-tune >= 68,75%
Effective MAC reduction >= 20%
```

Các ngưỡng này là tiêu chí go/no-go của vòng nghiên cứu tiếp theo, không phải số
liệu đã đo. Nếu không đạt, MAC20 bị loại hoặc phải được sinh lại.

### 10.5. Fine-tune công bằng và ổn định dữ liệu

MAC20 vượt qua full-val gate sẽ dùng cùng protocol với MAC15:

```text
Epoch:                    3
Learning rate:            0,001
Training sample/epoch:    50.000
Internal validation:      1.000 ảnh
Seed:                     42
```

Để tránh lặp lại lỗi rclone FUSE ở MAC18:

1. tạo manifest cố định của đúng 50.000 training sample;
2. copy trước các shard cần thiết vào disk local của Colab;
3. không để DataLoader đọc training shard trực tiếp qua FUSE;
4. lưu checkpoint local sau mỗi epoch rồi upload ngay lên Drive;
5. ghi SHA-256 và kiểm tra file sau upload;
6. hỗ trợ resume từ checkpoint gần nhất.

Nếu cần thêm manifest/resume hoặc thay đổi data loader dùng chung cho nhiều
nhánh, thay đổi đó phải được làm và commit trên `master` trước, sau đó mới đồng
bộ vào nhánh thí nghiệm này.

### 10.6. Chọn checkpoint và full benchmark

Screen mọi checkpoint hợp lệ trên cùng 5.000 ảnh, chọn theo Top-1 đã định trước,
rồi benchmark checkpoint được chọn trên đủ 50.000 ảnh. Bảng cuối phải có dense,
MAC15, MAC18, MAC20 và Uniform 3:4, cùng các trường:

- Top-1 và Top-5;
- dense/effective parameter và MAC;
- median/P95 latency và throughput;
- peak device memory;
- checkpoint SHA-256, scheme, branch, commit và seed.

Mục tiêu chấp nhận đề xuất cho MAC20 sau fine-tune là:

```text
Top-1 >= 69,45%                 # mất không quá khoảng 0,30 điểm so với dense
Effective MAC reduction >= 20%
```

MAC20 chỉ được gọi là tốt hơn về accuracy–complexity nếu đạt các ngưỡng này và
giảm resource nhiều hơn MAC18. Nếu không đạt, MAC18 vẫn là model được chọn.

### 10.7. Khi có Jetson

1. Benchmark từng cặp `layer × N:M` trên chính Jetson.
2. Thu thập latency, throughput, memory, power/energy và runtime version.
3. Xây hardware lookup table hoặc cost predictor được kiểm định.
4. Tính cost theo constraint triển khai, ví dụ:

   ```text
   Cost = alpha*latency + beta*energy + gamma*bandwidth + delta*memory
   ```

5. Chạy lại selector với cost Jetson và chỉ dùng pattern/kernel mà runtime trên
   board thực sự hỗ trợ.
6. So sánh scheme MAC-based với scheme Jetson-cost-based bằng benchmark
   end-to-end trên cùng board.

Chỉ benchmark trực tiếp trên Jetson mới cho phép kết luận speedup, memory hoặc
energy thực tế. Kết quả effective MAC trên T4 không thay thế bước này.

## 11. Kết quả MAC20 ngày 2026-08-12

Conditioned profile được chạy từ exact MAC18 epoch-1 checkpoint trên 5.000 ảnh:

- baseline Top-1/Top-5: 71,560%/89,860%;
- đủ 21 layer và 51 candidate monotonic;
- checkpoint SHA-256
  `3f4a81f0f0eec1a1f4c3ce2d16b4b2884f02d01730ea449d95c5785932058ca0`;
- checkpoint load exact, không có missing/unexpected key;
- đủ 14 validation shard, tổng 6.693.093.726 byte.

Selector target MAC 20% chỉ chuyển thêm hai layer từ 4:4 sang 3:4:

```text
SparseConv12_128-256-(1, 1)
SparseConv16_512-512-(3, 3)
```

Scheme đạt 20,003% giảm effective MAC, 20,332% giảm effective parameter và
estimated additive sensitivity 0,32 điểm Top-1. T4 hardware lookup cũ chỉ cung
cấp shape/MAC cho selector; measured masked-dense latency không được dùng làm
speedup objective.

### 11.1. Full validation trước và sau fine-tune

| Run | Top-1 | Top-5 | Giảm parameter | Giảm MAC | Median T4 | P95 T4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MAC20 trước fine-tune | 69,534% | 89,096% | 20,332% | 20,003% | 20,566 ms | 21,495 ms |
| MAC20 epoch 1 sau fine-tune | **69,660%** | **89,130%** | **20,332%** | **20,003%** | 19,948 ms | 21,510 ms |

Fine-tune dùng LR 0,001, ba epoch, 50.000 train sample/epoch, validation nội bộ
1.000 ảnh và seed 42. Mười hai train shard local chứa 52.296 row; loader được
gọi với `--train-expected-shards 12` và dừng đúng quota 50.000. Cả ba checkpoint
được screen trên cùng 5.000 ảnh:

| Epoch | Top-1 5k | Top-5 5k | SHA-256 |
| --- | ---: | ---: | --- |
| 1 | **71,400%** | 89,780% | `ef982b4c1f85ddaf452528587cd1d5deae56452b0588ae502d56a8af652a070b` |
| 2 | 71,340% | **89,960%** | `c902517f295c997dc5b3c20fc56a448c2f3ecb3bf960fbd67a81c942f4553032` |
| 3 | 71,260% | 89,600% | `442a9a99e8d7ff6688e11798502b1a9c0f2686d0ec47f965df069ce084fc41c9` |

Epoch 1 được chọn theo Top-1 đã định trước. Full Top-1 69,660% vượt gate
69,45%, chỉ thấp hơn dense 0,094 điểm và gần ngang MAC18 epoch 1 (chênh 0,006
điểm, không đủ để tuyên bố accuracy cao hơn). MAC20 giảm thêm khoảng 1,681 điểm
phần trăm MAC so với MAC18 nên hiện là điểm accuracy–complexity được chọn.

Latency T4 vẫn chậm hơn dense vì runtime tạo mask rồi gọi dense operator. Kết
quả này chỉ chứng minh accuracy và effective complexity, chưa chứng minh target
hardware improvement.

Artifact chính trên Drive:

```text
profiles/resnet18-m4-conditioned-mac18-5k-20260812.json
schemes/domino-staged-mac20-from-mac18-5k-20260812.txt[.json]
results/domino-staged-mac20-from-mac18-full-before-20260812.json
results/domino-staged-mac20-epoch1-full-after-20260812.json
runs/domino-staged-mac20-lr001-3epoch-train50k-20260812/model.pth-{1,2,3}
```

### 11.2. Kết quả stage MAC23

Conditioned profile tiếp theo bắt đầu từ MAC20 epoch 1, không quay lại checkpoint
dense hoặc MAC18. Profile 5.000 ảnh có baseline 71,400% Top-1 và selector chọn
thêm ba thay đổi:

```text
SparseConv10_128-256-(3, 3): 4:4 -> 3:4
SparseConv14_256-256-(3, 3): 3:4 -> 2:4
SparseConv15_256-512-(3, 3): 4:4 -> 3:4
```

Scheme đạt 23,190% giảm effective MAC, 24,747% giảm effective parameter và
estimated additive sensitivity 0,56 điểm Top-1. Full validation trước
fine-tune đạt 68,892% Top-1 và 88,682% Top-5, vượt gate 68,75%.

Fine-tune dùng cùng protocol ba epoch của MAC20. Screen 5.000 ảnh chọn epoch 1:

| Epoch | Top-1 5k | Top-5 5k | SHA-256 |
| --- | ---: | ---: | --- |
| 1 | **71,120%** | 89,640% | `e860b43630de8296b32fc6a3633592361a6c2451a33313d6aabced00e80763ac` |
| 2 | 70,920% | 89,540% | `67545e3dd7c67267e47c905860a2b3538d268a1d76e6549c5cffeec5b880ed79` |
| 3 | 70,800% | **89,760%** | `c92e3e94cd6e1fa5b970601726647b6b797317ae5384e283e22fea915bd42ba2` |

Full validation epoch 1 đạt 69,458% Top-1 và 88,994% Top-5. Nó thấp hơn dense
0,296 điểm Top-1 và thấp hơn MAC20 0,202 điểm, nhưng giảm thêm khoảng 3,186 điểm
phần trăm MAC so với MAC20. Vì vậy MAC20 và MAC23 đều là Pareto points:

- MAC20 ưu tiên accuracy: 69,660% Top-1, giảm 20,003% MAC;
- MAC23 ưu tiên complexity: 69,458% Top-1, giảm 23,190% MAC.

Masked-dense median T4 của MAC23 là 21,043 ms và không chứng minh sparse
speedup. Việc chọn một trong hai để deploy phải dựa trên lookup/kernel và
end-to-end measurement trên Jetson Nano.

Artifact MAC23 trên Drive:

```text
profiles/resnet18-m4-conditioned-mac20-5k-20260812.json
schemes/domino-staged-mac23-from-mac20-5k-20260812.txt[.json]
results/domino-staged-mac23-from-mac20-full-before-20260812.json
results/domino-staged-mac23-epoch1-full-after-20260812.json
runs/domino-staged-mac23-lr001-3epoch-train50k-20260812/model.pth-{1,2,3}
```

## 12. Nguồn tham khảo

- DominoSearch: `assets/DominoSearch.pdf`, đặc biệt mục tiêu complexity có thể là
  model size, FLOPs, latency hoặc energy và phần layer-wise penalty;
- [HALP: Hardware-Aware Latency Pruning](https://arxiv.org/abs/2110.10811);
- [GraNet: gradual pruning with neuroregeneration](https://arxiv.org/abs/2106.10404);
- [NVIDIA Jetson Nano specifications](https://developer.nvidia.com/embedded/jetson-nano);
- [TensorRT structured sparsity requirements](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/data-formats-tensors.html).
