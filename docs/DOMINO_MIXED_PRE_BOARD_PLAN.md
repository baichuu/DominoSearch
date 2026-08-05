# Domino mixed N:M: công việc hoàn tất trước khi có Jetson Nano

## 1. Mục tiêu của vòng này

Vòng thí nghiệm mới sửa đúng nhược điểm đã quan sát trên T4: selector cũ giảm
cost lý thuyết nhưng đặt sparsity rất mạnh vào layer nhạy, khiến Top-1 sau
fine-tune chỉ còn 51,962%. Thay vì chọn scheme chỉ theo parameter, MAC hoặc cost
phần cứng, quy trình mới dùng hai bảng độc lập:

1. **hardware profile**: chi phí của từng `layer × N:M`;
2. **sensitivity profile**: mức tăng loss và giảm Top-1/Top-5 khi chỉ prune layer
   đó, còn các layer khác giữ dense.

Bộ chọn tìm scheme có sensitivity ước lượng nhỏ nhất nhưng vẫn đạt target giảm
MAC hoặc hardware cost. Đây vẫn là pruning mixed N:M; không trộn quantization,
distillation hay thay kiến trúc.

## 2. Những gì có thể làm trước khi có board

- kiểm tra chính xác dense checkpoint và dense baseline;
- profile sensitivity trên T4 bằng ImageNet validation;
- tạo scheme với boundary protection;
- kiểm tra scheme trên validation subset;
- benchmark đủ 50.000 ảnh trước fine-tune;
- fine-tune và benchmark đủ 50.000 ảnh sau fine-tune;
- so sánh cùng MAC budget với Uniform 3:4;
- chuẩn bị protocol để sau này thay T4 hardware profile bằng Jetson profile.

Chưa có board thì không thể kết luận latency, energy hoặc speedup trên Jetson.
T4 chỉ dùng để tạo/đánh giá model và kiểm tra pipeline.

## 3. Tạo sensitivity profile trên Colab

Lần đầu nên dùng không gian `M=4`, gồm `2:4`, `3:4`, `4:4`. Nó tránh cấu hình
quá mạnh như `1:16`, đồng thời chứa đúng Uniform 3:4 để so sánh cùng budget.

Smoke test 5.000 ảnh:

```bash
cd /content/DominoSearch

python search/profile_layer_sensitivity.py \
  --model resnet18_sparse \
  --checkpoint /content/DominoSearch-artifacts/checkpoints/resnet18-dense-converted.pth \
  --m 4 \
  --candidate-n 2 3 4 \
  --layout NHWC \
  --device cuda \
  --dataset-format parquet \
  --parquet-root /content/drive/MyDrive/DominoSearch-data/imagenet-1k \
  --parquet-pattern 'data/validation-*.parquet' \
  --dataset-num-samples 50000 \
  --accuracy-batch-size 64 \
  --workers 2 \
  --max-eval-samples 5000 \
  --seed 42 \
  --output /content/DominoSearch-artifacts/profiles/resnet18-m4-sensitivity-5k.json
```

Profiler luôn dùng cùng prefix validation, preprocessing, checkpoint và seed.
Mỗi lần chỉ đổi N:M của một layer. File JSON ghi baseline, checkpoint SHA-256,
dataset manifest, Git branch/commit và mọi kết quả quan sát. Không dùng profile
5.000 ảnh làm kết quả accuracy cuối; nó chỉ phục vụ chọn scheme.

Chi phí là `21 layer × 3 candidate × 5.000 ảnh`, tức tương đương khoảng 6,3 lần
đánh giá đủ 50.000 ảnh. Có thể chạy `--max-eval-samples 1000` để kiểm tra lệnh
trước, nhưng sensitivity 5.000 ảnh ổn định hơn.

## 4. Tạo matching hardware profile trên T4

Hardware profile và sensitivity profile phải chứa đúng cùng layer/candidate:

```bash
python search/profile_layer_hardware.py \
  --model resnet18_sparse \
  --checkpoint /content/DominoSearch-artifacts/checkpoints/resnet18-dense-converted.pth \
  --m 4 \
  --candidate-n 2 3 4 \
  --layout NHWC \
  --device cuda \
  --warmup 30 \
  --iterations 100 \
  --repeats 7 \
  --latency-weight 1 \
  --energy-weight 0 \
  --bandwidth-weight 0 \
  --memory-weight 0 \
  --output /content/DominoSearch-artifacts/profiles/resnet18-m4-t4-hardware.json
```

Do implementation hiện tạo mask rồi gọi dense PyTorch operator, profile này có
thể không cho thấy sparse nhanh hơn. Vì thế vòng trước-board nên đặt target theo
`macs`; hardware profile vẫn được lưu để kiểm tra mapping và sau này thay bằng
profile đo trên Jetson.

## 5. Chọn scheme sensitivity-aware

Target đầu tiên là giảm khoảng 23% MAC để so trực tiếp với Uniform 3:4:

```bash
python search/select_sensitivity_aware_scheme.py \
  --hardware-profile /content/DominoSearch-artifacts/profiles/resnet18-m4-t4-hardware.json \
  --sensitivity-profile /content/DominoSearch-artifacts/profiles/resnet18-m4-sensitivity-5k.json \
  --target-metric macs \
  --target-reduction 0.23 \
  --sensitivity-metric top1_drop_percent \
  --max-estimated-sensitivity 1.0 \
  --minimum-n 2 \
  --protect-first-conv \
  --protect-linear \
  --output /content/DominoSearch-artifacts/schemes/domino-m4-sensitive-mac23.txt
```

Selector dùng Pareto dynamic programming. Với mỗi layer, nó tính MAC/cost tiết
kiệm và penalty sensitivity. Nó chọn tổng penalty nhỏ nhất trong các scheme đạt
target, không cho layer đầu/linear cuối sparse khi bật hai cờ bảo vệ. Giá trị
`estimated_additive_sensitivity` chỉ là proxy: ảnh hưởng giữa các layer không
hoàn toàn cộng được, nên scheme vẫn phải được benchmark end-to-end.

Nếu không có scheme dưới budget `1,0` điểm Top-1, lệnh fail rõ ràng. Khi đó thử
target thấp hơn như `0.15` hoặc tăng budget có giải trình; không được âm thầm bỏ
accuracy constraint.

## 6. Cổng kiểm tra trước khi fine-tune

Chạy benchmark 5.000 ảnh với scheme mới. Chỉ tiếp tục nếu checkpoint load exact,
sparsity đúng và Top-1 hợp lý. Sau đó chạy đủ 50.000 ảnh trước fine-tune:

```bash
python benchmark/benchmark_model.py \
  --run-name domino-sensitive-mac23-before \
  --pruning-method domino-mixed-nm \
  --experiment-status candidate \
  --model resnet18_sparse \
  --checkpoint /content/DominoSearch-artifacts/checkpoints/resnet18-dense-converted.pth \
  --scheme-file /content/DominoSearch-artifacts/schemes/domino-m4-sensitive-mac23.txt \
  --device cuda \
  --dataset-format parquet \
  --parquet-root /content/drive/MyDrive/DominoSearch-data/imagenet-1k \
  --dataset-num-samples 50000 \
  --accuracy-batch-size 64 \
  --workers 2 \
  --batch-size 1 --warmup 30 --iterations 100 --seed 42 \
  --output /content/DominoSearch-artifacts/results/domino-sensitive-mac23-before.json
```

Tiêu chí đề xuất trước fine-tune: Top-1 ít nhất 67,0%, MAC giảm xấp xỉ 23%,
không thiếu layer trong scheme. Sau đó fine-tune fixed-mask 3 epoch, learning
rate `0.001`, cùng 50.000 training sample/epoch như thí nghiệm trước. Benchmark
lại đầy đủ và so với dense + Uniform 3:4 bằng `benchmark/compare_results.py`.

## 7. Mapping sang Jetson Nano khi có board

Không chuyển các con số latency T4 sang Jetson bằng một hệ số. Khi có board:

1. cố định power mode bằng `sudo nvpmodel -m <mode>`;
2. khóa clock cho phép bằng `sudo jetson_clocks`;
3. ghi phiên bản JetPack, CUDA, cuDNN, PyTorch/runtime và nhiệt độ;
4. chạy lại `profile_layer_hardware.py` cho đúng `2:4`, `3:4`, `4:4`;
5. đo nhiều repeat sau warm-up, đồng thời log power bằng `tegrastats` hoặc cảm
   biến nguồn ngoài;
6. tạo cost mới từ latency/energy/bandwidth/memory thực đo;
7. chạy lại selector với sensitivity profile không đổi nếu model/checkpoint và
   preprocessing không đổi;
8. benchmark end-to-end dense và pruned trên cùng power mode.

Mapping cuối cùng là:

```text
(đặc trưng layer, N:M, trạng thái Jetson) -> latency, energy, bandwidth, memory
                                  -> normalized hardware cost
hardware cost + sensitivity       -> N:M được chọn cho từng layer
```

## 8. Các cải tiến tiếp theo đáng thử

Theo thứ tự ưu tiên:

1. **direct layer sensitivity** như pipeline trên: đơn giản, dễ kiểm chứng;
2. **iterative re-profiling**: chọn một số layer, đo lại sensitivity của bước kế
   tiếp trên model đã sparse để giảm sai số do giả định cộng;
3. **Taylor sensitivity** (`|w × gradient|`) để rút ngắn profiling, nhưng phải
   đối chiếu với direct sensitivity trước;
4. **Pareto sweep** ở target MAC 10%, 15%, 20%, 23%, 30% thay vì chọn một điểm;
5. **candidate theo khả năng runtime**: chỉ cho selector dùng pattern/kernel mà
   Jetson runtime thật sự khai thác;
6. **cost lookup thay predictor** khi số layer ít; predictor T4 hiện có sai số
   cao nên không nên dùng để đưa ra kết luận.

NetAdapt cho thấy metric trực tiếp trên platform đáng tin hơn proxy FLOPs. HALP
kết hợp latency lookup và importance/saliency trong bài toán chọn pruning. Taylor
importance là lựa chọn proxy gradient có cơ sở, nhưng direct validation delta
phù hợp hơn cho vòng đầu vì số layer ResNet-18 nhỏ và mục tiêu cần dễ giải thích.

Nguồn nghiên cứu chính:

- [NetAdapt — ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Tien-Ju_Yang_NetAdapt_Platform-Aware_Neural_ECCV_2018_paper.html);
- [HALP — arXiv 2110.10811](https://arxiv.org/abs/2110.10811);
- [Importance Estimation for Neural Network Pruning — CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html);
- [SynFlow — NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/46a4378f835dc8040c8057beb6a2da52-Abstract.html), tham khảo cho rủi ro layer collapse.

## 9. Trạng thái sau thí nghiệm 2026-08-05

Pipeline đã được chạy trên ResNet-18/ImageNet và Tesla T4. Scheme MAC15 sau ba
epoch fine-tune đạt 69,648% Top-1 trên đủ 50.000 ảnh, giảm 15,135% effective MAC
và 13,876% effective parameter. Nó chỉ thấp hơn dense 0,106 điểm Top-1.

Kết quả chứng minh hướng sensitivity-aware giữ accuracy tốt hơn các Domino mixed
run trước tại điểm resource đã thử. Nó không chứng minh runtime speedup: median
latency PyTorch là 13,396 ms, còn dense là 3,778 ms, vì implementation vẫn mask
rồi gọi dense operator.

MAC15 giảm MAC ít hơn Uniform 3:4, nên chưa phải so sánh cùng budget. Scheme
MAC23 cùng khoảng 23,1% MAC không đạt cổng accuracy trước fine-tune. Bước tiếp
theo là cải thiện sensitivity profile/interaction hoặc thử target trung gian,
sau đó đo lại cost trên Jetson khi có board.

Chi tiết protocol, bảng trước/sau fine-tune, checkpoint và artifact nằm tại
[`DOMINO_SENSITIVITY_AWARE_RESULTS_T4.md`](DOMINO_SENSITIVITY_AWARE_RESULTS_T4.md).
