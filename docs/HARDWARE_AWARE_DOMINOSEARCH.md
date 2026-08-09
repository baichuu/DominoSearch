# Hardware-aware DominoSearch theo từng layer

## 1. Mục tiêu

Hướng này chỉ thuộc nhánh `pruning-domino-mixed-nm`. Nó thay thành phần FLOPs
trong layer-wise penalty bằng cost đo trên phần cứng mục tiêu, nhưng giữ FLOPs
làm mặc định để bảo toàn hành vi DominoSearch gốc.

Với layer `i` và cấu hình `N:M`, cost được định nghĩa:

```text
C_i(N:M) = α × L_i/L_dense
         + β × E_i/E_dense
         + γ × B_i/B_dense
         + δ × W_i/W_dense
```

Trong đó:

- `L_i`: latency layer đo trên thiết bị;
- `E_i`: energy của layer, chỉ được dùng khi có power sensor đồng bộ;
- `B_i`: DRAM traffic đo bằng hardware counter phù hợp;
- `W_i`: peak system memory đo trên thiết bị;
- mẫu số là tổng giá trị dense của toàn bộ sparse layer;
- `α + β + γ + δ = 1` và mọi trọng số không âm.

Chuẩn hóa theo tổng dense giúp cộng các đại lượng khác đơn vị mà không để độ lớn
số học của byte lấn át millisecond. Profiler PyTorch hiện chỉ đo trực tiếp
latency nên bắt buộc dùng `α=1`, các trọng số khác bằng 0. Energy không được suy
đoán từ TDP; bandwidth và memory không được suy ra từ tensor shape rồi gọi là số
đo phần cứng. Các feature lý thuyết vẫn mô tả layer nhưng không đi vào measured
cost.

## 2. Hai nguồn cost

### Lookup

`lookup` lấy đúng số đã đo cho từng `layer × N:M`. Đây là nguồn nên dùng khi mọi
cấu hình ứng viên đều đã profile.

### Predictor

`predictor` là ridge regression trên `log(1 + metric)`. Đầu vào gồm loại layer,
channel vào/ra, số phần tử input/output, kernel, group, parameter, MAC và mật độ
`N/M`. JSON lưu hệ số cùng leave-one-layer-out MAE, MAPE và R².

Predictor có thể ước lượng `N:M` chưa đo, nhưng không được xem là đáng tin nếu sai
số validation lớn. Khi lookup đầy đủ, lookup là bằng chứng mạnh hơn predictor.

## 3. Tạo profile trên đúng phần cứng

Ví dụ Tesla T4, batch triển khai bằng 1:

```bash
python search/profile_layer_hardware.py \
  --model resnet18_sparse \
  --checkpoint /path/to/resnet18-dense.pth \
  --m 16 --candidate-n 1 2 4 8 16 \
  --layout NHWC --input-size 224 --batch-size 1 \
  --device cuda --warmup 30 --iterations 100 --repeats 7 \
  --bootstrap-resamples 2000 \
  --latency-cost-statistic ci95-high \
  --latency-weight 1 \
  --energy-weight 0 --bandwidth-weight 0 --memory-weight 0 \
  --seed 42 \
  --output /path/to/t4-resnet18-nm-profile.json
```

Script từ chối overwrite profile, kiểm tra checkpoint exact, ghi branch/commit,
phiên bản PyTorch/CUDA, tên GPU, protocol đo và toàn bộ shape layer. Mỗi candidate
lưu raw per-repeat block mean của CUDA Events và synchronized wall clock, median,
P95, standard deviation và bootstrap CI95%. `ci95-high` là lựa chọn bảo thủ khi
dùng latency làm cost; `median` dùng cho run đối chứng.

`energy_mj`, `bandwidth_bytes` và `memory_bytes` của profiler này là `null`.
Profiler fail nếu weight tương ứng khác 0. Chỉ collector Jetson tích hợp power
sensor, hardware counter và system-memory telemetry mới được điền các metric đó.

> Latency này phản ánh implementation PyTorch hiện có: tạo mask rồi gọi dense
> operator. Nó không đại diện cho FPGA hoặc sparse accelerator khác. Muốn tối ưu
> cho board nào phải chạy lại profiler bằng operator/runtime của board đó.

## 4. Chạy DominoSearch với hardware cost

Lookup chính xác:

```bash
python search/find_mix_from_dense_imagenet.py \
  --config search/script_resnet_ImageNet/configs/config_resnet18_img_mix_from_dense.yaml \
  --initial-checkpoint /path/to/resnet18-dense.pth \
  --cost-source hardware \
  --hardware-profile /path/to/t4-resnet18-nm-profile.json \
  --hardware-cost-mode lookup \
  --target-metric hardware --target_sparsity 0.20 \
  --dataset-format parquet --parquet-root /path/to/imagenet-1k \
  --train-num-samples 50000 --val-num-samples 1000 \
  --data-workers 2 --seed 42 \
  --model_dir /path/to/runs/domino-hardware-t4 \
  --scheme-output /path/to/schemes/domino-hardware-t4.txt
```

Để đánh giá predictor, chỉ đổi:

```bash
--hardware-cost-mode predictor
```

Search manifest lưu SHA-256 của profile, thiết bị, cost weights, predictor
validation, scheme và reduction đạt được. Profile phải khớp chính xác toàn bộ
sparse layer và đủ các ứng viên `1/2/4/8/16:16` khi dùng lookup.

Chỉ có thể tái sử dụng cùng số đo và đổi objective khi profile chứa số đo thật
cho mọi metric có weight dương. Ví dụ dưới đây chỉ hợp lệ với profile Jetson đã
tích hợp bandwidth và memory collector, không hợp lệ với profiler PyTorch hiện
tại:

```bash
--hardware-latency-weight 0.20 \
--hardware-energy-weight 0.00 \
--hardware-bandwidth-weight 0.40 \
--hardware-memory-weight 0.40
```

Đây là một objective khác latency-only và phải được ghi thành run riêng. Giảm
composite cost không đồng nghĩa latency giảm.

### Chọn scheme trực tiếp từ profile

Khi gradient search không thể hoàn tất ổn định, có thể tạo một baseline xác định
chỉ từ lookup table:

```bash
python search/select_scheme_from_hardware_profile.py \
  --hardware-profile /path/to/t4-resnet18-nm-profile.json \
  --output /path/to/schemes/hardware-profile-target3.txt \
  --target-reduction 0.03 --loss-metric parameters \
  --latency-weight 1.00 --energy-weight 0.00 \
  --bandwidth-weight 0.00 --memory-weight 0.00
```

Selector giải bài toán multiple-choice Pareto: đạt cost reduction yêu cầu với
ít parameter (hoặc MAC) bị loại nhất. Đây là **hardware-profile selection**,
không phải kết quả gradient search của DominoSearch; parameter/MAC chỉ là proxy
complexity, không phải predictor accuracy. File manifest cạnh scheme ghi rõ
method, profile SHA-256, weights, target và reduction đạt được.

## 5. Giao thức kết quả đầy đủ

Một kết quả hợp lệ cần:

1. dense baseline trên cùng T4/runtime;
2. profile layer-wise từ dense checkpoint;
3. scheme hardware-aware và manifest;
4. benchmark scheme trước fine-tune trên đủ 50.000 validation ảnh;
5. fine-tune cùng budget với Domino cũ và Uniform;
6. benchmark checkpoint sau fine-tune trên đủ 50.000 ảnh;
7. so sánh Top-1/Top-5, parameter, MAC, latency, throughput, memory và predicted
   hardware cost reduction;
8. ghi rõ lookup/predictor và sai số predictor.

Nếu không có cấu hình sparse nào giảm measured latency cost trên stack hiện tại,
kết luận đúng là “dense nằm trên Pareto frontier của runtime này”, không được đổi
sang MAC rồi vẫn gọi đó là hardware speedup.

## 6. Kết quả T4 ngày 2026-08-05

Profile thực tế gồm 21 layer × 5 ứng viên `1/2/4/8/16:16` (105 điểm), batch 1,
warm-up 30, 100 iteration và bảy repeat. Không có cấu hình sparse nào giảm
measured layer latency so với dense cùng layer. Energy không được đo. Predictor
latency leave-one-layer-out có MAE 0,255 ms, MAPE 69,85% và R² 0,577, nên kết quả
cuối dùng lookup thay vì predictor.

Selector với trọng số latency/bandwidth/memory `0,2/0,4/0,4`, target composite
3% và parameter loss đã chọn `1:16` cho bốn convolution sâu và linear cuối. Nó
đạt composite reduction dự đoán 3,102%, parameter reduction 37,22% và MAC
reduction 23,92%. Benchmark đủ 50.000 ảnh cho kết quả:

| Mốc | Top-1 % | Top-5 % | Median ms | P95 ms | sample/s |
| --- | ------: | ------: | --------: | -----: | -------: |
| Trước fine-tune | 0,250 | 0,900 | 6,709 | 7,272 | 149,05 |
| Sau 3 epoch × 50.000 sample | 51,962 | 77,920 | 6,923 | 7,954 | 144,45 |

Dense cùng protocol đạt Top-1 69,754%, median 3,725 ms và 268,44 sample/s. Vì
vậy scheme này không được gọi là tối ưu hay nhanh hơn: nó chỉ chứng minh pipeline
đo cost/chọn scheme hoạt động, đồng thời cho thấy objective phần cứng phải có
ràng buộc sensitivity/accuracy và cần sparse kernel thực sự.
