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
- `B_i`: byte input, output, effective weight và bias ước lượng;
- `W_i`: byte effective weight và bias;
- mẫu số là tổng giá trị dense của toàn bộ sparse layer;
- `α + β + γ + δ = 1` và mọi trọng số không âm.

Chuẩn hóa theo tổng dense giúp cộng các đại lượng khác đơn vị mà không để độ lớn
số học của byte lấn át millisecond. Với profile T4 hiện tại, mặc định dùng
`α=1`, các trọng số khác bằng 0. Energy không được suy đoán từ TDP.

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
  --latency-weight 1 \
  --energy-weight 0 --bandwidth-weight 0 --memory-weight 0 \
  --seed 42 \
  --output /path/to/t4-resnet18-nm-profile.json
```

Script từ chối overwrite profile, kiểm tra checkpoint exact, ghi branch/commit,
phiên bản PyTorch/CUDA, tên GPU, protocol đo và toàn bộ shape layer.

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

Có thể tái sử dụng cùng số đo nhưng đổi objective bằng cách truyền đủ bốn trọng
số. Ví dụ ưu tiên latency 20%, bandwidth 40% và memory weight 40%:

```bash
--hardware-latency-weight 0.20 \
--hardware-energy-weight 0.00 \
--hardware-bandwidth-weight 0.40 \
--hardware-memory-weight 0.40
```

Đây là một objective khác latency-only và phải được ghi thành run riêng. Giảm
composite cost không đồng nghĩa latency giảm.

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
