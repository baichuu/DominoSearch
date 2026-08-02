# Điểm tối ưu: Uniform N:M pruning

Branch: `pruning-uniform-nm`

Trạng thái: **đã có công cụ sinh scheme, sẵn sàng đánh giá; chưa có kết quả để
kết luận đã tối ưu**.

## Điểm tối ưu cụ thể

Hướng này giữ cấu trúc N:M giống nhau trên các layer để tạo baseline phần cứng đơn
giản. Không có search theo layer. Các biến cần khảo sát là:

1. Tỷ lệ `3:4`, `2:4`, `1:4`, `8:16`, `4:16`.
2. Prune toàn bộ sparse layer hoặc giữ layer đầu/cuối dense.
3. Giữ các convolution 1×1 dense hoặc prune giống các layer khác.
4. Accuracy trước fine-tune và sau SR-STE fine-tune.

Điểm khởi đầu đề xuất là `2:4`, sau đó thử `2:4` với layer đầu và layer cuối dense.
Đây chỉ là giả thuyết; benchmark mới quyết định cấu hình tốt hơn.

## Công cụ mới

`pruning/uniform_nm/generate_scheme.py` tạo dictionary đầy đủ cho đúng model,
kiểm tra `0 < N <= M`, kiểm tra số weight chia hết cho M và từ chối tên layer
không tồn tại. Công cụ cũng sinh manifest JSON chứa branch, commit, layer được bảo
vệ và sparsity lý thuyết.

Ví dụ:

```bash
python pruning/uniform_nm/generate_scheme.py \
  --model resnet18_sparse \
  --n 2 --m 4 \
  --keep-first-dense \
  --keep-last-dense \
  --output experiments/uniform-nm/resnet18-2of4-boundary-dense.txt
```

Không commit file scheme/manifest sinh ra nếu đó chỉ là artifact của một lần chạy.

## Benchmark trước fine-tune

```bash
python benchmark/benchmark_model.py \
  --run-name uniform-2of4-before-finetune \
  --pruning-method uniform-nm \
  --experiment-status candidate \
  --model resnet18_sparse \
  --checkpoint /path/to/dense_checkpoint.pth \
  --scheme-file experiments/uniform-nm/resnet18-2of4-boundary-dense.txt \
  --dataset-format imagefolder \
  --data-root /path/to/imagenet/val \
  --output results/uniform-2of4-before.json
```

Fine-tune dùng `train/classification_sparsity_level/train_imagenet.py` với file
scheme vừa sinh. Sau đó benchmark lại bằng đúng lệnh trên nhưng đổi checkpoint và
run name thành `after-finetune`.

## Điều kiện chứng minh hướng này có ích

- So với dense: báo cáo chênh lệch Top-1, effective MACs và parameters.
- So với Domino mixed N:M: phải so tại budget MAC hoặc parameter tương đương.
- Chỉ báo speedup nếu runtime/board thực sự hỗ trợ pattern N:M tương ứng.
- Nếu hai cấu hình accuracy gần nhau, ưu tiên cấu hình có pattern đơn giản hơn.

## Ma trận chạy tối thiểu

| ID | N:M | Layer đầu/cuối | 1×1 | Mục đích |
| --- | --- | --- | --- | --- |
| U0 | 4:4 | Dense | Dense | Baseline |
| U1 | 3:4 | Pruned | Pruned | Pruning nhẹ |
| U2 | 2:4 | Pruned | Pruned | Uniform chuẩn |
| U3 | 2:4 | Dense | Pruned | Kiểm tra boundary sensitivity |
| U4 | 2:4 | Dense | Dense | Pattern bảo thủ |
| U5 | 1:4 | Dense | Dense | Pruning mạnh |

Giữ nguyên checkpoint, dataset, seed, epoch fine-tune và benchmark settings giữa
các hàng. Các run thiếu một điều kiện trên chỉ được đánh dấu `debug`.
