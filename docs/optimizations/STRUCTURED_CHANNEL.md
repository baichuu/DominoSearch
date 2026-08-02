# Điểm tối ưu: Structured channel pruning

Branch: `pruning-structured-channel`

Trạng thái: **đã có one-shot hidden-channel pruning và persistent fine-tune mask;
chưa có compact export hoặc benchmark để kết luận speedup**.

## Điểm tối ưu cụ thể

Residual connection yêu cầu input/output của block giữ cùng shape. Vì vậy bản đầu
không xóa output channel của block một cách tùy ý. Công cụ chỉ prune hidden channel
an toàn:

- ResNet BasicBlock: output `conv1` và input tương ứng của `conv2`.
- ResNet Bottleneck: `conv1→conv2` và `conv2→conv3`.

BatchNorm gamma/beta của channel bị prune cũng được mask. Channel được chọn bằng
L1 norm của filter hoặc độ lớn BatchNorm gamma. Số channel giữ lại được làm tròn
theo `--alignment` để phù hợp vector width/FPGA datapath hơn.

## Tạo checkpoint và mask

```bash
python pruning/structured_channel/prune_checkpoint.py \
  --model resnet18_sparse \
  --checkpoint /path/to/dense_checkpoint.pth \
  --ratio 0.25 \
  --score l1 \
  --alignment 8 \
  --output experiments/structured/resnet18-channel25.pth
```

Lệnh sinh bốn artifact:

- checkpoint đã materialize zero;
- `.masks.pth` để giữ channel zero trong fine-tune;
- `.dense-scheme.txt` để tắt N:M, tránh trộn hai phương pháp;
- manifest JSON chứa site/channel và Git metadata.

Không được đặt output trùng checkpoint đầu vào.

## Fine-tune giữ mask

```bash
python train/classification_sparsity_level/train_imagenet.py \
  --config train/classification_sparsity_level/train_imagenet/configs/config_resnet18.yaml \
  --initial-checkpoint experiments/structured/resnet18-channel25.pth \
  --weight-mask-file experiments/structured/resnet18-channel25.pth.masks.pth \
  --schemes_file experiments/structured/resnet18-channel25.pth.dense-scheme.txt \
  --model_dir runs/structured-resnet18-channel25 \
  --epochs 120
```

Gradient bị mask và weight được ép về 0 sau mỗi optimizer step. Như vậy weight đã
prune không mọc lại trong fine-tune.

## Benchmark

```bash
python benchmark/benchmark_model.py \
  --run-name structured-channel25-before \
  --pruning-method structured-channel \
  --density-source nonzero \
  --experiment-status candidate \
  --checkpoint experiments/structured/resnet18-channel25.pth \
  --n 1 --m 1 \
  --dataset-format imagefolder \
  --data-root /path/to/imagenet/val \
  --output results/structured-channel25-before.json
```

## Ma trận tối thiểu

| ID | Channel target | Score | Alignment |
| --- | ---: | --- | ---: |
| S0 | 0% | — | 8 |
| S1 | 10% | L1 | 8 |
| S2 | 25% | L1 | 8 |
| S3 | 40% | L1 | 8 |
| S4 | 25% | BN gamma | 8 |
| S5 | 25% | L1 | 16 |

## Giới hạn và bước tiếp theo

Checkpoint hiện vẫn dùng tensor shape dense và chỉ materialize channel zero. Nó
đủ để đo accuracy, fine-tune và ước lượng effective non-zero MACs, nhưng dense
PyTorch convolution chưa bỏ phép tính của channel zero.

Bước tiếp theo sau khi chọn được tỷ lệ tốt là tạo **compact model export**: thực
sự cắt output dimension của conv1/conv2 trung gian, copy channel được giữ và xuất
sang runtime/FPGA. Chỉ benchmark compact model hoặc target board mới chứng minh
latency speedup. Không dùng latency của masked checkpoint để tuyên bố structured
pruning đã chạy nhanh hơn.
