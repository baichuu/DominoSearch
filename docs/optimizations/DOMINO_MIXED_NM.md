# Điểm tối ưu: DominoSearch mixed N:M

Branch: `pruning-domino-mixed-nm`

Trạng thái: **đã hỗ trợ search theo parameter hoặc FLOPs budget; sẵn sàng chạy
search/fine-tune, chưa có benchmark mới để kết luận đã tối ưu**.

## Điểm tối ưu cụ thể

Điểm mạnh cần kiểm chứng của DominoSearch là phân bổ N:M khác nhau theo độ nhạy
từng layer. Nhánh này tập trung vào hai objective tách biệt:

1. **Parameter reduction**: ưu tiên giảm model weight budget.
2. **FLOPs reduction**: ưu tiên layer đóng góp nhiều phép tính.

Code gốc chỉ bật điều kiện dừng theo model sparsity; đoạn dừng theo FLOPs bị
comment. Nhánh này đưa cả hai thành CLI và giữ mặc định cũ:

- `--target-metric params`: 0.5 ERK + 0.5 complexity.
- `--target-metric flops`: 0.2 ERK + 0.8 FLOPs, đúng thiết lập mô tả trong bài báo.
- `--vote-ratio 0.75`: supermajority mặc định.

Có thể override `--erk-weight` và `--cost-weight`, nhưng hai giá trị phải không âm
và tổng bằng 1. Mỗi kết quả search sinh thêm manifest JSON chứa branch, commit,
objective, target, giá trị đạt được và scheme đầy đủ.

## Chạy search theo parameter budget

```bash
python search/find_mix_from_dense_imagenet.py \
  --config search/script_resnet_ImageNet/configs/config_resnet18_img_mix_from_dense.yaml \
  --target-metric params \
  --target_sparsity 0.50 \
  --scheme-output experiments/domino/resnet18-params-50.txt
```

## Chạy search theo FLOPs budget

```bash
python search/find_mix_from_dense_imagenet.py \
  --config search/script_resnet_ImageNet/configs/config_resnet18_img_mix_from_dense.yaml \
  --target-metric flops \
  --target_sparsity 0.50 \
  --scheme-output experiments/domino/resnet18-flops-50.txt
```

`--target_sparsity 0.50` ở lệnh thứ hai nghĩa là giảm 50% FLOPs, không phải bắt
buộc giảm 50% parameter. Tên option cũ được giữ để không phá script hiện có.

## Điểm cần sweep

| ID | Metric | Target | ERK/cost | Mục đích |
| --- | --- | ---: | --- | --- |
| D0 | Params | 50% | 0.5/0.5 | Pruning vừa |
| D1 | Params | 70% | 0.5/0.5 | Compression cao |
| D2 | Params | 80% | 0.5/0.5 | So với paper |
| D3 | FLOPs | 50% | 0.2/0.8 | Compute budget |
| D4 | FLOPs | 70% | 0.2/0.8 | Compute thấp |
| D5 | FLOPs | 50% | 0.5/0.5 | Ablation penalty |

Mỗi scheme phải được đo trước fine-tune và sau SR-STE fine-tune.

## Benchmark

```bash
python benchmark/benchmark_model.py \
  --run-name domino-flops50-after-finetune \
  --pruning-method domino-mixed-nm \
  --experiment-status candidate \
  --checkpoint /path/to/finetuned_checkpoint.pth \
  --scheme-file experiments/domino/resnet18-flops-50.txt \
  --dataset-format imagefolder \
  --data-root /path/to/imagenet/val \
  --output results/domino-flops50-after.json
```

## Điều kiện chứng minh hướng này tốt hơn

- So với Uniform N:M tại **cùng effective parameter hoặc MAC budget**, không chỉ
  cùng sparsity label.
- Mixed N:M phải cải thiện Top-1 đủ lớn để bù chi phí search và phần cứng phức tạp.
- Parameter-target và FLOPs-target là hai kết quả khác nhau, không trộn trong một
  bảng nếu budget không tương đương.
- FLOPs giảm mới chỉ là complexity lý thuyết; speedup phải đo trên target runtime.

## Giới hạn hiện tại

- Search cần pretrained checkpoint và ImageNet nên chưa chạy được trong môi trường
  không có PyTorch/dataset.
- Objective hiện hỗ trợ parameter và FLOPs. Latency/energy lookup table là bước
  hardware-in-the-loop tiếp theo, không được giả lập thành số đo thật.
- Search vẫn dùng fixed M và candidate N theo implementation DominoSearch gốc.
