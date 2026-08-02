# Điểm tối ưu: Unstructured magnitude pruning

Branch: `pruning-unstructured-magnitude`

Trạng thái: **đã có global/local one-shot pruning và persistent fine-tune mask;
chưa có benchmark để kết luận hiệu quả**.

## Điểm tối ưu cụ thể

Hướng này loại từng weight có trị tuyệt đối nhỏ nhất. Nó là đối chứng về khả năng
giữ accuracy theo số non-zero, không phải mặc định là hướng tốt nhất cho FPGA.

Hai cách phân bổ:

- `global`: xếp hạng chung toàn bộ weight đủ điều kiện; layer tự nhận sparsity khác
  nhau theo phân bố magnitude.
- `local`: mỗi layer bị prune đúng gần tỷ lệ yêu cầu.

Layer convolution đầu và linear cuối được bảo vệ mặc định. Chỉ dùng `--prune-first`
hoặc `--prune-last` khi muốn thực hiện ablation riêng.

## Tạo checkpoint và persistent mask

```bash
python pruning/unstructured_magnitude/prune_checkpoint.py \
  --model resnet18_sparse \
  --checkpoint /path/to/dense_checkpoint.pth \
  --scope global \
  --sparsity 0.70 \
  --output experiments/unstructured/resnet18-global70.pth
```

Tool chọn chính xác số index nhỏ nhất bằng `topk`, materialize weight bằng 0 và
sinh:

- checkpoint pruned;
- `.masks.pth` để giữ zero trong fine-tune;
- `.dense-scheme.txt` để N:M không xen vào thí nghiệm;
- manifest JSON với sparsity từng layer, branch và commit.

## Fine-tune

```bash
python train/classification_sparsity_level/train_imagenet.py \
  --config train/classification_sparsity_level/train_imagenet/configs/config_resnet18.yaml \
  --initial-checkpoint experiments/unstructured/resnet18-global70.pth \
  --weight-mask-file experiments/unstructured/resnet18-global70.pth.masks.pth \
  --schemes_file experiments/unstructured/resnet18-global70.pth.dense-scheme.txt \
  --model_dir runs/unstructured-resnet18-global70 \
  --epochs 120
```

## Benchmark

```bash
python benchmark/benchmark_model.py \
  --run-name unstructured-global70-after \
  --pruning-method unstructured-magnitude \
  --density-source nonzero \
  --experiment-status candidate \
  --checkpoint /path/to/finetuned_checkpoint.pth \
  --n 1 --m 1 \
  --dataset-format imagefolder \
  --data-root /path/to/imagenet/val \
  --output results/unstructured-global70-after.json
```

## Ma trận tối thiểu

| ID | Scope | Eligible sparsity | Boundary |
| --- | --- | ---: | --- |
| M0 | — | 0% | Dense baseline |
| M1 | Global | 50% | Protected |
| M2 | Global | 70% | Protected |
| M3 | Global | 80% | Protected |
| M4 | Global | 90% | Protected |
| M5 | Local | 70% | Protected |
| M6 | Global | 70% | Prune first/last |

## Điều kiện so sánh

- So accuracy với N:M tại cùng số non-zero/effective parameters.
- Báo cáo sparsity thực tế toàn model vì layer đầu/cuối được bảo vệ làm tỷ lệ tổng
  thấp hơn tỷ lệ trên eligible weights.
- Global và local phải dùng cùng checkpoint, seed và fine-tune budget.
- Đo cả trước và sau fine-tune; không chỉ chọn checkpoint tốt nhất mà bỏ qua mức
  giảm accuracy ban đầu.

## Giới hạn phần cứng

Pattern unstructured là irregular và cần lưu index/mask. Checkpoint dense chứa zero
không tự nhỏ hơn và dense PyTorch operator không tự bỏ phép nhân zero. Effective
MAC/parameter từ `--density-source nonzero` chỉ là complexity lý thuyết.

Chỉ kết luận speedup khi có sparse storage/kernel hoặc FPGA datapath hỗ trợ pattern
irregular và latency end-to-end thực sự giảm. Nếu accuracy tốt nhưng latency không
giảm, hướng này vẫn có giá trị làm upper-bound đối chứng chứ chưa phải phương án
deployment tốt nhất.
