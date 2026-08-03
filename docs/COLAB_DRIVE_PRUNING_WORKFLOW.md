# Workflow pruning ImageNet từ Google Drive trên Colab

Tài liệu này là giao thức dùng chung cho `master` và bốn nhánh pruning. Mục tiêu
là giữ dataset, dense checkpoint, seed, fine-tune budget và benchmark settings
giống nhau để báo cáo có thể bảo vệ được.

## 1. Dữ liệu và artifact

Mount Drive trong Colab:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Các đường dẫn dùng trong ví dụ:

```text
Dataset:   /content/drive/MyDrive/DominoSearch-data/imagenet-1k
Artifacts: /content/drive/MyDrive/DominoSearch-artifacts
```

Dataset phải có đúng 294 shard train và 14 shard validation dưới thư mục `data/`.
Nếu train đã đủ như ảnh nhưng validation chưa có, chạy một cell Colab:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ILSVRC/imagenet-1k",
    repo_type="dataset",
    allow_patterns=["data/validation-*.parquet"],
    local_dir="/content/drive/MyDrive/DominoSearch-data/imagenet-1k",
    token=True,
    max_workers=2,
)
```

Kiểm tra nhanh:

```bash
python - <<'PY'
from imagenet_data import parquet_manifest

root = "/content/drive/MyDrive/DominoSearch-data/imagenet-1k"
for split, pattern, expected in (
    ("train", "data/train-*.parquet", 294),
    ("validation", "data/validation-*.parquet", 14),
):
    manifest = parquet_manifest(root, pattern)
    print(split, manifest)
    assert manifest["shard_count"] == expected
PY
```

Không đặt dataset, checkpoint, mask, scheme sinh ra, log hoặc result JSON trong
Git. Mỗi nhánh chỉ chứa code của phương pháp; artifact được lưu trên Drive.

## 2. Dense baseline bắt buộc

Chạy trên `master` trước mọi phương pháp:

```bash
git switch master
git pull --ff-only

python benchmark/benchmark_model.py \
  --run-name dense-resnet18-drive \
  --pruning-method dense \
  --experiment-status candidate \
  --model resnet18_sparse \
  --checkpoint /content/drive/MyDrive/DominoSearch-artifacts/checkpoints/resnet18-dense.pth \
  --n 1 --m 1 \
  --dataset-format parquet \
  --parquet-root /content/drive/MyDrive/DominoSearch-data/imagenet-1k \
  --accuracy-batch-size 64 \
  --workers 2 \
  --device cuda \
  --batch-size 1 \
  --warmup 30 \
  --iterations 100 \
  --seed 42 \
  --output /content/drive/MyDrive/DominoSearch-artifacts/results/dense-resnet18-drive.json
```

Kết quả tham chiếu đã đo trước đây là Top-1 `69,760%`, Top-5 khoảng `89,08%`,
11,689512 triệu parameter và 1,814073344 tỷ MAC/sample. Run mới chỉ hợp lệ nếu
checkpoint load không thiếu key, đủ 50.000 validation sample và accuracy gần mốc
này.

## 3. Smoke test trước khi chạy dài

Trên mỗi nhánh, chạy benchmark với `--max-eval-samples 1000` và
`--experiment-status debug`. Với fine-tune/search, có thể tạm đặt sample count nhỏ
chỉ để kiểm tra pipeline, nhưng không đưa số đó vào bảng kết quả cuối.

Đọc Parquet trực tiếp từ Drive dùng streaming:

- train được shuffle theo shard và buffer; seed đổi có kiểm soát theo epoch;
- validation không shuffle;
- worker và distributed rank nhận stream tách biệt;
- mặc định `train-num-samples=1.281.167`, `val-num-samples=50.000`.

## 4. Fine-tune dùng chung

Mọi nhánh gọi cùng entry point và chỉ khác scheme/checkpoint/mask của phương pháp:

```bash
python train/classification_sparsity_level/train_imagenet.py \
  --config train/classification_sparsity_level/train_imagenet/configs/config_resnet18.yaml \
  --dataset-format parquet \
  --parquet-root /content/drive/MyDrive/DominoSearch-data/imagenet-1k \
  --train-parquet-pattern 'data/train-*.parquet' \
  --val-parquet-pattern 'data/validation-*.parquet' \
  --train-num-samples 1281167 \
  --val-num-samples 50000 \
  --shuffle-buffer 10000 \
  --seed 42 \
  --data-workers 2 \
  --initial-checkpoint CHECKPOINT_TRUOC_FINE_TUNE \
  --schemes_file SCHEME_CUA_PHUONG_PHAP \
  --model_dir /content/drive/MyDrive/DominoSearch-artifacts/runs/TEN_RUN \
  --epochs SO_EPOCH_GIONG_NHAU
```

Structured và unstructured phải truyền thêm `--weight-mask-file`. Uniform và
Domino mixed dùng N:M scheme và không thêm mask của phương pháp khác.
`--data-workers` override riêng giá trị YAML mà không sửa config gốc. Trên mounted
Drive nên bắt đầu với 2–3 worker rồi chỉ tăng sau khi đo `Data` time trong log.

## 5. Trách nhiệm từng nhánh

| Nhánh | Artifact riêng | Benchmark density |
| --- | --- | --- |
| `pruning-uniform-nm` | Uniform scheme từ `generate_scheme.py` | `nm` |
| `pruning-domino-mixed-nm` | Layer-wise scheme từ search | `nm` |
| `pruning-structured-channel` | Zero checkpoint, mask, dense scheme | `nonzero` |
| `pruning-unstructured-magnitude` | Zero checkpoint, mask, dense scheme | `nonzero` |

Mỗi cấu hình phải có hai JSON: `before-finetune` và `after-finetune`. Cả hai dùng
đúng lệnh benchmark của dense, chỉ đổi pruning method, checkpoint, scheme và
`density-source` khi cần.

## 6. Search Domino mixed N:M từ Drive

Trên `pruning-domino-mixed-nm`:

```bash
python search/find_mix_from_dense_imagenet.py \
  --config search/script_resnet_ImageNet/configs/config_resnet18_img_mix_from_dense.yaml \
  --dataset-format parquet \
  --parquet-root /content/drive/MyDrive/DominoSearch-data/imagenet-1k \
  --train-parquet-pattern 'data/train-*.parquet' \
  --val-parquet-pattern 'data/validation-*.parquet' \
  --train-num-samples 1281167 \
  --val-num-samples 50000 \
  --shuffle-buffer 10000 \
  --seed 42 \
  --data-workers 2 \
  --target-metric params \
  --target_sparsity 0.50 \
  --model_dir /content/drive/MyDrive/DominoSearch-artifacts/runs/domino-params50 \
  --scheme-output /content/drive/MyDrive/DominoSearch-artifacts/schemes/domino-params50.txt
```

## 7. Sinh báo cáo Markdown và CSV

Sau khi có dense và các JSON pruning:

```bash
python benchmark/compare_results.py \
  /content/drive/MyDrive/DominoSearch-artifacts/results/dense-resnet18-drive.json \
  /content/drive/MyDrive/DominoSearch-artifacts/results/*-before.json \
  /content/drive/MyDrive/DominoSearch-artifacts/results/*-after.json \
  --csv /content/drive/MyDrive/DominoSearch-artifacts/reports/comparison.csv \
  --markdown /content/drive/MyDrive/DominoSearch-artifacts/reports/pruning-report.md \
  --title "So sánh pruning ResNet-18 trên ImageNet"
```

Script tự cảnh báo nếu sample count, dataset format, input shape, device, batch
size, warm-up hoặc iterations khác dense baseline. Báo cáo cuối cần tách ba loại
bằng chứng:

1. theoretical parameter/MAC reduction;
2. host PyTorch runtime trên Colab;
3. target-hardware runtime trên CPU/board/FPGA, chỉ khi đã đo thật.

## 8. Checklist trước khi đưa vào báo cáo

- Dense và pruned cùng checkpoint gốc, preprocessing, seed và validation 50k.
- Có cả trước và sau fine-tune.
- JSON ghi đúng branch/commit và worktree không dirty.
- Không gọi một run `final` nếu checkpoint load hoặc dataset manifest sai.
- Không dùng giảm MAC để khẳng định FPGA nhanh hơn.
- Chọn Pareto frontier theo accuracy, effective complexity, host runtime và memory;
  không chọn chỉ theo sparsity.
