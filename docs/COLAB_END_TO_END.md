# Chạy DominoSearch trên Google Colab từ đầu đến cuối

Tài liệu này hướng dẫn chạy ResNet-18 trên ImageNet theo đúng giao thức benchmark
của dự án: chuẩn bị Colab, lưu dữ liệu trên Google Drive, tạo dense checkpoint,
đo baseline, pruning, fine-tune, đo lại đủ 50.000 ảnh và sinh báo cáo.

Các cell được thiết kế để chạy lần lượt trong một notebook Colab mới. Hướng nên
chạy đầu tiên là **unstructured global magnitude 30%** vì đây là điểm
accuracy–complexity tốt nhất trong các kết quả hiện có. Ba hướng còn lại nằm ở
phần 10.

## 1. Chuẩn bị

Cần có:

- Google Colab runtime dùng GPU;
- Google Drive còn ít nhất khoảng 170 GB nếu tải cả train và validation;
- tài khoản Hugging Face đã được cấp quyền truy cập `ILSVRC/imagenet-1k`;
- khoảng vài giờ cho download và một thí nghiệm fine-tune giới hạn.

Trong Colab, chọn **Runtime → Change runtime type → T4 GPU**. Sau đó chạy:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

assert torch.cuda.is_available(), "Hãy bật GPU runtime trước khi tiếp tục"
```

```bash
!nvidia-smi
```

Không đổi GPU giữa dense và pruned benchmark. Nếu Colab cấp lại runtime/GPU khác,
hãy chạy lại toàn bộ latency benchmark trên cùng runtime.

## 2. Clone repository và cài dependency

```bash
!git clone https://github.com/baichuu/DominoSearch.git /content/DominoSearch
%cd /content/DominoSearch
!git switch master
!git pull --ff-only
!pip install -q -r requirements.txt
```

Kiểm tra môi trường:

```python
import torch
import torchvision

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
```

## 3. Mount Google Drive và khai báo đường dẫn

```python
from google.colab import drive

drive.mount("/content/drive")
```

```python
import os
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive")
DATA_ROOT = DRIVE_ROOT / "DominoSearch-data" / "imagenet-1k"
ARTIFACT_ROOT = DRIVE_ROOT / "DominoSearch-artifacts"

CHECKPOINT_ROOT = ARTIFACT_ROOT / "checkpoints"
SCHEME_ROOT = ARTIFACT_ROOT / "schemes"
RUN_ROOT = ARTIFACT_ROOT / "runs"
RESULT_ROOT = ARTIFACT_ROOT / "results"
REPORT_ROOT = ARTIFACT_ROOT / "reports"

for directory in (
    DATA_ROOT,
    CHECKPOINT_ROOT,
    SCHEME_ROOT,
    RUN_ROOT,
    RESULT_ROOT,
    REPORT_ROOT,
):
    directory.mkdir(parents=True, exist_ok=True)

# Các lệnh ! bên dưới nhận biến này từ environment.
os.environ["DS_DATA_ROOT"] = str(DATA_ROOT)
os.environ["DS_ARTIFACT_ROOT"] = str(ARTIFACT_ROOT)
os.environ["DS_CHECKPOINT_ROOT"] = str(CHECKPOINT_ROOT)
os.environ["DS_SCHEME_ROOT"] = str(SCHEME_ROOT)
os.environ["DS_RUN_ROOT"] = str(RUN_ROOT)
os.environ["DS_RESULT_ROOT"] = str(RESULT_ROOT)
os.environ["DS_REPORT_ROOT"] = str(REPORT_ROOT)

# Tăng độ ổn định khi tải shard lớn trực tiếp vào mounted Drive.
os.environ["HF_HOME"] = str(DRIVE_ROOT / "DominoSearch-data" / ".huggingface-cache")
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] = "1"

print("Dataset:", DATA_ROOT)
print("Artifacts:", ARTIFACT_ROOT)
```

Không lưu dataset/checkpoint/result vào Git repository. Drive giữ chúng qua lần
Colab reset; source code trong `/content/DominoSearch` có thể clone lại.

## 4. Đăng nhập Hugging Face và tải ImageNet

Trước tiên mở trang dataset `ILSVRC/imagenet-1k` trên Hugging Face, chấp nhận điều
kiện truy cập và tạo access token dạng **Read**. Không ghi token trực tiếp vào
notebook.

```python
from huggingface_hub import notebook_login

notebook_login()
```

### 4.1 Chỉ tải validation để kiểm tra baseline

Nếu muốn xác nhận pipeline trước, chỉ tải 14 validation shard:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ILSVRC/imagenet-1k",
    repo_type="dataset",
    allow_patterns=["data/validation-*.parquet"],
    local_dir=os.environ["DS_DATA_ROOT"],
    token=True,
    max_workers=2,
)
```

### 4.2 Tải train để fine-tune

Fine-tune cần thêm 294 training shard, khoảng 146 GB. Cell có thể chạy lại an
toàn; file đã hoàn tất sẽ được tái sử dụng.

```python
snapshot_download(
    repo_id="ILSVRC/imagenet-1k",
    repo_type="dataset",
    allow_patterns=["data/train-*.parquet"],
    local_dir=os.environ["DS_DATA_ROOT"],
    token=True,
    max_workers=2,
)
```

Không cần materialize 1,28 triệu ảnh thành ImageFolder. Repo đọc trực tiếp các
Parquet shard từ Drive.

### 4.3 Kiểm tra dataset

```python
from pathlib import Path

data_dir = Path(os.environ["DS_DATA_ROOT"]) / "data"
train_shards = sorted(data_dir.glob("train-*.parquet"))
val_shards = sorted(data_dir.glob("validation-*.parquet"))

print("Train shards:", len(train_shards))
print("Validation shards:", len(val_shards))
print("Validation bytes:", sum(path.stat().st_size for path in val_shards))

assert len(val_shards) == 14
# Bỏ comment khi chuẩn bị fine-tune:
# assert len(train_shards) == 294
```

Validation đã dùng trong báo cáo có 14 shard, 50.000 ảnh và tổng
6.693.093.726 byte.

## 5. Tạo dense ResNet-18 checkpoint

Nếu Drive đã có `resnet18-dense.pth`, không cần tạo lại. Nếu chưa có, tải weight
ResNet-18 chính thức của PyTorch và chuyển legacy serialization sang checkpoint
hiện đại. `weights_only=False` chỉ dùng ở đây vì file đến từ domain chính thức của
PyTorch.

```python
from pathlib import Path
from urllib.request import urlretrieve
import os
import torch

dense_checkpoint = Path(os.environ["DS_CHECKPOINT_ROOT"]) / "resnet18-dense.pth"
legacy_checkpoint = Path("/content/resnet18-5c106cde.pth")

if not dense_checkpoint.exists():
    urlretrieve(
        "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        legacy_checkpoint,
    )
    state = torch.load(
        legacy_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    torch.save({"state_dict": state}, dense_checkpoint)

print("Dense checkpoint:", dense_checkpoint)
print("Bytes:", dense_checkpoint.stat().st_size)
```

## 6. Smoke test dense baseline

Smoke test chỉ dùng 1.000 validation sample và phải mang trạng thái `debug`. Nó
kiểm tra checkpoint, dataset và CUDA trước khi chạy đủ 50.000 ảnh.

```bash
!python benchmark/benchmark_model.py \
  --run-name dense-resnet18-smoke \
  --pruning-method dense \
  --experiment-status debug \
  --model resnet18_sparse \
  --checkpoint "$DS_CHECKPOINT_ROOT/resnet18-dense.pth" \
  --n 1 --m 1 \
  --dataset-format parquet \
  --parquet-root "$DS_DATA_ROOT" \
  --accuracy-batch-size 64 \
  --workers 2 \
  --max-eval-samples 1000 \
  --device cuda \
  --batch-size 1 \
  --warmup 30 \
  --iterations 100 \
  --seed 42 \
  --output "$DS_RESULT_ROOT/dense-resnet18-smoke.json"
```

Smoke test hợp lệ khi:

- checkpoint không có missing/unexpected key;
- accuracy được đánh giá trên đúng 1.000 sample;
- latency dùng Tesla T4;
- JSON có `experiment.status = debug`.

## 7. Chạy dense baseline đầy đủ

```bash
!python benchmark/benchmark_model.py \
  --run-name dense-resnet18-imagenet-t4 \
  --pruning-method dense \
  --experiment-status candidate \
  --model resnet18_sparse \
  --checkpoint "$DS_CHECKPOINT_ROOT/resnet18-dense.pth" \
  --n 1 --m 1 \
  --dataset-format parquet \
  --parquet-root "$DS_DATA_ROOT" \
  --accuracy-batch-size 64 \
  --workers 2 \
  --device cuda \
  --batch-size 1 \
  --warmup 30 \
  --iterations 100 \
  --seed 42 \
  --output "$DS_RESULT_ROOT/dense-resnet18-imagenet-t4.json"
```

Mốc tham chiếu của checkpoint này:

- Top-1 khoảng 69,754%;
- Top-5 khoảng 89,078%;
- 11.689.512 parameter;
- 1.814.073.344 MAC/sample.

Nếu accuracy lệch lớn, dừng lại và kiểm tra checkpoint/dataset trước khi pruning.

## 8. Hướng khuyến nghị: unstructured global 30%

### 8.1 Chuyển đúng branch

```bash
!git switch pruning-unstructured-magnitude
!git pull --ff-only
!git status --short
```

`git status --short` phải không có output trước khi tạo artifact, để JSON ghi
`source.dirty = false`.

### 8.2 Tạo checkpoint, mask và dense N:M scheme

```bash
!python pruning/unstructured_magnitude/prune_checkpoint.py \
  --model resnet18_sparse \
  --checkpoint "$DS_CHECKPOINT_ROOT/resnet18-dense.pth" \
  --scope global \
  --sparsity 0.30 \
  --output "$DS_CHECKPOINT_ROOT/unstructured-global30-before.pth"
```

Script tạo bốn artifact mà không overwrite dense checkpoint:

```text
unstructured-global30-before.pth
unstructured-global30-before.pth.masks.pth
unstructured-global30-before.pth.dense-scheme.txt
unstructured-global30-before.pth.json
```

Layer đầu và linear cuối được bảo vệ mặc định. Vì vậy 30% là sparsity trên các
weight đủ điều kiện; reduction toàn model đo được khoảng 28,63%.

### 8.3 Benchmark trước fine-tune

Chạy smoke trước bằng cách thêm `--max-eval-samples 1000` và đổi status thành
`debug`. Khi smoke pass, chạy candidate đầy đủ:

```bash
!python benchmark/benchmark_model.py \
  --run-name unstructured-global30-before \
  --pruning-method unstructured-magnitude \
  --density-source nonzero \
  --experiment-status candidate \
  --model resnet18_sparse \
  --checkpoint "$DS_CHECKPOINT_ROOT/unstructured-global30-before.pth" \
  --scheme-file "$DS_CHECKPOINT_ROOT/unstructured-global30-before.pth.dense-scheme.txt" \
  --n 16 --m 16 \
  --dataset-format parquet \
  --parquet-root "$DS_DATA_ROOT" \
  --accuracy-batch-size 64 \
  --workers 2 \
  --device cuda \
  --batch-size 1 \
  --warmup 30 \
  --iterations 100 \
  --seed 42 \
  --output "$DS_RESULT_ROOT/unstructured-global30-before.json"
```

Kết quả tham chiếu trước fine-tune là Top-1 khoảng 69,218%.

## 9. Fine-tune unstructured và benchmark sau fine-tune

### 9.1 Xác nhận đã tải train shards

```python
from pathlib import Path
import os

train_shards = list((Path(os.environ["DS_DATA_ROOT"]) / "data").glob("train-*.parquet"))
assert len(train_shards) == 294, f"Thiếu train shard: chỉ có {len(train_shards)}/294"
```

### 9.2 Fine-tune tái lập thí nghiệm đã báo cáo

Lệnh sau dùng đúng limited budget đã báo cáo: 3 epoch, mỗi epoch 50.000 training
sample, validation nội bộ 1.000 sample, LR 0.01. Kết quả này phục vụ tái lập; LR
0.01 đã làm accuracy unstructured giảm, nên phần 9.5 đề xuất LR thấp hơn.

```bash
!python train/classification_sparsity_level/train_imagenet.py \
  --config train/classification_sparsity_level/train_imagenet/configs/config_resnet18.yaml \
  --schemes_file "$DS_CHECKPOINT_ROOT/unstructured-global30-before.pth.dense-scheme.txt" \
  --initial-checkpoint "$DS_CHECKPOINT_ROOT/unstructured-global30-before.pth" \
  --weight-mask-file "$DS_CHECKPOINT_ROOT/unstructured-global30-before.pth.masks.pth" \
  --dataset-format parquet \
  --parquet-root "$DS_DATA_ROOT" \
  --train-parquet-pattern 'data/train-*.parquet' \
  --val-parquet-pattern 'data/validation-*.parquet' \
  --train-num-samples 50000 \
  --val-num-samples 1000 \
  --shuffle-buffer 10000 \
  --data-workers 2 \
  --epochs 3 \
  --base_lr 0.01 \
  --seed 42 \
  --save-every-epoch \
  --model_dir "$DS_RUN_ROOT/unstructured-global30-3epoch-train50k"
```

Checkpoint cuối dự kiến:

```text
$DS_RUN_ROOT/unstructured-global30-3epoch-train50k/model.pth-3
```

### 9.3 Resume khi Colab bị ngắt

`--save-every-epoch` lưu `model.pth-1`, `model.pth-2`, `model.pth-3`. Chạy lại
đúng lệnh trên với cùng `--model_dir`; training code đọc checkpoint state trong
thư mục và tiếp tục. Không đổi scheme, mask, seed hoặc tổng số epoch khi resume.

Trước khi resume, kiểm tra:

```bash
!ls -lh "$DS_RUN_ROOT/unstructured-global30-3epoch-train50k"
```

### 9.4 Benchmark checkpoint sau fine-tune

```bash
!python benchmark/benchmark_model.py \
  --run-name unstructured-global30-after-train50k \
  --pruning-method unstructured-magnitude \
  --density-source nonzero \
  --experiment-status candidate \
  --model resnet18_sparse \
  --checkpoint "$DS_RUN_ROOT/unstructured-global30-3epoch-train50k/model.pth-3" \
  --scheme-file "$DS_CHECKPOINT_ROOT/unstructured-global30-before.pth.dense-scheme.txt" \
  --n 16 --m 16 \
  --dataset-format parquet \
  --parquet-root "$DS_DATA_ROOT" \
  --accuracy-batch-size 64 \
  --workers 2 \
  --device cuda \
  --batch-size 1 \
  --warmup 30 \
  --iterations 100 \
  --seed 42 \
  --output "$DS_RESULT_ROOT/unstructured-global30-after-train50k.json"
```

Kết quả đã ghi nhận với LR 0.01 là 68,572% Top-1, thấp hơn mốc trước fine-tune
69,218%. Không chọn checkpoint sau fine-tune chỉ vì nó được train thêm.

### 9.5 Thí nghiệm tiếp theo được khuyến nghị

Tạo thư mục run mới, giữ nguyên checkpoint/mask và đổi duy nhất learning rate:

```text
LR 0.001  → unstructured-global30-lr001
LR 0.0001 → unstructured-global30-lr0001
```

Thay `--base_lr` và `--model_dir` trong lệnh phần 9.2. Không overwrite run LR
0.01. Chỉ nhận checkpoint mới nếu:

- Top-1 sau fine-tune vượt 69,218%;
- effective parameter/MAC không đổi;
- mask vẫn được giữ;
- benchmark dùng cùng 50.000 validation sample và cùng GPU/runtime.

### 9.6 Nhánh gradual unstructured mới

Nhánh `pruning-unstructured-gradual` triển khai phương án tăng sparsity dần và
hiện chỉ mới sẵn sàng để chạy, chưa có kết quả ImageNet. Khi bắt đầu thí nghiệm:

```bash
!git switch pruning-unstructured-gradual
!git pull --ff-only
!python train/classification_sparsity_level/train_imagenet.py \
  --config train/classification_sparsity_level/train_imagenet/configs/config_resnet18.yaml \
  --initial-checkpoint "$DS_CHECKPOINT_ROOT/resnet18-dense.pth" \
  --gradual-pruning-target 0.30 \
  --gradual-pruning-start-epoch 0 \
  --gradual-pruning-end-epoch 3 \
  --gradual-pruning-frequency 1 \
  --gradual-pruning-power 3 \
  --gradual-pruning-scope global \
  --dataset-format parquet \
  --parquet-root "$DS_DATA_ROOT" \
  --train-num-samples 50000 \
  --val-num-samples 1000 \
  --shuffle-buffer 10000 \
  --data-workers 2 \
  --epochs 5 \
  --base_lr 0.001 \
  --seed 42 \
  --save-every-epoch \
  --model_dir "$DS_RUN_ROOT/gradual-global30-5epoch-train50k"
```

Epoch 0 bắt đầu dense; epoch 3 đạt target; epoch 4 giữ mask để phục hồi. Benchmark
checkpoint cuối:

```bash
!python benchmark/benchmark_model.py \
  --run-name gradual-global30-after-train50k \
  --pruning-method unstructured-gradual \
  --density-source nonzero \
  --experiment-status candidate \
  --model resnet18_sparse \
  --checkpoint "$DS_RUN_ROOT/gradual-global30-5epoch-train50k/model.pth-5" \
  --n 16 --m 16 \
  --dataset-format parquet \
  --parquet-root "$DS_DATA_ROOT" \
  --accuracy-batch-size 64 \
  --workers 2 --device cuda \
  --batch-size 1 --warmup 30 --iterations 100 --seed 42 \
  --output "$DS_RESULT_ROOT/gradual-global30-after-train50k.json"
```

Để có mốc “pruned trước fine-tune”, dùng JSON one-shot global 30% đã tạo ở phần
8.3. Chỉ kết luận gradual tốt hơn nếu Top-1 cuối vượt 69,218% ở cùng sparsity và
cùng benchmark protocol.

## 10. Chạy ba hướng pruning còn lại

Mỗi hướng nên chạy trong notebook/session riêng nhưng dùng cùng Drive, dense
checkpoint, dataset và benchmark flags.

### 10.1 Uniform 3:4 conservative

```bash
!git switch pruning-uniform-nm
!git pull --ff-only
!python pruning/uniform_nm/generate_scheme.py \
  --model resnet18_sparse \
  --n 3 --m 4 \
  --keep-first-dense \
  --keep-last-dense \
  --keep-1x1-dense \
  --output "$DS_SCHEME_ROOT/uniform-3of4-conservative.txt"
```

Benchmark trước fine-tune dùng dense checkpoint, scheme trên,
`--pruning-method uniform-nm`, `--density-source nm`, `--n 3 --m 4`. Fine-tune
dùng:

```bash
!python train/classification_sparsity_level/train_imagenet.py \
  --config train/classification_sparsity_level/train_imagenet/configs/config_resnet18.yaml \
  --schemes_file "$DS_SCHEME_ROOT/uniform-3of4-conservative.txt" \
  --initial-checkpoint "$DS_CHECKPOINT_ROOT/resnet18-dense.pth" \
  --dataset-format parquet \
  --parquet-root "$DS_DATA_ROOT" \
  --train-num-samples 50000 --val-num-samples 1000 \
  --shuffle-buffer 10000 --data-workers 2 \
  --epochs 3 --base_lr 0.01 --seed 42 --save-every-epoch \
  --model_dir "$DS_RUN_ROOT/uniform-3of4-3epoch-train50k"
```

Benchmark sau fine-tune dùng `model.pth-3` và cùng scheme. Không truyền
`--weight-mask-file` cho N:M.

### 10.2 Domino mixed N:M

```bash
!git switch pruning-domino-mixed-nm
!git pull --ff-only
!python search/find_mix_from_dense_imagenet.py \
  --config search/script_resnet_ImageNet/configs/config_resnet18_img_mix_from_dense.yaml \
  --initial-checkpoint "$DS_CHECKPOINT_ROOT/resnet18-dense.pth" \
  --target-metric params \
  --target_sparsity 0.23 \
  --dataset-format parquet \
  --parquet-root "$DS_DATA_ROOT" \
  --train-num-samples 50000 \
  --val-num-samples 1000 \
  --shuffle-buffer 10000 \
  --data-workers 2 \
  --seed 42 \
  --model_dir "$DS_RUN_ROOT/domino-search-params23" \
  --scheme-output "$DS_SCHEME_ROOT/domino-params23.txt"
```

Sau search:

1. Benchmark dense checkpoint với `--scheme-file domino-params23.txt`,
   `--pruning-method domino-mixed-nm`, `--density-source nm`, `--n 16 --m 16`.
2. Fine-tune dense checkpoint với `--schemes_file domino-params23.txt`, không
   truyền mask.
3. Benchmark `model.pth-3` bằng cùng scheme.

Target `params` không tương đương target MAC. Run đã báo cáo giảm 30,27%
parameter nhưng chỉ giảm 9,56% MAC; muốn so công bằng với Uniform, chạy thêm
`--target-metric flops` ở budget MAC tương đương và lưu sang run/scheme mới.

### 10.3 Structured channel 10% L1

```bash
!git switch pruning-structured-channel
!git pull --ff-only
!python pruning/structured_channel/prune_checkpoint.py \
  --model resnet18_sparse \
  --checkpoint "$DS_CHECKPOINT_ROOT/resnet18-dense.pth" \
  --ratio 0.10 \
  --score l1 \
  --alignment 8 \
  --output "$DS_CHECKPOINT_ROOT/structured-channel10-l1-before.pth"
```

Script tạo checkpoint, `.masks.pth`, `.dense-scheme.txt` và manifest. Benchmark
dùng `--pruning-method structured-channel`, `--density-source nonzero`, scheme
dense và `--n 16 --m 16`. Fine-tune giống phần 9.2 nhưng đổi sang artifact
structured tương ứng.

Artifact hiện là dense-shape checkpoint có zero/mask, chưa phải compact model.
Do đó latency gần dense chưa chứng minh structured runtime đã nhanh hơn.

## 11. Sinh báo cáo Markdown và CSV

### 11.1 Báo cáo một hướng

```bash
!python benchmark/compare_results.py \
  "$DS_RESULT_ROOT/dense-resnet18-imagenet-t4.json" \
  "$DS_RESULT_ROOT/unstructured-global30-before.json" \
  "$DS_RESULT_ROOT/unstructured-global30-after-train50k.json" \
  --csv "$DS_REPORT_ROOT/unstructured-global30-comparison.csv" \
  --markdown "$DS_REPORT_ROOT/unstructured-global30-report.md" \
  --title "ResNet-18 unstructured global 30% trên Tesla T4"
```

### 11.2 Báo cáo tất cả hướng

Sau khi hoàn tất bốn hướng, liệt kê tường minh dense, before và after JSON. Không
đưa smoke/debug JSON vào bảng:

```bash
!python benchmark/compare_results.py \
  "$DS_RESULT_ROOT/dense-resnet18-imagenet-t4.json" \
  "$DS_RESULT_ROOT/uniform-3of4-before.json" \
  "$DS_RESULT_ROOT/uniform-3of4-after.json" \
  "$DS_RESULT_ROOT/domino-params23-before.json" \
  "$DS_RESULT_ROOT/domino-params23-after.json" \
  "$DS_RESULT_ROOT/structured-channel10-before.json" \
  "$DS_RESULT_ROOT/structured-channel10-after.json" \
  "$DS_RESULT_ROOT/unstructured-global30-before.json" \
  "$DS_RESULT_ROOT/unstructured-global30-after-train50k.json" \
  --csv "$DS_REPORT_ROOT/all-pruning-comparison.csv" \
  --markdown "$DS_REPORT_ROOT/all-pruning-report.md" \
  --title "So sánh pruning ResNet-18 trên Tesla T4"
```

Tên JSON phải khớp tên bạn đã dùng ở các lệnh benchmark. Không dùng wildcard nếu
thư mục còn debug run, vì debug có thể bị trộn nhầm vào báo cáo.

## 12. Checklist chấp nhận kết quả

Trước khi kết luận một hướng tốt hơn, kiểm tra:

- [ ] Dense và pruned dùng cùng dense checkpoint gốc.
- [ ] Validation đủ 50.000 ảnh và cùng 14 shard.
- [ ] Checkpoint load không có missing/unexpected key.
- [ ] Source branch/commit đúng và `dirty = false`.
- [ ] Scheme/mask đúng phương pháp của branch.
- [ ] Có JSON trước và sau fine-tune.
- [ ] Cùng seed, preprocessing, input size và benchmark settings.
- [ ] Fine-tune budget giống nhau giữa các hướng.
- [ ] Parameter/MAC reduction không bị gọi là runtime speedup.
- [ ] Latency chỉ được so khi chạy trên cùng GPU/runtime.

## 13. Kết quả kỳ vọng và cách diễn giải

Mốc đã audit để kiểm tra notebook:

| Model | Top-1 % | Param giảm % | MAC giảm % | Median ms |
| --- | ---: | ---: | ---: | ---: |
| Dense | 69,754 | 0,00 | 0,00 | 3,725 |
| Unstructured global 30%, trước FT | 69,218 | 28,63 | 21,89 | 3,775 |
| Unstructured global 30%, sau FT LR 0.01 | 68,572 | 28,63 | 21,89 | 3,712 |

Latency unstructured gần dense và sai khác nhỏ nằm trong nhiễu của một lần đo.
Kết luận đúng hiện tại là **giảm độ phức tạp lý thuyết và giữ accuracy tốt**, chưa
phải **model chạy nhanh hơn**. Xem phân tích cả bốn hướng tại
[`PRUNING_EXPERIMENT_REPORT_T4.md`](PRUNING_EXPERIMENT_REPORT_T4.md).

## 14. Lỗi thường gặp

### `credential propagation was unsuccessful` khi mount Drive

Restart runtime, đăng nhập đúng Google account trong Colab rồi chạy lại
`drive.mount`. Không dùng token Drive trong notebook.

### Hugging Face trả 401/403

Đảm bảo đã chấp nhận điều kiện ImageNet, token có quyền Read và
`notebook_login()` báo thành công.

### Colab hết disk

Đảm bảo `local_dir` trỏ vào `/content/drive/MyDrive/...`, không phải `/content`.
Chỉ tải validation trước; chỉ tải train khi bắt đầu fine-tune.

### `torch.load` báo lỗi `weights_only=True`

Chạy cell chuyển checkpoint ở phần 5. Không sửa benchmark để load không an toàn
mọi checkpoint.

### Accuracy gần 0%

Thường do checkpoint load sai, N:M quá mạnh hoặc scheme không đúng model. Kiểm tra
missing/unexpected key và luôn chạy dense baseline trước.

### MAC giảm nhưng latency tăng

N:M hiện tạo mask rồi vẫn gọi dense PyTorch operator. Đây là hành vi dự kiến của
implementation hiện tại; không sửa số liệu để làm latency đẹp hơn.
