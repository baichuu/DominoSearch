# Trạng thái triển khai các hướng pruning

Tài liệu này là bản đồ chung để biết mỗi nhánh đang tối ưu điểm nào, code nằm ở
đâu và còn thiếu bằng chứng gì. `master` luôn giữ vai trò dense baseline cùng hạ
tầng dùng chung; implementation của từng phương pháp chỉ tồn tại trên đúng nhánh.

## Nguyên tắc trạng thái

- **Implemented**: đã có code tạo scheme/checkpoint/mask và hướng dẫn chạy.
- **Ready for evaluation**: code qua kiểm tra tĩnh nhưng chưa chạy benchmark thật.
- **Validated**: đã chạy dense/pruned trước và sau fine-tune cùng điều kiện.
- **Hardware validated**: đã đo trên đúng board/runtime mục tiêu.

Hiện tại các hướng mới đạt **Implemented / Ready for evaluation**. Chưa có
checkpoint, ImageNet và PyTorch runtime trong môi trường phát triển hiện tại, nên
không hướng nào được gọi là “tốt nhất” hoặc “đã tăng tốc”.

## Tổng quan

| Branch | Điểm tối ưu | Implementation commit | Trạng thái |
| --- | --- | --- | --- |
| `master` | Dense baseline và hạ tầng chung | `766a809` | Shared baseline |
| `pruning-uniform-nm` | Một N:M đồng nhất, có thể bảo vệ boundary/1×1 | `59ed0b1` | Ready for evaluation |
| `pruning-domino-mixed-nm` | Mixed N:M theo parameter hoặc FLOPs budget | `956077e` | Ready for evaluation |
| `pruning-structured-channel` | Hidden-channel pruning an toàn trong residual block | `f12b328` | Ready for evaluation |
| `pruning-unstructured-magnitude` | Global/local exact magnitude masks | `d47df96` | Ready for evaluation |

Các commit merge sau đó có thể làm branch tip thay đổi; cột implementation commit
chỉ ra commit chứa thay đổi cốt lõi của phương pháp.

## `master`: dense baseline

`master` không áp dụng một phương pháp pruning cụ thể. Các thay đổi dùng chung:

- `benchmark/benchmark_model.py`: JSON benchmark có Git/method/status và hỗ trợ
  density từ N:M hoặc số non-zero materialized.
- `benchmark/compare_results.py`: bảng hiển thị method, status và branch.
- `train/devkit/core/mask_utils.py`: exact checkpoint loading và persistent mask.
- `train/classification_sparsity_level/train_imagenet.py`: chỉ dùng mask khi truyền
  `--weight-mask-file`; mặc định không kích hoạt pruning mới.

Dense baseline dùng N=M, không có scheme sparse và không có mask.

## `pruning-uniform-nm`

Điểm tối ưu:

- sweep `3:4`, `2:4`, `1:4`, `8:16`, `4:16`;
- so prune toàn bộ với giữ layer đầu/cuối dense;
- thử giữ convolution 1×1 dense;
- tìm cấu hình đều, đơn giản cho accelerator nhưng vẫn giữ accuracy.

Code trên nhánh:

```text
pruning/uniform_nm/generate_scheme.py
docs/optimizations/UNIFORM_NM.md
```

Artifact: exact scheme text và manifest JSON. Fine-tune bằng SR-STE hiện có.

Bằng chứng còn thiếu: accuracy trước/sau fine-tune cho toàn bộ matrix U0–U5 và so
với mixed N:M ở cùng effective budget.

## `pruning-domino-mixed-nm`

Điểm tối ưu:

- search theo parameter reduction hoặc FLOPs reduction;
- model-size objective mặc định 0.5 ERK + 0.5 cost;
- FLOPs objective mặc định 0.2 ERK + 0.8 FLOPs theo bài báo;
- CLI cho vote ratio và ablation penalty weights;
- manifest chứa target và giá trị thực tế khi search dừng.

Code/tài liệu trên nhánh:

```text
search/find_mix_from_dense_imagenet.py
docs/optimizations/DOMINO_MIXED_NM.md
```

Bằng chứng còn thiếu: search/fine-tune các target 50–80%, sau đó so với uniform ở
cùng parameter hoặc MAC budget.

## `pruning-structured-channel`

Điểm tối ưu:

- chỉ prune hidden channel không phá residual output shape;
- hỗ trợ BasicBlock và Bottleneck;
- xếp hạng bằng L1 filter hoặc BatchNorm gamma;
- alignment channel theo vector width/datapath;
- materialize zero và giữ mask xuyên suốt fine-tune.

Code/tài liệu trên nhánh:

```text
pruning/structured_channel/prune_checkpoint.py
docs/optimizations/STRUCTURED_CHANNEL.md
```

Artifact hiện là masked dense-shape checkpoint, chưa phải compact model. Bằng
chứng còn thiếu: accuracy sweep 10–40%, compact export và latency thật trên
CPU/board/FPGA.

## `pruning-unstructured-magnitude`

Điểm tối ưu:

- global magnitude để tự phân bổ sparsity giữa layer;
- local magnitude làm đối chứng cùng tỷ lệ mỗi layer;
- chọn đúng số index nhỏ nhất bằng `topk`;
- mặc định bảo vệ conv đầu và linear cuối;
- materialize zero và persistent mask khi fine-tune.

Code/tài liệu trên nhánh:

```text
pruning/unstructured_magnitude/prune_checkpoint.py
docs/optimizations/UNSTRUCTURED_MAGNITUDE.md
```

Bằng chứng còn thiếu: accuracy sweep 50–90%, global-vs-local và kiểm tra sparse
runtime có thực sự hỗ trợ irregular pattern hay không.

## Trình tự benchmark chung

Mọi nhánh phải sinh ba kết quả:

1. Dense baseline.
2. Pruned trước fine-tune.
3. Pruned sau fine-tune.

Sau đó ghép bảng:

```bash
python benchmark/compare_results.py \
  results/dense.json \
  results/pruned-before.json \
  results/pruned-after.json \
  --csv results/comparison.csv
```

N:M dùng `--density-source nm`. Structured/unstructured checkpoint đã materialize
zero dùng `--density-source nonzero`.

## Điều kiện nâng trạng thái

Chỉ đổi từ `Ready for evaluation` sang `Validated` khi:

- checkpoint load exact, scheme/mask khớp toàn bộ layer dự kiến;
- cùng dataset, preprocessing, device, seed và fine-tune budget;
- có JSON cho dense, before fine-tune và after fine-tune;
- số non-zero/N:M đo được đúng yêu cầu;
- báo cáo Top-1/Top-5, effective parameters/MACs, latency và memory;
- kết quả debug không được trộn với candidate/final.

Chỉ đổi sang `Hardware validated` khi latency/throughput/resource được đo trên
đúng target. Effective MACs hoặc latency dense PyTorch không thay thế được phép đo
FPGA.
