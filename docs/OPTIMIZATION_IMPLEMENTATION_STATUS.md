# Trạng thái triển khai các hướng pruning

Tài liệu này là bản đồ chung để biết mỗi nhánh đang tối ưu điểm nào, code nằm ở
đâu và còn thiếu bằng chứng gì. `master` luôn giữ vai trò dense baseline cùng hạ
tầng dùng chung; implementation của từng phương pháp chỉ tồn tại trên đúng nhánh.

## Nguyên tắc trạng thái

- **Implemented**: đã có code tạo scheme/checkpoint/mask và hướng dẫn chạy.
- **Ready for evaluation**: code qua kiểm tra tĩnh nhưng chưa chạy benchmark thật.
- **Validated**: đã chạy dense/pruned trước và sau fine-tune cùng điều kiện.
- **Hardware validated**: đã đo trên đúng board/runtime mục tiêu.

Đến ngày 2026-08-04, mỗi hướng đã có một cấu hình được benchmark trước và sau
fine-tune trên đủ 50.000 ảnh ImageNet validation bằng Tesla T4. Vì mới có một
cấu hình mỗi hướng và fine-tune dùng budget giới hạn, trạng thái phù hợp là
**Validated on T4 (limited experiment)**, chưa phải hoàn tất toàn bộ sweep.

Số liệu và giới hạn diễn giải nằm tại
[`PRUNING_EXPERIMENT_REPORT_T4.md`](PRUNING_EXPERIMENT_REPORT_T4.md). Không hướng
nào được xác nhận tăng tốc runtime từ benchmark hiện tại.

## Tổng quan

| Branch | Điểm tối ưu | Implementation commit | Trạng thái |
| --- | --- | --- | --- |
| `master` | Dense baseline và hạ tầng chung | `d8254f6` | Baseline validated on T4 |
| `pruning-uniform-nm` | Một N:M đồng nhất, có thể bảo vệ boundary/1×1 | `59ed0b1` | 3:4 validated, limited FT |
| `pruning-domino-mixed-nm` | Mixed N:M theo parameter hoặc FLOPs budget | `956077e`, `77f4aac` | Params-23 validated, limited FT |
| `pruning-structured-channel` | Hidden-channel pruning an toàn trong residual block | `f12b328` | Channel-10 L1 validated, limited FT |
| `pruning-unstructured-magnitude` | Global/local exact magnitude masks | `d47df96` | Global-30 validated, limited FT |

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

Đã có bằng chứng cho Uniform 3:4 conservative: Top-1 tăng từ 66,094% trước
fine-tune lên 68,388% sau fine-tune; parameter/MAC giảm 23,49%/23,10%. Runtime
PyTorch hiện chậm hơn dense. Còn thiếu toàn bộ matrix U0–U5 và phép so mixed N:M
ở cùng effective budget.

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

Đã có params-23 scheme: Top-1 68,328% trước và 67,996% sau fine-tune; parameter
giảm 30,27% nhưng MAC chỉ giảm 9,56%. Còn thiếu search/fine-tune nhiều target và
phép so với Uniform ở cùng parameter hoặc MAC budget.

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

Đã có channel-10 L1: Top-1 51,066% trước và 66,672% sau fine-tune; parameter/MAC
giảm 8,99%/10,16%. Artifact vẫn là masked dense-shape checkpoint, chưa phải model
compact, nên latency hiện tại chưa chứng minh lợi ích của structured model
compact. Còn thiếu sweep, compact export và phép đo runtime sau compact.

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

Đã có global-30: Top-1 69,218% trước và 68,572% sau fine-tune; parameter/MAC giảm
28,63%/21,89%. Đây là điểm accuracy–complexity tốt nhất trong các run hiện có,
nhưng không giảm runtime. Còn thiếu sweep, global-vs-local và sparse runtime hỗ
trợ irregular pattern.

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
