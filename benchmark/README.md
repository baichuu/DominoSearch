# Benchmark pruning cho DominoSearch

Thư mục này tạo số liệu **trước và sau tối ưu** theo cùng một giao thức. Script
chính lưu toàn bộ kết quả thành JSON; script còn lại ghép nhiều JSON thành bảng
Markdown/CSV để so sánh.

Mỗi JSON còn lưu branch, commit, trạng thái dirty, seed, phương pháp pruning và
trạng thái thí nghiệm. Mặc định là `--experiment-status debug`; chỉ chuyển thành
`candidate` hoặc `final` sau khi các điều kiện self-check trong `AGENTS.md` đạt.

## Các chỉ số được đo

- **Top-1 / Top-5 accuracy**: độ chính xác trên tập validation.
- **Dense parameters**: tổng số tham số thật trong model PyTorch.
- **Effective parameters**: số tham số khác 0 theo lý thuyết N:M.
- **Dense / effective MACs**: phép nhân-cộng trước/sau N:M; một MAC là một phép
  multiply-accumulate.
- **Median / P95 latency**: trung vị và phân vị 95 của thời gian inference.
- **Throughput**: số sample/giây.
- **Peak GPU memory**: đỉnh bộ nhớ CUDA trong lúc đo, nếu dùng GPU.

`effective parameters` không phải kích thước file checkpoint. Repo hiện lưu tensor
dense và tạo mask lúc chạy. Muốn file nhỏ thật hoặc FPGA dùng ít BRAM hơn, bước
triển khai phải đóng gói riêng giá trị khác 0 và metadata của mask.

## Ba mốc bắt buộc nên so sánh

1. **Dense baseline**: checkpoint ban đầu, N=M.
2. **Pruned, chưa fine-tune**: cùng checkpoint ban đầu và scheme tìm được. Mốc này
   cho biết pruning trực tiếp làm giảm accuracy bao nhiêu.
3. **Pruned, đã fine-tune**: checkpoint sau dynamic sparse training và cùng scheme.
   Mốc này cho biết fine-tune phục hồi accuracy đến mức nào.

Ba lần phải giữ nguyên model, validation set, input size, batch size, device,
PyTorch, warm-up và số iteration. Không so latency giữa hai loại máy/board khác
nhau.

## Cài đặt và đo nhanh

```bash
pip install -r requirements.txt

# Không cần dataset: đo complexity và runtime bằng input ngẫu nhiên.
python benchmark/benchmark_model.py \
  --run-name dense-synthetic \
  --pruning-method dense \
  --experiment-status debug \
  --model resnet18_sparse \
  --checkpoint /path/to/dense_checkpoint.pth \
  --n 1 --m 1 \
  --device auto \
  --output results/dense-synthetic.json
```

Khi không có dataset, accuracy trong JSON là `not evaluated`.

## Đo đầy đủ với ImageFolder

`/path/to/imagenet/val` phải có mỗi class là một thư mục con:

```bash
python benchmark/benchmark_model.py \
  --run-name dense \
  --pruning-method dense \
  --model resnet18_sparse \
  --checkpoint /path/to/dense_checkpoint.pth \
  --n 1 --m 1 \
  --dataset-format imagefolder \
  --data-root /path/to/imagenet/val \
  --accuracy-batch-size 64 \
  --batch-size 1 \
  --warmup 30 \
  --iterations 100 \
  --output results/dense.json
```

Nếu dataset dùng file `val.txt`, mỗi dòng là `relative/image.jpg class_id`, thay
phần dataset bằng:

```bash
--dataset-format meta \
--val-root /path/to/imagenet \
--val-source /path/to/val.txt
```

## Đo pruning trước và sau fine-tune

```bash
# Cùng dense checkpoint, chỉ thêm mask: đo tổn thất do pruning.
python benchmark/benchmark_model.py \
  --run-name pruned-before-finetune \
  --pruning-method domino-mixed-nm \
  --model resnet18_sparse \
  --checkpoint /path/to/dense_checkpoint.pth \
  --scheme-file /path/to/searched_scheme.txt \
  --dataset-format imagefolder \
  --data-root /path/to/imagenet/val \
  --output results/pruned-before-finetune.json

# Checkpoint sau fine-tune: đo mức accuracy được phục hồi.
python benchmark/benchmark_model.py \
  --run-name pruned-after-finetune \
  --pruning-method domino-mixed-nm \
  --model resnet18_sparse \
  --checkpoint /path/to/finetuned_sparse_checkpoint.pth \
  --scheme-file /path/to/searched_scheme.txt \
  --dataset-format imagefolder \
  --data-root /path/to/imagenet/val \
  --output results/pruned-after-finetune.json
```

File scheme là dictionary do bước search của DominoSearch sinh ra. Có thể thêm
`--max-eval-samples 1000` để smoke test, nhưng không nên dùng tập con nhỏ làm kết
quả cuối vì sai số lớn.

Với checkpoint đã materialize số 0 của structured hoặc unstructured pruning, dùng
`--density-source nonzero`. Với N:M động của `SparseConv`/`SparseLinear`, giữ mặc
định `--density-source nm` vì dense weight trong checkpoint chưa chứa mask.

Fine-tune checkpoint có mask bằng `train_imagenet.py --initial-checkpoint ...
--weight-mask-file ...`. Mask được kiểm tra đúng tên/shape, nhân vào gradient và
được áp lại sau mỗi `optimizer.step`, nên weight đã prune không tự mọc lại.

## Tạo bảng so sánh

```bash
python benchmark/compare_results.py \
  results/dense.json \
  results/pruned-before-finetune.json \
  results/pruned-after-finetune.json \
  --csv results/comparison.csv
```

## Google Colab và FPGA

Trên Colab, chạy cùng lệnh với `--device cuda`, giữ nguyên runtime và kiểm tra GPU
bằng `nvidia-smi`. Colab chứng minh được accuracy, parameter/MAC theo N:M và chi
phí của implementation PyTorch hiện tại.

Repo hiện tạo mask rồi vẫn gọi dense PyTorch operator. Vì vậy latency Colab/CPU
**không chứng minh được tốc độ FPGA** và model sparse thậm chí có thể chậm hơn do
chi phí tạo mask. Để kết luận trên board, phải export cùng model/scheme sang
toolchain của board và đo end-to-end: latency, throughput, công suất, BRAM, DSP,
LUT và kích thước binary.

Một kết luận “tối ưu hiệu quả” nên báo cáo đồng thời mức giảm effective MAC/param,
chênh lệch Top-1 so với dense, và latency/throughput trên đúng phần cứng mục tiêu.
Không nên chỉ dùng FLOPs/MACs để khẳng định model chắc chắn chạy nhanh hơn.
