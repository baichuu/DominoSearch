# Hiểu về Sparsity và dự án DominoSearch

> Để đo rõ hiệu quả trước/sau tối ưu, xem code và giao thức tại
> [`benchmark/README.md`](../benchmark/README.md). Báo cáo tách riêng accuracy,
> độ phức tạp lý thuyết và tốc độ thực đo để tránh kết luận sai.

Tài liệu này dành cho người mới tiếp cận pruning, sparse neural network và mã nguồn DominoSearch. Mục tiêu là giúp bạn hiểu:

1. Dự án đang giải quyết vấn đề gì.
2. Các thuật ngữ machine learning xuất hiện trong mã nguồn.
3. Thuật toán DominoSearch hoạt động như thế nào.
4. Các thư mục và file chính có vai trò gì.
5. Cần chuẩn bị gì trước khi chạy dự án.

> Đây là research code đi kèm bài báo NeurIPS 2021. Tài liệu này giải thích ý tưởng và mã nguồn hiện có; nó không khẳng định repo sẽ chạy nguyên trạng trên PyTorch mới.

---

## 1. Bức tranh tổng thể

Một mạng neural như ResNet-50 chứa hàng chục triệu con số được gọi là **trọng số** (`weights`). Khi nhận một bức ảnh, model thực hiện rất nhiều phép nhân và phép cộng với những trọng số này để dự đoán ảnh thuộc lớp nào.

Ví dụ đơn giản:

```text
đầu vào   = [2, 3, 4]
trọng số  = [0.5, -0.2, 0.8]

kết quả = 2×0.5 + 3×(-0.2) + 4×0.8
```

Model càng lớn thì thường càng:

- Chiếm nhiều bộ nhớ.
- Tốn nhiều phép tính.
- Tốn điện hơn.
- Khó triển khai trên thiết bị nhỏ.

Tuy nhiên, không phải mọi trọng số đều quan trọng như nhau. DominoSearch tìm cách loại bỏ những trọng số kém quan trọng mà vẫn giữ độ chính xác của model ở mức tốt.

```text
ResNet dense đã được huấn luyện
              ↓
Tìm mức sparsity thích hợp cho từng layer
              ↓
Sinh dictionary: tên layer → [N, M]
              ↓
Fine-tune model với các mask N:M cố định
              ↓
Model sparse và checkpoint tốt nhất
```

Ý tưởng nổi bật là **không prune mọi layer với cùng một tỷ lệ**. Layer nhạy cảm được giữ nhiều trọng số hơn; layer dư thừa có thể bị prune mạnh hơn.

---

## 2. Kiến thức nền

### 2.1 Model, parameter và weight

**Model** là hàm biến input thành output. Trong bài toán ImageNet:

```text
ảnh đầu vào → model → xác suất của 1.000 lớp
```

**Parameter** là giá trị model học được trong quá trình training. Weight là loại parameter phổ biến nhất.

Model không được lập trình thủ công để biết hình dạng con chó hay con mèo. Nó học các weight từ dữ liệu bằng cách điều chỉnh chúng để giảm lỗi dự đoán.

### 2.2 Layer

Mạng neural gồm nhiều tầng, gọi là **layer**:

```text
Ảnh đầu vào
    ↓
Layer đầu: cạnh, màu và texture đơn giản
    ↓
Layer giữa: hình dạng và bộ phận
    ↓
Layer sâu: đặc trưng của vật thể
    ↓
Layer cuối: xác suất của từng lớp
```

Mỗi layer có tập weight riêng. Các layer không nhạy cảm với pruning giống nhau:

```text
Layer A rất nhạy cảm → prune nhẹ
Layer B dư thừa nhiều → prune mạnh
Layer C ở giữa        → mức prune trung bình
```

### 2.3 Dense và sparse

**Dense model** sử dụng toàn bộ weight:

```text
[0.8, -0.1, 0.5, 0.02]
```

**Sparse model** có một phần weight bằng 0:

```text
[0.8, 0, 0.5, 0]
```

**Sparsity** là tỷ lệ weight bị đưa về 0:

```text
4 weight, trong đó 2 weight bằng 0
sparsity = 2 / 4 = 50%
```

Công thức tổng quát:

```text
sparsity = số weight bằng 0 / tổng số weight
```

Sparsity 80% nghĩa là 80% weight bị loại và 20% được giữ lại.

### 2.4 Pruning

**Pruning** là quá trình loại bỏ weight được xem là ít quan trọng.

Một cách phổ biến là magnitude pruning: so sánh trị tuyệt đối của weight.

```text
Weight:  [0.90, -0.03, 0.70, 0.01]
Độ lớn:  [0.90,  0.03, 0.70, 0.01]

Sau khi bỏ hai số nhỏ nhất:
[0.90, 0, 0.70, 0]
```

Đây là một heuristic, tức quy tắc kinh nghiệm. Weight nhỏ thường ít ảnh hưởng hơn, nhưng điều đó không đúng tuyệt đối trong mọi trường hợp. Vì vậy model thường cần được fine-tune sau pruning.

### 2.5 Mask

**Mask** là tensor gồm 0 và 1, có cùng hình dạng với weight:

```text
Weight:       [0.8, -0.1, 0.6, 0.02]
Mask:         [1,    0,   1,    0]
Weight×Mask:  [0.8,  0,   0.6,  0]
```

- Mask bằng `1`: weight được giữ.
- Mask bằng `0`: weight bị loại khỏi forward pass.

### 2.6 N:M sparsity

Đây là khái niệm trung tâm của dự án.

**N:M sparsity** nghĩa là trong mỗi nhóm `M` weight, chỉ giữ đúng `N` weight.

Ví dụ `2:4`:

```text
Nhóm ban đầu:          [0.8, -0.1, 0.6, 0.02]
Giữ 2 số lớn nhất:     [0.8,  0,   0.6, 0]
```

Trong mỗi nhóm bốn số, hai số được giữ và hai số bị loại. Sparsity bằng 50%.

Với `M=16`, các scheme trong DominoSearch có ý nghĩa:

| Scheme  | Weight được giữ | Weight bị loại | Sparsity |
| ------- | --------------: | -------------: | -------: |
| `16:16` |           16/16 |           0/16 |       0% |
| `8:16`  |            8/16 |           8/16 |      50% |
| `4:16`  |            4/16 |          12/16 |      75% |
| `2:16`  |            2/16 |          14/16 |    87.5% |
| `1:16`  |            1/16 |          15/16 |   93.75% |

Công thức:

```text
sparsity = 1 - N/M
```

### 2.7 Unstructured và structured sparsity

**Unstructured sparsity** cho phép loại bất kỳ weight nào trong model. Cách này linh hoạt nhưng cấu trúc các số 0 không đều, khiến phần cứng khó tăng tốc hiệu quả.

**Structured sparsity** loại weight theo một quy luật. N:M là dạng **fine-grained structured sparsity**:

- `fine-grained`: quyết định ở mức weight nhỏ.
- `structured`: mỗi nhóm luôn tuân theo cấu trúc N:M.

Quy luật rõ ràng giúp phần cứng hoặc sparse kernel có khả năng xử lý hiệu quả hơn.

### 2.8 Layer-wise N:M scheme

**Layer-wise** nghĩa là riêng cho từng layer. **Scheme** nghĩa là phương án hoặc cấu hình.

Ví dụ:

```python
{
    "layer_1": [16, 16],
    "layer_2": [8, 16],
    "layer_3": [4, 16],
    "layer_4": [2, 16],
}
```

Cách đọc:

```text
layer_1: dense, không prune
layer_2: sparsity 50%
layer_3: sparsity 75%
layer_4: sparsity 87.5%
```

DominoSearch tìm dictionary loại này từ một dense model đã được huấn luyện.

---

## 3. Các khái niệm về model ảnh

### 3.1 Convolution

Convolution là phép toán chính trong nhiều model xử lý ảnh. Một kernel nhỏ, ví dụ 3×3, trượt qua ảnh hoặc feature map để phát hiện đặc trưng.

```text
Feature map
┌─────────────┐
│ ▣ ▣ ▣ . .   │
│ ▣ ▣ ▣ . .   │  ← kernel 3×3 đang nhìn vùng này
│ ▣ ▣ ▣ . .   │
│ . . . . .   │
└─────────────┘
```

Các kernel có thể học cách nhận ra cạnh, góc, texture, hình dạng và những đặc trưng phức tạp hơn.

### 3.2 SparseConv và SparseLinear

`SparseConv` vẫn thực hiện convolution, nhưng trước đó tạo mask N:M cho weight:

```text
Weight gốc
    ↓
Chia thành các nhóm M
    ↓
Giữ N weight lớn nhất trong mỗi nhóm
    ↓
Weight × mask
    ↓
Convolution
```

`SparseLinear` làm điều tương tự cho fully connected layer.

Implementation của pha search nằm tại [`devkit/sparse_ops/sparse_ops.py`](../devkit/sparse_ops/sparse_ops.py). Pha training dùng bản riêng tại [`train/devkit/sparse_ops/sparse_ops.py`](../train/devkit/sparse_ops/sparse_ops.py).

### 3.3 ResNet-18 và ResNet-50

ResNet là kiến trúc neural network cho computer vision. ResNet sử dụng residual connection, cho phép dữ liệu đi tắt qua một nhóm layer:

```text
x ─────────────────┐
│                  │
↓                  │
Conv → Conv        │
│                  │
└──── cộng với x ←─┘
```

Đường tắt này giúp huấn luyện mạng sâu ổn định hơn.

- ResNet-18 nhỏ và nhanh hơn.
- ResNet-50 lớn hơn, thường chính xác hơn nhưng tốn nhiều tài nguyên hơn.

Luồng search được repo chuẩn bị chủ yếu cho ResNet-18 và ResNet-50 trên ImageNet.

### 3.4 Pretrained model

**Pretrained model** là model đã được huấn luyện trước. DominoSearch không bắt đầu từ weight ngẫu nhiên:

```text
ResNet đã được huấn luyện trên ImageNet
              ↓
DominoSearch đánh giá layer nào có thể prune
              ↓
ResNet sparse
```

Điều này cung cấp một model dense chính xác làm điểm xuất phát và giảm chi phí so với training hoàn toàn từ đầu.

### 3.5 ImageNet

ImageNet là dataset phân loại ảnh lớn thường dùng để đánh giá computer vision model. Phiên bản phổ biến có khoảng 1,2 triệu ảnh training, 50.000 ảnh validation và 1.000 lớp.

Repo không chứa ImageNet. Người dùng phải tự chuẩn bị dataset và cấu hình đường dẫn. Pha search sử dụng cấu trúc dạng:

```text
imagenet/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── val/
    ├── class_1/
    ├── class_2/
    └── ...
```

Trên Colab, repo còn hỗ trợ opt-in `dataset-format=parquet` để stream các shard
ImageNet đã tải trực tiếp từ Google Drive. Cách này tránh tạo 1,28 triệu file ảnh
nhỏ và giữ nguyên loader ImageFolder/meta mặc định. Lệnh benchmark và cấu trúc
dataset được duy trì tại [`benchmark/README.md`](../benchmark/README.md).

---

## 4. DominoSearch hoạt động như thế nào?

Entry point của pha search là [`search/find_mix_from_dense_imagenet.py`](../search/find_mix_from_dense_imagenet.py).

### 4.1 Bước 1: bắt đầu từ dense model

Với `M=16`, mọi sparse layer bắt đầu ở `16:16`, tức chưa prune:

```yaml
N: 16
M: 16
```

Model tải pretrained weight và tiếp tục chạy trên ImageNet.

### 4.2 Bước 2: tạo threshold

**Threshold** là ngưỡng dùng để phân biệt weight mạnh và yếu.

```text
threshold = 0.2

|weight| > 0.2 → mạnh
|weight| < 0.2 → yếu
```

Ví dụ:

```text
[0.8, -0.1, 0.6, 0.02]

0.8  > 0.2 → mạnh
0.1  < 0.2 → yếu
0.6  > 0.2 → mạnh
0.02 < 0.2 → yếu
```

Trong DominoSearch, threshold được tính ở mức từng nhóm weight. Giá trị khởi tạo sử dụng trung bình của một phần các weight nhỏ nhất trong nhóm.

### 4.3 Bước 3: đếm weight sống sót

Chương trình kiểm tra mỗi nhóm có bao nhiêu weight lớn hơn threshold:

```text
Nhóm 1: 7 weight sống sót
Nhóm 2: 8 weight sống sót
Nhóm 3: 6 weight sống sót
...
```

Số này biểu thị layer có khả năng hoạt động với mức N thấp hơn hay không.

### 4.4 Bước 4: bỏ phiếu giữa các nhóm

Code đặt:

```python
vote_ratio = 0.75
```

Nếu ít nhất 75% số nhóm trong một layer cho thấy có thể giữ ít weight hơn, `N_intermediate` của layer giảm một đơn vị.

```text
100 nhóm trong layer
80 nhóm thỏa điều kiện

80% ≥ 75%
→ N_intermediate được giảm
```

Việc kiểm tra diễn ra định kỳ trong quá trình training.

### 4.5 Bước 5: chốt các mức N hợp lệ

`N_intermediate` có thể giảm dần:

```text
16 → 15 → 14 → ... → 8 → 7 → ... → 4 → 3 → 2 → 1
```

Nhưng scheme chỉ được chốt khi N thuộc tập lũy thừa của 2:

```text
16, 8, 4, 2, 1
```

Vì vậy một layer có thể lần lượt chuyển:

```text
16:16 → 8:16 → 4:16 → 2:16 → 1:16
```

### 4.6 Bước 6: tính lại ưu tiên

Khi scheme của một layer thay đổi, tổng sparsity và chi phí tính toán của model cũng thay đổi. DominoSearch tính lại normalization dựa trên ERK và FLOPs để định hướng các quyết định tiếp theo.

Trong code mặc định:

```python
w1 = 0.5  # ERK
w2 = 0.5  # FLOPs
```

Tức hai nguồn tín hiệu có tỷ trọng bằng nhau.

### 4.7 Bước 7: dừng khi đạt mục tiêu

Ví dụ:

```bash
--target_sparsity 0.80
```

Search tiếp tục cho đến khi sparsity toàn model đạt xấp xỉ 80%. Sau đó chương trình in scheme cuối cùng và thoát.

Ví dụ output rút gọn:

```python
{
    "SparseConv0_3-64-(7, 7)": [16, 16],
    "SparseConv1_64-64-(1, 1)": [8, 16],
    "SparseConv2_64-64-(3, 3)": [4, 16],
    "Linear0_2048-1000": [4, 16],
}
```

Tên `SparseConv2_64-64-(3, 3)` có thể đọc như sau:

- `SparseConv2`: sparse convolution thứ 2.
- `64-64`: 64 input channel và 64 output channel.
- `(3, 3)`: kernel 3×3.

### 4.8 Tại sao gọi là DominoSearch?

Khi một layer thay đổi scheme, mức ưu tiên của các layer còn lại được tính lại. Các quyết định xảy ra tuần tự và ảnh hưởng đến trạng thái tổng thể, tương tự các quân domino lần lượt ngã cho tới khi đạt mục tiêu sparsity.

---

## 5. ERK, FLOPs và penalty

### 5.1 ERK

ERK là viết tắt của **Erdős–Rényi Kernel**. Trong ngữ cảnh này, nó là heuristic phân bổ sparsity dựa trên hình dạng layer, ví dụ:

- Số input channel.
- Số output channel.
- Kích thước kernel.
- Tổng số parameter.

Trực giác đơn giản:

```text
Layer nhỏ, ít kết nối    → thường nên giữ dense hơn
Layer lớn, nhiều kết nối → thường có khả năng prune nhiều hơn
```

ERK không trực tiếp kết luận scheme cuối cùng. Nó cung cấp một tín hiệu để thuật toán điều chỉnh áp lực pruning giữa các layer.

### 5.2 FLOPs

FLOPs là số phép toán số thực cần thực hiện. Nó thường được dùng để ước lượng chi phí tính toán.

**Parameter count** và **FLOPs** không giống nhau:

- Parameter count: model lưu bao nhiêu weight.
- FLOPs: một lần chạy model cần bao nhiêu phép tính.

Một layer có thể không có quá nhiều parameter nhưng được áp dụng trên feature map lớn, do đó vẫn có FLOPs cao.

DominoSearch dùng thông tin FLOPs để quan tâm đến cả chi phí tính toán, không chỉ kích thước model.

### 5.3 Penalty và sparse decay

Penalty bổ sung một lực điều chỉnh vào gradient để khuyến khích cấu trúc weight phù hợp với pruning.

Có thể hình dung mục tiêu training gồm:

```text
loss dự đoán
+
penalty liên quan đến sparsity
```

Nếu chỉ tối ưu độ chính xác, model không có lý do để hình thành weight dễ prune. `sparse_decay` điều khiển độ mạnh của penalty này.

Trong pha search, code bật penalty định kỳ và có thể tăng `sparse_decay` khi số iteration đạt các mốc lớn.

---

## 6. Training và fine-tuning

Entry point của pha training là [`train/classification_sparsity_level/train_imagenet.py`](../train/classification_sparsity_level/train_imagenet.py).

### 6.1 Loss

**Loss** đo mức sai của dự đoán:

```text
Dự đoán đúng và tự tin → loss thấp
Dự đoán sai            → loss cao
```

Repo dùng cross-entropy loss cho phân loại ảnh.

### 6.2 Gradient

**Gradient** cho biết nên thay đổi mỗi weight theo hướng nào để loss giảm.

```text
weight hiện tại = 0.50
gradient đề xuất giảm weight
weight mới      = 0.48
```

### 6.3 Optimizer và SGD

**Optimizer** cập nhật weight từ gradient. Repo sử dụng SGD, viết tắt của Stochastic Gradient Descent.

Các tham số liên quan:

- `base_lr`: learning rate, độ lớn của bước cập nhật.
- `momentum`: giúp hướng cập nhật ổn định hơn.
- `weight_decay`: regularization để hạn chế weight quá lớn.
- `epochs`: số lần đi qua toàn bộ dataset.

### 6.4 Batch, iteration và epoch

Dataset lớn được chia thành các batch nhỏ để vừa GPU.

```text
1 iteration = xử lý một batch
1 epoch     = đi qua toàn bộ tập training một lần
```

Ví dụ:

```text
1.280.000 ảnh
batch size = 256

1 epoch ≈ 1.280.000 / 256 = 5.000 iteration
```

### 6.5 Fine-tune

Sau pruning, accuracy thường giảm vì model vừa mất nhiều kết nối. **Fine-tune** là tiếp tục training model sparse để các weight còn lại thích nghi.

```text
Dense accuracy:      76.5%
Ngay sau pruning:    70.0%
Sau fine-tune:       75.8%
```

Luồng training của repo:

1. Đọc dictionary scheme từ file text bằng `ast.literal_eval`.
2. Gán `[N, M]` cho từng `SparseConv` và `SparseLinear`.
3. Tạo magnitude-based mask trong mỗi forward pass.
4. Training bằng SR-STE.
5. Validate Top-1 và Top-5 accuracy.
6. Lưu checkpoint tốt nhất.

### 6.6 STE và SR-STE

Nếu backward chặn hoàn toàn gradient tại các weight bị mask, những weight đó không còn cơ hội thay đổi.

STE, viết tắt của **Straight-Through Estimator**, sử dụng:

```text
Forward:  dùng sparse weight sau mask
Backward: vẫn truyền gradient về dense weight
```

```text
Weight → mask → sparse weight → prediction
  ↑                                  │
  └────────── gradient ──────────────┘
```

SR-STE trong code còn thêm penalty cho phần weight bị prune. Điều này giúp refinement của cấu trúc sparse trong quá trình training.

---

## 7. Đánh giá model

### 7.1 Top-1 accuracy

Model đúng nếu lớp có xác suất cao nhất là nhãn thật:

```text
Nhãn thật: tiger

tiger  70%
lion   20%
cat    10%

→ Top-1 đúng
```

### 7.2 Top-5 accuracy

Model đúng nếu nhãn thật nằm trong năm dự đoán cao nhất:

```text
lion     35%
leopard  25%
tiger    20%  ← nhãn thật
cat      10%
dog       5%

→ Top-1 sai
→ Top-5 đúng
```

### 7.3 Checkpoint

Checkpoint thường lưu:

- Weight của model.
- Epoch hiện tại.
- Accuracy tốt nhất.
- Trạng thái optimizer.

Nhờ checkpoint, training có thể tiếp tục sau khi bị gián đoạn. Repo cũng lưu model có validation accuracy tốt nhất.

---

## 8. Distributed training

Training ResNet-50 trên ImageNet tốn nhiều thời gian, vì vậy repo hỗ trợ nhiều GPU.

Ví dụ:

```bash
--nproc_per_node=8
```

nghĩa là dùng tám process, thường tương ứng tám GPU:

```text
Batch tổng 256 ảnh

GPU 0: 32 ảnh
GPU 1: 32 ảnh
...
GPU 7: 32 ảnh
```

Các process đồng bộ gradient để cùng cập nhật một model.

Thuật ngữ thường gặp:

- `rank`: số thứ tự của process.
- `world_size`: tổng số process.
- `NCCL`: backend giao tiếp giữa GPU NVIDIA.
- `DistributedSampler`: chia dataset cho từng process.

Repo hiện giả định có CUDA và NCCL; không có CPU fallback hoàn chỉnh.

---

## 9. Layout NCHW và NHWC

Tensor ảnh thường có bốn chiều:

- `N`: số ảnh trong batch.
- `C`: số channel.
- `H`: chiều cao.
- `W`: chiều rộng.

Hai layout phổ biến:

```text
NCHW: batch, channel, height, width
NHWC: batch, height, width, channel
```

PyTorch chủ yếu dùng NCHW. Tuy nhiên, khi chia weight thành nhóm N:M, thứ tự chiều quyết định weight nào nằm chung nhóm. Repo có thể hoán vị weight để grouping theo NHWC, phù hợp hơn với một số cách tổ chức sparse data hoặc phần cứng.

---

## 10. Ví dụ tổng hợp

Giả sử model chỉ có ba layer:

```text
Layer A: 160 weight
Layer B: 320 weight
Layer C: 160 weight
Tổng:    640 weight
```

Ban đầu tất cả là `16:16`:

```text
Layer A giữ 160
Layer B giữ 320
Layer C giữ 160
Tổng giữ 640
Sparsity = 0%
```

Search nhận thấy:

- Layer A nhạy cảm, nên dùng `8:16`.
- Layer B dư thừa nhiều, nên dùng `2:16`.
- Layer C ở mức trung bình, nên dùng `4:16`.

Kết quả:

```text
Layer A: 160 × 8/16 = 80 weight được giữ
Layer B: 320 × 2/16 = 40 weight được giữ
Layer C: 160 × 4/16 = 40 weight được giữ

Tổng giữ = 160
Tổng gốc = 640
Sparsity = 1 - 160/640 = 75%
```

Output:

```python
{
    "LayerA": [8, 16],
    "LayerB": [2, 16],
    "LayerC": [4, 16],
}
```

Pha training sau đó áp scheme này và fine-tune model để phục hồi accuracy.

---

## 11. Bản đồ mã nguồn

### 11.1 Các thành phần chính

```text
DominoSearch/
├── README.md
├── docs/
│   └── UNDERSTANDING_DOMINOSEARCH.md
├── search/
│   ├── find_mix_from_dense_imagenet.py
│   ├── models/
│   │   └── resnet_sparse.py
│   └── script_resnet_ImageNet/
│       ├── configs/
│       └── *.run
├── devkit/
│   ├── core/
│   ├── dataset/
│   └── sparse_ops/
└── train/
    ├── classification_sparsity_level/
    │   ├── train_imagenet.py
    │   ├── models/
    │   └── train_imagenet/
    │       ├── configs/
    │       ├── schemes/
    │       └── *.sh
    └── devkit/
```

### 11.2 File nên đọc theo thứ tự

1. [`README.md`](../README.md): mục tiêu, lệnh mẫu và kết quả paper.
2. [`search/script_resnet_ImageNet/configs/config_resnet50_img_mix_from_dense.yaml`](../search/script_resnet_ImageNet/configs/config_resnet50_img_mix_from_dense.yaml): tham số search.
3. [`search/models/resnet_sparse.py`](../search/models/resnet_sparse.py): cách ResNet thay convolution thường bằng `SparseConv`.
4. [`devkit/sparse_ops/sparse_ops.py`](../devkit/sparse_ops/sparse_ops.py): mask, threshold và quyết định N:M trong pha search.
5. [`search/find_mix_from_dense_imagenet.py`](../search/find_mix_from_dense_imagenet.py): vòng lặp search và điều kiện dừng.
6. [`train/classification_sparsity_level/train_imagenet.py`](../train/classification_sparsity_level/train_imagenet.py): áp scheme và fine-tune.
7. [`train/devkit/sparse_ops/sparse_ops.py`](../train/devkit/sparse_ops/sparse_ops.py): SR-STE của pha training.

### 11.3 Hai thư mục `devkit`

Repo có hai implementation liên quan nhưng không hoàn toàn giống nhau:

- `devkit/`: được pha search sử dụng.
- `train/devkit/`: được pha training sử dụng.

Đây là một điểm dễ nhầm khi đọc import. Hãy luôn chú ý working directory và `sys.path` của command đang chạy.

---

## 12. File cấu hình YAML

YAML lưu tham số để không phải sửa trực tiếp Python code.

Ví dụ:

```yaml
model: resnet50_sparse
N: 16
M: 16
batch_size: 64
epochs: 120
layout: NHWC
data: your/data/repo
```

Ý nghĩa:

| Trường         | Ý nghĩa                                |
| -------------- | -------------------------------------- |
| `model`        | Kiến trúc được tạo                     |
| `N`, `M`       | Scheme N:M ban đầu                     |
| `batch_size`   | Số ảnh trong batch tổng                |
| `epochs`       | Số vòng qua dataset                    |
| `layout`       | Cách sắp xếp weight trước khi grouping |
| `data`         | Đường dẫn ImageNet trong pha search    |
| `finetue_lr`   | Learning rate trong search             |
| `sparse_decay` | Độ mạnh của sparse penalty             |
| `workers`      | Số worker đọc dữ liệu                  |
| `print_freq`   | Tần suất in log                        |

Lưu ý file training dùng `train_root`, `train_source`, `val_root` và `val_source`, vì nó sử dụng dataset loader khác pha search.

Khi dùng Parquet trên Drive, cả search và fine-tune nhận `parquet_root`, glob
train/validation, sample count và shuffle buffer qua CLI. Các trường cũ vẫn được
dùng mặc định nên không làm thay đổi thí nghiệm gốc.

---

## 13. Quy trình chạy trên PyTorch hiện đại

### 13.1 Pha search

Trước tiên sửa đường dẫn ImageNet trong file YAML, sau đó:

```bash
cd search/script_resnet_ImageNet

python -u ../find_mix_from_dense_imagenet.py \
  --target_sparsity 0.80 \
  --port 64485 \
  --config configs/config_resnet50_img_mix_from_dense.yaml
```

Theo README, pha này có thể mất vài giờ. Khi đạt mục tiêu, code in scheme ra terminal và tự lưu vào `<model_dir>/searched_scheme.txt`. Có thể chọn đường dẫn khác bằng `--scheme-output`.

### 13.2 Sử dụng scheme đã lưu

Mặc định scheme nằm trong `<model_dir>/searched_scheme.txt`. Có thể truyền file này trực tiếp cho `--schemes_file`, hoặc copy nó vào thư mục training, ví dụ:

```text
train/classification_sparsity_level/train_imagenet/schemes/resnet50_M16_0.80.txt
```

Nội dung phải là Python dictionary hợp lệ vì training đọc bằng `ast.literal_eval`.

### 13.3 Pha fine-tune

```bash
cd train/classification_sparsity_level/train_imagenet

torchrun --standalone --nproc-per-node=1 \
  ../train_imagenet.py \
  --config configs/config_resnet50.yaml \
  --base_lr 0.01 \
  --decay 0.0005 \
  --epochs 120 \
  --schemes_file schemes/resnet50_M16_0.80.txt \
  --model_dir resnet50/resnet50_0.80_M16
```

Đặt `--nproc-per-node` bằng số GPU khi chạy nhiều GPU. Trên Colab một GPU, có thể gọi trực tiếp `python ../train_imagenet.py ...`; code sẽ tự chuyển sang single-GPU mode mà không tạo distributed process group.

---

## 14. Những giới hạn cần biết trước khi chạy

Đây là research code từ khoảng năm 2021. Repository đã được cập nhật các điểm compatibility chính:

- Có `requirements.txt` cho các dependency ngoài PyTorch/torchvision.
- Single-GPU chạy trực tiếp bằng `python`; multi-GPU dùng `torchrun`.
- Distributed reduction dùng `dist.all_reduce` hiện đại.
- YAML được đọc bằng `yaml.safe_load`.
- Import `torch._six` đã được loại bỏ.
- Giả định có CUDA và NCCL.
- Search tải pretrained weight qua URL torchvision cũ.
- Search tự lưu scheme trước khi kết thúc khi đạt target sparsity.
- Có hai bản `devkit`, làm tăng nguy cơ import nhầm.
- Không có automated test hoặc license file trong repo hiện tại.

Vẫn nên ghi lại chính xác phiên bản Python, PyTorch, torchvision và CUDA của từng thí nghiệm để bảo đảm khả năng tái lập.

---

## 15. Model size, FLOPs và tốc độ thực tế

Ba khái niệm này có liên quan nhưng không đồng nhất.

### Giảm model size

Ít weight khác 0 hơn có thể giúp giảm dung lượng lưu trữ:

```text
Dense:  100 MB
Sparse:  20 MB về mặt lý thuyết
```

Trong thực tế còn cần metadata để mô tả vị trí các weight được giữ.

### Giảm FLOPs

Nếu có thể bỏ qua phép tính với zero weight, số phép toán lý thuyết giảm:

```text
Dense:  4 tỷ phép tính
Sparse: 1 tỷ phép tính
```

### Tăng tốc thực tế

Speedup là thời gian chạy thật giảm:

```text
Dense:  20 ms/ảnh
Sparse: 10 ms/ảnh
```

Model có nhiều zero weight không tự động chạy nhanh hơn. Nếu framework vẫn gọi dense convolution, GPU có thể vẫn thực hiện gần như cùng lượng công việc. Muốn có speedup thật cần:

- Phần cứng hỗ trợ N:M sparsity.
- Sparse kernel phù hợp.
- Runtime biết cách mã hóa và thực thi sparse weight.
- Scheme N:M tương thích với phần cứng đó.

Repo tập trung vào việc tìm scheme và training model sparse. Nó không cung cấp một inference runtime tối ưu hoàn chỉnh để chứng minh speedup trên mọi thiết bị.

---

## 16. Từ điển thuật ngữ nhanh

| Thuật ngữ     | Giải thích ngắn                                      |
| ------------- | ---------------------------------------------------- |
| Accuracy      | Tỷ lệ dự đoán đúng                                   |
| Batch         | Nhóm sample được xử lý cùng lúc                      |
| Checkpoint    | File lưu model và trạng thái training                |
| Dense         | Sử dụng toàn bộ weight                               |
| Epoch         | Một lượt đi qua toàn bộ training dataset             |
| ERK           | Heuristic phân bổ sparsity theo hình dạng layer      |
| Fine-tune     | Training tiếp một model đã có weight                 |
| FLOPs         | Số phép toán số thực ước tính                        |
| Forward pass  | Tính output từ input                                 |
| Gradient      | Tín hiệu cho biết cách cập nhật weight               |
| ImageNet      | Dataset phân loại ảnh lớn                            |
| Iteration     | Một lần xử lý batch và cập nhật model                |
| Layer         | Một tầng xử lý trong neural network                  |
| Learning rate | Độ lớn của bước cập nhật weight                      |
| Loss          | Số đo mức sai của model                              |
| Mask          | Tensor 0/1 dùng để giữ hoặc loại weight              |
| N:M           | Giữ N weight trong mỗi nhóm M weight                 |
| Optimizer     | Thuật toán cập nhật parameter                        |
| Parameter     | Giá trị model học được                               |
| Penalty       | Thành phần bổ sung để định hướng optimization        |
| Pretrained    | Đã được training trước                               |
| Pruning       | Loại weight kém quan trọng                           |
| ResNet        | Kiến trúc neural network có residual connection      |
| Scheme        | Cấu hình N:M của layer                               |
| SGD           | Một optimizer phổ biến                               |
| Sparse        | Có nhiều weight bằng 0                               |
| Sparsity      | Tỷ lệ weight bằng 0                                  |
| SR-STE        | Cách backward qua sparse mask kèm refinement penalty |
| Threshold     | Ngưỡng phân biệt weight mạnh và yếu                  |
| Weight        | Parameter dùng trong phép tính của model             |

---

## 17. Lộ trình học đề xuất

Nếu các khái niệm vẫn còn mới, có thể đọc và thực hành theo thứ tự:

1. Hiểu tensor, weight, forward pass và loss.
2. Hiểu convolution và kiến trúc ResNet ở mức trực giác.
3. Hiểu training loop: batch, loss, backward, optimizer.
4. Tự tạo một vector nhỏ và áp magnitude mask bằng tay.
5. Thử các scheme `2:4`, `4:8` và tính sparsity.
6. Đọc `Sparse.forward()` trong training sparse ops.
7. Đọc cách `SparseConv.forward()` gọi masked weight.
8. Đọc `check_sparsity_each_group()` của pha search.
9. Đọc `adjust_N_M_of_each_layer_based_on_each_group()` để thấy cách scheme được cập nhật.
10. Cuối cùng đọc toàn bộ `main()` và training loop.

Khi đọc code, luôn tự hỏi bốn câu:

```text
Input của hàm là gì?
Output của hàm là gì?
State nào của layer bị thay đổi?
Thay đổi đó ảnh hưởng forward/backward thế nào?
```

---

## 18. Tóm tắt cuối cùng

DominoSearch làm ba việc chính:

1. Lấy một ResNet pretrained đang hoạt động tốt.
2. Tìm mức N:M riêng cho từng layer cho đến khi đạt sparsity mục tiêu.
3. Fine-tune model với scheme đã tìm để phục hồi accuracy.

Điểm cốt lõi:

```text
Không prune tất cả layer như nhau.

Layer quan trọng  → giữ nhiều weight.
Layer ít nhạy cảm → giữ ít weight.
```

Đó là lý do output của DominoSearch không chỉ là một con số sparsity, mà là cả một dictionary cấu hình N:M cho từng layer.
