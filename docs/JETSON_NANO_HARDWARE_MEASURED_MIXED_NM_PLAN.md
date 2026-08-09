# Kế hoạch mixed N:M dựa trên số đo thật của Jetson Nano

## 1. Tóm tắt đề xuất

Mục tiêu là tạo một ResNet-18 mixed N:M có độ chính xác tốt và chứng minh được
lợi ích triển khai trên Jetson Nano bằng số liệu thực đo. Công việc được chia
thành hai vòng độc lập nhưng liên kết với nhau:

```text
Colab/ImageNet                         Jetson Nano
-----------------------------          --------------------------------
đo sensitivity từng layer              triển khai backend/kernel thật
chọn và fine-tune MAC candidates   ->  đo layer latency/memory/energy
full validation 50.000 ảnh             tạo hardware lookup
                                       chọn lại scheme theo measured cost
                                  <-   fine-tune scheme mới trên Colab
                                       benchmark end-to-end trên board
```

Colab tạo các ứng viên accuracy-complexity; Jetson quyết định ứng viên nào thực
sự có lợi cho deployment. FLOPs/MAC, latency T4 hoặc thông số lý thuyết không
được thay thế số đo Jetson.

Giải pháp đề xuất kết hợp:

- staged/conditioned sensitivity của nhánh `pruning-domino-mixed-nm`;
- direct target-platform measurement theo tinh thần NetAdapt;
- latency lookup và saliency-constrained allocation theo HALP;
- packed weight và runtime-aware candidate gating theo các nghiên cứu sparse
  kernel;
- end-to-end validation để phát hiện sai số của phép cộng layer latency.

Không đưa quantization, distillation, architecture replacement hoặc pruning
khác vào kết quả mixed N:M. Nếu nghiên cứu về sau, chúng phải là thí nghiệm kết
hợp riêng.

## 2. Câu hỏi nghiên cứu

### 2.1. Câu hỏi chính

1. Staged sensitivity-aware mixed N:M có tạo được Pareto point tốt hơn kết quả
   MAC18 hiện tại không?
2. Pattern N:M nào thực sự được runtime/kernel trên Jetson Nano khai thác?
3. Scheme chọn theo measured Jetson cost có tốt hơn scheme chọn theo MAC khi đo
   end-to-end trên cùng board không?
4. Lợi ích hoặc overhead đến từ pruning, packed representation, kernel hay
   framework?

### 2.2. Giả thuyết

- H1: conditioned profiling từ MAC18 có thể đạt ít nhất 20% effective MAC
  reduction mà mất không quá 0,30 điểm Top-1 sau fine-tune.
- H2: masked-dense PyTorch không giảm latency vì vẫn gọi dense convolution và
  thêm chi phí tạo mask.
- H3: chỉ các `layer × N:M` có packed kernel phù hợp mới có khả năng giảm
  latency hoặc DRAM traffic thật trên Jetson Nano.
- H4: candidate gating theo microbenchmark thật sẽ đáng tin cậy hơn việc cho
  selector sử dụng mọi pattern N:M dựa trên MAC.

H2-H4 chỉ là giả thuyết trước khi có board. Không ghi chúng như kết quả.

## 3. Baseline và phạm vi so sánh

### 3.1. Baseline accuracy-complexity

| Mốc | Top-1 | Top-5 | Effective MAC reduction | Effective param reduction |
| --- | -----: | -----: | ----------------------: | ------------------------: |
| Dense | 69,754% | 89,078% | 0% | 0% |
| MAC18 epoch 1 | 69,654% | 89,124% | 18,322% | 15,216% |

MAC18 là checkpoint đầu vào của stage tiếp theo. Scheme debug MAC20 cũ không
được dùng làm bằng chứng vì được sinh từ profile MAC15 chỉ có 1.000 ảnh.

### 3.2. Baseline paper

DominoSearch báo cáo ResNet-18:

| Mốc paper | Params | FLOPs | Top-1 |
| --- | ---: | ---: | ---: |
| Dense | 11,7M | 1.814M | 69,8% |
| DS, equal model size | 1,46M | 329M | 68,76% |
| DS, equal FLOPs | 1,29M | 227M | 67,98% |

Một kết quả chỉ được gọi là tốt hơn paper về accuracy-complexity khi so tại
constraint tương đương. MAC18 chính xác hơn nhưng chưa đạt mức compression của
hai điểm paper nên chưa phải so sánh ngang bằng.

### 3.3. Baseline deployment bắt buộc

Trên Jetson phải có ít nhất ba backend/run:

1. dense cuDNN hoặc TensorRT;
2. masked-dense PyTorch hiện tại;
3. packed N:M kernel, nếu feasibility gate đạt.

Mọi model dùng cùng kiến trúc, checkpoint nguồn, preprocessing, input 224×224,
batch 1, precision, power mode và điều kiện nhiệt.

## 4. Nguyên tắc xác định số liệu thật

Không tồn tại latency phần cứng hoàn toàn tách khỏi phần mềm: model luôn chạy qua
runtime, driver và kernel. Vì vậy báo cáo phải chỉ rõ `hardware + software stack`
đã đo và tách các tầng sau:

| Tầng đo | Nguồn | Ý nghĩa |
| --- | --- | --- |
| GPU execution | CUDA Events | Thời gian trên CUDA stream |
| Kernel/API timeline | Nsight Systems | Kernel, queue, copy và synchronization |
| Runtime end-to-end | C++ `clock_gettime` | Latency deployment, giảm ảnh hưởng Python |
| Application end-to-end | GPIO + logic analyzer, nếu có | Thời gian input-to-output vật lý |
| System memory/clock/thermal | `tegrastats`, sysfs | RAM, GPU/EMC clock, nhiệt độ, throttling |
| Power/energy | power meter hoặc shunt sensor ngoài | Phép đo độc lập với telemetry phần mềm |

Python được dùng để orchestration và accuracy evaluation. Kết luận latency chính
thức phải ưu tiên C++/CUDA/Nsight và phép đo ngoài thiết bị.

## 5. Giai đoạn A — Hoàn thiện candidates trên Colab

### 5.1. Kiểm tra đầu vào

- branch `pruning-domino-mixed-nm` và worktree sạch;
- exact MAC18 epoch-1 checkpoint SHA-256;
- exact MAC18 scheme SHA-256;
- đủ 14 validation Parquet shard, 50.000 ảnh;
- checkpoint load không có missing/unexpected key;
- scheme bao phủ đúng 21 sparse layer;
- `M=4`, candidates `2:4`, `3:4`, `4:4`;
- seed 42 và preprocessing giống dense baseline.

### 5.2. Conditioned sensitivity profile

Chạy smoke 1.000 ảnh, sau đó profile 5.000 ảnh. Profiler ghi partial JSON sau
từng candidate và có thể resume nếu Colab bị ngắt:

```bash
python search/profile_layer_sensitivity.py \
  --model resnet18_sparse \
  --checkpoint "$MAC18_CHECKPOINT" \
  --base-scheme-file "$MAC18_SCHEME" \
  --m 4 --candidate-n 2 3 4 \
  --layout NHWC --device cuda \
  --dataset-format parquet --parquet-root "$DATA_ROOT" \
  --dataset-num-samples 50000 \
  --accuracy-batch-size 64 --workers 2 \
  --max-eval-samples 5000 --seed 42 \
  --output "$PROFILE_ROOT/resnet18-m4-conditioned-mac18-5k.json"
```

Nếu runtime mất, chạy lại đúng lệnh và thêm `--resume`. Partial profile không
được đưa vào selector hoặc bảng kết quả.

### 5.3. Sinh candidates

Sinh MAC20, MAC21 và MAC23 từ cùng conditioned profile. Selector phải:

- chỉ giữ nguyên hoặc tăng sparsity từ MAC18;
- bảo vệ first convolution và classifier;
- fail nếu thiếu layer/candidate;
- tối thiểu hóa measured Top-1 sensitivity;
- ghi scheme, profile/checkpoint hash, target và achieved reduction.

### 5.4. Accuracy gates

| Gate | Dataset | Mục đích |
| --- | ---: | --- |
| A | 1.000 ảnh | kiểm tra pipeline, checkpoint và scheme |
| B | 5.000 ảnh | screening candidates cùng prefix |
| C | 50.000 ảnh | quyết định có fine-tune hay không |

MAC20 chỉ qua Gate C khi:

```text
Top-1 trước fine-tune >= 68,75%
Effective MAC reduction >= 20%
```

### 5.5. Fine-tune và lựa chọn

```text
epochs:                  3
learning rate:           0,001
training samples/epoch:  50.000
internal validation:     1.000
seed:                    42
save:                    every epoch
```

Screen mọi epoch trên cùng 5.000 ảnh, chọn checkpoint theo Top-1 đã định trước,
sau đó benchmark đủ 50.000 ảnh. Mục tiêu MAC20 sau fine-tune:

```text
Top-1 >= 69,45%
Effective MAC reduction >= 20%
```

Nếu không đạt, giữ MAC18 và ghi MAC20 là kết quả âm. Không tăng budget hoặc đổi
learning rate sau khi nhìn kết quả mà không tạo run riêng.

## 6. Giai đoạn B — Chuẩn hóa Jetson Nano

### 6.1. Ghi môi trường

- model/module Jetson chính xác;
- JetPack/L4T, CUDA, cuDNN, TensorRT và compiler;
- power supply, cooling và storage;
- power mode;
- CPU/GPU/EMC clocks;
- ambient/start/end temperature;
- process nền và swap;
- git commit, compiler flags, checkpoint/scheme SHA-256.

Chạy `nvpmodel` và `jetson_clocks` theo mode đã chọn. Nếu thay mode hoặc runtime,
toàn bộ dense/sparse latency phải đo lại.

### 6.2. Protocol latency

Cho mỗi backend/model:

1. khởi tạo process mới;
2. load model và pack weight trước vùng đo;
3. warm-up ít nhất 100 inference;
4. đo 1.000 inference, batch 1;
5. chạy ít nhất 10 repeat;
6. xen kẽ thứ tự dense/sparse;
7. lặp toàn protocol trong ít nhất ba session;
8. lưu raw samples, không chỉ summary.

Báo cáo median, P95, mean, standard deviation và bootstrap CI95%. Không kết luận
speedup nếu CI95% của chênh lệch cắt qua 0 hoặc chênh lệch dưới ngưỡng thực tế
đã định trước.

### 6.3. Protocol memory

Ghi tại ba thời điểm: idle, sau load và peak inference. Đối chiếu:

- RAM/largest free block từ `tegrastats`;
- process RSS/HWM từ `/proc/<pid>/status`;
- CUDA/TensorRT allocation;
- Nsight memory trace khi hỗ trợ.

Vì Jetson dùng unified memory, không dùng riêng
`torch.cuda.max_memory_allocated()` làm peak system memory.

### 6.4. Protocol power/energy

Ưu tiên power meter ngoài hoặc shunt sensor đã hiệu chuẩn. Chạy inference liên
tục trong cửa sổ 30–60 giây:

```text
active_energy = integral(power_active - power_idle) dt
energy_per_inference = active_energy / inference_count
```

`tegrastats` dùng làm telemetry đối chiếu clock/nhiệt/power rail, không phải bằng
chứng năng lượng duy nhất cho inference ngắn.

## 7. Giai đoạn C — Kernel feasibility

Jetson Nano Maxwell không có native Ampere sparse Tensor Core. Vì vậy không giả
định `2:4` nhanh hơn dense. Tiến hành theo cổng:

1. Nsight masked-dense để xác nhận bottleneck;
2. chọn 3–5 convolution 3×3 chiếm latency lớn nhất;
3. pack `2:4` values + indices offline;
4. triển khai hoặc port một CUDA SpMM/convolution kernel tối thiểu;
5. microbenchmark với đúng shape ResNet-18;
6. so kernel với cuDNN dense cùng input/output/precision.

Candidate chỉ được đánh dấu `eligible=true` khi:

```text
speedup median >= 5%
CI95% của speedup không cắt qua 0
output numerical check đạt tolerance
packing không nằm trong inference critical path
```

Nếu không đạt, không mở rộng kernel và không cho hardware selector dùng candidate
đó. Kết quả “dense nhanh hơn” là kết luận hợp lệ.

## 8. Giai đoạn D — Hardware lookup thật

Mỗi row phải chứa:

```json
{
  "layer": "SparseConv5_64-128-(3,3)",
  "n": 2,
  "m": 4,
  "backend": "packed_cuda",
  "eligible": true,
  "latency": {
    "median_ms": 0.39,
    "p95_ms": 0.43,
    "ci95_low_ms": 0.38,
    "ci95_high_ms": 0.41,
    "samples": 10000
  },
  "energy_mj": 1.27,
  "dram_read_bytes": 524288,
  "dram_write_bytes": 262144,
  "peak_memory_bytes": 3145728,
  "temperature_c": 48.5,
  "gpu_clock_mhz": 921,
  "emc_clock_mhz": 1600
}
```

Các con số trên chỉ minh họa schema, không phải kết quả Jetson. Metric chưa đo
thật phải là `null` và có weight bằng 0.

## 9. Hàm cost và selector

### 9.1. Cost theo constraint

Chuẩn hóa mỗi metric theo dense của cùng layer. Dùng upper CI để tránh chọn
candidate có median tốt do nhiễu:

```text
C_i(N:M) = alpha * latency_ci95_high_ratio
         + beta  * energy_ci95_high_ratio
         + gamma * measured_dram_bytes_ratio
         + delta * measured_peak_memory_ratio
```

Các weight không được chọn sau khi nhìn kết quả. Chúng phải xuất phát từ yêu cầu
deployment, ví dụ latency deadline, memory ceiling hoặc energy budget. Báo cáo
latency-only trước; composite objective là run riêng.

### 9.2. Bài toán chọn scheme

```text
minimize    estimated accuracy sensitivity

subject to  measured hardware cost <= budget
            memory <= board budget
            candidate eligible = true
            scheme monotonic from accepted stage
```

Lookup đầy đủ được ưu tiên hơn predictor. Predictor chỉ được dùng nếu validation
error đạt ngưỡng định trước và phải báo MAE/MAPE/R².

## 10. Giai đoạn E — Xác nhận end-to-end

Sau khi selector tạo scheme hardware-aware:

1. benchmark accuracy trước fine-tune trên 50.000 ảnh;
2. fine-tune trên Colab với budget công bằng;
3. benchmark accuracy sau fine-tune;
4. deploy checkpoint cuối lên Jetson;
5. chạy lại protocol end-to-end không profiler;
6. chạy một session Nsight riêng để giải thích kết quả;
7. so với dense và MAC-based candidate.

Không dùng tổng lookup latency làm kết quả cuối vì kernel launch, cache, fusion,
copy và scheduling làm tổng layer latency khác end-to-end latency.

## 11. Tiêu chí nghiệm thu

### 11.1. Accuracy-complexity

- checkpoint load exact;
- full ImageNet validation 50.000 ảnh;
- Top-1/Top-5 và before/after fine-tune đầy đủ;
- scheme đúng 21 sparse layer;
- measured sparsity khớp scheme;
- đạt gate accuracy đã công bố.

### 11.2. Deployment

Chỉ gọi là nhanh hơn trên Jetson nếu:

- C++ end-to-end median/P95 giảm;
- CUDA/Nsight kernel time giải thích được reduction;
- CI95% chứng minh chênh lệch ngoài nhiễu;
- ít nhất ba session cùng chiều;
- không có thermal/power-mode advantage;
- numerical output đúng;
- packing/setup cost được báo riêng;
- peak memory và energy được đo bằng nguồn phù hợp.

Nếu chỉ giảm MAC/parameter, kết luận là theoretical reduction. Nếu chỉ T4 nhanh
hơn, kết luận chỉ áp dụng T4/runtime đã đo.

## 12. Artifact và khả năng truy vết

Mỗi run phải lưu:

- raw latency/power/temperature samples;
- Nsight report và exported statistics;
- `tegrastats` log;
- environment manifest;
- kernel/backend build commit và compiler flags;
- layer lookup JSON;
- checkpoint/scheme SHA-256;
- accuracy benchmark JSON;
- compare-results Markdown/CSV;
- branch, commit, dirty state và seed.

Không commit checkpoint, dataset, Nsight binary report hoặc generated benchmark
artifact vào Git. Lưu chúng trên Drive/experiment storage và chỉ commit protocol,
schema, code cùng báo cáo tóm tắt.

## 13. Rủi ro và phương án dừng

| Rủi ro | Phát hiện | Quyết định |
| --- | --- | --- |
| MAC20 mất accuracy | full-val gate | giữ MAC18 |
| Colab bị ngắt | partial profile | resume đúng identity |
| masked sparse chậm | Nsight/kernel time | không gọi là speedup |
| custom kernel chậm | feasibility CI95% | dừng kernel/cấm candidate |
| power telemetry quá thưa | đối chiếu meter ngoài | không báo energy/inference từ telemetry |
| predictor sai số cao | held-out MAPE/R² | dùng lookup |
| lookup không dự đoán end-to-end | full-model benchmark | chọn theo end-to-end result |
| Nano không khai thác N:M | mọi candidate fail gate | báo kết quả âm, dense là runtime Pareto |

## 14. Trình tự thực hiện

### Ngay hiện tại, chưa có board

1. hoàn thiện profiler resume và test;
2. chuẩn bị schema raw measurement/CI;
3. khi Colab sẵn sàng, chạy MAC18-conditioned profile;
4. tạo và fine-tune MAC20 candidates;
5. đóng băng 2–3 checkpoint Pareto để deploy.

### Khi có Jetson Nano

1. chuẩn hóa board và dense C++ baseline;
2. đo masked-dense để phân rã overhead;
3. chạy kernel feasibility trên layer ưu tiên;
4. tạo measured lookup chỉ từ candidate hợp lệ;
5. chạy hardware-aware selector;
6. fine-tune scheme được chọn trên Colab;
7. benchmark end-to-end cuối trên Jetson;
8. lập bảng accuracy–latency–memory–energy và kết luận.

## 15. Đóng góp dự kiến

Đóng góp không phải là tuyên bố “sparsity làm Jetson nhanh hơn” từ FLOPs. Đóng
góp có thể kiểm chứng là:

> Một pipeline DominoSearch mixed N:M dùng conditioned accuracy sensitivity,
> measured Jetson lookup, uncertainty-aware cost và runtime/kernel eligibility
> để chỉ chọn các pattern có bằng chứng deployment thật.

Nếu không có N:M kernel nào vượt dense trên Jetson Nano, báo cáo vẫn trả lời được
câu hỏi nghiên cứu và chỉ ra giới hạn hardware–software của Maxwell một cách có
số liệu, thay vì đưa ra speedup không được chứng minh.

## 16. Nguồn nghiên cứu

- DominoSearch, `assets/DominoSearch.pdf`;
- NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications,
  ECCV 2018;
- HALP: Hardware-Aware Latency Pruning, 2021/2022;
- Efficient GPU Kernels for N:M-Sparse Weights in Deep Learning (nmSPARSE),
  MLSys 2023;
- Accelerating Sparse DNN Models without Hardware-Support via Tile-Wise
  Sparsity, SC 2020;
- NVIDIA CUDA Events, Nsight Systems và Jetson `tegrastats` documentation.
