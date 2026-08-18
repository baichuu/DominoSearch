# Báo cáo ngắn: tối ưu mixed N:M và kiểm chứng trên Jetson Nano

**Ngày báo cáo:** 17/08/2026  
**Phạm vi:** DominoSearch layer-wise mixed N:M, ResNet-18/ImageNet  
**Mục tiêu:** chọn mô hình cân bằng accuracy–complexity trên Colab, sau đó xác
minh bằng số đo thật rằng mô hình có lợi trên Jetson Nano.

## 0. Công việc đang thực hiện

Model đang nghiên cứu là **ResNet-18 phân loại ImageNet**: input RGB 224×224 và
output 1.000 lớp. Phương pháp duy nhất đang phát triển là **layer-wise mixed
N:M**: mỗi convolution được gán một cấu hình N:M riêng dựa trên độ nhạy accuracy,
thay vì dùng cùng một mức sparsity cho toàn mạng.

Hiện pipeline Colab đã hoàn thành các bước search, fine-tune và full validation
cho MAC20/MAC23. Công việc kế tiếp không phải tiếp tục giảm MAC bằng mọi giá mà
là triển khai representation/kernel có thể khai thác N:M, đo cost thật trên
Jetson Nano, rồi đưa cost đó trở lại selector. Quantization, distillation và các
loại pruning khác không nằm trong thí nghiệm này.

## 1. Kết quả hiện tại

| Mô hình         |       Top-1 |       Top-5 | Giảm MAC hiệu dụng | Giảm tham số hiệu dụng |
| --------------- | ----------: | ----------: | -----------------: | ---------------------: |
| Dense           |     69,754% |     89,080% |                 0% |                     0% |
| Mixed N:M MAC20 | **69,660%** | **89,130%** |        **20,003%** |                20,332% |
| Mixed N:M MAC23 |     69,458% |     88,994% |        **23,190%** |            **24,747%** |

- MAC20 chỉ giảm 0,094 điểm phần trăm Top-1 so với dense, phù hợp khi ưu tiên
  accuracy.
- MAC23 giảm 0,296 điểm Top-1 nhưng giảm thêm khoảng 3,19 điểm phần trăm MAC so
  với MAC20, phù hợp khi ưu tiên complexity.
- Hai mô hình là hai điểm Pareto khác nhau; chưa có cơ sở chọn một mô hình duy
  nhất trước khi đo trên board.
- Latency masked-dense trên T4 là 19,948 ms với MAC20 và 21,043 ms với MAC23.
  Kết quả này **không chứng minh sparse speedup** vì các lớp hiện chỉ tạo mask rồi
  vẫn gọi phép toán dense của PyTorch.

Chi tiết và artifact được ghi trong
[DOMINO_STAGED_MIXED_NM.md](DOMINO_STAGED_MIXED_NM.md).

## 2. Vấn đề cần giải quyết

Giảm tham số hay MAC chỉ là lợi ích lý thuyết. Latency thực còn phụ thuộc vào
layout dữ liệu, chi phí giải mã metadata, kernel launch, memory traffic, khả năng
song song, thư viện và các phép toán không được prune. Hơn nữa, structured
sparsity 2:4 được TensorRT hỗ trợ bằng phần cứng trên GPU kiến trúc Ampere; Jetson
Nano dùng GPU Maxwell 128 CUDA cores
[NVIDIA Jetson Nano specifications](https://developer.nvidia.com/embedded/jetson-nano),
nên không thể giả định có cùng sparse acceleration
[NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/io-formats-sparsity.html).

Vì vậy câu hỏi nghiên cứu trên board là:

> Với cùng input, precision và điều kiện vận hành, representation/kernel packed
> mixed N:M có giảm latency, memory và năng lượng so với dense hay không, trong
> khi vẫn tái tạo đúng đầu ra và accuracy của checkpoint đã chọn?

Không cần sửa Linux kernel để trả lời câu hỏi này. Kernel cần nghiên cứu ở đây
là **CUDA inference kernel**. Can thiệp kernel hệ điều hành chỉ hợp lý nếu công cụ
chuẩn không truy cập được counter cần thiết, và không phải bước mặc định.

## 3. Giải pháp thực hiện khi có board

```text
checkpoint + scheme
        |
        v
kiểm tra mask/layout --> pack offline --> kiểm tra số học với PyTorch
        |                                      |
        v                                      v
map layer -> operator -> CUDA kernel --> đo từng kernel/layer
        |                                      |
        +------------ cost table --------------+
                           |
                           v
             search/chọn MAC20 hoặc MAC23
                           |
                           v
              đo end-to-end và accuracy
```

### Bước 1 — Khóa baseline và môi trường

- Dùng cùng ResNet-18, checkpoint, ImageNet validation, preprocessing, input
  224×224, batch size và precision cho dense/MAC20/MAC23.
- Ghi JetPack/L4T, CUDA, cuDNN, TensorRT, compiler flags, commit, power mode và
  nhiệt độ môi trường.
- Chọn trước một power mode; kiểm tra bằng `nvpmodel -q`, cố định xung bằng
  `jetson_clocks`, và ghi xung/nhiệt/RAM bằng `tegrastats`. NVIDIA xác nhận các
  công cụ này phản ánh power mode, CPU/GPU/EMC frequency, nhiệt độ và RAM trên
  Jetson Nano
  [Jetson Nano power management](https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3276/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_nano.html).

### Bước 2 — Mapping chính xác từ model xuống hardware

Mỗi record trong cost table phải có khóa đầy đủ; không chỉ dùng tên layer:

| Nhóm    | Trường bắt buộc                                                |
| ------- | -------------------------------------------------------------- |
| Model   | layer, input/output/weight shape, stride, padding, groups      |
| Sparse  | N:M, trục grouping, mask hash, packed layout, metadata size    |
| Runtime | precision, batch, backend, algorithm/tactic, kernel name       |
| Board   | power mode, CPU/GPU/EMC clock, temperature, software versions  |
| Kết quả | median, P95, CI95, throughput, peak RAM, traffic, power/energy |

Mapping được xác nhận bằng profiler: PyTorch layer/ONNX node → runtime operator
→ kernel xuất hiện trên GPU timeline. Nếu fusion làm nhiều layer thành một
kernel, cost phải gắn với **fused group**, không được chia latency giả tạo cho
từng layer.

### Bước 3 — Ba baseline tách biệt

1. **Dense runtime:** cuDNN/TensorRT hoặc C++–CUDA dense tốt nhất trên cùng board.
2. **Masked-dense:** kiểm tra checkpoint/mask nhưng vẫn chạy dense; chỉ dùng làm
   đối chứng, không gọi là sparse acceleration.
3. **Packed mixed N:M:** pack weight offline thành values + metadata và chạy CUDA
   kernel chỉ đọc/tính các phần tử được giữ lại.

Đầu ra packed phải so với masked PyTorch trên nhiều input. Chỉ benchmark candidate
đạt ngưỡng sai số đã định cho precision tương ứng; sau đó chạy lại full validation
để xác nhận Top-1/Top-5.

### Bước 4 — Giao thức đo có thể lặp lại

| Đại lượng                         | Cách đo                                                                |
| --------------------------------- | ---------------------------------------------------------------------- |
| GPU kernel/layer latency          | CUDA Events đặt trong đúng stream, đồng bộ trước khi đọc               |
| End-to-end latency                | C++ `std::chrono::steady_clock`, bao quanh đúng phạm vi đã công bố     |
| Kernel, occupancy, memory traffic | Nsight Systems/Compute; profiler chạy riêng benchmark timing           |
| RAM/clock/temperature             | `tegrastats` và số liệu runtime; lấy baseline trước/sau cấp phát       |
| Công suất/năng lượng              | ưu tiên power meter ngoài tại nguồn vào; `tegrastats` là telemetry phụ |

CUDA launch là bất đồng bộ, do đó chỉ đo thời gian host quanh lời gọi kernel sẽ
sai. NVIDIA khuyến nghị đặt hai CUDA Events quanh công việc trong stream rồi đồng
bộ trước khi lấy elapsed time
[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).

Protocol đề xuất: warm-up tối thiểu 100 lần; 1.000 inference mỗi repeat; 10 repeat
cho mỗi candidate; ít nhất 3 phiên độc lập sau khi ổn định nhiệt; đảo/interleave
thứ tự dense và sparse để giảm bias do nhiệt/xung. Lưu toàn bộ mẫu thô, báo cáo
median, P95 và bootstrap CI95; batch 1 là kết quả chính, throughput batch lớn là
kết quả phụ. Không chạy profiler đồng thời với phép đo latency chính vì profiler
có overhead.

### Bước 5 — Điều kiện kết luận

Chỉ kết luận tối ưu thành công trên Jetson Nano khi đồng thời đạt:

- checkpoint và mask hợp lệ; packed output đạt kiểm tra số học;
- Top-1/Top-5 được giữ trong ngưỡng đã công bố;
- median và P95 end-to-end tốt hơn dense, CI95 không cho thấy kết quả do nhiễu;
- peak memory hoặc model binary giảm nếu đó là claim;
- không có thermal/power throttling và mọi điều kiện đo giống nhau;
- lợi ích vẫn tồn tại ở toàn model, không chỉ ở microbenchmark của một layer.

Nếu sparse kernel không thắng dense, kết quả vẫn có giá trị: cost table sẽ chỉ ra
layer/shape nào bị metadata, memory traffic hoặc launch overhead lấn át, để search
không chọn pattern đó. Đây là tư tưởng platform-aware đã được dùng trong
[NetAdapt](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Tien-Ju_Yang_NetAdapt_Platform-Aware_Neural_ECCV_2018_paper.php)
và latency lookup theo kernel/operator được nghiên cứu trong
[nn-Meter](https://www.microsoft.com/en-us/research/publication/nn-meter-towards-accurate-latency-prediction-of-deep-learning-model-inference-on-diverse-edge-devices/).

## 4. Kế hoạch trước và sau khi có board

**Hiện tại trên Colab:** giữ MAC20 và MAC23 làm hai candidate; hoàn tất artifact,
manifest, kiểm tra scheme/checkpoint và export test vectors. Colab chỉ dùng để xác
nhận accuracy và complexity, không dùng để dự đoán Jetson latency.

**Khi có Jetson Nano:** dựng dense C++ baseline → kiểm tra packed representation →
profile/mapping → thu cost table → tối ưu CUDA kernel nếu cần → đo end-to-end →
chọn MAC20 hoặc MAC23 theo Pareto accuracy–latency–memory–energy.

### Pipeline tổng thể và artifact của từng giai đoạn

| Giai đoạn          | Nơi chạy   | Input                        | Xử lý                                  | Output bắt buộc              |
| ------------------ | ---------- | ---------------------------- | -------------------------------------- | ---------------------------- |
| 1. Baseline        | Colab      | dense checkpoint             | full validation và complexity          | dense JSON                   |
| 2. Sensitivity     | Colab      | checkpoint + base scheme     | thử N:M theo từng layer                | conditioned profile JSON     |
| 3. Selection       | Colab      | profile + MAC budget         | chọn scheme ít mất accuracy            | scheme + manifest/hash       |
| 4. Fine-tune       | Colab      | checkpoint + scheme          | giữ mask cố định, phục hồi accuracy    | checkpoint mỗi epoch         |
| 5. Validation      | Colab      | candidate checkpoint         | đủ 50.000 ảnh                          | Top-1/Top-5 + MAC/param JSON |
| 6. Export          | Máy host   | checkpoint + scheme          | pack values/metadata, tạo test vectors | packed weights + manifest    |
| 7. Microbenchmark  | Jetson     | layer shape + packed weights | dense và sparse CUDA kernel            | raw layer/kernel cost JSON   |
| 8. Hardware search | Host/Colab | sensitivity + Jetson lookup  | chỉ chọn candidate eligible            | hardware-aware scheme        |
| 9. Deployment      | Jetson     | model cuối                   | C++ end-to-end benchmark               | raw samples + summary + logs |

### Trình tự đo cụ thể trên Jetson Nano

1. **Chuẩn hóa board:** reboot, tắt workload nền không cần thiết, chọn power mode,
   chạy `jetson_clocks`, ghi môi trường và đợi nhiệt độ ổn định. Không đổi setting
   giữa dense và sparse.
2. **Khởi tạo ngoài vùng đo:** load model, cấp phát buffer, chọn tactic và pack
   weight trước khi timing. Thời gian load/pack được đo riêng, không trộn vào
   steady-state inference.
3. **Kiểm tra đúng:** cùng test vector phải cho output dense/masked/packed nằm
   trong tolerance. Checkpoint phải load đủ key; scheme phải phủ đúng layer.
4. **Warm-up:** chạy tối thiểu 100 inference để ổn định cache, tactic và clock.
5. **Đo GPU:** đặt CUDA Events trong cùng stream quanh kernel/operator; đồng bộ
   event kết thúc rồi lưu từng sample, không chỉ lưu trung bình.
6. **Đo end-to-end:** dùng `steady_clock` quanh phạm vi input tensor đã sẵn sàng
   đến output đã sẵn sàng; đồng bộ CUDA trước điểm kết thúc.
7. **Lặp công bằng:** 1.000 inference/repeat, 10 repeat và ít nhất 3 session;
   interleave thứ tự dense–sparse để giảm bias nhiệt độ và clock.
8. **Telemetry riêng:** chạy session có `tegrastats`/Nsight để lấy RAM, clock,
   nhiệt, utilization và memory traffic. Không dùng session profiler làm latency
   chính vì profiler tạo overhead.
9. **Power:** đo idle trước, sau đó chạy inference liên tục 30–60 giây bằng power
   meter ngoài; trừ idle power và chia active energy cho số inference.
10. **Tổng hợp:** báo median, P95, mean, standard deviation và bootstrap CI95;
    so với dense bằng cùng raw protocol.

Phạm vi end-to-end phải ghi rõ. Kết quả deployment chính nên đo:

```text
input tensor đã ở RAM/GPU
    -> preprocessing/runtime enqueue
    -> toàn bộ CUDA kernels
    -> synchronization
    -> output tensor sẵn sàng
```

Nếu đo thêm camera-to-decision hoặc host-to-device copy thì báo thành metric
riêng, không trộn với model-only latency.

### Cách tính các chỉ số chính

```text
speedup                   = dense_latency / sparse_latency
latency_reduction         = 1 - sparse_latency / dense_latency
throughput                = số inference / tổng thời gian
active_energy             = integral(power_active - power_idle) dt
energy_per_inference      = active_energy / số inference
memory_reduction          = 1 - sparse_peak_memory / dense_peak_memory
```

`tegrastats` phù hợp để kiểm tra RAM, clock, nhiệt độ và throttling. Với năng
lượng/inference ngắn, power meter ngoài là bằng chứng chính vì sampling telemetry
có thể quá thưa.

Đầu ra cuối cùng gồm raw samples, environment manifest, layer/kernel mapping,
cost table, correctness report và bảng dense–MAC20–MAC23. Quy trình chi tiết nằm
trong
[JETSON_NANO_HARDWARE_MEASURED_MIXED_NM_PLAN.md](JETSON_NANO_HARDWARE_MEASURED_MIXED_NM_PLAN.md).

## 5. Nội dung trình bày ngắn

> Hiện tại đã tạo được hai mô hình mixed N:M hợp lệ. MAC20 gần như giữ nguyên
> accuracy, chỉ giảm 0,094 điểm Top-1 và giảm 20,003% MAC; MAC23 giảm 0,296 điểm
> Top-1 nhưng giảm 23,190% MAC. Đây mới là tối ưu accuracy–complexity, chưa phải
> bằng chứng tăng tốc vì PyTorch vẫn gọi dense operator. Khi có Jetson Nano,
> trọng số sẽ được đóng gói, đầu ra được kiểm tra với PyTorch và từng layer được
> mapping qua operator tới CUDA kernel. Kernel được đo bằng CUDA Events và
> end-to-end được đo bằng C++ trong điều kiện power, clock và nhiệt được cố định.
> Ba trường hợp dense, masked-dense và
> packed sparse sẽ được so sánh bằng raw samples, median, P95 và CI95. Chỉ khi
> full-model latency thật sự giảm mà accuracy vẫn đạt ngưỡng thì mới kết luận mô
> hình được tối ưu trên board.

## Tài liệu tham khảo chính

1. [DominoSearch — NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/ad68473a64305626a27c32a5408552d7-Abstract.html).
2. [NVIDIA CUDA Programming Guide — Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).
3. [NVIDIA Jetson Nano — Clock Frequency and Power Management](https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3276/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_nano.html).
4. [NVIDIA Jetson Nano — Technical Specifications](https://developer.nvidia.com/embedded/jetson-nano).
5. [NVIDIA TensorRT — Structured Sparsity](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/io-formats-sparsity.html).
6. [NetAdapt — ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Tien-Ju_Yang_NetAdapt_Platform-Aware_Neural_ECCV_2018_paper.php).
7. [nn-Meter — MobiSys 2021](https://www.microsoft.com/en-us/research/publication/nn-meter-towards-accurate-latency-prediction-of-deep-learning-model-inference-on-diverse-edge-devices/).
8. [nmSPARSE — MLSys 2023](https://proceedings.mlsys.org/paper_files/paper/2023/hash/a10deb4d5227a8ea307ea8ff3cb712f4-Abstract-mlsys2023.html).
