# Tổng quan nghiên cứu và giải pháp Hardware-Measured Mixed N:M cho Jetson Nano

## 1. Mục tiêu tài liệu

Tài liệu này tổng hợp các nghiên cứu liên quan đến:

- pruning có xét phần cứng mục tiêu;
- đo và dự đoán latency trên thiết bị edge;
- triển khai kernel cho N:M sparsity;
- đánh giá model sparse trên Jetson Nano;
- xây dựng một giải pháp thực tế cho DominoSearch mixed N:M.

Mục tiêu cuối cùng là trả lời câu hỏi:

> Làm thế nào tìm được cấu hình N:M riêng cho từng layer, giữ độ chính xác của
> ResNet-18, đồng thời chứng minh bằng số đo thực tế rằng model có lợi trên
> Jetson Nano?

Phạm vi active chỉ gồm DominoSearch layer-wise mixed N:M. Uniform N:M,
structured pruning và unstructured pruning chỉ được giữ làm kết quả đối chứng,
không được trộn mask hoặc checkpoint vào thí nghiệm này.

## 2. Vấn đề nghiên cứu

DominoSearch tìm cấu hình N:M riêng cho từng layer dưới một complexity budget.
Trong paper gốc, complexity chủ yếu là model size hoặc FLOPs. Tuy nhiên:

- effective MAC thấp không bảo đảm latency thấp;
- checkpoint chứa nhiều số 0 không tự động nhỏ hơn;
- PyTorch hiện tạo mask rồi vẫn gọi dense operator;
- tổng latency từng layer không nhất thiết bằng latency end-to-end;
- Jetson Nano Maxwell không có native Ampere Sparse Tensor Core;
- speedup phụ thuộc kernel, runtime, memory layout và shape của từng operator.

Do đó cần thay đổi từ:

```text
accuracy sensitivity + FLOPs/MAC proxy
```

sang:

```text
accuracy sensitivity + hardware cost thực đo + kernel eligibility
```

## 3. Nghiên cứu nền tảng

### 3.1. DominoSearch: layer-wise mixed N:M

DominoSearch tìm N:M riêng cho từng layer từ một candidate pool và cho phép
complexity là model size, FLOPs, latency hoặc energy. Paper thực nghiệm chủ yếu
với model size/FLOPs và thừa nhận chưa benchmark deployment latency/throughput.

Phần kế thừa:

- layer-wise mixed N:M;
- discrete N:M candidate pool;
- bắt đầu từ pretrained dense weights;
- tách bước search scheme và bước fine-tune phục hồi accuracy;
- giữ behavior gốc làm baseline.

Phần cần mở rộng:

- measured hardware cost trên Jetson;
- direct layer sensitivity tại trạng thái đã prune;
- kernel/runtime eligibility;
- uncertainty và end-to-end validation.

Nguồn: [DominoSearch: Find layer-wise fine-grained N:M sparse schemes from dense
networks, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/ad68473a64305626a27c32a5408552d7-Abstract.html).

### 3.2. NetAdapt: đo trực tiếp trên target platform

NetAdapt chỉ ra indirect metrics như MAC hoặc parameter không phản ánh đầy đủ
latency/energy. Phương pháp đơn giản hóa model từng bước, đo direct resource
metric trên thiết bị mục tiêu và tiếp tục cho đến khi đạt budget.

Phần áp dụng:

```text
accepted mixed N:M stage
→ sinh stage mới
→ đo trực tiếp
→ chỉ nhận stage qua accuracy/resource gate
→ lặp lại
```

Điểm quan trọng là không cần hiểu hoàn toàn microarchitecture nếu có thể đo trực
tiếp trên platform. Tuy nhiên, phép đo vẫn phải dùng deployment stack cố định.

Nguồn: [NetAdapt: Platform-Aware Neural Network Adaptation for Mobile
Applications, ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Tien-Ju_Yang_NetAdapt_Platform-Aware_Neural_ECCV_2018_paper.php).

### 3.3. HALP: latency lookup kết hợp saliency

HALP xây latency lookup table và kết hợp latency reduction potential với
importance/saliency. Việc phân bổ pruning toàn cục được giải như một bài toán
resource allocation/knapsack dưới latency constraint.

Phần áp dụng trực tiếp:

```text
Jetson layer/kernel lookup
+ conditioned Top-1 sensitivity
→ Pareto multiple-choice selector
```

Điểm khác biệt là HALP dùng structured pruning, còn dự án này giữ mixed N:M.
Chỉ kế thừa cách kết hợp measured lookup và sensitivity, không sử dụng filter
pruning của HALP.

Nguồn: [HALP: Hardware-Aware Latency Pruning](https://arxiv.org/abs/2110.10811).

## 4. Đo và dự đoán latency

### 4.1. nn-Meter: kernel-aware latency prediction

nn-Meter chỉ ra operator graph không luôn trùng với execution kernel vì runtime
có thể fusion hoặc biến đổi operator. Phương pháp phát hiện kernel thực thi và
profile ở kernel level, sau đó dùng adaptive sampling để giảm số cấu hình cần
đo.

Phần áp dụng:

- dùng Nsight xác định execution units thật;
- không mặc định `model latency = tổng Python layer latency`;
- lookup key nên gồm `kernel group + shape + N:M + backend`;
- đo lại end-to-end scheme cuối;
- chỉ dùng adaptive sampling khi search space lớn.

Nguồn: [nn-Meter: Towards Accurate Latency Prediction of Deep-Learning Model
Inference on Diverse Edge Devices, MobiSys 2021](https://www.microsoft.com/en-us/research/publication/nn-meter-towards-accurate-latency-prediction-of-deep-learning-model-inference-on-diverse-edge-devices/).

### 4.2. CloserToMe: transferable latency prediction

CloserToMe xây device behavior signature từ một tập workload đại diện, kết hợp
device capability vector và Hardware–Operation Dialogue Module để mô hình hóa
tương tác giữa operator và phần cứng.

Phần có thể áp dụng về sau:

- chọn representative probes;
- bổ sung memory hierarchy/compute capability vào device representation;
- transfer predictor từ một tập thiết bị sang board mới;
- giảm số phép đo khi hỗ trợ nhiều board.

Với một Jetson Nano và ResNet-18 chỉ khoảng `21 layer × 3 N:M = 63` điểm, full
lookup nên được ưu tiên hơn predictor. Full lookup sẽ là ground truth để đánh giá
predictor, không phải ngược lại.

Nguồn: [CloserToMe: A Unified Framework for Accurate and Transferable Latency
Prediction Across Heterogeneous Devices, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39779).

### 4.3. On Latency Predictors for Neural Architecture Search

Nghiên cứu MLSys 2024 phân tích predictor architecture, device representation,
operator encoding và sample selection trên nhiều bài toán chuyển giao thiết bị.

Phần áp dụng:

- chia train/test theo layer hoặc device, không chia ngẫu nhiên row cùng layer;
- báo MAE, MAPE, R² và ranking accuracy;
- so predictor với lookup;
- từ chối predictor khi sai số hoặc ranking không đạt ngưỡng định trước.

Nguồn: [On Latency Predictors for Neural Architecture Search, MLSys
2024](https://proceedings.mlsys.org/paper_files/paper/2024/hash/f03cb785864596fa5901f1359d23fd81-Abstract-Conference.html).

### 4.4. MAPLE: hardware descriptor cho unseen device

MAPLE xây hardware descriptor từ đặc tính và performance metrics của
microprocessor để dự đoán latency trên thiết bị chưa thấy.

Phần áp dụng chỉ cần thiết khi mở rộng từ Nano sang Xavier, Orin hoặc FPGA. Không
cần đưa vào vòng Jetson Nano đầu tiên vì measured lookup đầy đủ đáng tin cậy hơn.

Nguồn: [MAPLE: Microprocessor a Priori for Latency Estimation, CVPRW
2022](https://openaccess.thecvf.com/content/CVPR2022W/ECV/html/Abbasi_MAPLE_Microprocessor_a_Priori_for_Latency_Estimation_CVPRW_2022_paper.html).

## 5. Sparse kernel và algorithm–software co-design

### 5.1. nmSPARSE: dedicated N:M GPU kernels

nmSPARSE chỉ ra N:M sparsity không tạo real-world benefit nếu thiếu dedicated
kernel. Hệ thống pack sparse values/metadata và tổ chức computation để giảm
scattered memory access và load imbalance cho SpMV/SpMM.

Phần áp dụng:

- pack weight và indices offline;
- không tạo magnitude mask trong inference;
- benchmark theo matrix/layer shape;
- tổ chức workload đều giữa thread/warp;
- ưu tiên coalesced memory access;
- so với dense library mạnh nhất trên cùng thiết bị.

Paper đánh giá chủ yếu trên A100 nên không được dùng speedup của paper làm bằng
chứng cho Jetson Nano Maxwell.

Nguồn: [Efficient GPU Kernels for N:M-Sparse Weights in Deep Learning,
MLSys 2023](https://proceedings.mlsys.org/paper_files/paper/2023/hash/a10deb4d5227a8ea307ea8ff3cb712f4-Abstract-mlsys2023.html).

### 5.2. SparTA: specialized operator theo sparsity attribute

SparTA gắn sparsity pattern vào tensor abstraction, truyền attribute qua graph
và sinh specialized operator phù hợp pattern/phần cứng.

Phần áp dụng quan trọng nhất là candidate eligibility:

```text
candidate có kernel
AND numerical check đạt
AND kernel nhanh hơn dense ngoài CI95%
→ eligible=true

ngược lại
→ eligible=false
```

Selector không được chọn candidate `eligible=false` chỉ vì nó giảm MAC.

Nguồn: [SparTA: Deep-Learning Model Sparsity via Tensor-with-Sparsity-Attribute,
OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/zheng-ningxin).

### 5.3. Tile-Wise Sparsity

Tile-wise sparsity cho thấy algorithm và GPU execution cần được đồng thiết kế.
Paper pack weight offline, thay đổi layout, giảm uncoalesced access và xử lý load
imbalance/concurrency.

Không áp dụng tile-wise pruning vì nó thay đổi pruning pattern active. Chỉ kế
thừa nguyên tắc kernel/layout và cách tách kernel benchmark khỏi end-to-end
benchmark.

Nguồn: [Accelerating Sparse DNN Models without Hardware-Support via Tile-Wise
Sparsity](https://arxiv.org/abs/2008.13006).

### 5.4. Column-vector-wise sparse convolution

Nghiên cứu này triển khai sparse convolution GPU và đánh giá end-to-end trên
CNN/ResNet. Pattern không phải N:M nên không đưa vào pruning experiment, nhưng
có thể tham khảo direct sparse convolution, mapping workload và memory layout.

Nguồn: [Accelerating Sparse Convolution with Column-Vector-Wise Sparsity,
NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/c383e44d9a878d1982d9abb838bd5d8a-Paper-Conference.pdf).

### 5.5. Sparsity-aware end-to-end framework

SparTA và các hệ thống tương tự cho thấy chỉ thay weight bằng zero là chưa đủ.
Representation, compiler/runtime, operator generation và end-to-end graph đều
cần biết sparsity. Đây là lý do báo cáo phải tách:

```text
pruning-only result
vs. packed representation result
vs. custom-kernel deployment result
```

## 6. Nghiên cứu trực tiếp trên Jetson Nano

### 6.1. LightPrune

LightPrune dùng progressive latency-aware structured pruning và báo cáo trên
Jetson Nano. Paper hỗ trợ các nguyên tắc:

- đo trên đúng target hardware;
- progressive pruning;
- kết hợp hardware feedback với training stability;
- báo đồng thời accuracy, model size và measured latency.

Không áp dụng structured filter pruning của LightPrune vì khác phạm vi. Dataset,
model và pruning method cũng khác nên không dùng số speedup của paper làm
baseline cho ResNet-18 mixed N:M.

Nguồn: [LightPrune: Latency-Aware Structured Pruning for Efficient Deep
Inference on Embedded Devices, ICCVW 2025](https://openaccess.thecvf.com/content/ICCV2025W/EVW/html/Belhadi_LightPrune_Latency-Aware_Structured_Pruning_for_Efficient_Deep_Inference_on_Embedded_ICCVW_2025_paper.html).

### 6.2. Behaviour study for edge platforms

Nghiên cứu MLSys 2021 cho thấy một model có ít operation hoặc memory-access proxy
hơn vẫn có thể chậm hơn do mismatch với platform/runtime. Kết quả củng cố yêu
cầu profile đúng operator shape, backend và thiết bị.

Nguồn: [To Bridge Neural Network Design and Real-World Performance: A Behaviour
Study for Neural Networks, MLSys 2021](https://proceedings.mlsys.org/paper_files/paper/2021/hash/411e39b117e885341f25efb8912945f7-Abstract.html).

### 6.3. HarDNet: memory traffic là metric quan trọng

HarDNet chỉ ra latency edge có thể bị chi phối bởi memory traffic thay vì MAC.
Paper sử dụng profiler để đối chiếu memory traffic với inference latency.

Phần áp dụng:

- chỉ đưa DRAM traffic thật từ hardware counter vào cost;
- không dùng tensor byte estimate rồi gọi là measured bandwidth;
- kiểm tra model đang compute-bound hay memory-bound.

Không áp dụng kiến trúc HarDNet vì architecture replacement ngoài phạm vi.

Nguồn: [HarDNet: A Low Memory Traffic Network, ICCV
2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Chao_HarDNet_A_Low_Memory_Traffic_Network_ICCV_2019_paper.html).

## 7. Khoảng trống nghiên cứu

Các nghiên cứu hiện có giải quyết từng phần:

- DominoSearch: layer-wise N:M nhưng chưa deployment benchmark;
- NetAdapt/HALP: measured hardware feedback nhưng không phải mixed N:M;
- nn-Meter/CloserToMe: latency prediction nhưng không tối ưu accuracy N:M;
- nmSPARSE/SparTA: sparse kernel/runtime nhưng không chọn scheme bằng direct
  ImageNet sensitivity trên Jetson Nano;
- LightPrune: Jetson latency-aware pruning nhưng là structured pruning.

Khoảng trống phù hợp cho đề tài:

> Kết hợp conditioned accuracy sensitivity, measured Jetson kernel cost,
> uncertainty và runtime-aware N:M eligibility trong DominoSearch, sau đó xác
> nhận end-to-end trên Jetson Nano.

## 8. Giải pháp đề xuất

Tên làm việc:

> **Measured Kernel-Aware DominoSearch for Jetson Nano**

### 8.1. Pipeline tổng thể

```text
Dense ResNet-18
    ↓
Staged mixed N:M trên Colab
    ↓
MAC18 → MAC20/MAC21/MAC23 candidates
    ↓
Full ImageNet validation + fine-tune
    ↓
Jetson kernel/backend profiling
    ↓
Measured lookup + eligibility + CI95%
    ↓
Hardware-aware selector
    ↓
Fine-tune scheme được chọn
    ↓
C++ end-to-end Jetson benchmark
```

### 8.2. Giai đoạn Colab

1. Bắt đầu từ exact MAC18 epoch-1 checkpoint và scheme.
2. Chạy conditioned sensitivity profile trên 5.000 validation image.
3. Sinh MAC20, MAC21 và MAC23.
4. Screening 1.000/5.000 ảnh.
5. Full validation 50.000 ảnh.
6. Fine-tune 3 epoch, LR 0,001, 50.000 training sample/epoch.
7. Giữ các checkpoint trên accuracy–MAC Pareto frontier.

### 8.3. Giai đoạn Jetson

Ba backend tối thiểu:

1. dense cuDNN/TensorRT;
2. masked-dense PyTorch;
3. packed N:M CUDA kernel nếu feasibility gate đạt.

Đo:

- C++ end-to-end wall latency;
- CUDA Events kernel/stream latency;
- Nsight kernel/API/memory timeline;
- median, P95, standard deviation và CI95%;
- system peak memory;
- measured DRAM traffic nếu counter hỗ trợ;
- power/energy bằng thiết bị đo phù hợp;
- temperature, GPU/EMC clocks và throttling.

### 8.4. Kernel feasibility gate

Không viết toàn bộ kernel trước khi chứng minh feasibility. Bắt đầu với 3–5
convolution 3×3 chiếm latency lớn nhất và pattern `2:4`.

```text
eligible=true khi:
  median speedup >= 5%
  CI95% của speedup không cắt qua 0
  numerical output đạt tolerance
  packing không nằm trong critical inference path
```

Nếu không đạt, giữ `eligible=false` và không cho selector dùng candidate đó.

### 8.5. Hardware lookup

Lookup key:

```text
(kernel group, input/output shape, N:M, backend, precision, power mode)
```

Lookup value:

```text
latency distribution
energy distribution
measured DRAM traffic
peak memory
clock/temperature metadata
eligibility
```

### 8.6. Cost

Chỉ metric thực đo mới có weight dương:

```text
C_i(N:M) = alpha * latency_ci95_high_ratio
         + beta  * energy_ci95_high_ratio
         + gamma * measured_dram_traffic_ratio
         + delta * measured_peak_memory_ratio
```

Metric chưa đo phải là `null` và weight bằng 0. Cost weight được chọn từ yêu cầu
deployment trước khi xem kết quả.

### 8.7. Selector

```text
minimize    conditioned measured accuracy sensitivity

subject to  estimated measured hardware cost <= budget
            candidate eligible = true
            memory <= board budget
            scheme monotonic từ accepted stage trước
```

Lookup được ưu tiên. Predictor chỉ được dùng khi held-out error và ranking đạt
ngưỡng công bố trước.

### 8.8. End-to-end validation

Scheme cuối phải được benchmark lại toàn model. Không dùng tổng lookup làm kết
quả cuối vì fusion, cache, launch, synchronization và residual graph có thể làm
sai phép cộng layer latency.

So sánh tối thiểu:

```text
dense C++ backend
vs. MAC-based mixed N:M
vs. measured-hardware-selected mixed N:M
```

## 9. Tiêu chí kết luận

Chỉ gọi model tốt hơn về accuracy-complexity khi full ImageNet validation chứng
minh nó giảm effective resource nhiều hơn tại accuracy gate đã công bố.

Chỉ gọi model nhanh hơn trên Jetson khi:

- C++ end-to-end median và P95 giảm;
- CUDA/Nsight giải thích được nguồn speedup;
- CI95% của chênh lệch không cắt qua 0;
- ít nhất ba session độc lập cùng chiều;
- không có lợi thế giả do clock, temperature hoặc power mode;
- numerical output đúng;
- packing/setup cost được báo riêng.

Nếu không candidate N:M nào nhanh hơn dense, kết luận hợp lệ là:

> Jetson Nano/runtime/kernel được thử không khai thác hiệu quả mixed N:M; dense
> backend vẫn nằm trên deployment Pareto frontier.

Không được thay kết luận này bằng speedup dự đoán từ MAC.

## 10. Thứ tự ưu tiên triển khai

1. Conditioned sensitivity và staged candidates trên Colab.
2. Dense/masked C++–CUDA Jetson baseline.
3. nn-Meter-style kernel detection bằng Nsight.
4. Full measured lookup cho một Jetson Nano.
5. SparTA-style candidate eligibility.
6. nmSPARSE-inspired packed `2:4` feasibility kernel.
7. HALP-style measured lookup + sensitivity selector.
8. End-to-end validation.
9. CloserToMe/predictor extension khi có nhiều thiết bị hoặc search space lớn.

## 11. Đóng góp dự kiến

Đóng góp có thể trình bày với giảng viên:

> Mở rộng DominoSearch từ FLOPs/model-size constraint sang measured Jetson
> kernel cost; kết hợp conditioned direct sensitivity, uncertainty-aware lookup
> và runtime eligibility để chỉ chọn N:M có bằng chứng triển khai thật.

Đóng góp này vẫn có giá trị khi kết quả kernel là âm, vì nó xác định bằng dữ liệu
ranh giới giữa theoretical sparsity và deployment benefit trên Maxwell.

## 12. Danh mục tài liệu tham khảo

1. [DominoSearch: Find layer-wise fine-grained N:M sparse schemes from dense
   networks, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/ad68473a64305626a27c32a5408552d7-Abstract.html).
2. [NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications,
   ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Tien-Ju_Yang_NetAdapt_Platform-Aware_Neural_ECCV_2018_paper.php).
3. [HALP: Hardware-Aware Latency Pruning](https://arxiv.org/abs/2110.10811).
4. [nn-Meter: Towards Accurate Latency Prediction of Deep-Learning Model
   Inference on Diverse Edge Devices, MobiSys 2021](https://www.microsoft.com/en-us/research/publication/nn-meter-towards-accurate-latency-prediction-of-deep-learning-model-inference-on-diverse-edge-devices/).
5. [CloserToMe: A Unified Framework for Accurate and Transferable Latency
   Prediction Across Heterogeneous Devices, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39779).
6. [On Latency Predictors for Neural Architecture Search, MLSys
   2024](https://proceedings.mlsys.org/paper_files/paper/2024/hash/f03cb785864596fa5901f1359d23fd81-Abstract-Conference.html).
7. [MAPLE: Microprocessor a Priori for Latency Estimation, CVPRW
   2022](https://openaccess.thecvf.com/content/CVPR2022W/ECV/html/Abbasi_MAPLE_Microprocessor_a_Priori_for_Latency_Estimation_CVPRW_2022_paper.html).
8. [Efficient GPU Kernels for N:M-Sparse Weights in Deep Learning, MLSys
   2023](https://proceedings.mlsys.org/paper_files/paper/2023/hash/a10deb4d5227a8ea307ea8ff3cb712f4-Abstract-mlsys2023.html).
9. [SparTA: Deep-Learning Model Sparsity via Tensor-with-Sparsity-Attribute,
   OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/zheng-ningxin).
10. [Accelerating Sparse DNN Models without Hardware-Support via Tile-Wise
    Sparsity](https://arxiv.org/abs/2008.13006).
11. [Accelerating Sparse Convolution with Column-Vector-Wise Sparsity,
    NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/c383e44d9a878d1982d9abb838bd5d8a-Paper-Conference.pdf).
12. [LightPrune: Latency-Aware Structured Pruning for Efficient Deep Inference
    on Embedded Devices, ICCVW 2025](https://openaccess.thecvf.com/content/ICCV2025W/EVW/html/Belhadi_LightPrune_Latency-Aware_Structured_Pruning_for_Efficient_Deep_Inference_on_Embedded_ICCVW_2025_paper.html).
13. [To Bridge Neural Network Design and Real-World Performance: A Behaviour
    Study for Neural Networks, MLSys 2021](https://proceedings.mlsys.org/paper_files/paper/2021/hash/411e39b117e885341f25efb8912945f7-Abstract.html).
14. [HarDNet: A Low Memory Traffic Network, ICCV
    2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Chao_HarDNet_A_Low_Memory_Traffic_Network_ICCV_2019_paper.html).

## 13. Tài liệu dự án liên quan

- [Kế hoạch hardware-measured mixed N:M cho Jetson
  Nano](JETSON_NANO_HARDWARE_MEASURED_MIXED_NM_PLAN.md).
- [Hardware-aware DominoSearch](HARDWARE_AWARE_DOMINOSEARCH.md).
- [Domino staged mixed N:M](DOMINO_STAGED_MIXED_NM.md).
- [Benchmark protocol](../benchmark/README.md).
