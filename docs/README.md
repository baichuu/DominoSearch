# Tài liệu DominoSearch

Thư mục này tách kiến thức, kế hoạch và bằng chứng thực nghiệm thành các tài liệu
riêng để tránh nhầm kết quả đã đo với kết quả dự kiến.

## Thứ tự đọc đề xuất

1. [`UNDERSTANDING_DOMINOSEARCH.md`](UNDERSTANDING_DOMINOSEARCH.md): thuật ngữ,
   ý tưởng bài báo và cấu trúc dự án.
2. [`PRUNING_OPTIMIZATION_DIRECTIONS.md`](PRUNING_OPTIMIZATION_DIRECTIONS.md):
   bốn hướng pruning, ưu/nhược điểm và ma trận thí nghiệm tiếp theo.
3. [`COLAB_DRIVE_PRUNING_WORKFLOW.md`](COLAB_DRIVE_PRUNING_WORKFLOW.md): giao
   thức chạy công bằng trên ImageNet từ Google Drive.
4. [`OPTIMIZATION_IMPLEMENTATION_STATUS.md`](OPTIMIZATION_IMPLEMENTATION_STATUS.md):
   code của mỗi hướng nằm ở branch/commit nào và mức xác thực hiện tại.
5. [`PRUNING_EXPERIMENT_REPORT_T4.md`](PRUNING_EXPERIMENT_REPORT_T4.md): báo cáo
   số liệu thực tế trước/sau fine-tune trên Tesla T4 và kết luận hiện tại.

## Quy tắc đọc kết quả

- Parameter/MAC hiệu dụng là độ giảm lý thuyết theo mask hoặc số non-zero.
- Latency, throughput và peak memory là số đo của đúng runtime/thiết bị ghi trong
  báo cáo; không tự suy rộng sang thiết bị khác.
- Báo cáo thực nghiệm là nguồn cho câu hỏi “đã đo được gì”. Tài liệu hướng tối ưu
  là nguồn cho câu hỏi “nên thử gì tiếp theo”.
- JSON, checkpoint, mask, dataset và log không được commit vào repository. Báo
  cáo chỉ trích số liệu đã audit từ các artifact đó.
