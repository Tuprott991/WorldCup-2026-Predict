# World Cup 2026 Prediction Pipeline

Pipeline này đọc dataset `International football results from 1872 to 2026`, train các mô hình match-level, rồi chạy Monte Carlo để xuất bảng xác suất World Cup 2026.

## Chạy nhanh

```powershell
python main.py --model ensemble --mode pre_tournament --simulations 20000 --output-dir outputs/ensemble_pre
```

Các model có thể chọn:

- `elo`: Elo baseline.
- `dixon-coles`: Poisson goals model với Dixon-Coles low-score correction.
- `boosted`: XGBoost/LightGBM/CatBoost nếu có, fallback sang sklearn HistGradientBoosting.
- `xgboost`: ép dùng riêng XGBoost W-D-L model.
- `catboost`: ép dùng riêng CatBoost W-D-L model.
- `bayesian`: empirical-Bayes Poisson với shrinkage cho đội ít dữ liệu.
- `ensemble`: ensemble có calibration từ validation gần nhất; tự include LightGBM, XGBoost và CatBoost nếu package có trong env.

## Mode dữ liệu

- `pre_tournament`: cutoff mặc định `2026-06-10`, bỏ qua toàn bộ kết quả World Cup 2026 dù dataset đã có vài trận.
- `live`: cutoff mặc định là ngày World Cup 2026 mới nhất đã có score trong dataset, và dùng các kết quả đó khi simulate phần còn lại.

Ví dụ live:

```powershell
python main.py --model ensemble --mode live --simulations 20000 --output-dir outputs/ensemble_live
```

XGBoost với CUDA nếu môi trường có GPU/CUDA tương thích:

```powershell
python main.py --model xgboost --mode pre_tournament --simulations 20000 --xgboost-device cuda --output-dir outputs/xgboost_cuda_pre
```

Thanh tiến trình Monte Carlo dùng `tqdm` mặc định nếu package có sẵn. Thêm `--no-progress` để tắt.

## Output

Mỗi lần chạy sẽ ghi:

- `champion_probabilities.csv`: xác suất vô địch.
- `stage_probabilities.csv`: xác suất vào Round of 32, Round of 16, quarterfinal, semifinal, final, champion.
- `group_probabilities.csv`: xác suất xếp hạng 1-4 từng bảng.
- `group_projection.csv`: điểm, bàn thắng, bàn thua, hiệu số kỳ vọng.
- `match_predictions.csv`: xác suất từng fixture vòng bảng.
- `metadata.json`: cutoff, số trận train, weights/log-loss của ensemble.

Lưu ý: Round-of-32 dùng match slots chính thức; đội hạng ba được gán bằng heuristic backtracking hợp lệ thay vì hardcode toàn bộ 495 tổ hợp Annex C.
