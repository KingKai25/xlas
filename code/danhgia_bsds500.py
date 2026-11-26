import cv2
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path


def load_bsds500_ground_truth(mat_path: Path) -> np.ndarray:
    """
    Đọc file .mat của BSDS500 và hợp nhất đường biên chuẩn từ nhiều người đánh nhãn.
    Trả về ảnh nhị phân (0 và 255).
    """
    mat_data = sio.loadmat(str(mat_path))
    gt_cell = mat_data['groundTruth'][0]

    h, w = gt_cell[0]['Boundaries'][0, 0].shape
    merged_boundaries = np.zeros((h, w), dtype=np.uint8)

    for item in gt_cell:
        boundary = item['Boundaries'][0, 0]
        merged_boundaries = np.maximum(merged_boundaries, boundary)

    return (merged_boundaries * 255).astype(np.uint8)


def compute_edge_metrics(pred_edge: np.ndarray, gt_edge: np.ndarray, tolerance: int = 1) -> tuple:
    """
    Tính Precision, Recall và F1-Score giữa biên dự đoán và biên chuẩn BSDS500.
    tolerance = 1: Cho phép sai lệch vị trí biên tối đa 1 pixel.
    """
    pred_bin = (pred_edge > 127).astype(np.uint8)
    gt_bin = (gt_edge > 127).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
    gt_dilated = cv2.dilate(gt_bin, kernel)
    pred_dilated = cv2.dilate(pred_bin, kernel)

    tp_prec = np.sum(pred_bin & gt_dilated)
    tp_rec = np.sum(gt_bin & pred_dilated)

    total_pred = np.sum(pred_bin)
    total_gt = np.sum(gt_bin)

    precision = (tp_prec / total_pred) if total_pred > 0 else 0.0
    recall = (tp_rec / total_gt) if total_gt > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return round(precision, 4), round(recall, 4), round(f1_score, 4)


def evaluate_f1(f1: float) -> str:
    """Phân loại chất lượng dựa trên F1-Score phù hợp cho thuật toán truyền thống trên BSDS500"""
    if f1 >= 0.40: # 0.40 là ngưỡng cao để đánh giá các thuật toán truyền thống, vì các thuật toán này thường không đạt F1 quá cao so với các phương pháp học sâu
        return "Rất tốt (Sát biên chuẩn)" 
    elif f1 >= 0.30: # 0.30 là ngưỡng trung bình để đánh giá các thuật toán truyền thống, vì các thuật toán này thường không đạt F1 quá cao so với các phương pháp học sâu
        return "Tốt (Khôi phục biên hiệu quả)"
    elif f1 >= 0.20: # 0.20 là ngưỡng thấp để đánh giá các thuật toán truyền thống, vì các thuật toán này thường không đạt F1 quá cao so với các phương pháp học sâu
        return "Khá / Chấp nhận được"
    else: # Nếu F1 thấp hơn 0.20, chất lượng được đánh giá là kém
        return "Kém (Nhiễu nặng / Mất biên)"


def _save_excel_with_autofit(df: pd.DataFrame, excel_path: Path):
    """Hàm nội bộ hỗ trợ ghi file Excel và tự động căn chỉnh độ rộng cột"""
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='BSDS500_Metrics')
        worksheet = writer.sheets['BSDS500_Metrics']
        for column_cells in worksheet.columns:
            length = max(len(str(cell.value)) for cell in column_cells) + 4
            worksheet.column_dimensions[column_cells[0].column_letter].width = length


def run_bsds500_evaluation(img_origin, image_path: Path, project_root: Path, prep_chains: dict, edge_algos: dict):
    base_name = image_path.stem
    mat_path = project_root / "boanh" / "anhbsds500" / "groundtruth" / f"{base_name}.mat"

    if not mat_path.exists():
        print(f"❌ Lỗi: Không tìm thấy file Ground Truth tại '{mat_path}'!")
        print("   Hãy đảm bảo file nhãn .mat có cùng tên với ảnh và nằm trong 'boanh/anhbsds500/groundtruth/'.")
        return

    gt_edge = load_bsds500_ground_truth(mat_path)
    if gt_edge.shape[:2] != img_origin.shape[:2]:
        print(f"❌ Lỗi kích thước: Ảnh đầu vào {img_origin.shape[:2]} không khớp với Ground Truth {gt_edge.shape[:2]}!")
        return

    eval_dir = project_root / "boanh" / "danhgia_bsds500" / base_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Lưu ảnh Ground Truth chuẩn ra thư mục kết quả để tiện chèn vào báo cáo
    cv2.imwrite(str(eval_dir / f"GroundTruth_{base_name}.png"), gt_edge)

    num_rows = len(prep_chains)
    num_cols = len(edge_algos)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4.5 * num_cols, 3.8 * num_rows))
    fig.suptitle(f'ĐÁNH GIÁ VỚI CHUẨN BSDS500: {image_path.name}', fontsize=14, fontweight='bold')

    excel_data = []
    _ = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)  # Warm-up bộ nhớ đệm

    print("\n" + "="*108)
    print(f"ĐÁNH GIÁ ĐỘ CHÍNH XÁC VỚI GROUND TRUTH BSDS500: {image_path.name}")
    print("="*108)
    header = f"{'Chuỗi Tiền Xử Lý':<18} | {'Thuật Toán':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Đánh giá F1':<28}"
    print(header)
    print("-" * 108)

    for row_idx, (p_key, (p_name, p_func)) in enumerate(prep_chains.items()):
        t0 = time.perf_counter()
        preprocessed_img = p_func(img_origin)
        t_prep = (time.perf_counter() - t0) * 1000
        short_p_name = p_name.split(":")[0].strip()

        for col_idx, (e_key, (e_name, e_func)) in enumerate(edge_algos.items()):
            t1 = time.perf_counter()
            edge_img = e_func(preprocessed_img)
            t_edge = (time.perf_counter() - t1) * 1000
            total_time = round(t_prep + t_edge, 2)

            prec, rec, f1 = compute_edge_metrics(edge_img, gt_edge, tolerance=1)
            f1_comment = evaluate_f1(f1)

            print(f"{short_p_name:<18} | {e_name:<10} | {prec:>10.4f} | {rec:>10.4f} | {f1:>10.4f} | {f1_comment:<28}")

            excel_data.append({
                "Mã Chuỗi": short_p_name,
                "Chi tiết Tiền xử lý": p_name,
                "Thuật toán Biên": e_name,
                "Precision (Độ chính xác)": prec,
                "Recall (Độ phủ biên)": rec,
                "F1-Score (Chỉ số tổng hợp)": f1,
                "Đánh giá chất lượng": f1_comment,
                "Thời gian tiền xử lý (ms)": round(t_prep, 2),
                "Thời gian tìm biên (ms)": round(t_edge, 2),
                "Tổng thời gian xử lý (ms)": total_time
            })

            if num_rows == 1 and num_cols == 1:
                ax = axes
            elif num_rows == 1:
                ax = axes[col_idx]
            elif num_cols == 1:
                ax = axes[row_idx]
            else:
                ax = axes[row_idx, col_idx]

            ax.imshow(edge_img, cmap='gray')
            ax.set_title(f"{short_p_name} + {e_name}\n(F1: {f1:.3f} | P:{prec:.2f} R:{rec:.2f})", fontsize=9)
            ax.axis('off')

    print("="*108)

    # 1. Lưu ảnh lưới tổng hợp
    grid_img_path = eval_dir / f"TongHop_BSDS500_{base_name}.png"
    plt.tight_layout()
    plt.savefig(grid_img_path, dpi=200)

    # 2. Xuất file Excel (Tự động đổi tên nếu file cũ đang mở)
    excel_path = eval_dir / f"DanhGia_BSDS500_{base_name}.xlsx"
    df = pd.DataFrame(excel_data)

    try:
        _save_excel_with_autofit(df, excel_path)
    except PermissionError:
        timestamp = time.strftime("%H%M%S")
        excel_path = eval_dir / f"DanhGia_BSDS500_{base_name}_{timestamp}.xlsx"
        print(f"\n⚠️ File Excel cũ đang mở nên bị khóa! Tự động lưu sang file mới: '{excel_path.name}'")
        _save_excel_with_autofit(df, excel_path)

    print(f"\n✅ [HOÀN TẤT] Đã lưu kết quả đánh giá BSDS500 tại: '{eval_dir}'")
    print(f"   - Ảnh Ground Truth : GroundTruth_{base_name}.png")
    print(f"   - Ảnh tổng hợp     : {grid_img_path.name}")
    print(f"   - File Excel       : {excel_path.name}\n")

    plt.show()