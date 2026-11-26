import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import các module xử lý và đánh giá BSDS500
import chuoitienxuly as prep
import thuattoan as edge
import danhgia_bsds500 as bsds_eval

PREPROCESSING_CHAINS = {
    "B": ("Baseline: Chỉ Grayscale (Không tiền xử lý)", prep.chain_baseline_gray),
    "0": ("Chuỗi 0: Cơ bản (Grayscale + Gauss 3x3)", prep.chain_0_basic),
    "1": ("Chuỗi 1: Khử nhiễu muối tiêu (Median 5x5)", prep.chain_1_denoise_median),
    "2": ("Chuỗi 2: Tăng tương phản (Gauss 5x5 + CLAHE)", prep.chain_2_contrast_boost),
    "3": ("Chuỗi 3: Nâng cao (Bilateral + CLAHE + Morph)", prep.chain_3_advanced_morph)
}

EDGE_ALGORITHMS = {
    "0": ("Sobel", edge.sobel),
    "1": ("Prewitt", edge.prewitt),
    "2": ("Laplace", edge.laplace),
    "3": ("Canny", edge.canny)
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "boanh" / "anhbsds500" / "images"
OUTPUT_DIR = PROJECT_ROOT / "boanh" / "anhxuat"
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def choose_image_path() -> Path | None:
    """Chọn ảnh từ thư mục anhbsds500/images hoặc nhập đường dẫn trực tiếp."""
    raw_path = input(
        f"Nhập đường dẫn ảnh (Enter để chọn trong '{INPUT_DIR}'): "
    ).strip().strip('"').strip("'")

    if raw_path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidates = (
                Path.cwd() / candidate,
                PROJECT_ROOT / candidate,
                INPUT_DIR / candidate,
            )
            candidate = next((path for path in candidates if path.is_file()), candidates[0])
        if candidate.is_file():
            return candidate
        print(f"❌ Không tìm thấy file ảnh: '{candidate}'")
        return None

    image_paths = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ) if INPUT_DIR.is_dir() else []

    if not image_paths:
        print(f"❌ Không có ảnh trong thư mục '{INPUT_DIR}'.")
        return None
    if len(image_paths) == 1:
        return image_paths[0]

    print("\nCác ảnh tìm thấy trong bộ BSDS500:")
    for index, path in enumerate(image_paths, start=1):
        print(f"[{index}] {path.name}")
    while True:
        choice = input("Chọn số ảnh: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(image_paths):
            return image_paths[int(choice) - 1]
        print("❌ Lựa chọn không hợp lệ.")


def load_image(image_path: Path) -> np.ndarray | None:
    """Đọc ảnh hỗ trợ đường dẫn có dấu tiếng Việt."""
    try:
        encoded = np.fromfile(image_path, dtype=np.uint8)
    except OSError as error:
        print(f"❌ Không thể đọc file ảnh: {error}")
        return None

    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        print(f"❌ Không thể giải mã ảnh: '{image_path}'")
    return image


def save_image(image: np.ndarray, output_path: Path) -> bool:
    """Lưu ảnh hỗ trợ đường dẫn có dấu tiếng Việt."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        print(f"❌ Không thể mã hóa ảnh để lưu: '{output_path}'")
        return False

    try:
        encoded.tofile(output_path)
    except OSError as error:
        print(f"❌ Không thể lưu ảnh '{output_path}': {error}")
        return False
    return True


def display_menu(title, options):
    print(f"\n--- {title} ---")
    for key, (name, _) in options.items():
        print(f"[{key}] {name}")
    print("[A] Tất cả (All)")


def get_user_choice(title, options):
    display_menu(title, options)
    while True:
        choice = input("Nhập lựa chọn của bạn: ").strip().upper()
        if choice in options or choice == 'A':
            return choice
        print("❌ Lựa chọn không hợp lệ, vui lòng chọn lại!")


def get_custom_threshold(edge_choice: str) -> tuple:
    """Hỏi và kiểm tra tính hợp lệ của ngưỡng thủ công theo thuật toán đã chọn."""
    if edge_choice == "A":
        print("\nℹ️ Chạy tất cả thuật toán [A]: Hệ thống tự động dùng ngưỡng Otsu để đảm bảo công bằng.")
        return ()

    is_canny = (edge_choice == "3")
    prompt_msg = (
        "\nNhập 2 ngưỡng cho Canny dạng 'low, high' trong khoảng 0-255 (VD: 50, 150 | Enter để dùng Otsu): "
        if is_canny else
        "\nNhập 1 ngưỡng cắt trong khoảng 0-255 (VD: 100 | Enter để dùng Otsu): "
    )

    while True:
        raw_thresh = input(prompt_msg).strip()

        if not raw_thresh:
            return ()

        raw_parts = [item.strip() for item in raw_thresh.split(",")]
        try:
            values = [int(val) for val in raw_parts]
        except ValueError:
            print("❌ Lỗi: Ngưỡng phải là số nguyên (không chứa chữ cái hoặc ký tự lạ)!")
            continue

        if any(v < 0 or v > 255 for v in values):
            print("❌ Lỗi: Giá trị ngưỡng phải nằm trong khoảng từ 0 đến 255!")
            continue

        if is_canny:
            if len(values) != 2:
                print("❌ Lỗi: Thuật toán Canny yêu cầu nhập đủ 2 ngưỡng cách nhau bởi dấu phẩy (VD: 50, 150)!")
                continue
            if values[0] >= values[1]:
                print("❌ Lỗi: Ngưỡng thấp (threshold1) phải nhỏ hơn ngưỡng cao (threshold2)!")
                continue
            return (values[0], values[1])
        else:
            if len(values) != 1:
                print("❌ Lỗi: Thuật toán này chỉ nhận duy nhất 1 giá trị ngưỡng (VD: 100)!")
                continue
            return (values[0],)


def main():
    # 1. Chọn ảnh đầu vào
    image_path = choose_image_path()
    if image_path is None:
        return

    img_origin = load_image(image_path)
    if img_origin is None:
        return

    # 2. Chọn chế độ thực thi
    print("\n=== CHỌN CHẾ ĐỘ CHẠY ===")
    print("[1] Chế độ thủ công (Tự chọn chuỗi tiền xử lý & thuật toán, xuất ảnh lẻ vào 'anhxuat')")
    print("[2] Chế độ đánh giá tự động BSDS500 (Đối chiếu Ground Truth .mat -> Precision, Recall, F1)")

    mode = ""
    while mode not in ["1", "2"]:
        mode = input("Chọn chế độ (1 hoặc 2): ").strip()

    if mode == "2":
        bsds_eval.run_bsds500_evaluation(
            img_origin=img_origin,
            image_path=image_path,
            project_root=PROJECT_ROOT,
            prep_chains=PREPROCESSING_CHAINS,
            edge_algos=EDGE_ALGORITHMS
        )
        return

    # 3. Chế độ thủ công
    prep_choice = get_user_choice("CHỌN CHUỖI TIỀN XỬ LÝ", PREPROCESSING_CHAINS)
    edge_choice = get_user_choice("CHỌN THUẬT TOÁN PHÁT HIỆN BIÊN", EDGE_ALGORITHMS)
    custom_args = get_custom_threshold(edge_choice)
    print(f"ℹ️ Ngưỡng áp dụng: {custom_args if custom_args else 'Tự động (Otsu)'}")

    selected_preps = PREPROCESSING_CHAINS if prep_choice == 'A' else {prep_choice: PREPROCESSING_CHAINS[prep_choice]}
    selected_edges = EDGE_ALGORITHMS if edge_choice == 'A' else {edge_choice: EDGE_ALGORITHMS[edge_choice]}

    num_rows = len(selected_preps)
    num_cols = len(selected_edges)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    fig.suptitle(f'KẾT QUẢ THỬ NGHIỆM: {image_path.name}', fontsize=12, fontweight='bold')

    saved_count = 0
    for row_idx, (p_key, (p_name, p_func)) in enumerate(selected_preps.items()):
        preprocessed_img = p_func(img_origin)

        for col_idx, (e_key, (e_name, e_func)) in enumerate(selected_edges.items()):
            edge_img = e_func(preprocessed_img, *custom_args)
            output_path = OUTPUT_DIR / f"{image_path.stem}_prep{p_key}_edge{e_key}.png"
            if save_image(edge_img, output_path):
                saved_count += 1

            if num_rows == 1 and num_cols == 1:
                ax = axes
            elif num_rows == 1:
                ax = axes[col_idx]
            elif num_cols == 1:
                ax = axes[row_idx]
            else:
                ax = axes[row_idx, col_idx]

            ax.imshow(edge_img, cmap='gray')
            ax.set_title(f"{p_name.split(':')[0]}\n+ {e_name}", fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    print(f"\n✅ Đã lưu {saved_count} ảnh vào: '{OUTPUT_DIR}'")
    plt.show()


if __name__ == '__main__':
    main()