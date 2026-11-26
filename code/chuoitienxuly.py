import cv2

#file chứa các chuỗi tiền xử lý và để lựa chọn các chuỗi. ví dụ chuỗi base line (chuỗi đối chứng) bao gồm 1 bước là grayscale; chuỗi 0 (cơ bản) bao gồm 2 bước là grayscale, lọc gauss; chuỗi 1 để xử lý ảnh nhiễu muối tiêu bao gồm 2 bước là grayscale, lọc trung vị; chuỗi 2 bao gồm 3 bước là grayscale, tăng độ tương phản, lọc gauss 5x5; chuỗi 3 nâng cao cho ảnh nhiễu phức tạp bao gồm các bước là grayscale, lọc bilateral, tăng tương phản, phép toán Morphological Opening (Mở) làm sạch đốm nhiễu

def chain_0_basic(image):
    """
    Chuỗi 0 (Cơ bản - 2 bước): 
    Bước 1: Chuyển sang ảnh xám (Grayscale)
    Bước 2: Lọc Gauss nhẹ (3x3) để giảm nhiễu nền
    """
    # Kiểm tra nếu ảnh là ảnh màu (3 kênh), chuyển sang ảnh xám
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Chuyển sang ảnh xám
    else:
        gray = image.copy() # Nếu ảnh đã là ảnh xám, giữ nguyên

    # Áp dụng lọc Gauss 3x3 để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return blurred # Trả về ảnh đã được xử lý qua chuỗi cơ bản

def chain_1_denoise_median(image):
    """
    Chuỗi 1 (Chuyên khử nhiễu muối tiêu - 2 bước):
    Bước 1: Chuyển sang ảnh xám
    Bước 2: Lọc Trung vị (Median Filter 5x5) giữ sắc nét biên
    """
    # Kiểm tra nếu ảnh là ảnh màu (3 kênh), chuyển sang ảnh xám
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy() # Nếu ảnh đã là ảnh xám, giữ nguyên
    # Áp dụng lọc trung vị 5x5 để loại bỏ nhiễu muối tiêu, 5x5 là kích thước kernel phổ biến để giữ biên sắc nét    
    filtered = cv2.medianBlur(gray, 5)
    return filtered

def chain_2_contrast_boost(image):
    """
    Chuỗi 2 (Tăng tương phản + Khử nhiễu - 3 bước):
    Bước 1: Chuyển sang ảnh xám
    Bước 2: Lọc Gauss (5x5)
    Bước 3: Tăng cường độ tương phản cục bộ (CLAHE)
    """
    # Kiểm tra nếu ảnh là ảnh màu (3 kênh), chuyển sang ảnh xám
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy() # Nếu ảnh đã là ảnh xám, giữ nguyên
    # Áp dụng lọc Gauss 5x5 để giảm nhiễu nền trước khi tăng tương phản
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    # Tăng cường độ tương phản cục bộ bằng CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Tạo đối tượng CLAHE với giới hạn cắt là 2.0 và kích thước ô là 8x8 bởi vì CLAHE hoạt động trên các ô nhỏ để tránh tăng quá mức độ tương phản
    enhanced = clahe.apply(denoised)
    return enhanced

def chain_3_advanced_morph(image):
    """
    Chuỗi 3 (Nâng cao cho ảnh nhiễu phức tạp - 4 bước):
    Bước 1: Chuyển sang ảnh xám
    Bước 2: Lọc Bilateral (khử nhiễu nhưng bảo vệ biên)
    Bước 3: Tăng tương phản CLAHE
    Bước 4: Phép toán Morphological Opening (Mở) làm sạch đốm nhiễu
    """
    # Kiểm tra nếu ảnh là ảnh màu (3 kênh), chuyển sang ảnh xám
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy() # Nếu ảnh đã là ảnh xám, giữ nguyên
    # Áp dụng lọc Bilateral để khử nhiễu nhưng vẫn giữ biên sắc nét. Lọc Bilateral là một kỹ thuật lọc phi tuyến, giúp giảm nhiễu trong khi bảo vệ các cạnh của đối tượng trong ảnh.    
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75) # Áp dụng lọc Bilateral để khử nhiễu với d=9 (kích thước kernel), sigmaColor=75 (độ lệch chuẩn trong không gian màu) và sigmaSpace=75 (độ lệch chuẩn trong không gian không gian). Các giá trị này có thể được điều chỉnh tùy thuộc vào mức độ nhiễu và đặc điểm của ảnh.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Tạo đối tượng CLAHE với giới hạn cắt là 2.0 và kích thước ô là 8x8 để tăng cường độ tương phản cục bộ mà không làm quá sáng các vùng đồng nhất
    enhanced = clahe.apply(denoised) # Áp dụng CLAHE lên ảnh đã được khử nhiễu để tăng cường độ tương phản
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # Tạo kernel hình chữ nhật 3x3 để sử dụng trong phép toán Morphological Opening (Mở). Kernel này sẽ được sử dụng để loại bỏ các đốm nhiễu nhỏ trong ảnh.
    morph = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel) # Áp dụng phép toán Morphological Opening (Mở) để loại bỏ các đốm nhiễu nhỏ trong ảnh. Phép toán này bao gồm hai bước: Erosion (Xói mòn) và Dilation (Phóng đại). Erosion sẽ loại bỏ các điểm sáng nhỏ, trong khi Dilation sẽ phục hồi các vùng sáng lớn hơn. Kết quả là các đốm nhiễu nhỏ sẽ bị loại bỏ, trong khi các chi tiết quan trọng vẫn được giữ lại.
    return morph # Trả về ảnh đã được xử lý qua chuỗi nâng cao, bao gồm các bước khử nhiễu, tăng cường độ tương phản và loại bỏ đốm nhiễu nhỏ.

def chain_baseline_gray(image):
    """
    Chuỗi Baseline (Đối chứng): Chỉ chuyển sang ảnh xám, KHÔNG lọc nhiễu hay chỉnh tương phản
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()