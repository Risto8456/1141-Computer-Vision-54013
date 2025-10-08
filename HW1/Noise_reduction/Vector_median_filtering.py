# Color in image filtering
# Vector Median Filtering

import cv2
import numpy as np

def vector_median_filter(img, diameter=5):
    """
    實作：Vector Median Filtering
    Args:
        img : ndarray  彩色影像 (H, W, 3)
        diameter : int  鄰域視窗大小（奇數）
    Returns:
        result : ndarray 濾波後的影像
    """
    # 確保是 float32，方便計算
    img = img.astype(np.float32)
    half = diameter // 2
    h, w, c = img.shape

    # 輸出影像初始化
    result = np.zeros_like(img)

    # 對每個像素進行濾波
    for i in range(half, h - half):
        for j in range(half, w - half):

            # Step 1: 擷取鄰域
            patch = img[i - half:i + half + 1, j - half:j + half + 1, :]
            pixels = patch.reshape(-1, 3)  # (k^2, 3) 將鄰域像素展平為向量集合。

            # Step 2: 計算兩兩間歐氏距離總和
            # d(v_i, v_j) = sqrt( (R_i-R_j)^2 + (G_i-G_j)^2 + (B_i-B_j)^2 )
            # D(v_i) = sum_j d(v_i, v_j)
            # 使用 broadcasting 加速，diffs 用廣播計算所有向量之間的距離。
            diffs = pixels[:, np.newaxis, :] - pixels[np.newaxis, :, :]  # shape (N, N, 3)
            dist = np.sqrt(np.sum(diffs ** 2, axis=2))  # shape (N, N)
            sum_dist = np.sum(dist, axis=1)  # shape (N,)，sum_dist：每一個向量與所有其他向量的距離總和。

            # Step 3: 找出距離總和最小的像素
            min_idx = np.argmin(sum_dist) # 找出最靠近中位值的向量。
            median_vector = pixels[min_idx]

            # Step 4: 設定輸出像素
            result[i, j, :] = median_vector

    return np.clip(result, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    I = cv2.imread("HW1/noise.bmp")  # 讀入彩色影像
    result = vector_median_filter(I, diameter=5)

    cv2.imshow("Input", I)
    cv2.imshow("Vector Median Filtered", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
