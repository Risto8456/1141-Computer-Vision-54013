# Color in image filtering
# Joint bilateral filtering

import cv2
import numpy as np

def joint_bilateral_filter(I, G, diameter=9, sigma_s=12, sigma_r=25):
    """
    實作：Joint Bilateral Filtering（聯合雙邊濾波）
    Args:
        I : ndarray     要被平滑處理的影像（彩色圖像）
        G : ndarray     導引影像（guide image，灰階圖像)
        diameter : int  濾波視窗大小（必須為奇數，如 3、5、7...）
        sigma_s : float 空間權重標準差 (σ_s)，控制距離越遠權重衰減的速度
        sigma_r : float 亮度差權重標準差 (σ_r)，控制顏色差距對權重的影響程度
    Returns：經過 Joint Bilateral Filter 後的影像
    """
    # ---- Step 1: 資料型別處理 ----
    # 若輸入是彩色圖像，保留三通道；若是灰階，擴維成三維方便運算
    if I.ndim == 3:
        I = I.astype(np.float32)
    else:
        I = I[..., np.newaxis].astype(np.float32)

    # 導引影像轉成 float32，避免溢位
    G = G.astype(np.float32)

    # 視窗半徑與圖像長寬
    half = diameter // 2
    h, w = G.shape

    # 輸出影像初始化 (與輸入 I 同大小)
    result = np.zeros_like(I)

    # ---- Step 2: 建立空間權重 (spatial weight) ω_s(p, q) ----
    # 根據像素距離的高斯函數
    # ω_s(p,q) = exp(-||p-q||^2 / (2σ_s^2))
    x, y = np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1))
    spatial_weight = np.exp(-(x**2 + y**2) / (2 * sigma_s**2))

    # ---- Step 3: 對每一個像素進行濾波 ----
    # p 為中心像素座標
    for i in range(half, h - half):
        for j in range(half, w - half):

            # ---- Step 3.1: 擷取鄰域像素 ----
            # N_p 表示以 (i,j) 為中心的鄰域
            patch_I = I[i - half:i + half + 1, j - half:j + half + 1, :]  # 要濾波的彩色區塊
            patch_G = G[i - half:i + half + 1, j - half:j + half + 1]    # 導引影像的對應灰階區塊

            # ---- Step 3.2: 計算亮度權重 (range weight) ω_r ----
            # ω_r(G(p), G(q)) = exp(-( (G(p) - G(q))^2 / (2σ_r^2) ))
            diff = patch_G - G[i, j]
            range_weight = np.exp(-(diff**2) / (2 * sigma_r**2))

            # ---- Step 3.3: 計算聯合權重 (joint weight) ----
            # ω(p,q) = ω_s(p,q) × ω_r(G(p), G(q))
            weight = spatial_weight * range_weight

            # 擴張維度成 (k,k,1)，讓彩色三通道能同時乘上相同權重
            weight = weight[..., np.newaxis]

            # ---- Step 3.4: 計算加權平均 ----
            # J(p) = Σ ω(p,q) I(q) / Σ ω(p,q)
            numerator = np.sum(weight * patch_I, axis=(0, 1))  # 加權像素和
            denominator = np.sum(weight, axis=(0, 1))          # 總權重
            result[i, j, :] = numerator / denominator          # 加權平均值

    # ---- Step 4: 處理單通道情況 ----
    if result.shape[2] == 1:
        result = result.squeeze()

    # ---- Step 5: 回傳結果 ----
    return np.clip(result, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    I = cv2.imread("HW1/noise.bmp")       # 要濾波的影像
    G = cv2.imread("HW1/lena.bmp", cv2.IMREAD_GRAYSCALE)  # 導引影像
    result = joint_bilateral_filter(I, G, diameter=9, sigma_s=12, sigma_r=25)

    cv2.imshow("Input", I)
    cv2.imshow("Guide", G)
    cv2.imshow("Joint Bilateral Filtered", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
