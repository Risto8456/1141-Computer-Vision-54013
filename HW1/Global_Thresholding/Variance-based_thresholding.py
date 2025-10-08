# Global Thresholding : Variance-based thresholding

import cv2
import numpy as np

def global_thresholding_otsu(image):
    """
    使用 Otsu 方差法進行全域閾值處理
    image: 灰階影像 (numpy array)
    return：閾值、二值影像、可分離性η*
    """
    # 步驟 1. 計算直方圖與機率分佈
    L = 256 # 灰階級數
    hist = cv2.calcHist([image], [0], None, [L], [0, L]).ravel()
    hist_norm = hist / hist.sum() # 正規化直方圖，p_i
    '''步驟 1 解釋：
    1. `cv2.calcHist([image], [0], None, [L], [0, L])`

    * `cv2.calcHist(images, channels, mask, histSize, ranges)` 是 OpenCV 計算直方圖的函式。
    * `images`：一個影像清單，這裡傳 ` [image]`（即使只一張影像也需放在 list 裡）。
    * `channels`：要用哪個通道計算直方圖；灰階圖用 `[0]`（單通道），彩色圖會用 0/1/2 分別對應 B/G/R。
    * `mask`：可以傳入遮罩 (mask) 只對部分像素計數。`None` 表示對整張圖計數。
    * `histSize`：bin 的數量，`[256]` 代表 0–255 共有 256 個 bin（常見於 8-bit 灰階）。
    * `ranges`：灰階值的範圍 `[0, 256]`，實際範圍是左閉右開區間 [0, 256)。
    * 回傳值：一個 shape 為 `(256, 1)` 的陣列（預設 dtype 是 `float32`），每一個元素是該灰階值的**像素個數（count）**。

    2. `.ravel()`

    * `calcHist` 回傳 `(256,1)` 二維陣列；`.ravel()` 把它攤平成一維陣列 `(256,)`，方便後續用索引 `hist[i]` 取得第 `i` 個灰階的計數。
    * `ravel()` 與 `flatten()` 很相似，`ravel()` 可能回傳視圖而非複本（效能較好）。

    3. `hist_norm = hist / hist.sum()`

    * `hist.sum()` 為所有 bin 的總和，當 `mask=None` 時，這個和就是像素總數 `H * W`（或掩膜中非零像素數）。
    * `hist / hist.sum()` 把每個 bin 的**計數**除以總像素數 → 得到**機率分布**：
        [
        P(i) = \frac{\text{count of gray level } i}{\text{total pixels}}
        ]
        也就是 Otsu 演算法所需要的 `P(i)`（normalized histogram）。
    * 結果 `hist_norm` 的和等於 1（或非常接近 1，取決於浮點誤差）。
    '''

    # 步驟 2. 累積機率 P1(k)
    P1 = np.cumsum(hist_norm)
    # cumsum 是 numpy 的累積和函式，P1[k] = sum(P(i)) for i=0 to k

    # 步驟 3. 累積平均 m(k)
    intensity = np.arange(L) # arange 產生 0 到 255 的陣列
    m = np.cumsum(intensity * hist_norm) # m[k] = sum(i * P(i)) for i=0 to k

    # 步驟 4. 全域平均 mG
    mG = m[-1] # mG = sum(i * P(i)) for i=0 to 255(k=L-1)

    # 步驟 5. 計算類間方差 σ_{B}^{2}(k)
    numerator = (mG * P1 - m)**2
    denominator = P1 * (1 - P1)
    # 避免除以 0
    valid = denominator > 0         # 產生有效分母的布林陣列
    sigma_b2 = np.zeros_like(P1)    # 初始化類間方差陣列，全設為 0
    sigma_b2[valid] = numerator[valid] / denominator[valid] # 布林遮罩，只計算有效值
    '''步驟 5 解釋：
    當 P1(k) = 0 或 P1(k) = 1 時，代表整張圖全部屬於某一類（前景或背景），
    此時類間方差理論上就應該是 0。
    '''

    # 步驟 6. 找最大方差對應閾值
    max_val = np.max(sigma_b2)
    k_candidates = np.where(sigma_b2 == max_val)[0] # 找出所有最大值對應的 k，[0] 取出索引陣列
    k_star = int(np.mean(k_candidates))  # 若有多個最大值，取平均

    # 步驟 7. 類別分離度 η*，越大表示分離效果越好
    # 全域方差 σ_{G}^{2}
    sigma_g2 = np.sum(((intensity - mG)**2) * hist_norm)
    eta_star = sigma_b2[k_star] / sigma_g2 if sigma_g2 > 0 else 0

    # 產生二值圖像
    binary = np.where(image > k_star, 255, 0).astype(np.uint8)

    return k_star, binary, eta_star


if __name__ == "__main__":
    image = cv2.imread('HW1/lena.bmp', cv2.IMREAD_GRAYSCALE)

    # 執行 Otsu 演算法
    k_star, binary, eta = global_thresholding_otsu(image)
    print(f"Otsu threshold = {k_star}")
    print(f"Separability η* = {eta:.4f}")

    # 顯示結果
    cv2.imshow('Original', image)
    cv2.imshow('Global : Otsu Thresholding', binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
