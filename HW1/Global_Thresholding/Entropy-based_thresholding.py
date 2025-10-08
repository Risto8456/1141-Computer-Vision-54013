# Global Thresholding : Entropy-based thresholding

import cv2
import numpy as np

def global_thresholding_entropy(image):
    """
    使用 Entropy (最大熵法) 進行全域閾值處理
    image: 灰階影像 (numpy array)
    return：閾值、二值影像、最大熵值 H*
    """
    # 步驟 1. 計算直方圖與機率分佈
    L = 256
    hist = cv2.calcHist([image], [0], None, [L], [0, L]).ravel()
    hist_norm = hist / hist.sum()  # 正規化成機率分佈 P(i)

    # 步驟 2. 計算累積機率 (前景與背景)
    P1 = np.cumsum(hist_norm)          # 前景機率 P1(T)
    P2 = 1 - P1                        # 背景機率 P2(T)

    # 小常數，防止 log(0) 錯誤
    eps = 1e-12

    # 步驟 3. 對每個灰階分界 T，計算前景與背景熵
    H1 = np.zeros(L)
    H2 = np.zeros(L)

    for T in range(L):
        # 前景區間 0 ~ T
        if P1[T] > 0:
            p1 = hist_norm[:T+1] / (P1[T]) # 前景的條件機率分佈，P(i∣C1​) = p(i)/P1(T), i in [0,T]
            H1[T] = -np.sum(p1 * np.log(p1 + eps))
        else:
            H1[T] = 0

        # 背景區間 T+1 ~ 255
        if P2[T] > 0:
            p2 = hist_norm[T+1:] / (P2[T]) # 背景的條件機率分佈，P(i∣C2​) = p(i)/P2(T), i in [T+1,255]
            H2[T] = -np.sum(p2 * np.log(p2 + eps))
        else:
            H2[T] = 0

    # 步驟 4. 找最大熵值
    H_total = H1 + H2
    max_val = np.max(H_total)
    k_candidates = np.where(H_total == max_val)[0]
    k_star = int(np.mean(k_candidates))  # 若多個最大值取平均

    # 步驟 5. 產生二值圖像
    binary = np.where(image > k_star, 255, 0).astype(np.uint8)

    return k_star, binary, max_val


if __name__ == "__main__":
    image = cv2.imread('HW1/lena.bmp', cv2.IMREAD_GRAYSCALE)

    # 執行 Entropy-based 演算法
    k_star, binary, H_star = global_thresholding_entropy(image)
    print(f"Entropy threshold = {k_star}")
    print(f"Max entropy H* = {H_star:.4f}")

    # 顯示結果
    cv2.imshow('Original', image)
    cv2.imshow('Global : Entropy Thresholding', binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
