# Global Thresholding : Maximum Likelihood Thresholding
# 假設前景與背景各自服從 Gaussian 高斯分布
# 閾值 T 根據最大化聯合似然函數決定

import cv2
import numpy as np

def global_thresholding_mle(image):
    """
    使用 Maximum Likelihood (最大似然法) 進行全域閾值處理
    image: 灰階影像 (numpy array)
    return：閾值、二值影像、最大 log-likelihood
    """
    L = 256
    hist = cv2.calcHist([image], [0], None, [L], [0, L]).ravel()
    hist_norm = hist / hist.sum()  # 機率分佈 P(i)

    intensity = np.arange(L)
    P1 = np.cumsum(hist_norm)      # 前景累積機率
    P2 = 1 - P1                    # 背景累積機率
    eps = 1e-12                    # 防止除零與 log(0)

    log_likelihood = np.full(L, -np.inf)

    for T in range(1, L-1):  # 0 與 255 不合理（全部屬於單一類）
        # 前景區間 0 ~ T
        if P1[T] > 0:
            mu1 = np.sum(intensity[:T+1] * hist_norm[:T+1]) / P1[T] # 前景均值 μ1​
            sigma1_sq = np.sum(((intensity[:T+1] - mu1)**2) * hist_norm[:T+1]) / P1[T] # 前景變異數 σ1²​
            sigma1_sq = max(sigma1_sq, eps)  # 防止除零
            p1 = (1 / np.sqrt(2 * np.pi * sigma1_sq)) * np.exp(-0.5 * ((intensity[:T+1] - mu1)**2) / sigma1_sq) # 代入高斯分布公式
            log_L1 = np.sum(hist_norm[:T+1] * np.log(p1 + eps)) # log domain 避免 underflow
        else:
            log_L1 = -np.inf # 無效值

        # 背景區間 T+1 ~ 255
        if P2[T] > 0:
            mu2 = np.sum(intensity[T+1:] * hist_norm[T+1:]) / P2[T] # 背景均值 μ2​
            sigma2_sq = np.sum(((intensity[T+1:] - mu2)**2) * hist_norm[T+1:]) / P2[T] # 背景變異數 σ2²​
            sigma2_sq = max(sigma2_sq, eps)
            p2 = (1 / np.sqrt(2 * np.pi * sigma2_sq)) * np.exp(-0.5 * ((intensity[T+1:] - mu2)**2) / sigma2_sq)
            log_L2 = np.sum(hist_norm[T+1:] * np.log(p2 + eps))
        else:
            log_L2 = -np.inf # 無效值

        # 聯合似然（log domain）
        log_likelihood[T] = log_L1 + log_L2

    # 找最大似然對應的閾值
    max_val = np.max(log_likelihood)
    k_candidates = np.where(log_likelihood == max_val)[0]
    k_star = int(np.round(np.mean(k_candidates)))

    # 產生二值圖像
    binary = np.where(image > k_star, 255, 0).astype(np.uint8)

    return k_star, binary, max_val


if __name__ == "__main__":
    image = cv2.imread('HW1/lena.bmp', cv2.IMREAD_GRAYSCALE)

    # 執行 Maximum Likelihood 演算法
    k_star, binary, logL = global_thresholding_mle(image)
    print(f"ML threshold = {k_star}")
    print(f"Max log-likelihood = {logL:.12f}")
    
    # 顯示結果
    cv2.imshow('Original', image)
    cv2.imshow('Global : Maximum Likelihood Thresholding', binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
