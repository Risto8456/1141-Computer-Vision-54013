# Color in image filtering
# RGB to HSV
import numpy as np
import cv2

def rgb_to_hsv_manual(img_bgr):
    """手動實作 RGB -> HSV 轉換"""
    img = img_bgr.astype(np.float32) / 255.0  # normalize
    B, G, R = cv2.split(img)
    
    Cmax = np.maximum(np.maximum(R, G), B)
    Cmin = np.minimum(np.minimum(R, G), B)
    delta = Cmax - Cmin

    # Hue
    H = np.zeros_like(Cmax)
    mask = delta != 0
    # when Cmax == R
    idx = (Cmax == R) & mask
    H[idx] = 60 * (((G[idx] - B[idx]) / delta[idx]) % 6)
    # when Cmax == G
    idx = (Cmax == G) & mask
    H[idx] = 60 * (((B[idx] - R[idx]) / delta[idx]) + 2)
    # when Cmax == B
    idx = (Cmax == B) & mask
    H[idx] = 60 * (((R[idx] - G[idx]) / delta[idx]) + 4)
    H[~mask] = 0

    # Saturation
    S = np.zeros_like(Cmax)
    S[Cmax != 0] = delta[Cmax != 0] / Cmax[Cmax != 0]

    # Value
    V = Cmax

    hsv = cv2.merge([H, S, V])
    return hsv


def hsv_to_rgb_manual(hsv):
    """手動實作 HSV -> RGB 轉換"""
    H, S, V = cv2.split(hsv)

    C = V * S
    H_ = H / 60.0
    X = C * (1 - np.abs((H_ % 2) - 1))
    m = V - C

    # 建立空白輸出
    R1 = np.zeros_like(H)
    G1 = np.zeros_like(H)
    B1 = np.zeros_like(H)

    # 根據區段條件給值
    conds = [
        (0 <= H_) & (H_ < 1),
        (1 <= H_) & (H_ < 2),
        (2 <= H_) & (H_ < 3),
        (3 <= H_) & (H_ < 4),
        (4 <= H_) & (H_ < 5),
        (5 <= H_) & (H_ < 6),
    ]
    Z = np.zeros_like(H) # 零陣列
    rgb_values = [
        (C, X, Z),
        (X, C, Z),
        (Z, C, X),
        (Z, X, C),
        (X, Z, C),
        (C, Z, X),
    ]
    for cond, (r, g, b) in zip(conds, rgb_values):
        R1[cond], G1[cond], B1[cond] = r[cond], g[cond], b[cond]

    # 加上 m 並轉回 0~255
    R = (R1 + m) * 255
    G = (G1 + m) * 255
    B = (B1 + m) * 255

    rgb = cv2.merge([B, G, R]).astype(np.uint8)
    return rgb


def noise_reduction_hsv_manual(img_bgr, method="None"):
    """
    彩色影像雜訊抑制：RGB -> HSV -> 對 V 分量去噪 -> 回 RGB
    """
    hsv = rgb_to_hsv_manual(img_bgr)
    H, S, V = cv2.split(hsv)
    cv2.imshow("HSV(Output in RGB mode)", hsv)

    if method == "median": # 中值濾波
        V_denoised = cv2.medianBlur(V, 5)
    elif method == "gaussian": # 高斯濾波
        V_denoised = cv2.GaussianBlur(V, (5, 5), 0)
    elif method == "bilateral": # 雙邊濾波
        V_denoised = cv2.bilateralFilter(V, 9, 75, 75)
    else:
        V_denoised = V

    hsv_denoised = cv2.merge([H, S, V_denoised])
    img_bgr_denoised = hsv_to_rgb_manual(hsv_denoised)
    return img_bgr_denoised

if __name__ == "__main__":
    img = cv2.imread("HW1/noise.bmp")
    cv2.imshow("Original(RGB)", img)

    method = "bilateral" # "None", "median", "gaussian", "bilateral"
    result = noise_reduction_hsv_manual(img, method)
    cv2.imshow(f"Result(RGB), ({method})", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
