# Object Detection 物體偵測
# Generalized Hough Transform 廣義霍夫變換

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

curr_fold = os.path.dirname(os.path.abspath(__file__))          # 目前資料夾
temp_path = os.path.join(curr_fold, "data", "Template.png")     # 偵測物影像
ref_path = os.path.join(curr_fold, "data", "Refernce.png")      # 待偵測影像

# ==========================================================
# 1. 讀取影像(支援中文路徑)
# ==========================================================
temp_data = np.fromfile(temp_path, dtype=np.uint8)
ref_data = np.fromfile(ref_path, dtype=np.uint8)

template = cv2.imdecode(temp_data, cv2.IMREAD_COLOR)
reference = cv2.imdecode(ref_data, cv2.IMREAD_COLOR)

temp_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

# ==========================================================
# 2. Canny 邊緣 + Sobel 計算梯度方向
# ==========================================================
temp_edges = cv2.Canny(temp_gray, 80, 150)
ref_edges  = cv2.Canny(ref_gray, 80, 150)

# Sobel 計算梯度方向
def gradient_direction(img_gray):
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    directions = np.arctan2(gy, gx)  # range (-pi, pi)
    return directions

temp_dir = gradient_direction(temp_gray)
ref_dir  = gradient_direction(ref_gray)

# ==========================================================
# 3. 建立 R-table
# ==========================================================

# 定義參考點（範本中心）
h, w = temp_edges.shape
xc, yc = w // 2, h // 2

# 方向量化
NBINS = 60
def quantize_angle(theta):
    # theta ∈ (-pi, pi)
    bin = int(((theta + np.pi) / (2*np.pi)) * NBINS)
    return bin % NBINS

# R-table 結構：list of lists
R_table = [[] for _ in range(NBINS)]

# 掃描範本邊緣
ys, xs = np.where(temp_edges > 0)
for (x, y) in zip(xs, ys):
    phi = temp_dir[y, x]
    bin_id = quantize_angle(phi)

    dx = xc - x
    dy = yc - y
    r = np.hypot(dx, dy)
    alpha = np.arctan2(dy, dx)  # 從 edge 指到中心的角度

    R_table[bin_id].append((r, alpha))

print("R-table 建立完成")

# ==========================================================
# 4. 在 Reference 上進行投票
# ==========================================================
H, W = ref_edges.shape
accumulator = np.zeros((H, W), dtype=np.float32)

ys, xs = np.where(ref_edges > 0)
for (x, y) in zip(xs, ys):
    phi = ref_dir[y, x]
    bin_id = quantize_angle(phi)

    # 找不到對應方向的範本點：跳過
    if len(R_table[bin_id]) == 0:
        continue

    for (r, alpha) in R_table[bin_id]:
        # 預測中心位置
        xc_hat = int(x + r * np.cos(alpha))
        yc_hat = int(y + r * np.sin(alpha))

        if 0 <= xc_hat < W and 0 <= yc_hat < H:
            accumulator[yc_hat, xc_hat] += 1

# test
# plt.imshow(cv2.cvtColor(accumulator, cv2.COLOR_BGR2RGB))
print("投票完成")

# ==========================================================
# 5. 找出累加器最大值（偵測結果）
# ==========================================================
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(accumulator)
(cx, cy) = maxLoc  # maxLoc = (x, y)

print("偵測到中心點位置：", (cx, cy))
print("票數：", maxVal)

# ==========================================================
# 6. 畫出偵測結果（框出偵測物）
# ==========================================================

# 用 Template 大小作為框大小
th, tw = template.shape[:2]
x1 = cx - tw//2
y1 = cy - th//2
x2 = cx + tw//2
y2 = cy + th//2

# 框不超出 Reference 影像
rh, rw = reference.shape[:2]
x1 = max(0, x1)
y1 = max(0, y1)
x2 = min(rw, x2)
y2 = min(rh, y2)

# 畫框
output = reference.copy()
cv2.rectangle(output, (x1, y1), (x2, y2), (0,255,0), 1)

# 顯示
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Generalized Hough Transform Result")
plt.axis("off")
result_path = os.path.join(curr_fold, "result", "base.png")     # 偵測結果路徑
plt.savefig(result_path, bbox_inches='tight', pad_inches=0.1)
plt.show()
