# GHT_with_rotation.py
# 支援旋轉的 Generalized Hough Transform (純 numpy + cv2)
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm   # 進度條

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
# 1.5 (選填) 精準旋轉 Reference 影像（自動擴張避免裁切）
# ==========================================================
pre_rotate_angle = 45  # <-- 設定角度

if pre_rotate_angle != 0:
    angle = np.radians(pre_rotate_angle)
    (h, w) = ref_gray.shape

    # 計算旋轉後 bounding box 尺寸
    new_w = int(abs(w*np.cos(angle)) + abs(h*np.sin(angle)))
    new_h = int(abs(w*np.sin(angle)) + abs(h*np.cos(angle)))

    # 產生旋轉矩陣（注意要把中心移到新大小中）
    M = cv2.getRotationMatrix2D((w/2, h/2), pre_rotate_angle, 1.0)

    # 平移修正（避免裁切）
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # 執行旋轉
    reference = cv2.warpAffine(reference, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
    ref_gray = cv2.warpAffine(ref_gray, M, (new_w, new_h), flags=cv2.INTER_LINEAR)

    print(f"[1.5] Reference 已精準旋轉 {pre_rotate_angle} 度（無裁切）")
else:
    print("[1.5] Reference 未旋轉")


# ==========================================================
# 2. Canny 邊緣 + Sobel 計算梯度方向
# ==========================================================
temp_edges = cv2.Canny(temp_gray, 80, 150)
ref_edges  = cv2.Canny(ref_gray, 80, 150)

def gradient_direction(img_gray):
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    directions = np.arctan2(gy, gx)  # range (-pi, pi)
    return directions

temp_dir = gradient_direction(temp_gray)
ref_dir  = gradient_direction(ref_gray)

# ==========================================================
# 3. 建立 R-table（保留每個範本邊緣的 (r, alpha)）並以方向量化索引
# ==========================================================
h, w = temp_edges.shape
xc, yc = w // 2, h // 2  # 範本中心

NBINS = 120  # 方向量化數（template edge orientation bins）
def quantize_angle(theta, nbins=NBINS):
    # theta ∈ (-pi, pi) -> 0..nbins-1
    bin_id = int(((theta + np.pi) / (2*np.pi)) * nbins)
    return bin_id % nbins

# R-table: list of lists，每個 bin 儲存 (r, alpha, phi_t) 或 (r, alpha)
R_table = [[] for _ in range(NBINS)]

ys_t, xs_t = np.where(temp_edges > 0)
for (x, y) in zip(xs_t, ys_t):
    phi = temp_dir[y, x]             # 範本邊緣方向
    bin_id = quantize_angle(phi)

    dx = xc - x
    dy = yc - y
    r = np.hypot(dx, dy)
    alpha = np.arctan2(dy, dx)  # 從 edge 指到中心的角度 (range -pi..pi)

    # store r and alpha (polar vector from edge point to template center)
    R_table[bin_id].append((r, alpha))

print("R-table 建立完成，範本邊緣點數：", len(xs_t))

# ==========================================================
# 4. 在 Reference 上進行投票（加入旋轉維度）
# ==========================================================
H, W = ref_edges.shape

# Rotation bins: 要檢查的離散旋轉角度（弧度）
N_ROT = 120
rot_angles = np.linspace(-np.pi, np.pi, N_ROT, endpoint=False)

# 3D accumulator: (H, W, N_ROT)
accumulator = np.zeros((H, W, N_ROT), dtype=np.int32)

ys_r, xs_r = np.where(ref_edges > 0)
num_ref_edges = len(xs_r)
print("Reference 邊緣點數:", num_ref_edges, "開始投票（含旋轉）...")

# 預先計算 cos/sin
cos_rot = np.cos(rot_angles)
sin_rot = np.sin(rot_angles)

# tqdm 進度條（外層）
for (x, y, phi_r) in tqdm(zip(xs_r, ys_r, ref_dir[ys_r, xs_r]),
                           total=num_ref_edges,
                           desc="Voting (ref edges)",
                           ncols=80):

    # 每個離散旋轉角 psi
    for ridx, psi in enumerate(rot_angles):
        phi_t_needed = phi_r - psi
        bin_t = quantize_angle(phi_t_needed)

        entries = R_table[bin_t]
        if not entries:
            continue

        # entries: 多筆 (r, alpha)
        for (r, alpha) in entries:
            alpha_rot = alpha + psi
            xc_hat = int(round(x + r * np.cos(alpha_rot)))
            yc_hat = int(round(y + r * np.sin(alpha_rot)))

            if 0 <= xc_hat < W and 0 <= yc_hat < H:
                accumulator[yc_hat, xc_hat, ridx] += 1

print("投票完成")

# ==========================================================
# 5. 找出累加器最大值（偵測結果：中心 + 旋轉角）
# ==========================================================
# 找最大值的位置與旋轉 index
flat_idx = accumulator.argmax()
cy, cx, r_idx = np.unravel_index(flat_idx, accumulator.shape)
maxVal = accumulator[cy, cx, r_idx]
best_angle = rot_angles[r_idx]

print("偵測到中心點位置：", (cx, cy))
print("偵測到旋轉角（弧度）:", best_angle, "（度）:", np.degrees(best_angle))
print("票數：", int(maxVal))

# ==========================================================
# 6. 畫出偵測結果（旋轉框） - 使用範本大小並旋轉
# ==========================================================
th, tw = template.shape[:2]
# 範本四個角相對於範本中心的座標（以 x 向右，y 向下）
half_w = tw / 2.0
half_h = th / 2.0
corners = np.array([
    [-half_w, -half_h],
    [ half_w, -half_h],
    [ half_w,  half_h],
    [-half_w,  half_h]
])  # shape (4,2)  (x, y)

# 旋轉矩陣（注意：標準數學座標，y 向下會影響視覺，但用同一方式旋轉即可）
ca = np.cos(best_angle)
sa = np.sin(best_angle)
R = np.array([[ca, -sa],
              [sa,  ca]])

rotated_corners = (corners @ R.T)  # shape (4,2)
# 平移到偵測到的中心 (cx, cy)
rotated_corners[:, 0] += cx
rotated_corners[:, 1] += cy

# 產出可畫的整數座標（注意 cv2 期待 (x,y) 型式）
pts = np.int32(rotated_corners.reshape((-1, 1, 2)))

output = reference.copy()
# 畫多邊形（旋轉框）
cv2.polylines(output, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

# 在中心畫一個小圓點，並顯示角度
cv2.circle(output, (cx, cy), 3, (0, 0, 255), -1)
if (isinstance(pre_rotate_angle, int)):
    angle_text = f"{int(np.round(np.degrees(best_angle)))}deg"
else:
    angle_text = f"{np.degrees(best_angle)}deg"
cv2.putText(output, angle_text, (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1, cv2.LINE_AA)

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("GHT with rotation - Result")
plt.axis("off")
result_path = os.path.join(curr_fold, "result", f"angle{pre_rotate_angle}.png")     # 偵測結果路徑
plt.savefig(result_path, bbox_inches='tight', pad_inches=0.1)
plt.show()
