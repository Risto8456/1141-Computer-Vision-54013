# GHT_with_rotation_and_scaling.py
# 支援旋轉與縮放的 Generalized Hough Transform (純 numpy + cv2)
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
# 1.5 (選填) Reference 影像前處理：縮放 + 旋轉（無裁切）
# ==========================================================

# ---- 可調參數 ----
scale_factor = 0.5    # 影像縮放倍率（0.1 ~ 10），1.0 表示不縮放
pre_rotate_angle = 60 # 旋轉角度（度）0 表示不旋轉

# ---- Step 1：影像縮放（不強制）----
if scale_factor != 1.0:
    if not (0.1 <= scale_factor <= 10.0):
        raise ValueError("縮放倍率必須在 0.1 ~ 10 之間")

    new_w = int(reference.shape[1] * scale_factor)
    new_h = int(reference.shape[0] * scale_factor)

    reference = cv2.resize(reference, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    ref_gray = cv2.resize(ref_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    print(f"[1.5] Reference 已縮放倍率：{scale_factor}")
else:
    print("[1.5] Reference 未縮放")

# ---- Step 2：旋轉（自動擴張畫布，避免裁切）----
if pre_rotate_angle != 0:
    angle = np.radians(pre_rotate_angle)
    (h, w) = ref_gray.shape

    # 計算旋轉後能完整容納的 bounding box（不縮放內容）
    new_w = int(abs(w*np.cos(angle)) + abs(h*np.sin(angle)))
    new_h = int(abs(w*np.sin(angle)) + abs(h*np.cos(angle)))

    # 旋轉矩陣以原中心為基準（但輸出 canvas 是新的）
    M = cv2.getRotationMatrix2D((w/2, h/2), pre_rotate_angle, 1.0)

    # 平移補償：把內容移到新 canvas 的中心
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    reference = cv2.warpAffine(reference, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
    ref_gray = cv2.warpAffine(ref_gray, M, (new_w, new_h), flags=cv2.INTER_LINEAR)

    print(f"[1.5] Reference 已旋轉 {pre_rotate_angle} 度（自動擴張避免裁切）")
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

NBINS = 60  # 方向量化數（template edge orientation bins）
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
# 4. 在 Reference 上進行投票（加入旋轉 + 縮放維度）
# ==========================================================
H, W = ref_edges.shape

# Rotation bins
N_ROT = 36
rot_angles = np.linspace(-np.pi, np.pi, N_ROT, endpoint=False)

# Scale bins（假設要搜尋多個縮放倍率）
scale_factors = [1.0, 0.8, 1.2]  # 可以自行修改

# 4D accumulator: (H, W, N_ROT, N_SCALE)
accumulator = np.zeros((H, W, N_ROT, len(scale_factors)), dtype=np.int32)

ys_r, xs_r = np.where(ref_edges > 0)
print("Reference 邊緣點數：", len(xs_r), "，開始投票（含旋轉+縮放）...")

for (x, y) in zip(xs_r, ys_r):
    phi_r = ref_dir[y, x]

    for ridx, psi in enumerate(rot_angles):
        phi_t_needed = phi_r - psi
        bin_t = quantize_angle(phi_t_needed)

        entries = R_table[bin_t]
        if not entries:
            continue

        for sidx, s in enumerate(scale_factors):
            for (r, alpha) in entries:
                r_scaled = r * s               # 將範本半徑放大或縮小
                alpha_rot = alpha + psi        # 旋轉向量
                xc_hat = int(round(x + r_scaled * np.cos(alpha_rot)))
                yc_hat = int(round(y + r_scaled * np.sin(alpha_rot)))

                if 0 <= xc_hat < W and 0 <= yc_hat < H:
                    accumulator[yc_hat, xc_hat, ridx, sidx] += 1

print("投票完成")

# ==========================================================
# 5. 找出累加器最大值（中心 + 旋轉角 + 縮放倍率）
# ==========================================================
flat_idx = accumulator.argmax()
cy, cx, r_idx, s_idx = np.unravel_index(flat_idx, accumulator.shape)
maxVal = accumulator[cy, cx, r_idx, s_idx]

best_angle = rot_angles[r_idx]
best_scale = scale_factors[s_idx]

print("偵測到中心點位置：", (cx, cy))
print("偵測到旋轉角（度）：", np.degrees(best_angle))
print("偵測到縮放倍率：", best_scale)
print("票數：", int(maxVal))

# ==========================================================
# 6. 畫出偵測結果（旋轉框 + 縮放）
# ==========================================================
th, tw = template.shape[:2]
tw_scaled = tw * best_scale
th_scaled = th * best_scale

half_w = tw_scaled / 2.0
half_h = th_scaled / 2.0

# Template 四個角
corners = np.array([
    [-half_w, -half_h],
    [ half_w, -half_h],
    [ half_w,  half_h],
    [-half_w,  half_h]
])

# 旋轉矩陣
ca = np.cos(best_angle)
sa = np.sin(best_angle)
R = np.array([[ca, -sa],
              [sa,  ca]])

rotated_corners = (corners @ R.T)
rotated_corners[:, 0] += cx
rotated_corners[:, 1] += cy

pts = np.int32(rotated_corners.reshape((-1, 1, 2)))

output = reference.copy()
cv2.polylines(output, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

cv2.circle(output, (cx, cy), 3, (0, 0, 255), -1)
angle_text = f"{int(np.degrees(best_angle))}deg / scale:{best_scale:.2f}"
cv2.putText(output, angle_text, (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1, cv2.LINE_AA)

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("GHT with rotation + scale - Result")
plt.axis("off")
result_path = os.path.join(curr_fold, "result", f"scale{scale_factor}_angle{pre_rotate_angle}.png")     # 偵測結果路徑
plt.savefig(result_path, bbox_inches='tight', pad_inches=0.1)
plt.show()
