import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

img = None
win = "Interactive SLIC"
region_size = 20
compactness = 10
iterations = 10
# -----------------------------
# SLIC function
# -----------------------------
def run_slic():
    global img, region_size, compactness, iterations
    if img is None:
        return np.zeros((200, 400, 3), dtype=np.uint8)
    slic = cv2.ximgproc.createSuperpixelSLIC(
        img,
        algorithm=cv2.ximgproc.SLICO,
        region_size=max(5, region_size),
        ruler=max(1, compactness)
    )
    slic.iterate(max(1, iterations))
    mask = slic.getLabelContourMask()
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(img, img, mask=mask_inv)
    return result

# -----------------------------
# Trackbar callback functions
# -----------------------------
def on_region_size(v):
    global region_size
    region_size = max(5, v)
def on_compactness(v):
    global compactness
    compactness = max(1, v)
def on_iterations(v):
    global iterations
    iterations = max(1, v)
def open_file_dialog():
    global img
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="選擇一張影像",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")]
    )
    root.destroy()
    if not filepath:
        return
    try:
        # 使用 pathlib 讀取檔案 bytes，再用 cv2.imdecode 支援中文/非 ASCII 路徑
        data = Path(filepath).read_bytes()
        arr = np.frombuffer(data, dtype=np.uint8)
        temp = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if temp is None:
            print("讀檔失敗（cv2.imdecode 回傳 None）")
        else:
            img = temp
            print("成功讀取：", filepath)
    except Exception as e:
        print("讀檔失敗：", e)

# -----------------------------
# Main
# -----------------------------
cv2.namedWindow(win)
# 建立 Trackbars
cv2.createTrackbar("Region Size", win, region_size, 100, on_region_size)
cv2.createTrackbar("Compactness", win, compactness, 50, on_compactness)
cv2.createTrackbar("Iterations", win, iterations, 20, on_iterations)
print("=== 操作說明 ===")
print("按 O 開啟影像檔案")
print("按 Q 離開程式")
print("================\n")
while True:
    result = run_slic()
    cv2.imshow(win, result)
    key = cv2.waitKey(30) & 0xFF
    # 按 O 開啟檔案
    if key == ord('o') or key == ord('O'):
        open_file_dialog()
    # 按 Q 離開
    if key == ord('q') or key == ord('Q'):
        break
cv2.destroyAllWindows()
