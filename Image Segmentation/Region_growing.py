import cv2
import numpy as np

THRESHOLD = 10 # intensity difference threshold
seed_point = None
def on_mouse(event, x, y, flags, param):
    global seed_point
    if event == cv2.EVENT_LBUTTONDOWN:
        seed_point = (x, y)
        print(f"Seed selected at: {seed_point}")
def region_growing(img, seed, threshold=THRESHOLD):
    h, w = img.shape
    visited = np.zeros((h, w), dtype=bool)
    region = np.zeros((h, w), dtype=np.uint8)
    seed_value = img[seed[1], seed[0]]
    stack = [seed]
    while stack:
        x, y = stack.pop()
        if visited[y, x]:
            continue
        visited[y, x] = True
        region[y, x] = 255
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                if abs(int(img[ny, nx]) - int(seed_value)) < threshold:
                    stack.append((nx, ny))
    return region
# Main program
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select image", filetypes=[("Image files", "*.jpg *.png *.bmp")])
    data = np.fromfile(file_path, dtype=np.uint8)
    color_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    if file_path == "" or img is None:
        raise FileNotFoundError("Please place image in the folder.")
    cv2.namedWindow("Input")
    cv2.setMouseCallback("Input", on_mouse)
    
    # Trackbar UI
    def on_trackbar(val):
        global THRESHOLD
        THRESHOLD = val
        print(f"Threshold = {THRESHOLD}")
    cv2.createTrackbar("Threshold", "Input", THRESHOLD, 100, on_trackbar)

while True:
    cv2.imshow("Input", img)
    key = cv2.waitKey(30)
    
    if seed_point is not None:
        grown = region_growing(img, seed_point, THRESHOLD)
        cv2.imshow("Region Growing Result", grown)
        seed_point = None
    
    if key == 27: # ESC to exit
        break
cv2.destroyAllWindows()