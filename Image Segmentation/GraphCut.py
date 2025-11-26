"""
Multi-label image segmentation via alpha-expansion (graph cut / max-flow).
GUI to open image and set K (number of labels), lambda (smoothness), max iter.

Dependencies:
pip install numpy pillow matplotlib scikit-learn PyMaxflow
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from sklearn.cluster import KMeans
import maxflow
import math
import matplotlib.pyplot as plt

# ---------------- utility functions ----------------
def image_to_array(pil_img, max_size=400):
    # Resize for speed / display if too large
    w, h = pil_img.size
    scale = min(1.0, max_size / max(w, h))
    if scale < 1.0:
        pil_img = pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    arr = np.array(pil_img).astype(np.float32) / 255.0
    return arr

def kmeans_init_labels(img_arr, K):
    h, w, c = img_arr.shape
    X = img_arr.reshape(-1, 3)
    km = KMeans(n_clusters=K, n_init=4, random_state=0)
    lbl = km.fit_predict(X)
    centers = km.cluster_centers_
    return lbl.reshape(h, w), centers

def compute_unary_costs(img_arr, centers):
    """
    Simple unary: squared Euclidean distance to cluster centers.
    Returns unaryCosts: shape (K, H, W)
    """
    h, w, _ = img_arr.shape
    K = centers.shape[0]
    unary = np.zeros((K, h, w), dtype=np.float32)
    for k in range(K):
        diff = img_arr - centers[k].reshape(1,1,3)
        unary[k] = np.sum(diff**2, axis=2)
    # normalize
    unary = unary / (np.max(unary) + 1e-12)
    return unary

def compute_beta(img_arr):
    """
    Beta used in contrast-sensitive pairwise term:
    beta = 1 / (2 * mean(||Ip - Iq||^2))
    computed over 4-neighborhood differences
    """
    h, w, _ = img_arr.shape
    diffs = []
    for y in range(h):
        for x in range(w):
            p = img_arr[y,x]
            if x+1 < w:
                diffs.append(np.sum((p - img_arr[y, x+1])**2))
            if y+1 < h:
                diffs.append(np.sum((p - img_arr[y+1, x])**2))
    diffs = np.array(diffs, dtype=np.float32)
    mean_d = np.mean(diffs) if diffs.size>0 else 1e-6
    beta = 1.0 / (2.0 * mean_d + 1e-12)
    return beta

def build_pairwise_weights(img_arr, lam):
    """
    Precompute pairwise weights for 4-neighbors:
    w_pq = lambda * exp(-beta * ||Ip - Iq||^2)
    Return as two arrays: right_weights (h,w-1) between (y,x)<->(y,x+1)
                           down_weights  (h-1,w) between (y,x)<->(y+1,x)
    """
    h, w, _ = img_arr.shape
    beta = compute_beta(img_arr)
    right = np.zeros((h, max(0, w-1)), dtype=np.float32)
    down  = np.zeros((max(0, h-1), w), dtype=np.float32)
    for y in range(h):
        for x in range(w-1):
            diff = np.sum((img_arr[y,x] - img_arr[y, x+1])**2)
            right[y,x] = lam * math.exp(-beta * diff)
    for y in range(h-1):
        for x in range(w):
            diff = np.sum((img_arr[y,x] - img_arr[y+1, x])**2)
            down[y,x] = lam * math.exp(-beta * diff)
    return right, down

# ---------------- alpha-expansion core ----------------
def alpha_expansion(img_arr, init_labels, unary, right_w, down_w, max_iters=10):
    """
    Perform alpha-expansion iterations.
    img_arr: HxWx3 (not used inside except for dims)
    init_labels: HxW current labeling
    unary: K x H x W unary costs
    right_w: H x (W-1) pairwise weights for horizontal edges
    down_w: (H-1) x W pairwise weights for vertical edges
    returns labels
    """
    h, w, _ = img_arr.shape
    K = unary.shape[0]
    labels = init_labels.copy()
    N = h * w

    # Precompute neighbor indices mapping linear index
    def idx(y,x): return y * w + x

    for it in range(max_iters):
        changed = 0
        # iterate all labels as alpha (can randomize order)
        for alpha in range(K):
            # Build graph for binary optimization: each pixel has binary choice:
            # 0 -> keep current label, 1 -> move to alpha
            g = maxflow.Graph[float](N, N*4)
            nodes = g.add_nodes(N)
            # set t-links (unary costs)
            for y in range(h):
                for x in range(w):
                    p = idx(y,x)
                    cur = labels[y,x]
                    cost_alpha = float(unary[alpha,y,x])
                    cost_keep  = float(unary[cur,y,x])
                    # t-link: source->p = cost_alpha, p->sink = cost_keep
                    g.add_tedge(p, cost_alpha, cost_keep)

            # add pairwise (smoothness Potts): if neighbors assigned different binary values, penalty w_pq
            # horizontal edges
            for y in range(h):
                for x in range(w-1):
                    p = idx(y,x)
                    q = idx(y,x+1)
                    w_pq = float(right_w[y,x])
                    if w_pq > 0:
                        # add undirected edge between p and q with capacity w_pq
                        g.add_edge(p, q, w_pq, w_pq)
            # vertical edges
            for y in range(h-1):
                for x in range(w):
                    p = idx(y,x)
                    q = idx(y+1,x)
                    w_pq = float(down_w[y,x])
                    if w_pq > 0:
                        g.add_edge(p, q, w_pq, w_pq)

            flow = g.maxflow()
            # get segmentation: 1 = source side (assign alpha), 0 = sink side (keep)
            for y in range(h):
                for x in range(w):
                    p = idx(y,x)
                    seg = g.get_segment(p)  # 0 = sink, 1 = source
                    if seg == 1:
                        if labels[y,x] != alpha:
                            labels[y,x] = alpha
                            changed += 1
            # end of alpha move
        # end of all alpha
        # stop if no change
        if changed == 0:
            break
    return labels

# ---------------- GUI ----------------
class GraphCutGUI:
    def __init__(self, master):
        self.master = master
        master.title("Multi-label GraphCut (alpha-expansion)")

        self.img_pil = None
        self.img_arr = None
        self.labels = None

        frm = tk.Frame(master)
        frm.pack(padx=6, pady=6)

        btn_open = tk.Button(frm, text="Open Image", command=self.open_image)
        btn_open.grid(row=0, column=0)

        tk.Label(frm, text="K (clusters)").grid(row=0, column=1)
        self.k_var = tk.IntVar(value=3)
        tk.Spinbox(frm, from_=2, to=10, textvariable=self.k_var, width=4).grid(row=0, column=2)

        tk.Label(frm, text="lambda (smooth)").grid(row=0, column=3)
        self.lam_var = tk.DoubleVar(value=50.0)
        tk.Entry(frm, textvariable=self.lam_var, width=6).grid(row=0, column=4)

        tk.Label(frm, text="max iters").grid(row=0, column=5)
        self.iter_var = tk.IntVar(value=5)
        tk.Spinbox(frm, from_=1, to=20, textvariable=self.iter_var, width=4).grid(row=0, column=6)

        btn_run = tk.Button(frm, text="Run GraphCut", command=self.run_graphcut)
        btn_run.grid(row=0, column=7, padx=6)

        btn_save = tk.Button(frm, text="Save Result", command=self.save_result)
        btn_save.grid(row=0, column=8, padx=6)

        # display
        self.left_label = tk.Label(master)
        self.left_label.pack(side='left', padx=4, pady=4)
        self.right_label = tk.Label(master)
        self.right_label.pack(side='left', padx=4, pady=4)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
        if not path:
            return
        self.img_pil = Image.open(path).convert('RGB')
        arr = image_to_array(self.img_pil, max_size=300)
        self.img_arr = arr
        self.show_pil(self.img_pil, self.left_label)

    def show_pil(self, pil, label):
        # resize for display
        display = pil.copy()
        display.thumbnail((300,300), Image.LANCZOS)
        tkimg = ImageTk.PhotoImage(display)
        label.imgtk = tkimg
        label.config(image=tkimg)

    def run_graphcut(self):
        if self.img_arr is None:
            messagebox.showwarning("No image", "Open an image first")
            return
        K = int(self.k_var.get())
        lam = float(self.lam_var.get())
        max_iters = int(self.iter_var.get())

        h, w, _ = self.img_arr.shape

        # init labels with k-means (color)
        init_lbl, centers = kmeans_init_labels(self.img_arr, K)
        unary = compute_unary_costs(self.img_arr, centers)
        right_w, down_w = build_pairwise_weights(self.img_arr, lam)

        # run alpha-expansion
        labels = alpha_expansion(self.img_arr, init_lbl, unary, right_w, down_w, max_iters=max_iters)
        self.labels = labels

        # visualization
        vis = visualize_labels(labels)
        vis_pil = Image.fromarray((vis*255).astype(np.uint8))
        self.show_pil(vis_pil, self.right_label)
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1); plt.imshow(self.img_arr); plt.title("Input"); plt.axis('off')
        plt.subplot(1,2,2); plt.imshow(vis); plt.title(f"GraphCut K={K}"); plt.axis('off')
        plt.show()

    def save_result(self):
        if self.labels is None:
            messagebox.showwarning("No result", "Run segmentation first")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")])
        if not path:
            return
        vis = visualize_labels(self.labels)
        Image.fromarray((vis*255).astype(np.uint8)).save(path)
        messagebox.showinfo("Saved", f"Saved to {path}")

def visualize_labels(lbl):
    h, w = lbl.shape
    labels = np.unique(lbl)
    k = len(labels)
    rng = np.random.RandomState(1)
    colors = (rng.rand(k,3) * 255).astype(np.uint8)
    out = np.zeros((h,w,3), dtype=np.uint8)
    for i,lab in enumerate(labels):
        out[lbl==lab] = colors[i]
    return out

# ---------------- run app ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = GraphCutGUI(root)
    root.mainloop()
