import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Canvas, Button, Frame, Label, Entry
from PIL import Image, ImageTk
import os

gscale = 1.8 if "ANDROID_STORAGE" in os.environ else 1
plt.rcParams.update({'font.size': 14*gscale})

# Global variables
image = None
canvas_widget = None
focal = np.array([436.8,436.8], dtype=np.float32)  # fx, fy
center = np.array([480, 270], dtype=np.float32) # cx, cy
dist_coeffs = np.array([-0.226141, 0.045470, 0.000183, -0.000036, 0], dtype=np.float32) # k1, k2, p1, p2, k3
w_value = 0.00207  # Default w value

def apply_distortion_equidistance(points, center, w, dst_center):
    if w<=0:
        return points
    ru = np.linalg.norm(points - center, axis=2)
    rd = np.where(ru != 0, np.arctan(w * ru) / w, 0)
    distorted = np.where(ru[..., None] != 0, (points - center) * (rd / ru)[..., None] + dst_center, points)
    return distorted

def apply_distortion_zhang(points, center, focal, dist_coeffs, dst_center):
    k1, k2 = dist_coeffs[:2]
    normalized_points = (points - center) / focal
    ru = np.linalg.norm(normalized_points, axis=2)
    rd = ru + k1 * ru**3 + k2 * ru**5
    rd = np.where(ru != 0, rd, 0)
    distorted = np.where(ru[..., None] != 0, (points - center) * (rd / ru)[..., None] + dst_center, points)
    return distorted

def display_image_on_canvas(canvas_img):
    global canvas_widget, gscale
    height, width = canvas_img.shape[:2]
    new_width, new_height = int(width * gscale), int(height * gscale)
    canvas.config(width=new_width, height=new_height)
    canvas_img = (canvas_img * 255).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(canvas_img).resize((new_width, new_height), Image.NEAREST)
    img_tk = ImageTk.PhotoImage(img_pil)
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image_tk = img_tk

def show_image():
    global image
    display_image_on_canvas(image)

def undistort_image_equidistance():
    global image, center, w_value
    try:
        w = float(w_entry.get())
    except ValueError:
        w = w_value
    height, width, channels = image.shape
    new_height, new_width = int(height * 1.4), int(width * 1.4)
    new_center = np.array([new_width / 2, new_height / 2])
    new_image = np.zeros((new_height, new_width, channels), dtype=image.dtype)
    y, x = np.meshgrid(np.arange(new_height), np.arange(new_width), indexing='ij')
    points = np.stack([x, y], axis=-1)
    distorted_points = apply_distortion_equidistance(points, new_center, w, center).astype(int)
    mask = (0 <= distorted_points[:, :, 0]) & (distorted_points[:, :, 0] < width) & (0 <= distorted_points[:, :, 1]) & (distorted_points[:, :, 1] < height)
    new_image[mask] = image[distorted_points[mask, 1], distorted_points[mask, 0]]
    display_image_on_canvas(new_image)

def undistort_image_zhang():
    global image, center, focal, dist_coeffs
    height, width, channels = image.shape
    new_height, new_width = int(height * 1.4), int(width * 1.4)
    new_center = np.array([new_width / 2, new_height / 2])
    new_image = np.zeros((new_height, new_width, channels), dtype=image.dtype)
    y, x = np.meshgrid(np.arange(new_height), np.arange(new_width), indexing='ij')
    points = np.stack([x, y], axis=-1)
    distorted_points = apply_distortion_zhang(points, new_center, focal, dist_coeffs, center).astype(int)
    mask = (0 <= distorted_points[:, :, 0]) & (distorted_points[:, :, 0] < width) & (0 <= distorted_points[:, :, 1]) & (distorted_points[:, :, 1] < height)
    new_image[mask] = image[distorted_points[mask, 1], distorted_points[mask, 0]]
    display_image_on_canvas(new_image)

def exit_app():
    root.destroy()

if __name__ == "__main__":
    image = plt.imread("sample_radial.png")
    if image is None:
        print("Image not found!")
        exit(-1)

    root = Tk()
    root.title("Distortion Correction")

    canvas = Canvas(root)
    canvas.pack()

    button_frame = Frame(root)
    button_frame.pack()

    load_btn = Button(button_frame, text="Load Image", command=show_image)
    load_btn.grid(row=0, column=0, padx=5, pady=5)
    
    undistort_zhang_btn = Button(button_frame, text="Undistort Zhang", command=undistort_image_zhang)
    undistort_zhang_btn.grid(row=0, column=1, padx=5, pady=5)

    Label(button_frame, text="w:").grid(row=0, column=2, padx=5, pady=5)
    w_entry = Entry(button_frame, width=10)
    w_entry.grid(row=0, column=3, padx=5, pady=5)
    w_entry.insert(0, str(w_value))

    undistort_equidistance_btn = Button(button_frame, text="Undistort Equidistance", command=undistort_image_equidistance)
    undistort_equidistance_btn.grid(row=0, column=4, padx=5, pady=5)

    exit_btn = Button(button_frame, text="Exit", command=exit_app)
    exit_btn.grid(row=0, column=5, padx=5, pady=5)

    root.mainloop()
