import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tkinter import Tk, Canvas, Button, Frame, Label, Entry
from PIL import Image, ImageTk
import os

gscale = 1.8 if "ANDROID_STORAGE" in os.environ else 1

# Camera parameters
focal       = np.array([461.9, 462.6], dtype=np.float32)
center      = np.array([324.1, 190.5], dtype=np.float32)
dist_coeffs = np.array([-0.413928, 0.162300, -0.001240, -0.000541, 0.0], dtype=np.float32)

K = np.array([[focal[0],       0, center[0]],
              [       0, focal[1], center[1]],
              [       0,        0,         1]], dtype=np.float64)
dist = dist_coeffs.astype(np.float64)

image        = None
canvas_widget = None


def pan_tilt_roll_to_R(pan, tilt, roll):
    return (Rotation.from_euler('y', pan) *
            Rotation.from_euler('x', -tilt) *
            Rotation.from_euler('z', roll)).as_matrix()

def R_to_pan_tilt_roll(R):
    pan, tilt, roll =  Rotation.from_matrix(R).as_euler('yxz')  # (pan, tilt, roll)
    return pan, -tilt, roll
    
# Fixed: actual camera pose of the input image
CAMERA_TILT_DEG = -46.6
R_current = pan_tilt_roll_to_R(pan=0.0,
                                tilt=np.deg2rad(CAMERA_TILT_DEG),
                                roll=0.0)

vtilt_default = 0.0  # default virtual tilt (deg)

def pixel2virtualpixel(pts, K, dist, R_current, R_virtual):
    pts_ud = cv2.undistortPoints(
        pts.reshape(-1, 1, 2).astype(np.float32), K, dist, P=K
    ).reshape(-1, 2)

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    rays = np.stack([(pts_ud[:,0]-cx)/fx,
                     (pts_ud[:,1]-cy)/fy,
                     np.ones(len(pts_ud))], axis=1)

    R = R_virtual @ R_current.T
    rays_v = (R @ rays.T).T

    x = rays_v[:,0] / rays_v[:,2] * fx + cx
    y = rays_v[:,1] / rays_v[:,2] * fy + cy
    return np.stack([x, y], axis=1)


def virtual2imagepixel(pts, K, dist, R_current, R_virtual):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    rays = np.stack([(pts[:,0]-cx)/fx,
                     (pts[:,1]-cy)/fy,
                     np.ones(len(pts))], axis=1)

    R = R_current @ R_virtual.T
    rays_src = (R @ rays.T).T

    pts_2d, _ = cv2.projectPoints(
        rays_src.reshape(-1, 1, 3).astype(np.float32),
        np.zeros(3), np.zeros(3), K, dist
    )
    return pts_2d.reshape(-1, 2)

def warp_to_virtual_view(image, K, dist, R_current, R_virtual,
                       dst_scale=-1.0, dst_scale_max=3.0):
    sh, sw = image.shape[:2]
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    _, tilt, _ = R_to_pan_tilt_roll(R_current)

    virtual_y = cy + fy * np.tan(np.pi / 2 + tilt)
    visible_y1, visible_y2 = 0.0, float(sh - 1)
    if cy < virtual_y < visible_y2:
        visible_y2 = virtual_y
    elif visible_y1 < virtual_y < cy:
        visible_y1 = virtual_y
    visible_cy = (visible_y1 + visible_y2) / 2.0

    boundary_pts = np.array([
        [0,      visible_cy],
        [sw - 1, visible_cy],
        [sw / 2, visible_y1],
        [sw / 2, visible_y2],
    ])

    vpts = pixel2virtualpixel(boundary_pts, K, dist, R_current, R_virtual)

    x_scale = (vpts[1, 0] - vpts[0, 0]) / sw
    y_scale = (vpts[3, 1] - vpts[2, 1]) / sh
    scale_auto = max(x_scale, y_scale)

    if dst_scale <= 0:
        dst_scale = min(scale_auto, dst_scale_max)
    else:
        dst_scale = min(dst_scale, scale_auto)

    start_x = int(sw / 2 - sw / 2 * dst_scale)
    start_y = int(vpts[3, 1] - sh * dst_scale)
    start_y = int(vpts[2, 1])

    dw = int(sw * dst_scale + 0.5)
    dh = int(sh * dst_scale + 0.5)

    u, v = np.meshgrid(np.arange(dw, dtype=np.float64),
                       np.arange(dh, dtype=np.float64))
    virtual_coords = np.stack([u.ravel() + start_x,
                                v.ravel() + start_y], axis=1)

    src_coords = virtual2imagepixel(virtual_coords, K, dist, R_current, R_virtual)

    map_x = src_coords[:, 0].reshape(dh, dw).astype(np.float32)
    map_y = src_coords[:, 1].reshape(dh, dw).astype(np.float32)

    return cv2.remap(image, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0)

def display_image_on_canvas_old(img):
    global canvas
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    new_w, new_h = int(w * gscale), int(h * gscale)
    canvas.config(width=new_w, height=new_h)

    img_pil = Image.fromarray(img).resize((new_w, new_h), Image.NEAREST)
    img_tk  = ImageTk.PhotoImage(img_pil)
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image_tk = img_tk  # prevent GC

def display_image_on_canvas(img):
    global canvas, image
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    oh, ow = image.shape[:2]
    img_pil = Image.fromarray(img).resize((ow, oh), Image.BILINEAR)

    new_w, new_h = int(ow * gscale), int(oh * gscale)
    canvas.config(width=new_w, height=new_h)

    img_pil = img_pil.resize((new_w, new_h), Image.NEAREST)
    img_tk  = ImageTk.PhotoImage(img_pil)
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image_tk = img_tk

def show_original():
    display_image_on_canvas(image)

def apply_warp():
    try:
        vtilt_deg = float(tilt_entry.get())
    except ValueError:
        vtilt_deg = vtilt_default

    pan, _, roll = R_to_pan_tilt_roll(R_current)
    R_virtual = pan_tilt_roll_to_R(pan,
                                   np.deg2rad(vtilt_deg),
                                   roll)
    result = warp_to_virtual_view(image, K, dist, R_current, R_virtual)
    display_image_on_canvas(result)


def save_result():
    try:
        vtilt_deg = float(tilt_entry.get())
    except ValueError:
        vtilt_deg = vtilt_default

    pan, _, roll = R_to_pan_tilt_roll(R_current)
    R_virtual = pan_tilt_roll_to_R(pan,
                                   np.deg2rad(vtilt_deg),
                                   roll)
    result = warp_to_virtual_view(image, K, dist, R_current, R_virtual)
    cv2.imwrite("virtual_view.png", result)
    print("Saved: virtual_view.png")


def exit_app():
    root.destroy()

if __name__ == "__main__":
    image = cv2.imread("sample_distorted.png")
    if image is None:
        print("Image not found!")
        exit(-1)

    root = Tk()
    root.title("Warp to Virtual View")

    canvas = Canvas(root)
    canvas.pack()

    button_frame = Frame(root)
    button_frame.pack()

    Button(button_frame, text="Show Original",
           command=show_original).grid(row=0, column=0, padx=5, pady=5)

    Label(button_frame, text="Tilt (deg):").grid(row=0, column=1, padx=5, pady=5)
    tilt_entry = Entry(button_frame, width=8)
    tilt_entry.grid(row=0, column=2, padx=5, pady=5)
    tilt_entry.insert(0, str(vtilt_default))

    Button(button_frame, text="Warp Virtual View",
           command=apply_warp).grid(row=0, column=3, padx=5, pady=5)

    Button(button_frame, text="Save",
           command=save_result).grid(row=0, column=4, padx=5, pady=5)

    Button(button_frame, text="Exit",
           command=exit_app).grid(row=0, column=5, padx=5, pady=5)

    show_original()
    root.mainloop()