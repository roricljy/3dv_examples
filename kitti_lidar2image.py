import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tkinter import Tk, Canvas, Button, Frame, Label, Entry
from PIL import Image, ImageDraw, ImageTk
import os

gscale = 1.8 if "ANDROID_STORAGE" in os.environ else 1
plt.rcParams.update({'font.size': 14*gscale})

# Global variables
image = None
lidar_points = None
calib_data = None
canvas_widget = None

def display_image_on_canvas(canvas_img):
    global canvas_widget, gscale

    if canvas_img.dtype in [np.float32, np.float64] and canvas_img.max() <= 1.0:
        canvas_img = (canvas_img * 255).clip(0, 255).astype(np.uint8)
    else:
        canvas_img = canvas_img.astype(np.uint8)

    height, width = canvas_img.shape[:2]
    new_width, new_height = int(width * gscale), int(height * gscale)
    canvas.config(width=new_width, height=new_height)

    img_pil = Image.fromarray(canvas_img)
    img_pil = img_pil.resize((new_width, new_height), Image.NEAREST)

    img_tk = ImageTk.PhotoImage(img_pil)
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image_tk = img_tk

def display_lidar_on_canvas(points):
    img_size = 500  # image size
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    x_scaled = ((points[:, 0] - x_min) / (x_max - x_min) * (img_size - 1)).astype(int)
    y_scaled = ((points[:, 1] - y_min) / (y_max - y_min) * (img_size - 1)).astype(int)
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    z_scaled = ((points[:, 2] - z_min) / (z_max - z_min) * 255).astype(np.uint8)    
    for i in range(len(points)):
        img[y_scaled[i], x_scaled[i]] = [z_scaled[i], 255 - z_scaled[i], 128] 
    
    display_image_on_canvas(img)

def project_lidar_to_image(lidar_data, calibration_data, image):
    P2 = calibration_data['P2'].reshape(3, 4)
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = calibration_data['R0_rect'].reshape(3, 3)
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :] = calibration_data['Tr_velo_to_cam'].reshape(3, 4)

    lidar_points = np.hstack((lidar_data[:, :3], np.ones((lidar_data.shape[0], 1))))
    points_rect_3d = (R0_rect @ Tr_velo_to_cam @ lidar_points.T).T
    points_rect_3d = points_rect_3d[points_rect_3d[:, 2] > 0]

    pixels_cam2_2d = (P2 @ points_rect_3d.T).T
    pixels_cam2_2d[:, 0] /= pixels_cam2_2d[:, 2]
    pixels_cam2_2d[:, 1] /= pixels_cam2_2d[:, 2]

    depth = points_rect_3d[:, 2]
    depth_min, depth_max = depth.min(), depth.max()
    depth_normalized = np.clip((depth - depth_min) / (depth_max - depth_min) * 1.4, 0, 1)

    colormap = plt.colormaps.get_cmap("turbo")
    colors = (colormap(depth_normalized)[:, :3] * 255).astype(np.uint8)

    if image.dtype in [np.float32, np.float64] and image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)

    image_pil = Image.fromarray(image_uint8)
    draw = ImageDraw.Draw(image_pil)

    for i, point in enumerate(pixels_cam2_2d):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            color = tuple(colors[i])
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=color)

    return np.array(image_pil, dtype=image.dtype)
    
def show_image():
    global image
    display_image_on_canvas(image)

def read_kitti_lidar(file_path):
    scan = np.fromfile(file_path, dtype=np.float32)
    points = scan.reshape((-1, 4))
    return points
    
def read_calib_file(filepath):
    calib_data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if ':' in line:
                key, value = line.split(':', 1)
                calib_data[key] = np.array([float(x) for x in value.split()])
    return calib_data
    
def show_lidar():
    global lidar_points
    display_lidar_on_canvas(lidar_points) 

def show_mapping():
    global image, lidar_points, calib_data
    projected_image = project_lidar_to_image(lidar_points, calib_data, image)
    display_image_on_canvas(projected_image)

def exit_app():
    root.destroy()

if __name__ == "__main__":
    # load image
    image = plt.imread("kitti_image.png")
    if image is None:
        print("Image not found!")
        exit(-1)

    # load lidar
    lidar_file_path = 'kitti_lidar.bin'
    lidar_points = read_kitti_lidar(lidar_file_path)
    
    # load calibration data
    calib_file_path = 'kitti_calib.txt'
    calib_data = read_calib_file(calib_file_path)    

    root = Tk()
    root.title("Distortion Correction")

    canvas = Canvas(root)
    canvas.pack()

    button_frame = Frame(root)
    button_frame.pack()

    load_image_btn = Button(button_frame, text="Show Image", command=show_image)
    load_image_btn.grid(row=0, column=0, padx=5, pady=5)
    
    load_lidar_btn = Button(button_frame, text="Show LiDAR", command=show_lidar)
    load_lidar_btn.grid(row=0, column=1, padx=5, pady=5)

    mapping_btn = Button(button_frame, text="Show Mapping", command=show_mapping)
    mapping_btn.grid(row=0, column=2, padx=5, pady=5)

    exit_btn = Button(button_frame, text="Exit", command=exit_app)
    exit_btn.grid(row=0, column=3, padx=5, pady=5)

    root.mainloop()
