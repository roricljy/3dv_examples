import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Tk, Canvas, Button, Frame, Label, Entry
from PIL import Image, ImageTk
import os

gscale = 1.0
plt.rcParams.update({'font.size': 14*gscale})

# Global variables
image = None
canvas_widget = None

def normalize_angle(angle):
    return (angle + 2 * np.pi) % (2 * np.pi)

def erp_to_rect(erp_image, theta, hfov, vfov):
    erp_H, erp_W, channels = erp_image.shape
    f = erp_W / (2 * np.pi)
    rect_H = int(round(2 * f * np.tan(vfov / 2)))
    rect_W = int(round(2 * f * np.tan(hfov / 2)))
    rect_cx = rect_W / 2
    rect_cy = rect_H / 2

    rect_x, rect_y = np.meshgrid(np.arange(rect_W), np.arange(rect_H))
    xth = np.arctan((rect_x - rect_cx) / f)    
    xth_erp = normalize_angle(theta + xth)
    erp_x = (xth_erp * erp_W) / (2 * np.pi)
    yf = f / np.cos(xth)
    yth = np.arctan((rect_y - rect_cy) / yf)
    erp_y = (yth * erp_H) / np.pi + erp_H / 2
    
    erp_xi = np.round(erp_x).astype(np.int32)
    erp_yi = np.round(erp_y).astype(np.int32)
    
    erp_xi = np.clip(erp_xi, 0, erp_W - 1)
    erp_yi = np.clip(erp_yi, 0, erp_H - 1)
    rect_image = np.zeros((rect_H, rect_W, channels), dtype=erp_image.dtype)
    rect_image[:, :] = erp_image[erp_yi, erp_xi]

    return rect_image

def convert_erp():
    global image
    theta = np.radians(float(theta_entry.get()))
    hfov = np.radians(float(hfov_entry.get()))
    vfov = np.radians(float(vfov_entry.get()))
    rect_image = erp_to_rect(image, theta, hfov, vfov)
    display_image_on_canvas(rect_image)    

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

def exit_app():
    root.destroy()

if __name__ == "__main__":
    image = plt.imread("sample_erp.png")
    if image is None:
        print("Image not found!")
        exit(-1)

    root = Tk()
    root.title("Distortion Correction")

    canvas = Canvas(root)
    canvas.pack()

    button_frame = Frame(root)
    button_frame.pack()
    
    tk.Label(button_frame, text="Theta (degrees):").grid(row=0, column=0)
    theta_entry = tk.Entry(button_frame)
    theta_entry.grid(row=0, column=1)
    theta_entry.insert(0, "180")

    tk.Label(button_frame, text="HFOV (degrees):").grid(row=1, column=0)
    hfov_entry = tk.Entry(button_frame)
    hfov_entry.grid(row=1, column=1)
    hfov_entry.insert(0, "120")

    tk.Label(button_frame, text="VFOV (degrees):").grid(row=2, column=0)
    vfov_entry = tk.Entry(button_frame)
    vfov_entry.grid(row=2, column=1)
    vfov_entry.insert(0, "90")    

    load_btn = Button(button_frame, text="Input Image", command=show_image)
    load_btn.grid(row=3, column=0, padx=5, pady=5)
    
    undistort_zhang_btn = Button(button_frame, text="Rectify ERP", command=convert_erp)
    undistort_zhang_btn.grid(row=3, column=1, padx=5, pady=5)

    exit_btn = Button(button_frame, text="Exit", command=exit_app)
    exit_btn.grid(row=3, column=5, padx=5, pady=5)

    root.mainloop()
