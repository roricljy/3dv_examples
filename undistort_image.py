import numpy as np
import math
import matplotlib.pyplot as plt
from tkinter import Tk, Canvas, Button, Frame
from PIL import Image, ImageTk
import os
gscale = 2.5 if "ANDROID_STORAGE" in os.environ else 1
plt.rcParams.update({'font.size': 14*gscale})

# Global variables
image = None
canvas_widget = None
focal = np.array([461.9, 462.6], dtype=np.float32)  # fx, fy
center = np.array([324.1, 190.5], dtype=np.float32) # cx, cy
dist_coeffs = np.array([-0.413928, 0.162300, -0.001240, -0.000541, 0.0], dtype=np.float32) # k1, k2, p1, p2, k3

# Function to apply image distortion
def apply_distortion(point, center, focal, dist_coeffs):
    if np.array_equal(point, center):
        return point
    k1 = dist_coeffs[0]
    k2 = dist_coeffs[1]
    normalized_point = (point - center) / focal
    ru = np.linalg.norm(normalized_point)
    if ru == 0:
        return point    
    rd = ru + k1*ru**3 + k2*ru**5
    distorted = (point - center) * rd / ru + center
    return distorted

# Function to apply image distortion (enlarged)
def apply_distortion_enlarge(point, center, focal, dist_coeffs, dst_center):
    if np.array_equal(point, center):
        return point
    k1 = dist_coeffs[0]
    k2 = dist_coeffs[1]
    normalized_point = (point - center) / focal
    ru = np.linalg.norm(normalized_point)
    if ru == 0:
        return point    
    rd = ru + k1*ru**3 + k2*ru**5
    distorted = (point - center) * rd / ru + dst_center
    return distorted
    
# Function to undistort image distrotion
def undistort_point(point, focal, center, dist_coeffs, max_iterations=10, learning_rate = 0.2, epsilon=1e-6):
    fx, fy = focal[0], focal[1]  # Focal lengths
    cx, cy = center[0], center[1]  # Principal point
    k1, k2, p1, p2, k3 = dist_coeffs.flatten() 

    # Convert input points to normalized coordinates
    normalized_point = (point - [cx, cy]) / [fx, fy]
    xd, yd = normalized_point[0], normalized_point[1]
    rd = (xd**2 + yd**2)**0.5

    # Iterative undistortion process
    ru = rd
    for _ in range(max_iterations):
        residual = (ru + k1*ru**3 + k2*ru**5) - rd
        fp = 2*residual*(1 + 3*k1*ru**2 + 5*k2*ru**4)
        ru_new = ru - learning_rate * fp
        if abs(ru_new - ru) < epsilon:
            break
        ru = ru_new

    # undistort in pixel coordinates
    undistorted_point = (point - center) * ru/rd + center
    
    return undistorted_point

# Display the image on the Tkinter canvas
def display_image_on_canvas(canvas_img):
    global canvas_widget, gscale
    if canvas_widget is not None:
        canvas.delete("all")  # Clear existing canvas content
    height, width = canvas_img.shape[:2]
    # Resize the canvas to match the image size
    new_width = int(width * gscale)
    new_height = int(height * gscale)    
    canvas.config(width=new_width, height=new_height)
    # Convert image array to PhotoImage-compatible format
    if canvas_img.dtype != np.uint8:
        canvas_img = (canvas_img * 255).clip(0, 255).astype(np.uint8)    
    img_pil = Image.fromarray(canvas_img).resize((new_width, new_height), Image.NEAREST)
    img_tk = ImageTk.PhotoImage(img_pil)
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image_tk = img_tk  # Keep a reference to prevent garbage collection

def show_image():
    global image
    display_image_on_canvas(image)
    
def undistort_image():
    global image, focal, center, dist_coeffs
    height, width = image.shape[:2]
    new_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            distorted_point = apply_distortion(np.array([x, y]), center, focal, dist_coeffs)
            xd = int(round(distorted_point[0]))
            yd = int(round(distorted_point[1]))
            if 0 <= xd < width and 0 <= yd < height:
                new_image[y, x] = image[yd, xd]
    display_image_on_canvas(new_image)
    
def undistort_image_direct():
    global image, focal, center, dist_coeffs
    height, width = image.shape[:2]
    new_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            point = np.array([x, y], dtype=np.float32)
            corrected = undistort_point(point, focal, center, dist_coeffs)
            xu = int(round(corrected[0]))
            yu = int(round(corrected[1]))
            if 0 <= xu < width and 0 <= yu < height:
                new_image[yu, xu] = image[y, x]
    display_image_on_canvas(new_image)
                
def undistort_image_enlarge(scale = 1.4):
    global image, focal, center, dist_coeffs
    height, width, channels = image.shape
    new_height = int(height*scale)
    new_width = int(width*scale)
    new_center = np.array([new_width/2, new_height/2])
    new_image = np.zeros((new_height, new_width, channels), dtype=image.dtype)
    for new_y in range(new_height):
        for new_x in range(new_width):
            distorted_point = apply_distortion_enlarge(np.array([new_x, new_y]), new_center, focal, dist_coeffs, center)
            xd = int(round(distorted_point[0]))
            yd = int(round(distorted_point[1]))
            if 0 <= xd < width and 0 <= yd < height:
                new_image[new_y, new_x] = image[yd, xd]    
    display_image_on_canvas(new_image)

def exit_app():
    root.destroy()
                
# Main function
if __name__ == "__main__":
    # Read the image
    image = plt.imread("sample_distorted.png")
    if image is None:
        print("Image not found!")
        exit(-1)

    # Initialize Tkinter
    root = Tk()
    root.title("Distortion Correction")

    # Create canvas for image display
    canvas = Canvas(root)
    canvas.pack()    

    # Add buttons
    button_frame = Frame(root)
    button_frame.pack()
    load_btn = Button(button_frame, text="Load Image", command=show_image)
    load_btn.grid(row=0, column=0, padx=5, pady=5)
    undistort_btn = Button(button_frame, text="Undistort", command=undistort_image)
    undistort_btn.grid(row=0, column=1, padx=5, pady=5)
    undistort_direct_btn = Button(button_frame, text="UndistortDirect", command=undistort_image_direct)
    undistort_direct_btn.grid(row=0, column=2, padx=5, pady=5)
    undistort_enlarge_btn = Button(button_frame, text="UndistortEnlarge", command=undistort_image_enlarge)
    undistort_enlarge_btn.grid(row=0, column=3, padx=5, pady=5)
    exit_btn = Button(button_frame, text="Exit", command=exit_app)
    exit_btn.grid(row=0, column=4, padx=5, pady=5)
    root.mainloop()
