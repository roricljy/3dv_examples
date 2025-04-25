import cv2
import numpy as np
from tkinter import Tk, Label
from PIL import Image, ImageTk
import os
gscale = 2.5 if "ANDROID_STORAGE" in os.environ else 1

# Checkerboard settings
pattern_size = (4, 7)
square_size = 25.0  # mm

# Camera intrinsics (replace with your calibrated values)
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float64)
fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

dist_coeffs = np.zeros((5, 1))

# 3D object points (checkerboard)
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# Function: manual 3D to 2D projection
def project_points_manual(points_3d, rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    imgpts = []

    for point in points_3d:
        X_cam = R @ point.reshape(3, 1) + tvec
        X_cam = X_cam.flatten()
        x = (fx * X_cam[0] / X_cam[2]) + cx
        y = (fy * X_cam[1] / X_cam[2]) + cy
        imgpts.append((int(x), int(y)))
    return imgpts

# Function: draw cube
def draw_cube(frame, rvec, tvec):
    board_w = pattern_size[0] * square_size
    board_h = pattern_size[1] * square_size
    cube_height = square_size * 4

    cube_points = np.float32([
        [0, 0, 0], [board_w, 0, 0], [board_w, board_h, 0], [0, board_h, 0],
        [0, 0, -cube_height], [board_w, 0, -cube_height],
        [board_w, board_h, -cube_height], [0, board_h, -cube_height]
    ])

    imgpts = project_points_manual(cube_points, rvec, tvec)

    # Draw base
    cv2.drawContours(frame, [np.array(imgpts[:4])], -1, (0, 255, 0), 3)
    # Draw pillars
    for i in range(4):
        cv2.line(frame, imgpts[i], imgpts[i + 4], (255, 0, 0), 3)
    # Draw top
    cv2.drawContours(frame, [np.array(imgpts[4:])], -1, (0, 0, 255), 3)

def on_key(event):
    if event.keysym == 'Escape' or event.char == 'q':
        print("Exit key pressed. Closing window...")
        cap.release()
        window.destroy()
        
# Tkinter setup
window = Tk()
window.title("Real-Time Checkerboard with Cube")
if gscale>1:
    window.attributes('-fullscreen', True)
label = Label(window)
label.pack()

# Bind ESC or 'q' key to exit
window.bind('<Key>', on_key)

# Open camera
cap = cv2.VideoCapture(0)

# Termination criteria
criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

def update_frame():
    ret, frame = cap.read()
    if not ret:
        window.after(10, update_frame)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if found:
        corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        success, rvec, tvec = cv2.solvePnP(objp, corners_refined, camera_matrix, dist_coeffs)
        if success:
            cv2.drawChessboardCorners(frame, pattern_size, corners_refined, found)
            draw_cube(frame, rvec, tvec)

    # Convert to RGB & show in Tkinter
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    window.after(10, update_frame)
        
# Run update loop
update_frame()
window.mainloop()

# Release camera when window closes
cap.release()