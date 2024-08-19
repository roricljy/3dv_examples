import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_calib_file(filepath):
    calib_data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if ':' in line:
                key, value = line.split(':', 1)
                calib_data[key] = np.array([float(x) for x in value.split()])
    return calib_data

def read_kitti_lidar(file_path):
    scan = np.fromfile(file_path, dtype=np.float32)
    points = scan.reshape((-1, 4))
    return points
    
def visualize_lidar(points):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def project_lidar_to_image(lidar_data, calibration_data, image):
    P2 = calibration_data['P2'].reshape(3, 4)  # Projection matrix from rectified camera to image
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = calibration_data['R0_rect'].reshape(3, 3)  # Rectification matrix
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :] = calibration_data['Tr_velo_to_cam'].reshape(3, 4)  # Transform from velodyne to camera coordinates
    
    # Transform LiDAR points to camera coordinates
    lidar_points = np.hstack((lidar_data[:, :3], np.ones((lidar_data.shape[0], 1))))  # (N, 4)
    lidar_points_cam = (R0_rect @ Tr_velo_to_cam @ lidar_points.T).T
    
    # Filter points that are behind the camera (z > 0)
    lidar_points_cam = lidar_points_cam[lidar_points_cam[:, 2] > 0]
    
    # Project the points onto the image plane
    points_2d = (P2 @ lidar_points_cam.T).T
    points_2d[:, 0] /= points_2d[:, 2]
    points_2d[:, 1] /= points_2d[:, 2]
    
    # Normalize the depth (z) values for color mapping
    depth = lidar_points_cam[:, 2]
    depth_min, depth_max = depth.min(), depth.max()
    depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled = (depth_normalized * 255).astype(np.uint8)

    # Apply colormap (rainbow colorjet)
    colors = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_RAINBOW)
    
    # Draw the projected points on the image
    for i, point in enumerate(points_2d):
        x, y = int(point[0]), int(point[1])
        color = colors[i][0]
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 2, color.tolist(), -1)
    
    return image
    

# load image
image = cv2.imread("kitti_image.png")
if image is None:
    print("Image not found!")
    exit(-1)
cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Image", image)
cv2.waitKey(1)

# load lidar
lidar_file_path = 'kitti_lidar.bin'
lidar_points = read_kitti_lidar(lidar_file_path)
visualize_lidar(lidar_points)

# load calibration data
calib_file_path = 'kitti_calib.txt'
calib_data = read_calib_file(calib_file_path)

# project lidar to image
projected_image = project_lidar_to_image(lidar_points, calib_data, image)

# display the result
cv2.imshow('Projected LiDAR on Image', projected_image)
cv2.waitKey()