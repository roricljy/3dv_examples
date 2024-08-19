import cv2
import numpy as np

# 카메라 행렬 (예시)
camera_matrix = np.array([[600, 0, 320],
                          [0, 600, 240],
                          [0, 0, 1]], dtype=np.float32)

# 왜곡 계수 (k1, k2, p1, p2, k3)
dist_coeffs = np.array([-0.414, 0.162, 0.0, 0.0, 0.0], dtype=np.float32)

# 입력된 픽셀 좌표 (예시로 몇 개의 픽셀 좌표 사용)
points = np.array([[480, 290],
                   [320, 240],
                   [100, 200]], dtype=np.float32)

# 왜곡 보정된 픽셀 좌표 계산
undistorted_points = cv2.undistortPoints(points, camera_matrix, dist_coeffs, None, camera_matrix)

# 결과 출력
print("왜곡 보정된 픽셀 좌표:")
print(undistorted_points)