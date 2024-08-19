import cv2
import numpy as np

# 매칭된 쌍 (objectPoints: 3D 월드 좌표, imagePoints: 2D 이미지 좌표)
objectPoints = np.array([...], dtype=np.float32)  # 3D object points
imagePoints = np.array([...], dtype=np.float32)   # 2D image points

# 카메라 파라미터
fx = ...  # Focal length in x
fy = ...  # Focal length in y
cx = ...  # Principal point x
cy = ...  # Principal point y
k1, k2, p1, p2 = ...  # Distortion coefficients

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)  # Camera matrix

distCoeffs = np.array([k1, k2, p1, p2], dtype=np.float64)  # Distortion coefficients

# 카메라 pose 추정
success, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, K, distCoeffs)

# 회전 행렬 추출
R, _ = cv2.Rodrigues(rvec)
t = tvec

# 출력
print("Rotation Matrix:\n", R)
print("Translation Vector:\n", t)
