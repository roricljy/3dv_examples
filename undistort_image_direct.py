import cv2
import numpy as np
import math

# Main function
if __name__ == "__main__":
    # Read the image
    image = cv2.imread("sample_distorted.png")
    if image is None:
        print("Image not found!")
        exit(-1)

    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Image", image)
    cv2.waitKey(1)

    # Camera matrix
    camera_matrix = np.array([[461.9, 0, 324.1],
                              [0, 462.6, 190.5],
                              [0, 0, 1]], dtype=np.float32)

    # Distortion Coefficients (k1, k2, p1, p2, k3)
    dist_coeffs = np.array([-0.413928, 0.162300, -0.001240, -0.000541, 0.0], dtype=np.float32)

    # Apply distortion correction to the image
    height, width, channels = image.shape
    new_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            point = np.array([x, y], dtype=np.float32)
            corrected = cv2.undistortPoints(point, camera_matrix, dist_coeffs, None, camera_matrix)
            xu = int(round(corrected[0, 0, 0]))
            yu = int(round(corrected[0, 0, 1]))
            if 0 <= xu < width and 0 <= yu < height:
                new_image[yu, xu] = image[y, x]

    cv2.imshow("Corrected Image", new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()