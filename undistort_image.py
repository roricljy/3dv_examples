import cv2
import numpy as np
import math

# Function to apply the radial distortion
def apply_radial_distortion(point, center, focal, dist_coeffs):
    if np.array_equal(point, center):
        return point
    k1 = dist_coeffs[0]
    k2 = dist_coeffs[1]
    normalized_point = (point - center) / focal
    ru = np.linalg.norm(normalized_point)
    rd = ru + k1*ru**3 + k2*ru**5
    distorted = (point - center) * rd / ru + center
    return distorted

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

    # Camera parameters
    focal = np.array([461.9, 462.6], dtype=np.float32)  # fx, fy
    center = np.array([324.1, 190.5], dtype=np.float32) # cx, cy

    # Distortion Coefficients (k1, k2, p1, p2, k3)
    dist_coeffs = np.array([-0.413928, 0.162300, -0.001240, -0.000541, 0.0], dtype=np.float32)

    # Apply distortion correction to the image
    height, width = image.shape[:2]
    new_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            distorted_point = apply_radial_distortion(np.array([x, y]), center, focal, dist_coeffs)
            xd = int(round(distorted_point[0]))
            yd = int(round(distorted_point[1]))
            if 0 <= xd < width and 0 <= yd < height:
                new_image[y, x] = image[yd, xd]

    cv2.imshow("Corrected Image", new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()