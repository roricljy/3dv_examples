import cv2
import numpy as np
import math

# Function to apply the radial distortion
def apply_radial_distortion(point, center, w, dst_center):
    if w == 0 or np.array_equal(point, center):
        return point
    ru = np.linalg.norm(point - center)
    rd = np.arctan(w * ru) / w
    distorted = (point - center) * rd / ru + dst_center
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

    # Distortion Coefficients
    w = 0.00248

    # Apply distortion correction to the image
    height, width, channels = image.shape
    new_height = int(height*1.3)
    new_width = int(width*1.3)
    new_center = np.array([new_width/2, new_height/2])
    new_image = np.zeros((new_height, new_width, channels), dtype=image.dtype)
    for new_y in range(new_height):
        for new_x in range(new_width):
            distorted_point = apply_radial_distortion(np.array([new_x, new_y]), new_center, w, center)
            xd = int(round(distorted_point[0]))
            yd = int(round(distorted_point[1]))
            if 0 <= xd < width and 0 <= yd < height:
                new_image[new_y, new_x] = image[yd, xd]

    cv2.imshow("Corrected Image", new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()