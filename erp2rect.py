import cv2
import numpy as np

def erp_to_rect(erp_image, theta, hfov, vfov):
    erp_H, erp_W, channels = erp_image.shape
    f = erp_W/(2*np.pi)
    rect_H = int(round(2*f*np.tan(vfov/2)))
    rect_W = int(round(2*f*np.tan(hfov/2)))
    rect_cx = rect_W/2
    rect_cy = rect_H/2
    rect_image = np.zeros((rect_H, rect_W, channels), dtype=erp_image.dtype)
    for rect_y in range(rect_H):
        for rect_x in range(rect_W):
            xth = np.arctan((rect_x - rect_cx) / f)  # relative angle from center
            xth_erp = theta + xth                        # absolute angle in erp  
            erp_x = (theta + xth) * erp_W / (2*np.pi)    # angle to pixel

            yf = f / np.cos(xth)
            yth = np.arctan((rect_y - rect_cy) / yf)
            erp_y = yth * erp_H / np.pi + erp_H / 2

            erp_xi = int(round(erp_x))
            erp_yi = int(round(erp_y))
            if 0 <= erp_xi < erp_W and 0 <= erp_yi < erp_H:
                rect_image[rect_y, rect_x] = image[erp_yi, erp_xi]
    return rect_image

# Main function
if __name__ == "__main__":
    # Read the image
    image = cv2.imread("sample_erp.png")
    if image is None:
        print("Image not found!")
        exit(-1)

    cv2.namedWindow("erp image", 0)
    cv2.imshow("erp image", image)
    cv2.waitKey(1)

    # Rectify the erp image
    theta = np.radians(180)
    hfov = np.radians(120)
    vfov = np.radians(90)
    rect_image = erp_to_rect(image, theta, hfov, vfov)
    cv2.imshow("rectilinear image", rect_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()