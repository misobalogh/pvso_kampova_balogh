####################################
#      PVSO - Assignment 1         #
####################################
# Balogh Michal, Kampova Jana      #
# FEI STU                          #
# 2026                             #
####################################

####################################
import cv2
import os
import numpy as np
####################################

####################################
#          CAMERA SWITCH
USE_XIMEA = False
####################################


####################################
#          SETTINGS
####################################
SPACEBAR    = 32
IMG_DIR     = "imgs"
MAX_PICS    = 4

width       = 240
height      = 240
num_channels = 4 if USE_XIMEA else 3

taken_pics  = 0

os.makedirs(IMG_DIR, exist_ok=True)
mosaic = np.zeros((2 * width, 2 * height, num_channels), dtype=np.uint8)
####################################


def apply_on_quadrant(image, fun_to_apply, part_num):
    height, width = image.shape[:2]

    quad_height = height // 2
    quad_width  = width // 2

    row = part_num // 2
    col = part_num % 2
    y_start = row * quad_height
    y_end = y_start + quad_height
    x_start = col * quad_width
    x_end = x_start + quad_width

    roi = image[y_start:y_end, x_start:x_end, :]
    processed_roi = fun_to_apply(roi)

    image[y_start:y_end, x_start:x_end, :] = processed_roi
    return image

def sobel_filter(roi):
    """
    Apply 3x3 Sobel edge detection filter to the input ROI.

    - sobel_x detects horizontal edges
    - sobel_y detects vertical edges
    This gives us gradients. From these gradients, we can compute
    the magnitude of the gradient vector at each pixel, which represents
    the strength of the edge. The result is a combination of both directions.
    (gradient = how much the intensity changes in any direction).


    Args:
        roi: Region of interest (image quadrant) to process
    Returns:
        Processed ROI with edges highlighted (BGR image)
    """

    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    grad_x  = cv2.filter2D(gray, cv2.CV_64F, sobel_x, borderType=cv2.BORDER_DEFAULT)
    grad_y  = cv2.filter2D(gray, cv2.CV_64F, sobel_y, borderType=cv2.BORDER_DEFAULT)

    magnitude = np.sqrt(grad_x**2 + grad_y**2).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

def rotate90_filter(roi):
    row, col = roi.shape[:2]
    rotated = np.zeros_like(roi)
    for i in range(row):
        for j in range(col):
            rotated[j, row - 1 - i, :] = roi[i, j, :]
    return rotated

def red_channel_filter(roi):
    red = roi.copy()
    # OpenCV : BGR
    # B = 0 , G = 1 , R = 2
    if num_channels == 3:  # BGR
        red[0:240, 0:240, 0] = 0  # Blue
        red[0:240, 0:240, 1] = 0  # Green
    elif num_channels == 4: # RGBA
        red[:, :, 1] = 0
        red[:, :, 2] = 0
    return red



if USE_XIMEA:
    from ximea import xiapi

    cam = xiapi.Camera()
    print("Opening first Ximea camera...")
    cam.open_device()
    cam.set_exposure(50000)
    cam.set_param("imgdataformat", "XI_RGB32")
    cam.set_param("auto_wb", 1)
    print("Exposure was set to %i us" % cam.get_exposure())
    img = xiapi.Image()
    print("Starting data acquisition...")
    cam.start_acquisition()
else:
    print("Opening notebook camera...")
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Cannot open camera")


while taken_pics < MAX_PICS:
    if USE_XIMEA:
        cam.get_image(img)
        image = img.get_image_data_numpy()
        image = cv2.resize(image, (width, height))
    else:
        ret, image = cam.read()
        image = cv2.resize(image, (width, height))

    cv2.imshow("Preview", image)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break
    if k == 32:
        cv2.imwrite(f"{IMG_DIR}/image_{taken_pics}.jpeg", image)

        apply_on_quadrant(mosaic, lambda _: image, taken_pics)
        taken_pics += 1
        print(f"Captured image {taken_pics}/4")

if taken_pics == MAX_PICS:

    cv2.destroyWindow("Preview")

    apply_on_quadrant(mosaic, sobel_filter, 0)
    apply_on_quadrant(mosaic, rotate90_filter, 1)
    apply_on_quadrant(mosaic, red_channel_filter, 2)

    cv2.imshow("Mosaic", mosaic.astype(np.uint8))
    print("Press any key to exit...")
    cv2.waitKey(0)

    print(60*"#")
    print(24*" "+"Image info:")
    print(60*"#")
    print(f"{'Data type':<32}{str(mosaic.dtype):>28}")
    print(f"{'Dimensions':<32}{str(mosaic.shape):>28}")
    print(f"{'Height':<32}{str(mosaic.shape[0]):>28}")
    print(f"{'Width':<32}{str(mosaic.shape[1]):>28}")
    print(f"{'Channels':<32}{num_channels:>28}")
    color_format = 'BGR' if num_channels == 3 else 'RGBA' if num_channels == 4 else 'unknown'
    print(f"{'Color format':<32}{color_format:>28}")
    print(f"{'Total elements (h x w x ch)':<32}{mosaic.size:>28}")
    print(f"{'Number of pixels (h x w)':<32}{mosaic.shape[0] * mosaic.shape[1]:>28}")
    print(f"{'Memory size in bytes':<32}{mosaic.nbytes:>28}")
    print(60*"#")

####################################
#           Cleanup
####################################
if USE_XIMEA:
    print("Stopping acquisition...")
    cam.stop_acquisition()
    cam.close_device()
else:
    cam.release()
cv2.destroyAllWindows()

if "USE_XIMEA" in globals() and USE_XIMEA:
    print("Stopping acquisition...")
    cam.stop_acquisition()
    cam.close_device()
####################################
