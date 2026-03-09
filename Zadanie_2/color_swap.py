####################################
#      PVSO - Assignment 2         #
#      Task 3: Color swap          #
####################################
# Balogh Michal, Kampova Jana      #
# FEI STU                          #
# 2026                             #
####################################

####################################
import cv2
import os
import numpy as np
from enum import Enum
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

class SwapState(Enum):
    ORIGINAL = 0
    RED      = 1
    GREEN    = 2
    BLUE     = 3

    def next(self):
        return SwapState((self.value + 1) % len(SwapState))


#################################
# Color swap
#################################
def color_swap_filter(image, state: SwapState):
    if state == SwapState.ORIGINAL:
        return image

    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
    maska = None
    nova_farba_bgr = (0, 0, 0)

    # Filters
    if state == SwapState.RED:
        m1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        m2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        maska = cv2.bitwise_or(m1, m2)
        nova_farba_bgr = (180, 105, 255) # Pink

    elif state == SwapState.GREEN:
        maska = cv2.inRange(hsv, (50, 50, 50), (70, 255, 255))
        nova_farba_bgr = (0, 165, 255) # Orange

    elif state == SwapState.BLUE:
        maska = cv2.inRange(hsv, (110, 50, 50), (130, 255, 255))
        nova_farba_bgr = (0, 255, 255) # Yellow

    # Mask
    if maska is not None:
        if num_channels == 3:  # BGR
            camera = ()
            image[maska > 0] = nova_farba_bgr + camera

        elif num_channels == 4: # BGRA
            camera = (255,)
            image[maska > 0] = nova_farba_bgr + camera

    return image


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


current_state = SwapState.ORIGINAL

while True:
    if USE_XIMEA:
        cam.get_image(img)
        image = img.get_image_data_numpy()
        image = cv2.resize(image, (width, height))
    else:
        ret, image = cam.read()
        image = cv2.resize(image, (width, height))

    image = color_swap_filter(image, current_state)
    cv2.imshow("Preview", image)

    k = cv2.waitKey(1)
    if k == ord("q"):
        cv2.destroyWindow("Preview")
        break
    if k == 32:
        current_state = current_state.next()
        print(f"Switched to {current_state.name} filter")



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
