####################################
#      PVSO - Assignment 2         #
####################################
# Balogh Michal, Kampova Jana      #
# FEI STU                          #
# 2026                             #
####################################

####################################
import cv2
import os
import glob
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
SPACEBAR        = 32
IMG_DIR         = "imgs"
CALIB_DIR       = "imgs/calib"
RESULTS_DIR     = "imgs/results"
CALIB_FILE      = "camera_calib.npz"

# Number of inner corners of chessboard row and column (BOARD_W x BOARD_H)
BOARD_W         = 5
BOARD_H         = 7
MIN_CALIB_IMGS  = 10

os.makedirs(CALIB_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class UndistortMethod(Enum):
    CROP = 1
    REMAPP = 2
####################################


def capture_calibration_images():
    taken = 0

    if USE_XIMEA:
        from ximea import xiapi
        cam = xiapi.Camera()
        print("Opening first Ximea camera...")
        cam.open_device()
        cam.set_exposure(50000)
        cam.set_param("imgdataformat", "XI_RGB32")
        cam.set_param("auto_wb", 1)
        xi_img = xiapi.Image()
        cam.start_acquisition()
    else:
        print("Opening notebook camera...")
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            raise RuntimeError("Cannot open camera")

    while True:
        if USE_XIMEA:
            cam.get_image(xi_img)
            frame = xi_img.get_image_data_numpy()
        else:
            ret, frame = cam.read()
            if not ret:
                continue

        preview = frame.copy()

        cv2.imshow("Calibration preview", preview)
        k = cv2.waitKey(1)

        if k == ord("q"):
            break
        if k == SPACEBAR:
            path = os.path.join(CALIB_DIR, f"calib_{taken:03d}.jpg")
            cv2.imwrite(path, frame)
            taken += 1
            print(f"Saved image {taken}: {path}")

    cv2.destroyAllWindows()
    if USE_XIMEA:
        cam.stop_acquisition()
        cam.close_device()
    else:
        cam.release()
    return taken


def calibrate_camera():
    # --- BEGIN: Adapted from OpenCV documentation ---
    # Source: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    # Changes: replaced fixed board size with BOARD_W, BOARD_H constants;
    #          replaced hardcoded image path with CALIB_DIR variable.

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(BOARD_W-1,BOARD_H-1,0)
    objp = np.zeros((BOARD_H * BOARD_W, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(os.path.join(CALIB_DIR, "*.jpg"))

    for fname in sorted(images):
        img  = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (BOARD_W, BOARD_H), corners2, ret)
            cv2.imshow("img", img)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    # --- END: Adapted from OpenCV documentation ---

    print(60*"#")
    print(20*" " + "Calibration results:")
    print(60*"#")
    print("Camera matrix K:")
    print(mtx)

    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    print(f"\n{'fx (focal length x):':<25} {fx:.4f} px")
    print(f"{'fy (focal length y):':<25} {fy:.4f} px")
    print(f"{'cx (optical center x):':<25} {cx:.4f} px")
    print(f"{'cy (optical center y):':<25} {cy:.4f} px")

    print("\nDistortion coefficients (k1,k2,p1,p2,k3):")
    print(dist)

    # --- BEGIN: Adapted from OpenCV documentation ---
    # Source: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
     # --- END: Adapted from OpenCV documentation ---

    print(f"\n{'Mean reprojection error:':<25} {mean_error/len(objpoints):.6f} px")
    print(60*"#")
    np.savez(CALIB_FILE, mtx=mtx, dist=dist)
    print(f"\nCalibration parameters saved to: {CALIB_FILE}")

    return mtx, dist



def show_comparison(original, processed):
    orig_disp = cv2.resize(original, (640, 480))
    und_disp  = cv2.resize(processed, (orig_disp.shape[1], orig_disp.shape[0]))
    comparison = np.hstack([orig_disp, und_disp])

    cv2.imshow("Original vs Undistorted", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


####################################
#   FÁZA 3 – UNDISTORTION         #
####################################
def undistort_image(mtx, dist, img_path, method=UndistortMethod.CROP):
    img   = cv2.imread(img_path)
    h, w  = img.shape[:2]

    newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    if method == UndistortMethod.CROP:
        dst = cv2.undistort(img, mtx, dist, None, newmtx)
    elif method == UndistortMethod.REMAPP:
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newmtx, (w,h), cv2.CV_32F)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    x, y, rw, rh = roi
    dst = dst[y:y+rh, x:x+rw]
    show_comparison(img, dst)


####################################
#              Main                #
####################################
if __name__ == "__main__":
    existing = glob.glob(os.path.join(CALIB_DIR, "*.jpg"))
    if len(existing) < MIN_CALIB_IMGS:
        capture_calibration_images()

    mtx, dist = calibrate_camera()

    data = np.load(CALIB_FILE)
    mtx, dist = data["mtx"], data["dist"]
    img_path = sorted(glob.glob(os.path.join(CALIB_DIR, "*.jpg")))[0]

    undistort_image(mtx, dist, img_path, method=UndistortMethod.CROP)
    undistort_image(mtx, dist, img_path, method=UndistortMethod.REMAPP)
####################################
