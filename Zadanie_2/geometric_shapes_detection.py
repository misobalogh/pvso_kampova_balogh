####################################
#      PVSO - Assignment 2         #
####################################
# Balogh Michal, Kampova Jana      #
# FEI STU                          #
# 2026                             #
####################################

import cv2
import numpy as np
import os

####################################
#          CAMERA SWITCH
USE_XIMEA = False
####################################


####################################
#   Camera calibration parameters  #
####################################
CALIB_FILE = "camera_calib.npz"

_mtx = None
_dist = None

if os.path.exists(CALIB_FILE):
    _data = np.load(CALIB_FILE)
    _mtx = _data["mtx"]
    _dist = _data["dist"]
    print(f"Loaded calibration parameters from '{CALIB_FILE}'")
else:
    print(f"'{CALIB_FILE}' not found")

_undist_cache: dict = {}


def _get_undistort_maps(h, w):
    assert _mtx is not None and _dist is not None
    key = (h, w)
    if key not in _undist_cache:
        newmtx, roi = cv2.getOptimalNewCameraMatrix(_mtx, _dist, (w, h), 1, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(
            _mtx, _dist, None, newmtx, (w, h), cv2.CV_32FC1
        )
        _undist_cache[key] = (mapx, mapy, roi)
    return _undist_cache[key]


def undistort_frame(frame):
    if _mtx is None or _dist is None:
        return frame
    h, w = frame.shape[:2]
    mapx, mapy, roi = _get_undistort_maps(h, w)
    dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    x, y, rw, rh = roi
    return dst[y : y + rh, x : x + rw]


####################################
#   Hough circle NMS               #
####################################


def _nms_circles(circles, dist_thresh_ratio = 0.8):
    if len(circles) == 0:
        return np.empty((0, 3), dtype=np.int32)

    circles = circles.astype(float)
    order = np.argsort(-circles[:, 2])  # sort by radius descending
    circles = circles[order]

    suppressed = np.zeros(len(circles), dtype=bool)
    for i in range(len(circles)):
        if suppressed[i]:
            continue
        cx1, cy1, r1 = circles[i]
        for j in range(i + 1, len(circles)):
            if suppressed[j]:
                continue
            cx2, cy2, r2 = circles[j]
            d = np.hypot(cx1 - cx2, cy1 - cy2)
            # Suppress j if its centre falls within (dist_thresh_ratio * larger_r)
            # of circle i center
            if d < dist_thresh_ratio * max(r1, r2):
                suppressed[j] = True

    return circles[~suppressed].astype(np.int32)


####################################
#   Shape detection                #
####################################


def detect_shapes(frame):
    frame = undistort_frame(frame)

    output = frame.copy()
    if frame.ndim == 3 and frame.shape[2] == 4: # BGRA for Ximea
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    img_h, img_w = gray.shape

    debug_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    debug_blurred = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    debug_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(frame)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

    size = (img_w // 2, img_h // 2)
    top_row = np.hstack([cv2.resize(debug_gray, size), cv2.resize(debug_blurred, size)])
    bottom_row = np.hstack(
        [cv2.resize(debug_edges, size), cv2.resize(contour_img, size)]
    )
    cv2.imshow("Debug", np.vstack([top_row, bottom_row]))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)

        moment = cv2.moments(cnt)
        if moment["m00"] != 0:
            cx = int(moment["m10"] / moment["m00"])
            cy = int(moment["m01"] / moment["m00"])
        else:
            cx, cy = 0, 0

        n_pts = len(approx)

        if n_pts == 3:
            shape = "Triangle"
        elif n_pts == 4:
            bx, by, bw, bh = cv2.boundingRect(approx)
            aspect = bw / float(bh)
            shape = "Square" if 0.95 < aspect < 1.05 else "Rectangle"
        elif n_pts == 5:
            shape = "Pentagon"
        elif n_pts == 6:
            shape = "Hexagon"
        else:
            # Circles are handled by Hough
            continue

        color = (255, 255, 255)
        cv2.drawContours(output, [approx], -1, color, 2)
        cv2.circle(output, (cx, cy), 4, (0, 0, 0), -1)
        cv2.putText(
            output, shape, (cx - 40, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    hough_blur = cv2.GaussianBlur(gray, (9, 9), 2)
    raw = cv2.HoughCircles(
        hough_blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=70,
        param2=40,
        minRadius=10,
        maxRadius=0,
    )

    if raw is not None:
        candidates = np.round(raw[0]).astype(np.int32)
        circles = _nms_circles(candidates)

        for cx, cy, r in circles:
            cv2.circle(output, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(output, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(
                output,
                "Circle",
                (cx - 30, cy - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    return output


####################################
#   Entry point                    #
####################################


def main():
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
            print("Cannot open camera")
            return

    while True:
        if USE_XIMEA:
            cam.get_image(xi_img)
            frame = xi_img.get_image_data_numpy()
        else:
            ret, frame = cam.read()
            if not ret:
                continue
        result = detect_shapes(frame)
        cv2.imshow("Shape detection", result)
        if cv2.waitKey(1) == ord("q"):
            break

    if USE_XIMEA:
        cam.stop_acquisition()
        cam.close_device()
    else:
        cam.release()
    cv2.destroyAllWindows()

    img = cv2.imread("imgs/shapes/test1.jpg")
    if img is not None:
        result = detect_shapes(img)
        cv2.imshow("Shape detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    img = cv2.imread("imgs/shapes/test2.jpg")
    if img is not None:
        result = detect_shapes(img)
        cv2.imshow("Shape detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
