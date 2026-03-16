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
CAMERA_DISTANCE_CM = 30.0

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


####################################
#   Hyperparameter settings        #
####################################

class Hyperparameters:
    CONTROL_WINDOW = "Settings"

    TRACKBAR = {
        "camera_distance": "Z cm x10",
        "canny_high": "Canny high",
        "blur_kernel": "Blur k",
        "min_area": "Min area",
        "poly_eps": "Poly eps x1e3",
        "hough_dp": "H dp x10",
        "hough_min_dist": "H minDist",
        "hough_votes": "H votes",
        "hough_min_r": "H minR",
        "hough_max_r": "H maxR 0=auto",
        "nms_ratio": "NMS x100",
    }

    def __init__(self):
        cv2.namedWindow(self.CONTROL_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.CONTROL_WINDOW, 640, 520)

        cv2.createTrackbar(self.TRACKBAR["camera_distance"], self.CONTROL_WINDOW, int(CAMERA_DISTANCE_CM * 10), 3000, self._noop)
        cv2.createTrackbar(self.TRACKBAR["canny_high"], self.CONTROL_WINDOW, 150, 255, self._noop)
        cv2.createTrackbar(self.TRACKBAR["blur_kernel"], self.CONTROL_WINDOW, 2, 20, self._noop)
        cv2.createTrackbar(self.TRACKBAR["min_area"], self.CONTROL_WINDOW, 50, 2000, self._noop)
        cv2.createTrackbar(self.TRACKBAR["poly_eps"], self.CONTROL_WINDOW, 18, 120, self._noop)

        cv2.createTrackbar(self.TRACKBAR["hough_dp"], self.CONTROL_WINDOW, 10, 50, self._noop)
        cv2.createTrackbar(self.TRACKBAR["hough_min_dist"], self.CONTROL_WINDOW, 30, 500, self._noop)
        cv2.createTrackbar(self.TRACKBAR["hough_votes"], self.CONTROL_WINDOW, 40, 255, self._noop)
        cv2.createTrackbar(self.TRACKBAR["hough_min_r"], self.CONTROL_WINDOW, 10, 300, self._noop)
        cv2.createTrackbar(self.TRACKBAR["hough_max_r"], self.CONTROL_WINDOW, 0, 600, self._noop)
        cv2.createTrackbar(self.TRACKBAR["nms_ratio"], self.CONTROL_WINDOW, 80, 200, self._noop)

    def _noop(self, _):
        pass

    def read(self):
        canny_high = max(1, cv2.getTrackbarPos(self.TRACKBAR["canny_high"], self.CONTROL_WINDOW))
        canny_low = max(1, canny_high // 2)

        blur_k = 2 * cv2.getTrackbarPos(self.TRACKBAR["blur_kernel"], self.CONTROL_WINDOW) + 1

        hough_min_r = cv2.getTrackbarPos(self.TRACKBAR["hough_min_r"], self.CONTROL_WINDOW)
        hough_max_r = cv2.getTrackbarPos(self.TRACKBAR["hough_max_r"], self.CONTROL_WINDOW)
        if hough_max_r != 0 and hough_max_r <= hough_min_r:
            hough_max_r = hough_min_r + 1

        return {
            "camera_distance_cm": cv2.getTrackbarPos(self.TRACKBAR["camera_distance"], self.CONTROL_WINDOW) / 10.0,
            "canny_low": canny_low,
            "canny_high": canny_high,
            "blur_k": blur_k,
            "min_area": cv2.getTrackbarPos(self.TRACKBAR["min_area"], self.CONTROL_WINDOW),
            "poly_eps": max(0.001, cv2.getTrackbarPos(self.TRACKBAR["poly_eps"], self.CONTROL_WINDOW) / 1000.0),
            "hough_dp": max(0.1, cv2.getTrackbarPos(self.TRACKBAR["hough_dp"], self.CONTROL_WINDOW) / 10.0),
            "hough_min_dist": max(1, cv2.getTrackbarPos(self.TRACKBAR["hough_min_dist"], self.CONTROL_WINDOW)),
            "hough_param2": max(1, cv2.getTrackbarPos(self.TRACKBAR["hough_votes"], self.CONTROL_WINDOW)),
            "hough_min_r": hough_min_r,
            "hough_max_r": hough_max_r,
            "nms_ratio": cv2.getTrackbarPos(self.TRACKBAR["nms_ratio"], self.CONTROL_WINDOW) / 100.0,
        }

hyperparams_singleton = None


def get_hyperparams():
    global hyperparams_singleton
    if hyperparams_singleton is None:
        hyperparams_singleton = Hyperparameters()
    return hyperparams_singleton


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


def px_to_cm(pixels, axis="x", camera_distance_cm=None):
    """
    formula: real_size = pixel_size * Z / focal_length
    """
    if _mtx is None:
        return None
    if camera_distance_cm is None:
        camera_distance_cm = CAMERA_DISTANCE_CM
    f = _mtx[0, 0] if axis == "x" else _mtx[1, 1]
    return pixels * camera_distance_cm / f


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


def _rectangularity(cnt):
    area = cv2.contourArea(cnt)
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    rect_area = rw * rh
    if rect_area <= 1e-6:
        return 0.0, rw, rh
    return area / rect_area, rw, rh


####################################
#   Shape detection                #
####################################


def detect_shapes(frame):
    params = get_hyperparams().read()
    frame = undistort_frame(frame)

    output = frame.copy()
    if frame.ndim == 3 and frame.shape[2] == 4: # BGRA for Ximea
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (params["blur_k"], params["blur_k"]), 0)
    edges = cv2.Canny(blurred, params["canny_low"], params["canny_high"])

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
        if area < params["min_area"]:
            continue

        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        if peri <= 0:
            continue

        eps_ratio = max(params["poly_eps"], 0.018)
        approx = cv2.approxPolyDP(hull, eps_ratio * peri, True)

        moment = cv2.moments(cnt)
        if moment["m00"] != 0:
            cx = int(moment["m10"] / moment["m00"])
            cy = int(moment["m01"] / moment["m00"])
        else:
            cx, cy = 0, 0

        n_pts = len(approx)
        rectangularity, rw, rh = _rectangularity(hull)
        draw_cnt = approx

        if n_pts == 3:
            shape = "Triangle"
            # longest side as representative size
            pts = approx.reshape(-1, 2)
            sides_px = [np.linalg.norm(pts[i] - pts[(i+1) % 3]) for i in range(3)]
            size_cm = px_to_cm(max(sides_px), "x", params["camera_distance_cm"])
            size_label = f"{size_cm:.1f} cm" if size_cm is not None else ""
        elif n_pts <= 6 and rectangularity > 0.8:
            shape = "Square" if min(rw, rh) > 0 and max(rw, rh) / min(rw, rh) < 1.1 else "Rectangle"
            w_cm = px_to_cm(rw, "x", params["camera_distance_cm"])
            h_cm = px_to_cm(rh, "y", params["camera_distance_cm"])
            size_label = f"{w_cm:.1f} x {h_cm:.1f} cm" if w_cm is not None else ""
            box = cv2.boxPoints(cv2.minAreaRect(hull))
            draw_cnt = np.int32(box).reshape(-1, 1, 2)
        elif n_pts == 4:
            bx, by, bw, bh = cv2.boundingRect(approx)
            aspect = bw / float(bh)
            shape = "Square" if 0.95 < aspect < 1.05 else "Rectangle"
            w_cm = px_to_cm(bw, "x", params["camera_distance_cm"])
            h_cm = px_to_cm(bh, "y", params["camera_distance_cm"])
            if w_cm is not None:
                size_label = (f"{w_cm:.1f} x {h_cm:.1f} cm")
            else:
                size_label = ""
        elif n_pts == 5:
            shape = "Pentagon"
            pts = approx.reshape(-1, 2)
            sides_px = [np.linalg.norm(pts[i] - pts[(i+1) % 5]) for i in range(5)]
            size_cm = px_to_cm(np.mean(sides_px), "x", params["camera_distance_cm"])
            size_label = f"{size_cm:.1f} cm" if size_cm is not None else ""
        elif n_pts == 6:
            shape = "Hexagon"
            pts = approx.reshape(-1, 2)
            sides_px = [np.linalg.norm(pts[i] - pts[(i+1) % 6]) for i in range(6)]
            size_cm = px_to_cm(np.mean(sides_px), "x", params["camera_distance_cm"])
            size_label = f"{size_cm:.1f} cm" if size_cm is not None else ""
        else:
            # Circles are handled by Hough
            continue

        color = (255, 255, 255)
        cv2.drawContours(output, [draw_cnt], -1, color, 2)
        cv2.circle(output, (cx, cy), 4, (0, 0, 0), -1)
        cv2.putText(
            output, shape, (cx - 40, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
        if size_label:
            cv2.putText(
                output, size_label, (cx - 40, cy + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1
            )

    raw = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=params["hough_dp"],
        minDist=params["hough_min_dist"],
        param1=params["canny_high"],
        param2=params["hough_param2"],
        minRadius=params["hough_min_r"],
        maxRadius=params["hough_max_r"],
    )

    if raw is not None:
        candidates = np.round(raw[0]).astype(np.int32)
        circles = _nms_circles(candidates, params["nms_ratio"])

        for cx, cy, r in circles:
            cv2.circle(output, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(output, (cx, cy), 3, (0, 0, 255), -1)
            diam_cm = px_to_cm(2 * r, "x", params["camera_distance_cm"])
            circle_label = "Circle"
            if diam_cm is not None:
                circle_label += f"  r={diam_cm/2:.1f} cm"
            cv2.putText(
                output,
                circle_label,
                (cx - 50, cy - r - 10),
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

    img = cv2.imread("imgs/shapes/test1.jpg")
    if img is not None:
        result = detect_shapes(img)
        cv2.imshow("Shape detection", result)
        cv2.waitKey(0)
        cv2.destroyWindow("Shape detection")

    img = cv2.imread("imgs/shapes/test2.jpg")
    if img is not None:
        result = detect_shapes(img)
        cv2.imshow("Shape detection", result)
        cv2.waitKey(0)
        cv2.destroyWindow("Shape detection")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
