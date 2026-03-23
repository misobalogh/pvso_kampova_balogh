import numpy as np
import cv2

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 320
DISPLAY_SCALE = 1

IMAGE_DIR = "imgs"


def global_threshold(gray: np.ndarray, thresh: int = 128) -> np.ndarray:
    out = np.zeros_like(gray, dtype=np.uint8)
    out[gray > thresh] = 255
    return out


def otsu_threshold(gray: np.ndarray) -> (np.ndarray, int):  # type: ignore
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)

    weight_b = 0.0
    sum_b = 0.0
    max_var = 0.0
    threshold = 0

    for t in range(256):
        weight_b += hist[t]
        if weight_b == 0:
            continue

        weight_f = total - weight_b
        if weight_f == 0:
            break

        sum_b += t * hist[t]
        mean_b = sum_b / weight_b
        mean_f = (sum_all - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2

        if var_between > max_var:
            max_var = var_between
            threshold = t

    out = global_threshold(gray, threshold)
    return out, threshold


def adaptive_mean_threshold(
    gray: np.ndarray, block_size: int = 15, C: int = 5
) -> np.ndarray:
    half = block_size // 2
    padded = cv2.copyMakeBorder(gray, half, half, half, half, cv2.BORDER_REPLICATE)
    out = np.zeros_like(gray, dtype=np.uint8)

    integral = cv2.integral(padded, sdepth=cv2.CV_64F)

    h, w = gray.shape
    for y in range(h):
        y1 = y
        y2 = y + block_size
        for x in range(w):
            x1 = x
            x2 = x + block_size
            sum_region = (
                integral[y2, x2]
                - integral[y1, x2]
                - integral[y2, x1]
                + integral[y1, x1]
            )
            mean = sum_region / (block_size * block_size)
            out[y, x] = 255 if gray[y, x] > (mean - C) else 0

    return out


def apply_on_quadrant(image: np.ndarray, src: np.ndarray, part_num: int) -> np.ndarray:
    height, width = image.shape[:2]
    quad_height = height // 2
    quad_width = width // 2
    row = part_num // 2
    col = part_num % 2
    y_start = row * quad_height
    y_end = y_start + quad_height
    x_start = col * quad_width
    x_end = x_start + quad_width
    image[y_start:y_end, x_start:x_end] = src
    return image


def _on_trackbar(_value: int) -> None:
    # Callback needs to exist for trackbar
    pass


def camera_mosaic_demo(width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT) -> None:
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera, skipping live camera demo.")
        return

    manual_window = "Manual threshold mosaic: gray/global/otsu/adaptive"
    # opencv_window = 'OpenCV threshold mosaic: original/global/otsu/adaptive'
    # cv2.namedWindow(opencv_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(manual_window, cv2.WINDOW_NORMAL)

    display_w = int(2 * width * DISPLAY_SCALE)
    display_h = int(2 * height * DISPLAY_SCALE)
    cv2.resizeWindow(manual_window, display_w, display_h)
    # cv2.resizeWindow(opencv_window, display_w, display_h)

    cv2.createTrackbar("Global threshold", manual_window, 128, 255, _on_trackbar)
    cv2.createTrackbar("Adaptive block", manual_window, 31, 99, _on_trackbar)
    cv2.createTrackbar("Adaptive C", manual_window, 5, 50, _on_trackbar)

    print("Live mosaic running. Press q to quit.")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to read frame from camera")
            break

        frame = cv2.resize(frame, (width, height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        thresh = cv2.getTrackbarPos("Global threshold", manual_window)
        block = cv2.getTrackbarPos("Adaptive block", manual_window)
        if block < 3:
            block = 3
        if block % 2 == 0:
            block += 1
        c_val = cv2.getTrackbarPos("Adaptive C", manual_window)

        manual_global = global_threshold(gray, thresh)
        manual_otsu, _ = otsu_threshold(gray)
        manual_adaptive = adaptive_mean_threshold(gray, block_size=block, C=c_val)

        # cv_global = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
        # cv_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # cv_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                     cv2.THRESH_BINARY, block, c_val)

        manual_mosaic = np.zeros((2 * height, 2 * width, 3), dtype=np.uint8)
        apply_on_quadrant(manual_mosaic, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0)
        apply_on_quadrant(
            manual_mosaic, cv2.cvtColor(manual_global, cv2.COLOR_GRAY2BGR), 1
        )
        apply_on_quadrant(
            manual_mosaic, cv2.cvtColor(manual_otsu, cv2.COLOR_GRAY2BGR), 2
        )
        apply_on_quadrant(
            manual_mosaic, cv2.cvtColor(manual_adaptive, cv2.COLOR_GRAY2BGR), 3
        )

        # opencv_mosaic = np.zeros((2 * height, 2 * width, 3), dtype=np.uint8)
        # apply_on_quadrant(opencv_mosaic, frame, 0)
        # apply_on_quadrant(opencv_mosaic, cv2.cvtColor(cv_global, cv2.COLOR_GRAY2BGR), 1)
        # apply_on_quadrant(opencv_mosaic, cv2.cvtColor(cv_otsu, cv2.COLOR_GRAY2BGR), 2)
        # apply_on_quadrant(opencv_mosaic, cv2.cvtColor(cv_adaptive, cv2.COLOR_GRAY2BGR), 3)

        display_mosaic = cv2.resize(
            manual_mosaic,
            (int(2 * width * DISPLAY_SCALE), int(2 * height * DISPLAY_SCALE)),
            interpolation=cv2.INTER_NEAREST,
        )

        cv2.imshow(manual_window, display_mosaic)
        # cv2.imshow(opencv_window, opencv_mosaic)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


def compare_images(name: str, img1: np.ndarray, img2: np.ndarray):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    same_pixels = np.sum(img1 == img2)
    total = img1.size
    print(
        f"{name}: MSE={mse:.2f}, same_pixels={same_pixels}/{total} ({100.0 * same_pixels / total:.2f}%)"
    )


def main():
    camera_mosaic_demo()

    image_path = f"{IMAGE_DIR}/angrycat.jpg"
    img = cv2.imread(image_path)
    if img is None:
        print(f"Unable to read image from {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ours
    bin_global = global_threshold(gray, 128)
    bin_otsu, otsu_t = otsu_threshold(gray)
    bin_adaptive = adaptive_mean_threshold(gray, block_size=31, C=10)

    # OpenCV
    _, cv_global = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    _, cv_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv_adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10
    )

    print(f"Otsu threshold: {otsu_t}")

    compare_images("Global threshold", bin_global, cv_global)
    compare_images("Otsu threshold", bin_otsu, cv_otsu)
    compare_images("Adaptive mean", bin_adaptive, cv_adaptive)

    # cv2.imshow('Original', img)
    # cv2.imshow('Gray', gray)
    # cv2.imshow('Manual Global', bin_global)
    # cv2.imshow('OpenCV Global', cv_global)
    # cv2.imshow('Manual Otsu', bin_otsu)
    # cv2.imshow('OpenCV Otsu', cv_otsu)
    # cv2.imshow('Manual Adaptive', bin_adaptive)
    # cv2.imshow('OpenCV Adaptive', cv_adaptive)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
