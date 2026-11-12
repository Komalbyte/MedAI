import cv2
import numpy as np

def detect_skin(bgr):
    """Returns (cropped_image, skin_fraction, found)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 48, 80), (20, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return bgr, 0.0, False

    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[i]
    frac = float(area) / (bgr.shape[0] * bgr.shape[1])

    if frac < 0.15:
        return bgr, frac, False
    return bgr[y:y + h, x:x + w], frac, True

if __name__ == "__main__":
    img = cv2.imread("test.jpg")
    if img is None:
        print("test.jpg not found")
    else:
        crop, frac, found = detect_skin(img)
        cv2.imwrite("test_crop.jpg", crop)
        print("skin found:", found, "| skin area: %.0f%%" % (frac * 100))
