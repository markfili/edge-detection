import glob
import os

import cv2 as cv
import numpy as np


class Contour:
    def __init__(self, raw=None):
        self.raw = None
        if raw is not None:
            self.contour = raw
            self.hull = cv.convexHull(raw)
            self.area = cv.contourArea(raw)
        else:
            self.area = 0

    def better_than(self, other):
        return self.area > 0 and self.area > other.area


def to_cv_value(gimpH, gimpS, gimpV):
    opencvH = gimpH / 2
    opencvS = (gimpS / 100) * 255
    opencvV = (gimpV / 100) * 255
    print(f"[{gimpH} {gimpS} {gimpV}] to [{opencvH} {opencvS} {opencvV}]")
    return [opencvH, opencvS, opencvV]


def change_brightness(image, value=30):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    v = cv.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv.merge((h, s, v))
    return cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)


def show_image(title, image, pos=0):
    title = "{} {}".format(title, pos)
    cv.imshow(title, image)
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.moveWindow(title, 0, 250 * pos)


def load_image(path):
    image = cv.imread(path)
    height, width, _ = image.shape
    image = cv.resize(image, [int(width * 0.4), int(height * 0.4)])
    return image


class EdgeDetector:

    def __init__(self, images):
        self.images = images
        self.threshold = 230
        self.k_size = 5
        self.blur_kernel_size = 7
        self.erode_kernel_size = self.k_size
        self.canny_upper = 255
        self.canny_lower = 20
        self.b = 0
        self.g = 0
        self.r = 255
        # 0, 0, 77
        self.h = 40
        self.s = 0
        self.v = 70
        # 360, 28, 100
        self.h_u = 360
        self.s_u = 46
        self.v_u = 100
        self.brightness = 30
        self.last_brightness = self.brightness

        self.display_settings()
        os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')

        pass

    def display_settings(self):
        self.settings_win = "settings"
        cv.namedWindow(self.settings_win, cv.WINDOW_FREERATIO)
        # cv.createTrackbar("threshold", self.settings_win, self.threshold, 255, self.update_threshold)
        # cv.createTrackbar("b", self.settings_win, self.b, 255, self.update_b)
        # cv.createTrackbar("g", self.settings_win, self.g, 255, self.update_g)
        # cv.createTrackbar("r", self.settings_win, self.r, 255, self.update_r)
        cv.createTrackbar("h l", self.settings_win, self.h, 360, self.update_h_l)
        cv.createTrackbar("s l", self.settings_win, self.s, 100, self.update_s_l)
        cv.createTrackbar("v l", self.settings_win, self.v, 100, self.update_v_l)
        cv.createTrackbar("h u", self.settings_win, self.h_u, 360, self.update_h_u)
        cv.createTrackbar("s u", self.settings_win, self.s_u, 100, self.update_s_u)
        cv.createTrackbar("v u", self.settings_win, self.v_u, 100, self.update_v_u)
        cv.createTrackbar("brightness", self.settings_win, self.brightness, 100, self.update_brightness)
        # cv.createTrackbar("canny lower", self.settings_win, self.canny_lower, 255, self.update_canny_lower)
        # cv.createTrackbar("canny upper", self.settings_win, self.canny_upper, 255, self.update_canny_upper)
        # cv.createTrackbar("blur k size", self.settings_win, self.blur_kernel_size, 255, self.update_blur_k_size)
        cv.moveWindow(self.settings_win, 50, 50)
        cv.resizeWindow(self.settings_win, 600, 0)
        # cv.createTrackbar("erode k size", self.settings_win, self.erode_kernel_size, 255, self.update_erode_k_size)

    def update_brightness(self, b):
        self.brightness = b

    def update_threshold(self, threshold):
        self.threshold = threshold

    def update_blur_k_size(self, k_size):
        if k_size % 2 == 1:
            self.blur_kernel_size = k_size

    def update_erode_k_size(self, k_size):
        self.erode_kernel_size = k_size

    def update_canny_lower(self, canny):
        self.canny_lower = canny

    def update_canny_upper(self, canny):
        self.canny_upper = canny

    def update_b(self, b):
        self.b = b

    def update_g(self, g):
        self.g = g

    def update_r(self, r):
        self.r = r

    def update_h_l(self, h):
        self.h = h

    def update_s_l(self, s):
        self.s = s

    def update_v_l(self, v):
        self.v = v

    def update_h_u(self, h):
        self.h_u = h

    def update_s_u(self, s):
        self.s_u = s

    def update_v_u(self, v):
        self.v_u = v

    def run(self):
        for image_path in self.images:
            print(image_path)
            image = load_image(image_path)

            lab = cv.cvtColor(image, cv.COLOR_BGR2Lab)
            show_image("lab", lab)
            while 1:
                k = cv.waitKey(5) & 0xFF
                if k == 27:
                    cv.destroyAllWindows()
                    exit(1)
                elif k == 32:
                    break
                pass

    # paper sticks out through applying mask by defining HSV colors
    # best so far
    def mask_out_paper(self, image):
        brighter = change_brightness(image, self.brightness)
        hsv = cv.cvtColor(brighter, cv.COLOR_BGR2HSV)

        while 1:
            if self.brightness != self.last_brightness:
                print("brightness change")
                brighter = change_brightness(image, self.brightness)
                hsv = cv.cvtColor(brighter, cv.COLOR_BGR2HSV)
                self.last_brightness = self.brightness

            mask = self.hsv_mask(hsv)
            bitwise = cv.bitwise_and(image, image, mask=mask)
            # canny = cv.Canny(bitwise, self.canny_lower, self.canny_upper)
            # contour = self.find_contours(canny)
            #
            # if contour.area > 0:
            #     cv.drawContours(image, [contour.contour], -1, (70, 255, 154), thickness=2)
            #     cv.drawContours(image, [contour.hull], -1, (255, 70, 155), thickness=2)

            show_image("results", image)
            show_image("mask", mask, 1)
            show_image("bitwise", bitwise, 2)
            show_image("hsv", hsv, 3)
            k = cv.waitKey(5) & 0xFF
            if k == 27:
                cv.destroyAllWindows()
                exit(1)
            elif k == 32:
                break
            pass

    def hsv_mask(self, hsv):
        lower_white = np.array(to_cv_value(self.h, self.s, self.v))
        upper_white = np.array(to_cv_value(self.h_u, self.s_u, self.v_u))
        mask = cv.inRange(hsv, lower_white, upper_white)
        return mask

    def original_mask(self, image):
        lower_white = np.array([self.b, self.g, self.r])
        upper_white = np.array([255, 255, 255])
        mask = cv.inRange(image, lower_white, upper_white)
        return mask

    def bitwise_mask(self, frame, hsv):
        lower_white = np.array([self.b, self.g, self.r], dtype="uint8")
        upper_white = np.array([255, 255, 255], dtype="uint8")
        mask = cv.inRange(frame, lower_white, upper_white)
        res = cv.bitwise_and(frame, frame, mask=mask)
        return res

    def contours(self, image):
        image = cv.medianBlur(image, self.blur_kernel_size)
        th, image = cv.threshold(image, self.threshold, 255, cv.THRESH_TRUNC)
        image = cv.Canny(image, self.canny_upper, self.canny_lower)

        # kernel = cv.getStructuringElement(cv.MORPH_RECT, [self.erode_kernel_size, self.erode_kernel_size])
        # image = cv.dilate(threshed, kernel)
        # image = cv.erode(image, kernel)

        show_image("image", image, 1)

    def hsv_threshold(self, hsv):
        h, s, v = cv.split(hsv)
        for i, channel in enumerate([h, s, v]):
            th, threshed = cv.threshold(channel, self.threshold, 255, cv.THRESH_BINARY_INV)
            show_image("threshed", threshed, i)

    def find_contours(self, image):
        image = cv.Canny(image, self.canny_lower, self.canny_upper)
        contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        max_contour = Contour()
        for raw_contour in contours:
            new_contour = Contour(raw_contour)
            if new_contour.better_than(max_contour):
                max_contour = new_contour
            pass
        pass
        return max_contour


if __name__ == '__main__':
    images_paths = glob.glob('/Users/marko/Testing/3dfoot/success/*')
    detector = EdgeDetector(images_paths)
    detector.run()
