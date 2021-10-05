import glob
from enum import Enum

import cv2 as cv
import numpy as np


class ContourInfo:
    def __init__(self, contour):
        self.contour = contour
        self.area = cv.contourArea(contour)
        self.arc = cv.arcLength(contour, True) * 0.02
        self.approx = cv.approxPolyDP(contour, self.arc, False)
        self.corners = len(self.approx)
        self.bounding_rect = cv.boundingRect(contour)

    def area_covered(self):
        x, y, box_width, box_height = self.bounding_rect
        bounding_box_area = box_width * box_height
        contour_area = self.area
        percentage = contour_area / bounding_box_area
        return int(percentage * 100)


def draw_contours(contours: list, output):
    # loop over the contours
    for contour in contours:
        x, y, w, h = contour.bounding_rect
        color = (255 - contour.area * w % 255, 255 - contour.area * h * w % 255, 255 - contour.area * x % 255)
        cv.drawContours(output, [contour.contour], -1, color, 5)
        cv.rectangle(output, (x, y), (x + w, y + h), color, 2)
    pass


def draw_corners(corners, image):
    if corners is not None:
        for i in corners:
            x, y = i.ravel()
            cv.circle(image, (int(x), int(y)), 3, (0, 0, 255), 15)


def stop_execution():
    key = cv.waitKey(1)
    converted_key = key & 0xFF
    return converted_key == ord('q')


def next_frame():
    key = cv.waitKey(1)
    converted_key = key & 0xFF
    return converted_key == ord('e')


def change_brightness(image, value=30):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    v = cv.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv.merge((h, s, v))
    return cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)


def add_text(image, text, pos):
    cv.putText(image, text, pos, fontScale=1, fontFace=cv.FONT_HERSHEY_PLAIN, color=255)


def show_image(window, image):
    cv.imshow(window, image)


class ProcessModes(Enum):
    MODE_CANNY = 1
    MODE_MORPH = 2
    MODE_THRESHOLD = 3
    MODE_MASK = 4


class EdgeDetector:

    def __init__(self, mode=ProcessModes.MODE_CANNY):
        self.settings_window = "Settings"
        self.process_mode = mode
        self.hsv_sensitivity = 15
        self.noise_kernel_size = 3
        self.brightness = 30
        self.max_thresh_value = 255
        self.thresh_value = 85
        self.min_dist = 50
        self.features_quality = 0.01
        self.features_max_corners = 25
        self.features_corners_distance = 50
        self.canny_lower = 50
        self.canny_upper = 150
        self.screen_width = 640
        self.screen_height = 480
        self.source_images = None
        self.source_videos = None
        self.source_use_camera = False
        self.mode_switcher = {
            ProcessModes.MODE_CANNY: self.attempt_with_canny,
            ProcessModes.MODE_MORPH: self.attempt_with_morph,
            ProcessModes.MODE_THRESHOLD: self.attempt_with_threshold,
            ProcessModes.MODE_MASK: self.attempt_with_mask
        }
        # Flip it
        # cv.flip(frame, +1, dst=frame)
        # if self.record:
        #     self.video_name = session_name + ".mp4"
        #     self.four2c = cv.VideoWriter_fourcc(*'mp4v')
        #     self.out = cv.VideoWriter(self.video_name, self.four2c, 20.0, (self.screen_width, self.screen_height))

    def images(self, images=None):
        self.source_images = images

    def videos(self, videos=None):
        self.source_videos = videos

    def use_camera(self, use_camera=True):
        self.source_use_camera = use_camera

    def run(self):
        should_run = True
        self.show_settings()
        if self.source_images:
            while should_run:
                for image in self.source_images:
                    frame = cv.imread(image)
                    frame = cv.resize(frame, None, fx=0.1, fy=0.1)
                    while should_run:
                        result, processed = self.find_edges(frame)
                        show_image("Processed", processed)
                        show_image("Result", result)

                        height, width = frame.shape[:2]
                        cv.moveWindow("Result", 20, 100)
                        cv.moveWindow("Processed", 40 + width, 100)
                        if next_frame():
                            break
                        elif stop_execution():
                            should_run = False
                            cv.destroyAllWindows()
                            break

        elif self.source_use_camera:
            video = cv.VideoCapture(0)
            video.set(3, self.screen_width)
            video.set(4, self.screen_height)
            while video.isOpened():
                success, frame = video.read()

                result, processed = self.find_edges(frame)

                cv.flip(result, 1, dst=result)
                cv.flip(processed, 1, dst=processed)

                show_image("Processed", processed)
                show_image("Result", result)

                height, width = frame.shape[:2]
                cv.moveWindow("Result", 20, 100)
                cv.moveWindow("Processed", 40 + width, 100)
                if stop_execution():
                    cv.destroyAllWindows()
                    break

    def find_edges(self, frame):
        frame_copy = frame.copy()
        # Read frame
        result = self.prepare_frame(frame_copy)
        # find corners and contours
        corners = self.get_features(result)
        contours = self.get_contours(result)

        # lines = self.get_lines(result)
        # if lines is not None:
        #     for i in range(0, len(lines)):
        #         line = lines[i][0]
        #         cv.line(frame, (line[0], line[1]), (line[2], line[3]), (200, 255, 100), 13)

        draw_contours(contours, frame_copy)
        draw_corners(corners, frame_copy)
        return frame_copy, result

    def get_features(self, frame):
        corners = cv.goodFeaturesToTrack(frame, self.features_max_corners, self.features_quality,
                                         self.features_corners_distance)
        return corners

    def get_contours(self, output):
        contours, hierarchy = cv.findContours(output, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        parsed_contours = []
        for contour in contours:
            parsed_contours.append(ContourInfo(contour))

        result_contours = sorted(parsed_contours, key=ContourInfo.area_covered, reverse=True)[:5]

        return result_contours

    def get_lines(self, frame):
        return cv.HoughLinesP(frame, 1, np.pi / 180, 50, None, 5, 10)

    def prepare_frame(self, frame):
        frame = change_brightness(frame, self.brightness)
        process_function = self.mode_switcher.get(self.process_mode, "Null")
        return process_function(frame)

    def attempt_with_canny(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.GaussianBlur(frame, (self.noise_kernel_size, self.noise_kernel_size), 0)
        frame = cv.erode(frame, None, iterations=2)
        frame = cv.dilate(frame, None, iterations=2)
        frame = cv.Canny(frame, self.canny_lower, self.canny_upper)
        return frame

    def attempt_with_morph(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.Canny(frame, self.canny_lower, self.canny_upper)
        frame = cv.GaussianBlur(frame, (self.noise_kernel_size, self.noise_kernel_size), 10, sigmaY=10)
        frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, (self.noise_kernel_size, self.noise_kernel_size))
        return frame

    def attempt_with_threshold(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        val, frame = cv.threshold(frame, self.thresh_value, self.max_thresh_value, cv.THRESH_BINARY)
        return frame

    def attempt_with_mask(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame = cv.GaussianBlur(frame, (self.noise_kernel_size, self.noise_kernel_size), 0)
        sensitivity = self.hsv_sensitivity
        lower_white = np.array([0, 0, 255 - sensitivity])
        upper_white = np.array([255, sensitivity, 255])
        frame = cv.inRange(frame, lower_white, upper_white)
        frame = cv.erode(frame, None, iterations=2)
        frame = cv.dilate(frame, None, iterations=2)
        return frame

    def show_settings(self):
        cv.namedWindow(self.settings_window)
        settings_switcher = {
            ProcessModes.MODE_CANNY: self.settings_canny,
            ProcessModes.MODE_MORPH: self.settings_morph,
            ProcessModes.MODE_THRESHOLD: self.settings_threshold,
            ProcessModes.MODE_MASK: self.settings_mask
        }
        settings_switcher.get(self.process_mode)()
        cv.createTrackbar("Features quality", self.settings_window, int(self.features_quality * 100), 10,
                          self.set_features_quality)
        cv.createTrackbar("Features max corners", self.settings_window, self.features_max_corners, 200,
                          self.set_features_max_corners)
        cv.createTrackbar("Features corners dist", self.settings_window, self.features_corners_distance, 1000,
                          self.set_max_corners_distance)
        cv.createTrackbar("Brightness", self.settings_window, self.brightness + 50, 100, self.set_brightness)
        cv.createTrackbar("Noise kernel", self.settings_window, self.noise_kernel_size, 10, self.set_noise_kernel_size)

        cv.moveWindow(self.settings_window, 40 + 2 * self.screen_width, 100)

    def settings_canny(self):
        cv.createTrackbar("Canny upper", self.settings_window, self.canny_upper, 255, self.set_canny_upper)
        cv.createTrackbar("Canny lower", self.settings_window, self.canny_lower, 255, self.set_canny_lower)

    def settings_morph(self):
        pass

    def settings_threshold(self):
        cv.createTrackbar("Threshold value", self.settings_window, self.thresh_value, 255, self.set_thresh_value)

    def settings_mask(self):
        cv.createTrackbar("HSV sensitivity", self.settings_window, self.hsv_sensitivity, 255, self.set_hsv_sensitivity)

    def set_thresh_value(self, value):
        self.thresh_value = value

    def set_max_thresh_value(self, value):
        self.max_thresh_value = value

    def set_features_max_corners(self, value):
        self.features_max_corners = value

    def set_features_quality(self, value):
        self.features_quality = (int(value) / 100) + 0.01

    def set_brightness(self, value):
        self.brightness = value - 50

    def set_noise_kernel_size(self, value):
        if value % 2 == 0:
            value = value + 1
        self.noise_kernel_size = value

    def set_max_corners_distance(self, value):
        self.features_corners_distance = value

    def set_canny_lower(self, value):
        self.canny_lower = value

    def set_canny_upper(self, value):
        self.canny_upper = value

    def set_hsv_sensitivity(self, value):
        self.hsv_sensitivity = value

    def is_paper_present(self, path):
        original = cv.imread(path)
        resized = cv.resize(original, None, fx=0.1, fy=0.1)
        prepared = self.prepare_frame(resized)
        contours = self.get_contours(prepared)
        lines = self.get_lines(prepared)
        if lines is not None:
            for i in range(0, len(lines)):
                line = lines[i][0]
                cv.line(resized, (line[0], line[1]), (line[2], line[3]), (200, 255, 100), 13)
        is_present = PaperDetection(False, "Not present: Unknown reason")
        if len(contours):
            biggest = contours[0]
            draw_contours([biggest], resized)
            show_image("present", resized)
            cv.waitKey()
            # print(biggest.area_covered())
            if biggest.area_covered() > 55:
                is_present = PaperDetection(True, "Paper found on screen.")
            else:
                # contour area too small
                is_present = PaperDetection(False, "Not present: Not enough paper visible")
        else:
            # no contours found in frame
            is_present = PaperDetection(False, "Not present: No paper present in image")
        return is_present
        pass


class PaperDetection:
    def __init__(self, is_present, reason):
        self.is_present = is_present
        self.reason = reason


if __name__ == '__main__':
    images_paths = glob.glob('resources/images/*')
    detector = EdgeDetector(mode=ProcessModes.MODE_CANNY)
    # detector.images(images_paths)
    detector.use_camera()
    detector.run()

    # image = images_paths[0]
    # result = detector.is_paper_present(image)
    # print(result.is_present)
    # print(result.reason)
