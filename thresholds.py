import cv2 as cv
import numpy as np

main_win_name = "threshold"
titles = {
    0: "Binary",
    1: "Binary Inverted",
    2: "Threshold Truncated",
    3: "Threshold to Zero",
    4: "Threshold to Zero Inverted",
    7: "Threshold Mask",
    8: "Threshold Otsu",
}


def on_trackbar_change(val):
    # 0: Binary
    # 1: Binary Inverted
    # 2: Threshold Truncated
    # 3: Threshold to Zero
    # 4: Threshold to Zero Inverted
    # 8: Threshold Otsu
    for threshold_type in 0, 1, 2, 3, 4, 8:

        thresh = src_blur

        threshold_value = cv.getTrackbarPos("Threshold", main_win_name)
        threshold_value, thresh = cv.threshold(thresh, threshold_value, 255, threshold_type)

        de_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        iterations = cv.getTrackbarPos("Iterations", main_win_name)
        thresh = cv.dilate(thresh, de_kernel, iterations=iterations)

        thresh = cv.erode(thresh, de_kernel, iterations=iterations)

        canny = cv.getTrackbarPos("Canny", main_win_name)
        thresh = cv.Canny(thresh, canny, canny * 3)

        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        max_corners = cv.getTrackbarPos("Corners", main_win_name)
        # corners = cv.goodFeaturesToTrack(thresh, max_corners, 0.01, 100)

        thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)

        hulls_corners = ''
        paper_area = 0
        for contour in contours:
            hull = cv.convexHull(contour)

            hull_approx = cv.approxPolyDP(hull, 3, True)
            hull_approx_area = cv.contourArea(hull_approx)
            if (len(hull_approx) == 4) & (hull_approx_area > paper_area):
                paper_area = hull_approx_area

            hulls_corners += '{},'.format(len(hull_approx))
            # rect = cv.minAreaRect(hull)
            # box = cv.boxPoints(rect)
            # box = np.int0(box)
            # cv.drawContours(thresh, [box], -1, (0, 0, 255), thickness=3)
            # cv.drawContours(thresh, [contour], -1, (0, 255, 0), thickness=2)
            cv.drawContours(thresh, [hull_approx], -1, (255, 0, 255), thickness=4)
            # cv.drawContours(thresh, [hull], -1, (255, 0, 0), thickness=4)

        cv.putText(thresh, "contours count {}".format(len(contours)), (10, 10), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        cv.putText(thresh, "all approx corners {}".format(hulls_corners), (10, 30), cv.FONT_HERSHEY_PLAIN, 1,
                   (255, 255, 255))
        cv.putText(thresh, "threshold {}".format(threshold_value), (10, 50), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        cv.putText(thresh, "img {} {} {}".format(len(thresh[0]), len(thresh), len(thresh[0]) * len(thresh)), (10, 70), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        cv.putText(thresh, "paper area {}".format(paper_area), (10, 90), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        # lines = cv.HoughLinesP(thresh, 1, np.pi / 180, threshold_value, None, 0, 0)
        # if lines is not None:
        #     for i in range(0, len(lines)):
        #         line = lines[i][0]
        #         cv.line(thresh, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 3)

        # corners = np.int0(corners)
        # for i in corners:
        #     x, y = i.ravel()
        #     cv.circle(thresh, (x, y), 10, 255, -1)

        y = 0
        x_multiplier = threshold_type

        if threshold_type > 4:
            y = len(src)
            x_multiplier = threshold_type - 7

        x = x_multiplier * len(src[0])

        win_name = "{} {}".format(main_win_name, titles.get(threshold_type))
        cv.imshow(win_name, thresh)
        cv.moveWindow(win_name, x, y)


if __name__ == '__main__':
    print(cv.getVersionString())
    src = cv.imread('resources/images/20200526_121130.jpg', cv.IMREAD_GRAYSCALE)
    src = cv.resize(src, None, fx=0.09, fy=0.09)
    cv.namedWindow(main_win_name, cv.WINDOW_AUTOSIZE)
    cv.imshow(main_win_name, src)
    # src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_blur = cv.GaussianBlur(src, (7, 7), 0)
    threshold = 190
    cv.createTrackbar("Threshold", main_win_name, threshold, 255, on_trackbar_change)
    cv.createTrackbar("Canny", main_win_name, 60, 255, on_trackbar_change)
    cv.createTrackbar("Iterations", main_win_name, 2, 10, on_trackbar_change)
    cv.createTrackbar("Corners", main_win_name, 4, 50, on_trackbar_change)
    on_trackbar_change(-1)
    cv.waitKey()
