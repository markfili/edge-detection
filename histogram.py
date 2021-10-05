import cv2 as cv

main_win_name = "Histograms"


def on_threshold_change(val):
    if __name__ == '__main__':
        src = cv.imread('resources/images/20200526_121053.jpg')
        src = cv.resize(src, None, fx=0.09, fy=0.09)
        cv.namedWindow(main_win_name, cv.WINDOW_AUTOSIZE)
        cv.imshow(main_win_name, src)
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2HSV)
        src_blur = cv.GaussianBlur(src_gray, (3, 3), 0)

        cv.createTrackbar("threshold value", main_win_name, 100, 255, on_threshold_change)
        on_threshold_change(200)
        cv.waitKey()
