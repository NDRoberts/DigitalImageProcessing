import cv2
import glob
import imutils
import numpy as np

def main():
    image_list = glob.glob('./real_test/*.jpg')
    image_list.sort()
    image_set = [cv2.imread(img) for img in image_list]
    for img in image_set:
        height, width = img.shape[0:2]
        horizontals = []
        verticals = []
        grays = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        show(grays)
        thold = cv2.threshold(grays, 240, 255, cv2.THRESH_BINARY)
        show(thold[1])

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
        # for _ in range(5):
        #     cv2.dilate(thold[1], kernel, thold[1])
        #     show(thold[1])
        # for _ in range(5):
        #     cv2.erode(thold[1], kernel, thold[1])
        #     show(thold[1])

        edges = cv2.Canny(thold[1], 250, 255)
        show(edges)
        min_line_length = min(width, height) / 2
        max_line_gap = 0 #min_line_length * 0.9
        votes_required = 1
        lines = cv2.HoughLines(edges, 1, np.pi / 180, votes_required, min_line_length, max_line_gap)
        for ln in range(5):
            rho, theta = lines[ln][0]
            print("Rho:", rho, "     Theta:", theta)
            if within(theta, 1.29, 1.85) or within(theta, 4.43, 4.99):
                horizontals.append(theta)
                print('horzontial')
            elif within(theta, 0.00, 0.28) or within(theta, 6.0, 6.28) or within(theta, 2.86, 3.42):
                verticals.append(theta)
                print('vercital')
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*a)
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*a)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        show(img)
        for x in horizontals:
            for y in verticals:
                intersect = np.abs(x - y)
                if intersect >= 1.518 and intersect <= 1.623:
                    print("True Corner found:", x, "and", y)
        exit()
        # show(thold)


def show(thing):
    '''Shorthand method to display an image and wait; used for testing'''
    cv2.imshow("Image", thing)
    cv2.waitKey()


def within(val, low, high):
    if val >= low and val < high:
        return True
    return False

if __name__ == '__main__':
    main()