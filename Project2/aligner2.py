import cv2
import glob
import imutils
import numpy as np

def main():
    image_list = glob.glob('./images/*.png')
    image_list.sort()
    image_set = [cv2.imread(img) for img in image_list]
    for img in image_set:
        height, width = img.shape[0:2]
        grays = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        show(grays)
        thold = cv2.threshold(grays, 250, 255, cv2.THRESH_BINARY)
        # thold = np.uint8(thold)
        # print(type(thold), thold)
        show(thold[1])
        edges = cv2.Canny(thold[1], 254, 255)
        show(edges)
        min_line_length = min(width, height) / 2
        max_line_gap = 0 #min_line_length * 0.9
        votes_required = 1
        lines = cv2.HoughLines(edges, 1, np.pi / 180, votes_required, min_line_length, max_line_gap)
        for ln in range(2):
            rho, theta = lines[ln][0]
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
        # show(thold)


def show(thing):
    '''Shorthand method to display an image and wait; used for testing'''
    cv2.imshow("Image", thing)
    cv2.waitKey()



if __name__ == '__main__':
    main()