import cv2
import glob
import imutils
import numpy as np

def main():
    image_list = glob.glob('./real_test/*.jpg')
    image_list.sort()
    image_set = [cv2.imread(img) for img in image_list]
    for img in image_set:
        stangle = get_angles(img)
        # align(img, stangle)
        exit()
        # show(thold)


def get_angles(img):
    height, width = img.shape[0:2]
    horizontals = []
    verticals = []
    occurrences = []

    grays = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show(grays)
    thold = cv2.threshold(grays, 240, 255, cv2.THRESH_BINARY)
    show(thold[1])

    dkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    ekernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    cv2.dilate(thold[1], dkernel, thold[1])
    show(thold[1])
    cv2.erode(thold[1], ekernel, thold[1])
    show(thold[1])
    cv2.dilate(thold[1], dkernel, thold[1])
    show(thold[1])

    edges = cv2.Canny(thold[1], 250, 255)
    show(edges)
    min_line_length = min(width, height) / 2
    max_line_gap = 0 #min_line_length * 0.9
    votes_required = 1
    lines = cv2.HoughLines(edges, 1, np.pi / 180, votes_required, min_line_length, max_line_gap)
    for ln in range(10):
        rho, theta = lines[ln][0]
        print("Rho:", rho, "     Theta:", theta)
        if within(theta, 1.43, 1.71) or within(theta, 4.57, 4.85):
            horizontals.append(theta)
        elif within(theta, 0.00, 0.14) or within(theta, 6.14, 6.28) or within(theta, 3.00, 3.28):
            verticals.append(theta)
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
    for h in horizontals:
        if h not in get_keys(occurrences):
            occurrences.append((h, countemup(h, horizontals)))
    for v in verticals:
        if v not in get_keys(occurrences):
            occurrences.append((v, countemup(v, verticals)))
    in_order = sorted(occurrences, key=get_val, reverse=True)
    print(in_order[0][0], "is the best angle, having occurred", in_order[0][1], "times.")
    best = in_order[0][0]
    return best


def get_val(thing):
    return thing[1]


def get_keys(lst):
    result = []
    for k in range(len(lst)):
        result.append(lst[k][0])
    return result


def align(img, theta):
    '''Rotate an image based on a given angle'''
    shift = np.zeros((629, 2))
    # Table of adjustment values for rotation; depends on quadrant, which involves rho
    for i in range(629):
        if i < 14:
            shift[i] = [0, 0]
        elif i in range(143, 158):
            shift[i] = [90, 270]
        elif i in range(158, 171):
            shift[i] = [90, 270]
        elif i in range(300,315):
            shift[i] = [180, 180]
        elif i in range(315, 328):
            shift[i] = [0, 0]
        elif i in range(457,472):
            shift[i] = [0, 0]
        elif i in range(472,485):
            shift[i] = [0, 0]
        elif i in range(614, 629):
            shift[i] = [0, 0]
    angle = theta * (180 / np.pi)
    aligned = imutils.rotate_bound(img, angle + shift[int(angle * 100)])
    show(aligned)


def show(thing):
    '''Shorthand method to display an image and wait; used for testing'''
    cv2.imshow("Image", thing)
    cv2.waitKey()


def within(val, low, high):
    '''Because Python won't let me check range() of float values'''
    if val >= low and val < high:
        return True
    return False


def countemup(a, values):
    '''Tally how frequently an angle appears in a list'''
    total = 0
    for n in range(len(values)):
        if a == values[n]:
            total += 1
    return total


if __name__ == '__main__':
    main()