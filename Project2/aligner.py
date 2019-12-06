import cv2
import imutils
import math
import numpy as np


def is_vertical(angle):
    """Assess whether the angle is close to being vertical"""
    if ((angle < 0.14) or (angle > 6.14) or (angle > 3.00 and angle < 3.28)):
        return True
    return False

def is_horizontal(angle):
    """Assess whether the angle is close to being horizontal"""
    if ((angle > 1.43 and angle < 1.71) or (angle > 4.57 and angle < 4.85)):
        return True
    return False

def compare_vert(src, pt_a, pt_b):
    """Compare the average luminosities of either side of a near-vertical line"""
    height, width = src.shape[0:2]
    comparison_offset = int(min(height, width) / 100)
    a_weight, b_weight, total = 0, 0, 0
    x, y = pt_a
    dy = pt_b[1] - pt_a[1]
    dx = pt_b[0] - pt_a[0]
    doh = np.gcd(dy, dx)
    dx = int((dx/doh)/10)
    dy = int((dy/doh)/10)
    while (y < height) and (x < width):
        a_weight += src[y][max(x-comparison_offset, 0)]
        b_weight += src[y][min(x+comparison_offset, x+(width-1-x))]
        total += 1
        x += dx
        y += dy
    a_weight /= total
    b_weight /= total
    return (a_weight, b_weight)


def compare_horiz(src, pt_a, pt_b):
    """Compare the average luminosities of either side of a near-vertical line"""
    height, width = src.shape[0:2]
    comparison_offset = int(min(height, width) / 100)
    a_weight, b_weight, total = 0, 0, 0
    x, y = pt_a
    dy = pt_b[1] - pt_a[1]
    dx = pt_b[0] - pt_a[0]
    doh = np.gcd(dy, dx)
    dx = int(dx/doh)
    dy = int(dy/doh)
    while (y < height) and (x < width):
        a_weight += src[max(y-comparison_offset, 0)][x]
        b_weight += src[min(y+comparison_offset, y+(height-1-y))][x]
        total += 1
        x += dx
        y += dy
    a_weight /= total
    b_weight /= total
    return (a_weight, b_weight)


def main():
    # Load an image, make a copy to fiddle with
    file_path = './janky3.bmp'
    original = cv2.imread(file_path)
    image = np.copy(original)
    im_height, im_width = image.shape[0:2]
    cv2.imshow("Original", image)
    cv2.waitKey()

    # Convert image to grayscale for edge-finding and luminance comparison
    grays = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_threshold = 253
    max_threshold = 255
    edges = cv2.Canny(grays, min_threshold, max_threshold, apertureSize=3)
    outer_edges = []

    # Use linear Hough transform to find lines in the image
    min_line_length = min(im_width, im_height) * 0.9
    max_line_gap = min_line_length * 0.9
    votes_required = 1
    lines = cv2.HoughLines(edges, 1, np.pi / 180, votes_required, min_line_length, max_line_gap)

    # Examine the most likely lines found
    for ln in range(10):
        for rho, theta in lines[ln]:
            if (is_vertical(theta)):
                print(theta, rho, " - I think line ", ln, "is a VERTICAL.")
            elif (is_horizontal(theta)):
                print(theta, rho, " - I think line ", ln, "is a HORIZONTAL.")
            else:
                print(theta, rho, " - I don't think line", ln, "is useful.")

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            x3 = 0
            y3 = 0
            x4 = im_width
            y4 = im_height

            pt_0 = (x0, y0)
            cv2.circle(image, pt_0, 4, (0,255,0), 2)

            # Determine the point at which the current line intersects the image boundaries
            #     (Formula courtesy of Wikipedia)
            # intercept = (
            #     ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) /
            #     ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)),
            #     ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) /
            #     ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            # )
            # For TOP intercept, x3 = 0, x4 = im_width, y3 = 0, y4 = 0

            if x1 == x2 or y1 == y2:
                print("DANGER WILL ROBINSON")
                continue
                

            # print("Coordinates: x1 =", x1, "   y1 =", y1, "   x2 =", x2, "   y2 =", y2)
            top_intercept = (
                int(((x1*y2 - y1*x2) * (0 - im_width) - (x1 - x2) * (0*0 - 0*im_width)) /
                ((x1 - x2) * (0 - 0) - (y1 - y2) * (0 - im_width))),
                0
            )
            # For BOTTOM intercept, x3 = 0, x4 = im_width, y3 = im_height, y4 = im_height
            bottom_intercept = (
                int(((x1*y2 - y1*x2) * (0 - im_width) - (x1 - x2) * (0*im_height - im_height*im_width)) /
                ((x1 - x2) * (im_height - im_height) - (y1 - y2) * (0 - im_width))),
                im_height - 1
            )
            # For LEFT intercept, x3 = 0, x4 = 0, y3 = 0, y4 = im_height
            left_intercept = (
                0,
                int(((x1*y2 - y1*x2) * (0 - im_height) - (y1 - y2) * (0*im_height - 0*0)) /
                ((x1 - x2) * (0 - im_height) - (y1 - y2) * (0 - 0)))
            )
            # For RIGHT intercept, x3 = im_width, x4 = im_width, y3 = 0, y4 = im_height
            right_intercept = (
                im_width - 1,
                int(((x1*y2 - y1*x2) * (0 - im_height) - (y1 - y2) * (im_width*im_height - 0*im_height)) /
                ((x1 - x2) * (0 - im_height) - (y1 - y2) * (im_width - im_width)))
            )

            # Any given line will intersect exactly 2 of the 4 edges
            top = (top_intercept[0] >= 0) and (top_intercept[0] <= im_width)
                # print("Line", ln, "touches the TOP edge;")
            bottom = (bottom_intercept[0] >= 0) and (bottom_intercept[0] <= im_width)
                # print("Line", ln, "touches the BOTTOM edge;")
            left = (left_intercept[1] >= 0) and (left_intercept[1] <= im_height)
                # print("Line", ln, "touches the LEFT edge;")
            right = (right_intercept[1] >= 0) and (right_intercept[1] <= im_height)
                # print("Line", ln, "touches the RIGHT edge;")

            sides = [top, bottom, left, right]
            intercepts = [top_intercept, bottom_intercept, left_intercept, right_intercept]

            # Call comparison function to evaluate luminance based on which sides the line hits
            side_a = 0
            side_b = 0
            if top and left:
                if is_vertical(theta):
                    side_a, side_b = compare_vert(grays, top_intercept, left_intercept)
                elif is_horizontal(theta):
                    side_a, side_b = compare_horiz(grays, left_intercept, top_intercept)
            elif top and right:
                if is_vertical(theta):
                    side_a, side_b = compare_vert(grays, top_intercept, right_intercept)
                elif is_horizontal(theta):
                    side_a, side_b = compare_horiz(grays, top_intercept, right_intercept)
            elif left and bottom:
                if is_vertical(theta):
                    side_a, side_b = compare_vert(grays, left_intercept, bottom_intercept)
                elif is_horizontal(theta):
                    side_a, side_b = compare_horiz(grays, left_intercept, bottom_intercept)
            elif right and bottom:
                if is_vertical(theta):
                    side_a, side_b = compare_vert(grays, right_intercept, bottom_intercept)
                elif is_horizontal(theta):
                    side_a, side_b = compare_horiz(grays, bottom_intercept, right_intercept)
            elif top and bottom and is_vertical(theta):
                side_a, side_b = compare_vert(grays, top_intercept, bottom_intercept)
            elif left and right and is_horizontal(theta):
                side_a, side_b = compare_horiz(grays, left_intercept, right_intercept)

            # If one side or the other (but not both) is mostly white, the line is probably an outer edge
            if (side_a > 225 or side_b > 225) and (np.abs(side_a - side_b) > 40):
                print(f"I am confident that line {ln} is an outer edge.")
                outer_edges.append(theta)

            cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.imshow(f'img {ln}', image)
            cv2.waitKey()

    # If only one outer edge found, assume it is to be used for alignment
    if len(outer_edges) == 1:
        print("I only found one outer edge, at angle", outer_edges[0])
        print("So I guess that's what I'll align to.")
        bangle = outer_edges[0] * 100
        # Image should be rotated slightly right or slightly left, based on angle
        if( (bangle >= 143 and bangle <= 157) or
            (bangle >= 300 and bangle <= 314) or
            (bangle >= 457 and bangle <= 471) or
            (bangle >= 614 and bangle <= 628)):
            align_to = 180 + (180/np.pi) * outer_edges[0] 
            align_to = (180/np.pi) * outer_edges[0]
        elif( (bangle >= 0 and bangle <= 14) or
            (bangle >= 157 and bangle <= 171) or
            (bangle >= 314 and bangle <= 328) or
            (bangle >= 471 and bangle <= 485)):
            align_to = (180/np.pi) * outer_edges[0]
        rotated = imutils.rotate(original, align_to)
        
        print("C'est bien?")
        cv2.imshow("Rotated image", rotated)
        cv2.waitKey()


    # If exactly 2 outer edges found, determine whether intersection is likely to be a corner
    if len(outer_edges) == 2:
        print("I found outer edges at angles", outer_edges[0], "and", outer_edges[1])
        corner_angle = np.abs(outer_edges[0] - outer_edges[1])
        print("The difference between their angles is", corner_angle)
        if corner_angle >= 1.518 and corner_angle <= 1.623:
            print("That's close to 90 degrees, so I think I'll use it for alignment.")
            rotated = imutils.rotate(original, (180/np.pi) * outer_edges[0])
            print("How does this look?")
            cv2.imshow("Rotated image", rotated)
            cv2.waitKey()

if __name__ == '__main__':
    main()