import cv2
import math
import numpy
import random
from pathlib import Path

class iNDRIS:

    def __init__(self):
        print("\nIt's NDR's Imaging Suite! v0.0.1")
        print("Come, let us process an image together.")
        running = True
        while (running):
            img = choose_img()
            print("Displaying original image...")
            cv2.imshow('Original Image', img)
            mode = func_menu(img)
            #filtered = apply_filter(img)
            #hgram = gen_histogram(img)
            #print("Displaying histogram...")
            #cv2.imshow('Histogram', hgram)
            #cv2.waitKey(0)
            #print("Displaying contrast-stretched image...")
            #cs_img = stretch_contrast(img)
            #cv2.imshow('Contrast-stretched Image', cs_img)
            #cv2.waitKey(0)
            #cs_hgram = gen_histogram(cs_img)
            #print("Displaying contrast-stretched histogram...")
            #cv2.imshow('Contrast-stretched Histogram', cs_hgram)
            #cv2.waitKey(0)
            
            cv2.destroyAllWindows()
            again = input("Would you like to load another image (A), or quit (Q)? ")
            if again == 'q' or again == 'Q':
                running = False
            elif again != 'a' and again != 'A':
                print("I don't know what that means, so I'll assume you want to load another.")
        print("iNDRIS has terminated.  Have a nice day.")

filter_kernels = {
    "average (1/9)": [[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]],

    "average (1/16)": [[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]],

    "1stderiv-v": [[ 0, 0, 0],
                   [-1, 0, 1],
                   [ 0, 0, 0]],
    
    "1stderiv-h": [[0, -1,  0],
                   [0,  0,  0],
                   [0,  1,  0]],

    "laplace": [[0,  1,  0],
                [1, -4,  1],
                [0,  1,  0]],

    "sobel-v": [[-1,  0,  1],
                [-2,  0,  2],
                [-1,  0,  1]],

    "sobel-h": [[-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]],
    
    "simplesharp": [[-1, -1, -1],
                    [-1,  8, -1],
                    [-1, -1, -1]],
    
    "gauss-like": [[1,  4,  7,  4, 1],
                 [4, 16, 26, 16, 4],
                 [7, 26, 41, 26, 7],
                 [4, 16, 26, 16, 4],
                 [1,  4,  7,  4, 1]],
    
    "roberts1": [[-1, 0],
                 [ 0, 1]],
    "roberts2": [[0, -1],
                 [1, 0]],

    "lens": [[0,  0,  0, 16,  8, 16,  0,  0,  0],
            [ 0,  0,  0,  8,  4,  8,  0,  0,  0],
            [ 0,  0,  8,  4,  2,  4,  8,  0,  0],
            [16,  8,  4,  2,  1,  2,  4,  8, 16],
            [ 8,  4,  2,  1,  0,  1,  2,  4,  8],
            [16,  8,  4,  2,  1,  2,  4,  8, 16],
            [ 0,  0,  0,  4,  2,  4,  8,  0,  0],
            [ 0,  0,  0,  8,  4,  8,  0,  0,  0],
            [ 0,  0,  0, 16,  8, 16,  0,  0,  0]],
    
    "wave-v": [[-4, -2, 0, 2, 4],
               [-4, -2, 0, 2, 4],
               [-4, -2, 0, 2, 4],
               [-4, -2, 0, 2, 4],
               [-4, -2, 0, 2, 4]],
    "wave-h": [[-4, -4, -4, -4, -4],
               [-2, -2, -2, -2, -2],
               [ 0,  0,  0,  0,  0],
               [ 2,  2,  2,  2,  2],
               [ 4,  4,  4,  4,  4]]
}

affirmatives = {'y', 'yes', 'Y', 'Yes', 'YES'}
negatives = {'n', 'no', 'N', 'No', 'NO"'}

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def func_menu(image):
    available_funcs = {
        1: ["Generate Histogram", lambda i: gen_histogram(i)],
        2: ["Add Mask/Frame", lambda i: add_frame(i)],
        3: ["Stretch Contrast", lambda i: stretch_contrast(i)],
        4: ["Image Negative", lambda i: img_negative(i)],
        5: ["Apply Spatial Filter", lambda i: apply_filter(i)],
        6: ["Threshhold", lambda i: threshhold(i)],
        7: ["Circular Mask", lambda i: circle_mask(i)],
        8: ["Log Transformation", lambda i: log_transform(i)],
        9: ["Gamma Adjustment", lambda i: gamma_adjust(i)],
        10: ["Otsu's Method Threshold", lambda i: thresh_by_otsu(i)]
    }
    print("\nThe following functions are available.  Please choose one.")
    for func in available_funcs.keys():
        print(func, ": ", available_funcs[func][0])
    selected_func = int(input(">> "))
    while selected_func not in available_funcs.keys():
        print("I don't know how to do that.  Please choose a known function.")
        selected_func = input(">> ")
    available_funcs[selected_func][1](image)

    return available_funcs[selected_func]

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def choose_img():
    '''Select an image from one of the recognized source folders'''
    proceed = False
    folders = {
        'c': ".\\custom_images\\",
        'C': ".\\custom_images\\",
        's': ".\\standard_test_images\\",
        'S': ".\\standard_test_images\\",
        'o': ".\\output_images\\",
        'O': ".\\output_images\\",
        'w': ".\\wikimedia_images\\",
        'W': ".\\wikimedia_images\\"
    }
    print("Would you like to load a [S]tandard image, a [C]ustom image, a previous [O]utput image, or an image from [W]ikimedia Commons?")
    source = input(">> ")
    while source not in folders.keys():
        print("I'm sorry, I didn't understand that.  Please choose a recognized image source.")
        source = input(">> ")
    img_lib = folders[source]    
    img_list = list_files(img_lib)

    print("Please enter the number of the image to process.")
    img_num = int(input(">> "))
    while img_num not in range(len(img_list)):
        print("That's not even a real number.  Try again.")
        img_num = int(input(">> "))
    print("Selected: ", img_list[img_num])
    img_name = ".\\" + str(img_list[img_num])

    img = cv2.imread(img_name)
    return img

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def choose_filter():
    '''Select a filter from available kernels'''
    print("\nChoose an available filter.")
    print("(Please note that standard spatial filters may take some time.)")
    for key in filter_kernels.keys():
        print(key)
    filter = input(">> ")

    while filter.lower() not in filter_kernels.keys():
        print("I'm not familiar with that filter.  Please choose one from the menu.")
        filter = input(">> ")

    return filter_kernels[filter]

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def gen_histogram(source):
    '''Generate a displayable histogram for a given image at a given scale'''
    width, height, depth = source.shape[0:3]
    print("Please enter a display scale (256 or 512 recommended).")
    scale = int(input(">> "))
    hist_scale = scale / 256
    hists = [0 for i in range(256)]

    for i in range(width):
        for j in range(height):
            hists[source[i][j][0]] += 1
    
    scalar = max(hists)/scale
    rel_hists = [0 for k in range(256)]
    for k in range(256):
        rel_hists[k] = int(hists[k]/scalar)
    
    hist_img = [[[255 for m in range(scale)] for n in range(scale)] for o in range(depth)]

    for row in range(scale):
        for col in range(scale):
            if row <= rel_hists[int(col/hist_scale)]:
                hist_img[scale - row - 1][col] = 0
    
    hist_img = numpy.uint8(hist_img)
    print("Displaying histogram...")
    cv2.imshow("Histogram", hist_img)
    cv2.waitKey(0)
    save_check(hist_img)
    
    return hist_img

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def list_files(base):
    '''List all available files in a given base folder'''
    files_list = []
    for path in Path(base).iterdir():
        files_list.append(path)
    for i in range(len(files_list)):
        print(i, ": ", files_list[i])
    return files_list

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def stretch_contrast(image):
    '''Stretch image contrast'''
    width, height, depth = image.shape[0:3]
    abs_max = 255
    abs_min = 0
    rel_max = 0
    rel_min = 255
    contrasted = [[[0 for k in range(depth)] for j in range(height)] for i in range(width)]
    for i in range(width):
        for j in range(height):
            for k in range(depth):
                if image[i][j][k] > rel_max:
                    rel_max = image[i][j][k]
                if image[i][j][k] < rel_min:
                    rel_min = image[i][j][k]
    for i in range(width):
        for j in range(height):
            for k in range(depth):
                contrasted[i][j][k] = (image[i][j][k]-rel_min) * ((abs_max-abs_min)/(rel_max-rel_min))
    contrasted = numpy.uint8(contrasted)
    print("Displaying contrast-stretched image...")
    cv2.imshow("Contrast-stretched image", contrasted)
    cv2.waitKey(0)
    save_check(contrasted)

    return contrasted

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def add_frame(image):
    print("Feature not yet implemented.  Try again later.")

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def apply_filter(image):
    '''Filter an image based on a kernel; size of kernel may vary, but must be square'''
    width, height, depth = image.shape[0:3]
    filtering = True
    total_filters = 0
    filter_pass = {}
    final_img = numpy.zeros((width, height, depth))
    
    while filtering:
        kernel = choose_filter()
        k_bound = int(len(kernel)/2)
        result = numpy.zeros((width, height, depth))
        rel_max = 0
        rel_min = 0
        for m in range(width):
            for n in range(height):
                for q in range(depth):
                    new_pix_value = 0
                    weight = 0
                    for o in range(len(kernel)):
                        for p in range(len(kernel[o])):
                            target_x = m + (o - k_bound)
                            target_y = n + (p - k_bound)
                            weight += kernel[o][p]
                            if target_x < 0 or target_x >= width:
                                target_x = m
                            if target_y < 0 or target_y >= height:
                                target_y = n
                            new_pix_value += image[target_x][target_y][q] * kernel[o][p]
                            if new_pix_value > rel_max:
                                rel_max = new_pix_value
                            if new_pix_value < rel_min:
                                rel_min = new_pix_value
                    if weight > 0:
                        new_pix_value = int(new_pix_value / weight)
                    result[m][n][q] = new_pix_value
        for m in range(width):
            for n in range(height):
                for o in range(depth):
                    # result[m][n][o] = 255 * ((result[m][n][o]-rel_min) / (rel_max - rel_min))
                    if result[m][n][o] < 0:
                        result[m][n][o] = 0
        result = numpy.uint8(result)
        print("Displaying filtered image...")
        cv2.imshow('Filtered Image', result)
        cv2.waitKey(0)
        filter_again = input("Would you like to apply an additional filter? (Y/N)\n>> ")
        if filter_again.lower() not in affirmatives:
            filtering = False
        filter_pass[total_filters] = result
        total_filters += 1
    print("Combining selected filters...")
    for img in filter_pass:
        final_img += filter_pass[img]
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if final_img[x][y][z] > 255:
                    final_img[x][y][z] =255
                elif final_img[x][y][z] < 0:
                    final_img[x][y][z] = 0
    final_img = numpy.uint8(final_img)
    print("Displaying final result...")
    cv2.imshow("Final Result",final_img)
    cv2.waitKey(0)
    save_check(final_img)

    return final_img

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def threshhold(image):
    width, height, depth = image.shape[0:3]
    thresh = numpy.zeros((width, height, depth))
    print("Please enter a threshhold value between 0 and 255.")
    tpoint = int(input(">> "))
    while tpoint not in range(256):
        print("0 to 255, if you please.")
        tpoint = int(input(">> "))
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if image[x][y][z] > tpoint:
                    thresh[x][y][z] = 255
                else:
                    thresh[x][y][z] = 0
    thresh = numpy.uint8(thresh)
    print("Displaying threshhold-adjusted image...")
    cv2.imshow("Threshhold-adjusted image", thresh)
    cv2.waitKey(0)
    save_check(thresh)
    return thresh

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def circle_mask(image):
    width, height, depth = image.shape[0:3]
    masked = numpy.zeros((width, height, depth))
    center = [int(width/2), int(height/2)]
    print("This image is", width, "by", height, ".")
    print("Please enter an outer radius.")
    orad = int(input(">> "))
    print("Please enter an inner radius (0 for no inner mask).")
    irad = int(input(">> "))
    for x in range(width):
        for y in range(height):
            for z in range (depth):
                distance = int(((x - center[0])**2 + (y - center[1])**2)**(0.5))
                if (distance > orad) or (distance < irad):
                    masked[x][y][z] = 0
                else:
                    masked[x][y][z] = image[x][y][z]
    masked = numpy.uint8(masked)
    print("Displaying masked image...")
    cv2.imshow("Masked image", masked)
    cv2.waitKey(0)
    save_check(masked)
    return masked

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def img_negative(image):
    width, height, depth = image.shape[0:3]
    negative_img = numpy.zeros((width, height, depth))
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                negative_img[x][y][z] = abs(image[x][y][z] - 255)
    negative_img = numpy.uint8(negative_img)
    print("Displaying negative image...")
    cv2.imshow("Image Negative", negative_img)
    cv2.waitKey(0)
    save_check(negative_img)
    return negative_img

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def log_transform(image):
    width, height, depth = image.shape[0:3]
    result = numpy.zeros((width, height, depth))
    print("Please enter a C value.")
    c = float(input(">> "))
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                #newval = c * math.log1p(image[x][y][z])
                result[x][y][z] = c * (math.e ** image[x][y][z])
    result = normalize(numpy.uint8(result))
    print("Displaying log-transformed image...")
    cv2.imshow("Log-transformed image", result)
    cv2.waitKey(0)
    save_check(result)
    return result

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def gamma_adjust(image):
    width, height, depth = image.shape[0:3]
    result = numpy.zeros((width, height, depth))
    print("Please enter a scalar multiple.")
    c = int(input(">> "))
    print("Please enter a value for gamma.")
    gamma = float(input(">> "))
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                result[x][y][z] = c * (image[x][y][z] ** gamma)
    result = normalize(numpy.uint8(result))
    print("Displaying gamma-adjusted image...")
    cv2.imshow("Gamma-adjusted image", result)
    cv2.waitKey(0)
    save_check(result)
    return result

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def normalize(image):
    width, height, depth = image.shape[0:3]
    normal = {}
    result = numpy.zeros((width, height, depth))
    abs_max = 255
    abs_min = 0
    rel_max = 0
    rel_min = 255
    for m in range(width):
        for n in range(height):
            for o in range(depth):
                if image[m][n][o] > rel_max:
                    rel_max = image[m][n][o]
                if image[m][n][o] < rel_min:
                    rel_min = image[m][n][o]
    for m in range(width):
         for n in range(height):
            for o in range(depth):
                result[m][n][o] = int(abs_max * ((image[m][n][o]-rel_min) / (rel_max - rel_min)))
    result = numpy.uint8(result)
    return result

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def seek_circle(image):
    width, height, depth = image.shape[0:3]
    accumulator = numpy.zeros((width, height, 360))
    found = numpy.zeros((width, height, depth))

    for x in range(width):
        for y in range(height):
            if image[x][y][0] == 255:
                for r in range(35, 45):
                    for t in range(0, 360):
                        a = int(x - r * math.cos(t * math.pi / 180))
                        b = int(y - r * math.sin(t * math.pi / 180))
                        accumulator[a][b][r] +=1

    max_intersect = [0, 0, 0]
    biggest = 0
    for a in range(512):
        for b in range(512):
            for r in range(512):
                if accumulator[a][b][r] > biggest:
                    max_intersect = [a, b, r]
                    biggest = accumulator[a][b][r]
    for x in range(width):
        for y in range(height):
            dist = int((x - max_intersect[0])**2 + (y - max_intersect[1])**2)**(0.5)
            if (dist > max_intersect[2] - 1) and (dist < max_intersect[2] + 1):
                #image[x][y][2] = 255
                #image[x][y][1] = 0
                #image[x][y][0] = 255
                found[x][y][0] += 128
                found[x][y][2] += 128
    cv2.imshow("Found circle", found)
    cv2.waitKey(0)
    return found

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def find_median(lst):
    lst.sort()
    mindex = int(len(lst) / 2)
    return int(lst[mindex])

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def get_neighborhood(image, x, y, z):
    width, height = image.shape[0:2]
    result = numpy.zeros(25)
    num = 0
    for i in range(-2, 3):
        for j in range(-2, 3):
            nx = x + i
            ny = y + j
            if nx < 0 or nx > width-1:
                nx = x
            if ny < 0 or ny > height-1:
                ny = y
            result[num] = image[nx][ny][z]
            num += 1
    return result

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def median_filter(image):
    width, height, depth = image.shape[0:3]
    result = numpy.zeros((width, height, depth))

    for x in range(width):
        for y in range(height):
            for z in range(depth):
                nbd = get_neighborhood(image, x, y, z)
                result[x][y][z] = int(find_median(nbd))
    result = numpy.uint8(result)
    return result

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def gen_hist_array(image):
    width, height, depth = image.shape[0:3]
    luv_img = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
    histo = numpy.zeros(256)
    for m in range(width):
        for n in range(height):
            histo[luv_img[m][n][0]] += 1
    return histo

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def otsu_thresh(hist):
    total_px = sum(hist)
    wcv = {}
    for t in range(256):
        w_b = sum(hist[0:t])/total_px
        w_f = sum(hist[t:256])/total_px
        mu_b = 0
        mu_f = 0
        var_b = 0
        var_f = 0

        for m in range(t):
            mu_b += (m * hist[m])
        mu_b /= max(sum(hist[0:t]), 1)
        for v in range(t):
            var_b += (v-mu_b)**2 * hist[v]
        var_b /= max(sum(hist[0:t]), 1)

        for m in range(t, 256):
            mu_f += (m * hist[m])
        mu_f /= max(sum(hist[t:256]), 1)
        for v in range(t, 256):
            var_f += (v-mu_f)**2 * hist[v]
        var_f /= max(sum(hist[t:256]), 1)

        wcv[t] = (w_b * var_b) + (w_f * var_f)
    
    index_min = 255
    for x in range(256):
        if wcv[x] < wcv[index_min]:
            index_min = x
    return index_min


# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def thresh_by_otsu(image):
    width, height, depth = image.shape[0:3]
    result = numpy.zeros((width, height))
    yuv_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    lum = yuv_img[:,:,0]
    hgram = gen_hist_array(image)
    threshy = otsu_thresh(hgram)
    print("Result:", threshy)
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if lum[x][y] > threshy:
                    result[x][y] = 255
    cv2.imshow("Otsu Threshold", result)
    cv2.waitKey()
    save_check(result)
    return result

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def greyscale(image):
    result = numpy.zeros(image.shape[0:2])
    yimg = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    result = yimg[:,:,0]
    return result

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def array_variance(source):
    totes = sum(source)
    mean = totes / len(source)
    result = []
    for i in range (len(source)-1):
        result[i] = (source[i] + source[i+1]) / 2
    return result
    
# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

def save_check(image):
    opath = "./output_images/"
    yorn = input("Would you like to save this image to a file? (Y/N)\n>> ")
    if yorn in affirmatives:
        fname = input("Please enter a name to save under: ")
        write_to = opath + fname + ".png"
        cv2.imwrite(write_to, image)

# / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / #

if __name__ == "__main__":
       iNDRIS()
