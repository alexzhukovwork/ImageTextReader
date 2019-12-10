import cv2
import numpy as np
from matplotlib import pyplot as plt

SPACE_BOUND = 6


def find_corners(bound):
    c1 = [bound[3][0], bound[0][1]]
    c2 = [bound[1][0], bound[0][1]]
    c3 = [bound[1][0], bound[2][1]]
    c4 = [bound[3][0], bound[2][1]]
    return [c1, c2, c3, c4]


def find_thresh(data):
    bin_size = 50
    #  find density and bounds of histogram of data
    density,  bds = np.histogram(data, bins=bin_size)
    #  normalize the histogram values
    norm_dens = density/float(sum(density))
    #  find discrete cumulative density function
    cum_dist = norm_dens.cumsum()
    #  initial values to be overwritten
    fn_min = np.inf
    thresh = -1
    bounds = range(1,  bin_size)
    #  begin minimization routine
    for itr in range(0,  bin_size):
        if itr == bin_size-1:
            break
        p1 = np.asarray(norm_dens[0:itr])
        p2 = np.asarray(norm_dens[itr+1:])
        q1 = cum_dist[itr]
        q2 = cum_dist[-1] - q1
        b1 = np.asarray(bounds[0:itr])
        b2 = np.asarray(bounds[itr:])
        # find means
        m1 = np.sum(p1*b1)/q1
        m2 = np.sum(p2*b2)/q2
        # find variance
        v1 = np.sum(((b1-m1)**2)*p1)/q1
        v2 = np.sum(((b2-m2)**2)*p2)/q2

        # calculate minimization function and replace values
        # if appropriate
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = itr

    return thresh, bds[thresh]


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# function takes two rectangles of corners and combines them into a single
# rectangle
def merge_boxes(c1, c2):
    new_rect = []
    # find new corner for the top left
    cx = min(c1[0][0], c2[0][0])
    cy = min(c1[0][1], c2[0][1])
    new_rect.append([cx, cy])
    # find new corner for the top right
    cx = max(c1[1][0], c2[1][0])
    cy = min(c1[1][1], c2[1][1])
    new_rect.append([cx, cy])
    # find new corner for bottm right
    cx = max(c1[2][0], c2[2][0])
    cy = max(c1[2][1], c2[2][1])
    new_rect.append([cx, cy])
    # find new corner for bottm left
    cx = min(c1[3][0], c2[3][0])
    cy = max(c1[3][1], c2[3][1])
    new_rect.append([cx, cy])
    return new_rect


# given a list of corners that represent the corners of a box, 
# find the center of that box
def find_center_coords(c1):
    width = abs(c1[0][0]-c1[1][0])
    height = abs(c1[0][1]-c1[3][1])
    return [c1[0][0] + (width / 2.0), c1[0][1] + (height / 2.0)]


# take two points and find their slope
def find_slope(p1, p2):
    if p1[0]-p2[0] == 0:
        return np.inf

    return (p1[1]-p2[1])/(p1[0]-p2[0])


# takes point and set of corners and checks if the point is within the bounds
def is_inside(p1, c1):
    if c1[0][0] <= p1[0] <= c1[1][0] and c1[0][1] <= p1[1] <= c1[2][1]:
        return True
    else:
        return False


def find_area(c1):
    return abs(c1[0][0]-c1[1][0])*abs(c1[0][1]-c1[3][1])


def get_lines(all_letters, incoming_image):
    all_letters.sort(key=lambda letter_item: letter_item.getY() + letter_item.getHeight())

    avg = 0
    num = 0

    for letter in all_letters:
        avg += letter.getY()
        num += 1

    avg /= num
    prev = 0
    num = 0
    error = 10

    for letter in all_letters:
        for letter_ in all_letters:
            if abs(letter_.getY() - letter.getY()) < avg - error and all_letters.__contains__(letter):
                all_letters.remove(letter)

        prev = letter.getY()
        num += 1

    lines_ = [[[]]]

    for letter in all_letters:
        lines_.append(((letter.getY(), 0), (letter.getY(), incoming_image.shape[1])))

    lines_.pop(0)

    lines_ = list(set(lines_))

    return lines_


def parse_img(image):
    binding_bx = []
    corners = []
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    th3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.bitwise_not(th3)

    contours,  heirar = cv2.findContours(th3,  cv2.RETR_CCOMP,  cv2.CHAIN_APPROX_SIMPLE)
    for num in range(0,  len(contours)):
        if heirar[0][num][3] == -1:
            left = tuple(contours[num][contours[num][:, :,  0].argmin()][0])
            right = tuple(contours[num][contours[num][:, :,  0].argmax()][0])
            top = tuple(contours[num][contours[num][:, :,  1].argmin()][0])
            bottom = tuple(contours[num][contours[num][:, :,  1].argmax()][0])
            binding_bx.append([top,  right,  bottom,  left])

    for bx in binding_bx:
        corners.append(find_corners(bx))

    err = 2
    area = []

    for corner in corners:
        area.append(find_area(corner))

    area = np.asarray(area)
    avg_area = np.mean(area)
    std_area = np.std(area)
    """
    :type outlier: list
    """
    outlier = (area < avg_area - std_area)

    for num in range(0,  len(outlier)):
        dot = False
        if outlier[num]:
            black = np.zeros((len(image), len(image[0])), np.uint8)
            cv2.rectangle(black,  (corners[num][0][0],  corners[num][0][1]),  (corners[num][2][0],  corners[num][2][1]), 
                          (255,  255),  -1)
            fin = cv2.bitwise_and(th3,  black)
            temp_cnt,  temp_h = cv2.findContours(fin,  cv2.RETR_CCOMP,  cv2.CHAIN_APPROX_SIMPLE)
            for cnt in temp_cnt:
                rect = cv2.minAreaRect(cnt)
                axis1 = rect[1][0] / 2.0
                axis2 = rect[1][1] / 2.0
                if (axis1 != 0 and axis2 != 0):
                    ratio = axis1 / axis2
                    #  calculate ratio of axis
                    #  if ratio is close to 1 (circular),  then most likely a dot
                    if 1.0 - err < ratio < err + 1.0:
                        dot = True
            #  if contour is a dot,  we want to connect it to the closest
            #  bounding box that is beneath it
            if dot:
                best_corner = corners[num]
                closest = np.inf
                for crn in corners:
                    #  go through each set of corners
                    #  find width and height of bounding box
                    width = abs(crn[0][0] - crn[1][0])
                    height = abs(crn[0][1] - crn[3][1])
                    #  check to make sure character is below in position (greater y value)
                    if corners[num][0][1] > crn[0][1]:
                        continue
                        #  if it's above the dot we don't care
                    elif dist(corners[num][0],  crn[0]) < closest and crn != corners[
                        num]:
                        #  and (findSlope(findCenterCoor(corners[num]), crn[0])) > 0:
                        #  if(findArea(mergeBoxes(corners[num], crn))> avg_area+std_area):
                        #      continue
                        #  check the distance if it is below the dot
                        cent = find_center_coords(crn)
                        best_corner = crn
                        closest = dist(corners[num][0],  crn[0])
                #  modify the coordinates of the pic to include the dot
                #  print(best_corner)
                new_corners = merge_boxes(corners[num], best_corner)
                corners.append(new_corners)
                #  print(new_corners)
                corners[num][0][0] = 0
                corners[num][0][1] = 0
                corners[num][1][0] = 0
                corners[num][1][1] = 0
                corners[num][2][0] = 0
                corners[num][2][1] = 0
                corners[num][3][0] = 0
                corners[num][3][1] = 0
                best_corner[0][0] = 0
                best_corner[0][1] = 0
                best_corner[1][0] = 0
                best_corner[1][1] = 0
                best_corner[2][0] = 0
                best_corner[2][1] = 0
                best_corner[3][0] = 0
                best_corner[3][1] = 0

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #  Take letters and turn them into objects
    AllLetters = []
    counter = 0
    d = 0
    index = 0
    lastCornerX = 0

    preparing = {corners[i][0][0]: corners[i] for i in range(0,  len(corners))}
    sortedDict = dict(sorted(preparing.items(),  key=lambda kv: (kv[1],  kv[0])))

    #  if (sortedDict.__contains__(0)):
    #     sortedDict.pop(0)
    #   values = list(sortedDict.values())
    #   letters = [[]]
    #    word_count = 0

    for bx in sortedDict.values():
        width = abs(bx[1][0] - bx[0][0])
        height = abs(bx[3][1] - bx[0][1])

        if width * height == 0:
            continue

        # newLetter = Letter.Letter([bx[0][0], bx[0][1]], [height, width], counter)
        # AllLetters.append(newLetter)
        #
        # counter += 1
        # c = 1
        # crop_img = th3[bx[0][1] - c:bx[3][1] + c, bx[0][0] - c:bx[1][0] + c]
        # plt.imshow(crop_img, 'gray')
        # corner = bx[0][0]
        # distVal = corner - lastCornerX
        #
        # if index > 0:
        #     print(str(index) + " " + str(distVal))
        #     d += distVal
        #
        # lastCornerX = bx[1][0]
        # index += 1
        #
        # letters[word_count].append(crop_img)

    return AllLetters


if __name__ == "__main__":
    img = cv2.imread('TwoLines.png', 0)
    AllLetters = parse_img(img)
    lines = get_lines(AllLetters, img)

    for i in range(len(lines)):
        imgLine = img[lines[i][0][0]:lines[i][0][0] + lines[i][1][0],  lines[i][0][1]:lines[i][0][1] + lines[i][1][1]]
        plt.imshow(imgLine)
        plt.show()
        letters = parse_img(imgLine)

        for l in letters:
            IMG = imgLine[l.getY():l.getY()+l.getHeight(),  l.getX():l.getX()+l.getWidth()]
            plt.imshow(IMG)
            plt.show()






