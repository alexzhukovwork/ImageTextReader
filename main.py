import os, sys
from PIL import Image
import Letter
import cv2
import glob
import numpy as np
import NN.nn
import time
from numba import cuda
import multiprocessing as mp

SPACE_BOUND = 6
LETTER_NUM = 32


def to_letter(label: list):
    """
    :raise: ValueError
    """
    return chr(ord('А') + label.index(1.))


def findCorners(bound):
    c1 = [bound[3][0], bound[0][1]]
    c2 = [bound[1][0], bound[0][1]]
    c3 = [bound[1][0], bound[2][1]]
    c4 = [bound[3][0], bound[2][1]]
    return [c1, c2, c3, c4]


def findThresh(data):
    Binsize = 50
    # find density and bounds of histogram of data
    density, bds = np.histogram(data, bins=Binsize)
    # normalize the histogram values
    norm_dens = (density) / float(sum(density))
    # find discrete cumulative density function
    cum_dist = norm_dens.cumsum()
    # initial values to be overwritten
    fn_min = np.inf
    thresh = -1
    bounds = range(1, Binsize)
    # begin minimization routine
    for itr in range(0, Binsize):
        if (itr == Binsize - 1):
            break;
        p1 = np.asarray(norm_dens[0:itr])
        p2 = np.asarray(norm_dens[itr + 1:])
        q1 = cum_dist[itr]
        q2 = cum_dist[-1] - q1
        b1 = np.asarray(bounds[0:itr])
        b2 = np.asarray(bounds[itr:])
        # find means
        m1 = np.sum(p1 * b1) / q1
        m2 = np.sum(p2 * b2) / q2
        # find variance
        v1 = np.sum(((b1 - m1) ** 2) * p1) / q1
        v2 = np.sum(((b2 - m2) ** 2) * p2) / q2

        # calculate minimization function and replace values
        # if appropriate
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = itr

    return thresh, bds[thresh]


def dist(P1, P2):
    return np.sqrt((P1[0] - P2[0]) ** 2 + (P1[1] - P2[1]) ** 2)


# function takes two rectangles of corners and combines them into a single
# rectangle
def mergeBoxes(c1, c2):
    newRect = []
    # find new corner for the top left
    cx = min(c1[0][0], c2[0][0])
    cy = min(c1[0][1], c2[0][1])
    newRect.append([cx, cy])
    # find new corner for the top right
    cx = max(c1[1][0], c2[1][0])
    cy = min(c1[1][1], c2[1][1])
    newRect.append([cx, cy])
    # find new corner for bottm right
    cx = max(c1[2][0], c2[2][0])
    cy = max(c1[2][1], c2[2][1])
    newRect.append([cx, cy])
    # find new corner for bottm left
    cx = min(c1[3][0], c2[3][0])
    cy = max(c1[3][1], c2[3][1])
    newRect.append([cx, cy])
    return newRect


# given a list of corners that represent the corners of a box,
# find the center of that box
def findCenterCoor(c1):
    width = abs(c1[0][0] - c1[1][0])
    height = abs(c1[0][1] - c1[3][1])
    return ([c1[0][0] + (width / 2.0), c1[0][1] + (height / 2.0)])


# take two points and find their slope
def findSlope(p1, p2):
    if (p1[0] - p2[0] == 0):
        return np.inf

    return (p1[1] - p2[1]) / (p1[0] - p2[0])


# takes point and set of corners and checks if the point is within the bounds
def isInside(p1, c1):
    if (p1[0] >= c1[0][0] and p1[0] <= c1[1][0] and p1[1] >= c1[0][1] and p1[1] <= c1[2][1]):
        return True
    else:
        return False


def findArea(c1):
    return abs(c1[0][0] - c1[1][0]) * abs(c1[0][1] - c1[3][1])


def getLines(AllLetters, img):
    AllLetters.sort(key=lambda letter: letter.getY() + letter.getHeight())

    avg = 0
    num = 0

    for letter in AllLetters:
        avg += letter.getHeight()
        num += 1

    avg /= num
    num = 0
    error = avg / 10
    max_height = 0

    for letter in AllLetters:
        if max_height < letter.getHeight():
            max_height = letter.getHeight()

        for l in AllLetters:
            if abs(l.getY() - letter.getY()) < avg + error:
                AllLetters.remove(l)

        num += 1

    lines = [[[]]]

    for letter in AllLetters:
        lines.append(((letter.getY(), 0), (max_height, img.shape[1])))

    lines.pop(0)

    lines = list(set(lines))
    lines.sort(key=lambda letter: letter[0][0])

    return lines


def parseImg(img):
    bndingBx = []
    corners = []
    # blur = cv2.GaussianBlur(img, (5, 5), 0)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)
    th3 = cv2.bitwise_not(th3)

    err = 0

    contours, heirar = cv2.findContours(th3, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for num in range(0, len(contours)):
        if (heirar[0][num][3] == -1):
            left = tuple(contours[num][contours[num][:, :, 0].argmin()][0])
            right = tuple(contours[num][contours[num][:, :, 0].argmax()][0])
            top = tuple(contours[num][contours[num][:, :, 1].argmin()][0])
            bottom = tuple(contours[num][contours[num][:, :, 1].argmax()][0])
            bndingBx.append([top, right, bottom, left])

    for bx in bndingBx:
        corners.append(findCorners(bx))

    Area = []

    for corner in corners:
        Area.append(findArea(corner))

    Area = np.asarray(Area)
    avgArea = np.mean(Area)
    stdArea = np.std(Area)
    outlier = (Area < avgArea - stdArea)

    for num in range(0, len(outlier)):
        dot = False
        if (outlier[num]):
            black = np.zeros((len(img), len(img[0])), np.uint8)
            cv2.rectangle(black, (corners[num][0][0], corners[num][0][1]), (corners[num][2][0], corners[num][2][1]),
                          (255, 255), -1)
            fin = cv2.bitwise_and(th3, black)
            tempCnt, tempH = cv2.findContours(fin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in tempCnt:
                rect = cv2.minAreaRect(cnt)
                axis1 = rect[1][0] / 2.0
                axis2 = rect[1][1] / 2.0
                if (axis1 != 0 and axis2 != 0):
                    ratio = axis1 / axis2  # calculate ratio of axis
                    # if ratio is close to 1 (circular), then most likely a dot
                    if ratio > 1.0 - err and ratio < err + 1.0:
                        dot = True
            # if contour is a dot, we want to connect it to the closest
            # bounding box that is beneath it
            if dot:
                bestCorner = corners[num]
                closest = np.inf
                for crn in corners:  # go through each set of corners
                    # find width and height of bounding box
                    width = abs(crn[0][0] - crn[1][0])
                    height = abs(crn[0][1] - crn[3][1])
                    # check to make sure character is below in position (greater y value)
                    if (corners[num][0][1] > crn[0][1]):
                        continue  # if it's above the dot we don't care
                    elif dist(corners[num][0], crn[0]) < closest and crn != corners[
                        num]:  # and (findSlope(findCenterCoor(corners[num]),crn[0])) > 0:
                        # if(findArea(mergeBoxes(corners[num],crn))> avgArea+stdArea):
                        #     continue
                        # check the distance if it is below the dot
                        cent = findCenterCoor(crn)
                        bestCorner = crn
                        closest = dist(corners[num][0], crn[0])
                # modify the coordinates of the pic to include the dot
                # print(bestCorner)
                newCorners = mergeBoxes(corners[num], bestCorner)
                corners.append(newCorners)
                # print(newCorners)
                corners[num][0][0] = 0
                corners[num][0][1] = 0
                corners[num][1][0] = 0
                corners[num][1][1] = 0
                corners[num][2][0] = 0
                corners[num][2][1] = 0
                corners[num][3][0] = 0
                corners[num][3][1] = 0
                bestCorner[0][0] = 0
                bestCorner[0][1] = 0
                bestCorner[1][0] = 0
                bestCorner[1][1] = 0
                bestCorner[2][0] = 0
                bestCorner[2][1] = 0
                bestCorner[3][0] = 0
                bestCorner[3][1] = 0

    ###############################################
    # Take letters and turn them into objects
    AllLetters = []
    counter = 0
    d = 0
    index = 0
    lastCornerX = 0

    #  preparing = {corners[i][0][0]: corners[i] for i in range(0, len(corners))}
    #  sortedDict = dict(sorted(preparing.items(), key=lambda kv: (kv[1], kv[0])))

    # if (sortedDict.__contains__(0)):
    #    sortedDict.pop(0)

    #  values = list(sortedDict.values())

    #  letters = [[]]
    #   word_count = 0

    for bx in corners:
        width = abs(bx[1][0] - bx[0][0])
        height = abs(bx[3][1] - bx[0][1])

        if width * height == 0:
            continue

        newLetter = Letter.Letter([bx[0][0], bx[0][1]], [height, width], counter)
        AllLetters.append(newLetter)
    #      counter += 1
    #       c = 1
    #     crop_img = th3[bx[0][1] - c:bx[3][1] + c, bx[0][0] - c:bx[1][0] + c]
    # plt.imshow(crop_img, 'gray')
    #        corner = bx[0][0]
    #      distVal = corner - lastCornerX

    #       if index > 0:
    #          print(str(index) + " " + str(distVal))
    #           d += distVal

    #      lastCornerX = bx[1][0]
    #     index += 1

    #     letters[word_count].append(crop_img)

    AllLetters.sort(key=lambda letter: letter.getX())

    return AllLetters


def resize(str):
    size = 128, 128

    outfile = str

    try:
        im = Image.open(str)
        im.thumbnail(size, Image.ANTIALIAS)
        im.save(outfile, "JPEG")
    except IOError:
        print
        "cannot create thumbnail for '%s'" % str


def preprocess_image(file):
    image = cv2.imread(file)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)
    return gray_image


def parse_path(path):
    items = []
    labels = []
    for dir_name in range(0, LETTER_NUM):
        real_path = f"Data/{path}/{str(dir_name)}/*.png"
        files = glob.glob(real_path)
        for file in files:
            gray_image = preprocess_image(file)
            items.append(gray_image)
            label = [1. if i is dir_name else 0. for i in range(0, LETTER_NUM)]
            labels.append(label)
    items = np.array(items, dtype='int32')  # as mnist
    labels = np.array(labels, dtype='int32')  # as mnist
    items = np.reshape(items, [items.shape[0], items.shape[1] * items.shape[2]])
    return items, labels


def img_list_to_nparray(imgs):
    items = []

    for img in imgs:
        #       gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        items.append(gray_image)

    items = np.array(items, dtype='int32')  # as mnist
    items = np.reshape(items, [items.shape[0], items.shape[1] * items.shape[2]])
    items = items.swapaxes(0, 1)
    return items


def prepare_nn_data():
    train_items, train_labels = parse_path("Train")
    test_items, test_labels = parse_path("Test")
    return train_items, train_labels, test_items, test_labels


def is_trained():
    return os.path.exists("Weights.npy")


def prepare_train_data(path, need_generate_new_data=False):
    if need_generate_new_data:
        img = cv2.imread(path, 0)

        AllLetters = parseImg(img)

        lines = getLines(AllLetters, img)
        error = 10

        for i in range(len(lines)):
            line_images = img[lines[i][0][0] - error:lines[i][0][0] + lines[i][1][0] + error,
                          lines[i][0][1]:lines[i][0][1] + lines[i][1][1]]

            im = Image.fromarray(line_images)
            im.save("Letters/" + str(i) + ".jpeg")
            letters = parseImg(line_images)
            j = 0

            for l in letters:
                img_in_line = line_images[l.getY():l.getY() + l.getHeight(), l.getX():l.getX() + l.getWidth()]
                if not os.path.exists("Data/Train/" + str(j)):
                    os.mkdir("Data/Train/" + str(j))

                resized = cv2.resize(img_in_line, (28, 28), interpolation=cv2.INTER_AREA)
                im = Image.fromarray(resized)

                im.save("Data/Train/" + str(j) + "/" + str(i) + " " + str(j) + ".png")

                j += 1

    x_train, y_train, x_test, y_test = prepare_nn_data()
    # plt.figure(figsize=[6, 6])

    x_train = x_train.swapaxes(0, 1)
    y_train = y_train.swapaxes(0, 1)
    x_test = x_test.swapaxes(0, 1)
    y_test = y_test.swapaxes(0, 1)

    return x_train, y_train, x_test, y_test


def get_weights():
    if not is_trained():
        x_train, y_train, x_test, y_test = prepare_train_data("DataSet.png")

        weights = NN.nn.model(x_train, y_train, 1000, 0.001)
        save_weights(weights)
    else:
        weights = np.load("Weights.npy", allow_pickle=True)
    #
    return weights


def save_weights(weights):
    np.save("Weights.npy", weights, allow_pickle=True)


def print_prediction(prediction):
    string = ""
    for p in prediction:
        # a = np.mean(p == y_train[0])
        string += to_letter(p.tolist())

    return string


def print_letters(letters):
    final_str = ""

    for letter in letters:
        for word in letter:
            prepared = img_list_to_nparray(word)
            prediction = NN.nn.check(prepared, weights).swapaxes(0, 1)
            final_str += print_prediction(prediction) + " "

        final_str += "\n"

    final_str = final_str.replace("ЬЫ", "Ы")
    final_str = final_str.replace("Ы ", "Ы")
    final_str = final_str.replace(" Ы", "Ы ")
    final_str = final_str.replace("Й", "И")

    print(final_str)


def split_word(lines_img):
    lines_img.pop(lines_img.__len__() - 1)

    letters = [[[]]]

    letters.pop(0)
    word_number = 0
    prevX = 0

    avg = 0
    word_count = 0

    for i in range(len(lines_img)):

        for j in range(len(lines_img[i])):
            l = lines_img[i][j]

            avg += l[1].getWidth()

            word_count += 1

    prevLetter = 0

    for i in range(len(lines_img)):
        letters.append([[]])
        word_number = 0
        for j in range(len(lines_img[i])):
            l = lines_img[i][j]
            currentX = l[1].getX()

            if j > 0 and currentX - prevX > prevLetter + prevLetter / 1.8:
                word_number += 1
                letters[i].append([])

            letters[i][word_number].append(l[0])
            prevLetter = l[1].getWidth()
            prevX = l[1].getX()

    return letters


def get_lines_img(img, lines):
    error = 0
    lines_img = [[]]

    for i in range(len(lines)):
        line_images = img[lines[i][0][0] - error:lines[i][0][0] + lines[i][1][0] + error,
                      lines[i][0][1]:lines[i][0][1] + lines[i][1][1]]
        letters = parseImg(line_images)

        lines_img.append([])

        for j in range(len(letters)):
            img_in_line = line_images[
                          letters[j].getY():letters[j].getY() + letters[j].getHeight(),
                          letters[j].getX():letters[j].getX() + letters[j].getWidth()
                          ]

            resized = cv2.resize(img_in_line, (28, 28), interpolation=cv2.INTER_AREA)
            lines_img[i].append([resized, letters[j]])

    return lines_img


results = []
letters = []

def get_lines_img_async(img, lines, pool):
    results.clear()
    letters.clear()

    for i in range(len(lines)):
        pool.apply_async(lines_async, args=(img, lines[i], i), callback=callback_lines)
        #lines_async(img, lines[i], i, pool)

    pool.close()
    pool.join()

    pool = mp.Pool(mp.cpu_count())

    for i in range(len(letters)):
        for j in range(len(letters[i][0])):
            pool.apply_async(get_letter, args=(letters[i][1], letters[i][0][j], i, j), callback=collect_result)
        #collect_result(get_letter(line_images, letters[j], i, j))

    pool.close()
    pool.join()

    return results


def lines_async(img, l, i):

    line_images = img[
                  l[0][0]:l[0][0] + l[1][0],
                  l[0][1]:l[0][1] + l[1][1]
                  ]

    return [[parseImg(line_images), line_images], i]


def callback_lines(letter):
    i = letter[1]

    while len(letters) <= i:
        letters.append([])

    letters[i] = letter[0]


def get_letter(line_images, l, i, j):
    img_in_line = line_images[
                  l.getY():l.getY() + l.getHeight(),
                  l.getX():l.getX() + l.getWidth()]

    resized = cv2.resize(img_in_line, (28, 28), interpolation=cv2.INTER_AREA)
    return [[resized, l], i, j]


def collect_result(result):
    i = result[1]
    j = result[2]

    while len(results) <= i:
        results.append([])

    while len(results[i]) <= j:
        results[i].append([])

    results[i][j] = result[0]


if __name__ == "__main__":
    pool = mp.Pool(2)

    weights = get_weights()
    img = cv2.imread("test4.png", 0)

    start = time.time()

    lines = getLines(parseImg(img), img)
    lines_img = get_lines_img_async(img, lines, pool)

 #   lines_img = get_lines_img(img, lines)

    letters = split_word(lines_img)

    print_letters(letters)

    print(time.time() - start)

    print("Number of processors: ", mp.cpu_count())
