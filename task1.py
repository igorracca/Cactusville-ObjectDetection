import cv2
import numpy as np
import matplotlib.pyplot as plt

# TRACKBARS
THRESHOLD_TB = False
HSV_TB_CACTI = False
HSV_TB_BG = True
HSV_TB = True
CONTOUR_TB = True

# DEBUG MODE
DEBUG = True

def show(n, i):
    if DEBUG == True:
        cv2.imshow(n,i)

def empty(a):
    pass

def createTrackBars():
    if THRESHOLD_TB:
        cv2.namedWindow("Parameters")
        cv2.resizeWindow("Parameters", 640, 240)
        cv2.createTrackbar("Threshold1", "Parameters", 150, 255, empty)
        cv2.createTrackbar("Threshold2", "Parameters", 150, 255, empty)

    if HSV_TB:
        cv2.namedWindow("HSV")
        cv2.setTrackbarPos('VMax', 'HSV', 255)
        cv2.createTrackbar('HMin','HSV', 0, 179, empty) # Hue is from 0-179 for Opencv
        cv2.createTrackbar('SMin','HSV', 0, 255, empty)
        cv2.createTrackbar('VMin','HSV', 0, 255, empty)
        cv2.createTrackbar('HMax','HSV', 0, 179, empty)
        cv2.createTrackbar('SMax','HSV', 0, 255, empty)
        cv2.createTrackbar('VMax','HSV', 0, 255, empty)
        # Set default value for MAX HSV trackbars.
        cv2.setTrackbarPos('HMax', 'HSV', 179)
        cv2.setTrackbarPos('SMax', 'HSV', 255)
        cv2.setTrackbarPos('VMax', 'HSV', 255)

    if CONTOUR_TB:
        cv2.namedWindow("Contour")
        cv2.createTrackbar('Area Min', 'Contour', 0, 10000, empty)
        cv2.createTrackbar('Area Max', 'Contour', 0, 100000, empty)

def getCountours(img, img_contour):
    countours, hierarchu = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #cv2.CHAIN_APPROX_NONE

    if CONTOUR_TB:
        aMin = cv2.getTrackbarPos('Area Min', 'Contour')
        aMax = cv2.getTrackbarPos('Area Max', 'Contour')
    else:
        aMin = 1500
        aMax = 25000

    for c in countours:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if area > aMin and area < aMax:
            cv2.drawContours(img_contour, c, -1, (255, 255, 255), 3)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(img_contour, (x,y), (x+w, y+h), (255, 0, 0), 5)
        else:
            #print(area)
            print(len(approx))

def getWoodBackground (image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2.fill(255)
    img2[output == max_label] = 0
    return img2

createTrackBars()

while True:

    img = cv2.imread("HW/g1/rgb/326.jpg")
    # show("rgb", img)

    # Blur
    img_blur = cv2.blur(img, (5, 5))
    img_blur = cv2.GaussianBlur(img, (7, 7), 1)
    # show("blur", img_blur)
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    # get current positions of hsv trackbars
    hMin = cv2.getTrackbarPos('HMin', 'HSV')
    sMin = cv2.getTrackbarPos('SMin', 'HSV')
    vMin = cv2.getTrackbarPos('VMin', 'HSV')

    hMax = cv2.getTrackbarPos('HMax', 'HSV')
    sMax = cv2.getTrackbarPos('SMax', 'HSV')
    vMax = cv2.getTrackbarPos('VMax', 'HSV')

    # CACTI MASK
    if HSV_TB_CACTI:
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])
    else:
        lower = np.array([2, 130, 0])
        upper = np.array([41, 255, 136])
    maskCacti = cv2.inRange(img_hsv, lower, upper)
    show("maskCacti", maskCacti)

    # WOOD BG
    if HSV_TB_BG:
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])
    else:
        lower = np.array([0, 0, 0])
        upper = np.array([44, 255, 255])
    maskWood = cv2.inRange(img_blur, lower, upper)
    show('maskWood', maskWood)
    cv2.imshow('maskWood', maskWood)

    # # closing
    # kernel = np.ones((15, 15))
    # woodClosing = cv2.morphologyEx(maskWood, cv2.MORPH_CLOSE, kernel)
    # show('woodClosing', woodClosing)

    woodBg = np.zeros_like(maskWood)  # step 1
    for val in np.unique(maskWood)[1:]:  # step 2
        mask = np.uint8(maskWood == val)  # step 3
        labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]  # step 4
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # step 5
        woodBg[labels == largest_label] = val  # step 6

    woodBg = cv2.bitwise_not(woodBg)

    # Canny filter
    if THRESHOLD_TB:
        threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    else:
        threshold1 = 0
        threshold2 = 118
    edges = cv2.Canny(img_blur, threshold1, threshold2)
    show("canny", edges)

    minLineLength = 100
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100, lines=np.array([]),
                            minLineLength=minLineLength, maxLineGap=80)
    a, b, c = lines.shape
    for i in range(a):
        x = lines[i][0][0] - lines[i][0][2]
        y = lines[i][0][1] - lines[i][0][3]
        if x != 0:
            if abs(y / x) < 1:
                cv2.line(img_blur, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255), 1,
                         cv2.LINE_AA)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gray = cv2.morphologyEx(img_blur, cv2.MORPH_CLOSE, se)
    cv2.imshow('img', gray)


    mask = cv2.bitwise_and(maskCacti, woodBg)
    show('and', mask)
    final = cv2.bitwise_or(mask, edges)
    show('or', final)

    img_countour = img.copy()
    # Countours
    getCountours(final, img_countour)
    cv2.imshow("countour", img_countour)

    # img_countour = img.copy()
    # # Countours
    # getCountours(img_erosion, img_countour)
    # cv2.imshow("countour", img_countour)

    # # red
    # lower = np.array([0, 50, 50])
    # upper = np.array([10, 255, 255])
    # lower = np.array([hMin, sMin, vMin])
    # upper = np.array([hMax, sMax, vMax])
    # maskRed = cv2.inRange(img_hsv, lower, upper)
    # # cv2.imshow("mask0", mask0)
    # # yellow
    # lower = np.array([10, 50, 50])
    # upper = np.array([15, 255, 100])
    # mask2 = cv2.inRange(img_hsv, lower, upper)
    # # show("mask2", mask2)
    # # yellow
    # lower = np.array([15, 50, 100])
    # upper = np.array([30, 255, 255])
    # mask1 = cv2.inRange(img_hsv, lower, upper)
    # # show("mask1", mask1)
    #
    #
    #
    # # join my masks
    # # mask = mask0 + mask1 + mask2
    # cv2.imshow("mask", mask)
    #
    # # Dilation
    # kernel = np.ones((4,4))
    # img_dil = cv2.dilate(img_canny, kernel, iterations=1)
    # show("dilation", img_dil)
    #
    # kernel = np.ones((4, 4))
    # img_grad = cv2.morphologyEx(img_canny, cv2.MORPH_GRADIENT, kernel)
    # show("img_grad", img_grad)

    # mask
    # lower = np.array([0, 50, 0])
    # upper = np.array([111, 255, 255])
    # mask = cv2.inRange(img_hsv, lower, upper)
    # show("mask", mask)
    # # erosion
    # kernel = np.ones((5, 5), np.uint8)
    # img_erosion = cv2.erode(mask, kernel, iterations=1)
    # show("img_erosion", img_erosion)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

