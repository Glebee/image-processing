import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def NLMeans(image):
    dst = cv.fastNlMeansDenoisingColored(image, None, 8, 8, 7, 21)
    b, g, r = cv.split(dst)
    rgb_dst = cv.merge([r, g, b])
    return rgb_dst

def Gaussian(image):
    dst = cv.GaussianBlur(image, (7, 7), cv.BORDER_DEFAULT)
    b, g, r = cv.split(dst)
    rgb_dst = cv.merge([r, g, b])
    return rgb_dst

def Bilateral(image):
    dst = cv.bilateralFilter(image, 14, 75, 75)
    b, g, r = cv.split(dst)
    rgb_dst = cv.merge([r, g, b])
    return rgb_dst

def CalcOfDamageAndNonDamage(image):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    image_erode = cv.erode(image, kernel)
    hsv_img = cv.cvtColor(image_erode, cv.COLOR_BGR2HSV)
    markers = np.zeros((image.shape[0], image.shape[1]), dtype="int32")
    markers[90: 140, 90: 140] = 255
    markers[236: 255, 0: 20] = 1
    markers[0: 20, 0: 20] = 1
    markers[0: 20, 236: 255] = 1
    markers[236: 255, 236: 255] = 1
    leafs_area_BGR = cv.watershed(image_erode, markers)
    healthy_part = cv.inRange(hsv_img, (36, 25, 25), (86, 255, 255))
    ill_part = leafs_area_BGR - healthy_part
    mask = np.zeros_like(image, np.uint8)
    mask[leafs_area_BGR > 1] = (255, 0, 255)
    mask[ill_part > 1] = (0, 0, 255)
    return mask

src = cv.imread("12.jpg")
NLMimg = NLMeans(src)
GAimg = Gaussian(src)
BIimg = Bilateral(src)
b, g, r = cv.split(src)
rgb_img = cv.merge([r, g, b])
maskNLM = CalcOfDamageAndNonDamage(NLMimg)
maskGA = CalcOfDamageAndNonDamage(GAimg)
maskBI = CalcOfDamageAndNonDamage(BIimg)

plt.subplot(421), plt.imshow(rgb_img)
plt.subplot(423), plt.imshow(NLMimg)
plt.subplot(427), plt.imshow(GAimg)
plt.subplot(425), plt.imshow(BIimg)
plt.subplot(424), plt.imshow(maskNLM)
plt.subplot(428), plt.imshow(maskGA)
plt.subplot(426), plt.imshow(maskBI)
plt.show()