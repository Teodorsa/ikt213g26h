import cv2
import numpy as np

image = cv2.imread('images/lambo.png', 1)



def SobelEdgeDetection(image):
    grayScaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(grayScaled, (3, 3), 0)

    sobelX = cv2.Sobel(src=blurred, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
    sobelY = cv2.Sobel(src=blurred, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=1)
    sobelXY = cv2.magnitude(sobelX, sobelY)

    cv2.imwrite('images/sobel.png', cv2.normalize(sobelXY, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))

SobelEdgeDetection(image)



def CannyEdgeDetection(image, threshold1, threshold2):
    grayScaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(grayScaled, (3, 3), 0)

    edges = cv2.Canny(blurred, threshold1, threshold2)

    cv2.imwrite('images/canny.png', edges)

CannyEdgeDetection(image,50,50)



image2 = cv2.imread('images/shapes-1.png', 1)
template = cv2.imread('images/shapes_template.jpg', 0)

def templateMatch(image, template):
    grayScaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(grayScaled, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite('images/templateMatch.png', image)

templateMatch(image2, template)



def resize(image,  scaleFactor: int, upOrDown: str):
    rows, cols, _channels = map(int, image.shape)

    if upOrDown == "up":
        cv2.imwrite('images/upScaled.png', cv2.pyrUp(image, dstsize=(scaleFactor * cols, scaleFactor * rows)))

    elif upOrDown == 'down':
        cv2.imwrite('images/downScaled.png', cv2.pyrDown(image, dstsize=(cols // scaleFactor, rows // scaleFactor)))

resize(image, scaleFactor=2, upOrDown="up")
resize(image, scaleFactor=2, upOrDown="down")