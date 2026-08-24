import cv2
import numpy as np



image = cv2.imread('images/iris-1.png', 1)



def padding(image, borderWidth):
    paddedImage = cv2.copyMakeBorder(image,
                       borderWidth,
                       borderWidth,
                       borderWidth,
                       borderWidth,
                       borderType=cv2.BORDER_REFLECT)
    return paddedImage

cv2.imwrite("images/paddedImage.png", padding(image, 100))



def crop(image, x_0, x_1, y_0, y_1):
    return image[y_0:y_1, x_0:x_1]

cv2.imwrite("images/croppedImage.png", crop(image, 200, image.shape[1] - 130, 200, image.shape[0] - 130))



def resize(image, width, height ):
    return cv2.resize(image, (width, height))

cv2.imwrite("images/resizedImage.png", resize(image, 200, 200))



def copy(image, emptyPictureArray):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            emptyPictureArray[y, x] = image[y, x]

    return emptyPictureArray

height = image.shape[0]
width = image.shape[1]

emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)

cv2.imwrite("images/copiedImage.png", copy(image, emptyPictureArray))



def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite("images/grayscaleImage.png", grayscale(image))



def hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imwrite("images/hsvImage.png", hsv(image))



# An Important note for this task is that when values exceed 255 that
# value does not make sense on the color scale (0-255). Therefore, the
# result may be unpredictable. Using a function like clip() or wrapping
# around to 0 might be safer alternatives so that the values you are
# seeing are expected. This also applies when going below 0.
def hue_shifted(image, emptyPictureArray, hue):
    copiedImage = copy(image, emptyPictureArray)

    copiedImage[:, :, 0] = copiedImage[:, :, 0] + hue
    copiedImage[:, :, 1] = copiedImage[:, :, 1] + hue
    copiedImage[:, :, 2] = copiedImage[:, :, 2] + hue

    return copiedImage

height = image.shape[0]
width = image.shape[1]

emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)

cv2.imwrite("images/shiftedImage.png", hue_shifted(image, emptyPictureArray, 50))



def smoothing(image):
    blurredImage = cv2.GaussianBlur(image,
                                    ksize=(15,15),
                                    sigmaX=0,
                                    borderType=cv2.BORDER_DEFAULT)
    return blurredImage

cv2.imwrite("images/smoothedImage.png", smoothing(image))



def rotation(image, angle):
    if angle == 90:
        rotatedImage = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        rotatedImage =  cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        rotatedImage = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else :
        rotatedImage = image
    return rotatedImage

angle = int(input("Enter the angle in degrees (90, 180 or 270): "))
cv2.imwrite("images/rotatedImage.png", rotation(image, angle))