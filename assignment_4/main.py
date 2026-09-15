import cv2
import numpy as np



image = cv2.imread('images/reference_img.png')
image2 = cv2.imread('images/align_this.jpg')



def HarrisCornerDetector(referenceImage):
    gray = cv2.cvtColor(referenceImage, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 5, 3, 0.05)

    dst = cv2.dilate(dst, None)

    referenceImage[dst > 0.05 * dst.max()] = [0, 0, 255]

    cv2.imwrite('images/harris.png', referenceImage)

HarrisCornerDetector(image)



def ImageAlignerSIFT(imageToAlign, referenceImage, maxFeatures, goodMatchPercent):
    gray1 = cv2.cvtColor(imageToAlign, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(referenceImage, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=maxFeatures)

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    FLANN_INDEX_KDTREE = 1
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)

    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < goodMatchPercent * n.distance:
            good.append(m)

    if len(good) < 4:
        print("Error, not enough good matches!")
        return None

    srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5.0)

    height, width = referenceImage.shape[:2]

    aligned = cv2.warpPerspective(imageToAlign, M, (width, height))

    cv2.imwrite('images/aligned.png', aligned)

    matchesMask = mask.ravel().tolist()

    drawParams = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matchesMask,
        flags=2
        )

    matchesImage = cv2.drawMatches(gray1, kp1, gray2, kp2, good, None, **drawParams)

    cv2.imwrite('images/matches.png', matchesImage)

ImageAlignerSIFT(image2, image, maxFeatures=100000, goodMatchPercent=0.7)