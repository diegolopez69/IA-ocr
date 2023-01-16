# Importing the cv2 and matplotlib library
import cv2 
from matplotlib import pyplot as plt
import numpy as np

# Reading the image
image = cv2.imread("image3.jpeg")

# Showing the image
cv2.imshow('Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Scaling the image by 60%
scale=60
newWidth = int(image.shape[1] * scale / 100)
newHeight = int(image.shape[0] * scale / 100)
newDimension = (newWidth, newHeight)

# Resizing the image
resizedImage = cv2.resize(image, newDimension, interpolation = cv2.INTER_AREA)
cv2.imshow('Image',resizedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Saving the resized image
cv2.imwrite("resizedParts.png", resizedImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# Converting the resized image to grayscale
grayImage=cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)

# Showing the grayscale image
cv2.imshow('Image', grayImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Saving the grayscale image
cv2.imwrite("resizedPartsGray.png", grayImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# Applying threshold to the grayscale image
estimatedThreshold, thresholdImage=cv2.threshold(grayImage,90,255,cv2.THRESH_BINARY)

# Showing the threshold image
cv2.imshow('Image', thresholdImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Saving the threshold image
cv2.imwrite("resizedPartsThreshold.png", thresholdImage, [cv2.IMWRITE_PNG_COMPRESSION, 0]) 

# Plotting the grayscale and threshold images with histograms
plt.figure(figsize=(14, 12))
plt.subplot(2,2,1), plt.imshow(grayImage,'gray'), plt.title('Grayscale Image')
plt.subplot(2,2,2), plt.hist(grayImage.ravel(), 256), plt.title('Color Histogram of Grayscale Image')
plt.subplot(2,2,3), plt.imshow(thresholdImage,'gray'), plt.title('Binary (Thresholded)  Image')
plt.subplot(2,2,4), plt.hist(thresholdImage.ravel(),256), plt.title('Color Histogram of Binary (Thresholded) Image')
plt.savefig('fig1.png')
plt.show()


# threshold the grayscale image using Otsu's method
estimatedThreshold, thresholdImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# show the thresholded image
cv2.imshow('Image', thresholdImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save the thresholded image
cv2.imwrite("resizedPartsThreshold.png", thresholdImage, [cv2.IMWRITE_PNG_COMPRESSION, 0]) 

# find contours in the thresholded image
contours, hierarchy = cv2.findContours(thresholdImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# create a copy of the resized image
resizedImageCopy = np.copy(resizedImage)

# draw contours on the copy of the resized image
for i, c in enumerate(contours):
    areaContour = cv2.contourArea(c)
    if areaContour < 2000 or 100000 < areaContour:
        continue
    cv2.drawContours(resizedImageCopy, contours, i, (255, 10, 255), 4)

# Display the image using the imshow function
cv2.imshow('Image', resizedImageCopy)
# Wait for a key press event
cv2.waitKey(0)
# Close all open windows
cv2.destroyAllWindows()
# Save the image to a file
cv2.imwrite("resizedPartsContours.png", resizedImageCopy, [cv2.IMWRITE_PNG_COMPRESSION, 0])