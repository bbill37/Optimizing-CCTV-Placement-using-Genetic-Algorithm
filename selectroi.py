import cv2
import numpy as np


# Read image
image = cv2.imread("art.png")

# Select ROI
r = cv2.selectROI("select the area", image)

# Crop image
cropped_image = image[int(r[1]):int(r[1]+r[3]),
                      int(r[0]):int(r[0]+r[2])]

# Display cropped image
cv2.imshow("Cropped image", cropped_image)
cv2.waitKey(0)

print(int(r[0])) # x0
print(int(r[1])) # y0
print(int(r[2])) # x1
print(int(r[3])) # y1