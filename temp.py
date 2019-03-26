import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

for i in range(0, 1):
    foldername = 'B/7/'
    onlyfiles = [f for f in listdir(foldername)]
    for filename in onlyfiles:
        dir = foldername + filename
        frame = cv2.imread(dir)
        blur = cv2.GaussianBlur(frame, (3, 3), 0)

        # Convert to HSV color space
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Create a binary image with where white will be skin colors and rest is black
        mask2 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
        # Kernel for morphological transformation
        kernel = np.ones((5, 5))

        # Apply morphological transformations to filter out the background noise
        dilation = cv2.dilate(mask2, kernel, iterations=4)
        erosion = cv2.erode(dilation, kernel, iterations=1)

        # Apply Gaussian Blur and Threshold
        filtered = cv2.GaussianBlur(erosion, (5, 5), 100)
        ret, thresh = cv2.threshold(filtered, 127, 255, 0)

        # Kernel matrices for morphological transformation


        # Find contours of the filtered frame
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find Max contour area (Assume that hand is in the frame)
        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        # Print bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ##### Show final image ########
        # cv2.imshow('Dilation',frame)
        cropped = frame[y:y + h, x:x + w]
        # img = cv2.resize(cropped,(32,32))
        # cv2.imshow('Cropped',cropped)
        cv2.imwrite(dir, cropped)
        print("done")
        ###############################

cv2.waitKey(0)
cv2.destroyAllWindows()
print(onlyfiles)