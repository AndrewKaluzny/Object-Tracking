import cv2
import numpy as np
from collections import deque
import imutils

# for cv2.createTracker()
def nothing(x):
    pass
# define webcam
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Rmin','image',0,255, nothing)
cv2.createTrackbar('Rmax','image',255,255, nothing)
cv2.createTrackbar('Gmin','image',0,255, nothing)
cv2.createTrackbar('Gmax','image',255,255, nothing)
cv2.createTrackbar('Bmin','image',0,255, nothing)
cv2.createTrackbar('Bmax','image',255,255, nothing)
while True:
    webcam = cv2.VideoCapture(0)
    # define color filtration sliders
    img = np.zeros((1,512,3), np.uint8)
    cv2.imshow('image', img)
    pts = deque(maxlen = 32)
    while webcam.isOpened():
        ret, frame = webcam.read()
        if 'webcam.avi' and not ret:
            break
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #img = imutils.resize(img, width = 200)
        #img = imutils.resize(img, height = 500)




        # Select Color Filtration
        rMin = cv2.getTrackbarPos("Rmin", "image")
        rMax = cv2.getTrackbarPos("Rmax", "image")
        gMin = cv2.getTrackbarPos("Gmin", "image")
        gMax = cv2.getTrackbarPos("Gmax", "image")
        bMin = cv2.getTrackbarPos("Bmin", "image")
        bMax = cv2.getTrackbarPos("Bmax", "image")

        lowerRange = np.array([bMin, gMin, rMin])
        upperRange = np.array([bMax, gMax, rMax])

        # For tennis ball in my room
        #lowerRange = np.array([0, 52, 0])
        #upperRange = np.array([75, 255, 255])

        # Draw Mask
        mask = cv2.inRange(hsv, lowerRange, upperRange)
        cleanMask = cv2.bitwise_and(frame, frame, mask=mask)
        bwmask = cv2.cvtColor(cleanMask, cv2.COLOR_BGR2GRAY)
        bwmask = cv2.erode(bwmask, None, iterations=2)
        bwmask = cv2.dilate(bwmask, None, iterations=2)
        cv2.imshow("Final Mask", bwmask)

        # Find Contours
        contArray = cv2.findContours(bwmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        if len(contArray) > 0:
            c = max(contArray, key = cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if radius > 8:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 10)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        pts.appendleft(center)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            webcam.release()
            cv2.destroyAllWindows()
            exit(0)

        for i in xrange(1, len(pts)):
            if pts[i-1] is None or pts[i] is None:
                continue
            thickness = int(np.sqrt(32/float(i+1)))
            # FOR POINTS:
            # cv2.circle(frame, pts[i], int(32/(np.sqrt(i))-1), (255, 0, 0), -1)
            # FOR LINES:
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), int(16/(np.sqrt(i))))
        cv2.imshow("Raw Frame", frame)
webcam.release()
cv2.destroyAllWindows()