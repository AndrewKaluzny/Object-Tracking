import cv2
import numpy as np
from collections import deque
import imutils

# for cv2.createTracker()
def nothing(x):
    pass
# define webcam
cv2.namedWindow('image')
cv2.createTrackbar('Rmin','image',0,255, nothing)
cv2.createTrackbar('Rmax','image',0,255, nothing)
cv2.createTrackbar('Gmin','image',0,255, nothing)
cv2.createTrackbar('Gmax','image',0,255, nothing)
cv2.createTrackbar('Bmin','image',0,255, nothing)
cv2.createTrackbar('Bmax','image',0,255, nothing)
while True:
    cxo = 0
    cyo = 0
    webcam = cv2.VideoCapture('webcam.avi')
    # define color filtration sliders
    img = np.zeros((1,512,3), np.uint8)
    cv2.imshow('image', img)
    pts = deque(maxlen = 32)
    ptsb = deque(maxlen = 32)
    while webcam.isOpened():
        ret, frame = webcam.read()
        if 'webcam.avi' and not ret:
            break
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Select Color Filtration
        bMin = cv2.getTrackbarPos("Bmin", "image")
        bMax = cv2.getTrackbarPos("Bmax", "image")
        gMin = cv2.getTrackbarPos("Gmin", "image")
        gMax = cv2.getTrackbarPos("Gmax", "image")
        rMin = cv2.getTrackbarPos("Rmin", "image")
        rMax = cv2.getTrackbarPos("Rmax", "image")

        lowerRange = np.array([bMin, gMin, rMin])
        upperRange = np.array([bMax, gMax, rMax])

        lowerRange = np.array([0, 27, 69])
        upperRange = np.array([241, 255, 141])

        # Draw Mask
        mask = cv2.inRange(hsv, lowerRange, upperRange)
        cleanMask = cv2.bitwise_and(frame, frame, mask=mask)
        bwmask = cv2.cvtColor(cleanMask, cv2.COLOR_BGR2GRAY)
        bwmask = cv2.erode(bwmask, None, iterations=2)
        bwmask = cv2.dilate(bwmask, None, iterations=2)
        #cv2.imshow("Final Mask", bwmask)

        # Find Contours
        contArray = cv2.findContours(bwmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        if len(contArray) > 0:
            # currently finds largest contour on the whole frame
            c = max(contArray, key = cv2.contourArea)
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            # adjust to find the largest contour within a fixed distance of the previous contour (relative to contour size?)

            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if radius > 8:
                # cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 10)
                cxn = int(M["m10"] / M["m00"])
                cyn = int(M["m01"] / M["m00"])
                if cxo != 0 or cyo != 0:
                    if (abs(cxo - cxn) < 100) and (abs(cyo - cyn) < 100):
                        center = (cxn, cyn)
                        cxo = cxn
                        cyo = cyn
                else:
                    center = (cxn, cyn)
                    cxo = cxn
                    cyo = cyn

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
            #cv2.circle(frame, pts[i], int(32/(np.sqrt(i))-1), (255, 0, 0), -1)
            # FOR LINES:
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), int(16/(np.sqrt(i))))
        cv2.imshow("Updated Frame", frame)
webcam.release()
cv2.destroyAllWindows()