import cv2
import numpy as np
from matplotlib import pyplot as plt

"""Creates a video file titled "webcam2.avi" and replays it to the user"""

print("Start of program!\n")

# ''' # START Save to File
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('webcam2.avi',fourcc, 20.0, (640,480))
raw_input("Press Enter to start recording")
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:

        # write the flipped frame
        out.write(frame)


        #cv2.circle(frame, (100, 100), 100, (0, 255, 255), 3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
# ''' # END Save to file

print("Finished recording!\n")
#raw_input("Press Enter to continue...")

# C:\Users\Andrew\PycharmProjects\Integrator\
# ''' # START Playback From File
cap = cv2.VideoCapture('webcam2.avi')

print(cap.isOpened())

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        cv2.imshow('frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
# ''' # END Playback from file