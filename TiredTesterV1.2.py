from scipy.spatial import distance
from imutils.video import VideoStream
from imutils import face_utils
import cv2
import imutils
import dlib

def eyeAspectRatio(eye):
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    
    c = distance.euclidean(eye[0], eye[3])
    return (a+b)/(2*c)

eyeCloseThreshold = 0.15
eyeOpenThreshold = 0

frameMaxForBlink_fast = 3
frameMaxForBlink_slow = 10
6
count = 0
slow_blinks = 1
fast_blinks = 10
blink_factor = 0
fatigueThreshold = 4

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

leftStart = 42
leftEnd = 48
rightStart = 36
rightEnd = 42

print("starting video stream")
print("Analyzing profile now. Please keep eyes relaxed.")
vs = VideoStream(src=0).start()
sumRatio = 0

i = 0

while i < 300:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray,0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[leftStart:leftEnd]
        rightEye = shape[rightStart:rightEnd]

        leftEyeRatio = eyeAspectRatio(leftEye)
        rightEyeRatio = eyeAspectRatio(rightEye)
        ratio = (leftEyeRatio + rightEyeRatio) / 2
        if ratio != 0:
            sumRatio += ratio
            i+=1
            print(i)

eyeOpenThreshold = (sumRatio / 300) - 0.02
eyeCloseThreshold = eyeOpenThreshold - 0.05
print(eyeOpenThreshold)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray,0)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[leftStart:leftEnd]
        rightEye = shape[rightStart:rightEnd]
        
        leftEyeRatio = eyeAspectRatio(leftEye)
        rightEyeRatio = eyeAspectRatio(rightEye)
        
        ratio = (leftEyeRatio + rightEyeRatio) / 2
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)
        ###############
        if ratio < eyeCloseThreshold:
            count += 1
        elif ratio > eyeOpenThreshold:
            if count >= frameMaxForBlink_slow:
                slow_blinks += 1
            elif count >= frameMaxForBlink_fast:
                fast_blinks += 1
            count = 0
        ################
        cv2.putText(frame, "Slow Blinks: {}".format(slow_blinks), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Fast Blinks: {}".format(fast_blinks), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Ratio: {:.2f}".format(ratio), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    blink_factor = fast_blinks/slow_blinks
    
    if blink_factor < fatigueThreshold:
        cv2.putText(frame, "take a break", (150,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
        
    cv2.imshow("Frame",frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()
vs.stop()
