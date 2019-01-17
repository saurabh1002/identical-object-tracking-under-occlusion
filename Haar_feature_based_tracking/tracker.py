import numpy as np
import cv2
import sys
from random import randint

swap_0_1 = 0

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
font = cv2.FONT_HERSHEY_SIMPLEX

def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]:
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)

  return tracker

def haar_classifier():
    while(1):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cup_cascades = cup_cascade.detectMultiScale(gray, 2, 5, 0, minSize=(128,128), maxSize=(128,128))
        if (len(cup_cascades) == 2):
            
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cup_cascades = cup_cascade.detectMultiScale(gray, 2, 5, 0, minSize=(128,128), maxSize=(128,128))
            global center_old
            center_old = [0 for i in range(len(cup_cascades))]
            for i, (x,y,w,h) in enumerate(cup_cascades):
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
                center_old[i] = x+64
            sep = 0
            for i in range(len(center_old)):
                for j in range(len(center_old)):
                    if (i != j):
                        sep = sep + abs(center_old[i] - center_old[j])
            if (sep > 128*(len(center_old)**2 - len(center_old))):
                # print('initial centers', center_old)
                break
            cv2.imshow('img',frame)


    return cup_cascades, frame


def initiate_tracker(cup_cascades, trackertype):
    trackerType = trackertype
    global bboxes, colors, Trackers
    bboxes = [0 for i in range(len(cup_cascades))]
    colors = [0 for i in range(len(cup_cascades))]
    Trackers = [0 for i in range(len(cup_cascades))]

    for i, (x,y,w,h) in enumerate(cup_cascades):
        bboxes[i] = (x,y,w,h)
        colors[i] = (randint(0, 255), randint(0, 255), randint(0, 255))
    # print('lengths', len(bboxes), len(colors))

    for i, bbox in enumerate(bboxes):
        Trackers[i] = cv2.MultiTracker_create()
    # print('length tracker', len(Trackers))

    for i in range(len(Trackers)):
        # print('added tracker', i)
        Trackers[i].add(createTrackerByName(trackerType), frame, bboxes[i])

    return bboxes, colors, Trackers

cup_cascade = cv2.CascadeClassifier('../cup_tracking_cascades/cascade.xml')

cap = cv2.VideoCapture('../videos/output2.avi')
cup_cascades, frame = haar_classifier()
bboxes, colors, Trackers = initiate_tracker(cup_cascades, "BOOSTING")

while cap.isOpened():
    k = -1
    success, frame = cap.read()
    if not success:
        break
    global center_new, d_center
    center_new = [0 for i in range(len(cup_cascades))]
    d_center = [0 for i in range(len(cup_cascades))]

    for j in range(len(Trackers)):
        print('inside for',j)
        success, boxes = Trackers[j].update(frame)
        if (success):
            for i, newbox in enumerate(boxes):
                print('inside for 2',i)
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                center_new[j] = p1[0]+64
                cv2.rectangle(frame, p1, p2, colors[j], 2, 1)
                cv2.putText(frame, str(j), p1, font, 4, colors[j], 2, 1)
                cv2.putText(frame, str(swap_0_1), (600,440), font, 2, (0,0,0), 4, 1)
                cv2.imshow('img', frame)
    print("old", center_old)
    print("new", center_new)
    for i in range(len(Trackers)):
        d_center[i] = center_new[i] - center_old[i]
    print('diff', d_center)
    for i in range(len(Trackers)):
        if d_center[i]<-5:
            print(i,"is moving left")
        elif d_center[i]> 5:
            print(i,"is moving right")
    print(abs(center_new[0] - center_new[1]) - abs(center_old[0] - center_old[1]))
    if (abs(center_new[0] - center_new[1]) - abs(center_old[0] - center_old[1]) < 0):
        if ((((d_center[0] > 5) and (d_center[1] < -5)) or ((d_center[0] < -5) and (d_center[1] > 5))) and (abs(center_new[0] - center_new[1]) < 120)):
            swap_0_1 = swap_0_1+1
            print(swap_0_1)
            cup_cascades, frame = haar_classifier()
            bboxes, colors, Trackers = initiate_tracker(cup_cascades, "BOOSTING")
    for i in range(len(Trackers)):
        center_old[i] = center_new[i]
    # cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break

cap.release()
cv2.destroyAllWindows()
