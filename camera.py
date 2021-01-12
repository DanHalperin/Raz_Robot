import cv2
import numpy as np
import keyboard
import imutils
import serial
import time
import math

cap = cv2.VideoCapture(0)
pts = [2000]
pts1 = [2000]
pts2 = [2000]
angle = 0
greenXYR = []
centerX = 0
centerY = 0
radius = 0
angel = 0

port = 'COM9'
bluetooth = serial.Serial(port, 9600)
bluetooth.flushInput()

while (1):
    bluetooth.flushInput()
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # color1-red
    lower_red = np.array([171,101,174])
    upper_red = np.array([179,252,255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # color2-blue
    lower_blue = np.array([90, 120, 140])
    upper_blue = np.array([120, 255, 255])
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    res1 = cv2.bitwise_and(frame, frame, mask=mask1)

    # color3-green
    lower_green = np.array([40, 50, 105])
    upper_green = np.array([85, 230, 255])
    mask2 = cv2.inRange(hsv, lower_green, upper_green)
    res2 = cv2.bitwise_and(frame, frame, mask=mask2)

    # red
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # blue
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask1 = cv2.erode(mask1, None, iterations=2)
    mask1 = cv2.dilate(mask1, None, iterations=2)
    # green
    mask2 = cv2.inRange(hsv, lower_green, upper_green)
    mask2 = cv2.erode(mask2, None, iterations=2)
    mask2 = cv2.dilate(mask2, None, iterations=2)

    # RED COLOR
    cnts_red = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    cnts_red = imutils.grab_contours(cnts_red)
    center_red = None
    if len(cnts_red) > 0:
        c_red = max(cnts_red, key=cv2.contourArea)
        ((x1, y1), radius_red) = cv2.minEnclosingCircle(c_red)
        M_red = cv2.moments(c_red)
        center_red = (int(M_red["m10"] / M_red["m00"]), int(M_red["m01"] / M_red["m00"]))
        center_redX = int(M_red["m10"] / M_red["m00"])
        center_redY = int(M_red["m01"] / M_red["m00"])
        if radius_red > 10:
            cv2.circle(frame, (int(x1), int(y1)), int(radius_red),
                       (0, 0, 255), 2)
            cv2.circle(frame, center_red, 5, (0, 0, 255), -1)
    pts1.append(center_red)
    for i in range(1, len(pts1)):
        if pts1[i - 1] is None or pts1[i] is None:
            continue
        thickness1 = 2
        # cv2.line(frame, pts1[i-1], pts1[i], (0, 0, 255), thickness1)

    # BLUE COLOR
    cnts = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        centerX = int(M["m10"] / M["m00"])
        centerY = int(M["m01"] / M["m00"])
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (255, 0, 0), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    pts.append(center)
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        #thickness = 2
        #cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # GREEN COLOR
    cnts_green = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    cnts_green = imutils.grab_contours(cnts_green)
    center_green = None
    if len(cnts_green) > 0:
        c_green = max(cnts_green, key=cv2.contourArea)

        ##### eyalll
        cnts1_green = cv2.findContours(mask2.copy(), cv2.RETR_CCOMP,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cnts1_green = imutils.grab_contours(cnts1_green)
        greenXYR = []
        sum = 0
        radius1 = 0
        for kkk in range(len(cnts1_green)):
            ((x2, y2), radius_green) = cv2.minEnclosingCircle(cnts1_green[kkk])
            M_green = cv2.moments(cnts1_green[kkk])
            center_green = (int(M_green["m10"] / M_green["m00"]), int(M_green["m01"] / M_green["m00"]))
            center_greenX = int(x2)
            center_greenY = int(y2)
            radius_green = int(radius_green)
            radius1 = radius_green
            if radius_green >= radius1 and sum < 3 and radius_green > 10:
                sum += 1
                # print(sum,')',int(center_greenX),' ', int(center_greenY),' radius=',radius_green)
                greenXYR.append([center_greenX, center_greenY, radius_green])
                if radius_green > 10:
                    cv2.circle(frame, (int(x2), int(y2)), int(radius_green),
                               (0, 255, 255), 2)
                    # cv2.circle(frame, center_green, 5, (0, 0, 255), -1)
            pts2.append(center_green)
            for i in range(1, len(pts2)):
                if pts2[i - 1] is None or pts2[i] is None:
                    continue
        ######## eyalll
        ((x2, y2), radius_green) = cv2.minEnclosingCircle(c_green)
        M_green = cv2.moments(c_green)
        center_green = (int(M_green["m10"] / M_green["m00"]), int(M_green["m01"] / M_green["m00"]))
        center_greenX = int(M_green["m10"] / M_green["m00"])
        center_greenY = int(M_green["m01"] / M_green["m00"])
        if radius_green > 10:
            cv2.circle(frame, (int(x2), int(y2)), int(radius_green),
                       (0, 255, 255), 2)
            cv2.circle(frame, center_green, 5, (0, 0, 255), -1)
    pts2.append(center_green)
    for i in range(1, len(pts2)):
        if pts2[i - 1] is None or pts2[i] is None:
            continue
        thickness2 = 2
        # cv2.line(frame, pts2[i-1], pts2[i], (0, 0, 255), thickness2)

    # func1
    obst = 0
    for check in range(len(greenXYR)):
        check1 = greenXYR[check]
        checkR = check1[2]
        checkX = check1[0]
        checkY = check1[1]
        if (centerX is not 0) and (centerY is not 0) and (radius is not 0):
            if ((abs(centerX - checkX) - radius - checkR) - 50 < 0) and (
                    (abs(centerY - checkY) - radius - checkR) - 50 < 0):
                if centerY - checkY > 0:
                    obst += 1
                    bluetooth.write(str.encode(str(int(angel - 90))))
                    time.sleep(0.1)
                    print("obst avoid", angel - 90)
                else:
                    obst += 1
                    bluetooth.write(str.encode(str(int(angel + 90))))
                    time.sleep(0.1)
                    print("obst avoid", angel + 90)
    # func2
    if (center_red is not None and center is not None):
        distX = center_redX - centerX
        distY = center_redY - centerY
        angel = abs(math.atan2(distX, distY))
        angel = int((angel * 180) / math.pi) - 90  # angel positive - right

        # func3
        if abs(abs(angel) - abs(angle)) > 0 and obst == 0:
            # print(angel , "-angle")
            bluetooth.write(str.encode(str(int(angel))))
            time.sleep(0.1)
            angle = angel
            print("GO", angel)
        # func4
        distXS = abs(distX) - radius_red - radius
        distYS = abs(distY) - radius_red - radius
        if distXS < 0 and distYS < 0:
            bluetooth.write(str.encode(str(int(999))))
            time.sleep(0.1)
            print("COMPLETE")
            break

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask + mask1 + mask2)
    # cv2.imshow('res',res+res1+res2)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

