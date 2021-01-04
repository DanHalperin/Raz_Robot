import cv2
import imutils
import numpy as np
import time

import serial

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)

GREEN = 0
RED = 1
BLUE = 2
K = 2

FORWARD = 0
RIGHT = 1
LEFT = 2

OBJECTS_TYPE = {RED: ("Target", (0, 0, 255)), GREEN: ("Obsticle", (0, 255, 0)), BLUE: ("Robot", (255, 0, 0))}
GOING_TO_TARGET = 0
PASSING_BY_OBSTACLE = 1

PORT = 'COM9'
BLUETOOTH = serial.Serial(PORT, 9600)
BLUETOOTH.flushInput()

myColors = [[50,72,111,82,145,172],  #green
            [171,101,174,179,252,255],#red
            [66,48,153,110,255,255]]  #blue

def create_color_trackbar(name="tracker"):
    def nothing(x):
        pass
    cv2.namedWindow(name)
    cv2.createTrackbar('1_low', name, 0, 255, nothing)
    cv2.createTrackbar('2_low', name, 0, 255, nothing)
    cv2.createTrackbar('3_low', name, 0, 255, nothing)
    cv2.createTrackbar('1_high', name, 0, 255, nothing)
    cv2.createTrackbar('2_high', name, 0, 255, nothing)
    cv2.createTrackbar('3_high', name, 0, 255, nothing)
    cv2.setTrackbarPos("1_high", name, 255)
    cv2.setTrackbarPos("2_high", name, 255)
    cv2.setTrackbarPos("3_high", name, 255)
    return name

def get_color_trackbar_values(name="tracker"):
    hl = cv2.getTrackbarPos('1_low', name)
    sl = cv2.getTrackbarPos('2_low', name)
    vl = cv2.getTrackbarPos('3_low', name)
    hh = cv2.getTrackbarPos('1_high', name)
    sh = cv2.getTrackbarPos('2_high', name)
    vh = cv2.getTrackbarPos('3_high', name)
    return hl, sl, vl, hh, sh, vh


def findColor(img,myColors):

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    newPoints, cnts = [], []
    for i, color in enumerate(myColors):
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        p, c = get_contours(mask, i)
        if p:
            newPoints += p
            cnts += c
    if newPoints:
        newPoints, indices = remove_noise_contours(newPoints)
        cnts = np.array(cnts)[indices]
    return np.array(newPoints).astype(int), cnts


def get_contours(img, color):
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)

    points, res_contours = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            ((mid_x, mid_y), r) = cv2.minEnclosingCircle(cnt)
            mid_x, mid_y = int(mid_x), int(mid_y)
            if mid_x != 0 and mid_y != 0:
                points.append((mid_x, mid_y, r, color))
                res_contours.append(cnt)
    return points, res_contours


def find_angel(pt1, pt2):
    pt1, pt2 = np.array(pt1), np.array(pt2)
    temp = pt2 - pt1
    radians = np.arctan2(temp[1], temp[0])
    return 180 * radians / np.pi


def find_distance(pt1, pt2):
    pt1, pt2 = np.array(pt1)[:2], np.array(pt2)[:2]
    return np.sqrt(np.sum(np.square(pt1 - pt2)))


def is_too_close_to_object(robot, obj):
    xr, yr, rr, _ = robot
    xo, yo, ro, _ = obj
    dis = find_distance((xr, yr), (xo, yo))
    if dis < ro or dis < rr:
        return False
    return dis <= ro + 2 * rr


def remove_noise_contours(points):

    if len(points) < K:
        return [], []

    res, indices = [], []
    points = np.array(points)
    xy = points[:, :2]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, means = cv2.kmeans(np.float32(xy), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    for i, mean in enumerate(means):
        curr_points = points[label.flatten() == i]
        dominant_p = max(curr_points, key=lambda x: x[2])
        res.append(dominant_p)
        indices.append(np.array(np.where(np.all(points==dominant_p, axis=1))).flatten()[0])

    return np.array(res), np.array(indices)


def draw_cnt(p, cnt):
    p = tuple(p)
    text, heu = OBJECTS_TYPE[p[3]]
    x, y, h, w = cv2.boundingRect(cnt)
    cv2.rectangle(imgResult, (x, y), (x + h, y + w), heu, 2)
    cv2.putText(imgResult, text,
                (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                heu, 2)
    cv2.drawContours(imgResult, cnt, -1, (90, 26, 170), 3)
    cv2.circle(imgResult, p[:2], p[2] + 10, (60, 200, 110), 2)


if __name__ == '__main__':

    iter = 0
    curr_state = GOING_TO_TARGET
    while True:

        success, img = cap.read()
        imgResult = img.copy()
        newPoints, cnts = findColor(img, myColors)

        # Organize points according to role (Robot, obsticles, target)
        indices = np.where(newPoints == BLUE)[0]
        robot = newPoints[indices]
        robot_cnt = cnts[indices]

        indices = np.where(newPoints == RED)[0]
        target = newPoints[indices]
        target_cnt = newPoints[indices]

        if not robot or not target:
            continue

        indices = np.where(newPoints == GREEN)[0]
        obstacles = newPoints[indices]
        obstacles_cnts = newPoints[indices]

        # Move car
        angle_to_target = find_angel(robot, target)
        angles_to_obstacles = [find_angel(robot, x) for x in obstacles]

        if curr_state == GOING_TO_TARGET:
            BLUETOOTH.write(str.encode(str(FORWARD)))

            for obst in obstacles:
                if is_too_close_to_object(robot, obst):
                    curr_state = PASSING_BY_OBSTACLE
                    break
        # curr_state = PASSING_BY_OBSTACLE
        else:
            
            BLUETOOTH.write(str.encode(str(RIGHT)))




        # if len(newPoints) > K - 1:
        #
        #     ob1 = tuple(newPoints[0])
        #     ob2 = tuple(newPoints[-1])
        #
        #     print("Current Frame:\n----------------------")
        #     # print(find_angel(*newPoints[:2]))
        #     print(f"points {ob1}, {ob2}")
        #     print(find_distance(ob1, ob2))
        #     print("r's", ob1[2], ob2[2])
        #     print("\n\n")
        #     r1, r2= int(ob1[2]), int(ob2[2])
        #     for i in [0, -1]:
        #         draw_cnt(newPoints[i], cnts[i])
        #     cv2.line(imgResult, ob1[:2], ob2[:2], (0, 0, 0), 2)
        #     cv2.imshow("Result", imgResult)
        #     # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     #     continue

        cv2.imshow("Result",imgResult)
        time.sleep(0.1)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        iter += 1