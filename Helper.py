import cv2
import imutils
import numpy as np
import time
import serial
from scipy.spatial.distance import cdist

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)
cap.set(10,150)

GREEN = 0
RED = 1
BLUE = 2
DAN = 3

FORWARD = 0
RIGHT = 1
LEFT = 2

OBJECTS_TYPE = {RED: ("Target", (0, 0, 255)), GREEN: ("Obsticle", (0, 255, 0)), BLUE: ("Robot", (255, 0, 0)), DAN: ("Dan", (0, 0, 0))}
AREAS_THRESH = [250, 250, 250]
GOING_TO_TARGET = 0
AVOID_OBST = 1


PORT = 'COM9'
#BLUETOOTH = serial.Serial(PORT, 9600)
#BLUETOOTH.flushInput()

# COLORS = [[0, 100, 0, 113, 199, 170],    #green
#             [171,101,174,179,252,255], #red
#             [66,48,153,110,255,255]]#   #blue
#             # list((12, 91, 83, 60, 255, 251))]

COLORS = [[46, 62, 42, 90, 199, 212],    #green
            [0, 47, 183, 255, 255, 255], #red
            [95, 109, 84, 255, 255, 255]]#   #blue


def print_title(title):
    num_tiles = 8
    print(" " * num_tiles + title)
    print("_" * (num_tiles * 2 + len(title)))


def print_message_to_img(img, text):
    cv2.putText(img, text, (20, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)


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
    cv2.setTrackbarPos("1_high", name, 179)
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

def get_new_colors_range():
    while True:

        success, img = cap.read()
        if cv2.getTrackbarPos('1_high', "tracker") == -1:
            create_color_trackbar("tracker")
        x = get_color_trackbar_values()
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # imgHSV = img
        lower = np.array(x[0:3])
        upper = np.array(x[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        kernel = np.ones((5,5), dtype="uint8")
        # mask = cv2.erode(mask, kernel, iterations=1)
        # mask = cv2.dilate(mask, kernel, iterations=1)
        cv2.imshow("tracker", cv2.bitwise_and(imgHSV, imgHSV, mask=mask))
        print(x)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)

def find_angle(pt1, pt2):
    pt1, pt2 = np.array(pt1), np.array(pt2)
    diff = pt2 - pt1
    radians = np.arctan2(diff[1], diff[0])
    deg = - (180 * radians / np.pi)
    return deg % 360


def find_distance(pt1, pt2):
    pt1, pt2 = np.array(pt1)[:2], np.array(pt2)[:2]
    return np.sqrt(np.sum(np.square(pt1 - pt2)))


def is_too_close_to_object(robot, obj):
    robot, obj = np.array(robot), np.array(obj)
    obj = obj[None] if obj.ndim == 1 else obj
    xr, yr, rr, _ = robot
    for o in obj:
        xo, yo, ro, _ = o
        dis = find_distance((xr, yr), (xo, yo))
        if dis <= ro + rr:
            return True
    return False


def draw_cnt(img, points, cnts):

    points = np.array(points).astype(int)
    if points.ndim < 2:
        points = points[None]
        cnts = [cnts]

    for i in range(points.shape[0]):
        p, cnt = tuple(points[i]), cnts[i]
        text, heu = OBJECTS_TYPE[p[3]]
        x, y, h, w = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + h, y + w), heu, 2)
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7, heu, 2)
        cv2.drawContours(img, cnt, -1, (90, 26, 170), 3)
        cv2.circle(img, p[:2], p[2] + 10, (60, 200, 110), 2)

def angle_to_origin(c=BLUE):

    orig_pos = tuple(np.array([FRAME_WIDTH / 2, FRAME_HEIGHT / 2], dtype=int))
    blue_range = tuple(COLORS[c])
    lower, upper = blue_range[:3], blue_range[3:]

    while True:
        success, img = cap.read()
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(imgHSV, lower, upper)

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)
        if len(contours):
            largest = max(contours, key=lambda x: cv2.contourArea(x))
            current_pos, r = cv2.minEnclosingCircle(largest)
            current_pos = tuple(int(x) for x in current_pos)

            p = (*current_pos, int(r), c)
            draw_cnt(img, p, largest)
            k = cv2.waitKey(1)
            if k == ord("q"):
                print(find_angle(orig_pos, current_pos), orig_pos, current_pos)
            elif k == ord("r"):
                orig_pos = current_pos
            elif k == 27:
                exit(0)

            cv2.line(img, orig_pos, current_pos, (0, 0, 0), thickness=2)
        cv2.imshow("Angles from Origin", img)


def get_hsv_mask(img, c):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if c == RED:
        mask1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        return cv2.bitwise_or(mask2,  mask1)

    elif c == GREEN:
        return cv2.inRange(hsv, (35, 25, 25), (75, 255, 255))

    elif c == BLUE:
        return cv2.inRange(hsv, (112, 200, 70), (128, 255, 255))
    else:
        raise Exception("Wrong color input")



