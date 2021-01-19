import cv2
import imutils
import numpy as np
import time
import serial
from scipy.spatial.distance import cdist


frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
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
PASSING_BY_OBSTACLE = 1

PORT = 'COM9'
#BLUETOOTH = serial.Serial(PORT, 9600)
#BLUETOOTH.flushInput()

# myColors = [[0, 100, 0, 113, 199, 170],    #green
#             [171,101,174,179,252,255], #red
#             [66,48,153,110,255,255]]#   #blue
#             # list((12, 91, 83, 60, 255, 251))]

myColors = [[46, 62, 42, 90, 199, 212],    #green
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


def findObjects(img, myColors):

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
    if len(newPoints):
        newPoints, indices = remove_noise_contours(newPoints)
        cnts = [cnts[i] for i in indices]
    return np.array(newPoints).astype(int), cnts


def get_contours(img, color):
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)

    points, res_contours = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > AREAS_THRESH[color]:
            AREAS_THRESH[color] = max(AREAS_THRESH[color], area * 0.5)
            ((mid_x, mid_y), r) = cv2.minEnclosingCircle(cnt)
            mid_x, mid_y = int(mid_x), int(mid_y)
            if mid_x != 0 and mid_y != 0:
                points.append((mid_x, mid_y, r, color))
                res_contours.append(cnt)
    return points, res_contours


def find_angle(pt1, pt2):
    pt1, pt2 = np.array(pt1), np.array(pt2)
    diff = pt1 - pt2
    radians = np.arctan2(diff[1], diff[0])
    return (180 * radians / np.pi + 180) % 360


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

    if len(points) < 2:
        return np.array(points), np.arange(len(points))

    points = np.array(points)
    xy = points[:, :2].astype("float32")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Use cross-validation to find the correct amount of clusters in the image, in order to remove the colliding
    # classifications.
    K = 2
    while True:

        res, indices, found_k = [], [], True
        ret, label, means = cv2.kmeans(xy, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        for i, mean in enumerate(means):
            curr_points = points[label.flatten() == i]
            ind = np.argmax(curr_points, axis=0)[2]
            dominant_p = curr_points[ind]
            dist = cdist(curr_points[:, :2], dominant_p[:2][None])

            if len(np.where(dist > dominant_p[2])[0]):
                K += 1
                found_k = False
                break

            res.append(dominant_p)
            indices.append(np.array(np.where(np.all(points == dominant_p, axis=1))).flatten()[0])
        if found_k:
            return np.array(res), np.array(indices)


def draw_cnt(img, p, cnt):

    p = tuple(p)
    text, heu = OBJECTS_TYPE[p[3]]
    x, y, h, w = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + h, y + w), heu, 2)
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7, heu, 2)
    cv2.drawContours(img, cnt, -1, (90, 26, 170), 3)
    cv2.circle(img, p[:2], p[2] + 10, (60, 200, 110), 2)


def get_new_colors_range():
    while True:

        success, img = cap.read()
        if cv2.getTrackbarPos('1_high', "tracker") == -1:
            create_color_trackbar("tracker")
        x = get_color_trackbar_values()
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array(x[0:3])
        upper = np.array(x[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        kernel = np.ones((5,5), dtype="uint8")
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        cv2.imshow("tracker", mask)
        print(x)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)


def find_initinal_rob_angel():
    """
    Find the initial angel of the robot, by moving it forward for 2 seconds, and then find the angel between the
    start and end positions.
    :return: Angel of the robot within the camera capture.
    """

    def _find_local_pos(stopby="key", time_num=3):
        assert stopby in ["key", "time"], "Stopby has to be either 'key' for stopping by pressing on 'q' " \
                                          "or 'time', for stopping after an amount in seconds given in arg 'time_num'"

        blue_range = tuple(myColors[BLUE])
        lower, upper = blue_range[:3], blue_range[3:]
        t_end = time.time() + time_num
        while True:
            success, img = cap.read()
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(imgHSV, lower, upper)
            mask_show = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = imutils.grab_contours(contours)
            if len(contours):
                largest = max(contours, key=lambda x: cv2.contourArea(x))
                current_pos, r = cv2.minEnclosingCircle(largest)
                p = (*[int(x) for x in current_pos], int(r), BLUE)
                draw_cnt(img, p, largest)
                if stopby == "key" and cv2.waitKey(1) & 0xFF == ord('q'):
                    return current_pos
                elif stopby == "time" and cv2.waitKey(1) and time.time() > t_end:
                    return current_pos
            if stopby == "key":
                print_message_to_img(img, "Locating Robot. Press 'q' when ready.")
            else:
                print_message_to_img(img, f"On timer: {int(t_end - time.time()) + 1} sec left")
            cv2.imshow("Result", np.hstack((img, mask_show)))

    first_pos = np.array(_find_local_pos())
    # BLUETOOTH.write(str.encode(str(0)))
    second_pos = np.array(_find_local_pos(stopby="time", time_num=2))
    return find_angle(first_pos, second_pos), second_pos


def fix_obst_and_target():
    print("1) Make sure the desired targets and obstacles are marked. When marked, press 'q'. ")

    obst, obst_cnt = [], []
    tar, tar_cnt = [], []
    while True:
        success, imgResult = cap.read()
        newPoints, cnts = findObjects(imgResult, myColors)
        indices = np.where(newPoints == GREEN)[0]
        if len(indices):
            obst = newPoints[indices]
            obst_cnt = [cnts[i] for i in indices]
            [draw_cnt(imgResult, p, c) for p, c in zip(obst, obst_cnt)]

        indices = np.where(newPoints == RED)[0]
        if len(indices):
            tar = newPoints[indices]
            tar_cnt = [cnts[i] for i in indices]
            [draw_cnt(imgResult, p, c) for p, c in zip(tar, tar_cnt)]

        print_message_to_img(imgResult, "Fixing objects. Press 'q' when ready.")
        cv2.imshow("Result", imgResult)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return obst, obst_cnt, tar, tar_cnt

def angle_to_origin():

    orig_pos, iter = None, 0
    blue_range = tuple(myColors[BLUE])
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
            if not iter:
                orig_pos = current_pos
                iter += 1
                continue

            p = (*current_pos, int(r), BLUE)
            draw_cnt(img, p, largest)
            k = cv2.waitKey(1)
            if k == ord("q"):
                print(find_angle(orig_pos, current_pos), orig_pos, current_pos)
            elif k == 27:
                return

            cv2.line(img, orig_pos, current_pos, (0, 0, 0), thickness=2)
            cv2.imshow("Angles from Origin", np.hstack((img, np.flip(img, 0))))



if __name__ == '__main__':

    # get_new_colors_range()
    # 1) Preprocessing :
    # - (Fix obstacles and target)
    # - Find the initial angle of the robot, and it's current position in the video
    obst, obst_cnt, tar, tar_cnt = fix_obst_and_target()
    robot_angle, robot_pos = find_initinal_rob_angel()

    tar = np.squeeze(tar)
    assert tar.ndim == 1, f"Invalid number of targets: {tar.ndim}"

    while True:

        # 2) Find angle from robot to target
        direction_angle = find_angle(robot_pos, tar[:2])

        # 3) Correct robot angle to direction_angle and move forward.
        pass

        # 4) if reached target, stop process.
        pass

        # 5) If reached obstacle, change to "avoid obstacle" mode
        pass




    # iter = 0
    # curr_state = GOING_TO_TARGET
    # g_obstacles = None
    # g_targets = None
    # obstacles, target = [], []
    # while True:
    #     success, imgResult = cap.read()
    #     newPoints, cnts = findColor(imgResult, myColors)
    #
    #     # Organize points according to role (Robot, obsticles, target)
    #     indices = np.where(newPoints == BLUE)[0]
    #     if len(indices):
    #         robot = newPoints[indices]
    #         robot_cnt = cnts[indices]
    #         [draw_cnt(imgResult, p, c) for p, c in zip(robot, robot_cnt)]
    #
    #     indices = np.where(newPoints == RED)[0]
    #     if len(indices):
    #         target = newPoints[indices]
    #         target_cnt = cnts[indices]
    #         [draw_cnt(imgResult, p, c) for p, c in zip(target, target_cnt)]
    #
    #     indices = np.where(newPoints == GREEN)[0]
    #     if len(indices):
    #         obstacles = newPoints[indices]
    #         obstacles_cnts = cnts[indices]
    #         [draw_cnt(imgResult, p, c) for p, c in zip(obstacles, obstacles_cnts)]
    #
    #     indices = np.where(newPoints == DAN)[0]
    #     if len(indices):
    #         dan = newPoints[indices]
    #         dan_cnt = cnts[indices]
    #         [draw_cnt(imgResult, p, c) for p, c in zip(dan, dan_cnt)]
    #
    #     cv2.imshow("Result", imgResult)
    #     time.sleep(0.1)
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         g_obstacles = obstacles
    #         g_targets = target
    #         break

    # iter += 1


# if not robot or not target:
#     continue
#
# indices = np.where(newPoints == GREEN)[0]
# obstacles = newPoints[indices]
# obstacles_cnts = newPoints[indices]
#
# # Move car
# angle_to_target = find_angel(robot, target)
# angles_to_obstacles = [find_angel(robot, x) for x in obstacles]
#
# if curr_state == GOING_TO_TARGET:
#     BLUETOOTH.write(str.encode(str(FORWARD)))
#
#     for obst in obstacles:
#         if is_too_close_to_object(robot, obst):
#             curr_state = PASSING_BY_OBSTACLE
#             break
# # curr_state = PASSING_BY_OBSTACLE
# else:
#
#     BLUETOOTH.write(str.encode(str(RIGHT)))

