from PathFinder import *

def find_object_by_color(img, c):

    # color = COLORS[c]
    # imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower, upper = np.array(color[:3]), np.array(color[3:])
    # mask = cv2.inRange(imgHSV, lower, upper)
    mask = get_hsv_mask(img, c)
    p, c = get_contours(mask, c)

    newPoints, cnts = [], []
    if len(p):
        newPoints, indices = remove_noise_contours(p)
        cnts = [c[i] for i in indices]
    return np.array(newPoints, dtype=int), cnts


def get_contours(img, color):

    contours = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    areas = [cv2.contourArea(x) for x in contours]
    if len(areas):
        size_threshold = max(max(areas) * 0.8, AREAS_THRESH[color])
    points, res_contours = [], []

    for i, cnt in enumerate(contours):
        if areas[i] > size_threshold:
            center, r = cv2.minEnclosingCircle(cnt)
            points.append((*np.array(center, dtype=int), r, color))
            res_contours.append(cnt)
    return points, res_contours


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


def extract_largest(p, c):

    if len(c):
        largest = max(enumerate(c), key=lambda x: cv2.contourArea(x[1]))[0]
        return p[largest], c[largest]
    return p, c


def find_robot_initial_angle():
    """
    Find the initial angel of the robot, by moving it forward for 2 seconds, and then find the angel between the
    start and end positions.
    :return: Angel of the robot within the camera capture.
    """

    def _find_local_pos(stopby="key", time_num=3):
        assert stopby in ["key", "time"], "Stopby has to be either 'key' for stopping by pressing on 'q' " \
                                          "or 'time', for stopping after an amount in seconds given in arg 'time_num'"
        t_end = time.time() + time_num
        while True:
            success, img = cap.read()
            p, cnt = extract_largest(*find_object_by_color(img, BLUE))
            draw_cnt(img, p, cnt)
            if stopby == "key":
                print_message_to_img(img, "Locating Robot. Press 'q' when ready.")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return p[:2]
            elif stopby == "time":
                print_message_to_img(img, f"On timer: {int(t_end - time.time()) + 1} sec left")
                if cv2.waitKey(1) and time.time() > t_end:
                    return p[:2]
            cv2.imshow("Result", img)

    first_pos = _find_local_pos()
    # BLUETOOTH.write(str.encode(str(0)))
    second_pos = _find_local_pos(stopby="time", time_num=1)
    return find_angle(first_pos, second_pos)


def fix_obst_and_target():
    print("1) Make sure the desired targets and obstacles are marked. When marked, press 'q'. ")

    while True:
        success, imgResult = cap.read()
        obst, obst_cnts = find_object_by_color(imgResult, GREEN)
        tar, tar_cnt = find_object_by_color(imgResult, RED)
        if len(obst):
            [draw_cnt(imgResult, p, c) for p, c in zip(obst, obst_cnts)]
        if len(tar):
            [draw_cnt(imgResult, p, c) for p, c in zip(tar, tar_cnt)]

        print_message_to_img(imgResult, "Fixing objects. Press 'q' when ready.")
        cv2.imshow("Result", imgResult)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

def show_objects():
    print("1) Make sure the desired targets and obstacles are marked. When marked, press 'q'. ")

    while True:
        success, imgResult = cap.read()
        obst, obst_cnts = find_object_by_color(imgResult, GREEN)
        tar, tar_cnt = find_object_by_color(imgResult, RED)
        rob, rob_cnt = find_object_by_color(imgResult, BLUE)
        if len(obst):
            [draw_cnt(imgResult, p, c) for p, c in zip(obst, obst_cnts)]
        if len(tar):
            [draw_cnt(imgResult, p, c) for p, c in zip(tar, tar_cnt)]
        if len(rob):
            [draw_cnt(imgResult, p, c) for p, c in zip(rob, rob_cnt)]

        print_message_to_img(imgResult, "Fixing objects. Press 'q' when ready.")
        cv2.imshow("Result", imgResult)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()


def follow_route(angle, route, brake_dis=10):
    """
    Follows a route given by tuples of indicies upon the grid (Bound by the values FRAME_WIDTH and FRAME_HEIGHT)
    :param angle: current angle of obj in camera
    :param tar: final destination
    :param obst: obsticales on the route. -> listof(tuples(4))
    :param robot: object (i.e a robot, a snake player, etc) -> tuple(4)
    :param route: list of tuples, ordered by the route to some target. -> listof(tuples(2))
    :return: boolean - success of getting to target.
    """

    curr_angle = angle
    for next_pos in route:
        while True:
            _, img = cap.read()
            obst, obst_cnts = find_object_by_color(img, GREEN)
            tar, tar_cnt = extract_largest(*find_object_by_color(img, RED))
            robot, robot_cnt = extract_largest(*find_object_by_color(img, BLUE))

            if is_too_close_to_object(robot, obst):
                # Recalculate path
                new_route = search_path(img)
                return follow_route(curr_angle, new_route)

            if is_too_close_to_object(robot, tar):
                return True

            if is_too_close_to_object(robot, (*next_pos, brake_dis, None)):
                break

            curr_angle = find_angle(robot[:2], next_pos)
            # SEND ANGLE TO ROBOT

            points = [robot, *obst, tar]
            cnts = [robot_cnt, *obst_cnts, tar_cnt]
            draw_cnt(img, points, cnts)

    return False

def search_path(img):


    return []



if __name__ == '__main__':

    get_new_colors_range()
    # angle_to_origin(GREEN)
    # 1) Preprocessing :
    # - (Fix obstacles and target)
    # - Find the initial angle of the robot, and it's current position in the video
    state = GOING_TO_TARGET
    show_objects()
    curr_direction = find_robot_initial_angle()

    while True:
        _, img = cap.read()

        # 2) Find angle from robot to target
        obst, obst_cnts = find_object_by_color(img, GREEN)
        tar, tar_cnt = extract_largest(*find_object_by_color(img, RED))
        robot, robot_cnt = extract_largest(*find_object_by_color(img, BLUE))

        # 3) If state == GOING_TO_TARGET, adjust angel to destination and go to target.
        if state == GOING_TO_TARGET:
            angle_to_tar = find_angle(robot, tar)
            # If adjustment needed: stop, adjust and continue loop

            if is_too_close_to_object(robot, obst):
                pass

        # move forward

        # 4) if reached target, stop process.
        if is_too_close_to_object(robot, tar):
            print("Hooray!!!!!!!")
            exit(0)

        # 5) If reached obstacle, change to "avoid obstacle" mode
        pass

        # 2.5) Show objects on screen
        points = [robot, *obst, tar]
        cnts = [robot_cnt, *obst_cnts, tar_cnt]
        draw_cnt(img, points, cnts)

        cv2.imshow("Result", img)
        if cv2.waitKey(1) == 27:
            exit(0)




