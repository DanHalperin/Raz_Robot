from bisect import bisect_right
from collections import deque

from PathFinder import *


def find_object_by_color(img, c):

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
            out_vid.write(img)

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
        out_vid.write(imgResult)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return


def follow_route(angle, route, rob, tar, brake_dis=10):
    """
    Follows a route given by tuples of indicies upon the grid (Bound by the values FRAME_WIDTH and FRAME_HEIGHT)
    :param angle: current angle of obj in camera
    :param tar: final destination
    :param robot: object (i.e a robot, a snake player, etc) -> tuple(4)
    :param route: list of tuples, ordered by the route to some target. -> listof(tuples(2))
    :return: boolean - success of getting to target.
    """

    curr_angle = angle
    for i, next_pos in enumerate(route):

        if d(next_pos, route[-1]) >= d(rob[:2], route[-1]):
            continue

        while True:
            _, img = cap.read()

            obst, obst_cnts = find_object_by_color(img, GREEN)
            tar, tar_cnt = extract_largest(*find_object_by_color(img, RED))
            robot, robot_cnt = extract_largest(*find_object_by_color(img, BLUE))
            draw_route(img, [tuple(robot[:2])] + route[i:])

            # if is_too_close_to_object(robot, obst):
            #     # Recalculate path
            #     new_route = search_path(rob, obst, tar, curr_angle)
            #     return follow_route(curr_angle, new_route, robot, tar)

            if is_too_close_to_object(robot, tar, mode="in"):
                return True

            if is_too_close_to_object(robot, (*next_pos, brake_dis, -1)):
                break

            curr_angle = find_angle(robot[:2], next_pos)
            # SEND ANGLE TO ROBOT

            points = [robot, *obst, tar]
            cnts = [robot_cnt, *obst_cnts, tar_cnt]
            draw_cnt(img, points, cnts)

            print_message_to_img(img, "Following Route")
            cv2.imshow(WINDOW_NAME, img)
            out_vid.write(img)
            cv2.waitKey(1)

    return False


def search_path(img, rob, obst, tar, init_angle):

    iter, success, path = 0, True, []
    while success:
        print_message_to_img(img, "Finding Path")
        cv2.imshow(WINDOW_NAME, img)
        if cv2.waitKey(1) & iter:
            break

        success, path = _search_path(img, rob, obst, tar, init_angle)
        iter += 1

    assert success, "Couldn't find a path"
    return path


def _search_path(img, rob, obst, tar, init_angle):

    # Find initial direction
    curr_pos, tar_pos = tuple(np.array(rob[:2], dtype=int)), tuple(np.array(tar[:2], dtype=int))
    curr_direction = angle_to_direction(init_angle)
    g_scores, f_scores = {curr_pos: 0}, deque([d(curr_pos, tar_pos)])
    directions_stack, stack = deque([curr_direction]), deque([curr_pos])
    came_from = {curr_pos: None}

    cv2.circle(img, tuple(rob[:2]), rob[2], color=(255, 0, 0), thickness=2)
    [cv2.circle(img, tuple(x[:2]), x[2], color=(0, 255, 0), thickness=2) for x in obst]
    cv2.circle(img, tuple(tar[:2]), tar[2], color=(0, 0, 255), thickness=2)

    while len(stack):
        curr_pos = stack.popleft()
        curr_g = g_scores[curr_pos]
        curr_direction = directions_stack.popleft()
        _ = f_scores.popleft()

        directions = (np.arange(curr_direction - 1, curr_direction + 2) % 8).astype(int)
        next_steps = DIRECTIONS_ADDONS[directions] + curr_pos
        for i, t_pos in enumerate(next_steps):
            t_pos, t_point = tuple(t_pos), (*t_pos, rob[2], -1)
            cv2.circle(img, t_pos, radius=2, color=(255, 0, 0), thickness=2)

            if is_too_close_to_object(t_point, obst) and is_in_board_limits(t_point):
                continue

            if is_too_close_to_object(t_point, tar, mode="in"):
                came_from[t_pos] = curr_pos
                return True, extract_path(came_from, t_pos)

            g = curr_g + 1
            f = d(t_pos, tar[:2]) + g

            if g < g_scores.get(t_pos, np.inf):
                g_scores[t_pos] = g
                ind = bisect_right(f_scores, f)
                stack.insert(ind, t_pos)
                f_scores.insert(ind, f)
                directions_stack.insert(ind, directions[i])
                came_from[t_pos] = curr_pos

        cv2.circle(img, curr_pos, radius=2, color=(0, 0, 255), thickness=2)
        cv2.imshow(WINDOW_NAME, img)
        out_vid.write(img)
        time.sleep(0.2)
        cv2.waitKey(1)


    return False, []


if __name__ == '__main__':
    # get_new_colors_range()
    # angle_to_origin(GREEN)
    # 1) Preprocessing :
    # - (Fix obstacles and target)
    # - Find the initial angle of the robot, and it's current position in the video

    show_objects()
    init_angle = find_robot_initial_angle()

    _, img = cap.read()

    obst, obst_cnts = find_object_by_color(img, GREEN)
    tar, tar_cnt = extract_largest(*find_object_by_color(img, RED))
    robot, robot_cnt = extract_largest(*find_object_by_color(img, BLUE))

    # while True:
    #     draw_route(img, [[107, 384], [305, 312], [528, 309]])
    #     cv2.imshow("Result", img)
    #
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         exit(0)

    path = search_path(img, robot, obst, tar, init_angle)
    follow_route(init_angle, path, robot, tar)




