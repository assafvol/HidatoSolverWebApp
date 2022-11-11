import math
import cv2
import numpy as np
from matplotlib import pyplot as plt

def pre_process(img):
    """Receives colored image and returns binary image"""

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)
    return img_threshold


def find_biggest_contour(contours):
    """Receives a list of contours a returns the contour of biggest area of biggest area considering only contour of
       area > 50 and of polygon approximation of more than or equal to 20 corners (smallest hidato of side with 3
       hexagons has 30 corners). Also returns the points of the polygon approximation the perimeter
       of the biggest contour  for potential further fitting of polygon"""

    biggest_polygon = np.array([])
    biggest_contour = np.array([])
    biggest_contour_perimeter = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.003 * perimeter, True)
            if area > max_area and len(polygon) >= 20:
                biggest_contour = contour
                biggest_polygon = polygon
                biggest_contour_perimeter = perimeter
                max_area = area
    return biggest_contour, biggest_polygon, biggest_contour_perimeter


def fit_contour_to_a_possible_hidato_size(contour, polygon, perimeter):
    """In case number of points of the polygon is not valid for a hidato (30,42,54,66... or in general
    6 + 12i for non negative integer i), refit new polygons by until getting to the nearest valid number of points
    and return the newly fitted polygon. If the number of points is valid for a hidato, simply return the input
     polygon"""
    n = len(polygon)
    if n in [30, 42, 54, 66, 78, 90]:
        return polygon
    else:
        alpha = 0.003
        delta = 0.0001
        goal_n = min([30, 42, 54, 66, 78, 90], key=lambda x: abs(x - n))
        sign = np.sign(n - goal_n)

        curr_polygon = polygon
        curr_n = n
        while np.sign(curr_n - goal_n) == sign:
            alpha += sign * delta
            curr_polygon = cv2.approxPolyDP(contour, alpha * perimeter, True)
            curr_n = len(curr_polygon)

        if curr_n != goal_n:
            raise Exception("Failed to fit valid polygon for hidato to contour")
        else:
            return curr_polygon


def apply_perspective_transform(v, mat):
    """Returns a vector which is the result of apply mat to v
    Arguments:
    v - numpy vector
    mat - numpy matrix"""
    v_t = []
    for p in v:
        px = (mat[0][0] * p[0] + mat[0][1] * p[1] + mat[0][2]) / (mat[2][0] * p[0] + mat[2][1] * p[1] + mat[2][2])
        py = (mat[1][0] * p[0] + mat[1][1] * p[1] + mat[1][2]) / (mat[2][0] * p[0] + mat[2][1] * p[1] + mat[2][2])
        p_after = (int(px), int(py))
        v_t.append(p_after)
    return np.array(v_t)


# def find_corners(contour):
#     shape = contour.shape
#     contour = contour.reshape((shape[0], shape[2]))
#     corner_idxs = np.sum((contour - center) ** 2, axis=1).argsort()[-4:]
#     return contour[corner_idxs]


def reorder(points):
    new_points = np.zeros_like(points)
    new_points[0] = min(points, key=lambda el: sum(el))
    new_points[1] = min(points, key=lambda el: el[0] - el[1])
    new_points[2] = max(points, key=lambda el: el[0] - el[1])
    new_points[3] = max(points, key=lambda el: sum(el))
    return new_points


def distance(v1, v2, weights =(1,1)):
    """distance between two 1d vectors represented bynumpy arrays"""

    return np.sqrt(np.sum(((v1 - v2) ** 2) * np.array(weights)))


def get_corner_indices(labels):
    corner_idxs = set()
    n = len(labels)
    for i in range(n + 4):
        if len({labels[(i - 1) % n], labels[i % n], labels[(i + 1) % n]}) == 3:
            corner_idxs.add(i % n)
    return sorted(list(corner_idxs))


def find_rectangle_corners(corner_idxs, contour):
    n = len(contour)
    top_left_idx = min(corner_idxs, key=lambda c_idx: distance(v1=(0, 0), v2=contour[c_idx], weights=(0.4, 0.6)))
    bottom_left_idx = (corner_idxs[(corner_idxs.index(top_left_idx) + 2) % 6] + 1) % n
    bottom_right_idx = corner_idxs[(corner_idxs.index(top_left_idx) + 3) % 6]
    top_right_idx = (corner_idxs[(corner_idxs.index(top_left_idx) + 5) % 6] + 1) % n

    return np.array([top_left_idx, bottom_left_idx, top_right_idx, bottom_right_idx])


def find_hexagons(circum, corners_warped):
    """strategy : find middle y point of every row of hexagons, find unique x points excluding first and last.
    then the center are: for odd_rows even xs """
    all_x = sorted(circum[:, 0])
    all_y = sorted(circum[:, 1])
    n_rows = n_cols = len(circum) // 6
    width = max(all_x) - min(all_x)
    height = max(all_y) - min(all_y)
    h = height / n_rows
    w = width / n_cols
    # x = get_different_values(all_x, accuracy=w / 5)[1:-1]
    x = get_lattice_x_values(all_x)[1:-1]
    #     y = get_different_values(all_y,accuracy=h/5)
    top_left_w, bottom_left_w, top_right_w, bottom_right_w = corners_warped
    y_r = np.sort(circum[circum[:, 0] >= min(top_right_w[0], bottom_right_w[0])][:, 1])
    y_l = np.sort(circum[circum[:, 0] <= max(top_left_w[0], bottom_left_w[0])][:, 1])
    y = np.mean(np.stack([y_l, y_r], axis=1), axis=1)
    y = [np.mean([y[i], y[i + 1]]) for i in range(1, len(y) - 1, 2)]

    rows = [[] for _ in range(n_rows)]
    x_e = x[0::2]
    x_o = x[1::2]
    n_e, n_o = len(x_e), len(x_o)
    mid = n_rows // 2
    rows[mid] = [[x_e[i], y[mid]] for i in range(0, n_cols)]
    for j in range(1, (n_rows - 1) // 2 + 1):
        if j % 2 == 1:
            rows[mid + j] = [[x_o[k], y[mid + j]] for k in range(j // 2, n_o - j // 2)]
            rows[mid - j] = [[x_o[k], y[mid - j]] for k in range(j // 2, n_o - j // 2)]
        else:
            rows[mid + j] = [[x_e[k], y[mid + j]] for k in range(j // 2, n_e - j // 2)]
            rows[mid - j] = [[x_e[k], y[mid - j]] for k in range(j // 2, n_e - j // 2)]
    return rows, h, w


def get_lattice_x_values(all_x):
    n_rows = len(all_x) // 6
    n_s = (n_rows + 1) // 2
    group_lengths = iter([2] + [4] * (n_s-1) + [2] * n_rows + [4]* (n_s-1) + [2])
    x = []
    i = 0
    while i < len(all_x):
        group_len = next(group_lengths)
        x.append(np.mean(all_x[i:i + group_len]))
        i += group_len
    return x


def get_different_values(lst, accuracy):
    groups = []
    curr_group = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] < accuracy:
            curr_group.append(lst[i])
        else:
            groups.append(curr_group)
            curr_group = [lst[i]]
    if tuple(groups[-1]) != tuple(curr_group):
        groups.append(curr_group)
    return [np.mean(group) for group in groups]


def get_boxes(lattice, h, w, img):
    height_img, width_img = img.shape[:2] if len(img.shape) > 2 else img.shape
    boxes = [[] for _ in range(len(lattice))]
    for i, row in enumerate(lattice):
        for j, (x, y) in enumerate(row):
            left = max(int(x - w // 2), 0)
            right = min(int(x + w // 2), width_img)
            up = max(int(y - h // 2), 0)
            down = min(int(y + h // 2), height_img)
            box = img[up:down + 1, left:right + 1]
            boxes[i].append(box)
    return boxes


def plot_boxes(boxes):
    n_hexagons = sum(len(row) for row in boxes)
    q = math.ceil(np.sqrt(n_hexagons))
    l = n_hexagons // q + 1
    fig, axes = plt.subplots(nrows=l, ncols=q)

    n = 0
    for i, row in enumerate(boxes):
        for j, box in enumerate(row):
            r = n // q
            c = n % q
            axes[r, c].imshow(box)
            n += 1

    plt.show()
    plt.tight_layout()


def crop_from_image(img, bbox):
    img_width, img_height = img.shape
    x, y, w, h = bbox
    x_pad = w // 7
    y_pad = h // 7
    up = max(y - y_pad, 0)
    down = min(y + h + y_pad, img_height)
    left = max(x - x_pad, 0)
    right = min(x + w + x_pad, img_height)
    return img[up:down, left:right]


def get_optimal_font_scale(text, width):
    for scale in reversed(range(1, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if new_width <= width:
            return scale/10
    return 1


def put_text_centered(img, txt, center, scale, color=(0, 255, 0), thickness=3):
    (text_width, text_height), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,
                                                   scale, thickness)
    img_with_text = cv2.putText(img,
                                txt,
                                (int(center[0] - text_width / 2), int(center[1] + text_height / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                scale,
                                color,
                                thickness,
                                cv2.LINE_AA)
    return img_with_text


def get_four_hidato_corners(polygon):
    # vectors = np.roll(polygon, -1, axis=0) - polygon
    # k_means_model = KMeans(n_clusters=6)
    # vectors_labels = k_means_model.fit_predict(vectors)
    vectors_labels = cluster_by_angles(polygon)
    corner_indices = get_corner_indices(vectors_labels)
    four_corners = polygon[find_rectangle_corners(corner_indices, polygon)]
    return four_corners

def cluster_by_angles(polygon):
    vectors = np.roll(polygon, -1, axis=0) - polygon
    angles = (np.arctan2(vectors[:,1],vectors[:,0]) % (2*np.pi)) * (360/(2*np.pi))
    
    order = np.argsort(angles)
    original_order = np.argsort(order)
    
    angles = angles[order]
    diffs = (-(np.roll(angles,shift=1) - angles)) % 360
    transition_diff = np.sort(diffs)[-6]

    labels = np.cumsum(diffs >= transition_diff) % 6
    return labels[original_order]

def transform_perspective(polygon, four_corners, img, img_size):
    img_width, img_height = img_size
    n_points = len(polygon)
    # 2 * n_s - 1 = n_points/6 where n_s is the number of hexagons along one of the sides of the hidato and n_points
    # is the number of points in the hidato polygon.
    n_s = (n_points / 6 + 1) / 2
    x_1 = ((n_s / 2) / (2 * n_s - 1)) * img_width
    x_2 = ((n_s / 2 + n_s - 1) / (2 * n_s - 1)) * img_width

    # perspective transformation of image, hidato polygon, and polygon four corners
    pts1 = np.float32(four_corners)
    pts2 = np.float32([[x_1, 0], [x_1, img_height], [x_2, 0], [x_2, img_height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (img_width, img_height))
    polygon_warped = apply_perspective_transform(polygon, matrix)
    corners_warped = apply_perspective_transform(four_corners, matrix)

    return imgWarpColored, polygon_warped, corners_warped, matrix


# def remove_hexagonal_grid_and_noise(imgWarpBinary, imgWarpGray, noise_size):
#     nb_components, output, stats, _ = cv2.connectedComponentsWithStats(imgWarpBinary, connectivity=4)
#     sizes = stats[:, -1]

#     # Remove largest component (assumed to be the hexagonal grid)
#     largest_components_label = np.argmax(sizes[1:]) + 1
#     imgWarpBinary[output == largest_components_label] = 0
#     imgWarpGray[output == largest_components_label] = 255
#     imgWarpGray[output == 0] = 255

#     # remove all components whose area is small than noise size
#     for i, size in enumerate(sizes):
#         if i != 0 and size < noise_size:
#             imgWarpBinary[output == i] = 0
#             imgWarpGray[output == i] = 255

#     return imgWarpBinary, imgWarpGray

def remove_hexagonal_grid_and_noise(imgWarpBinary, imgWarpGray, noise_size):
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(imgWarpBinary, connectivity=4)
    sizes = stats[:, -1]

    max_size = max(sizes[1:])
    keep = np.nonzero((noise_size <= sizes) & (sizes < max_size))[0] 
    mask = np.isin(output,keep)
    
    imgWarpBinary = np.where(mask,imgWarpBinary,0)
    imgWarpGray = np.where(mask,imgWarpGray,255)

    return imgWarpBinary, imgWarpGray


