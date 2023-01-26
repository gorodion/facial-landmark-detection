import numpy as np
from itertools import chain
from typing import List


def filter_img_paths(img_paths_list, check_box=False):
    img_paths = []
    for img_path in chain(*img_paths_list):
        if check_box and not img_path.with_suffix('.box').exists():
            continue
        
        pts_path = img_path.with_suffix('.pts')
        pts = read_pts(pts_path)
        if pts.min() < 0:
            continue

        if pts.shape == (68, 2):
            img_paths.append(img_path)
    return img_paths

    
def read_pts(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))


def expand_box(img, pts, rect=None, random=False, k=0.25):
    h, w = img.shape[:2]

    if rect is None:
        xmin, ymin, xmax, ymax = (*np.min(pts, 0), *np.max(pts, 0))
    else:
        xmin, ymin, xmax, ymax = rect

    xmin, ymin, xmax, ymax = (int(x) for x in (xmin, ymin, xmax, ymax))
    box_width, box_height = xmax - xmin, ymax - ymin

    if random:
        dx1, dy1, dx2, dy2 = ((k * np.random.random(4) + 0.25) * [box_width, box_height, box_width, box_height]).astype(int)
    else:
        dx1, dy1, dx2, dy2 = (k * np.array([box_width, box_height, box_width, box_height])).astype(int)

    xmin1 = xmin - dx1
    ymin1 = ymin - dy1
    xmax1 = xmax + dx2
    ymax1 = ymax + dy2

    top_pad = -min(ymin1, 0)
    bottom_pad = max(ymax1 - h, 0)
    left_pad = -min(xmin1, 0)
    right_pad = max(xmax1 - w, 0)

    xmin1 = max(xmin1, 0)
    ymin1 = max(ymin1, 0)

    img = np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)))

    pts[:, 0] += left_pad
    pts[:, 1] += top_pad

    rect = [xmin1, ymin1, xmax1, ymax1]

    return img, pts, rect


def get_rect(pts: np.ndarray):
    return [*pts.min(0), *pts.max(0)]


def get_iou(rect1, rect2):
    left = max(rect1[0], rect2[0])
    top = max(rect1[1], rect2[1])
    right = min(rect1[2], rect2[2])
    bottom = min(rect1[3], rect2[3])

    if bottom < top or right < left:
        return 0

    inter_area = max(((right - left) * (bottom - top)), 0)
    area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

    iou = inter_area / (area1 + area2 - inter_area)
    assert iou >= 0
    return iou