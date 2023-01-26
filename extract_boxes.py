from tqdm import tqdm
import dlib
from pathlib import Path
import numpy as np

from utils import filter_img_paths, expand_box, read_pts, get_iou
from data import Dataset, make_loader
from models import BaselineModel
from metrics import calc_ced, count_ced_auc
from config import LOG_DIR, BATCH_SIZE, DATA_DIR, PREDICTOR_PATH, TEST_PATH_300W, TEST_PATH_MENPO


def calc_ced(logits, targets, width, height):
    nf = (width * height) ** 0.5
    ceds = (((logits - targets) ** 2).sum(-1) ** 0.5).mean(-1) / nf
    return ceds


def dlib_process(pred_path, detector, predictor):
    print(pred_path, 'processing')

    scores = []
    img_paths = []

    for img_path in tqdm(list(Path(pred_path).glob('*.jpg'))):
        pts_path = img_path.with_suffix('.pts')

        pts = read_pts(pts_path)

        if pts.shape != (68, 2):
            continue

        img = dlib.load_rgb_image(str(img_path))

        gt_rect = [*pts.min(0), *pts.max(0)]

        wbox, hbox = pts.max(0) - pts.min(0)

        dets = detector(img, 1)

        if len(dets) == 0:
            print('faces not found', img_path)
            continue

        det = max(dets, key=lambda det: get_iou([det.left(), det.top(), det.right(), det.bottom()], gt_rect))

        pred_rect = [det.left(), det.top(), det.right(), det.bottom()]

        iou = get_iou(pred_rect, gt_rect)

        if iou == 0:
            print('max iou is zero', img_path)
            continue

        if iou < 0.45:
            continue

        img_paths.append(img_path)

        shape = predictor(img, det)
        points = np.array([[point.x, point.y] for point in shape.parts()])

        ced = calc_ced(points, pts, wbox, hbox)
        scores.append(ced)

    return img_paths, scores


def create_box_files(img_paths, detector):
    print('Creating .box files')
    for img_path in tqdm(img_paths):
        img = dlib.load_rgb_image(str(img_path))
        pts = read_pts(img_path.with_suffix('.pts'))
        gt_rect = [*pts.min(0), *pts.max(0)]

        dets = detector(img, 1)
        det = max(dets, key=lambda det: get_iou([det.left(), det.top(), det.right(), det.bottom()], gt_rect))
        pred_rect = [det.left(), det.top(), det.right(), det.bottom()]

        box_path = img_path.with_suffix('.box')
        with open(box_path, 'w') as f:
            f.write(' '.join(map(str, pred_rect)))


def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    for pred_path in [TEST_PATH_300W, TEST_PATH_MENPO]:
        img_paths, scores = dlib_process(pred_path, detector, predictor)
        create_box_files(img_paths, detector)

        if pred_path == TEST_PATH_MENPO:
            scores = np.sort(scores)
            np.save('scores_dlib.npy', scores)
            print('scores_dlib.npy', 'saved')


if __name__ == '__main__':
    main()
