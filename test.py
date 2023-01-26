import cv3
from tqdm import trange
import torch
import numpy as np
from pathlib import Path
import os.path

from utils import filter_img_paths, expand_box, read_pts
from data import Dataset, make_loader
from models import BaselineModel
from metrics import count_ced_auc
from config import LOG_DIR, BATCH_SIZE, DATA_DIR, TEST_PATH_300W, TEST_PATH_MENPO, DEVICE



def calc_ced_val(logits, targets, width, height):
    nf = (width * height) ** 0.5

    targets = targets * torch.tensor([width, height])
    logits = logits * torch.tensor([width, height])
    ced = (((logits - targets) ** 2).sum(-1) ** 0.5).mean(-1) / nf
    return ced



def get_ceds(model, test_ds):
    ceds = []
    for idx in trange(len(test_ds)):
        img = cv3.imread(test_ds.img_paths[idx])
        pts = read_pts(test_ds.pts_paths[idx])

        rect = list(map(int, open(test_ds.box_paths[idx]).read().split()))
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        img, pts, rect = expand_box(img, pts, rect=rect, k=test_ds.k)

        xmin, ymin, xmax, ymax = rect
        pts = (pts - [xmin, ymin]) / [xmax-xmin, ymax-ymin]
        img_pad = img.copy()
        img = cv3.crop(img, ymin, ymax, xmin, xmax)


        img = test_ds.post_transforms(image=img)['image']

        pred = model(img[None].to(DEVICE)).detach().cpu().reshape(-1, 2)
        pred_abs = pred.numpy() * [xmax-xmin, ymax-ymin] + [xmin, ymin]


        targets = pts.reshape(-1)
        targets = torch.FloatTensor(targets)


        pts_abs = targets.reshape(-1, 2).numpy() * [xmax-xmin, ymax-ymin] + [xmin, ymin]

        xmin, ymin, xmax, ymax = (*pred_abs.min(0), *pred_abs.max(0))

        img_expand, pts_expand, rect = expand_box(img_pad, pts_abs, rect=[xmin, ymin, xmax, ymax], k=0.5)
        x0,y0,x1,y1 = rect
        img_crop = cv3.crop(img_expand, y0,y1,x0,x1)

        pts_new = (pts_expand - [x0, y0]) / [x1-x0, y1-y0]

        img_new = test_ds.post_transforms(image=img_crop)['image']

        new_pred = model(img_new[None].to(DEVICE)).detach().cpu().reshape(-1, 2)

        h, w = img_crop.shape[:2]

        ced = calc_ced_val(new_pred, torch.tensor(pts_new), w, h)
        ceds.append(ced)
    return ceds
        

def main():

    for path in [TEST_PATH_MENPO, TEST_PATH_300W]:
        print(path)
        test_paths = filter_img_paths([Path(path).glob('*.jpg')], check_box=True)

        test_ds, _ = make_loader(test_paths, mode='test', batch_size=BATCH_SIZE, num_workers=0, k=0.3)
        model = BaselineModel()
        model_path = os.path.join(LOG_DIR, 'checkpoints/model.best.pth')
        model.load_state_dict(torch.load(model_path))
        model.to(DEVICE)
        model.eval()

        ceds = get_ceds(model, test_ds)
        scores = np.sort(ceds)
        auc = count_ced_auc(scores)[0]
        print(auc)


if __name__ == '__main__':
    main()