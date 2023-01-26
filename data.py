from typing import List
import numpy as np
import cv2, cv3
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from utils import read_pts, expand_box

STD_INIT = 20
STD_END = 1

class Dataset:
    SIZE = 64

    def __init__(self, img_paths: List[Path], mode='train', k=0.25, heatmap=False, std=50):
        assert mode in ['train', 'val', 'test', 'demo']
        self.img_paths = img_paths
        self.pts_paths = [path.with_suffix('.pts') for path in img_paths]
        self.box_paths = [path.with_suffix('.box') for path in img_paths]
        self.mode = mode
        self.k = k
        self.std = std
        self.heatmap = heatmap

        self.x = torch.linspace(0, self.SIZE, self.SIZE)
        self.y = torch.linspace(0, self.SIZE, self.SIZE)

        
    def __getitem__(self, idx):
        img = cv3.imread(self.img_paths[idx])
        pts = read_pts(self.pts_paths[idx])
        
        if self.mode in ('train', 'demo'):
            w, h = pts.max(0) - pts.min(0)
            img, pts, rect = expand_box(img, pts, random=True, k=self.k)
            
            out = self.aug_transforms(image=img, keypoints=pts)
            img = out['image']
            pts = np.array(out['keypoints'])
        elif self.mode == 'val':
            w, h = pts.max(0) - pts.min(0)
            img, pts, rect = expand_box(img, pts, k=self.k)
        else:  # test
            rect = list(map(int, open(self.box_paths[idx]).read().split()))
            w, h = rect[2] - rect[0], rect[3] - rect[1]
            img, pts, rect = expand_box(img, pts, rect=rect, k=self.k)
            
        xmin, ymin, xmax, ymax = rect
        pts = (pts - [xmin, ymin]) / [xmax-xmin, ymax-ymin]
        img_pad = img.copy()
        img = cv3.crop(img, ymin, ymax, xmin, xmax)
            
        if self.mode == 'demo':
            return img, pts
            
            
        img = self.post_transforms(image=img)['image']

        if self.heatmap:
            pts = pts * self.SIZE

            targets = []
            for i, (x0, y0) in enumerate(pts):
                target = torch.outer(gaussian_1d(self.x, x0, std=self.std), 
                                    gaussian_1d(self.y, y0, std=self.std))
                targets.append(target)
            targets = torch.stack(targets)
        else:
            targets = pts.reshape(-1)
            targets = torch.FloatTensor(targets)

        if self.mode == 'train':
            return {
                'features': img,
                'targets': targets,
            }
        if self.mode in ('val', 'test'):
            return {
                'features': img,
                'targets': targets,
                'height': h,
                'width': w,
            }
        # if self.mode == 'test':
        #     return {
        #         'features': img,
        #         'targets': targets,
        #         'height': h,
        #         'width': w,
        #         'xmin': xmin,
        #         'ymin': ymin,
        #         'xmax': xmax,
        #         'ymax': ymax,
        #         'img_pad': img_pad
        #     }

    def __len__(self):
        return len(self.img_paths)

    aug_transforms =  albu.Compose([
        albu.ShiftScaleRotate(rotate_limit=20, scale_limit=0.3, border_mode=cv2.BORDER_CONSTANT),
        albu.RandomBrightnessContrast(p=0.2),
        albu.GaussianBlur(p=0.25)
        # albu.HorizontalFlip(),
    ], keypoint_params=albu.KeypointParams(format='xy', remove_invisible=False))


    post_transforms = albu.Compose([
        albu.Resize(SIZE, SIZE),
        albu.Normalize(),
        ToTensorV2()
    ])


def make_loader(*data_args, mode=None, batch_size=128, num_workers=0, **data_kwargs):
    assert mode in ('train', 'val', 'test')
    is_train = mode == 'train'
    dataset = Dataset(
        *data_args,
        mode=mode,
        **data_kwargs,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=is_train,
        shuffle=is_train,
        num_workers=num_workers
    )
    return dataset, dataloader