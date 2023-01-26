import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from catalyst import dl, utils
from catalyst import metrics
import numpy as np

from metrics import calc_ced, count_ced_auc


class CustomRunner(dl.SupervisedRunner):
    def on_loader_start(self, runner):
        super().on_loader_start(runner)

        self.meters = {
            'ced': metrics.AccumulativeMetric(keys=['ceds'], compute_on_call=False),
        }
        self.meters['ced'].reset(self.batch_size, len(self.loader.dataset))

        # self.loader.dataset.std = STD_INIT + (STD_END - STD_INIT) * runner.epoch_step / runner.num_epochs
        # self.loader_metrics['heatmap_std'] = self.loader.dataset.std

    def handle_batch(self, batch):
        x = batch['features']
        y = batch['targets']

        logits = self.model(x)
        self.batch['logits'] = logits

        if not self.is_train_loader:
            with torch.no_grad():
                # pts_gt = masks2pts(y)
                # pts_pred = masks2pts(logits)
                pts_pred = logits
                pts_gt = y
                ceds = calc_ced(pts_pred, pts_gt, batch['width'], batch['height'])
                self.meters['ced'].update(ceds=ceds)

    def on_loader_end(self, runner):
        if not self.is_train_loader:
            scores = np.sort(self.meters['ced'].compute()['ceds'].numpy())
            auc = count_ced_auc(scores)[0]
            self.loader_metrics['ced'] = auc

        super().on_loader_end(runner)