import numpy as np
import torch

def calc_ced(logits: torch.Tensor, targets: torch.Tensor, width, height):
    nf = (width * height) ** 0.5
    scale = torch.stack((width, height), axis=1).unsqueeze(1)

    targets = targets.reshape(-1, 68, 2) * scale
    logits = logits.reshape(-1, 68, 2) * scale
    ceds = (((logits - targets) ** 2).sum(-1) ** 0.5).mean(-1) / nf
    return ceds



def count_ced_auc(errors):
  if not isinstance(errors, list):
    errors = [errors]

  aucs = []
  for error in errors:
    auc = 0
    proportions = np.arange(error.shape[0], dtype=np.float32) / error.shape[0]
    assert (len(proportions) > 0)

    step = 0.01
    for thr in np.arange(0.0, 1.0, step):
      gt_indexes = [idx for idx, e in enumerate(error) if e >= thr]
      if len(gt_indexes) > 0:
        first_gt_idx = gt_indexes[0]
      else:
        first_gt_idx = len(error) - 1
      auc += proportions[first_gt_idx] * step
    aucs.append(auc)
  return aucs