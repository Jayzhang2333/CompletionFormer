"""
    CompletionFormer
    ======================================================================

    CompletionFormerMetric implementation
"""


import torch
from . import BaseMetric

class CompletionFormerMetric(BaseMetric):
    def __init__(self, args):
        super(CompletionFormerMetric, self).__init__(args)

        self.args = args
        self.t_valid = 0.1
        self.max_depth = args.max_depth

        self.metric_name = [
            'RMSE', 'MAE', 'iRMSE', 'iMAE', 'iAbsRel', 'REL',
            'SILog',                         # ← added here
            'D^1', 'D^2', 'D^3', 'D102', 'D105', 'D110'
        ]

    def evaluate(self, sample, output, mode=None):
        with torch.no_grad():
            pred = output['pred'].detach()
            gt   = sample['gt'].detach()

            pred_inv = 1.0 / (pred + 1e-8)
            gt_inv   = 1.0 / (gt   + 1e-8)

            # 1) valid‐pixel mask
            mask = (gt > self.t_valid) & (gt < self.max_depth)
            num_valid = mask.sum().float()

            # 2) apply mask
            pred     = pred[mask]
            gt       = gt[mask]
            pred_inv = pred_inv[mask]
            gt_inv   = gt_inv[mask]

            # zero out tiny values to avoid inf’s
            pred_inv[pred <= self.t_valid] = 0.0
            gt_inv[gt   <= self.t_valid] = 0.0

            # 3) standard RMSE & MAE
            diff     = pred - gt
            diff_abs = diff.abs()
            diff_sqr = diff.pow(2)

            rmse = torch.sqrt(diff_sqr.mean())
            mae  = diff_abs.mean()

            # 4) inverse‐depth metrics: iRMSE, iMAE, iAbsRel
            diff_inv     = pred_inv - gt_inv
            diff_inv_abs = diff_inv.abs()
            diff_inv_sqr = diff_inv.pow(2)

            irmse  = torch.sqrt(diff_inv_sqr.mean())
            imae   = diff_inv_abs.mean()

            diff_inv_rel = diff_inv_abs / (gt_inv + 1e-8)
            iabsrel      = diff_inv_rel.mean()

            # 5) AbsRel on regular depth
            rel = (diff_abs / (gt + 1e-8)).mean()

            # 6) SILog
            eps = 1e-8
            log_pred = torch.log(pred + eps)
            log_gt   = torch.log(gt   + eps)
            alpha    = (log_gt - log_pred).mean()
            silog    = torch.sqrt(((log_pred - log_gt + alpha).pow(2)).mean())

            # 7) δ thresholds
            r1    = gt / (pred + eps)
            r2    = pred / (gt   + eps)
            ratio = torch.max(r1, r2)

            del_1   = (ratio < 1.25   ).float().mean()
            del_2   = (ratio < 1.25**2).float().mean()
            del_3   = (ratio < 1.25**3).float().mean()
            del_102 = (ratio < 1.02   ).float().mean()
            del_105 = (ratio < 1.05   ).float().mean()
            del_110 = (ratio < 1.10   ).float().mean()

            # 8) stack in precisely the same order as metric_name
            result = torch.stack([
                rmse, mae,      # RMSE, MAE
                irmse, imae,    # iRMSE, iMAE
                iabsrel,        # iAbsRel
                rel,            # REL
                silog,          # SILog
                del_1, del_2, del_3,
                del_102, del_105, del_110
            ], dim=0)

            return result.unsqueeze(0)
