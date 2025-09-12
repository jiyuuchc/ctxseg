import numpy as np

from skimage.measure import regionprops
from .utils import clean_up_mask

def box_intersection(boxes_a, boxes_b):
    """Compute pairwise intersection areas between boxes.

    Args:
      boxes_a: [..., N, 2d]
      boxes_b: [..., M, 2d]

    Returns:
      its: [..., N, M] representing pairwise intersections.
    """
    minimum = np.minimum
    maximum = np.maximum
    boxes_a = np.array(boxes_a)
    boxes_b = np.array(boxes_b)

    ndim = boxes_a.shape[-1] // 2
    assert ndim * 2 == boxes_a.shape[-1]
    assert ndim * 2 == boxes_b.shape[-1]

    min_vals_1 = boxes_a[..., None, :ndim]  # [..., N, 1, d]
    max_vals_1 = boxes_a[..., None, ndim:]
    min_vals_2 = boxes_b[..., None, :, :ndim]  # [..., 1, M, d]
    max_vals_2 = boxes_b[..., None, :, ndim:]

    min_max = minimum(max_vals_1, max_vals_2)  # [..., N, M, d]
    max_min = maximum(min_vals_1, min_vals_2)

    intersects = maximum(0, min_max - max_min)  # [..., N, M, d]

    return intersects.prod(axis=-1)


def _analyze_mask(mask):
    mask = clean_up_mask(mask)
    rps = regionprops(mask)
    if len(rps) == 0:
        areas = np.zeros([0], dtype=int)
        bboxes = np.zeros([0, 4], dtype=int)
    else:
        areas = np.stack([rp.area for rp in rps])
        bboxes = np.stack([rp.bbox for rp in rps])
    
    return mask, areas, bboxes


class LabelMetrics:
    """Compute various metrics based on labels"""
    def __init__(self):
        self.pred_areas = []
        self.gt_areas = []
        self.pred_scores = []
        self.gt_scores = []
        self.ious = []

    def _update(self, pred_its, pred_areas, gt_areas):
        n_pred, n_gt = pred_its.shape

        if n_gt > 0:
            pred_best = pred_its.max(axis=1)
            pred_best_matches = pred_its.argmax(axis=1)

            assert (pred_best <= pred_areas).all()
            assert (pred_best <= gt_areas[pred_best_matches]).all()

            pred_dice = pred_best * 2 / (pred_areas + gt_areas[pred_best_matches])
            pred_ious = pred_best / (pred_areas + gt_areas[pred_best_matches] - pred_best)

        else:
            pred_dice = np.zeros([n_pred])
            pred_ious = np.zeros([n_pred])

        if n_pred > 0:
            gt_best = pred_its.max(axis=0)
            gt_best_matches = pred_its.argmax(axis=0)

            assert (gt_best <= gt_areas).all()
            assert (gt_best <= pred_areas[gt_best_matches]).all()

            gt_dice = gt_best * 2 / (gt_areas + pred_areas[gt_best_matches])
            # gt_ious = gt_best / (gt_areas + pred_areas[gt_best_matches] - gt_best)
        
        else:
            gt_dice = np.zeros([n_gt])

        return pred_dice, gt_dice, pred_ious


    def update(self, pred_mask, gt_mask):
        pred_mask, pred_areas, pred_bboxes = _analyze_mask(pred_mask)
        gt_mask, gt_areas, gt_bboxes = _analyze_mask(gt_mask)

        box_its = box_intersection(pred_bboxes, gt_bboxes)

        def _get_its(pid, gid):
            r0, r1 = np.split(pred_bboxes[pid], 2)
            box = tuple(slice(a,b) for a, b in zip(r0, r1))
            return np.count_nonzero( (gt_mask[box] == gid+1) & (pred_mask[box]==pid+1))

        mask_its = np.zeros_like(box_its, dtype=int)
        ids = np.where(box_its > 0)
        mask_its[ids] = [_get_its(pid, gid) for pid, gid in zip(*ids)]

        pred_scores, gt_scores, ious = self._update(mask_its, pred_areas, gt_areas)

        self.pred_areas.append(pred_areas)
        self.gt_areas.append(gt_areas)
        self.pred_scores.append(pred_scores)
        self.gt_scores.append(gt_scores)
        self.ious.append(ious)

    def _compute(self, gt_areas, pred_areas, gt_scores, pred_scores, ious,  iou_threshold):
        n_gts = len(gt_areas)
        n_preds = len(pred_areas)
        n_tps = np.count_nonzero(np.array(ious) >= iou_threshold)

        if n_preds == 0:
            pred_dice = 0
        else:
            pred_dice = (pred_areas / pred_areas.sum() * pred_scores).sum()

        if n_gts == 0:
            gt_dice = 0
        else:
            gt_dice = (gt_areas / gt_areas.sum() * gt_scores).sum()

        dice = (pred_dice + gt_dice) / 2

        return dict(
            n_preds = n_preds,
            n_gts = n_gts,
            n_tps = n_tps,
            accuracy = n_tps / n_preds if n_preds > 0 else float('nan'),
            recall = n_tps / n_gts if n_gts > 0 else float('nan'),
            f1 = n_tps / ((n_preds * n_gts) ** .5) if n_tps > 0 else 0,
            instance_dice = dice,
            ap = n_tps /(n_gts + n_preds - n_tps) if n_gts + n_preds > 0 else float('nan'),
        )
    
    def compute(self, iou_threshold=.5, micros=False):
        if len(self.pred_areas) == 0:
            return None
        # micro stats
        if micros:
            micros = []
            for k in range(len(self.pred_areas)):
                micros.append(self._compute(
                    self.gt_areas[k],
                    self.pred_areas[k], 
                    self.gt_scores[k],
                    self.pred_scores[k],
                    self.ious[k],
                    iou_threshold,
                ))

        # macro stats
        pred_areas = np.concatenate(self.pred_areas)
        pred_scores = np.concatenate(self.pred_scores)
        gt_areas = np.concatenate(self.gt_areas)
        gt_scores = np.concatenate(self.gt_scores)
        ious = np.concatenate(self.ious)

        macros = self._compute(gt_areas, pred_areas, gt_scores, pred_scores, ious, iou_threshold)

        if micros:
            return macros, micros
        else:
            return macros


class MultiLabelMetrics:
    def __init__(self, n, *args, **kwargs):
        self.metrics = [LabelMetrics(*args, **kwargs) for _ in range(n)]

    def update(self, pred_masks, gt_mask):
        gt_mask, gt_areas, gt_bboxes = _analyze_mask(gt_mask)
        if gt_bboxes.shape[0] > 0:
            gt_bms = [gt_mask[box[0]:box[2], box[1]:box[3]] == gid + 1 for gid, box in enumerate(gt_bboxes)]
        else:
            gt_bms = []

        assert len(self.metrics) == len(pred_masks), f"{len(self.metrics)} != {len(pred_masks)}"
        for metric, pred_mask in zip(self.metrics, pred_masks):
            pred_mask, pred_areas, pred_bboxes = _analyze_mask(pred_mask)
            box_its = box_intersection(pred_bboxes, gt_bboxes)

            mask_its = np.zeros_like(box_its, dtype=int)
            for pid, gid in zip(*np.where(box_its > 0)):
                box = gt_bboxes[gid]
                pred_bm = pred_mask[box[0]:box[2], box[1]:box[3]] == pid + 1
                mask_its[pid, gid] = np.count_nonzero(
                    gt_bms[gid] & pred_bm
                )

            pred_scores, gt_scores, ious = metric._update(mask_its, pred_areas, gt_areas)

            metric.pred_areas.append(pred_areas)
            metric.gt_areas.append(gt_areas)
            metric.pred_scores.append(pred_scores)
            metric.gt_scores.append(gt_scores)
            metric.ious.append(ious)


    def compute(self, iou_threshold=.5, micros=False):
        import jax
        import pandas as pd
        macros = []
        df = None
        for i, metric in enumerate(self.metrics):
            if micros:
                ma, mi = metric.compute(iou_threshold=iou_threshold, micros=micros)
                mi = jax.tree.map(lambda *x: x, *mi)
                df_ = pd.DataFrame.from_dict(mi)
                df_['sample'] = i
                if df is not None:
                    df = pd.concat([df, df_], axis = 0)
                else:
                    df = df_
            else:
                ma = metric.compute(iou_threshold=iou_threshold, micros=micros)

            macros.append(ma)
        
        return macros, df
