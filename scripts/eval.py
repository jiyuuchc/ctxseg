import pickle
import numpy as np
import tifffile
import pandas as pd
from absl import app, flags
from pathlib import Path
from tqdm import tqdm
from ctxseg.segmentation.flow import flow_to_mask
from ctxseg.segmentation.utils import remove_small_instances, clean_up_mask
from ctxseg.segmentation.metrics import LabelMetrics, MultiLabelMetrics

flags.DEFINE_string("logpath", None, "dir for inference results", required=True)
flags.DEFINE_string("dataset", None, "")
flags.DEFINE_integer("niter", 1000, "")
flags.DEFINE_float("threshold", 0.1, "")
flags.DEFINE_float("maxflowerr", 0., "")
flags.DEFINE_integer("minsize", 100, "Min cell size")

FLAGS = flags.FLAGS

def _eval(dataset):
    metric = None

    root = Path(FLAGS.logpath)
    params = dict(
        niter=FLAGS.niter,
        threshold=FLAGS.threshold,
        max_flow_err=FLAGS.maxflowerr,
    )

    ids = [fn.name.split("_label")[0] for fn in (root/dataset).glob("*_label*")]

    for img_n in tqdm(ids):
        # img = tifffile.imread(root/dataset/f"{img_n}.tif")
        gt = tifffile.imread(root/dataset/f"{img_n}_label.tif")
        pred = tifffile.imread(root/dataset/f"{img_n}_pred.tif")

        gt = remove_small_instances(gt, 100) # hack correct test dataset error

        label = flow_to_mask(np.moveaxis(pred, 0, -1), **params)

        # optionally remove small cells
        label = np.stack([
            clean_up_mask(remove_small_instances(x, FLAGS.minsize))
            for x in label
        ])

        tifffile.imwrite(root/dataset/f"{img_n}_output.tif", label.astype("uint16"))

        if metric is None:
            metric = MultiLabelMetrics(label.shape[0])

        metric.update(label, gt, sample_id=img_n)
    
    return metric


def _keep_max(metric, micro_results):
    micro_results = micro_results.set_index(np.arange(micro_results.shape[0]))
    idx = micro_results.groupby(['image_id'])['instance_dice'].idxmax()
    sid = micro_results.loc[idx].set_index('image_id')['sample']
    sid = dict(sid)
    
    mm = LabelMetrics()
    for k, sample_id in enumerate(metric.sample_ids):
        sample_n = sid[sample_id]
        m = metric.metrics[sample_n]
        mm.pred_areas.append(m.pred_areas[k])
        mm.gt_areas.append(m.gt_areas[k])
        mm.pred_scores.append(m.pred_scores[k])
        mm.gt_scores.append(m.gt_scores[k])
        mm.ious.append(m.ious[k])

    return mm


def main(_):
    p = Path(FLAGS.logpath)
    if FLAGS.dataset is None:
        datasets = [x.name for x in p.iterdir() if x.is_dir() and not x.name.startswith(".")]
    else:
        datasets = [FLAGS.dataset]

    for ds_name in datasets:
        print(f"============ {ds_name} ==========")

        metric = _eval(ds_name)

        if metric is not None:
            macro_results, micro_results = metric.compute(micros=True)
            grouped = micro_results.groupby('image_id')

            print("Macro metrics - mean")
            print(pd.DataFrame.from_records(macro_results).mean(axis=0))
            print("Macro metrics - max")
            mm = _keep_max(metric, micro_results).compute()
            print(pd.Series(mm))

            print("Micro metrics - mean:")
            print(grouped.mean().mean()[['accuracy', 'recall', 'f1', 'instance_dice']])
            print("Micro metrics - max:")
            print(grouped.max().mean()[['accuracy', 'recall', 'f1', 'instance_dice']])
            print()

            with open(p/ds_name/f"metric.pkl", "wb") as f:
                pickle.dump(metric, f)
        else:
            import warnings
            warnings.warn(f"{ds_name} contains no data")

if __name__ == "__main__":
    app.run(main)
