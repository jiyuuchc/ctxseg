import pickle
import numpy as np
import tifffile
import pandas as pd
from absl import app, flags
from pathlib import Path
from tqdm import tqdm
from skimage.measure import regionprops
from ctxseg.segmentation.flow import flow_to_mask
from ctxseg.segmentation.utils import remove_small_instances, clean_up_mask
from ctxseg.segmentation.metrics import LabelMetrics, MultiLabelMetrics

flags.DEFINE_string("logpath", None, "dir for inference results", required=True)
flags.DEFINE_string("dataset", None, "")
flags.DEFINE_integer("niter", 500, "")
flags.DEFINE_float("threshold", 0.1, "")
flags.DEFINE_float("maxflowerr", 0., "")
flags.DEFINE_float("stepsize", 0.5, "")
flags.DEFINE_integer("minsize", 100, "Min cell size")
flags.DEFINE_float("scorethreshold", 0., "")
flags.DEFINE_integer("nsamples", -1, "number of samples to analyze")
flags.DEFINE_boolean('fast', False, "whether to recompute mask")

FLAGS = flags.FLAGS

def _get_mask(flow):
    amptitude = np.sqrt((flow ** 2).sum(axis=-1, keepdims=True))
    flow = np.where(amptitude > FLAGS.threshold, flow, 0)
    # flow = np.where(amptitude > min_value, flow/amptitude, 0)
    mask = flow_to_mask(flow, niter=FLAGS.niter, step_size=FLAGS.stepsize)
    mask = np.stack([remove_small_instances(m, FLAGS.minsize) for m in mask])

    return mask

def _eval(dataset):
    metric = None

    root = Path(FLAGS.logpath)

    ids = [fn.name.split("_label")[0] for fn in (root/dataset).glob("*_label*")]

    for img_n in tqdm(ids):
        gt = tifffile.imread(root/dataset/f"{img_n}_label.tif")
        gt = remove_small_instances(gt, 100) # hack correct test dataset error

        out_file = root/dataset/f"{img_n}_output.tif"
        if FLAGS.fast and out_file.exists():
            label = tifffile.imread(out_file)
        else:
            pred = tifffile.imread(root/dataset/f"{img_n}_pred.tif") 
            pred = np.moveaxis(pred, 0, -1)
            label = _get_mask(pred)

            tifffile.imwrite(root/dataset/f"{img_n}_output.tif", label.astype("uint16"))
        
            if FLAGS.scorethreshold > 0:
                new_labels = []
                for pred_i, label_i in zip(pred, label):
                    score_img = (pred_i ** 2).sum(axis=-1)
                    label_i = clean_up_mask(label_i).astype(int)
                    props = regionprops(label_i, intensity_image=score_img)
                    scores = np.array([p.mean_intensity for p in props])
                    lut = np.arange(label_i.max() + 1, dtype=int)
                    lut[1:] = np.where(scores >= FLAGS.scorethreshold, lut[1:], 0)
                    new_labels.append(lut[label_i])
                label = np.stack(new_labels)

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
