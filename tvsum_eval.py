#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import f1_score, precision_score, recall_score

from run_va import run
from utils.summary_utils import smooth, load_flat_scores


# Настройки
TS_ROOT = Path("D:/Desktop/Cursovaya/2020/data/video/tvsum50_ver_1_1/ydata-tvsum50-v1_1/")
REPORT_FOLDER = Path("D:/Desktop/Cursovaya/2020/code/reports/va_debug/")
GPU_MEM = 512
SBD_THRESHOLD = 0.5
# Так в исходном матлабовском скрипте оценки
SUMMARY_PERCENT = 0.15

### Относительные пути. Зависят от того, как распакованы внутренние архивы
TS_DATA_FOLDER = TS_ROOT / "ydata-tvsum50-data/data/"
TS_INFO_PATH = TS_DATA_FOLDER / "ydata-tvsum50-info.tsv"
TS_ANNO_PATH = TS_DATA_FOLDER / "ydata-tvsum50-anno.tsv"
TS_VIDEO_FOLDER = TS_ROOT / "ydata-tvsum50-video/video/"
###


def get_averaged_scores(anno_path: Path) -> dict:
    res = {}
    with open(str(anno_path)) as anno_file:
        for idx, line in enumerate(anno_file, 1):
            parts = line.split("\t")
            assert len(parts) == 3, f"Line {idx}"
            vid_id = parts[0]
            score_str = parts[2]
            scores = [int(s) for s in score_str.split(",")]
            res.setdefault(vid_id, [])
            res[vid_id].append(scores)
    assert len(res) == 50, len(res)
    for k, v in res.items():
        assert len(v) == 20, f"N scores for {k}: {len(v)} (expected 20)"
        res[k] = np.mean(v, axis=0)
        assert len(res[k]) == len(v[0]), f"Unequal scoring for {k}"
    return res


def run_assessment(video_folder: Path, out_folder: Path, gpu_mem: int, sbd_threshold: float = 0.5) -> None:
    videos = list(video_folder.glob("*.mp4"))
    assert len(videos) == 50, f"Expected 50 videos in folder, got {len(videos)}"
    for idx, vid_path in enumerate(videos, 1):
        print(f">>>>> {idx} / {len(videos)}")
        vid_out_folder = out_folder / vid_path.stem
        vid_out_folder.mkdir(parents=True, exist_ok=True)
        run(vid_path=vid_path, out_path=vid_out_folder, 
            gpu_mem=gpu_mem, sbd_threshold=sbd_threshold)


def scale_to_ts(scores: np.ndarray, to_min=1, to_max=5) -> np.ndarray:
    return minmax_scale(scores, (to_min, to_max))


def compute_metrics(report_root_dir: Path,
                    summary_percent: float,
                    avg_gt_scores: dict) -> pd.DataFrame:
    res = {
        "id": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }
    for score_path in report_root_dir.rglob("*.pkl"):
        vid_id = score_path.parent.name
        #
        gt_scores = avg_gt_scores[vid_id]
        scores = load_flat_scores(score_path)
        scores = scale_to_ts(scores)
        scores = smooth(scores)
        n_gt_scores = len(gt_scores)
        n_pred_scores = len(scores)
        assert n_gt_scores == n_pred_scores, f"Expected {n_gt_scores} scores, got {n_pred_scores}" 
        #
        _pc = (1 - summary_percent) * 100
        pc_for_predicted = np.percentile(scores, _pc)
        pc_for_gt = np.percentile(gt_scores, _pc)
        #
        mask_pred = scores >= pc_for_predicted
        mask_true = gt_scores >= pc_for_gt
        #
        res["f1"].append(f1_score(mask_true, mask_pred))
        res["precision"].append(precision_score(mask_true, mask_pred))
        res["recall"].append(recall_score(mask_true, mask_pred))
        res["id"].append(vid_id)
    return pd.DataFrame(res)


def main():
    run_assessment(video_folder=TS_VIDEO_FOLDER,
                   out_folder=REPORT_FOLDER,
                   gpu_mem=GPU_MEM, sbd_threshold=SBD_THRESHOLD)
    avg_gt_scores = get_averaged_scores(TS_ANNO_PATH)
    metrics_df = compute_metrics(report_root_dir=REPORT_FOLDER,
                                 summary_percent=SUMMARY_PERCENT,
                                 avg_gt_scores=avg_gt_scores)
    metrics_path = REPORT_FOLDER / "tvsum_metrics.csv"
    metrics_df.to_csv(str(metrics_path), sep="\t", index=False)
    with open(metrics_path) as f:
        f.write(f"mean\t{metrics_df.f1.mean()}\t{metrics_df.precision.mean()}\t{metrics_df.recall.mean()}\n")


if __name__ == '__main__':
    main()
