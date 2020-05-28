from pathlib import Path
import pickle

import cv2
import numpy as np
from scipy.signal import lfilter, lfilter_zi, filtfilt, butter

from utils.gpu_safe import video_reader


def load_shot_scores(report_path: Path) -> list:
    assert report_path.exists()
    # Прекрасно работает и без str, но инспекции PyCharm раздражают
    with open(str(report_path), "rb") as rep_file:
        report = pickle.load(rep_file)
    return report["frame_scores_per_shots"]


def load_flat_scores(report_path: Path) -> np.ndarray:
    return np.hstack(load_shot_scores(report_path))


def get_fps(vid_path) -> float:
    return cv2.VideoCapture(str(vid_path)).get(cv2.CAP_PROP_FPS)


def get_cut_percentile(flat_scores: np.ndarray, fps: float, max_len_secs: int) -> float:
    time_percent = 1 - max_len_secs / (len(flat_scores) / fps)
    percentile = np.percentile(flat_scores, time_percent * 100)
    return percentile


def get_cut_percentile_for_frames(flat_scores: np.ndarray, max_frames: int) -> float:
    frame_percent = 1 - max_frames / len(flat_scores)
    percentile = np.percentile(flat_scores, frame_percent * 100)
    return percentile


def select_best_frames(flat_scores: np.ndarray, fps, max_len_secs: int) -> tuple:
    cut_percentile = get_cut_percentile(flat_scores, fps, max_len_secs)
    res = np.argwhere(flat_scores >= cut_percentile).ravel().astype(int)
    res.sort()
    return res, cut_percentile


def smooth(x, window_len=200, window='filtfilt', order=2):
    """
    Из коллекции сниппетов SciPy:
    smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal"""

    if window == "filtfilt":
        b, a = butter(order, 0.01)

        # Apply the filter to xn.  Use lfilter_zi to choose the initial condition
        # of the filter.
        zi = lfilter_zi(b, a)
        z, _ = lfilter(b, a, x, zi=zi * x[0])

        # Apply the filter again, to have a result filtered at an order
        # the same as filtfilt.
        z2, _ = lfilter(b, a, z, zi=zi * z[0])

        # Use filtfilt to apply the filter.
        y = filtfilt(b, a, x)
        return y
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':
        # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def filter_video_by_scores(vid_path, out_dir, flat_scores, required_time):
    vid_path = str(vid_path)
    out_fname = Path(vid_path).stem + f"_best_{required_time}s.mp4"
    out_path = Path(out_dir) / out_fname
    #
    flat_scores = smooth(flat_scores, window_len=200, window="filtfilt")
    #
    fps = cv2.VideoCapture(str(vid_path)).get(cv2.CAP_PROP_FPS)
    reader = video_reader(vid_path)
    frame = next(reader)
    out_stream = cv2.VideoWriter(filename=str(out_path), apiPreference=cv2.CAP_FFMPEG,
                                 fourcc=cv2.VideoWriter_fourcc(*"MP4V"),
                                 fps=fps, frameSize=tuple(frame.shape[:2][::-1]))
    filtered_indices, filter_pc = select_best_frames(flat_scores=flat_scores, fps=fps,
                                                     max_len_secs=required_time)
    if flat_scores[0] >= filter_pc:
        out_stream.write(frame)

    frame_cntr = 1
    for idx in sorted(filtered_indices[1:]):
        frame = None
        while frame_cntr <= idx:
            frame = next(reader)
            frame_cntr += 1
        out_stream.write(frame)
    out_stream.release()
