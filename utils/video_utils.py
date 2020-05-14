#!/usr/bin/env python
# coding: utf-8
from pytube import YouTube
from urllib.error import HTTPError
from pathlib import Path
import cv2
import tensorflow as tf
from utils.utils import low_pass_filter, extract_patches
from IPython.display import clear_output
import numpy as np


def download_video(link: str, save_dir: Path, filename: str, quality_list: list, check_exist=False):
    vid = YouTube(link)
    for res in quality_list:
        full_name = f"{filename}_{res}"
        if check_exist:
            path = save_dir / f"{full_name}.mp4"
            if path.exists():
                print(f"{full_name} already exists, skipping")
                continue
        streams = vid.streams.filter(res=res, file_extension="mp4", fps=30)
        if not streams:
            print(f"No '{res}' mp4 stream available with 30 fps for {filename}")
        try:
            streams[0].download(output_path=str(save_dir), filename=full_name)
        except HTTPError as e:
            print(f"Failed to download {filename} with {res}: {e} (Pytube3 bug)")


def video_reader(path, batch_size=1, shape=None):
    vc = cv2.VideoCapture(str(path))
    success, frame = vc.read()
    if success and shape is not None:
        frame = tf.image.resize(frame, shape[::-1]).numpy()
    batch = []
    while success:
        batch.append(frame)
        if len(batch) == batch_size:
            batch = np.array(batch)
            if batch_size == 1:
                batch = batch[0]
            yield batch
            batch = []
        success, frame = vc.read()
        if success and shape is not None:
            frame = tf.image.resize(frame, shape[::-1]).numpy()
    if len(batch) > 0:
        batch = np.array(batch)
        if batch_size == 1:
            batch = batch[0]
        yield batch


def count_frames(path):
    return sum([1 for _ in video_reader(path)]) - 1


def assess_file(model, video_path: Path, n_patches, mos_weights, batch_size=1000) -> np.ndarray:
    assert video_path.exists(), "Video path does not exist"
    assert batch_size > 0, "Batch size must be a positive integer"
    frame_quality = []
    vr = video_reader(video_path, batch_size=batch_size)
    for cntr, batch in enumerate(vr):
        if batch_size == 1:
            batch = [batch]
        scores = assess_frames(model, batch,
                               n_patches=n_patches, mos_weights=mos_weights)
        frame_quality.append(scores)
    return np.array(frame_quality).ravel()


def assess_frames(model, frames: list, n_patches: int, mos_weights) -> np.ndarray:
    assert len(frames), "'frames' must contain at least one image (3d-array)"
    ds = tf.data.Dataset.from_tensor_slices(frames)
    shape = model.input_shape[1:3]
    frame_quality = []
    if n_patches <= 0:
        prepared_ds = ds.map(lambda img: low_pass_filter(tf.image.resize([img], shape)))
        frame_quality = model.predict(prepared_ds).ravel()
    else:
        assert shape[0] == shape[1]
        prepared_ds = ds.map(lambda img: low_pass_filter(extract_patches(img, patch_size=shape[0],
                                                                         patches_per_side=n_patches)))
        for patch_set in prepared_ds:
            pred_mos_batch = model.predict_on_batch(patch_set)
            pred_mos_batch = tf.reshape(pred_mos_batch, [-1])  # Flatten in tf way
            assert len(mos_weights) == len(pred_mos_batch), \
                f"{mos_weights}, {pred_mos_batch}"
            res_img_mos = np.average(pred_mos_batch.numpy(), weights=mos_weights)
            frame_quality.append(res_img_mos)
        frame_quality = np.array(frame_quality)
    assert len(frame_quality), f"{len(frames)} frames"
    return frame_quality


def filter_video(model, inp_path: str, out_path: str, threshold: float, fps=30, size=(1280, 720)):
    assert Path(inp_path).exists()
#     assert 0 < threshold < 1
    frame_reader = video_reader(inp_path)
    out_stream = cv2.VideoWriter(filename=str(out_path), apiPreference=cv2.CAP_FFMPEG, 
                                 fourcc=cv2.VideoWriter_fourcc(*"MP4V"), 
                                 fps=fps, frameSize=tuple(size))
    for cntr, frame in enumerate(frame_reader):
        print(f"Processing frame {cntr}")
        img_data = low_pass_filter(tf.image.resize([frame], model.input_shape[1:3]))
        score = model.predict(img_data)[0][0]
        clear_output(True)
        if score < threshold:
            continue
        out_stream.write(frame)
        if cntr >= 1000:
            break
    out_stream.release()


def download_youtube_videos(save_dir: Path, json_settings: dict, check_exist=False):
    assert save_dir.exists(), "Save dir does not exist"
    for title, source in json_settings.items():
        res_list = source["quality"]
        link = source["link"]
        print(f"Downloading {title}")
        download_video(link=link, 
                       save_dir=save_dir, 
                       filename=title, 
                       quality_list=res_list, 
                       check_exist=check_exist)


def drop_bad_frames(inp_path: Path, out_path: Path,
                    frame_scores: np.ndarray, threshold: float):
    assert inp_path.exists(), "Input video file not found at this location"
    vr = video_reader(inp_path)
    frame = next(vr)
    out_stream = cv2.VideoWriter(filename=str(out_path), apiPreference=cv2.CAP_FFMPEG,
                                 fourcc=cv2.VideoWriter_fourcc(*"MP4V"),
                                 fps=30, frameSize=tuple(frame.shape[:2][::-1]))
    if frame_scores[0] >= threshold:
        out_stream.write(frame)
    for idx, frame in enumerate(vr, 1):
        if frame_scores[idx] < threshold:
            continue
        out_stream.write(frame)
    out_stream.release()


# Почитать про оценку суммаризаторов видео
# Подготовить датасет
# Почитать про неудачные кадры
