from pathlib import Path
from urllib.error import HTTPError

import cv2
import numpy as np
from pytube import YouTube


def count_frames(path):
    return sum([1 for _ in video_reader(path)]) - 1


def video_reader(path, batch_size=1, shape=None):
    vc = cv2.VideoCapture(str(path))
    success, frame = vc.read()
    if success and shape is not None:
        frame = cv2.resize(frame, shape)
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
            # frame = tf.image.resize(frame, shape[::-1]).numpy()
            frame = cv2.resize(frame, shape)
    if len(batch) > 0:
        batch = np.array(batch)
        if batch_size == 1:
            batch = batch[0]
        yield batch


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
