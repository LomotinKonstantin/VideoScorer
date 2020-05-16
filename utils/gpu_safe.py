import cv2
import numpy as np


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
