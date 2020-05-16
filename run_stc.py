from argparse import ArgumentParser
from pathlib import Path
import pickle

import cv2
from fastai.basic_train import load_learner
from fastai.vision import Image, np2model_tensor
from fastai import torch_core
import numpy as np
from PIL import Image as PIL_Image

STC_SHAPE = (375, 666)
SHOT_TYPES = ("CU", "ECU", "EWS", "LS", "MS", "MCU")


def get_shot_gen(reader, n_frames, shape, max_size: int):
    while n_frames > 0:
        real_size = min(n_frames, max_size)
        res = np.empty((real_size, *shape))
        for idx, frame in enumerate(reader):
            res[idx, ...] = frame
            if idx == real_size - 1:
                break
        yield res.astype(np.uint8)
        n_frames -= real_size


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


def _create_shot_type_predictor(stc_predictor, target_shape: tuple):
    coef = 1 / 255

    def predict_shot_type(frame: np.ndarray):
        pil_img = PIL_Image.fromarray(frame)
        # Приводим к форме модели STC для лучшего результата
        if frame.shape[:-1] != STC_SHAPE:
            pil_img = pil_img.resize(target_shape)
        # Нормируем
        img_data = np.array(pil_img).astype(np.float32) * coef
        # fastai.vision требует каналы первыми
        img_data = np.moveaxis(img_data, -1, 0)
        # Конвертируем np.ndarray -> pytorch.Tensor и заворачиваем в fastai.vision.Image
        img_data = Image(np2model_tensor(img_data))
        pred_arr = stc_predictor.predict(img_data)[2]
        return pred_arr.numpy()

    return predict_shot_type


def predict_stc_probas(frames: np.ndarray, stc_predictor):
    """

    :param stc_predictor:
    :param frames: uint8, [0, 255], [b, w, h, c]
    :return:
    """
    predict = np.vectorize(_create_shot_type_predictor(target_shape=STC_SHAPE,
                                                       stc_predictor=stc_predictor),
                           signature="(n,m,l)->(k)")
    probas = predict(frames)
    return probas


def get_st(shots, vid_path, stc_predictor):
    reader = video_reader(vid_path, shape=STC_SHAPE[::-1])
    all_stc_probas = []
    for idx, (start, end) in enumerate(shots):
        shot_gen = get_shot_gen(reader, n_frames=end - start, shape=(*STC_SHAPE, 3), max_size=500)
        stc_probas = []
        for shot_uint8 in shot_gen:
            stc_probas_batch = predict_stc_probas(frames=shot_uint8, stc_predictor=stc_predictor)
            stc_probas.append(stc_probas_batch)
        stc_probas = np.concatenate(stc_probas, axis=0)
        assert len(stc_probas) == end - start, f"{stc_probas.shape}[0] != {start} - {end}"
        all_stc_probas.append(stc_probas)
    return all_stc_probas


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("vid_path", action="store")
    arg_parser.add_argument("out_dir", action="store")
    arg_parser.add_argument("gpu_mem", action="store", type=int)
    return arg_parser.parse_args()


def main():
    args = get_args()
    gpu_mem = args.gpu_mem
    if gpu_mem < 0:
        torch_core.defaults.device = 'cpu'
    vid_path = args.vid_path
    out_dir = Path(args.out_dir)
    sb_fpath = out_dir / "tmp"
    with open(sb_fpath, "rb") as sb_file:
        shots = pickle.load(sb_file)
    #
    cur_folder = Path(__file__).parent.absolute()
    stc_predictor = load_learner(cur_folder / "stc", "shot-type-classifier.pkl")
    probas = get_st(shots=shots, vid_path=vid_path,
                    stc_predictor=stc_predictor)

    with open(out_dir / "shot_types", "wb") as st_file:
        pickle.dump(probas, st_file)


if __name__ == '__main__':
    main()
