from collections import Counter
from datetime import timedelta
import logging
from pathlib import Path
import pickle
from time import time

import cv2
import numpy as np
from PIL import Image as PIL_Image

from utils.utils import low_pass_filter, crop_img, extract_patches
from utils.video_utils import video_reader
from utils.TFLiteModel import TFLiteModel

cv2.ocl.setUseOpenCL(False)

TID_MAX_MOS = 9.
LIVE_MAX_MOS = 100.
TID_TO_LIVE_MOS_COEF = LIVE_MAX_MOS / TID_MAX_MOS
SHOT_TYPES = ("CU", "ECU", "EWS", "LS", "MS", "MCU")
STC_SHAPE = (375, 666)

GOOD_TRANSITIONS = (
    {"ECU", "MCU"}, {"CU", "MS"},
    {"MCU", "LS"}, {"MS", "EWS"},
    # Исключения
    {"MCU", "CU"}, {"MS", "LS"}
)


def decode_stc_prediction(pred):
    return SHOT_TYPES[pred.argmax()]


# def load_transnet(checkpoint_path: str = "./TransNet/model/transnet_model-F16_L3_S2_D256"):
#     params = TransNetParams()
#     params.CHECKPOINT_PATH = checkpoint_path
#     net = TransNet(params)
#     return net


def _create_predictor(detector_model: TFLiteModel,
                      iqa_model,
                      patch_weights: np.ndarray,
                      patch_size: int, patches_per_side: int,
                      min_square=0.3):
    def predict(frame: np.ndarray):
        bb = detector_model.get_bounding_box(frame, min_square)
        if bb is not None:
            frame = crop_img(frame, bb)
        patches = extract_patches(frame, patch_size, patches_per_side)
        pred_mos_batch = iqa_model.predict_on_batch(low_pass_filter(patches))
        pred_mos_batch = np.ravel(pred_mos_batch)
        assert len(pred_mos_batch.shape) == 1, pred_mos_batch.shape
        weighted_mos = np.average(pred_mos_batch, weights=patch_weights)
        return weighted_mos

    return predict


def iqa_assess_frames(iqa_model, iqa_detector: TFLiteModel,
                      patch_weights: np.ndarray, frames: np.ndarray):
    """
    :param iqa_model:
    :param iqa_detector:
    :param patch_weights:
    :param frames: фреймы подходящего для iqa_model размера
                    float, [0, 1], [b, w, h, c]
    :return:
    """
    assert len(frames), "'frames' must contain at least one image (3d-array)"
    n_patches = int(np.sqrt(len(patch_weights)))
    assert n_patches ** 2 == len(patch_weights), len(patch_weights)
    patch_size = iqa_model.input_shape[1]
    preprocess_func = np.vectorize(_create_predictor(detector_model=iqa_detector,
                                                     iqa_model=iqa_model,
                                                     patch_weights=patch_weights,
                                                     patch_size=patch_size,
                                                     patches_per_side=n_patches),
                                   signature="(n,m,k)->()")
    # На этом моменте молимся
    t = time()
    assert len(frames.shape) == 4, frames.shape
    iqa_scores = preprocess_func(frames)
    assert len(iqa_scores) == len(frames), f"{iqa_scores} iqa scores"
    return iqa_scores, time() - t





def predict_stc_probas(frames: np.ndarray):
    """

    :param frames: uint8, [0, 255], [b, w, h, c]
    :return:
    """
    predict = np.vectorize(_create_shot_type_predictor(STC_SHAPE),
                           signature="(n,m,l)->(k)")
    probas = predict(frames)
    return probas


def _create_face_finder(face_cascade: cv2.CascadeClassifier):

    def find_faces(frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    return find_faces


def faces_bboxes(frames: np.ndarray, face_cascade: cv2.CascadeClassifier) -> list:
    """
    :param frames: uint8, [0, 255], [b, w, h, c]
    :param face_cascade:
    :return:
    """
    all_faces = list(
        map(_create_face_finder(face_cascade), frames)
    )
    return all_faces


def aggregate_scores_for_shot(iqa_scores: np.ndarray,
                              stc_matrix: np.ndarray,
                              face_boxes: list) -> np.ndarray:
    """
    Формула из головы для оценки шота.
    :param iqa_scores:
    :param stc_matrix:
    :param face_boxes:
    :return:
    """
    # Коэффициент понижения качества для каждого типа кадра
    # в порядке ("CU", "ECU", "EWS", "LS", "MS", "MCU")
    # Оценка ближайшего плана должна быть занижена (=== завышенные требования)
    st_weights = np.array(
        [0.6, 0.6, 0., 0., 0.1, 0.5]
    )
    # Сначала считаем итоговые веса планов как скалярное произведение
    # весов на вероятности.
    # Потом умножаем получившийся коэффициент на оценку IQA
    weighted_iqa = np.multiply(stc_matrix.dot(st_weights), iqa_scores)
    # Количество лиц на каждом кадре
    n_faces_for_frames = np.array(list(map(len, face_boxes)))
    # Пока непонятно, что делать с этой информацией
    frame_scores = weighted_iqa + n_faces_for_frames * 0.5
    return frame_scores


def _most_common(flat_list):
    return Counter(flat_list).most_common()[0][0]


def get_start_end_shot_types(stc_matrix: np.ndarray, area: 0.3) -> tuple:
    """
    Извлечь начальный и конечный типы плана из шота.
    :param stc_matrix:
    :param area:
    :return:
    """
    n_frames = len(stc_matrix)
    start_area = int(n_frames * area) + 1
    end_area = int(n_frames * (1 - area))
    start_decoded = list(map(decode_stc_prediction, stc_matrix[:start_area]))
    end_decoded = list(map(decode_stc_prediction, stc_matrix[end_area:]))
    start_st = _most_common(start_decoded)
    end_st = _most_common(end_decoded)
    return start_st, end_st


def count_good_transitions(shot_start_end_types: list) -> int:
    """
    'Многолетним опытом было установлено,
    что наиболее гладко воспринимается стык между планами,
    находящимися на приведенной выше шкале через один.
    То есть, общий план монтируется с 1-м средним и наоборот,
    2-й средний с крупным, и т.д.
    Исключения: крупный план монтируется с деталью, общий план с дальним.'
    :return:
    """
    idx = 1
    n_good_transitions = 0
    while idx < len(shot_start_end_types):
        cur_pair = shot_start_end_types[idx - 1]
        next_pair = shot_start_end_types[idx]
        transition = {cur_pair[1], next_pair[0]}
        if transition in GOOD_TRANSITIONS:
            n_good_transitions += 1
        idx += 1
    return n_good_transitions


def reshape_video(video: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    def _reshape_frame(frame: np.ndarray) -> np.ndarray:
        pil_img = PIL_Image.fromarray(frame)
        # Приводим к форме модели STC для лучшего результата
        pil_img = pil_img.resize((new_w, new_h))
        return np.array(pil_img)

    reshaper = np.vectorize(_reshape_frame, signature="(m,n,k)->(u,v,k)")
    return reshaper(video)


def get_shot(reader, n_frames, shape):
    res = np.empty((n_frames, *shape))
    for idx, frame in enumerate(reader):
        res[idx, ...] = frame
        if idx == n_frames - 1:
            break
    return res.astype(np.uint8)


def assess_video(vid_path: Path,
                 out_path: Path,
                 iqa_model,
                 iqa_detector_wrapper: TFLiteModel,
                 iqa_patch_weights: np.ndarray,
                 is_tid: bool,
                 face_cascade: cv2.CascadeClassifier) -> dict:
    """
    :param out_path:
    :param face_cascade:
    :param is_tid:
    :param iqa_patch_weights: массив весов для патчей
    :param iqa_detector_wrapper: объект TFLiteModel
    :param vid_path: путь к видеофайлу
    :param iqa_model: подразумевается, что используется модель с детектором и патчами
    :return:
    """
    logger = logging.getLogger("assess_video")
    total_time = time()
    logger.info("Loading video")
    logger.info("Loading shot boundaries")
    sb_fpath = out_path / "tmp"
    with open(str(sb_fpath), "rb") as sb_file:
        shots = pickle.load(sb_file)
    pix_coef = 1 / 255
    shot_start_end_types = []
    video_scores = []
    total_iqa_time = 0
    total_stc_time = 0
    total_detector_time = 0
    reader = video_reader(vid_path, shape=STC_SHAPE[::-1])
    for idx, (start, end) in enumerate(shots):
        logger.info(f"Processing shot {idx + 1} / {len(shots)}")
        # Собираем шот
        shot_uint8 = get_shot(reader, n_frames=end - start, shape=(*STC_SHAPE, 3))
        # IQA
        # Для IQA каждый шот надо привести к интервалу [0, 1]
        # Для скорости не делим на 255, а умножаем на 1/255
        logger.info("\tPredicting IQA")
        iqa_shot_arr = shot_uint8.astype(np.float32) * pix_coef
        iqa_scores, iqa_time = iqa_assess_frames(frames=iqa_shot_arr, iqa_model=iqa_model,
                                                 iqa_detector=iqa_detector_wrapper,
                                                 patch_weights=iqa_patch_weights)
        total_iqa_time += iqa_time
        logger.info(f"\tDone ({timedelta(seconds=iqa_time)})")
        # Если модель TID*, переводим MOS в шкалу [0, 100]
        if is_tid:
            iqa_scores *= TID_TO_LIVE_MOS_COEF

        # STC
        logger.info("\tPredicting shot types")
        t = time()
        stc_probas = predict_stc_probas(frames=shot_uint8)
        stc_time = time() - t
        total_stc_time += stc_time
        logger.info(f"\tDone ({timedelta(seconds=stc_time)})")

        # Faces
        logger.info("\tDetecting faces")
        t = time()
        shot_faces = faces_bboxes(frames=shot_uint8, face_cascade=face_cascade)
        detector_time = time() - t
        total_detector_time += detector_time
        logger.info(f"\tDone ({timedelta(seconds=detector_time)})")

        # Собираем оценки
        logger.info("\tAggregating scores")
        shot_scores = aggregate_scores_for_shot(iqa_scores=iqa_scores,
                                                stc_matrix=stc_probas,
                                                face_boxes=shot_faces)
        video_scores.append(shot_scores)
        shot_types = get_start_end_shot_types(stc_probas, area=0.3)
        shot_start_end_types.append(shot_types)

    # Теперь надо оценить переходы
    logger.info("Assessing transitions")
    n_good_transition = count_good_transitions(shot_start_end_types)
    # Доля хороших переходов
    n_transitions = len(shot_start_end_types) - 1
    if n_transitions > 0:
        gt_percent = n_good_transition / n_transitions
    else:
        gt_percent = 1
    total_time = time() - total_time
    logger.info(f"Good transitions: {n_good_transition} ({gt_percent * 100:.2f}%)")
    n_frames = len(np.hstack(video_scores))
    result = {
        "frame_scores_per_shots": video_scores,
        "n_shots": len(shots),
        "n_good_transitions": n_good_transition,
        "gt_percent": gt_percent,
        "iqa_time": total_iqa_time,
        # "sbd_time": total_sbd_time,
        "stc_time": total_stc_time,
        "detector_time": total_detector_time,
        "total_time": total_time,
        "mean_fps": n_frames / total_time,
        "n_frames": n_frames,
    }
    return result
