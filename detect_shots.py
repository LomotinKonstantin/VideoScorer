from argparse import ArgumentParser
from pathlib import Path
import pickle

import ffmpeg
import numpy as np
from TransNet.transnet import TransNet, TransNetParams
from TransNet.transnet_utils import scenes_from_predictions
import tensorflow as tf
from tensorflow_core.config.experimental import set_virtual_device_configuration, \
    VirtualDeviceConfiguration


def get_shot_boundaries(net: TransNet, video: np.ndarray, threshold: float):
    # resized_video = reshape_video(video, net.params.INPUT_WIDTH, net.params.INPUT_HEIGHT)
    predicts = net.predict_video(video)
    # Порог 0.5 - оптимально
    scenes = scenes_from_predictions(predicts, threshold)
    return scenes


def read_to_array(video_path: Path, width: int or None, height: int or None):
    video_path = str(video_path)
    if width is not None and height is not None:
        video_stream, err = (
            ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='rgb24',
                                            s=f'{width}x{height}').run(capture_stdout=True)
        )
        video = np.frombuffer(video_stream, np.uint8).reshape([-1, height, width, 3])
    else:
        video_stream, err = (
            ffmpeg.input(video_path).output('pipe:', format='rawvideo',
                                            pix_fmt='rgb24').run(capture_stdout=True)
        )
        video = np.frombuffer(video_stream, np.uint8)
    return video


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("vid_path", action="store")
    arg_parser.add_argument("out_dir", action="store")
    arg_parser.add_argument("sbd_threshold", action="store", type=float)
    arg_parser.add_argument("gpu_mem", action="store", type=int)
    return arg_parser.parse_args()


def main():
    args = get_args()
    gpu_mem = args.gpu_mem
    if gpu_mem > 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                set_virtual_device_configuration(gpus[0],
                                                 [VirtualDeviceConfiguration(memory_limit=gpu_mem)])
            except Exception as e:
                print(e)
                exit(1)
    else:
        tf.config.experimental.set_visible_devices([], 'GPU')
    vid_path = args.vid_path
    out_dir = Path(args.out_dir)
    sbd_threshold = args.sbd_threshold
    params = TransNetParams()
    cur_folder = Path(__file__).parent.absolute()
    transnet_cp_path = cur_folder / "TransNet/model/transnet_model-F16_L3_S2_D256"
    params.CHECKPOINT_PATH = str(transnet_cp_path.absolute())
    net = TransNet(params)
    video_arr_255_uint8 = read_to_array(video_path=vid_path,
                                        width=net.params.INPUT_WIDTH,
                                        height=net.params.INPUT_HEIGHT)
    sb = get_shot_boundaries(net=net, video=video_arr_255_uint8, threshold=sbd_threshold)
    with open(out_dir / "tmp", "wb") as sb_file:
        pickle.dump(sb, sb_file)


if __name__ == '__main__':
    main()
