from argparse import ArgumentParser
import json

from tensorflow.keras.models import load_model

from utils.va_utils import *


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--path", required=True, action="store")
    arg_parser.add_argument("--gpu_mem", required=True, action="store", type=int)
    arg_parser.add_argument("--out_dir", required=True, action="store")
    return arg_parser.parse_args()


def main():
    args = get_args()
    vid_path = Path(args.path)
    out_path = Path(args.out_dir)
    gpu_mem = args.gpu_mem
    logging.basicConfig(filename=out_path / "report.log",
                        level=logging.INFO, filemode="a",
                        format="[%(asctime)s]\t%(message)s")
    cur_folder = Path(__file__).parent.absolute()
    with open("config.json") as cf:
        config = json.load(cf)
    fps = cv2.VideoCapture(str(vid_path)).get(cv2.CAP_PROP_FPS)
    logging.info(f"Video FPS: {fps}")
    # IQA
    logging.info("Loading IQA model")
    iqa_model_path = cur_folder / config["iqa_model"]
    is_tid = "tid" in str(iqa_model_path)
    logging.info(f"Is TID: {is_tid}")
    iqa_model = load_model(str(iqa_model_path))
    logging.info("Loading IQA detector")
    iqa_detector_path = config["iqa_detector"]
    iqa_detector = TFLiteModel(iqa_detector_path)
    patch_w_path = config["patch_weights"]
    logging.info("Loading weights")
    with open(patch_w_path, "rb") as wf:
        patch_weights = pickle.load(wf)
    # Faces
    logging.info("Loading face detector")
    cascade_path = str(cur_folder / "detector" / "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    #
    logging.info("Starting assessment")
    result = assess_video(vid_path=vid_path, iqa_model=iqa_model,
                          iqa_detector_wrapper=iqa_detector,
                          iqa_patch_weights=patch_weights,
                          is_tid=is_tid,
                          face_cascade=face_cascade,
                          out_path=out_path,
                          gpu_mem=gpu_mem)
    logging.info(f"Assessment done. Time elapsed: {timedelta(seconds=result['total_time'])}")
    logging.info(f"Frames processed: {result['n_frames']}")
    logging.info(f"Total IQA time: {timedelta(seconds=result['iqa_time'])}")
    logging.info(f"Total detector time: {timedelta(seconds=result['detector_time'])}")
    logging.info(f"Mean processing performance: {result['mean_fps']:0.2f} FPS")
    scores = np.hstack(result["frame_scores_per_shots"])
    mean_score = np.mean(scores)
    logging.info(f"Mean score: {mean_score}:0.3f")
    logging.info("Saving scores")
    with open(out_path / "result_dict.pkl", "wb") as sf:
        pickle.dump(result, sf)
    logging.info("Done")


if __name__ == '__main__':
    main()
