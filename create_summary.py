from argparse import ArgumentParser
from pathlib import Path

from utils.summary_utils import filter_video_by_scores, load_flat_scores
from run_va import run


def parse_args() -> dict:
    parser = ArgumentParser()
    # Путь к исходному видео
    parser.add_argument("-in", action="store", help="Путь к исходному видео")
    # Папка, куда сохранить результат и отчет по оценке
    parser.add_argument("-out", action="store",
                        help="Папка, куда сохранить результат и отчет по оценке")
    # Длина summary в секундах
    parser.add_argument("-t", action="store", type=int, help="Длина summary в секундах")
    # [Optional] Количество видеопамяти в Мб, которое попытаются занять фреймворки
    parser.add_argument("--gpu_mem", action="store", type=int, default=-1,
                        help="Количество видеопамяти в Мб, которое попытаются занять фреймворки")
    # [Optional] Порог вероятности для переходов
    parser.add_argument("--sbd_threshold", action="store", type=float, default=0.5,
                        help="Порог вероятности для переходов")
    return vars(parser.parse_args())


def main():
    # Assert без сообщения
    args = parse_args()
    inp_path = Path(args["in"])
    assert inp_path.exists(), f"Video '{str(inp_path)}' is inaccessible"
    out_path = Path(args["out"])
    assert out_path.is_dir(), f"Output path must be a directory, got {str(out_path)}"
    duration = args["t"]
    assert isinstance(duration, int)
    gpu_mem = args["gpu_mem"]
    assert isinstance(gpu_mem, int)
    sbd_threshold = args["sbd_threshold"]
    assert isinstance(sbd_threshold, float)
    #
    print("Running the assessment")
    run(vid_path=inp_path, out_path=out_path,
        sbd_threshold=sbd_threshold, gpu_mem=gpu_mem)
    print("Assessment is done")
    #
    print("Summarization")
    report_path = out_path / "result_dict.pkl"
    scores = load_flat_scores(report_path)
    filter_video_by_scores(vid_path=inp_path, out_dir=out_path,
                           flat_scores=scores, required_time=duration)


if __name__ == '__main__':
    main()
