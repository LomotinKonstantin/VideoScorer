from argparse import ArgumentParser
from datetime import timedelta
import logging
from pathlib import Path
import subprocess
from time import time


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--path", required=True, action="store")
    arg_parser.add_argument("--gpu_mem", required=True, action="store", type=int)
    arg_parser.add_argument("--out_dir", required=True, action="store")
    arg_parser.add_argument("--sbd_threshold", required=False,
                            type=float, action="store", default=0.5)
    return arg_parser.parse_args()


def run_sbd(vid_path: Path, sbd_threshold: int, gpu_mem: int, out_dir: Path):
    sb_fpath = Path("tmp")
    # Удаляем результаты прошлого запуска
    if sb_fpath.exists():
        sb_fpath.unlink()
    try:
        subprocess.run(["python", "detect_shots.py",
                        str(vid_path),
                        str(out_dir),
                        str(sbd_threshold),
                        str(gpu_mem)], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"detect_shots.py caused an error :{e}\nTerminating.")
        exit(1)


def run_scoring(vid_path: Path, out_path: Path, gpu_mem: str):
    args = [f"--path {str(vid_path)}",
            f"--gpu_mem {str(gpu_mem)}",
            f"--out_dir {str(out_path)}"]
    try:
        command = f"python score.py {args[0]} {args[1]} {args[2]}"
        subprocess.run(command,
                       check=True)
    except subprocess.CalledProcessError as e:
        logging.info(f"score.py caused an error :{e}\nTerminating.")
        exit(1)


def main():
    args = get_args()
    vid_path = Path(args.path)
    out_path = Path(args.out_dir)
    out_path.mkdir(exist_ok=True)
    gpu_mem = args.gpu_mem
    sbd_threshold = args.sbd_threshold
    #
    logging.basicConfig(filename=out_path / "report.log",
                        level=logging.INFO, filemode="w",
                        format="[%(asctime)s]\t%(message)s")
    logger = logging.getLogger("Launcher")
    #
    t = time()
    run_sbd(vid_path=vid_path, sbd_threshold=sbd_threshold,
            gpu_mem=gpu_mem, out_dir=out_path)
    sbd_time = time() - t
    logger.info(f"SBD script has taken {timedelta(seconds=sbd_time)}")
    for hdl in list(logger.handlers):
        logger.removeHandler(hdl)
        hdl.flush()
        hdl.close()
    #
    run_scoring(vid_path=vid_path, out_path=out_path, gpu_mem=gpu_mem)


if __name__ == '__main__':
    main()
