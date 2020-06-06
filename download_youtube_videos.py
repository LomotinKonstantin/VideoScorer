from argparse import ArgumentParser
import json
from pathlib import Path

from utils.gpu_safe import download_youtube_videos


if __name__ == '__main__':
    """
    Скачать видео, указанные в youtube_ds.json
    и поместить в dest_folder
    """
    # Parsing args, initialising dest folder
    path_arg_name = "dest_folder"
    argparser = ArgumentParser()
    argparser.add_argument(path_arg_name, action="store")
    args = argparser.parse_args()
    save_dir = Path(vars(args)[path_arg_name])
    save_dir.mkdir(exist_ok=True)
    # Loading dataset description
    dataset_dict = json.load(open("youtube_ds.json"))
    # Downloading
    download_youtube_videos(save_dir=save_dir, json_settings=dataset_dict, check_exist=True)
