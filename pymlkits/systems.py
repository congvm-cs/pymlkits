# systems.py

import os
from glob import glob


def is_existing(fpath):
    if not os.path.isfile(fpath):
        return False
    else:
        return True


def make_dir(fpath):
    os.makedirs(fpath, exist_ok=True)


def load_files_from_dir(directory, pattern='*.*', only_file=True, get_full_path=True):
    dir_to_search = os.path.join(directory, pattern)
    all_paths = glob(dir_to_search, recursive=True)
    if only_file:
        all_paths = [p for p in all_paths if is_existing(p)]
    if get_full_path:
        all_paths = [os.path.abspath(p) for p in all_paths if is_existing(p)]
    return all_paths
