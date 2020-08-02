import sys
sys.path.append('..')

from pymlkits.systems import load_files_from_dir

dir_to_search = './'
all_file_paths = load_files_from_dir(dir_to_search, only_file=True, get_full_path=False)
print(all_file_paths)