import os
from pathlib import Path


def change_cwd_for_jupyter():
    os.chdir(Path(os.getcwd()).parent)
