import os
from os.path import expanduser


def work_dir():
    os.chdir(expanduser('~') + '/dronevision/')
