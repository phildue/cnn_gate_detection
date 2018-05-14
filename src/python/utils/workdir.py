import os
from os.path import expanduser

import sys


def cd_work():
    print("Platform: ", sys.platform)
    if sys.platform == 'win32' or sys.platform == 'win64':
        wd = 'e:/doc/study/thesis/mavv'
    else:
        wd = expanduser('~') + '/dronevision/'

    os.chdir(wd)
    print("Working Directory: ", os.getcwd())


def home():
    return expanduser('~')
