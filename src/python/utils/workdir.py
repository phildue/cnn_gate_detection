import os
from os.path import expanduser

import sys


def cd_work():
    print("Platform: ", sys.platform)
    if sys.platform is ('win32' or 'win64'):
        wd = 'e:/doc/study/thesis/mavv'
    else:
        wd = expanduser('~') + '/dronevision/'

    print("Working Directory: ", wd)

    os.chdir(wd)


def home():
    return expanduser('~')
