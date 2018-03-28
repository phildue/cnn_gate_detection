import os
from os.path import expanduser

import sys


def cd_work():
    if sys.platform is 'win32' or 'win64':
        os.chdir(expanduser('~') + '/Documents/doc/study/thesis/mavv')
    else:
        os.chdir(expanduser('~') + '/dronevision/')


def home():
    return expanduser('~')
