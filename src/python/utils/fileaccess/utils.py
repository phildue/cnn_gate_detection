import os
import pickle
from pprint import pprint


def load_file(filename: str):
    if filename.endswith('.txt'):
        with open(filename, 'r') as f:
            return f.read()
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)


def save_file(file, name: str, path: str = './', verbose=True):
    if not os.path.exists(path):
        os.makedirs(path)

    if name.endswith('.pkl'):
        with open(path + name, 'wb') as f:
            pickle.dump(file, f)
    elif name.endswith('.txt'):
        with open(path + name, 'w') as f:
            pprint(file, stream=f)
    else:
        name += '.pkl'
        with open(path + name, 'wb') as f:
            pickle.dump(file, f)

    if verbose:
        print("File stored at: " + path + name)


def create_dirs(dir_names: [str]):
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
