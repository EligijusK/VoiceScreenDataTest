import json as _json
from easydict import EasyDict as _EasyDict
import os as _os
import shutil as _shutil


def save_json(path, data):
    with open(path, "w") as outfile:
        _json.dump(data, outfile, indent=4)


def load_json(path):
    with open(path, "r") as outfile:
        return _EasyDict(_json.load(outfile))


def copy_file(in_file, out_file, override=True):
    if not _os.path.exists(in_file):
        raise Exception("File \"%s\" does not exist!" % in_file)

    if _os.path.exists(out_file):
        if override:
            _os.remove(out_file)
        else:
            raise Exception("File \"%s\" alread exists!" % out_file)

    _shutil.copy(in_file, out_file)
