import os as _os
from easydict import EasyDict as _EasyDict

__BASE_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), ".."))

DIRS = _EasyDict({
    "BASE_DIR": __BASE_DIR,
    "INPUTS": "/media/ratchet/hdd/Dataset/voice_analysis_2",
    "OUTPUT_MODELS": "%s/output_models_old" % __BASE_DIR,
})

TRAINING = _EasyDict({
    "BATCH_SIZE": 4,
    "TRAINGING_SET_RATIO": 0.8,
    "LEARNING_RATE_INITIAL": 1e-4,
    "EPOCHS": 50
})

MODEL = _EasyDict({
    "SAMPLING_RATE": 8000,
    "DEEPSPEECH": {
        "HIDDEN_NODES": 1024,
        "OUTPUT_CLASS": 4,
        # "OUTPUT_CLASS": 3,
        "DROPOUT_RATE": 0,
        "OUTPUT_SIZE": 16
    },
})