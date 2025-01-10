from time import sleep

import utils.numba_cache

import os
import imp
import sys
import cv2
import torch
import torchaudio
import numpy as np
import subprocess as proc
import configs.config as cfg
from utils.files import load_json
# from utils.gradcam import get_gradcam
from utils.resize_image import resize_image
from utils.analyse_audio import get_cochleagram
from utils.mfcc_image import spectrogram_image, gray_inferno
from utils.my_voice_analysis import mysp_resample_and_process


def load_model(name, network_name, use_gpu, path_model, path_settings, path_checkpoint):
    try:
        device = "cpu"
        if use_gpu and torch.cuda.is_available():
            x = torch.tensor([0.0], device="cuda")

            if "cuda" in x.device.type:
                device = "cuda"
    except:
        device = "cpu"

    print("Use device for network: '%s'" % device)
    sys.stdout.flush()

    if not getattr(sys, "frozen", False):
        py_path = "%s/__init__.py" % path_model
        pyc_path = "%s/__init__.pyc" % path_model

        py_path = pyc_path if os.path.exists(pyc_path) else py_path
        print("Loading network: %s" % py_path)
        sys.stdout.flush()
        module_model = imp.load_package(f"_{name}", py_path)
    else:
        import model_voice as module_model

    settings = load_json(path_settings)

    return getattr(module_model, network_name)(
        path_weights=path_checkpoint,
        device=device,
        **settings
    ), settings


def calc_svi(AVE, SHI, Vox):
    return 1.425 - float(AVE) * 0.5 + float(SHI) * 0.0625 - float(Vox) * 0.0925


def calc_m(AVE, F0, Prob0, Prob1, Prob2):
    return 180.518 + 295.34 * float(AVE) - 0.3 * float(F0) - 1.876 * (float(Prob2) * 100) - 1.72 * (
                float(Prob1) * 100) - 0.336 * (float(Prob0) * 100)


# kls_count = cfg.MODEL.DEEPSPEECH.OUTPUT_CLASS - 1

class_model, path_model_class, path_settings_class, path_checkpoint_class, path_model_index, path_settings_index, path_checkpoint_index, dirPath, opath = sys.argv[
                                                                                                                                                          1:]
# fpath = os.path.abspath(fpath.replace("\"", ""))
opath = os.path.abspath(opath.replace("\"", ""))
sys.argv = [sys.argv[0]]

opath_csv = "%s.csv" % opath

if not os.path.exists(os.path.dirname(opath_csv)):
    os.mkdir(os.path.dirname(opath_csv))

file_exists = os.path.exists(opath_csv)
if file_exists:
    os.remove(opath_csv)

arr = os.listdir(dirPath)
for index in arr:
    fpath = dirPath + index + "/oz.wav"
    sample_rate = 8000
    res = proc.run("ffmpeg -i \"%s\" -y -acodec pcm_u8 -ar 44000 -ac 1 .tmp.wav" % fpath, shell=True)
    res.check_returncode()

    n_samp = 80

    wav, in_sample_rate = torchaudio.load("./.tmp.wav", normalize=True)
    wav = torchaudio.transforms.Resample(orig_freq=in_sample_rate, new_freq=sample_rate)(wav)
    # wav = torch.mean(wav, axis=0, keepdims=True)
    mfcc = torchaudio.transforms.MFCC(sample_rate, n_samp)(wav)

    _, w = wav.shape
    n_coch, im_size = 200, 1000
    extra_zeros = n_samp - w % n_samp
    _wav = np.zeros([1, w + extra_zeros])
    _wav[0, 0:w] = wav

    cgram = get_cochleagram(_wav, sample_rate, n_samp, downsample=n_coch, nonlinearity="power")
    cgram = gray_inferno(resize_image(cgram / cgram.max(), im_size))
    # image = gray_inferno(resize_image(mfcc[0].cpu().numpy(), im_size))

    # cv2.imwrite("./mels.png", image)
    cv2.imwrite("%s/%s-cochs.png" % (os.path.dirname(opath_csv), os.path.splitext(os.path.basename(fpath))[0]), cgram)

    network_class, class_settings = load_model("network_class", class_model, True, path_model_class,
                                               path_settings_class, path_checkpoint_class)
    kls_count = class_settings["class_count"]
    t_pr = network_class.predict(mfcc).detach().cpu()[0]
    t_kls = torch.softmax(t_pr[..., 0:kls_count], -1).numpy()
    del network_class

    if kls_count == 1:  # special binary case
        kls_count = 2
        t_kls = torch.tensor([1 - t_kls[0], t_kls[0]])

    network_index, _ = load_model("network_index", "StrippedNetworkIndex", True, path_model_index, path_settings_index,
                                  path_checkpoint_index)
    t_pr = network_index.predict(mfcc).detach().cpu()[0]
    t_idx = (t_pr * torch.tensor([60, 50])).numpy()
    del network_index

    mysp_res = mysp_resample_and_process(".tmp.wav")

    if "error" in mysp_res:
        mysp_res = {}

    print("mysp:", str(mysp_res))
    metric_headers = list(mysp_res.keys())
    metric_values = [str(mysp_res[k]) for k in metric_headers]
    ave_value = str(mysp_res["AVE"]) if "AVE" in mysp_res else "-"
    svi_value = "-" if ave_value == "-" else "%.2f" % calc_svi(ave_value, t_idx[0], t_idx[1])
    f0_value = mysp_res["f0_mean"] if "f0_mean" in mysp_res else "-"
    m_value = "-" if svi_value == "-" or f0_value == "-" or kls_count < 3 else "%.2f" % calc_m(ave_value, f0_value,
                                                                                               *t_kls[0:3])

    print("Audio belongs to group: '%i'" % t_kls.argmax())
    print("\nConfidence breakdown:")
    for gr, conf in enumerate(t_kls):
        print("\tGroup '%i' confidece: %-6.2f %%" % (gr, conf * 100))

    print("\nAudio metrics:")
    print("\t%-12s %-12s" % ("SHI-LT", t_idx[0]))
    print("\t%-12s %-12s" % ("IINFVo-LT", t_idx[1]))
    for h, rts in zip(metric_headers, metric_values):
        print("\t%-12s %-12s" % (h, rts))
    print("\t%-12s %-12s" % ("SVI", svi_value))

    needs_header = not os.path.exists(opath_csv)

    with open(opath_csv, "a", encoding="utf-8") as f:
        if needs_header:
            f.write("Patient no.,Class,")
            f.write(",".join(["Probability %i" % i for i in range(kls_count)]))
            f.write(",SHI-LT,IINFVo-LT,")
            f.write(",".join(metric_headers))
            f.write(",SVI,M")

        f.write("\n")
        f.write(index)
        f.write(",")
        f.write(str(t_kls.argmax()))
        f.write(",")
        f.write(",".join(["%.2f" % (prob * 100) for prob in t_kls]))
        f.write(",")
        f.write(",".join(["%.2f" % v for v in t_idx]))
        f.write(",")
        f.write(",".join(metric_values))
        f.write(",")
        f.write(svi_value)
        f.write(",")
        f.write(m_value)
    sleep(0.5)
    os.remove(".tmp.wav")
