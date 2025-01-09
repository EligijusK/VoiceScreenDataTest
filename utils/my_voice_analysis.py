import sys
import traceback
import parselmouth
from parselmouth.praat import call, run_file
import glob
import pandas as pd
import numpy as np
import scipy
from scipy.stats import binom
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind
import os
import torch
import torchaudio
from module_diva import evaluate_audio

sourcerun = "utils/myspsolution.praat"
headers = ["gender","mood","p-value/sample","syllables","pauses","speech_rate","articulation_rate","duration_speaking","duration_total","balance","f0_mean","f0_SD","f0_MD","f0_min","f0_max","f0_quan25","f0_quan75","pronunciation_posteriori"]

def _myspsyl(objects):
    # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    # print(objects[0])
    # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
    z1 = str(objects[1])
    z2 = z1.strip().split()
    z3 = int(z2[0])  # will be the integer number 10
    z4 = float(z2[3])  # will be the floating point number 8.3
    # print("number_ of_syllables=", z3)

    return {"syllables": z3}


def _mysppaus(objects):
    # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    # print(objects[0])
    # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
    z1 = str(objects[1])
    z2 = z1.strip().split()
    z3 = int(z2[1])  # will be the integer number 10
    z4 = float(z2[3])  # will be the floating point number 8.3
    # print("number_of_pauses=", z3)

    return {"pauses": z3}


def _myspsr(objects):
    # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    # print(objects[0])
    # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
    z1 = str(objects[1])
    z2 = z1.strip().split()
    z3 = int(z2[2])  # will be the integer number 10
    z4 = float(z2[3])  # will be the floating point number 8.3
    # print("rate_of_speech=", z3, "# syllables/sec original duration")

    return {"speech_rate": z3}


def _myspatc(objects):
    # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    # print(objects[0])
    # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
    z1 = str(objects[1])
    z2 = z1.strip().split()
    z3 = int(z2[3])  # will be the integer number 10
    z4 = float(z2[3])  # will be the floating point number 8.3
    # print("articultion_rate=", z3, "# syllables/sec speaking duration")

    return {"articulation_rate": z3}


def _myspst(objects):
    # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    # print(objects[0])
    # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
    z1 = str(objects[1])
    z2 = z1.strip().split()
    z3 = int(z2[3])  # will be the integer number 10
    z4 = float(z2[4])  # will be the floating point number 8.3
    # print("speaking_duration=", z4,
    #       "# sec only speaking duration without pauses")

    return {"duration_speaking": z4}


def _myspod(objects):
    # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    # print(objects[0])
    # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
    z1 = str(objects[1])
    z2 = z1.strip().split()
    z3 = int(z2[3])  # will be the integer number 10
    z4 = float(z2[5])  # will be the floating point number 8.3
    # print("original_duration=", z4,
    #       "# sec total speaking duration with pauses")

    return {"duration_total": z4}


def _myspbala(objects):
    # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    # print(objects[0])
    # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
    z1 = str(objects[1])
    z2 = z1.strip().split()
    z3 = int(z2[3])  # will be the integer number 10
    z4 = float(z2[6])  # will be the floating point number 8.3
    # print("balance=", z4, "# ratio (speaking duration)/(original duration)")

    return {"balance": z4}

def _myspf0mean(objects):
    # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    # print(objects[0])
    # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
    z1 = str(objects[1])
    z2 = z1.strip().split()
    z3 = int(z2[3])  # will be the integer number 10
    z4 = float(z2[7])  # will be the floating point number 8.3
    # print("f0_mean=", z4, "# Hz global mean of fundamental frequency distribution")

    return {"f0_mean": z4}

def _myspf0sd(objects):
    # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    # print(objects[0])
    # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
    z1 = str(objects[1])
    z2 = z1.strip().split()
    z3 = int(z2[3])  # will be the integer number 10
    z4 = float(z2[8])  # will be the floating point number 8.3
    # print("f0_SD=", z4,
    #       "# Hz global standard deviation of fundamental frequency distribution")

    return {"f0_SD": z4}

def _myspf0med(objects):
    # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    # print(objects[0])
    # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
    z1 = str(objects[1])
    z2 = z1.strip().split()
    z3 = int(z2[3])  # will be the integer number 10
    z4 = float(z2[9])  # will be the floating point number 8.3
    # print("f0_MD=", z4, "# Hz global median of fundamental frequency distribution")

    return {"f0_MD": z4}

def _myspf0min(objects):
    # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    # print(objects[0])
    # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
    z1 = str(objects[1])
    z2 = z1.strip().split()
    z3 = int(z2[10])  # will be the integer number 10
    z4 = float(z2[10])  # will be the floating point number 8.3
    # print("f0_min=", z3, "# Hz global minimum of fundamental frequency distribution")

    return {"f0_min": z3}


def _myspf0max(objects):
    # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    # print(objects[0])
    # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
    z1 = str(objects[1])
    z2 = z1.strip().split()
    z3 = int(z2[11])  # will be the integer number 10
    z4 = float(z2[11])  # will be the floating point number 8.3
    # print("f0_max=", z3, "# Hz global maximum of fundamental frequency distribution")

    return {"f0_max": z3}


def _myspf0q25(objects):
    # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    # print(objects[0])
    # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
    z1 = str(objects[1])
    z2 = z1.strip().split()
    z3 = int(z2[12])  # will be the integer number 10
    z4 = float(z2[11])  # will be the floating point number 8.3
    # print("f0_quan25=", z3,
    #       "# Hz global 25th quantile of fundamental frequency distribution")

    return {"f0_quan25": z3}


def _myspf0q75(objects):
    # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    # print(objects[0])
    # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
    z1 = str(objects[1])
    z2 = z1.strip().split()
    z3 = int(z2[13])  # will be the integer number 10
    z4 = float(z2[11])  # will be the floating point number 8.3
    # print("f0_quan75=", z3,
    #       "# Hz global 75th quantile of fundamental frequency distribution")

    return {"f0_quan75": z3}


def _mysptotal(objects):
    # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    # print(objects[0])
    # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
    z1 = str(objects[1])
    z2 = z1.strip().split()
    z3 = np.array(z2)
    z4 = np.array(z3)[np.newaxis]
    z5 = z4.T
    return {"number_ of_syllables": z5[0, :], "number_of_pauses": z5[1, :], "rate_of_speech": z5[2, :], "articulation_rate": z5[3, :], "speaking_duration": z5[4, :],
            "original_duration": z5[5, :], "balance": z5[6, :], "f0_mean": z5[7, :], "f0_std": z5[8, :], "f0_median": z5[9, :], "f0_min": z5[10, :], "f0_max": z5[11, :],
            "f0_quantile25": z5[12, :], "f0_quan75": z5[13, :]
            }


def _mysppron(objects):
    # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    # print(objects[0])
    # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
    z1 = str(objects[1])
    z2 = z1.strip().split()
    z3 = int(z2[13])  # will be the integer number 10
    z4 = float(z2[14])  # will be the floating point number 8.3
    db = binom.rvs(n=10, p=z4, size=10000)
    a = np.array(db)
    b = np.mean(a)*100/10

    return {"pronunciation_posteriori": b}

    # print("Pronunciation_posteriori_probability_score_percentage= :%.2f" % (b))


def _myspgend(objects):
    # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    # print(objects[0])
    # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
    z1 = str(objects[1])
    z2 = z1.strip().split()
    z3 = float(z2[8])  # will be the integer number 10
    z4 = float(z2[7])  # will be the floating point number 8.3
    if z4 <= 114:
        g = 101
        j = 3.4
    elif z4 > 114 and z4 <= 135:
        g = 128
        j = 4.35
    elif z4 > 135 and z4 <= 163:
        g = 142
        j = 4.85
    elif z4 > 163 and z4 <= 197:
        g = 182
        j = 2.7
    elif z4 > 197 and z4 <= 226:
        g = 213
        j = 4.5
    elif z4 > 226:
        g = 239
        j = 5.3
    else:
        return {"error": True}

    def teset(a, b, c, d):
        d1 = np.random.wald(a, 1, 1000)
        d2 = np.random.wald(b, 1, 1000)
        d3 = ks_2samp(d1, d2)
        c1 = np.random.normal(a, c, 1000)
        c2 = np.random.normal(b, d, 1000)
        c3 = ttest_ind(c1, c2)
        y = ([d3[0], d3[1], abs(c3[0]), c3[1]])
        return y
    nn = 0
    mm = teset(g, j, z4, z3)
    while (mm[3] > 0.05 and mm[0] > 0.04 or nn < 5):
        mm = teset(g, j, z4, z3)
        nn = nn+1
    nnn = nn
    if mm[3] <= 0.09:
        mmm = mm[3]
    else:
        mmm = 0.35
    if z4 > 97 and z4 <= 114:
        return {"gender": "male", "mood": "none", "p-value/sample": mmm}
        # print(
        #     "a Male, mood of speech: Showing no emotion, normal, p-value/sample size= :%.2f" % (mmm), (nnn))
    elif z4 > 114 and z4 <= 135:
        return {"gender": "male", "mood": "reading", "p-value/sample": mmm}
        # print(
        #     "a Male, mood of speech: Reading, p-value/sample size= :%.2f" % (mmm), (nnn))
    elif z4 > 135 and z4 <= 163:
        return {"gender": "male", "mood": "passion", "p-value/sample": mmm}
        # print(
        #     "a Male, mood of speech: speaking passionately, p-value/sample size= :%.2f" % (mmm), (nnn))
    elif z4 > 163 and z4 <= 197:
        return {"gender": "female", "mood": "none", "p-value/sample": mmm}
        # print("a female, mood of speech: Showing no emotion, normal, p-value/sample size= :%.2f" % (mmm), (nnn))
    elif z4 > 197 and z4 <= 226:
        return {"gender": "female", "mood": "reading", "p-value/sample": mmm}
        # print(
        #     "a female, mood of speech: Reading, p-value/sample size= :%.2f" % (mmm), (nnn))
    elif z4 > 226 and z4 <= 245:
        return {"gender": "female", "mood": "passion", "p-value/sample": mmm}
        # print(
        #     "a female, mood of speech: speaking passionately, p-value/sample size= :%.2f" % (mmm), (nnn))
    else:
        return {"error": True}


def mysp_process(filename, dirname):
    sound = dirname+"/"+filename+".wav"
    path = dirname+"/"

    try:
        objects = run_file(sourcerun, -20, 2, 0.3, "yes",
                           sound, path, 80, 400, 0.01, capture_output=True)

        res = {}

        # print("_myspgend", _myspgend(objects), flush=True)
        # print("_myspsyl", _myspsyl(objects), flush=True)
        # print("_mysppaus", _mysppaus(objects), flush=True)
        # print("_myspsr", _myspsr(objects), flush=True)
        # print("_myspatc", _myspatc(objects), flush=True)
        # print("_myspst", _myspst(objects), flush=True)
        # print("_myspod", _myspod(objects), flush=True)
        # print("_myspbala", _myspbala(objects), flush=True)
        # print("_myspf0mean", _myspf0mean(objects), flush=True)
        # print("_myspf0sd", _myspf0sd(objects), flush=True)
        # print("_myspf0med", _myspf0med(objects), flush=True)
        # print("_myspf0min", _myspf0min(objects), flush=True)
        # print("_myspf0max", _myspf0max(objects), flush=True)
        # print("_myspf0q25", _myspf0q25(objects), flush=True)
        # print("_myspf0q75", _myspf0q75(objects), flush=True)
        # print("_mysppron", _mysppron(objects), flush=True)
        # print("_myspgend", _myspgend(objects), flush=True)
        
        res.update(_myspgend(objects))
        res.update(_myspsyl(objects))
        res.update(_mysppaus(objects))
        res.update(_myspsr(objects))
        res.update(_myspatc(objects))
        res.update(_myspst(objects))
        res.update(_myspod(objects))
        res.update(_myspbala(objects))
        res.update(_myspf0mean(objects))
        res.update(_myspf0sd(objects))
        res.update(_myspf0med(objects))
        res.update(_myspf0min(objects))
        res.update(_myspf0max(objects))
        res.update(_myspf0q25(objects))
        res.update(_myspf0q75(objects))
        res.update(_mysppron(objects))
        res.update(_myspgend(objects))

        return res
    except Exception as e:
        return {"error": True, "reason": e}


def mysp_resample_and_process(fpath):
    # sample_rate = 44000

    # wav, in_sample_rate = torchaudio.load(fpath, normalize=True)
    # wav = torchaudio.transforms.Resample(
    #     orig_freq=in_sample_rate, new_freq=sample_rate)(wav)
    # wav = torch.mean(wav, axis=0, keepdims=True)

    # torchaudio.save("_tmp.wav", wav, sample_rate)

    out = mysp_process(os.path.splitext(os.path.basename(fpath))[0], os.path.realpath(os.path.dirname(fpath)))

    res = dict([(k, out[k] if k in out else "-") for k in headers])

    try:
        res.update(evaluate_audio("AMPEX_DIVA.exe", os.path.realpath(fpath)))
    except Exception as e:
        print(traceback.format_exc(), file=sys.stdout)

    # os.unlink("_tmp.wav")

    return res
