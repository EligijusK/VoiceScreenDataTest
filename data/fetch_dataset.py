import os
import re
import torch
import operator
import functools
import torchaudio
import numpy as np
from glob import glob
from tqdm import tqdm
from math import ceil
import configs.config as cfg

regexp = re.compile("\/group_(?P<groupid>\d)+\/((?P<filename>[\w\d\s]+).wav)")


def read_wav(fpath, sample_rate):
    wav, in_sample_rate = torchaudio.load(fpath, normalize=True)
    wav = torchaudio.transforms.Resample(
        orig_freq=in_sample_rate, new_freq=sample_rate)(wav)
    wav = torch.mean(wav, axis=0, keepdims=True)
    mfcc = torchaudio.transforms.MFCC(sample_rate, 80)(wav)

    return mfcc


def get_dataset_index(*, dir_input, training_ratio, reduced_dataset=False, sample_rate=8000):
    dataset_list = [f for f in glob(
        "%s/dataset_**/dataset.csv" % dir_input, recursive=True)]
    all_samples = []
    ind_groups = [0.3, 0.55, 1]
    num_groups = len(ind_groups)

    groups = [[] for _ in range(num_groups)]
    training = [[] for _ in range(num_groups)]
    testing = [[] for _ in range(num_groups)]

    for fname in dataset_list:
        dname = os.path.dirname(fname)
        with open(fname, "r") as f:
            lines = f.readlines()

            header = [(i, s) for i, s in enumerate([s.strip()
                                                    for s in lines[0].split(",")])]
            wav_lines = [[s.strip() for s in l.split(",")] for l in lines[1:]]

            idx_index, _ = next(filter(lambda h: "File" in h[1], header))
            idx_shi, _ = next(filter(lambda h: "SHI" in h[1], header))
            idx_totals = [i for i, _ in filter(
                lambda h: h[1].startswith("Total"), header)]

            wav_list = [f for f in glob(
                "%s/group_**/*.wav" % dname, recursive=True)]

            for info in tqdm(wav_lines):
                index = "%s.wav" % info[idx_index].zfill(3)
                shi = float(info[idx_shi]) / 60
                totals = float(
                    np.mean([float(info[idx]) / 50 for idx in idx_totals]))

                fpath = next(filter(lambda f: f.endswith(index), wav_list))

                mfcc = read_wav(fpath, sample_rate)

                all_samples.append((mfcc, (shi, totals), fpath))

        for sample in all_samples:
            inp, (out_shi, out_total), fname = sample

            for i, off in zip(range(0, num_groups), ind_groups):
                if out_shi <= off:
                    groups[i].append(sample)
                    break

        min_elements = np.min([len(a) for a in groups])
        all_elements = [g[0:min_elements] for g in groups]

        for i, g in enumerate(all_elements):
            len_samples = len(g)
            len_total = min(len_samples, reduced_dataset) \
                if reduced_dataset is not None             \
                else len_samples
            len_training = ceil(len_total * training_ratio)
            len_testing = len_total - len_training

            training[i] = g[0:len_training]
            testing[i] = g[len_training:len_training+len_testing]

        training, testing = [
            functools.reduce(operator.iconcat, a, [])
            for a in [training, testing]
        ]

        return training, testing

from pathlib import Path
from itertools import chain

def get_dataset_class(*, dir_input, training_ratio, reduced_dataset=False, sample_rate=8000):
    num_classes = cfg.MODEL.DEEPSPEECH.OUTPUT_CLASS

    # sound_list = [f for f in glob(
    #     "%s/dataset_**/group_**/*.wav" % dir_input, recursive=True)]
    
    datasets = dict(map(lambda d: (d.stem, d), filter(
        lambda d: d.is_dir(), sorted(list(Path(dir_input).iterdir())))))
    files = dict(map(lambda d: (d[0], sorted(
        list(filter(lambda f: f.suffix == ".wav", d[1].iterdir())))), datasets.items()))

    sound_list = list(chain(*zip(*files.values())))

    groups = [[] for i in range(num_classes)]
    training = [[] for i in range(num_classes)]
    testing = [[] for i in range(num_classes)]

    for fpath in tqdm(sound_list, desc="Loading data"):
        # match = regexp.search(fpath)

        # if match is None:
        #     continue

        # group_id, filename = int(match["groupid"]), match["filename"]

        # if group_id == 3:
        #     continue

        group_id = 0

        mfcc = read_wav(fpath, sample_rate)

        groups[group_id].append((mfcc, group_id, fpath))

    for all_samples, training_samples, testing_samples in zip(groups, training, testing):
        len_samples = len(all_samples)
        len_total = min(len_samples, reduced_dataset) \
            if reduced_dataset is not None             \
            else len_samples
        len_training = ceil(len_total * training_ratio)
        len_testing = len_total - len_training

        data_training = all_samples[0:len_training]
        data_testing = all_samples[len_training:len_training+len_testing]

        training_samples.extend(data_training)
        testing_samples.extend(data_testing)

    training, testing = [
        functools.reduce(operator.iconcat, a, [])
        for a in [training, testing]
    ]

    return training, testing


def get_dataset_avqi(*, dir_input, training_ratio, reduced_dataset=False, sample_rate=8000):
    with open("%s/dataset.csv" % dir_input, "r") as f:
        lines = [l.split(",") for l in f.readlines()]

    re_dataset = re.compile("^dataset_\d+$")
    dataset_offsets = list(map(lambda t: t[0], filter(lambda t: re_dataset.match(t[1]) is not None, enumerate(lines[0]))))


    all_samples = []

    groups = [[] for _ in range(num_groups)]
    training = [[] for _ in range(num_groups)]
    testing = [[] for _ in range(num_groups)]

    for fname in dataset_list:
        dname = os.path.dirname(fname)
        with open(fname, "r") as f:
            lines = f.readlines()

            header = [(i, s) for i, s in enumerate([s.strip()
                                                    for s in lines[0].split(",")])]
            wav_lines = [[s.strip() for s in l.split(",")] for l in lines[1:]]

            idx_index, _ = next(filter(lambda h: "File" in h[1], header))
            idx_shi, _ = next(filter(lambda h: "SHI" in h[1], header))
            idx_totals = [i for i, _ in filter(
                lambda h: h[1].startswith("Total"), header)]

            wav_list = [f for f in glob(
                "%s/group_**/*.wav" % dname, recursive=True)]

            for info in tqdm(wav_lines):
                index = "%s.wav" % info[idx_index].zfill(3)
                shi = float(info[idx_shi]) / 60
                totals = float(
                    np.mean([float(info[idx]) / 50 for idx in idx_totals]))

                fpath = next(filter(lambda f: f.endswith(index), wav_list))

                mfcc = read_wav(fpath, sample_rate)

                all_samples.append((mfcc, (shi, totals), fpath))

        for sample in all_samples:
            inp, (out_shi, out_total), fname = sample

            for i, off in zip(range(0, num_groups), ind_groups):
                if out_shi <= off:
                    groups[i].append(sample)
                    break

        min_elements = np.min([len(a) for a in groups])
        all_elements = [g[0:min_elements] for g in groups]

        for i, g in enumerate(all_elements):
            len_samples = len(g)
            len_total = min(len_samples, reduced_dataset) \
                if reduced_dataset is not None             \
                else len_samples
            len_training = ceil(len_total * training_ratio)
            len_testing = len_total - len_training

            training[i] = g[0:len_training]
            testing[i] = g[len_training:len_training+len_testing]

        training, testing = [
            functools.reduce(operator.iconcat, a, [])
            for a in [training, testing]
        ]

        return training, testing