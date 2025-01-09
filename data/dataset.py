import abc
import torch
import random
import torchaudio
import numpy as np
import configs.config as cfg
from configs.args import args
from data.batches import Batch
import torch.nn.functional as F
import data.fetch_dataset as fetch
from torch.utils.data import DataLoader


class RandomClip:
    def __init__(self, sample_rate, min_length, max_length):
        self.min_len = min_length
        self.max_len = max_length
        self.sample_rate = sample_rate

        self.vad = torchaudio.transforms.Vad(
            sample_rate=sample_rate, trigger_level=7.0)

    def __call__(self, audio_data):
        audio_length = audio_data.shape[-1]
        clip_dur = random.uniform(self.min_len, self.max_len)
        clip_length = int(clip_dur * self.sample_rate)

        if audio_length > clip_length:
            offset = random.randint(0, audio_length - clip_length)
            audio_data = audio_data[..., offset:(offset + clip_length)]

        return self.vad(audio_data)  # remove silences at the beggining/end


def collate_fn(sampler, batch):
    # t_wav, t_kls, t_rts = [], [], []
    t_mfcc, t_kls, t_fname, t_wav = [], [], [], []

    # for wav, klass, rts in batch:
    for mfcc, klass, fname in batch:
        t_mfcc.append(sampler(mfcc))
        t_kls.append(klass)
        t_fname.append(fname)
        # t_wav.append(wav)
        # t_rts.append(rts)

    max_wav = int((torch.tensor([t.shape[-1] for t in t_mfcc])).max())
    t_mfcc = torch.concat([
        F
        .pad(t, [0, max_wav - int(t.shape[-1]), 0, 0])
        .view(1, t.shape[1], max_wav)
        for t in t_mfcc
    ], 0)

    t_kls = torch.tensor(t_kls)
    # t_rts = torch.tensor(np.array(t_rts))

    return t_mfcc, t_kls, t_fname, t_wav # , t_rts


class BaseDataset(abc.ABC):
    def __init__(self, *, dir_input, training_ratio, balance_training=True, balance_testing=None):
        super(BaseDataset, self).__init__()

        self.training, self.testing = type(self).get_dataset(
            dir_input=dir_input,
            training_ratio=training_ratio
        )

    def gen_batches(self, batch_size, test_batch_size=None, num_workers=0, use_cache=True, shuffle_train=True, shuffle_test=False, drop_last=True):
        prefetch_factor = 5 if num_workers > 0 else 2
        persistent_workers = True if num_workers > 0 else False

        def foo_collate(x): return x

        training_batch = DataLoader(
            Batch(name="training", elements=self.training),
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            shuffle=shuffle_train,
            collate_fn=lambda batch: collate_fn(foo_collate, batch),
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers
        ) if len(self.training) > 0 else None

        testing_batch = DataLoader(
            Batch(name="testing", elements=self.testing),
            batch_size=batch_size if test_batch_size is None else test_batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=lambda batch: collate_fn(foo_collate, batch),
            shuffle=shuffle_test,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers
        )

        return training_batch, testing_batch

    @staticmethod
    @abc.abstractstaticmethod
    def get_dataset(*, dir_input, training_ratio):
        pass


class DatasetClass(BaseDataset):
    @staticmethod
    def get_dataset(*, dir_input, training_ratio):
        return fetch.get_dataset_class(
            dir_input=dir_input,
            training_ratio=training_ratio,
            reduced_dataset=args.reduced_dataset,
            sample_rate=cfg.MODEL.SAMPLING_RATE
        )


class DatasetIndex(BaseDataset):
    @staticmethod
    def get_dataset(*, dir_input, training_ratio):
        return fetch.get_dataset_index(
            dir_input=dir_input,
            training_ratio=training_ratio,
            reduced_dataset=args.reduced_dataset,
            sample_rate=cfg.MODEL.SAMPLING_RATE
        )


class DatasetAVQI(BaseDataset):
    @staticmethod
    def get_dataset(*, dir_input, training_ratio):
        return fetch.get_dataset_avqi(
            dir_input=dir_input,
            training_ratio=training_ratio,
            reduced_dataset=args.reduced_dataset,
            sample_rate=cfg.MODEL.SAMPLING_RATE
        )
