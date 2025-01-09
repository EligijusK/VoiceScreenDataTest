import os
import re
import imp
import abc
import atexit
import shutil
import numpy as np
import configs.config as cfg
from configs.args import args
from easydict import EasyDict
from time import time as get_time
from data.dataset import BaseDataset
from utils.sys_colors import GREEN, RESET
from utils.kill_miners import kill_miners
import torch.utils.tensorboard as tensorboard
from .models.base_model import BaseModel
from network.scaled_optimizer import ScaledOptimizer
from utils.files import load_json, save_json, copy_file
from network.state import create_empty_state, set_state, copy_state


class BaseNetwork(abc.ABC):
    def __init__(self, *, checkpoint: str = None, checkpoint_type: str = None):
        super(BaseNetwork, self).__init__()

        if checkpoint == "latest":
            checkpoint = self.find_last_checkpoint()
            print("%sUsing latest checkpoint, found: '%s'%s" %
                  (GREEN, checkpoint, RESET))

        self.checkpoint = round(get_time()) if not checkpoint else checkpoint
        settings_path = "%s/settings.json" % self.dir_save

        if checkpoint and os.path.exists(settings_path):
            self.settings = load_json(settings_path)
        else:
            self.settings = EasyDict({
                "learning_rate": args.learning_rate,
                "class_count": cfg.MODEL.DEEPSPEECH.OUTPUT_CLASS,
                "model": self.ModelKey
            })

        self.net: BaseModel = self.initModel()

        self.current_state = create_empty_state()

        if checkpoint:
            self.net.compile(
                learning_rate=args.learning_rate,
                mixed_precision=args.mixed_precision
            )
            self.restore(checkpoint_type=checkpoint_type)

    @abc.abstractmethod
    def initModel(self):
        pass

    def find_last_checkpoint(self) -> str:
        _, dirs_checkpoints, _ = next(os.walk(cfg.DIRS.OUTPUT_MODELS))

        last_checkpoint, last_time = None, 0

        regexp = re.compile("^chk_(?P<checkpoint>\d+)$")

        for dir_checkpoint_name in dirs_checkpoints:
            matches = regexp.match(dir_checkpoint_name)

            if matches is None:
                continue

            dir_checkpoint = "%s/%s" % (cfg.DIRS.OUTPUT_MODELS,
                                        dir_checkpoint_name)

            time = os.path.getmtime(dir_checkpoint)

            if last_time < time:
                last_checkpoint = int(matches.groupdict()["checkpoint"])
                last_time = time

        return last_checkpoint

    def train(self, *, dataset: BaseDataset):
        model = self.net
        net_model = self.net.model

        dir_save = self.dir_save
        dir_save_cpy = "%s/file_copies" % dir_save
        dir_save_cpy_network = "%s/network" % dir_save_cpy
        dir_save_cpy_models = "%s/models" % dir_save_cpy_network

        last_state = self.current_state.last_state
        best_state = self.current_state.best_state
        best_train, best_test = last_state.train.loss, last_state.test.loss

        os.makedirs(dir_save, exist_ok=True)
        os.makedirs(dir_save_cpy, exist_ok=True)
        os.makedirs(dir_save_cpy_network, exist_ok=True)
        os.makedirs(dir_save_cpy_models, exist_ok=True)
        base_dir = cfg.DIRS.BASE_DIR

        Network = type(self)

        [
            Network.copy_file(
                "%s/network/%s.py" % (base_dir, name),
                 "%s/%s.py" % (dir_save_cpy_network, name)
            ) for name in [
                "base_network",
                "network_class",
                "network_index",
                "base_stripped_network",
                "stripped_class_network",
                "stripped_index_network",
                "__init__"
            ]
        ]

        Network.copy_file(
            "%s/network/__init__.py" %
            base_dir, "%s/__init__.py" % dir_save_cpy_network
        )

        dir_network = "%s/network/models" % base_dir
        _, dirs, files = next(os.walk(dir_network))

        for path_model in files:
            Network.copy_file(
                "%s/%s" % (dir_network, path_model),
                "%s/%s" % (dir_save_cpy_models, path_model)
            )

        for dir in filter(lambda d: not d.startswith("__"), dirs):
            _, _, files = next(os.walk("%s/%s" % (dir_network, dir)))

            out_subdir = "%s/%s" % (dir_save_cpy_models, dir)
            os.makedirs(out_subdir, exist_ok=True)

            for path_model in files:
                Network.copy_file(
                    "%s/%s/%s" % (dir_network, dir, path_model),
                    "%s/%s" % (out_subdir, path_model)
                )

        summary_training, summary_testing = [
            tensorboard.SummaryWriter(log_dir="%s/logs_%s" % (dir_save, p))
            for p in ["training", "testing"]
        ]

        atexit.register(self.cleanup)

        path_latest = "%s/model" % dir_save
        path_best_test = "%s/model_best_test" % dir_save
        path_best_train = "%s/model_best_train" % dir_save
        path_model_info = "%s/model_info.json" % dir_save

        data_training, data_testing = dataset.gen_batches(
            args.batch_size,
            num_workers=args.workers >> 1
        )

        save_json("%s/settings.json" % dir_save, self.settings)

        kls = type(self)

        def do_epoch(epoch):
            nonlocal best_train, best_train, best_test

            learning_rate = model.opts.learning_rate
            print("%s---------------------------------------\n%sEpoch: %i (LR: %.8E)" %
                  (GREEN, RESET, epoch + 1, learning_rate))

            train_result = model.fit(data_training)
            epoch_results = train_result["history"][0]

            train_loss_total, test_loss_total = kls._write_log(
                model=model,
                data_testing=data_testing,
                args=args,
                last_state=last_state,
                epoch=epoch,
                summary_training=summary_training,
                summary_testing=summary_testing,
                epoch_results=epoch_results
            )

            model.save_weights("%s.pth" % path_latest)

            if best_train > train_loss_total:
                best_train = train_loss_total

                best_state.train.epoch = epoch + 1
                set_state(best_state.train.train, train_loss_total)

                if epoch >= args.testing_set:
                    set_state(best_state.train.test, test_loss_total)

                copy_file("%s.pth" % path_latest, "%s.pth" % path_best_train)

            if epoch >= args.testing_set and best_test > test_loss_total:
                best_test = test_loss_total

                best_state.test.epoch = epoch + 1
                set_state(best_state.test.train, train_loss_total)
                set_state(best_state.test.test, test_loss_total)

                copy_file("%s.pth" % path_latest, "%s.pth" % path_best_test)

            save_json(path_model_info, self.current_state)

        model.compile(
            learning_rate=args.learning_rate,
            mixed_precision=args.mixed_precision
        )

        kill_miners()

        if np.isfinite(args.epochs):
            for epoch in range(last_state.epoch, args.epochs):
                do_epoch(epoch)
        else:
            epoch = last_state.epoch

            while True:
                do_epoch(epoch)
                epoch = epoch + 1

        summary_training.close()
        summary_testing.close()

    @classmethod
    @abc.abstractclassmethod
    def _write_log(*, model, data_testing, args, last_state, epoch, summary_training, summary_testing, epoch_results):
        pass

    @property
    @abc.abstractproperty
    def ModelKey(self) -> str:
        pass

    @staticmethod
    @abc.abstractstaticmethod
    def globals():
        pass

    @property
    def ModelConstructor(self) -> BaseModel:
        py_path = py_path = "%s/file_copies/network/models/__init__.py" % self.dir_save

        if os.path.isfile(py_path):
            module_model = imp.load_package("_models", py_path)

            return getattr(module_model, self.ModelKey)
        else:
            return self.globals()[self.ModelKey]

    def cleanup(self):
        if self.current_state.last_state.epoch == 0:
            shutil.rmtree(self.dir_save)

    def predict(self, inputs: np.array) -> np.array:
        return self.net.predict(inputs)

    def restore(self, *, checkpoint_type: str):
        dir_save = self.dir_save

        json_path = "%s/model_info.json" % dir_save

        if os.path.exists(json_path):
            model_info = load_json(json_path)
            last_epoch = model_info.last_state.epoch

            if model_info.model_name != self.ModelKey:
                raise Exception("Attempted to load wrong network '%s' expected '%s'" % (self.ModelKey, model_info.model_name))

            if checkpoint_type == "model":
                self.current_state = model_info
            elif checkpoint_type == "model_best_train" or checkpoint_type == "model_best_test":
                cur_state = model_info.last_state

                if checkpoint_type == "model_best_train":
                    best_state = model_info.best_state.train
                else:
                    best_state = model_info.best_state.test

                cur_state.epoch = best_state.epoch
                copy_state(cur_state.train, best_state.train)
                copy_state(cur_state.test, best_state.test)
            else:
                raise Exception(
                    "Invalid checkpoint type: %s" % checkpoint_type)

            self.current_state.last_state.epoch = last_epoch
        path = "%s/%s.pth" % (dir_save, checkpoint_type)

        if os.path.exists(path):
            self.net.restore(path, args.load_optimizer)

    @staticmethod
    def get_dir_save(checkpoint: str) -> str:
        return "%s/chk_%s" % (cfg.DIRS.OUTPUT_MODELS, checkpoint)

    @property
    def dir_save(self) -> str:
        return type(self).get_dir_save(self.checkpoint)

    @staticmethod
    def copy_file(src, dst):
        if not os.path.exists(dst):
            shutil.copy(src, dst)
