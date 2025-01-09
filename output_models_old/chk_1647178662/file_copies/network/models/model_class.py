import torch
from .m5_class import M5Class as M5
from torch import nn
from tqdm import tqdm
from .base_model import BaseModel
from .full_model_class import FullModelClass as FullModel
from network.scaled_optimizer import ScaledOptimizer


class ModelClass(BaseModel):
    def __init__(self, class_count):
        super(ModelClass, self).__init__()

        self.unwrapped_model = M5(class_count)

        network = FullModel(self.unwrapped_model)
        network = nn.DataParallel(network).cuda()

        self.model = network

    def compile(self, learning_rate, mixed_precision):
        self._do_compile(self.unwrapped_model.parameters(), learning_rate, mixed_precision)

    def do_step(self, is_training, t_data):
        # t_inp, t_out_kls, t_out_rts = t_data
        t_inp, t_out_kls = t_data
        t_inp = t_inp.float()
        # t_out_rts = t_out_rts.float()

        # t_inp, t_out_kls, t_out_rts = [
        t_inp, t_out_kls = [
            t.cuda().contiguous()
            # for t in [t_inp, t_out_kls, t_out_rts]
            for t in [t_inp, t_out_kls]
        ]

        with torch.cuda.amp.autocast(enabled=self.opts.use_mixed_precision):
            loss_class, accuracy_class = \
                self.model(t_inp, t_out_kls)
                # self.model(t_inp, t_out_kls, t_out_rts)

            if is_training:
                self.opts.backward(loss_class)

        return {
            "loss": {"class": loss_class.detach()},
            "accuracy": {"class": accuracy_class.detach()}
        }

    def do_epoch(self, is_training, dataset: torch.utils.data.DataLoader):
        loss_name = "loss" if is_training else "eval loss"
        its, epoch_metrics = 0, {"loss": {"loss": 0}, "accuracy": {}}
        epoch_loss, epoch_acc = epoch_metrics["loss"], epoch_metrics["accuracy"]

        it_tqdm = tqdm(dataset)

        for data in it_tqdm:
            if is_training:
                self.opts.zero_grad()

            step_metrics = self.do_step(is_training, data)
            loss_metrics, acc_metrics = step_metrics["loss"], step_metrics["accuracy"]

            for key in loss_metrics:
                value = loss_metrics[key]

                if key not in epoch_loss:
                    epoch_loss[key] = value
                else:
                    epoch_loss[key] += value

                epoch_loss["loss"] += value

            for key in acc_metrics:
                value = acc_metrics[key]

                if key not in epoch_acc:
                    epoch_acc[key] = value
                else:
                    epoch_acc[key] += value

            its += 1

            it_tqdm.set_description("%s: %.4E" %
                                    (loss_name, epoch_loss["loss"] / its))

            if is_training:
                self.opts.step_batch(epoch_loss["loss"])

        for key in epoch_loss:
            epoch_loss[key] /= its

        for key in epoch_acc:
            epoch_acc[key] /= its

        return epoch_metrics
