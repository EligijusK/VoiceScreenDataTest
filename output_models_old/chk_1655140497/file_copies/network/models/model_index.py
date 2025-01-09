import torch
from torch import nn
from tqdm import tqdm
from .base_model import BaseModel
from .m5_index import M5Index as M5
from network.scaled_optimizer import ScaledOptimizer
from .full_model_index import FullModelIndex as FullModel


class ModelIndex(BaseModel):
    def __init__(self):
        super(ModelIndex, self).__init__()

        self.unwrapped_model = M5()

        network = FullModel(self.unwrapped_model)
        network = nn.DataParallel(network).cuda()

        self.model = network

    def compile(self, learning_rate, mixed_precision):
        self._do_compile(self.unwrapped_model.parameters(), learning_rate, mixed_precision)

    def do_step(self, is_training, t_data):
        t_inp, t_out, _ = t_data
        t_inp = t_inp.float()

        t_inp, t_out = [
            t.cuda().contiguous()
            for t in [t_inp, t_out]
        ]

        with torch.cuda.amp.autocast(enabled=self.opts.use_mixed_precision):
            loss = \
                self.model(t_inp, t_out)

            if is_training:
                self.opts.backward(loss)

        return {
            "loss": {"loss": loss.detach()}
        }

    def do_epoch(self, is_training, dataset: torch.utils.data.DataLoader):
        loss_name = "loss" if is_training else "eval loss"
        its, epoch_metrics = 0, {"loss": {"loss": 0}}
        epoch_loss = epoch_metrics["loss"]

        it_tqdm = tqdm(dataset)

        for data in it_tqdm:
            if is_training:
                self.opts.zero_grad()

            step_metrics = self.do_step(is_training, data)
            loss_metrics = step_metrics["loss"]

            for key in loss_metrics:
                value = loss_metrics[key]

                if key not in epoch_loss:
                    epoch_loss[key] = value
                else:
                    epoch_loss[key] += value

                epoch_loss["loss"] += value

            its += 1

            it_tqdm.set_description("%s: %.4E" %
                                    (loss_name, epoch_loss["loss"] / its))

            if is_training:
                self.opts.step_batch(epoch_loss["loss"])

        for key in epoch_loss:
            epoch_loss[key] /= its

        return epoch_metrics
