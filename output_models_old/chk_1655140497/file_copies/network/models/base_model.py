import abc
import torch
from network.scaled_optimizer import ScaledOptimizer

class BaseModel(abc.ABC):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.unwrapped_model = None
        self.model = None
        self.opts = None

    def _do_compile(self, params, learning_rate, mixed_precision):
        self.opts = ScaledOptimizer(
            learning_rate, mixed_precision=mixed_precision)
        self.opts.compile(params)

    @abc.abstractmethod
    def do_step(self, is_training, t_data):
        pass

    @abc.abstractmethod
    def do_epoch(self, is_training, dataset: torch.utils.data.DataLoader):
        pass

    def save_weights(self, path):
        torch.save({
            "state_model": self.unwrapped_model.state_dict(),
            "state_optimizer": self.opts.state_dict()
        }, path)

    def evaluate(self, dataset):
        self.model.eval()

        with torch.no_grad():
            return self.do_epoch(False, dataset)

    def predict(self, t_in):
        net = self.unwrapped_model
        net.eval()

        with torch.no_grad():
            t_in = t_in.float().cuda().contiguous()
            t_pr = net(t_in)

            return t_pr

    def fit(self, dataset, epochs=1):
        self.model.train()

        history = []

        for i in range(epochs):
            epoch_metrics = self.do_epoch(True, dataset)

            history.append(epoch_metrics)

            self.opts.step_epoch(epoch_metrics["loss"])

        return {"history": history}

    def restore(self, path, load_optimizer):
        state = torch.load(path)

        self.unwrapped_model.load_state_dict(state["state_model"])

        if load_optimizer:
            self.opts.load_state_dict(state["state_optimizer"])