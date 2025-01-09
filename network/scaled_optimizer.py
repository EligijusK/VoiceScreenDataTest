import torch as _torch
import torch.optim as _optim
from network.optimizer import Optimizer as _Optimizer


class GroupScaledOptimizer():
    def __init__(self, mixed_precision=True):
        self.use_mixed_precision = mixed_precision
        self.scaler: _torch.cuda.amp.GradScaler = None

    def compile(self):
        self.scaler = _torch.cuda.amp.GradScaler(
            enabled=self.use_mixed_precision)

    def backward(self, *optimize):
        scaler = self.scaler

        for loss, optimizers in optimize:
            [opt.optimizer.zero_grad() for opt in optimizers]
            scaler.scale(loss).backward(retain_graph=True)
            [scaler.step(opt.optimizer) for opt in optimizers]

        scaler.update()


class ScaledOptimizer(_Optimizer):
    def __init__(self, learning_rate, mixed_precision=True):
        super(ScaledOptimizer, self).__init__(learning_rate)

        self.use_mixed_precision = mixed_precision
        self.scaler: _torch.cuda.amp.GradScaler = None

    def compile(self, parameters):
        super(ScaledOptimizer, self).compile(parameters)

        self.scaler = _torch.cuda.amp.GradScaler(
            enabled=self.use_mixed_precision)

    def backward(self, loss):
        optimizer, scaler = self.optimizer, self.scaler

        optimizer.zero_grad()
        scaler.scale(loss).backward(retain_graph=True)
        scaler.step(optimizer)
        scaler.update()
