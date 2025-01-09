import torch as _torch
import torch.optim as _optim


class Optimizer:
    def __init__(self, learning_rate):
        super(Optimizer, self).__init__()

        self.initial_lr = learning_rate
        self.weight_decay = 1e-5
        self.optimizer: _optim.Adam = None
        self.per_batch_scheduler = True
        self.scheduler: _optim.lr_scheduler._LRScheduler = None
        self.enabled = True

    def get_state(self): return self.optimizer.state_dict()
    def set_state(self, state): self.optimizer.load_state_dict(state)

    def load_state_dict(self, state): self.optimizer.load_state_dict(state)

    def step_batch(self, loss):
        if self.scheduler is not None and self.per_batch_scheduler:
            self.scheduler.step(loss if _torch.isfinite(loss) else _torch.tensor(100., dtype=loss.dtype, device=loss.device))

    def step_epoch(self, train_loss):
        if self.scheduler is not None and not self.per_batch_scheduler:
            self.scheduler.step(train_loss)

    def compile(self, parameters):
        self.optimizer = _optim \
            .AdamW(parameters, lr=self.initial_lr)

        self.scheduler = _optim.lr_scheduler \
            .CosineAnnealingWarmRestarts(self.optimizer, 500, eta_min=1e-7)

        # self.scheduler = _optim.lr_scheduler \
        #     .ReduceLROnPlateau(self.optimizer, patience=1000)

    @property
    def learning_rate(self): return self.optimizer.param_groups[0]["lr"]
    @learning_rate.setter
    def learning_rate(self, lr): self.optimizer.param_groups[0]["lr"] = lr

    def state_dict(self): return self.optimizer.state_dict()

    def backward(self, loss):
        optimizer = self.optimizer

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def drop(self):
        self.scheduler = None
        self.optimizer = None

    def zero_grad(self): self.optimizer.zero_grad()
