import torch
from torch import nn
import torch.autograd as grad
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101
from torchaudio.models import Wav2Letter

model_type = "resnet18"

class Clamp(nn.Module):
    def __init__(self, min, max):
        super().__init__()

        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)

class M5Index(nn.Module):
    def __init__(self):
        super().__init__()

        if model_type == "m5":
            self.model = nn.Sequential(
                nn.Conv1d(80, 32, kernel_size=80, stride=16),
                nn.BatchNorm1d(32),
                nn.Relu(),
                # nn.MaxPool1d(1),
                nn.Conv1d(32, 32, kernel_size=1),
                nn.BatchNorm1d(32),
                nn.Relu(),
                # nn.MaxPool1d(1),
                nn.Conv1d(32, 2 * 32, kernel_size=1),
                nn.BatchNorm1d(2 * 32),
                nn.Relu(),
                # nn.MaxPool1d(1),
                nn.Conv1d(2 * 32, 2 * 32, kernel_size=1),
                nn.BatchNorm1d(2 * 32),
                nn.Relu(),
                # nn.MaxPool1d(1),
                nn.Conv1d(2 * 32, 2, kernel_size=1),
                nn.ReLU(),
                Clamp(0, 1)
            )
        elif model_type == "resnet18" or model_type == "resnet101":
            if model_type == "resnet18":
                self.model = resnet18()
            elif model_type == "resnet101":
                self.model = resnet101()
            else:
                raise Exception("Bad resnet")

            self.model.conv1 = nn.Conv2d(1, self.model.conv1.out_channels,
                                        kernel_size=self.model.conv1.kernel_size[0],
                                        stride=self.model.conv1.stride[0],
                                        padding=self.model.conv1.padding[0])
            self.model.fc = nn.Sequential(
                # nn.Linear(self.model.fc.in_features, 4 + 8)
                nn.Linear(self.model.fc.in_features, 2),
                nn.ReLU(),
                Clamp(0, 1)
            )
        else:
            raise Exception("Bad model")

    def forward(self, t_in):
        x = t_in

        if model_type == "m5":
            x, _ = self.model(x).max(axis=-1)
        elif model_type == "resnet18" or model_type == "resnet101":
            x = x.unsqueeze(1)
            x = self.model(x)
        else:
            raise Exception("Bad model")

        return x
