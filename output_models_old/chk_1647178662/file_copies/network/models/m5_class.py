import torch
from torch import nn
import torch.autograd as grad
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101
from torchaudio.models import Wav2Letter

model_type = "resnet101"

class M5Class(nn.Module):
    def __init__(self, kls_count):
        super().__init__()

        self.class_count = kls_count
        
        if model_type == "wav2letter":
            self.model = Wav2Letter(kls_count, "mfcc", 80)
        elif model_type == "m5":
            self.model = nn.Sequential(
                nn.Conv1d(80, 32, kernel_size=80, stride=16),
                nn.BatchNorm1d(32),
                # nn.MaxPool1d(1),
                nn.Conv1d(32, 32, kernel_size=1),
                nn.BatchNorm1d(32),
                # nn.MaxPool1d(1),
                nn.Conv1d(32, 2 * 32, kernel_size=1),
                nn.BatchNorm1d(2 * 32),
                # nn.MaxPool1d(1),
                nn.Conv1d(2 * 32, 2 * 32, kernel_size=1),
                nn.BatchNorm1d(2 * 32),
                # nn.MaxPool1d(1),
                nn.Conv1d(2 * 32, kls_count, kernel_size=1),
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
                nn.Linear(self.model.fc.in_features, kls_count)
            )
        else:
            raise Exception("Bad model")

    def forward(self, t_in):
        x = t_in

        if model_type == "wav2letter" or model_type == "m5":
            x, _ = self.model(x).max(axis=-1)
        elif model_type == "resnet18" or model_type == "resnet101":
            x = x.unsqueeze(1)
            x = self.model(x)
        else:
            raise Exception("Bad model")

        return x
