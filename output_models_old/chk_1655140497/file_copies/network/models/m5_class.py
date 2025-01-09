import torch
from torch import nn
import torch.autograd as grad
import torch.nn.functional as F
from torchvision.models import convnext_tiny
from torchvision.models import resnext50_32x4d as resnext
from torchvision.models import shufflenet_v2_x1_0 as shufflenet
from torchvision.models import resnet18, resnet101, resnet34, resnet50
from torchaudio.models import Wav2Letter

model_type = "resnet18"


class M5Class(nn.Module):
    def __init__(self, kls_count):
        super().__init__()

        self.class_count = kls_count

        if model_type == "shufflenet":
            self.model = shufflenet(True)
            self.model.conv1[0] = nn.Conv2d(1, self.model.conv1[0].out_channels,
                                            kernel_size=self.model.conv1[0].kernel_size[0],
                                            stride=self.model.conv1[0].stride[0],
                                            padding=self.model.conv1[0].padding[0])
            # self.model.fc = nn.Sequential(
            #     nn.Linear(self.model.fc.in_features, kls_count)
            # )
            self.model.fc = nn.Sequential(
                nn.BatchNorm1d(self.model.fc.in_features),
                self.model.fc,
                nn.Linear(self.model.fc.out_features, kls_count),
                nn.Sigmoid()
            )
        elif model_type == "wav2letter":
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
        elif model_type == "convnext_tiny":
            self.model = convnext_tiny(pretrained=True)

            self.model.features = nn.Sequential(
                nn.Conv2d(1, 3, 1),
                nn.BatchNorm2d(3),
                self.model.features,
            )

            # self.model.features[0][0] = nn.Conv2d(1, self.model.features[0][0].out_channels,
            #                              kernel_size=self.model.features[0][0].kernel_size[0],
            #                              stride=self.model.features[0][0].stride[0],
            #                              padding=self.model.features[0][0].padding[0])
            self.model.classifier = nn.Sequential(
                self.model.classifier,
                nn.Linear(self.model.classifier[2].out_features, kls_count),
                nn.Sigmoid()
            )

        elif model_type == "resnext":
            self.model = resnext(True)

            self.model.conv1 = nn.Conv2d(1, self.model.conv1.out_channels,
                                         kernel_size=self.model.conv1.kernel_size[0],
                                         stride=self.model.conv1.stride[0],
                                         padding=self.model.conv1.padding[0])
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, kls_count),
                nn.Sigmoid()
            )
        elif model_type.startswith("resnet"):
            if model_type == "resnet18":
                self.model = resnet18(True)
            elif model_type == "resnet34":
                self.model = resnet34(True)
            elif model_type == "resnet50":
                self.model = resnet50(True)
            elif model_type == "resnet101":
                self.model = resnet101(True)
            else:
                raise Exception("Bad resnet")

            self.model.conv1 = nn.Sequential(
                nn.Conv2d(1, 3, 1),
                nn.BatchNorm2d(3),
                nn.ReLU(),
                self.model.conv1,
            )

            # self.model.conv1 = nn.Conv2d(1, self.model.conv1.out_channels,
            #                              kernel_size=self.model.conv1.kernel_size[0],
            #                              stride=self.model.conv1.stride[0],
            #                              padding=self.model.conv1.padding[0])
            self.model.fc = nn.Sequential(
                self.model.fc,
                nn.BatchNorm1d(self.model.fc.out_features),
                nn.ReLU(),
                nn.Linear(self.model.fc.out_features, kls_count),
                # nn.Linear(self.model.fc.in_features, kls_count),
                nn.Sigmoid()
            )
        else:
            raise Exception("Bad model")

    def forward(self, t_in):
        x = t_in

        if model_type == "wav2letter" or model_type == "m5":
            x, _ = self.model(x).max(axis=-1)
        elif model_type.startswith("resnet") or model_type == "shufflenet" or model_type == "resnext" or model_type == "convnext_tiny":
            x = x.unsqueeze(1)
            x = self.model(x)
        else:
            raise Exception("Bad model")

        return x
