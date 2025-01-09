import os
import torch
import numpy as np
from .models.m5_class import M5Class as _M5
from .base_stripped_network import BaseStrippedNetwork


class StrippedNetworkClass(BaseStrippedNetwork):
    def _init_model(*args, **kwargs):
        return _M5(class_count)
