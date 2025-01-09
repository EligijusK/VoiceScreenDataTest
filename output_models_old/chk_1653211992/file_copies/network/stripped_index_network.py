import os
import torch
import numpy as np
from .models.m5_index import M5Index as _M5
from .base_stripped_network import BaseStrippedNetwork


class StrippedNetworkIndex(BaseStrippedNetwork):
    def _init_model(*args, **kwargs):
        return _M5()
