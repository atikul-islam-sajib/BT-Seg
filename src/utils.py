import os
import torch
import torch.nn as nn
import joblib


def dump(value, filename):
    if value is not None and filename is not None:
        joblib.dump(value=value, filename=filename)
    else:
        raise Exception("Value or filename cannot be None".capitalize())


def load(filename):
    if filename is not None:
        return joblib.load(filename)
    else:
        raise Exception("Filename cannot be None".capitalize())


def device_init(device="mps"):
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
