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
