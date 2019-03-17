import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_batch(tens):
    size = tens.size()
    dim = len(size)
    if dim > 2:
        return True
    elif dim == 2 and size[0] != 1:
        return True
    else:
        return False