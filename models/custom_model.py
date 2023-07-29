import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch



class CustomModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # define model

        if self.args.pretrained:
            # load pretrained weights


    def forward(self, x):
        return


    