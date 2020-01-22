import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, features, bottleneck=1024):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(nn.Linear(features, bottleneck),
                                   nn.BatchNorm1d(bottleneck),
                                   nn.ReLU(),
                                   nn.Linear(bottleneck, features),
                                   nn.BatchNorm1d(features),
                                   nn.ReLU())

    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity
        out = F.relu(out)
        return out


class PBSNet(nn.Module):
    def __init__(self, in_features, out_features=4, bottleneck=1024):
        super(PBSNet, self).__init__()
        self.upsample = nn.Linear(in_features, bottleneck - 5)
        self.b0 = BasicBlock(bottleneck)
        self.b1 = BasicBlock(bottleneck)
        self.fc = nn.Linear(bottleneck, out_features)
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def mix(a, b, lambda_):
        return lambda_*a + (1-lambda_)*b

    def forward(self, x, assessment_id, shift=None, lambda_=None):
        assessment_code = torch.zeros(x.size(0), 5)
        assessment_code[torch.arange(assessment_code.size(0)).unsqueeze(1), assessment_id] = 1.

        out = F.relu(self.upsample(x))
        out = self.b0(torch.cat([out, assessment_code], dim=1))
        out_ = self.b1(out)
        out = self.softmax(self.fc(out_))

        if shift is None or lambda_ is None:
            return out
        else:
            out_mix = self.mix(out_, out_.roll(shift, dims=0), lambda_)
            out_mix = self.softmax(self.fc(out_mix))
            return out, out_mix

