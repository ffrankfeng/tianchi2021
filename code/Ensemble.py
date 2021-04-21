import torch.nn as nn
import torch.nn.functional as F
class Ensemble(nn.Module):
    def __init__(self, model1, model2, model3):
        super(Ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def forward(self, x):
        logits1 = self.model1(x)
        logits2 = self.model2(x)
        logits3 = self.model3(x)

        # fuse logits
        logits_e = (logits1 + logits2 + logits3) / 3

        return logits_e


class Ensemble2(nn.Module):
    def __init__(self, model1, model2):
        super(Ensemble2, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        logits1 = self.model1(x)
        logits2 = self.model2(x)

        # fuse logits
        logits_e = (logits1 + logits2) / 2

        return logits_e
class Ensemble3_hrn(nn.Module):
    def __init__(self, model1, model2, model3):
        super(Ensemble3_hrn, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def forward(self, x):
        logits1 = self.model1(x)
        logits2 = self.model2(x)
        x1 = F.interpolate(x, size=[224, 224], mode='bilinear')
        logits3 = self.model3(x1)
        # fuse logits
        logits_e = (logits1 + logits2 + logits3) / 3

        return logits_e

class Ensemble4(nn.Module):
    def __init__(self, model1, model2, model3, model4):
        super(Ensemble4, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4

    def forward(self, x):
        logits1 = self.model1(x)
        logits2 = self.model2(x)
        logits3 = self.model3(x)
        x1 = F.interpolate(x, size=[224, 224], mode='bilinear')
        logits4 = self.model4(x1)
        # fuse logits
        logits_e = (logits1 + logits2 + logits3 + logits4) / 4

        return logits_e