from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as models
import sys
import timm


sys.path.append("..")
from Ensemble import Ensemble, Ensemble2, Ensemble4, Ensemble3_hrn


def load_model(model_name):
    model = None

    if 'resnet152_ddn_jpeg'.__eq__(model_name):
        print("load model resnet152_ddn_jpeg")
        m = models.resnet152(pretrained=False)
        weight = './weights/jpeg_ddn_resnet152/jpeg_ddn_resnet152.pth'
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pretrained_model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        loaded_state_dict = torch.load(weight)
        pretrained_model.load_state_dict(loaded_state_dict)
        model = pretrained_model

    elif 'wide_resnet101_2_dnn_jpeg'.__eq__(model_name):
        print("load model wide_resnet101_2_dnn_jpeg")
        m = models.wide_resnet101_2(pretrained=False)
        weight = './weights/jpeg_ddn_wide_resnet101/jpeg_ddn_wide_resnet101.pth'
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        loaded_state_dict = torch.load(weight)
        model.load_state_dict(loaded_state_dict)

    elif 'densenet161_ddn_jpeg'.__eq__(model_name):
        print("load model densenet161_ddn_jpeg")

        m = models.densenet161(pretrained=False)
        weight = './weights/jpeg_ddn_densenet161/jpeg_ddn_densenet161.pth'

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model = NormalizedModel(model=m, mean=image_mean, std=image_std)
        loaded_state_dict = torch.load(weight)
        model.load_state_dict(loaded_state_dict)

    elif 'hrnet_w64_ddn_jpeg'.__eq__(model_name):
        print("load model: hrnet_w64_ddn_jpeg")
        model = timm.create_model('hrnet_w64', pretrained=False)
        image_mean = torch.tensor([0.5000, 0.5000, 0.5000]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.5000, 0.5000, 0.5000]).view(1, 3, 1, 1)
        model = NormalizedModel(model=model, mean=image_mean, std=image_std)
        weight= './weights/jpeg_ddn_hrnet_w64/jpeg_ddn_hrnet_w64.pth'
        loaded_state_dict = torch.load(weight)
        model.load_state_dict(loaded_state_dict)

    elif 'Ensemble_dsn161_jpeg_rn162_jpeg_wrn101_jpeg'.__eq__(model_name):
        print("load model: Ensemble_dsn161_jpeg_rn162_jpeg_wrn101_jpeg")
        model1 = load_model("densenet161_ddn_jpeg")
        model2 = load_model("resnet152_ddn_jpeg")
        model3 = load_model("wide_resnet101_2_dnn_jpeg")
        model = Ensemble(model1, model2, model3)
    elif 'Ensemble_dsn161_jpeg_wrn101_jpeg_hrn_jpeg'.__eq__(model_name):
        print("load model: Ensemble_dsn161_jpeg_wrn101_jpeg_hrn_jpeg")
        model1 = load_model("densenet161_ddn_jpeg")
        model2 = load_model("wide_resnet101_2_dnn_jpeg")
        model3 = load_model("hrnet_w64_ddn_jpeg")
        model = Ensemble3_hrn(model1, model2, model3)
    elif 'Ensemble_dsn161_jpeg_wrn101_jpeg'.__eq__(model_name):
        print("load model: Ensemble_dsn161_jpeg_wrn101_jpeg")
        model1 = load_model("densenet161_ddn_jpeg")
        model2 = load_model("wide_resnet101_2_dnn_jpeg")
        model = Ensemble2(model1, model2)
    elif 'Ensemble_dsn161_jpeg_rn162_jpeg_wrn101_jpeg_hrn_jpeg'.__eq__(model_name):
        print("load model: Ensemble_dsn161_jpeg_rn162_jpeg_wrn101_jpeg_hrn_jpeg")
        model1 = load_model("densenet161_ddn_jpeg")
        model2 = load_model("resnet152_ddn_jpeg")
        model3 = load_model("wide_resnet101_2_dnn_jpeg")
        model4 = load_model("hrnet_w64_ddn_jpeg")
        model = Ensemble4(model1, model2, model3,model4)

    elif 'Ensemble_dsn161_jpeg_rn162_jpeg'.__eq__(model_name):
        print("load model: Ensemble_dsn161_jpeg_rn162_jpeg")
        model1 = load_model("densenet161_ddn_jpeg")
        model2 = load_model("resnet152_ddn_jpeg")
        model = Ensemble2(model1, model2)

    else:
        print("can not load model")

    return model


def save_checkpoint(state: OrderedDict, filename: str = 'checkpoint.pth', cpu: bool = False) -> None:
    if cpu:
        new_state = OrderedDict()
        for k in state.keys():
            newk = k.replace('module.', '')  # remove module. if model was trained using DataParallel
            new_state[newk] = state[k].cpu()
        state = new_state
    if torch.__version__ >= '1.6.0':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.counter = 0

    def append(self, val):
        self.values.append(val)
        self.counter += 1

    @property
    def val(self):
        return self.values[-1]

    @property
    def avg(self):
        return sum(self.values) / len(self.values)

    @property
    def last_avg(self):
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg


class NormalizedModel(nn.Module):
    """
    Wrapper for a model to account for the mean and std of a dataset.
    mean and std do not require grad as they should not be learned, but determined beforehand.
    mean and std should be broadcastable (see pytorch doc on broadcasting) with the data.
    Args:

        model (nn.Module): model to use to predict
        mean (torch.Tensor): sequence of means for each channel
        std (torch.Tensor): sequence of standard deviations for each channel
    """

    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
        super(NormalizedModel, self).__init__()

        self.model = model
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = (input - self.mean) / self.std
        return self.model(normalized_input)
    def forward_attention(self, input: torch.Tensor,input2) -> torch.Tensor:
        normalized_input = (input - self.mean) / self.std
        normalized_input2 = (input2 - self.mean) / self.std
        return self.model.forward_attention(normalized_input,normalized_input2)
    def feature_map(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = (input - self.mean) / self.std
        return self.model.feature_map(normalized_input)
    def feature_map2(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = (input - self.mean) / self.std
        return self.model.feature_map2(normalized_input)



def requires_grad_(model:nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def squared_l2_norm(x: torch.Tensor) -> torch.Tensor:
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return squared_l2_norm(x).sqrt()

if __name__ == '__main__':
    model = load_model("ecaresnet269d")
    # load_model("nfnet_f7")
    total = sum([param.nelement() for param in model.parameters()])
    print(total)