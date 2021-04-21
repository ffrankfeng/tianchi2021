import os
import shutil
from typing import Optional
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lpips
import sys
sys.path.append("..")



class M_DI2_FGSM_Attacker:
    def __init__(self,
                 steps: int,
                 quantize: bool = True,
                 levels: int = 256,
                 max_norm: Optional[float] = None,
                 div_prob: float = 0.9,
                 loss_amp: float = 100.0,
                 low_bound: int = 224,
                 momentum: int = 1,
                 return_delta: bool = False,
                 device: torch.device = torch.device('cpu')) -> None:

        self.steps = steps
        self.need_lpiprecover = False
        self.need_gaussianBlur = False
        self.need_fid = False
        self.quantize = quantize
        self.levels = levels
        self.max_norm = max_norm
        self.div_prob = div_prob
        self.loss_amp = loss_amp
        self.low = low_bound
        self.momentum = momentum
        self.return_delta = return_delta
        self.device = device


    def input_diversity(self, image, low=270):
        high = 299
        if random.random() > self.div_prob:
            return image
        rnd = random.randint(low, high)
        rescaled = F.interpolate(image, size=[rnd, rnd], mode='bilinear')
        h_rem = high - rnd
        w_rem = high - rnd
        pad_top = random.randint(0, h_rem)
        pad_bottom = h_rem - pad_top
        pad_left = random.randint(0, w_rem)
        pad_right = w_rem - pad_left
        padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)
        return padded

    def psnr(self, delta):
        mse = torch.mean((delta) ** 2)
        if mse < 1.0e-10:
            return torch.Tensor([100]).cuda()
        PIXEL_MAX = 1
        return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


    def attack(self,
               model: nn.Module,
               inputs: torch.Tensor,
               labels_true: torch.Tensor) -> torch.Tensor:
        # gaussian = GaussianBlurConv(channels=3).to(DEVICE)
        batch_size = inputs.shape[0]
        delta = torch.zeros_like(inputs, requires_grad=True)

        # setup optimizer
        optimizer = optim.SGD([delta], lr=1, momentum=self.momentum)

        # for choosing best results
        best_loss = 1e4 * torch.ones(inputs.size(0), dtype=torch.float, device=self.device)
        best_delta = torch.zeros_like(inputs)
        for step in range(self.steps):
            if self.max_norm:
                delta.data.clamp_(-self.max_norm, self.max_norm)
                if self.quantize:
                    delta.data.mul_(self.levels - 1).round_().div_(self.levels - 1)

            adv = inputs + delta
            div_adv = self.input_diversity(adv, low=self.low)
            logits = model(div_adv)

            ce_loss_true = F.cross_entropy(logits, labels_true, reduction='none')
            # ce_loss_target = F.cross_entropy(logits, labels_target, reduction='none')
            loss = self.loss_amp - ce_loss_true
            # if self.loss_type == 'psnr':
            #     psnrloss = self.psnr(delta)
            #     loss -= 0.05*psnrloss
                # if (step+1)%5 ==0:
                # print("step:", str(step),": loss = ",str(torch.mean(loss).item()),": lpipscore = ",str(psnrloss.item()),": ce_loss_true = ",str(torch.mean(ce_loss_true).item()))
            # if self.need_fid == True:
            #     fid = calculate_fid_given_paths(adv, inputs, DEVICE, 2048)
            #     loss += 0.5*fid
                # print(str(step)," fid:",str(fid.item()), "ce_loss_true",str(torch.mean(ce_loss_true).item()))
            # loss += max(lpipscore,0.2)*10*self.loss_amp
            is_better = loss < best_loss

            best_loss[is_better] = loss[is_better]
            best_delta[is_better] = delta.data[is_better]

            loss = torch.mean(loss)
            optimizer.zero_grad()
            loss.backward()
            # renorm gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            # if self.need_gaussianBlur:
            #     delta.data = gaussian(delta.data)
            optimizer.step()

            # avoid out of bound
            delta.data.add_(inputs)
            delta.data.clamp_(0, 1).sub_(inputs)
        if self.return_delta :
            return best_delta
        else:
            advs = inputs + best_delta
            return advs

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
