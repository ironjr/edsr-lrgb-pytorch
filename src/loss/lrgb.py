import torch
import torch.nn as nn
import torch.nn.functional as F

from .autoencoder import AutoEncoder


class LRGBLoss(nn.Module):
    def __init__(
        self,
        checkpoint: str,
        loss_type: str = 'L1',
        ychan_only: bool = False,
        match_training_size: bool = False,
        remap: dict = None,
    ) -> None:
        super(LRGBLoss, self).__init__()
        self.loss_type = loss_type.lower()
        self.ychan_only = ychan_only
        self.match_training_size = match_training_size
        self.remap = remap

        #  coeffs = [65.738, 129.057, 25.064]
        #  coeffs = torch.Tensor(coeffs).view(1, 3, 1, 1) / 256.0
        #  self.register_buffer('coeffs', coeffs)

        self.encoder = AutoEncoder()
        self.encoder.load_state_dict(torch.load(checkpoint))
        for p in self.parameters():
            p.requires_grad = False

        if self.loss_type in ('l1', 'mae'):
            self.dist = F.l1_loss
        elif self.loss_type in ('l2', 'mse'):
            self.dist = F.mse_loss
        elif self.loss_type in ('huber', 'smoothl1', 'sl1'):
            self.dist = F.smooth_l1_loss
        else:
            raise ValueError(
                'Unknown loss type [{:s}] is detected'.format(loss_type))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(Loss={self.loss_type}, "
            f"Y={self.ychan_only}, Resize={self.match_training_size})"
        )

    @torch.cuda.amp.autocast(False)
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self.remap is not None:
            x = kwargs[self.remap['x']]
            y = kwargs[self.remap['y']]
        x_feat = self._get_features(x)
        with torch.no_grad():
            y_feat = self._get_features(y)
        return self.dist(x_feat, y_feat)

    def state_dict(self, *args, **kwargs):
        return {}

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        #  if self.match_training_size:
        #      x = F.interpolate(
        #          x, size=(256, 256), align_corners=False, mode='bilinear')
        #  if self.ychan_only and x.size(1) == 3:
        #      x = x.mul(self.coeffs).sum(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        return self.encoder.encode(x)
        #  if self.n_GPUs > 1:
        #      return P.data_parallel(self.model, x, range(self.n_GPUs))
        #  else:
        #      return self.model(x)


def instantiate(opt: dict, loss_opt: dict):
    kwargs = {
        #  'checkpoint': '../pretrained/model_latest.pt',
        'checkpoint': '../../LRGB/experiment/div2k_long/model/model_latest.pt',
        'loss_type': 'l1',
        'ychan_only': False,
        'match_training_size': False,
        'remap': None,
    }
    for k in kwargs.keys():
        if k in loss_opt:
            kwargs[k] = loss_opt[k]
    loss = LRGBLoss(**kwargs)
    return loss
