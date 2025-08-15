import torch
from torch import nn
from ldm.models.attention.models import Decoder
from ldm.util import instantiate_from_config
from ldm.util import rearrange


class Correlation(nn.Module):
    def __init__(self, k, corr_confg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = torch.hub.load(
            "/mnt/pfs/asdfe1/model/facebookresearch_dinov2_main",
            "dinov2_vits14",
            source="local",
            trust_repo=True,
        )
        self.backbone.require_grad_(False)
        self.backbone.eval()
        self.correlation_block: Decoder = instantiate_from_config(corr_confg)
        self.proj_out = nn.Linear(instantiate_from_config["param"]["d_model"], k * k)
        self.patch_size = 14

    def forward(self, input, reference, input_mask=None, reference_mask=None):
        b, c, h, w = input.shape
        input_feature_dict = self.backbone.forward_features(input)
        input_feature = input_feature_dict["x_norm_patchtokens"].detach()
        reference_feature_dict = self.backbone.forward_features(reference)
        reference_feature = reference_feature_dict["x_norm_patchtokens"].detach()
        corr_map = self.correlation_block(
            input_feature, input_mask, reference_feature, reference_mask
        )
        corr_map = self.proj_out(corr_map)
        corr_map = corr_map.softmax(dim=-1)
        corr_map = rearrange(
            corr_map,
            "b (h w) (k1 k2) -> b h w k1 k2",
            h=h // self.patch_size,
            w=w // self.patch_size,
            k1=self.k,
            k2=self.k,
        )
        return corr_map

    def loss(self, input, reference, gt):
        corr_map = self(input, reference)
        b, h, w, k1, k2 = corr_map.shape
        corr_grid_x = torch.arange(k1).view(1, 1, 1, k1, 1).expand(b, h, w, k1, k2).float()
        corr_grid_y = torch.arange(k2).view(1, 1, 1, 1, k2).expand(b, h, w, k1, k2).float()
        
        corr_pos_x = torch.sum(corr_map * corr_grid_x, dim=(-1,-2)).unsqueeze(-1)
        corr_pos_y = torch.sum(corr_map * corr_grid_y, dim=(-1,-2)).unsqueeze(-1)
        corr_pos = torch.cat([corr_pos_x, corr_pos_y], dim=-1)
        loss = nn.functional.mse_loss(corr_pos, gt)
        return loss
        
        
        
        
