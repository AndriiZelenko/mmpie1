import math
import torch
from torch import nn
from typing import Tuple, List
from timm import create_model




class Encoder(nn.Module):
    def __init__(self, 
                 Backbone: str = 'resnet50',
                 Pretrained: bool = True,
                 FreezeLayers: List[str]=['layer2', 'layer3', 'layer4'],
                 OutIndecies: List[int] = (1,2,3,4)
                 ):
        super().__init__()
        self.FreezeLayers = FreezeLayers
        self.model = create_model(Backbone, pretrained=Pretrained, out_indices=OutIndecies, features_only = True)
        self.freeze_layers(FreezeLayers)



    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        features = self.model(pixel_values)

        out = []
        for feature_map in features:
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append([feature_map, mask])

        return out

    def freeze_layers(self, layers):
        for name, parameter in self.model.named_parameters():
            if name in layers:
                parameter.requires_grad = False

    def unfreeze_layers(self, layers):
        for name, parameter in self.model.named_parameters():
            if name in layers:
                parameter.requires_grad = True




class DetrSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    
