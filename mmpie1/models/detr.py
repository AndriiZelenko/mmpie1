import math 
import torch
from torch import nn, Tensor
from timm import create_model
from typing import Dict, List, Optional, Tuple, Union
from mmpie1.models.layers import DetrEncoderLayer, DetrFrozenBatchNorm2d, DetrMHAttentionMap
from mmpie1.models.utils import DetrLearnedPositionEmbedding
from mmpie1.models.utils.utils import DetrSinePositionEmbedding


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, target_len: Optional[int] = None):
    """
    Expands attention_mask from `[batch_size, seq_len]` to `[batch_size, 1, target_seq_len, source_seq_len]`.
    """
    batch_size, source_len = mask.size()
    target_len = target_len if target_len is not None else source_len

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, target_len, source_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

def replace_batch_norm(model):
    r"""
    Recursively replace all `torch.nn.BatchNorm2d` with `DetrFrozenBatchNorm2d`.

    Args:
        model (torch.nn.Module):
            input model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_module = DetrFrozenBatchNorm2d(module.num_features)

            new_module.weight.data.copy_(module.weight)
            new_module.bias.data.copy_(module.bias)
            new_module.running_mean.data.copy_(module.running_mean)
            new_module.running_var.data.copy_(module.running_var)

            model._modules[name] = new_module

        if len(list(module.children())) > 0:
            replace_batch_norm(module)


# class DetrFrozenBatchNorm2d(nn.Module):
#     """
#     BatchNorm2d where the batch statistics and the affine parameters are fixed.

#     Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
#     torchvision.models.resnet[18,34,50,101] produce nans.
#     """

#     def __init__(self, n):
#         super().__init__()
#         self.register_buffer("weight", torch.ones(n))
#         self.register_buffer("bias", torch.zeros(n))
#         self.register_buffer("running_mean", torch.zeros(n))
#         self.register_buffer("running_var", torch.ones(n))

#     def _load_from_state_dict(
#         self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
#     ):
#         num_batches_tracked_key = prefix + "num_batches_tracked"
#         if num_batches_tracked_key in state_dict:
#             del state_dict[num_batches_tracked_key]

#         super()._load_from_state_dict(
#             state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
#         )

#     def forward(self, x):
#         # move reshapes to the beginning
#         # to make it user-friendly
#         weight = self.weight.reshape(1, -1, 1, 1)
#         bias = self.bias.reshape(1, -1, 1, 1)
#         running_var = self.running_var.reshape(1, -1, 1, 1)
#         running_mean = self.running_mean.reshape(1, -1, 1, 1)
#         epsilon = 1e-5
#         scale = weight * (running_var + epsilon).rsqrt()
#         bias = bias - running_mean * scale
#         return x * scale + bias




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
        replace_batch_norm(self.model)
        self.freeze_layers(FreezeLayers)
        self.intermediate_channel_sizes = self.model.feature_info.channels() 
        


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




# class DetrSinePositionEmbedding(nn.Module):
#     """
#     This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
#     need paper, generalized to work on images.
#     """

#     def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.temperature = temperature
#         self.normalize = normalize
#         if scale is not None and normalize is False:
#             raise ValueError("normalize should be True if scale is passed")
#         if scale is None:
#             scale = 2 * math.pi
#         self.scale = scale

#     def forward(self, pixel_values, pixel_mask):
#         if pixel_mask is None:
#             raise ValueError("No pixel mask provided")
#         y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
#         x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
#         if self.normalize:
#             y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
#             x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

#         dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
#         dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

#         pos_x = x_embed[:, :, :, None] / dim_t
#         pos_y = y_embed[:, :, :, None] / dim_t
#         pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
#         pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
#         pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
#         return pos
    

    
# class DetrLearnedPositionEmbedding(nn.Module):
#     """
#     This module learns positional embeddings up to a fixed maximum size.
#     """

#     def __init__(self, embedding_dim=256):
#         super().__init__()
#         self.row_embeddings = nn.Embedding(50, embedding_dim)
#         self.column_embeddings = nn.Embedding(50, embedding_dim)

#     def forward(self, pixel_values, pixel_mask=None):
#         height, width = pixel_values.shape[-2:]
#         width_values = torch.arange(width, device=pixel_values.device)
#         height_values = torch.arange(height, device=pixel_values.device)
#         x_emb = self.column_embeddings(width_values)
#         y_emb = self.row_embeddings(height_values)
#         pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
#         pos = pos.permute(2, 0, 1)
#         pos = pos.unsqueeze(0)
#         pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
#         return pos
    


# class DetrAttention(nn.Module):
#     """
#     Multi-headed attention from 'Attention Is All You Need' paper.

#     Here, we add position embeddings to the queries and keys (as explained in the DETR paper).
#     """

#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#         dropout: float = 0.0,
#         bias: bool = True,
#     ):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
#         self.scaling = self.head_dim**-0.5

#         self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

#     def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
#         return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

#     def with_pos_embed(self, tensor: torch.Tensor, object_queries: Optional[Tensor], **kwargs):
#         position_embeddings = kwargs.pop("position_embeddings", None)

#         if position_embeddings is not None:
#             object_queries = position_embeddings

#         return tensor if object_queries is None else tensor + object_queries

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         object_queries: Optional[torch.Tensor] = None,
#         key_value_states: Optional[torch.Tensor] = None,
#         spatial_position_embeddings: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#         **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         """Input shape: Batch x Time x Channel"""

#         position_embeddings = kwargs.pop("position_ebmeddings", None)
#         key_value_position_embeddings = kwargs.pop("key_value_position_embeddings", None)

#         if position_embeddings is not None:
#             object_queries = position_embeddings

#         if key_value_position_embeddings is not None:
#             spatial_position_embeddings = key_value_position_embeddings

#         # if key_value_states are provided this layer is used as a cross-attention layer
#         # for the decoder
#         is_cross_attention = key_value_states is not None
#         batch_size, target_len, embed_dim = hidden_states.size()

#         # add position embeddings to the hidden states before projecting to queries and keys
#         if object_queries is not None:
#             hidden_states_original = hidden_states
#             hidden_states = self.with_pos_embed(hidden_states, object_queries)

#         # add key-value position embeddings to the key value states
#         if spatial_position_embeddings is not None:
#             key_value_states_original = key_value_states
#             key_value_states = self.with_pos_embed(key_value_states, spatial_position_embeddings)

#         # get query proj
#         query_states = self.q_proj(hidden_states) * self.scaling
#         # get key, value proj
#         if is_cross_attention:
#             # cross_attentions
#             key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
#             value_states = self._shape(self.v_proj(key_value_states_original), -1, batch_size)
#         else:
#             # self_attention
#             key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
#             value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)

#         proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
#         query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
#         key_states = key_states.view(*proj_shape)
#         value_states = value_states.view(*proj_shape)

#         source_len = key_states.size(1)

#         attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))


#         if attention_mask is not None:
#             if attention_mask.size() != (batch_size, 1, target_len, source_len):
#                 raise ValueError(
#                     f"Attention mask should be of size {(batch_size, 1, target_len, source_len)}, but is"
#                     f" {attention_mask.size()}"
#                 )
#             attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, source_len) + attention_mask
#             attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, source_len)

#         attn_weights = nn.functional.softmax(attn_weights, dim=-1)

#         if output_attentions:
#             # this operation is a bit awkward, but it's required to
#             # make sure that attn_weights keeps its gradient.
#             # In order to do so, attn_weights have to reshaped
#             # twice and have to be reused in the following
#             attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
#             attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
#         else:
#             attn_weights_reshaped = None

#         attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

#         attn_output = torch.bmm(attn_probs, value_states)

#         if attn_output.size() != (batch_size * self.num_heads, target_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(batch_size, self.num_heads, target_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )

#         attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
#         attn_output = attn_output.transpose(1, 2)
#         attn_output = attn_output.reshape(batch_size, target_len, embed_dim)

#         attn_output = self.out_proj(attn_output)

#         return attn_output, attn_weights_reshaped


# class DetrEncoderLayer(nn.Module):
#     def __init__(self, embedded_dimension, 
#                  encoder_attention_heads, 
#                  attention_dropout, 
#                  dropout, activation_dropout, encoder_ffn_dim):
#         super().__init__()
#         self.embed_dim = embedded_dimension
#         self.self_attn = DetrAttention(
#             embed_dim=self.embed_dim,
#             num_heads=encoder_attention_heads,
#             dropout=attention_dropout,
#         )
#         self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
#         self.dropout = dropout
#         self.activation_fn = nn.ReLU()
#         self.activation_dropout = activation_dropout
#         self.fc1 = nn.Linear(self.embed_dim, encoder_ffn_dim)
#         self.fc2 = nn.Linear(encoder_ffn_dim, self.embed_dim)
#         self.final_layer_norm = nn.LayerNorm(self.embed_dim)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: torch.Tensor,
#         object_queries: torch.Tensor = None,
#         output_attentions: bool = False,
#         **kwargs,
#     ):
#         """
#         Args:
#             hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
#             attention_mask (`torch.FloatTensor`): attention mask of size
#                 `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
#                 values.
#             object_queries (`torch.FloatTensor`, *optional*):
#                 Object queries (also called content embeddings), to be added to the hidden states.
#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#         """
#         position_embeddings = kwargs.pop("position_embeddings", None)


#         if position_embeddings is not None:

#             object_queries = position_embeddings

#         residual = hidden_states
#         hidden_states, attn_weights = self.self_attn(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             object_queries=object_queries,
#             output_attentions=output_attentions,
#         )

#         hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
#         hidden_states = residual + hidden_states
#         hidden_states = self.self_attn_layer_norm(hidden_states)

#         residual = hidden_states
#         hidden_states = self.activation_fn(self.fc1(hidden_states))
#         hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

#         hidden_states = self.fc2(hidden_states)
#         hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

#         hidden_states = residual + hidden_states
#         hidden_states = self.final_layer_norm(hidden_states)

#         if self.training:
#             if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
#                 clamp_value = torch.finfo(hidden_states.dtype).max - 1000
#                 hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

#         outputs = (hidden_states,)

#         if output_attentions:
#             outputs += (attn_weights,)

#         return outputs
    
# class DetrMHAttentionMap(nn.Module):
#     """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

#     def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True, std=None):
#         super().__init__()
#         self.num_heads = num_heads
#         self.hidden_dim = hidden_dim
#         self.dropout = nn.Dropout(dropout)

#         self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
#         self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

#         self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

#     def forward(self, q, k, mask: Optional[Tensor] = None):
#         q = self.q_linear(q)
#         k = nn.functional.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
#         queries_per_head = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
#         keys_per_head = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
#         weights = torch.einsum("bqnc,bnchw->bqnhw", queries_per_head * self.normalize_fact, keys_per_head)

#         if mask is not None:
#             weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), torch.finfo(weights.dtype).min)
#         weights = nn.functional.softmax(weights.flatten(2), dim=-1).view(weights.size())
#         weights = self.dropout(weights)
#         return weights


class DetrEncoder(nn.Module):
    def __init__(self, num_encoder_layers: int, 
                 hidden_size: int, 
                 num_attention_heads: int,
                 encoder_ffn_dim: int = 2048, 
                 dropout_rate: float = 0.1, 
                 attention_dropout: float = 0.1, 
                 activation_dropout: float = 0.1, 
                 layerdrop: float = 0.0, 
                 init_std: float = 0.02, 
                 xavier_std: float = 1e-4):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.layerdrop = layerdrop
        self.init_std = init_std
        self.xavier_std = xavier_std

        # encoder_layer = DetrEncoderLayer(hidden_size, num_attention_heads, attention_dropout, dropout_rate, activation_dropout, encoder_ffn_dim)
        self.layers = nn.ModuleList([DetrEncoderLayer(hidden_size, num_attention_heads, attention_dropout, dropout_rate, activation_dropout, encoder_ffn_dim) for _ in range(num_encoder_layers)])

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, DetrMHAttentionMap):
                nn.init.zeros_(module.k_linear.bias)
                nn.init.zeros_(module.q_linear.bias)
                nn.init.xavier_uniform_(module.k_linear.weight, gain=self.xavier_std)
                nn.init.xavier_uniform_(module.q_linear.weight, gain=self.xavier_std)
            elif isinstance(module, DetrLearnedPositionEmbedding):
                nn.init.uniform_(module.row_embeddings.weight)
                nn.init.uniform_(module.column_embeddings.weight)
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
                
                module.weight.data.normal_(mean=0.0, std=self.init_std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.init_std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

   

    def forward(self, inputs_embeds, 
                attention_mask=None, 
                object_queries=None, 
                output_attentions=False, 
                output_hidden_states=False,
                return_dict = True):
        hidden_states = self.dropout(inputs_embeds)

        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = []
        all_attentions = []

        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states.append(hidden_states)

            dropout_probability = torch.rand([])
            if self.training and dropout_probability < self.layerdrop:  # skip the layer
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(hidden_states, attention_mask, object_queries, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions.append(layer_outputs[1])

        if output_hidden_states:
            encoder_states.append(hidden_states)

        if not output_attentions:
            all_attentions = None
        if return_dict:
            return {'hidden_state': hidden_states, 'encoder_states': encoder_states, 'all_attentions': all_attentions}
        return hidden_states, encoder_states, all_attentions




class DetrDetection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedded_dimension = cfg.TransformerEncoder.embedded_dimension
        self.Encoder = Encoder(Backbone = cfg.FeatureExtractorEncoder.Backbone,
                               Pretrained = cfg.FeatureExtractorEncoder.Pretrained,
                               FreezeLayers = cfg.FeatureExtractorEncoder.FreezeLayers,
                               OutIndecies = cfg.FeatureExtractorEncoder.OutIndecies)
        if cfg.TransformerEncoder.pos_encoding_type == 'sine':
            self.PosEncoding = DetrSinePositionEmbedding(embedding_dim=self.embedded_dimension/ 2, 
                                                          temperature=cfg.TransformerEncoder.temperature, 
                                                          normalize=True, 
                                                          scale=None)
        elif cfg.TransformerEncoder.pos_encoding_type == 'learned':
            self.PosEncoding = DetrLearnedPositionEmbedding(embedding_dim=self.embedded_dimension)
        else:
            raise ValueError(f"Unknown positional encoding type: {cfg.TransformerEncoder.pos_encoding_type}")
        
        self.InputProjection = nn.Conv2d(self.Encoder.intermediate_channel_sizes[-1], self.embedded_dimension, kernel_size=1)
        self.TransformerEncoder = DetrEncoder(num_encoder_layers=cfg.TransformerEncoder.num_encoder_layers, 
                                                hidden_size=self.embedded_dimension, 
                                                num_attention_heads=cfg.TransformerEncoder.num_attention_heads,
                                                encoder_ffn_dim=cfg.TransformerEncoder.encoder_ffn_dimension,
                                                dropout_rate=cfg.TransformerEncoder.dropout_rate,
                                                attention_dropout=cfg.TransformerEncoder.attention_dropout_rate,
                                                activation_dropout=cfg.TransformerEncoder.activation_droput_rate,
                                                layerdrop=cfg.TransformerEncoder.layer_dropout_rate, 
                                                init_std=cfg.TransformerEncoder.init_std,
                                                xavier_std=cfg.TransformerEncoder.xavier_std)


    def load_from_hf(self, hf_model):

        hf_pretrained_encoder = hf_model.model.encoder.state_dict()
        hf_pretrained_backbone = hf_model.model.backbone.conv_encoder.state_dict()
        hf_pretrained_projection = hf_model.model.input_projection.state_dict()
        hf_pretrained_position = hf_model.model.backbone.position_embedding.state_dict()

        res_enc = self.Encoder.load_state_dict(hf_pretrained_backbone, strict=True)
        res_pos_enc = self.PosEncoding.load_state_dict(hf_pretrained_position, strict=True)
        res_inp_proj = self.InputProjection.load_state_dict(hf_pretrained_projection, strict=True)
        res_trans_enc = self.TransformerEncoder.load_state_dict(hf_pretrained_encoder, strict=True)

        print(f'Encoder: {res_enc}')
        print(f'Positional Encoding: {res_pos_enc}')
        print(f'Input Projection: {res_inp_proj}')
        print(f'Transformer Encpder: {res_trans_enc}')


    def forward(self, pixel_values, pixel_mask = None, 
                output_attentions = False, 
                output_hidden_states = False, 
                return_dict = False):
        
        if pixel_mask is None:
            pixel_mask = torch.ones(pixel_values.shape[:-1], dtype=torch.bool, device=pixel_values.device)

        features = self.Encoder(pixel_values, pixel_mask)
        object_queries_list = []
        for feature_map, mask in features:
            object_queries_list.append(self.PosEncoding(feature_map, mask).to(feature_map.dtype))
        
        feature_map, mask = features[-1]
        cd = {'feature_map': feature_map}
        projected_feature_map = self.InputProjection(feature_map)
        cd['projected_feature_map'] = projected_feature_map
        
        flattened_features = projected_feature_map.flatten(2).permute(0,2,1)
        object_queries = object_queries_list[-1].flatten(2).permute(0,2,1)
        cd['object_queries'] = object_queries
        flattened_mask = mask.flatten(1)

        encoder_outputs = self.TransformerEncoder(inputs_embeds = flattened_features, 
                                                    attention_mask = flattened_mask, 
                                                    object_queries = object_queries,
                                                    output_attentions = output_attentions,
                                                    output_hidden_states = output_hidden_states,
                                                    return_dict = return_dict)
        
        
        return encoder_outputs, cd


# inputs_embeds, 
# attention_mask=None, 
# object_queries=None, 
# output_attentions=False, 
# output_hidden_states=False,
# return_dict = True):


# def _expand_mask(mask, dtype):
#     batch_size, seq_len = mask.shape
#     mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
#     mask = mask.repeat(1, 1, seq_len, 1)  # [batch_size, 1, seq_len, seq_len]
#     mask = mask.view(batch_size, 1, seq_len, seq_len)
#     mask = mask.to(dtype=dtype)
#     return mask


if __name__ == '__main__':
    pass