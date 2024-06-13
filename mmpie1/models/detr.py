import math 
import torch
from torch import nn, Tensor
from timm import create_model
from typing import Dict, List, Optional, Tuple, Union
from mmpie1.models.layers import DetrEncoderLayer, DetrDecoderLayer,  DetrFrozenBatchNorm2d, DetrMHAttentionMap
from mmpie1.models.utils import DetrLearnedPositionEmbedding, DetrSinePositionEmbedding, expand_mask



# def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, target_len: Optional[int] = None):
#     """
#     Expands attention_mask from `[batch_size, seq_len]` to `[batch_size, 1, target_seq_len, source_seq_len]`.
#     """
#     batch_size, source_len = mask.size()
#     target_len = target_len if target_len is not None else source_len

#     expanded_mask = mask[:, None, None, :].expand(batch_size, 1, target_len, source_len).to(dtype)

#     inverted_mask = 1.0 - expanded_mask

#     return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)




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




class DetrEncoder(nn.Module):
    def __init__(self, num_layers: int, 
                 embedded_dimension: int, 
                 num_attention_heads: int,
                 ffn_dim: int = 2048, 
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
        self.layers = nn.ModuleList([DetrEncoderLayer(embedded_dimension, num_attention_heads, attention_dropout, dropout_rate, activation_dropout, ffn_dim) for _ in range(num_layers)])

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
            attention_mask = expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            dropout_probability = torch.rand([])
            if self.training and dropout_probability < self.layerdrop:  # skip the layer
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(hidden_states, attention_mask, object_queries, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if return_dict:
            return {'last_hidden_state': hidden_states, 'hidden_states': encoder_states, 'attentions': all_attentions}
        return hidden_states, encoder_states, all_attentions

class DetrDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for DETR:

    - object_queries and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: DetrConfig
    """
    
    def __init__(self, 
                 num_layers: int, 
                 embedded_dimension: int, 
                 num_attention_heads: int,
                 layerdrop: float = 0.0,
                 attention_dropout_rate: float = 0.0,
                 activation_dropout_rate: float = 0.0, 
                 dropout_rate: float = 0.1,
                 ffn_dim: int = 2048,
                 aux_loss: bool = False,
                 init_std: float = 0.02,
                 xavier_std: float = 1e-4):
        super().__init__()
        self.dropout = dropout_rate
        self.layerdrop = layerdrop
        self.aux_loss = aux_loss
        self.xavier_std = xavier_std
        self.init_std = init_std

        self.layers = nn.ModuleList([DetrDecoderLayer(embedded_dimension, num_attention_heads, attention_dropout_rate, dropout_rate, activation_dropout_rate, ffn_dim) 
                                     for _ in range(num_layers)])

        self.layernorm = nn.LayerNorm(embedded_dimension)

        self.gradient_checkpointing = False
        
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

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        object_queries=None,
        query_position_embeddings=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The query embeddings that are passed into the decoder.

            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on certain queries. Mask values selected in `[0, 1]`:

                - 1 for queries that are **not masked**,
                - 0 for queries that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            object_queries (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Object queries that are added to the queries and keys in each cross-attention layer.
            query_position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                , *optional*): Position embeddings that are added to the values and keys in each self-attention layer.

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
            input_shape = inputs_embeds.size()[:-1]



        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
            encoder_attention_mask = expand_mask(
                encoder_attention_mask, inputs_embeds.dtype, target_len=input_shape[-1]
            )

        # optional intermediate hidden states
        intermediate = () if self.aux_loss else None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                object_queries=object_queries,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if self.aux_loss:
                hidden_states = self.layernorm(hidden_states)
                intermediate += (hidden_states,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # finally, apply layernorm
        hidden_states = self.layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # stack intermediate decoder activations
        if self.aux_loss:
            intermediate = torch.stack(intermediate)
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions, intermediate]
                if v is not None
            )
        else:
            if self.aux_loss:
                return {'last_hidden_state': hidden_states, 'hidden_states': all_hidden_states, 'attentions': all_self_attns, 'cross_attentions': all_cross_attentions, 'intermediate_hidden_states': intermediate}
            else:
                res_dict = {'last_hidden_state': hidden_states, 'hidden_states': all_hidden_states, 'attentions': all_self_attns, 'cross_attentions': all_cross_attentions}

            return res_dict




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
        self.query_position_embeddings = nn.Embedding(cfg.num_queries,self.embedded_dimension)
        self.TransformerEncoder = DetrEncoder(num_layers=cfg.TransformerEncoder.num_layers, 
                                                embedded_dimension=self.embedded_dimension, 
                                                num_attention_heads=cfg.TransformerEncoder.num_attention_heads,
                                                ffn_dim=cfg.TransformerEncoder.ffn_dimension,
                                                dropout_rate=cfg.TransformerEncoder.dropout_rate,
                                                attention_dropout=cfg.TransformerEncoder.attention_dropout_rate,
                                                activation_dropout=cfg.TransformerEncoder.activation_dropout_rate,
                                                layerdrop=cfg.TransformerEncoder.layer_dropout_rate, 
                                                init_std=cfg.TransformerEncoder.init_std,
                                                xavier_std=cfg.TransformerEncoder.xavier_std)


        self.TransformerDecoder = DetrDecoder(num_layers=cfg.TransformerDecoder.num_layers,
                                                embedded_dimension=self.embedded_dimension,
                                                num_attention_heads=cfg.TransformerDecoder.num_attention_heads,
                                                ffn_dim = cfg.TransformerDecoder.ffn_dimension,
                                                dropout_rate=cfg.TransformerDecoder.dropout_rate,
                                                attention_dropout_rate=cfg.TransformerDecoder.attention_dropout_rate,
                                                activation_dropout_rate=cfg.TransformerDecoder.activation_dropout_rate,
                                                layerdrop=cfg.TransformerDecoder.layer_dropout_rate,
                                                init_std=cfg.TransformerEncoder.init_std,
                                                xavier_std=cfg.TransformerEncoder.xavier_std
                                                )

    def load_from_hf(self, hf_model):

        hf_pretrained_encoder = hf_model.model.encoder.state_dict()
        hf_pretrained_backbone = hf_model.model.backbone.conv_encoder.state_dict()
        hf_pretrained_projection = hf_model.model.input_projection.state_dict()
        hf_pretrained_position = hf_model.model.backbone.position_embedding.state_dict()
        hf_pretrained_decoder = hf_model.model.decoder.state_dict()


        if self.query_position_embeddings.num_embeddings ==  hf_model.model.query_position_embeddings.num_embeddings:
            hf_pretrained_query_position_embeddings = hf_model.model.query_position_embeddings.state_dict()
            res_q_emb = self.query_position_embeddings.load_state_dict(hf_pretrained_query_position_embeddings, strict=True)

        else:
            print(f'Query Position Embeddings: {self.query_position_embeddings.num_embeddings} vs {hf_model.model.query_position_embeddings.num_embeddings}')
            res_q_emb = 'Different shape'
        res_enc = self.Encoder.load_state_dict(hf_pretrained_backbone, strict=True)
        res_pos_enc = self.PosEncoding.load_state_dict(hf_pretrained_position, strict=True)
        res_inp_proj = self.InputProjection.load_state_dict(hf_pretrained_projection, strict=True)
        res_trans_enc = self.TransformerEncoder.load_state_dict(hf_pretrained_encoder, strict=True)
        res_trans_dec = self.TransformerDecoder.load_state_dict(hf_pretrained_decoder, strict=True)


        print(f'Encoder: {res_enc}')
        print(f'Positional Encoding: {res_pos_enc}')
        print(f'Input Projection: {res_inp_proj}')
        print(f'Transformer Encpder: {res_trans_enc}')
        print(f'Transformer decoder: {res_trans_dec}')
        print(f'Loadded query pos embedding to encoder: {res_q_emb}')


    def forward(self, pixel_values, pixel_mask = None, 
                output_attentions = False, 
                output_hidden_states = False, 
                return_dict = False):
        
        batch_size, num_channels, height, width = pixel_values.shape

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
        

        query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        queries = torch.zeros_like(query_position_embeddings)
        
        if return_dict:
            encoder_output_data = encoder_outputs['last_hidden_state']
        else:
            encoder_output_data = encoder_outputs[0]

        decoder_outputs = self.TransformerDecoder(
            inputs_embeds=queries,
            attention_mask=None,
            object_queries=object_queries,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_output_data,
            encoder_attention_mask=flattened_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )
        # decoder_outputs = None
        return encoder_outputs, decoder_outputs, cd


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