from transformers import DetrForObjectDetection
from timm import create_model
import sys
sys.path.append('/home/andrii/mmpie1/')
from mmpie1.data.MMPieDM import MMPieDataModule
from mmpie1.models.detr import Encoder, DetrSinePositionEmbedding, DetrAttention, DetrEncoder
from omegaconf import OmegaConf
import torch
from torch import nn
from PIL import Image
import requests


# Load the Hugging Face model
hf_detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
hf_detr.eval()

# Extract pretrained weights
hf_pretrained_encoder = hf_detr.model.encoder.state_dict()
hf_pretrained_backbone = hf_detr.model.backbone.conv_encoder.state_dict()
hf_pretrained_projection = hf_detr.model.input_projection.state_dict()
hf_pretrained_position = hf_detr.model.backbone.position_embedding.state_dict()

# Define your model
embedded_dimension = 256

# Define encoder, position embedding, and input projection classes if not defined
# Assuming Encoder, DetrSinePositionEmbedding, and DetrEncoder are defined as shown previously

encoder = Encoder(Backbone='resnet50', Pretrained=True, FreezeLayers=['layer2', 'layer3', 'layer4'], OutIndecies=[1, 2, 3, 4])
position = DetrSinePositionEmbedding(embedding_dim=embedded_dimension / 2, temperature=10000, normalize=True, scale=None)
input_projection = nn.Conv2d(encoder.intermediate_channel_sizes[-1], embedded_dimension, kernel_size=1)
transformer_encoder = DetrEncoder(num_encoder_layers=6, hidden_size=embedded_dimension, num_attention_heads=8, dropout_rate=0, attention_dropout=0)

# Ensure models are in evaluation mode before loading weights
encoder.eval()
position.eval()
input_projection.eval()
transformer_encoder.eval()
def print_specific_weights(model, weight_name):
    weights = model.state_dict()[weight_name]
    print(f"{weight_name} weights: {weights}")

# Print specific weights before loading
print("Weights before loading:")
print_specific_weights(transformer_encoder, 'layers.0.self_attn.k_proj.weight')
print_specific_weights(transformer_encoder, 'layers.0.self_attn.k_proj.bias')

# Load pretrained weights into your model
transformer_encoder.load_state_dict(hf_pretrained_encoder, strict=True)
encoder.load_state_dict(hf_pretrained_backbone, strict=True)
input_projection.load_state_dict(hf_pretrained_projection, strict=True)
position.load_state_dict(hf_pretrained_position, strict=True)

# Print specific weights after loading
print("Weights after loading:")
print_specific_weights(transformer_encoder, 'layers.0.self_attn.k_proj.weight')
print_specific_weights(transformer_encoder, 'layers.0.self_attn.k_proj.bias')

# Compare weights immediately after loading
def compare_weights(your_model, hf_model):
    your_state_dict = your_model.state_dict()
    hf_state_dict = hf_model.state_dict()

    for name, param in your_state_dict.items():
        if name in hf_state_dict:
            hf_param = hf_state_dict[name]
            if param.shape != hf_param.shape:
                print(f"Shape mismatch at {name}: {param.shape} vs {hf_param.shape}")
            else:
                diff = torch.abs(param - hf_param)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()
                print(f"{name} - Max difference: {max_diff}, Mean difference: {mean_diff}")
        else:
            print(f"{name} not found in Hugging Face model")

# Compare the encoder weights
print("Comparing encoder weights...")
compare_weights(transformer_encoder, hf_detr.model.encoder)

# Compare the backbone weights
print("Comparing backbone weights...")
compare_weights(encoder, hf_detr.model.backbone.conv_encoder)

# Compare the input projection weights
print("Comparing input projection weights...")
compare_weights(input_projection, hf_detr.model.input_projection)

# Compare the position embedding weights
print("Comparing position embedding weights...")
compare_weights(position, hf_detr.model.backbone.position_embedding)