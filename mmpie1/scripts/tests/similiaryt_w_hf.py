from transformers import DetrForObjectDetection
from timm import create_model
import sys
sys.path.append('/home/andrii/mmpie1/')
from mmpie1.data.MMPieDM import MMPieDataModule
from mmpie1.models.detr import Encoder, DetrEncoder, DetrDecoder, DetrDetection
from mmpie1.models.utils import  DetrSinePositionEmbedding
from omegaconf import OmegaConf
import torch
from torch import nn
from PIL import Image
import requests
import torchvision.transforms as T




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

# encoder = Encoder(Backbone='resnet50', Pretrained=True, FreezeLayers=['layer2', 'layer3', 'layer4'], OutIndecies=[1, 2, 3, 4])
# position = DetrSinePositionEmbedding(embedding_dim=embedded_dimension / 2, temperature=10000, normalize=True, scale=None)
# input_projection = nn.Conv2d(encoder.intermediate_channel_sizes[-1], embedded_dimension, kernel_size=1)
# transformer_encoder = DetrEncoder(num_encoder_layers=6, hidden_size=embedded_dimension, num_attention_heads=8, dropout_rate=0, attention_dropout=0)
# transformer_encoder = DetrDecoder(num_decoder_layers=6, hidden_size=embedded_dimension, num_attention_heads=8, dropout_rate=0, attention_dropout=0)


import torch
torch.manual_seed(42)



model_cfg = OmegaConf.load("/home/andrii/mmpie1/mmpie1/configs/model/detr.yaml")
model = DetrDetection(model_cfg)


model.eval()
model.load_from_hf(hf_detr)


def print_specific_weights(model, weight_name):
    weights = model.state_dict()[weight_name]
    print(f"{weight_name} weights: {weights}")




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

def compare_tensors(tensor1, tensor2, name="tensor"):
    if tensor1.shape != tensor2.shape:
        print(f"{name} shape mismatch: {tensor1.shape} vs {tensor2.shape}")
        return

    diff = torch.abs(tensor1 - tensor2)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    print(f"{name} - Max difference: {max_diff}, Mean difference: {mean_diff}")


# Compare the encoder weights
print("Comparing backbone weights...")
compare_weights(model.Encoder, hf_detr.model.backbone.conv_encoder)
print('---')
print('\n'*5)

# Compare the backbone weights
print("Comparing encoder weights...")
compare_weights(model.TransformerEncoder, hf_detr.model.encoder)

print('---')
print('\n'*5)

# Compare the input projection weights
print("Comparing input projection weights...")
compare_weights(model.InputProjection ,  hf_detr.model.input_projection)

print('---')
print('\n'*5)

# Compare the position embedding weights
print("Comparing position embedding weights...")
compare_weights(model.PosEncoding, hf_detr.model.backbone.position_embedding)
print('---')
print('\n'*5)

print("Comaring decoder weights... ")
compare_weights(model.TransformerDecoder, hf_detr.model.decoder)


print('checking similiairty of outputs')

from transformers import DetrForObjectDetection
from timm import create_model
import sys
sys.path.append('/home/andrii/mmpie1/')
from mmpie1.data.MMPieDM import MMPieDataModule
# from mmpie1.models.detr import Encoder, DetrSinePositionEmbedding, DetrAttention, DetrEncoder
from mmpie1.models.detr import DetrDetection
from omegaconf import OmegaConf
import torch
from torch import nn
from PIL import Image
import requests


import torch
torch.manual_seed(42)

embedded_dimension = 256

data_cfg = OmegaConf.load("/home/andrii/mmpie1/mmpie1/configs/dataset/codetr_conf.yaml")
dm = MMPieDataModule(data_cfg)

model_cfg = OmegaConf.load("/home/andrii/mmpie1/mmpie1/configs/model/detr.yaml")
model = DetrDetection(model_cfg)
hf_detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

model.load_from_hf(hf_detr)

device = torch.device('cuda:1')
hf_detr.to(device)
model.to(device)
hf_detr.eval()
model.eval()

url = 'http://images.cocodataset.org/train2017/000000310645.jpg'
im = Image.open(requests.get(url, stream=True).raw)



transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


device = torch.device('cuda:1')
img = transform(im).unsqueeze(0).to(device)
img_ori = img.clone()

pixel_mask = torch.ones((1, img.shape[2],img.shape[3]), device=device)
pixel_mask_ori = pixel_mask.clone()

oute, outd,  cd = model(img, pixel_mask,    
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True)

hf_features, hf_object_queries = hf_detr.model.backbone(img_ori, pixel_mask_ori)
hf_feature_map, hf_mask = hf_features[-1]


hf_proj = hf_detr.model.input_projection(hf_feature_map)

hf_flattened_features = hf_proj.flatten(2).permute(0, 2, 1)
hf_object_queries = hf_object_queries[-1].flatten(2).permute(0, 2, 1)

hf_flattened_mask = hf_mask.flatten(1)

hf_encoder_outputs = hf_detr.model.encoder(
    inputs_embeds=hf_flattened_features,
    attention_mask=hf_flattened_mask,
    object_queries=hf_object_queries,
    output_attentions=True,
    output_hidden_states=True,
    return_dict=True,
)

query_position_embeddings = hf_detr.model.query_position_embeddings.weight.unsqueeze(0).repeat(1, 1, 1)
queries = torch.zeros_like(query_position_embeddings)
hf_decoder_outputs = hf_detr.model.decoder(           
            inputs_embeds=queries,
            attention_mask=None,
            object_queries=hf_object_queries,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=hf_encoder_outputs[0],
            encoder_attention_mask=hf_flattened_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )



for key in outd.keys():
    print('-'*20)
    if isinstance(outd[key], torch.Tensor):
        compare_tensors(outd[key], hf_decoder_outputs[key], key)
    else:
        for m , h in zip(outd[key], hf_decoder_outputs[key]):
            compare_tensors(m, h, key)








