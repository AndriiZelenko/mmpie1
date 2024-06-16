from transformers import DetrForObjectDetection
from timm import create_model
import sys
sys.path.append('/home/andrii/mmpie1/')
from mmpie1.data.MMPieDM import MMPieDataModule
# from mmpie1.models.detr import Encoder, DetrSinePositionEmbedding, DetrEncoder
from mmpie1.models.detr import DetrDetection, DetrTrainer
from mmpie1.models.utils import post_process_object_detection
from omegaconf import OmegaConf
import torch
from torch import nn
from PIL import Image
import requests
from mmpie1.models.utils import center_to_corners_format_torch
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch


torch.manual_seed(42)

embedded_dimension = 256

data_cfg = OmegaConf.load("/home/andrii/mmpie1/mmpie1/configs/dataset/codetr_conf.yaml")
dm = MMPieDataModule(data_cfg)

model_cfg = OmegaConf.load("/home/andrii/mmpie1/mmpie1/configs/model/detr.yaml")
model = DetrDetection(model_cfg)

lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-2


datamodule = MMPieDataModule(data_cfg)
model = DetrTrainer(model_cfg=model_cfg, 
                    lr=lr, lr_backbone=lr_backbone, weight_decay=weight_decay,
                    idx2class = dm.train_dataset.idx_2_class, preprocessor=datamodule.preprocessing,
                    )

wandb_logger = WandbLogger(project='mmpie_detr', log_model=True)

# Initialize the trainer
trainer = pl.Trainer(
    max_epochs=40,
    # fast_dev_run=True,  # Flag to quickly test the script
    devices=[1], accelerator="gpu", logger = wandb_logger
)

# Fit the model
trainer.fit(model, datamodule=datamodule)
