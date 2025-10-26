from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import cv2
from typing import List, Dict, Tuple, Optional
import pandas as pd
import seaborn as sns
from load import parse_pascal_voc_xml, get_dataset_files
from dataset import create_combined_dataset, DryWallQADataset
from visualize import visualize_samples, detailed_sample_analysis, get_sample_indices, dataset_summary
from train import setup_model, setup_data_loaders, train_model, configure_trainable_parameters, get_training_config, train_one_epoch, evaluate

# Set up paths
BASE_DIR = Path("/Users/prashanthreddyduggirala/10x")
CRACKS_DATASET = BASE_DIR / "cracks.v1i.voc"
DRYWALL_DATASET = BASE_DIR / "Drywall-Join-Detect.v1i.voc"

print("Imports successful!")

# Load both datasets
print("Loading crack detection dataset...")
cracks_files = get_dataset_files(CRACKS_DATASET)

print("Loading drywall joint detection dataset...")
drywall_files = get_dataset_files(DRYWALL_DATASET)

# Create the combined dataset
print("Creating combined dataset...")
train_images, train_annotations, train_prompts, train_labels = create_combined_dataset(['train'], cracks_files, drywall_files)
val_images, val_annotations, val_prompts, val_labels = create_combined_dataset(['valid'], cracks_files, drywall_files)

print(f"Combined datasets created!")
print(f"   Total train samples: {len(train_images)}")
print(f"   Total validation samples: {len(val_images)}")


train_crack_count = train_labels.count('crack')
train_drywall_count = train_labels.count('drywall_joint')
val_crack_count = val_labels.count('crack')
val_drywall_count = val_labels.count('drywall_joint')


# Create dataset instance
train_dataset = DryWallQADataset(train_images, train_annotations, train_prompts)
print(f"\nTrain Dataset instance created with {len(train_dataset)} samples")

val_dataset = DryWallQADataset(val_images, val_annotations, val_prompts)
print(f"Validation Dataset instance created with {len(val_dataset)} samples")

model, processor, device = setup_model()

encoder_params_list, decoder_params_list = configure_trainable_parameters(model)

config = get_training_config(
    batch_size=16,
    encoder_lr=1e-6,
    decoder_lr=1e-4,
    num_epochs=10,
    save_every=None,
    eval_every=170,
    optim_weight_decay=1e-5,
    checkpoint_base_path="/Users/prashanthreddyduggirala/10x/checkpoints/"
)

train_loader, val_loader = setup_data_loaders(train_dataset, val_dataset, processor, config)

