# Prompted Segmentation for Drywall QA: Implementation Guide

Based on your task requirements, you need to build a text-conditioned segmentation model for two specific tasks: segmenting cracks and segmenting taping areas in drywall images. Here's a comprehensive approach to tackle this problem:

## Recommended Approach: CLIPSeg Fine-tuning

**CLIPSeg** is the most suitable model for your task because it natively supports text-conditioned segmentation[1][2][3]. It combines a frozen CLIP encoder with a lightweight decoder that can generate binary segmentation masks from text prompts.

### Key Advantages:
- **Text-conditioned**: Designed specifically for prompted segmentation[2][4]
- **Lightweight**: Only 1.1M trainable parameters in the decoder[4]
- **Binary outputs**: Produces PNG masks with values {0,255} as required[1]
- **Pre-trained**: Available on Hugging Face with strong baselines[2][5]

## Implementation Strategy

### 1. Model Architecture

CLIPSeg uses a **frozen CLIP ViT-B/16** encoder with a compact decoder that's modulated by text embeddings using FiLM (Feature-wise Linear Modulation)[4]. The decoder processes visual features at 1/16 resolution and upsamples to full image size.

```python
# Basic CLIPSeg setup
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
```

### 2. Dataset Preparation

You'll need to process both Roboflow datasets:

Datasets are in Pascal VOC format



**Dataset 1 (Taping areas)**: located in Drywall-Join-Detect.v1i.voc folder

Map to prompts like:
- "segment taping area"
- "segment joint/tape" 
- "segment drywall seam"

**Dataset 2 (Cracks)**: located in cracks.v1i.voc folder

Map to prompts like:
- "segment crack"
- "segment wall crack"

### 3. Fine-tuning Approach

Since CLIPSeg uses a frozen CLIP encoder, you'll primarily fine-tune the decoder[2][6]:

```python
# Freeze CLIP encoder, only train decoder
for param in model.clip.parameters():
    param.requires_grad = False

# Keep decoder trainable
for param in model.decoder.parameters():
    param.requires_grad = True
```

### 4. Training Configuration

Based on successful implementations[4][6]:
- **Learning rate**: 1e-4 with AdamW optimizer
- **Batch size**: 8 (adjust based on GPU memory)
- **Image size**: 640 x 640 x 3
- **Loss function**: Binary cross-entropy for mask prediction
- **Augmentations**: Random scaling, cropping, color jittering

### 5. Alternative: EVF-SAM for Higher Performance

If you need state-of-the-art results, consider **EVF-SAM**[7], which achieved better performance than CLIPSeg by using early vision-language fusion with BEIT-3. However, it's more complex to implement and requires more computational resources.

## Implementation Steps

### Step 1: Environment Setup
use uv
```bash
uv init
uv add transformers torch torchvision datasets pillow
```

### Step 2: Data Loading
```python
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class DryWallDataset(Dataset):
    def __init__(self, images, masks, prompts, processor):
        self.images = images
        self.masks = masks  
        self.prompts = prompts
        self.processor = processor
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')
        prompt = self.prompts[idx]
        
        inputs = self.processor(
            text=[prompt], 
            images=[image], 
            return_tensors="pt",
            padding=True
        )
        
        # Convert mask to tensor
        mask_tensor = torch.tensor(np.array(mask) / 255.0, dtype=torch.float)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'labels': mask_tensor
        }
```

### Step 3: Training Loop
```python
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F

def train_model(model, dataloader, num_epochs=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'], 
                pixel_values=batch['pixel_values']
            )
            
            # Resize predictions to match ground truth
            preds = F.interpolate(
                outputs.logits, 
                size=batch['labels'].shape[-2:],
                mode='bilinear'
            )
            
            loss = criterion(preds.squeeze(), batch['labels'])
            loss.backward()
            optimizer.step()
```

### Step 4: Inference and Evaluation
```python
def generate_masks(model, processor, images, prompts):
    results = []
    
    for image_path, prompt in zip(images, prompts):
        image = Image.open(image_path)
        
        inputs = processor(
            text=[prompt],
            images=[image], 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Convert to binary mask
        preds = torch.sigmoid(outputs.logits)
        mask = (preds > 0.5).float() * 255
        
        # Save with required naming convention
        mask_array = mask.squeeze().cpu().numpy().astype(np.uint8)
        mask_image = Image.fromarray(mask_array, mode='L')
        
        filename = f"{image_id}__{prompt.replace(' ', '_')}.png"
        mask_image.save(filename)
        results.append(filename)
    
    return results
```

## Evaluation Metrics

Implement **mIoU** and **Dice coefficient** as specified in your rubric[1]:

```python
def calculate_iou(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / union

def calculate_dice(pred, target):
    intersection = (pred * target).sum()
    return (2 * intersection) / (pred.sum() + target.sum())
```

## Expected Performance

CLIPSeg typically achieves:
- **mIoU**: 0.3-0.5 on complex segmentation tasks[4]
- **Inference time**: ~50ms per image on GPU[4]
- **Model size**: ~150MB (frozen CLIP + decoder)[4]

For better results, consider data augmentation, prompt engineering with multiple text variations per category, and potentially combining with SAM-based approaches for post-processing[8][9].

This approach should provide a solid foundation for your drywall QA segmentation task with the flexibility to handle both crack detection and taping area segmentation through natural language prompts.

Sources
[1] Prompted_Segmentation_for_Drywall_QA.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86012097/ed85e419-12af-4c75-94cc-5aca54aa615a/Prompted_Segmentation_for_Drywall_QA.pdf

[2] CLIPSeg https://huggingface.co/docs/transformers/en/model_doc/clipseg

[3] Image Segmentation Using Text and Image Prompts https://www.scribd.com/document/633554920/Luddecke-Image-Segmentation-Using-Text-and-Image-Prompts-CVPR-2022-paper-pdf

[4] Image Segmentation Using Text and Image Prompts https://openaccess.thecvf.com/content/CVPR2022/papers/Luddecke_Image_Segmentation_Using_Text_and_Image_Prompts_CVPR_2022_paper.pdf

[5] Zero-shot image segmentation with CLIPSeg https://huggingface.co/blog/clipseg-zero-shot

[6] Fine-tune CLIPSeg with (image, mask) dataset https://discuss.huggingface.co/t/fine-tune-clipseg-with-image-mask-dataset/36935

[7] EVF-SAM: Early Vision-Language Fusion for Text-Prompted ... - arXiv https://arxiv.org/abs/2406.20076

[8] Exploring ways to text prompt SAM 2 — Sieve Blog https://www.sievedata.com/blog/segment-anything-2-sam-2-text-prompting

[9] Text prompts - segment-geospatial https://samgeo.gishub.org/examples/text_prompts/

[10] Segment Any Text: A Universal Approach for Robust ... https://aclanthology.org/2024.emnlp-main.665/

[11] GitHub - byrkbrk/prompting-for-segmentation: Image segmentation using FastSAM and (negative and positive) prompts in PyTorch https://github.com/byrkbrk/prompting-for-segmentation

[12] From Text Segmentation to Smart Chaptering: A Novel ... https://aclanthology.org/2024.eacl-long.25.pdf

[13] Welcome to segmentation_models_pytorch’s documentation!¶ https://segmentation-modelspytorch.readthedocs.io/en/latest/

[14] Unified Segmentation-Conditioned Diffusion for Precise Visual Text ... https://arxiv.org/html/2507.00992v1

[15] segmentation-models-pytorch https://pypi.org/project/segmentation-models-pytorch/0.0.3/

[16] Text prompt? · Issue #4 · facebookresearch/segment-anything https://github.com/facebookresearch/segment-anything/issues/4

[17] Text4Seg: Reimagining Image Segmentation as Text ... https://arxiv.org/html/2410.09855v1

[18] qubvel-org/segmentation_models.pytorch https://github.com/qubvel-org/segmentation_models.pytorch

[19] TI2V-Zero: Zero-Shot Image Conditioning for Text-to-Video ... https://openaccess.thecvf.com/content/CVPR2024/papers/Ni_TI2V-Zero_Zero-Shot_Image_Conditioning_for_Text-to-Video_Diffusion_Models_CVPR_2024_paper.pdf

[20] SAM 2 – Promptable Segmentation for Images and Videos https://learnopencv.com/sam-2/

[21] Sam2 text prompts - segment-geospatial - samgeo.gishub.org https://samgeo.gishub.org/examples/sam2_text_prompts/

[22] FICE: Text-conditioned fashion-image editing with guided ... https://www.sciencedirect.com/science/article/pii/S0031320324007738

[23] PyTorch Implementation of various Semantic Segmentation models (deeplabV3+, PSPNet, Unet, ...) https://www.reddit.com/r/deeplearning/comments/cmq4ko/pytorch_implementation_of_various_semantic/

[24] Segment Anything Model (SAM) - Ultralytics YOLO Docs https://docs.ultralytics.com/models/sam/

[25] Controllable and Efficient Multi-Class Pathology Nuclei ... https://papers.miccai.org/miccai-2024/160-Paper0251.html

[26] Models and pre-trained weights — Torchvision main ... https://docs.pytorch.org/vision/main/models.html

[27] GRES: Generalized Referring Expression Segmentation https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_GRES_Generalized_Referring_Expression_Segmentation_CVPR_2023_paper.pdf

[28] ClearCLIP: Decomposing CLIP Representations for Dense Vision-Language Inference https://eccv.ecva.net/virtual/2024/poster/1141

[29] Advancing Referring Expression Segmentation Beyond Single Image https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Advancing_Referring_Expression_Segmentation_Beyond_Single_Image_ICCV_2023_paper.pdf

[30] GitHub - mc-lan/ClearCLIP: [ECCV2024] ClearCLIP: Decomposing CLIP Representations for Dense Vision-Language Inference https://github.com/mc-lan/ClearCLIP

[31] Weakly-supervised 3D Referring Expression Segmentation https://openreview.net/forum?id=cSAAGL0cn0

[32] Explore the Potential of CLIP for Training-Free Open Vocabulary Semantic Segmentation https://arxiv.org/abs/2407.08268

[33] Advancing Referring Expression Segmentation Beyond ... https://arxiv.org/abs/2305.12452

[34] VLTSeg: Simple Transfer of CLIP-Based Vision-Language ... https://huggingface.co/papers/2312.02021

[35] CLIP for Segmentation CLIPSEG - Purnasai G https://purnasai.github.io/talks/clipseg

[36] Generalized Referring Expression Segmentation Driven by ... https://www.sciencedirect.com/science/article/abs/pii/S0031320325011872

[37] Ensembling CLIP-based Vision-Language Models for... - OpenReview https://openreview.net/forum?id=IJJJT1vIdX

[38] timojl/clipseg: This repository contains the code of ... https://github.com/timojl/clipseg

[39] Papers with Code - Referring Expression Segmentation https://paperswithcode.com/task/referring-expression-segmentation

[40] CRIS: CLIP-Driven Referring Image Segmentation https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_CRIS_CLIP-Driven_Referring_Image_Segmentation_CVPR_2022_paper.pdf

[41] Adding CLIPSeg automatic masking to Stable Diffusion https://mybyways.com/blog/adding-clipseg-automatic-masking-to-stable-diffusion

[42] VEGA: Visual Expression Guidance for Referring ... https://openreview.net/forum?id=O1b8uIQCZb

[43] GitHub - MuhammadAliS/CLIP: PyTorch implementation of OpenAI's CLIP model for image classification, visual search, and visual question answering (VQA). https://github.com/MuhammadAliS/CLIP

[44] How to split text into a dataset for finetuning? https://www.reddit.com/r/LocalLLaMA/comments/1aey1tb/how_to_split_text_into_a_dataset_for_finetuning/

[45] Crack Segmentation Dataset https://docs.ultralytics.com/datasets/segment/crack-seg/

[46] Fine-Tuning a Semantic Segmentation Model on a Custom ... https://huggingface.co/learn/cookbook/en/semantic_segmentation_fine_tuning_inference

[47] ultralytics/docs/en/datasets/segment/crack-seg.md at main · ultralytics/ultralytics https://github.com/ultralytics/ultralytics/blob/main/docs/en/datasets/segment/crack-seg.md

[48] Finetuning clip can be done locally with decent results (even if you ... https://huggingface.co/blog/herooooooooo/clip-finetune

[49] WAS: Dataset and Methods for Artistic Text Segmentation https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07220.pdf

[50] crack Instance Segmentation Dataset and Pre-Trained Model by marieam https://universe.roboflow.com/marieam/crack-bphdr-bl00w

[51] Fine Tuning CLIP Model for Optimal Performance https://www.labellerr.com/blog/fine-tuning-clip-on-custom-dataset/
[52] crack Instance Segmentation Model by University https://universe.roboflow.com/university-bswxt/crack-bphdr


[53] A beginner's guide to fine-tuning the CLIP model for your ... https://github.com/mlfoundations/open_clip/discussions/812

[54] Crack-seg https://docs.ultralytics.com/tr/datasets/segment/crack-seg/

[55] GitHub - zer0int/CLIP-fine-tune, or: SDXL + training the text encoder - Why you should (imo) train CLIP separately from U-Net. With code + instructions. https://www.reddit.com/r/StableDiffusion/comments/1cgyjvt/
github_zer0intclipfinetune_or_sdxl_training_the/

[56] Unit 2: Fine-Tuning, Guidance and Conditioning https://huggingface.co/learn/diffusion-course/en/unit2/1

[57] GitHub - fbrandao2k/YoloV7_Crack_Detection_Segmentation: YoloV7 used trained for Crack Detection and Segmentation using database from Roboflow https://github.com/fbrandao2k/YoloV7_Crack_Detection_Segmentation

[58] Fine-Tuning the CLIP Foundation Model for Image ... https://www.alexanderthamm.com/en/blog/fine-tuning-the-clip-foundation-model-for-image-classification/
