import cv2
import numpy as np
from typing import List
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from load import parse_pascal_voc_xml


class DryWallQADataset(Dataset):
    """
    Unified dataset for both crack detection and drywall joint detection.
    Handles both polygon and bounding box annotations.
    """
    
    def __init__(self, image_paths, annotation_paths, text_prompts, transform=None):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.text_prompts = text_prompts
        self.transform = transform
        
        assert len(image_paths) == len(annotation_paths) == len(text_prompts), \
            "All inputs must have the same length"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Load annotation
        annotation_path = self.annotation_paths[idx]
        annotation_data = parse_pascal_voc_xml(annotation_path)
        
        # Get text prompt
        text_prompt = self.text_prompts[idx]
        
        # Create binary mask from annotations
        mask = self._create_mask(annotation_data)
        
        sample = {
            'image': image,
            'mask': mask,
            'text_prompt': text_prompt,
            'filename': annotation_data['filename'],
            'image_path': str(image_path),
            'annotation_path': str(annotation_path)
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def _create_mask(self, annotation_data):
        """Create binary mask from annotation data"""
        height, width = annotation_data['height'], annotation_data['width']
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for obj in annotation_data['objects']:
            if 'polygon' in obj:
                # Handle polygon annotations (cracks)
                points = obj['polygon']
                if len(points) > 2:
                    # Convert to numpy array for OpenCV
                    pts = np.array(points, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], 255)
            
            elif 'bbox' in obj:
                # Handle bounding box annotations (drywall joints)
                bbox = obj['bbox']
                mask[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']] = 255
        
        return Image.fromarray(mask, mode='L')

def create_combined_dataset(splits: List[str], cracks_files: dict, drywall_files: dict):
    """Create a combined dataset with appropriate text prompts"""
    all_images = []
    all_annotations = []
    all_prompts = []
    all_labels = []  # Track dataset source
    
    # Text prompts for different tasks
    crack_prompts = [
        "crack",
        "wall crack", 
        "concrete crack",
        "surface crack",
        "structural crack",
        "segment crack",
        "segment wall crack",
    ]
    
    drywall_prompts = [
        "drywall joint",
        "taping area",
        "drywall seam",
        "joint tape",
        "wall joint",
        "segment taping area",
        "segment joint/tape",
        "segment drywall seam",
        "wall joint line"
    ]
    
    # Add crack detection samples
    for split in splits:
        if split in cracks_files:
            images = cracks_files[split]['images']
            annotations = cracks_files[split]['annotations']
            
            for img_path, ann_path in zip(images, annotations):
                all_images.append(img_path)
                all_annotations.append(ann_path)
                all_prompts.append(np.random.choice(crack_prompts))
                all_labels.append('crack')
    
    # Add drywall joint detection samples
    for split in splits:
        if split in drywall_files:
            images = drywall_files[split]['images']
            annotations = drywall_files[split]['annotations']
            
            for img_path, ann_path in zip(images, annotations):
                all_images.append(img_path)
                all_annotations.append(ann_path)
                all_prompts.append(np.random.choice(drywall_prompts))
                all_labels.append('drywall_joint')
    
    return all_images, all_annotations, all_prompts, all_labels

print("DryWallQADataset class defined!")
print("create_combined_dataset function defined!")
print("\nDataset features:")
print("- Handles both polygon (crack) and bounding box (drywall) annotations")
print("- Multiple text prompts per category for data augmentation")
print("- Creates binary masks for CLIPSeg training")
