# Evaluation Metrics: mIoU and Dice Coefficient
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import jaccard_score
from process import process_predicted_mask
from train import collate_fn


def calculate_iou(pred_mask, true_mask, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) for binary masks.
    
    Args:
        pred_mask (torch.Tensor or np.ndarray): Predicted mask
        true_mask (torch.Tensor or np.ndarray): Ground truth mask
        threshold (float): Threshold for binarizing predictions
    
    Returns:
        float: IoU score
    """
    # Convert to numpy if tensors
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()
    
    # Binarize prediction
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    true_binary = (true_mask > threshold).astype(np.uint8)
    
    # Flatten for calculation
    pred_flat = pred_binary.flatten()
    true_flat = true_binary.flatten()
    
    # Calculate intersection and union
    intersection = np.sum(pred_flat & true_flat)
    union = np.sum(pred_flat | true_flat)
    
    # Handle edge case where union is 0
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def calculate_dice(pred_mask, true_mask, threshold=0.5):
    """
    Calculate Dice coefficient for binary masks.
    
    Args:
        pred_mask (torch.Tensor or np.ndarray): Predicted mask
        true_mask (torch.Tensor or np.ndarray): Ground truth mask
        threshold (float): Threshold for binarizing predictions
    
    Returns:
        float: Dice coefficient
    """
    # Convert to numpy if tensors
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()
    
    # Binarize prediction
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    true_binary = (true_mask > threshold).astype(np.uint8)
    
    # Flatten for calculation
    pred_flat = pred_binary.flatten()
    true_flat = true_binary.flatten()
    
    # Calculate intersection
    intersection = np.sum(pred_flat & true_flat)
    
    # Calculate Dice coefficient: 2 * intersection / (|A| + |B|)
    dice = (2.0 * intersection) / (np.sum(pred_flat) + np.sum(true_flat))
    
    # Handle edge case where both masks are empty
    if np.sum(pred_flat) == 0 and np.sum(true_flat) == 0:
        return 1.0
    
    return dice

def calculate_pixel_accuracy(pred_mask, true_mask, threshold=0.5):
    """
    Calculate pixel-wise accuracy.
    
    Args:
        pred_mask (torch.Tensor or np.ndarray): Predicted mask
        true_mask (torch.Tensor or np.ndarray): Ground truth mask
        threshold (float): Threshold for binarizing predictions
    
    Returns:
        float: Pixel accuracy
    """
    # Convert to numpy if tensors
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()
    
    # Binarize prediction
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    true_binary = (true_mask > threshold).astype(np.uint8)
    
    # Calculate accuracy
    correct_pixels = np.sum(pred_binary == true_binary)
    total_pixels = pred_binary.size
    
    return correct_pixels / total_pixels

# Comprehensive Evaluation Function
def evaluate_model_with_metrics(model, val_loader, val_labels, device, num_samples=None):
    """
    Evaluate model with mIoU and Dice metrics, broken down by prompt type.
    
    Args:
        model: Trained CLIPSeg model
        val_loader: Validation DataLoader
        val_labels: List of labels ('crack' or 'drywall_joint')
        device: Torch device
        num_samples: Number of samples to evaluate (None for all)
    
    Returns:
        dict: Comprehensive metrics dictionary
    """
    model.eval()
    
    # Initialize metric storage
    metrics = {
        'crack': {'iou': [], 'dice': [], 'pixel_acc': []},
        'drywall_joint': {'iou': [], 'dice': [], 'pixel_acc': []},
        'overall': {'iou': [], 'dice': [], 'pixel_acc': []}
    }
    
    sample_count = 0
    
    print("🔍 Evaluating model performance...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Evaluation')):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            
            # Handle CLIPSeg output shape
            logits = outputs.logits
            if len(logits.shape) == 3:
                logits = logits.unsqueeze(1)
            
            # Resize predictions to match ground truth
            preds = F.interpolate(
                logits,
                size=labels.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            preds = preds.squeeze(1)  # Remove channel dimension
            preds = torch.sigmoid(preds)  # Apply sigmoid
            
            # Process each sample in the batch
            batch_size = preds.shape[0]
            batch_start_idx = batch_idx * val_loader.batch_size
            
            for i in range(batch_size):
                if num_samples and sample_count >= num_samples:
                    break
                    
                sample_idx = batch_start_idx + i
                if sample_idx >= len(val_labels):
                    break
                
                # Get individual prediction and ground truth
                pred_mask = preds[i]
                true_mask = labels[i]
                label_type = val_labels[sample_idx]

                pred_mask = process_predicted_mask(pred_mask.cpu().numpy())
                
                # Calculate metrics
                iou = calculate_iou(pred_mask, true_mask)
                dice = calculate_dice(pred_mask, true_mask)
                pixel_acc = calculate_pixel_accuracy(pred_mask, true_mask)
                
                # Store metrics by category
                metrics[label_type]['iou'].append(iou)
                metrics[label_type]['dice'].append(dice)
                metrics[label_type]['pixel_acc'].append(pixel_acc)
                
                # Store overall metrics
                metrics['overall']['iou'].append(iou)
                metrics['overall']['dice'].append(dice)
                metrics['overall']['pixel_acc'].append(pixel_acc)
                
                sample_count += 1
            
            if num_samples and sample_count >= num_samples:
                break
            
            # Free memory
            del outputs, preds, logits
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Calculate statistics
    results = {}
    for category in ['crack', 'drywall_joint', 'overall']:
        if metrics[category]['iou']:  # Only if we have samples
            results[category] = {
                'mIoU': np.mean(metrics[category]['iou']),
                'mIoU_std': np.std(metrics[category]['iou']),
                'mDice': np.mean(metrics[category]['dice']),
                'mDice_std': np.std(metrics[category]['dice']),
                'mPixel_Acc': np.mean(metrics[category]['pixel_acc']),
                'mPixel_Acc_std': np.std(metrics[category]['pixel_acc']),
                'sample_count': len(metrics[category]['iou'])
            }
        else:
            results[category] = {
                'mIoU': 0.0, 'mIoU_std': 0.0,
                'mDice': 0.0, 'mDice_std': 0.0,
                'mPixel_Acc': 0.0, 'mPixel_Acc_std': 0.0,
                'sample_count': 0
            }
    
    return results, metrics


# Visualize Sample Predictions with Metrics
def visualize_predictions_with_metrics(model, processor, val_dataset, val_labels, device, num_samples=6):
    """
    Visualize model predictions along with calculated metrics.
    """
    model.eval()
    
    # Get sample indices from both categories
    crack_indices = [i for i, label in enumerate(val_labels) if label == 'crack']
    drywall_indices = [i for i, label in enumerate(val_labels) if label == 'drywall_joint']
    
    # Select samples
    num_per_category = num_samples // 2
    selected_crack = np.random.choice(crack_indices, min(num_per_category, len(crack_indices)), replace=False)
    selected_drywall = np.random.choice(drywall_indices, min(num_per_category, len(drywall_indices)), replace=False)
    sample_indices = list(selected_crack) + list(selected_drywall)
    
    # Create figure
    fig, axes = plt.subplots(4, len(sample_indices), figsize=(4*len(sample_indices), 16))
    fig.suptitle('Sample Predictions with Metrics', fontsize=16, fontweight='bold')
    
    with torch.no_grad():
        for i, sample_idx in enumerate(sample_indices):
            # Get sample
            sample = val_dataset[sample_idx]
            label_type = val_labels[sample_idx]
            
            # Process through model
            debug_batch = collate_fn([sample], processor)
            outputs = model(
                input_ids=debug_batch['input_ids'].to(device),
                attention_mask=debug_batch['attention_mask'].to(device),
                pixel_values=debug_batch['pixel_values'].to(device)
            )
            
            # Process prediction
            logits = outputs.logits
            if len(logits.shape) == 3:
                logits = logits.unsqueeze(1)
            
            preds = F.interpolate(
                logits,
                size=(640, 640),
                mode='bilinear',
                align_corners=False
            )
            
            preds = preds.squeeze(1).squeeze(0)
            preds = torch.sigmoid(preds)
            
            # Calculate metrics
            true_mask = np.array(sample['mask']) / 255.0
            pred_mask = preds.cpu().numpy()
            
            iou = calculate_iou(pred_mask, true_mask)
            dice = calculate_dice(pred_mask, true_mask)
            pixel_acc = calculate_pixel_accuracy(pred_mask, true_mask)
            
            # Plot image
            axes[0, i].imshow(sample['image'])
            axes[0, i].set_title(f'{label_type.replace("_", " ").title()}', fontweight='bold')
            axes[0, i].axis('off')
            
            # Plot ground truth
            axes[1, i].imshow(true_mask, cmap='gray')
            axes[1, i].set_title(f'GT: "{sample["text_prompt"]}"', fontweight='bold')
            axes[1, i].axis('off')
            
            # Plot prediction
            axes[2, i].imshow(pred_mask, cmap='gray')
            axes[2, i].set_title(f'Prediction (Raw)', fontweight='bold')
            axes[2, i].axis('off')
            
            # Plot processed prediction
            processed_pred = process_predicted_mask(pred_mask)
            processed_iou = calculate_iou(processed_pred/255.0, true_mask)
            processed_dice = calculate_dice(processed_pred/255.0, true_mask)
            
            axes[3, i].imshow(processed_pred, cmap='gray')
            axes[3, i].set_title(f'Processed\nIoU: {processed_iou:.3f} | Dice: {processed_dice:.3f}', 
                               fontweight='bold', fontsize=10)
            axes[3, i].axis('off')
            
            # Add metrics text box on the image
            metrics_text = f'Raw Metrics:\nIoU: {iou:.3f}\nDice: {dice:.3f}\nPix Acc: {pixel_acc:.3f}'
            axes[0, i].text(0.02, 0.98, metrics_text, transform=axes[0, i].transAxes,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return sample_indices
