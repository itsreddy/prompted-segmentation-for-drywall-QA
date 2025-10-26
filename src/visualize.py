import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import Counter
from train import collate_fn


def visualize_samples(train_dataset, train_labels, sample_indices):
    

    # Visualize sample data
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Sample Images and Masks from train Dataset', fontsize=16, fontweight='bold')


    for i, sample_idx in enumerate(sample_indices):
        sample = train_dataset[sample_idx]
        image = sample['image']
        mask = sample['mask']
        prompt = sample['text_prompt']
        label_type = train_labels[sample_idx]
        
        # Convert mask to numpy for visualization
        mask_np = np.array(mask)
        
        # Display image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f'Image: {label_type}', fontweight='bold')
        axes[0, i].axis('off')
        
        # Display mask
        axes[1, i].imshow(mask_np, cmap='gray')
        axes[1, i].set_title(f'Mask: "{prompt}"', fontweight='bold')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def detailed_sample_analysis(train_dataset, train_labels, sample_indices):
    # Print detailed analysis
    print("🔍 DETAILED SAMPLE ANALYSIS:")
    print("=" * 40)

    for i, sample_idx in enumerate(sample_indices):
        sample = train_dataset[sample_idx]
        label_type = train_labels[sample_idx]
        mask_np = np.array(sample['mask'])
        
        # Calculate mask statistics
        mask_pixels = np.sum(mask_np > 0)
        total_pixels = mask_np.shape[0] * mask_np.shape[1]
        coverage = mask_pixels / total_pixels * 100
        
        print(f"\nSample {i+1} ({label_type}):")
        print(f"  Text prompt: '{sample['text_prompt']}'")
        print(f"  Mask coverage: {coverage:.2f}% of image")
        print(f"  Mask pixels: {mask_pixels:,} / {total_pixels:,}")
        print(f"  Filename: {sample['filename']}")


def get_sample_indices(train_labels, num_samples=4, random_seed=42):
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    # Find samples from different categories
    crack_indices = [i for i, label in enumerate(train_labels) if label == 'crack']
    drywall_indices = [i for i, label in enumerate(train_labels) if label == 'drywall_joint']

    # Show 2 crack samples and 2 drywall samples
    random_drywall_indices = np.random.randint(0, len(drywall_indices), num_samples)
    random_crack_indices = np.random.randint(0, len(crack_indices), num_samples)
    sample_indices = [
        crack_indices[random_crack_indices[0]], 
        crack_indices[random_crack_indices[1]], 
        drywall_indices[random_drywall_indices[2]], 
        drywall_indices[random_drywall_indices[3]]
    ]
    return sample_indices


def visualize_preds_and_masks(model, processor, test_dataset, test_labels, device, random_seed=42):
    np.random.seed(random_seed)
    # Visualize prediction on test data
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Sample Images and Masks from test Dataset', fontsize=16, fontweight='bold')

    # Find samples from different categories
    crack_indices = [i for i, label in enumerate(test_labels) if label == 'crack']
    drywall_indices = [i for i, label in enumerate(test_labels) if label == 'drywall_joint']

    # Show 2 crack samples and 2 drywall samples
    if crack_indices.__len__() >= 2 and drywall_indices.__len__() >= 2:
        random_crack_indices = np.random.randint(0, len(crack_indices), 2)
        random_drywall_indices = np.random.randint(0, len(drywall_indices), 2)
        sample_indices = [crack_indices[random_crack_indices[0]], crack_indices[random_crack_indices[1]], drywall_indices[random_drywall_indices[0]], drywall_indices[random_drywall_indices[1]]]
    else:
        sample_indices = np.random.choice(len(test_dataset), 4, replace=False)

    for i, sample_idx in enumerate(sample_indices):
        sample = test_dataset[sample_idx]
        image = sample['image']
        mask = sample['mask']
        prompt = sample['text_prompt']
        label_type = test_labels[sample_idx]
        
        # Convert mask to numpy for visualization
        mask_np = np.array(mask)

    # predicted mask
        # Forward pass
        debug_batch = collate_fn([sample], processor=processor)
        with torch.no_grad():
            outputs = model(
                input_ids=debug_batch['input_ids'].to(device),
                attention_mask=debug_batch['attention_mask'].to(device),
                pixel_values=debug_batch['pixel_values'].to(device)
            )
        
        # CLIPSeg outputs logits at 352x352, need to upsample to 640x640
        # Add channel dimension if needed
        logits = outputs.logits
        if len(logits.shape) == 3:  # [batch, H, W] -> [batch, 1, H, W]
            logits = logits.unsqueeze(1)
        
        # Resize predictions to match ground truth (352x352 -> 640x640)
        preds = F.interpolate(
            logits,
            size=(640, 640),
            mode='bilinear',
            align_corners=False
        )
        
        # Remove channel dimension for loss calculation
        preds = preds.squeeze(1)
        preds = preds.squeeze(0)  # [batch, 1, 640, 640] -> [640, 640]
        preds = torch.sigmoid(preds)  # Apply sigmoid to get probabilities
        
        # Display image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f'Image: {label_type}', fontweight='bold')
        axes[0, i].axis('off')
        
        # Display mask
        axes[1, i].imshow(mask_np, cmap='gray')
        axes[1, i].set_title(f'Mask: "{prompt}"', fontweight='bold')
        axes[1, i].axis('off')

        # Display predicted mask
        axes[2, i].imshow(preds.cpu().numpy(), cmap='gray')
        axes[2, i].set_title(f'Predicted Mask: "{prompt}"', fontweight='bold')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

def dataset_summary(train_dataset, train_prompts, train_labels, train_crack_count, train_drywall_count):
    # Final dataset analysis and summary
    print("FINAL DATASET ANALYSIS")
    print("=" * 50)

    # Analyze text prompts distribution
    prompt_counts = Counter(train_prompts)

    print("\nText Prompts Distribution:")
    for prompt, count in prompt_counts.most_common():
        percentage = count / len(train_prompts) * 100
        print(f"  '{prompt}': {count} samples ({percentage:.1f}%)")

    # Calculate mask coverage statistics for each category
    print(f"\nMask Coverage Analysis:")
    crack_coverages = []
    drywall_coverages = []

    # Sample a subset for analysis (to avoid processing all 6k images)
    sample_size = min(100, len(train_dataset))
    sample_indices = np.random.choice(len(train_dataset), sample_size, replace=False)

    for idx in sample_indices:
        sample = train_dataset[idx]
        label_type = train_labels[idx]
        mask_np = np.array(sample['mask'])
        
        mask_pixels = np.sum(mask_np > 0)
        total_pixels = mask_np.shape[0] * mask_np.shape[1]
        coverage = mask_pixels / total_pixels * 100
        
        if label_type == 'crack':
            crack_coverages.append(coverage)
        else:
            drywall_coverages.append(coverage)

    if crack_coverages:
        print(f"  Crack mask coverage: {np.mean(crack_coverages):.2f}% ± {np.std(crack_coverages):.2f}%")
        print(f"    Range: {np.min(crack_coverages):.2f}% - {np.max(crack_coverages):.2f}%")

    if drywall_coverages:
        print(f"  Drywall mask coverage: {np.mean(drywall_coverages):.2f}% ± {np.std(drywall_coverages):.2f}%")
        print(f"    Range: {np.min(drywall_coverages):.2f}% - {np.max(drywall_coverages):.2f}%")

    # Dataset readiness summary
    print(f"\nDATASET READY FOR CLIPSEG TRAINING!")
    print("=" * 50)
    print(f"Total train samples: {len(train_dataset):,}")
    print(f"Text prompts: {len(set(train_prompts))} unique prompts")
    print(f"Image format: 640x640x3 (RGB)")
    print(f"Mask format: 640x640 (binary)")
    print(f"Categories: 2 (cracks + drywall joints)")
    print(f"Data balance: {train_crack_count:,} cracks ({train_crack_count/len(train_labels)*100:.1f}%) + {train_drywall_count:,} joints ({train_drywall_count/len(train_labels)*100:.1f}%)")

    print(f"\nReady for CLIPSeg fine-tuning with text prompts!")
    print("   Next steps: Load CLIPSeg model and implement training loop")


def plot_training_curves(train_losses, val_losses, step_eval_history=None, save_path=None):
    """
    Plot training and validation loss curves, including step-based evaluations.
    
    Args:
        train_losses: List of training losses (per epoch)
        val_losses: List of validation losses (per epoch)
        step_eval_history: List of step-based evaluation results (optional)
        save_path: Optional path to save the plot
    """
    if step_eval_history and len(step_eval_history) > 0:
        # Include step-based evaluation plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Epoch-based losses
        axes[0].plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss')
        axes[0].plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Epoch-based Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Step-based evaluation
        steps = [eval_data['step'] for eval_data in step_eval_history]
        step_val_losses = [eval_data['val_loss'] for eval_data in step_eval_history]
        step_train_losses = [eval_data['train_loss'] for eval_data in step_eval_history]
        
        axes[1].plot(steps, step_train_losses, 'b.-', label='Train Loss')
        axes[1].plot(steps, step_val_losses, 'r.-', label='Validation Loss')
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Step-based Evaluation')
        axes[1].legend()
        axes[1].grid(True)
        
        # Training loss detail
        axes[2].plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Train Loss')
        axes[2].set_title('Training Loss Detail')
        axes[2].legend()
        axes[2].grid(True)
        
    else:
        # Original two-panel plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss')
        axes[0].plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Train Loss')
        axes[1].set_title('Training Loss Detail')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
