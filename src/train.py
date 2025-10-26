import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from pathlib import Path
import time
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss

# Import custom modules
from load import get_dataset_files
from dataset import create_combined_dataset, DryWallQADataset


def configure_trainable_parameters(model, verbose=True):
    if verbose:
        print("CONFIGURING TRAINABLE PARAMETERS")
        print("=" * 50)

    encoder_params_list = list(model.clip.parameters())
    decoder_params_list = list(model.decoder.parameters())

    # keep encoder trainable
    for param in encoder_params_list:
        param.requires_grad = True

    # Keep decoder trainable
    for param in decoder_params_list:
        param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    if verbose:
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    return encoder_params_list, decoder_params_list


def setup_model(model_name="CIDAS/clipseg-rd64-refined", device=None, checkpoint_path=None, verbose=True):
    """
    Setup CLIPSeg model and processor with configurable parameters.
    
    Args:
        model_name (str): HuggingFace model identifier
        device (str): Device to use ('cuda', 'cpu', or None for auto-detection)
        checkpoint_path (str, optional): Path to a saved checkpoint to load model weights from
        verbose (bool): Whether to print setup information
    
    Returns:
        tuple: (model, processor, device)
    """
    if verbose:
        print("SETTING UP CLIPSEG MODEL")
        print("=" * 50)

    # Load CLIPSeg model and processor
    processor = CLIPSegProcessor.from_pretrained(model_name)
    model = CLIPSegForImageSegmentation.from_pretrained(model_name)

    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    model = model.to(device)
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        if verbose:
            print(f"Loading checkpoint from: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if verbose:
                print(f"Checkpoint loaded successfully!")
                if 'epoch' in checkpoint:
                    print(f"   Epoch: {checkpoint['epoch']}")
                if 'val_loss' in checkpoint:
                    print(f"   Validation Loss: {checkpoint['val_loss']:.4f}")
                if 'train_loss' in checkpoint:
                    print(f"   Training Loss: {checkpoint['train_loss']:.4f}")
                    
        except FileNotFoundError:
            if verbose:
                print(f"Checkpoint file not found: {checkpoint_path}")
                print("   Continuing with pre-trained weights...")
        except Exception as e:
            if verbose:
                print(f"Error loading checkpoint: {str(e)}")
                print("   Continuing with pre-trained weights...")
    
    if verbose:
        print(f"Model loaded and moved to: {device}")

    return model, processor, device


def get_training_config(verbose=True, **kwargs):
    """
    Get training configuration parameters.
    """
    config = {key: value for key, value in kwargs.items()}

    if verbose:
        print("🔧 TRAINING CONFIGURATION")
        print("=" * 50)
        print(f"Batch size: {config['batch_size']}")
        print(f"Learning rate: {config['encoder_lr']}")
        print(f"Number of epochs: {config['num_epochs']}")
        if config['save_every'] is not None:
            print(f"Save frequency: Every {config['save_every']} steps")
        else:
            print(f"Save frequency: Only best model at epoch end")
        if config['eval_every'] is not None:
            print(f"Eval frequency: Every {config['eval_every']} steps")
        else:
            print(f"Eval frequency: Only at epoch end")
        print(f"Eval configuration: {config['eval_every']}")

    return config


def collate_fn(batch, processor):
    """Custom collate function for CLIPSeg training"""
    images = [sample['image'] for sample in batch]
    masks = [sample['mask'] for sample in batch]
    prompts = [sample['text_prompt'] for sample in batch]
    
    # Process inputs with CLIPSeg processor
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True
    )
    
    # Convert masks to tensors
    mask_tensors = []
    for mask in masks:
        mask_array = np.array(mask)
        mask_tensor = torch.tensor(mask_array / 255.0, dtype=torch.float32)
        mask_tensors.append(mask_tensor)
    
    # Stack masks
    ground_truth_masks = torch.stack(mask_tensors)
    
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'pixel_values': inputs['pixel_values'],
        'labels': ground_truth_masks
    }

def setup_data_loaders(
    train_dataset,
    val_dataset,
    processor,
    config,
    num_workers=0,
    verbose=True
):
    """
    Setup train and validation data loaders.
    
    Args:
        train_dataset: Training dataset instance
        val_dataset: Validation dataset instance
        processor: CLIPSeg processor for collate function
        config: Configuration dictionary containing batch_size
        num_workers (int): Number of worker processes
        verbose (bool): Whether to print setup information
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    if verbose:
        print("SETTING UP DATA LOADERS")
        print(f"Batch size: {config['batch_size']}")
        print("=" * 50)
    
    # Create collate function with processor
    def collate_fn_wrapper(batch):
        return collate_fn(batch, processor)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn_wrapper,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn_wrapper,
        num_workers=num_workers
    )

    if verbose:
        print(f"Data loaders created:")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        print(f"   Samples per epoch: {len(train_dataset)}")

    return train_loader, val_loader


# Fixed Training and evaluation functions
def train_one_epoch(
    model, 
    train_loader, 
    optimizer, 
    criterion, 
    device, 
    epoch, 
    save_every=None, 
    eval_every=None,
    val_loader=None,
    checkpoint_path_prefix="checkpoint", 
    verbose=True
):
    """Train for one epoch with optional step-based evaluation and checkpointing"""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    global_step = epoch * num_batches  # Calculate global step for this epoch
    
    # Store step-based evaluation results
    step_eval_results = []
    
    # Create progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        current_step = global_step + batch_idx
        
        optimizer.zero_grad()
        
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
        
        # CLIPSeg outputs logits at 352x352, need to upsample to 640x640
        # Add channel dimension if needed
        logits = outputs.logits
        if len(logits.shape) == 3:  # [batch, H, W] -> [batch, 1, H, W]
            logits = logits.unsqueeze(1)
        
        # Resize predictions to match ground truth (352x352 -> 640x640)
        preds = F.interpolate(
            logits,
            size=labels.shape[-2:],  # (640, 640)
            mode='bilinear',
            align_corners=False
        )
        
        # Remove channel dimension for loss calculation
        preds = preds.squeeze(1)  # [batch, 1, 640, 640] -> [batch, 640, 640]
        
        # Calculate loss
        loss = criterion(preds, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Update progress bar
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Avg Loss': f'{avg_loss:.4f}'})
        
        # Run evaluation every eval_every steps if specified
        if eval_every is not None and val_loader is not None and current_step > 0 and current_step % eval_every == 0:
            if verbose:
                print(f"\n📊 Running evaluation at step {current_step}...")
            
            # Run evaluation
            val_loss = evaluate(model, val_loader, criterion, device)
            step_eval_results.append({
                'step': current_step,
                'epoch': epoch,
                'batch_idx': batch_idx,
                'val_loss': val_loss,
                'train_loss': avg_loss
            })
            
            if verbose:
                print(f"📊 Step {current_step} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Set model back to training mode after evaluation
            model.train()
        
        # Save checkpoint every save_every steps if specified
        if save_every is not None and current_step > 0 and current_step % save_every == 0:
            checkpoint_path = f"{checkpoint_path_prefix}_step_{current_step}.pth"
            checkpoint = {
                'epoch': epoch,
                'step': current_step,
                'batch_idx': batch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'avg_loss': avg_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            if verbose:
                print(f"\n💾 Checkpoint saved at step {current_step}: {checkpoint_path}")
        
        # Free memory
        del outputs, preds, loss, logits
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return total_loss / num_batches, step_eval_results

def evaluate(model, val_loader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation', leave=False):
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
            
            # Resize predictions
            preds = F.interpolate(
                logits,
                size=labels.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            preds = preds.squeeze(1)
            
            # Calculate loss
            loss = criterion(preds, labels)
            total_loss += loss.item()
            
            # Free memory
            del outputs, preds, loss, logits
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return total_loss / num_batches

def create_datasets(cracks_dataset_path, drywall_dataset_path, verbose=True):
    """
    Create train and validation datasets from the given dataset paths.
    
    Args:
        cracks_dataset_path (str): Path to cracks dataset
        drywall_dataset_path (str): Path to drywall dataset
        verbose (bool): Whether to print dataset creation information
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    if verbose:
        print("CREATING DATASETS")
        print("=" * 50)
    
    # Load dataset files
    if verbose:
        print("Loading crack detection dataset...")
    cracks_files = get_dataset_files(Path(cracks_dataset_path))
    
    if verbose:
        print("Loading drywall joint detection dataset...")
    drywall_files = get_dataset_files(Path(drywall_dataset_path))
    
    # Create combined datasets
    if verbose:
        print("Creating combined dataset...")
    train_images, train_annotations, train_prompts, train_labels = create_combined_dataset(['train'], cracks_files, drywall_files)
    val_images, val_annotations, val_prompts, val_labels = create_combined_dataset(['valid'], cracks_files, drywall_files)
    
    # Create dataset instances
    train_dataset = DryWallQADataset(train_images, train_annotations, train_prompts)
    val_dataset = DryWallQADataset(val_images, val_annotations, val_prompts)
    
    if verbose:
        print(f"Combined datasets created!")
        print(f"   Total train samples: {len(train_images)}")
        print(f"   Total validation samples: {len(val_images)}")
        print(f"   Train Dataset instance created with {len(train_dataset)} samples")
        print(f"   Validation Dataset instance created with {len(val_dataset)} samples")
    
    return train_dataset, val_dataset


def train_model(
    model,
    train_loader,
    val_loader,
    train_dataset,
    device,
    config,
    checkpoint_path="best_clipseg_drywall_model.pth",
    verbose=True
):
    """
    Train the CLIPSeg model.
    
    Args:
        model: CLIPSeg model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        train_dataset: Training dataset (for sample count)
        device: Training device
        config: Training configuration dictionary (must include 'save_every' and 'eval_every' keys)
        checkpoint_path: Path to save best model checkpoint
        verbose: Whether to print training information
    
    Returns:
        dict: Training results with losses and metrics, including step_eval_history
        
    Note:
        Three types of operations are performed:
        1. Best model checkpoint: Saved when validation loss improves (at checkpoint_path)
        2. Step checkpoints: Saved every config['save_every'] steps (if save_every is not None)
        3. Step evaluations: Run every config['eval_every'] steps (if eval_every is not None)
    """
    if verbose:
        print("STARTING CLIPSEG TRAINING")
        print("=" * 50)

    # Setup optimizer and loss function
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=config['learning_rate']
    )
    criterion = BCEWithLogitsLoss()

    # Training history
    train_losses = []
    val_losses = []
    step_eval_history = []  # Store step-based evaluation results
    best_val_loss = float('inf')

    if verbose:
        print(f"Training Details:")
        print(f"   Optimizer: AdamW (lr={config['learning_rate']})")
        print(f"   Loss function: BCEWithLogitsLoss")
        print(f"   Device: {device}")
        print(f"   Model input: 640x640 -> 352x352 (internal)")
        print(f"   Model output: 352x352 -> 640x640 (upsampled)")

    # Start training
    start_time = time.time()

    for epoch in range(config['num_epochs']):
        if verbose:
            print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
            print("-" * 30)
        
        # Train for one epoch with save_every checkpointing and eval_every evaluation
        epoch_start = time.time()
        train_loss, step_evals = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            criterion, 
            device, 
            epoch,
            save_every=config.get('save_every'),
            eval_every=config.get('eval_every'),
            val_loader=val_loader,
            checkpoint_path_prefix=checkpoint_path.replace('.pth', ''),
            verbose=verbose
        )
        
        # Store step-based evaluation results
        step_eval_history.extend(step_evals)
        
        # Evaluate on validation set (end of epoch)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        if verbose:
            # Print epoch results
            print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if verbose:
                print(f"🏆 New best validation loss: {val_loss:.4f}")
            
            # Save model checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            if verbose:
                print(f"Model saved as '{checkpoint_path}'")

    # Training complete
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\nTRAINING COMPLETED!")
        print("=" * 50)
        print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final train loss: {train_losses[-1]:.4f}")
        print(f"Final validation loss: {val_losses[-1]:.4f}")

    # Create results dictionary
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'step_eval_history': step_eval_history,  # Include step-based evaluation results
        'best_val_loss': best_val_loss,
        'total_time': total_time,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'epochs_completed': len(train_losses),
        'samples_processed': len(train_dataset) * len(train_losses)
    }

    return results
