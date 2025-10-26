Report: Text-Prompted Segmentation for Drywall Quality Assurance

Problem Statement

Given an image of a drywall surface, the goal is to identify and segment a specific region of interest, such as a crack or a taped joint, based on a text prompt.

Approach

I first investigated the feasibility of this solution by prompting large language models (like GPT and Gemini) with a sample of images to identify cracks and seams. The positive responses gave me the confidence to build a text-conditioned segmentation model.

CLIPSeg was selected as the most suitable model. It combines a pre-trained CLIP encoder with a lightweight, trainable decoder. This architecture is ideal for our task for several key reasons:

Lightweight: The decoder has only 1.1M trainable parameters (150.7M parameters in total), making it efficient.

Prompt-Based: It natively supports segmentation from text prompts.

Binary Output: It is straightforward to configure for generating binary masks (values of 0 or 255) as required.

Model Architecture: CLIPSeg

CLIPSeg extends OpenAI's CLIP (Contrastive Language-Image Pre-training) model. CLIP itself was trained on 400 million image-text pairs using a contrastive learning methodology, teaching it to align visual and textual information in a shared embedding space. This allows it to perform zero-shot classification with high accuracy.

CLIPSeg leverages this capability for segmentation, enabling it to handle three distinct scenarios:

Referring Expression Segmentation: Segmenting based on complex text queries.

Zero-Shot Segmentation: Segmenting unseen object categories.

One-Shot Segmentation: Segmenting based on an example image.

Architecture and Key Components

The model consists of a vision encoder, a text encoder (both from CLIP), and a new segmentation decoder.

Vision Encoder: A Vision Transformer (ViT) that processes the image by dividing it into patches. Its self-attention mechanism allows it to capture global context across the entire image, which is crucial for understanding relationships between different regions.

Decoder: A lightweight, transformer-based decoder that generates the segmentation mask. It uses two key mechanisms:

U-Net-inspired Skip Connections: These connections feed features from multiple layers of the encoder directly to the decoder. This provides the multi-scale information needed to capture both high-level semantic meaning and fine-grained spatial details (like edges).

Cross-Attention: This mechanism allows the decoder to condition its output on the text embedding, ensuring it only segments the object described in the prompt (e.g., "crack" vs. "joint").

A critical detail is that the decoder outputs a low-resolution segmentation map (352x352), which is then upsampled to match the original input image size (640x640).

Data Preparation

The dataset was sourced from Roboflow in Pascal VOC format (images and XML annotations) and contained two categories: drywall cracks and drywall joints.

Dataset Statistics

Category

Train

Validation

Test

Total

Cracks

5,164

201

4

5,369

Drywall Joints

820

202

-

1,022

Combined

5,984

403

4

6,391

EXAMINING SAMPLE ANNOTATIONS
==================================================

Sample Crack Annotation:
File: 00002_jpg.rf.0091d2541bc223a680ba21f7e98ef810.jpg
Image size: 640x640x3
Number of objects: 1
  Object 1:
    Class: NewCracks - v2 2024-05-18 10-54pm
    Bounding box: (29, 300) to (641, 530)
    Polygon points: 30 points
    First 3 points: [(504.375, 340.0), (544.375, 321.25), (594.375, 318.75)]

Sample Drywall Joint Annotation:
File: 2000x1500_0_resized_jpg.rf.0240db143fd724683ce9b3dd3114d20b.jpg
Image size: 640x640x3
Number of objects: 1
  Object 1:
    Class: drywall-join
    Bounding box: (253, 1) to (389, 638)








Annotation and Data Insights

Annotation Type: Cracks were annotated with precise polygons, while drywall joints were annotated with simpler bounding boxes. The model would need to learn to output both mask styles based on the text prompt.

Data Imbalance: The training set was heavily imbalanced, consisting of 86.3% crack images and only 13.7% joint images. In contrast, the validation set was deliberately balanced (49.9% cracks, 50.1% joints) to provide a more reliable measure of performance on both tasks.

Text Prompts Distribution:
  'segment crack': 761 samples (12.7%)
  'surface crack': 752 samples (12.6%)
  'crack': 744 samples (12.4%)
  'concrete crack': 741 samples (12.4%)
  'structural crack': 735 samples (12.3%)
  'segment wall crack': 722 samples (12.1%)
  'wall crack': 709 samples (11.8%)
  'wall joint': 112 samples (1.9%)
  'taping area': 103 samples (1.7%)
  'joint tape': 99 samples (1.7%)
  'wall joint line': 99 samples (1.7%)
  'drywall joint': 87 samples (1.5%)
  'segment joint/tape': 84 samples (1.4%)
  'segment taping area': 83 samples (1.4%)
  'segment drywall seam': 81 samples (1.4%)
  'drywall seam': 72 samples (1.2%)

Mask Coverage Analysis:
  Crack mask coverage: 4.87% Â± 4.88%
    Range: 0.19% - 28.89%
  Drywall mask coverage: 16.56% Â± 13.84%
    Range: 1.45% - 57.62%








Sample images and masks from the training dataset.

Methodology and Training

Hardware: M4 Pro MacBook Pro (36 GB RAM, 14-core CPU, 20-core GPU).

Framework: Training was performed on the CPU.

Training Methodology

My first attempt involved freezing the CLIP encoder and training only the 1.1M parameters in the decoder. The performance was poor; the model acted as a simple edge detector and appeared to be overfitting to the more numerous crack images.

The second, successful approach involved unfreezing the entire model and applying discriminative learning rates. This technique uses a much lower learning rate for the pre-trained encoder to fine-tune it, and a higher learning rate for the decoder to learn the new segmentation task.

Model Component

Learning Rate

Rationale

Encoder (CLIP Backbone)

1e-6

Gently adapt pre-trained weights without "catastrophic forgetting."

Decoder (Segmentation Head)

1e-3

Rapidly learn the new task-specific features from scratch.

The model was trained for 12 epochs using the AdamW optimizer (which includes weight decay for regularization) and a BCEWithLogitsLoss function.

TRAINING CONFIGURATION
==============================
Batch size: 16
Learning rate: 0.0001
Number of epochs: 8
Save frequency: Every 500 steps
âœ… Data loaders created:
   Train batches: 374
   Validation batches: 26
   Samples per epoch: 5984








Post-Processing of Predicted Masks

To convert the blurry, grayscale probability maps from the model into clean, binary masks that resemble the ground truth, a sequence of image processing techniques was applied.

Noise Reduction (Denoising): First, a Gaussian Blur or Bilateral Filter can be applied to the raw, grayscale predicted mask. This smooths out general noise and pixelated textures, creating more uniform regions while preserving the main edges.

Binarization using Thresholding: This is the critical step to convert the grayscale image to binary. Otsu's Binarization was identified as the most robust method. It automatically assumes the image contains foreground and background pixels and finds the optimal threshold to separate them, which is more effective than a simple, manually-chosen threshold.

Morphological Operations: These operations refine the shape of the binary mask.

Opening (Erosion followed by Dilation): This operation is excellent for removing small, isolated white noise pixels (salt noise) located outside the main segmented shape.

Closing (Dilation followed by Erosion): This operation is used to fill small black holes (pepper noise) inside the main white shape, making the final mask solid.

This pipeline effectively transforms the blurry, noisy prediction into a clean, solid mask that is much closer to the ground truth.

Results and Evaluation

Training Performance

The model trained for approximately 2.6 hours (157.8 minutes). The training process showed a steady decrease in loss, with the best validation loss of 0.1347 achieved at epoch 8.

Epoch 1/12
------------------------------
Epoch 1 completed in 801.3s
   Train Loss: 0.1450
   Val Loss: 0.2099
New best validation loss: 0.2099
Model saved as 'best_clipseg_drywall_model.pth'

Epoch 2/12
------------------------------
Epoch 2 completed in 796.4s
   Train Loss: 0.1173
   Val Loss: 0.1783
New best validation loss: 0.1783
Model saved as 'best_clipseg_drywall_model.pth'

Epoch 3/12
------------------------------
Epoch 3 completed in 801.8s
   Train Loss: 0.1062
   Val Loss: 0.1631
New best validation loss: 0.1631
Model saved as 'best_clipseg_drywall_model.pth'

Epoch 4/12
------------------------------
Epoch 4 completed in 787.2s
   Train Loss: 0.1002
   Val Loss: 0.1516
New best validation loss: 0.1516
Model saved as 'best_clipseg_drywall_model.pth'

Epoch 5/12
------------------------------
Epoch 5 completed in 786.5s
   Train Loss: 0.0959
   Val Loss: 0.1460
New best validation loss: 0.1460
Model saved as 'best_clipseg_drywall_model.pth'

Epoch 6/12
------------------------------
Epoch 6 completed in 775.4s
   Train Loss: 0.0926
   Val Loss: 0.1416
New best validation loss: 0.1416
Model saved as 'best_clipseg_drywall_model.pth'

Epoch 7/12
------------------------------
Epoch 7 completed in 786.9s
   Train Loss: 0.0899
   Val Loss: 0.1363
New best validation loss: 0.1363
Model saved as 'best_clipseg_drywall_model.pth'

Epoch 8/12
------------------------------
Epoch 8 completed in 801.3s
   Train Loss: 0.0872
   Val Loss: 0.1347
New best validation loss: 0.1347
Model saved as 'best_clipseg_drywall_model.pth'

Epoch 9/12
------------------------------
Epoch 9 completed in 786.6s
   Train Loss: 0.0848
   Val Loss: 0.1408

Epoch 10/12
------------------------------
Epoch 10 completed in 784.2s
   Train Loss: 0.0825
   Val Loss: 0.1419

Epoch 11/12
------------------------------
Epoch 11 completed in 781.0s
   Train Loss: 0.0807
   Val Loss: 0.1388

Epoch 12/12
------------------------------
Epoch 12 completed in 790.9s
   Train Loss: 0.0787
   Val Loss: 0.1358








Training and validation loss curves over 12 epochs.

The final model demonstrated a strong ability to produce accurate segmentation masks for both cracks and drywall joints, correctly responding to the different text prompts. As seen in the validation output, the raw predicted masks are grayscale probability maps and are not as clean as the binary ground-truth masks.

Comparison of original images, ground truth masks, and the model's predicted output masks from the validation set.

Quantitative Evaluation

To provide a complete picture of model performance, a comprehensive evaluation system was implemented to calculate key segmentation metrics (mIoU, Dice Coefficient, Pixel Accuracy) for the entire validation set.

This system provides a detailed breakdown of performance, including overall metrics, category-specific results (cracks vs. drywall joints), and a comparison of raw vs. post-processed predictions.

📊 EVALUATION RESULTS
==================================================
Overall (403 samples):
   mIoU:        0.5488 ± 0.1740
   mDice:       0.6903 ± 0.1659
   Pixel Acc:   0.9410 ± 0.0659

Cracks (201 samples):
   mIoU:        0.5069 ± 0.1820
   mDice:       0.6516 ± 0.1768
   Pixel Acc:   0.9628 ± 0.0317

Drywall Joints (202 samples):
   mIoU:        0.5905 ± 0.1547
   mDice:       0.7287 ± 0.1444
   Pixel Acc:   0.9193 ± 0.0820


🎯 PERFORMANCE SUMMARY
==============================
Best performing category by mIoU: Drywall Joints (0.5905)
Overall model performance:
   - mIoU: 0.5488 (Higher is better)
   - mDice: 0.6903 (Higher is better)
   - Pixel Acc: 0.9410 (Higher is better)
✅ Good segmentation performance (mIoU > 0.5)


A sample-by-sample analysis shows the effect of the post-processing pipeline:

Sample 1 (crack - 'structural crack'):
   Raw Prediction:       IoU=0.6738, Dice=0.8051, Acc=0.9377
   Processed Prediction: IoU=0.7189, Dice=0.8365, Acc=0.9445
   📈 Processing improved IoU by 0.0451

Sample 2 (crack - 'structural crack'):
   Raw Prediction:       IoU=0.3514, Dice=0.5201, Acc=0.9883
   Processed Prediction: IoU=0.3328, Dice=0.4994, Acc=0.9796
   📉 Processing reduced IoU by 0.0186

Sample 3 (crack - 'surface crack'):
   Raw Prediction:       IoU=0.6462, Dice=0.7851, Acc=0.8924
   Processed Prediction: IoU=0.7104, Dice=0.8306, Acc=0.9114
   📈 Processing improved IoU by 0.0642

Sample 4 (drywall_joint - 'segment drywall seam'):
   Raw Prediction:       IoU=0.7700, Dice=0.8700, Acc=0.9717
   Processed Prediction: IoU=0.7939, Dice=0.8851, Acc=0.9742
   📈 Processing improved IoU by 0.0239

Sample 5 (drywall_joint - 'segment taping area'):
   Raw Prediction:       IoU=0.6709, Dice=0.8031, Acc=0.9508
   Processed Prediction: IoU=0.6507, Dice=0.7884, Acc=0.9436
   📉 Processing reduced IoU by 0.0202

Sample 6 (drywall_joint - 'wall joint line'):
   Raw Prediction:       IoU=0.6337, Dice=0.7758, Acc=0.9499
   Processed Prediction: IoU=0.6503, Dice=0.7881, Acc=0.9506
   📈 Processing improved IoU by 0.0166


The results indicate that the model performs better on drywall joints (mIoU 0.59) than on cracks (mIoU 0.51), which is interesting given the training data imbalance. The post-processing pipeline generally improves the IoU, though it can occasionally reduce performance on already difficult-to-segment samples.

Training Footprint

Total parameters: 150,747,746
Trainable parameters: 150,747,746
Training time: 157.8 minutes (2.6 hours)
Memory usage during training: 1.59 GB
Model file size on disk: 583.88 MB
Average inference time per image: 241.04 ms








Conclusion

This project successfully demonstrates that a fine-tuned CLIPSeg model can perform text-conditioned segmentation for drywall quality assurance.

The key to success was unfreezing the entire model and applying discriminative learning rates. This strategy allowed the powerful, pre-trained encoder to adapt to our specific domain (drywall textures) while preventing catastrophic forgetting of its generalized knowledge.

The model is flexible, handling both crack (polygon) and joint (bounding box) segmentation from a single set of weights. The raw outputs are probability maps which can be refined using a post-processing pipeline (denoising, Otsu's thresholding, and morphological operations) to create clean, binary masks. The quantitative evaluation, with an overall mIoU of 0.55, confirms this is a viable approach.

The training imbalance is a concern, yet the model performed better on the under-represented drywall joints class. This suggests the distinct, boxed-shape annotations for joints may have been easier for the model to learn than the complex, polygon-based crack annotations.

A primary limitation is the model's internal 352x352 processing resolution. While the output is upsampled, this low internal resolution limits the fine-grained detail it can capture, which is a likely factor in the lower performance on hairline cracks.

Future work should focus on data augmentation to balance the training set and explore techniques to process images at a higher resolution to further improve segmentation quality.