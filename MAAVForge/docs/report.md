# MAAVForge: AI-Based Restoration of Degraded Images for Semiconductor Inspection

## 1. Executive Summary
This report outlines the end-to-end technical approach taken by team **MAAVForge** for Problem Statement 1 (PS1) of the 2026 Semiconductor Hackathon. The objective was to restore degraded grayscale semiconductor images (128×128) affected by noise and low resolution, reconstructing them into clean, high-resolution (256×256) outputs. 

Instead of employing a traditional two-stage pipeline (e.g., denoise first, then upscale), we engineered a unified, frequency-aware Deep Learning architecture: **NAFNet-SR**. Our solution mathematically reconstructs the original microchip physics with high fidelity, operating efficiently with ~4.26 million parameters and achieving ~62 ms/image inference speeds on an NVIDIA T4 GPU.

---

## 2. Dataset and Preprocessing
### 2.1 The Data Challenge
The provided dataset contained pairs of semiconductor images:
*   **Input:** 128×128 pixel images suffering from significant multiplicative speckle noise and Gaussian-like degradation.
*   **Target (Ground Truth):** 256×256 pixel clean, high-resolution images.

The primary difficulty lay in the nature of semiconductor structures (e.g., repeating FinFET logic gates, metal interconnects, and vias). Traditional denoising algorithms (like Gaussian blur or Non-Local Means) tend to smooth out high-frequency edges, destroying the exact geometric tolerances that inspection tools rely on.

### 2.2 Dataset Splitting & Handling
We utilized a robust data-loading pipeline using PyTorch's `Dataset` and `DataLoader` classes.
*   **Total Images:** 3,200 `.npy` format arrays.
*   **Training Set:** 2,880 images (90%).
*   **Validation Set:** 320 images (10%).
*   **Data Augmentation:** To maximize the model's robustness and prevent overfitting, we applied random 90° rotations, horizontal flips, and vertical flips during training. This forced the network to learn structural physics rather than memorizing spatial pixel locations.

---

## 3. Model Architecture: Frequency-Aware NAFNet-SR
We designed a custom architecture based on the **Nonlinear Activation Free Network (NAFNet)**, augmented with a sub-pixel convolution upsampling module.

### 3.1 Why NAFNet?
Standard Convolutional Neural Networks (CNNs) rely heavily on nonlinear activation functions (ReLU, GELU, Sigmoid). Recent research demonstrated that these functions are not strictly necessary for state-of-the-art image restoration. NAFNet replaces complex activation functions with a computationally cheap **SimpleGate** (element-wise multiplication of feature maps split in the channel dimension). This allowed us to build a deep, powerful feature extractor that remains highly efficient for real-time inference.

### 3.2 The Encoder-Decoder Backbone
*   **Input Layer:** A 3×3 convolution projecting the 1-channel grayscale input into a 32-channel feature space.
*   **Encoder:** Consists of 3 hierarchical stages with 2, 2, and 4 NAFBlocks respectively. Spatial downsampling is achieved using strided convolutions, doubling the channel dimensions at each step to extract deep, semantic structural features.
*   **Bottleneck:** A dense block of 6 NAFBlocks operating at the lowest spatial resolution, capturing global context and macro-structures.
*   **Decoder:** Symmetrical to the encoder, with 4, 2, and 2 NAFBlocks. Spatial upsampling is performed using transposed convolutions to gradually rebuild the image resolution while concatenating skip connections from the encoder to preserve fine local details.

### 3.3 The Super-Resolution Head (PixelShuffle)
The raw NAFNet backbone operates purely in the 128×128 spatial domain. To achieve the mandatory 256×256 output, we appended an **Efficient Sub-Pixel Convolution (PixelShuffle)** layer at the output of the decoder. 
*   Unlike traditional transposed convolutions, which frequently cause "checkerboard" artifacts in the final image, PixelShuffle processes features in the low-resolution space and rearranges the channel dimension into spatial blocks to upscale by exactly 2×.
*   A **Global Bicubic Residual Connection** was added directly from the noisy input to the final output. This forces the heavy neural network to focus *only* on predicting the missing high-frequency details (the noise and the sharp edges), rather than wasting capacity trying to reconstruct the entire macro-image from scratch.

---

## 4. Loss Function Engineering
To train the model to respect the exact physics of the semiconductor patterns, we engineered a custom composite loss function containing three distinct mathematical terms:

`Total Loss = 1.0 * Charbonnier Loss + 0.1 * SSIM Loss + 0.05 * Focal Frequency Loss`

### 4.1 Charbonnier Loss (Weight: 1.0)
A robust, differentiable variant of L1 (Mean Absolute Error). Standard L2 (Mean Squared Error) heavily penalizes outliers, causing models to output blurry, "safe" averages. Charbonnier loss linearly penalizes errors, encouraging sharper edge reconstruction.

### 4.2 Structural Similarity Index (SSIM) Loss (Weight: 0.1)
While Charbonnier loss measures pixel-by-pixel accuracy, it ignores human visual perception and structural geometry. SSIM measures the correlation of luminance, contrast, and structure between the prediction and the ground truth. By maximizing SSIM, we explicitly train the model to preserve the sharp, rectangular nature of the logic gates and vias.

### 4.3 Focal Frequency Loss (FFL) (Weight: 0.05)
This is the core innovation of our pipeline. Neural networks suffer from "spectral bias"—they easily learn low-frequency shapes (backgrounds) but struggle with high-frequency details (microscopic edges). 
By computing the 2D Fast Fourier Transform (FFT) of both the prediction and the ground truth, FFL dynamically measures the gap in the frequency domain. It adaptively applies heavier loss penalties to the exact frequencies (usually high-frequency edges) that the model is struggling to reconstruct, preventing "smoothed out" predictions.

---

## 5. Training Methodology & Optimization
The model was trained on an NVIDIA T4 GPU for 100 epochs, taking approximately 1 hour and 45 minutes to converge.

*   **Optimizer:** `AdamW`, to prevent weight decay from destroying critical convolution filters.
*   **Learning Rate Scheduler:** Cosine Annealing, gradually decaying the learning rate from $1 \times 10^{-3}$ down to $1 \times 10^{-6}$ for smooth convergence in the final epochs.
*   **Mixed Precision (AMP):** We utilized PyTorch's Automatic Mixed Precision (FP16). This halved our VRAM usage and accelerated matrix multiplications on the Tensor Cores without sacrificing numerical stability.
*   **Gradient Clipping:** Capped at a norm of 1.0 to prevent exploding gradients during the volatile early phases of FFL calculation.

---

## 6. Inference and Deployment
To ensure maximum accuracy during the final automated judging, we implemented **Test-Time Augmentation (TTA)** in the `evaluate.py` script.

### 8-Fold Geometric TTA
When the script evaluates an unseen image, it does not process it once. It processes it 8 separate times:
1.  Original
2.  Rotated 90°, 180°, 270°
3.  Horizontally Flipped (Original, 90°, 180°, 270°)

The model predicts the clean image for all 8 variations. The predictions are then mathematically reverse-transformed back to the original orientation and averaged. This entirely eliminates directional bias in the convolution filters and dramatically smooths out any remaining high-frequency hallucination artifacts. 

Despite running 8 inferences per image, the lightweight nature of NAFNet keeps the total processing time at approximately **62 ms per image**, making it highly viable for real-time deployment in Applied Materials' industrial inspection tools.
