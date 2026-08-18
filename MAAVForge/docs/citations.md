# Project Citations & References

The MAAVForge architecture and methodology for the Semiconductor Hackathon are built upon the following peer-reviewed research papers. These foundational works govern our architecture choice, upsampling methodology, and loss functions.

## 1. Core Architecture: NAFNet
Our primary feature extraction backbone is based on the Nonlinear Activation Free Network (NAFNet), which demonstrates that complex activation functions are unnecessary for state-of-the-art image restoration.

> **Chen, L., Chu, X., Zhang, X., & Sun, J. (2022).** Simple baselines for image restoration. *European Conference on Computer Vision (ECCV)*. Cham: Springer Nature Switzerland.

## 2. Upsampling Module: PixelShuffle
To transform the 128x128 feature maps into the target 256x256 resolution without transposed-convolution checkerboard artifacts, we utilized the Efficient Sub-Pixel Convolutional Neural Network method.

> **Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A. P., Bishop, R., ... & Wang, Z. (2016).** Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. *Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)*.

## 3. Loss Metric: Focal Frequency Loss (FFL)
To prevent the neural network from suffering "spectral bias" (ignoring high-frequency semiconductor edges), we incorporated a dynamic Fourier-domain loss function.

> **Jiang, L., Dai, B., Wu, W., & Loy, C. C. (2021).** Focal frequency loss for image reconstruction and synthesis. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.

## 4. Loss Metric: Structural Similarity (SSIM)
To train the model on human-perceptual geometry rather than raw pixel differences, we incorporated the Structural Similarity Index as a core component of the loss function.

> **Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).** Image quality assessment: from error visibility to structural similarity. *IEEE transactions on image processing*, 13(4), 600-612.
