# PyTorch Variational Autoencoder (VAE) Implementation

A comprehensive PyTorch implementation of Variational Autoencoders (VAE) for image generation, featuring a convolutional encoder-decoder architecture with support for various image datasets.

## Overview

This project implements a complete VAE pipeline that learns meaningful latent representations of images by optimizing the Evidence Lower BOund (ELBO). The model consists of an encoder that maps images to a latent space and a decoder that reconstructs images from latent codes, enabling both reconstruction and generation of new samples.

## Key Features

- **Convolutional VAE Architecture** with encoder-decoder structure
- **Reparameterization Trick** for backpropagation through stochastic sampling
- **ELBO Optimization** combining reconstruction loss and KL divergence
- **Flexible Image Support** adaptable to different resolutions and channels
- **Latent Space Generation** for creating new samples and interpolations


## Variational Inference Framework

- **Encoder**: Maps images x to latent distributions q_φ(z|x)
- **Decoder**: Reconstructs images from latent codes p_θ(x|z)
- **Prior**: Standard Gaussian p(z) = N(0, I) in latent space

## Mathematical Foundation

### Evidence Lower Bound (ELBO)
The VAE maximizes the ELBO as a tractable approximation to the log-likelihood:

$$\log p(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - \text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

### Reparameterization Trick
Enables gradient flow through stochastic sampling:

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \text{where } \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

### Loss Function
**Reconstruction Loss (Binary Cross-Entropy)**
$$\mathcal{L}_{\text{recon}} = \text{BCE}(\mathbf{x}, \hat{\mathbf{x}})$$

**KL Divergence (Closed-form for Gaussians)**
$$\mathcal{L}_{\text{KL}} = -\frac{1}{2}  \sum\_{j=1}^{J} \left(1 + \log (\sigma_j^2) - \mu_j^2 - \sigma_j^2\right)$$

**Total Loss**
$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}\_{\text{KL}}$$

## Architecture Details

### Encoder Network
- **Input Processing**: Convolutional layers with ReLU activations
- **Feature Extraction**: Conv2d layers with progressive downsampling
- **Latent Mapping**: Fully connected layers outputting μ and log σ²

### Decoder Network
- **Latent Projection**: Linear layer expanding latent codes to feature maps
- **Upsampling**: Transposed convolutions for spatial reconstruction
- **Output Generation**: Sigmoid activation for pixel values in [0,1]

### Core Implementation
```python
def forward(self, x):
    mu, log_var = self.encode(x)
    sigma = torch.sqrt(torch.exp(log_var))  # σ = sqrt(exp(log σ²))
    eps = torch.randn_like(sigma)           # ε ~ N(0,I)
    z = mu + eps * sigma                    # Reparameterization trick
    x_recon = self.decode(z)
    return x_recon, mu, log_var
```

## Training Process

1. **Forward Pass**: Encode input to latent distribution parameters
2. **Sampling**: Apply reparameterization trick to sample latent codes
3. **Reconstruction**: Decode latent codes back to image space
4. **Loss Computation**: Calculate reconstruction loss + KL divergence
5. **Optimization**: Backpropagate and update parameters

## Results

- **Image Reconstruction**: High-fidelity reconstruction of input images
- **Latent Space Generation**: Novel sample generation via prior sampling
- **Smooth Interpolation**: Continuous transitions between different samples

## Technical Highlights

- **Numerical Stability** using log-variance parameterization
- **Convolutional Architecture** preserving spatial structure
- **Closed-form KL Divergence** for Gaussian distributions
- **Flexible Resolution Support** through adaptive architecture design

## References

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Kingma & Welling
