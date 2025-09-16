import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision import transforms
from tqdm import tqdm
from model import VariationalAutoEncoder
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMPUT_DIM = None
H_DIM = None
Z_DIM = 128
BATCH_SIZE = 2048
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

# Dataset Loading 
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(128),
    transforms.ToTensor(),
])


# dataset = datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
dataset = datasets.CelebA(root='../dataset', split='train', download=True, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# Model and optimizer
model = VariationalAutoEncoder(z_dim=Z_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
reconstruction_loss_fn = nn.BCELoss(reduction='sum') # per-pixel Binary Cross Entropy Loss

# Training Loop
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))

    for i, (x, _) in loop:
        # Forward pass (keep spatial dims for conv net)
        x = x.to(DEVICE)
        x_reconstructed, mu, log_var = model(x)

        # Compute Loss
        recon_loss = reconstruction_loss_fn(x_reconstructed, x)

        # Compute KL Divergence (using sigma) = - 1/2 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # kl_divergence = -0.5* torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Backpropagation and Optimization
        # Loss = -ELBO = reconstruction_loss + KL Divergence
        total_loss = recon_loss + kl_divergence

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loop.set_description(f'Epoch [{epoch+1}/{NUM_EPOCHS}]')
        loop.set_postfix(recon_loss=recon_loss.item()/BATCH_SIZE, kl_divergence=kl_divergence.item()/BATCH_SIZE, loss=total_loss.item()/BATCH_SIZE)


import random
def inference(model, idx_or_n=0, by_class=True, num_examples=10):
    """
    If by_class=True: original behavior (for class datasets)
    If by_class=False: reconstruct and sample random faces from CelebA (no classes)
    """
    if by_class:
        images = {}
        for x, y in dataset:
            if y not in images:
                images[y] = x
            if len(images) >= 10:
                break
        encodings = []
        for d in range(10):
            with torch.no_grad():
                img = images[d].unsqueeze(0).to(DEVICE)
                mu, log_var = model.encode(img)
                encodings.append((mu, log_var))
        mu, log_var = encodings[idx_or_n]
        for example in range(num_examples):
            sigma = torch.sqrt(torch.exp(log_var))
            eps = torch.randn_like(sigma)
            z = mu + eps * sigma
            x_reconstructed = model.decode(z)
            save_image(x_reconstructed, f'results/class_{idx_or_n}_example_{example}.png')
    else:
        # Reconstruct random faces
        for i in range(num_examples):
            idx = random.randint(0, len(dataset)-1)
            img, _ = dataset[idx]
            with torch.no_grad():
                img = img.unsqueeze(0).to(DEVICE)
                x_reconstructed, mu, log_var = model(img)
                save_image(x_reconstructed, f'results/celebA_recon_{i}.png')
        # Generate random faces from latent space
        z_dim = model.fc_mu.out_features
        for i in range(num_examples):
            with torch.no_grad():
                z = torch.randn(1, z_dim).to(DEVICE)
                x_generated = model.decode(z)
                save_image(x_generated, f'results/celebA_random_{i}.png')

if __name__ == '__main__':
    # Uncomment for class datasets (e.g., MNIST, CIFAR-10)
    # for idx in range(10):
    #     inference(model, idx, by_class=True)

    # Run CelebA-style inference (random reconstructions and generations)
    inference(model, idx_or_n=0, by_class=False, num_examples=10)

    # Save final trained model
    torch.save(model.state_dict(), 'vae_celeba_final.pth')
    print('Model saved to vae_celeba_final.pth')