'''
The training pipeline for full GAN setup
Defines
 - Generator_v2
 - Discriminator_v2
 - Losses (incl. Perceptual Loss)
 - Training Loop
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import transforms
from torchvision.models import vgg16

from burst_dataset import BurstOnlyDataset
from discriminator_dataset import DiscriminatorDataset
from generator import ResNetBurstGenerator
from discriminator import BurstDiscriminator

# ------------------------
# 1. Hyperparameters
# ------------------------
num_epochs = 10
batch_size = 8
learning_rate = 2e-4
burst_size = 10
image_size = (256, 256)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# 2. Datasets and Loaders
# ------------------------
burst_dataset = BurstOnlyDataset("bursts_dir", burst_size=burst_size, image_size=image_size)
disc_dataset = DiscriminatorDataset("bursts_dir", "real_images_dir", burst_size=burst_size, image_size=image_size)

burst_loader = DataLoader(burst_dataset, batch_size=batch_size, shuffle=True)
disc_loader = DataLoader(disc_dataset, batch_size=batch_size, shuffle=True)

# ------------------------
# 3. Models
# ------------------------
generator = ResNetBurstGenerator(in_frames=burst_size).to(device)
discriminator = BurstDiscriminator(burst_size=burst_size).to(device)

# Perceptual loss model
vgg = vgg16(pretrained=True).features[:16].to(device).eval()  # relu_2_2 features
for param in vgg.parameters():
    param.requires_grad = False

def perceptual_loss(fake, real):
    fake_vgg = vgg(fake)
    real_vgg = vgg(real)
    return nn.functional.l1_loss(fake_vgg, real_vgg)

# ------------------------
# 4. Losses and Optimisers
# ------------------------
adversarial_loss = nn.BCELoss()
optim_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optim_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# ------------------------
# 5. Training Loop
# ------------------------
for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    total_g_loss, total_d_loss = 0.0, 0.0

    loop = tqdm(zip(burst_loader, disc_loader), total=min(len(burst_loader), len(disc_loader)), desc=f"Epoch {epoch+1}/{num_epochs}")
    for burst_batch, real_batch in loop:
        burst_batch = burst_batch.to(device)
        real_batch = real_batch[1].to(device)  # take the real image only

        # === Train Discriminator ===
        fake_imgs = generator(burst_batch).detach()
        pred_real = discriminator(burst_batch, real_batch)
        pred_fake = discriminator(burst_batch, fake_imgs)

        real_labels = torch.ones_like(pred_real)
        fake_labels = torch.zeros_like(pred_fake)

        loss_d_real = adversarial_loss(pred_real, real_labels)
        loss_d_fake = adversarial_loss(pred_fake, fake_labels)
        loss_d = 0.5 * (loss_d_real + loss_d_fake)

        optim_D.zero_grad()
        loss_d.backward()
        optim_D.step()

        # === Train Generator ===
        fake_imgs = generator(burst_batch)
        pred_fake = discriminator(burst_batch, fake_imgs)

        loss_g_adv = adversarial_loss(pred_fake, real_labels)
        loss_g_perc = perceptual_loss(fake_imgs, real_batch)
        loss_g = loss_g_adv + 10.0 * loss_g_perc

        optim_G.zero_grad()
        loss_g.backward()
        optim_G.step()

        total_d_loss += loss_d.item()
        total_g_loss += loss_g.item()

        loop.set_postfix(G_Loss=total_g_loss / (loop.n + 1), D_Loss=total_d_loss / (loop.n + 1))

    print(f"\n📘 Epoch {epoch+1}/{num_epochs} | Gen Loss: {total_g_loss:.4f} | Disc Loss: {total_d_loss:.4f}")

# ------------------------
# 6. Generator Evaluation and Visual Preview
# ------------------------
generator.eval()
sample_output_dir = "output_dir/Video Project 14/Video Project 14_burst_00"

frame_tensors = []
for i in range(10):
    frame_path = os.path.join(sample_output_dir, f"frame_{i:02d}.jpg")
    if os.path.exists(frame_path):
        img = Image.open(frame_path).convert("RGB").resize((256, 256))
        img_tensor = transforms.ToTensor()(img)
        frame_tensors.append(img_tensor)

if not frame_tensors:
    print("⚠️ No frames found in the specified directory.")
else:
    grid = make_grid(frame_tensors, nrow=10, padding=2)
    plt.figure(figsize=(20, 4))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title("Example Burst Frames (Pre-Generator Input)")
    plt.show()

# ------------------------
# 7. Save Final Trained Models
# ------------------------
os.makedirs("trained_models_dir", exist_ok=True)
torch.save(generator.state_dict(), 'trained_models_dir/final_GAN_v2_generator.pth')
torch.save(discriminator.state_dict(), 'trained_models_dir/final_GAN_v2_discriminator.pth')
