import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid

from generator_v2 import ResNetBurstGenerator

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetBurstGenerator(in_frames=10).to(device)
model.load_state_dict(torch.load("generator_final_v2.pth", map_location=device))
model.eval()

# 2. Load burst frames
burst_dir = "output_dir/Video Project 14/Video Project 14_burst_00"
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

burst_frames = []
original_frames = []

for i in range(10):
    frame_path = os.path.join(burst_dir, f"frame_{i:02d}.jpg")
    if os.path.exists(frame_path):
        img = Image.open(frame_path).convert("RGB").resize((256, 256))
        original_frames.append(transforms.ToTensor()(img))
        burst_frames.append(transform(img))

if not burst_frames:
    print("⚠️ No burst frames found.")
    exit()

burst_tensor = torch.stack(burst_frames, dim=0).unsqueeze(0).to(device)  # [1, 10, 3, H, W]

# 3. Generate output
with torch.no_grad():
    generated = model(burst_tensor)[0].cpu()

# 4. Unnormalize
denorm = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.5]*3),
    transforms.Normalize(mean=[-0.5]*3, std=[1.]*3)
])
generated_img = denorm(generated).clamp(0, 1)

# 5. Visualisation
grid_burst = make_grid(original_frames, nrow=10, padding=2)
plt.figure(figsize=(20, 4))
plt.imshow(grid_burst.permute(1, 2, 0).numpy())
plt.axis("off")
plt.title("Input Burst Frames")
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(generated_img.permute(1, 2, 0).numpy())
plt.axis("off")
plt.title("Generated Scene")
plt.show()
