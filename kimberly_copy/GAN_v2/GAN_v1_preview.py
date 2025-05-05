import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from generator import ResNetBurstGenerator
from burst_dataset import BurstOnlyDataset

# ------------------------
# 1. Config & Setup
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "trained_models_dir/final_GAN_v1_generator.pth"
burst_dir = "bursts_dir"
burst_size = 10
image_size = (256, 256)

# ------------------------
# 2. Load Model
# ------------------------
model = ResNetBurstGenerator(in_frames=burst_size).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ------------------------
# 3. Load Dataset and Sample
# ------------------------
dataset = BurstOnlyDataset(burst_dir, burst_size=burst_size, image_size=image_size)
burst_tensor = dataset[0]  # Only burst is returned (shape: [10, 3, H, W])
input_burst = burst_tensor.unsqueeze(0).to(device)  # add batch dimension

# ------------------------
# 4. Generate Output
# ------------------------
with torch.no_grad():
    output = model(input_burst)[0].cpu()  # remove batch dimension
target = burst_tensor[0]  # frame_00 is the reference

# ------------------------
# 5. Visualise
# ------------------------
denorm = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.5] * 3),
    transforms.Normalize(mean=[-0.5] * 3, std=[1.] * 3)
])

def show_image(tensor, title):
    img = denorm(tensor).clamp(0, 1).permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
show_image(target, "Target (frame_00)")
plt.subplot(1, 2, 2)
show_image(output, "Generated Output")
plt.tight_layout()
plt.show()
