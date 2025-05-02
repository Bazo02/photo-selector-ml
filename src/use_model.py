import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
from train import PairwiseNet  # assumes use_model.py lives alongside train.py

def load_image(path, device):
    """Load an image, apply the same transforms you used in training, and move to device."""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open(path).convert("RGB")
    return preprocess(img).unsqueeze(0).to(device)  # add batch dim

def main(img1_path, img2_path, model_path="pairwise_model.pth"):
    # 1. Set up device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PairwiseNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load and preprocess the two images
    img1 = load_image(img1_path, device)
    img2 = load_image(img2_path, device)

    # 3. Run inference
    with torch.no_grad():
        logits = model(img1, img2)
        prob = torch.sigmoid(logits).item()  # probability that img1 is sharper than img2

    # 4. Interpret result
    print(f"Probability image 1 is sharper: {prob:.3f}")
    if prob > 0.5:
        print(f"🖼️  {img1_path} is predicted better than {img2_path}")
    else:
        print(f"🖼️  {img2_path} is predicted better than {img1_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python use_model.py path/to/image1.jpg path/to/image2.jpg")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
