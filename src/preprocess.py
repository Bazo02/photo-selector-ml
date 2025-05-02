from PIL import Image  # Pillow: bildehåndtering
import torch  # PyTorch for tensor-operasjoner
import os  # Fil- og stioperasjoner

# Laster inn ett bilde og bruker en torchvision-transform
def load_and_preprocess(image_path, transform):
    # Sjekk at filen finnes
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Bilde ikke funnet: {image_path}")
    # Åpne bilde og konverter til RGB-kanaler
    image = Image.open(image_path).convert("RGB")
    # Anvend transformasjon (f.eks. Resize, ToTensor)
    return transform(image)

# Forbehandler en liste av bilder til en batch-tensor
def preprocess_batch(image_paths, transform):
    tensors = []
    for path in image_paths:
        # Last inn og forbered hvert bilde
        tensor = load_and_preprocess(path, transform)
        tensors.append(tensor)
    # Stakk alle bildene til én tensor [batch, C, H, W]
    return torch.stack(tensors)
