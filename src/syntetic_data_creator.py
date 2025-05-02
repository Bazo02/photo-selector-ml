# -*- coding: utf-8 -*-
"""
synthetic_data_creator.py (tilpasset Mac)

Generates synthetic burst sequences from all raw images (found anywhere under Raw_Image_Data),
creates degraded variants, and writes pairwise_labels.json.
"""

import random
import shutil
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

# === CONFIGURATION ===

ROOT_PATH    = Path('/Users/alexbazo/Desktop/Project_4630')
RAW_PATH     = ROOT_PATH / 'Raw_Image_Data'
SYN_PATH     = ROOT_PATH / 'Synthetic_Data'
IMAGES_PATH  = SYN_PATH / 'Images'
JSON_PATH    = SYN_PATH / 'pairwise_labels.json'

# Optional preview settings
N_PREVIEW    = 12
SHOW_PREVIEW = False

# How many variants per burst (including the original)
BURST_LENGTH = 5

# Supported image extensions
IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")

# === COLLECT ALL RAW IMAGES RECURSIVELY ===

all_raw = []
for ext in IMG_EXTS:
    all_raw.extend(RAW_PATH.rglob(ext))

if not all_raw:
    raise RuntimeError(f"No image files found under {RAW_PATH}")

if len(all_raw) < N_PREVIEW:
    raise RuntimeError(f"Only {len(all_raw)} images found; need at least {N_PREVIEW} to preview")

# Preview a random sample if desired
if SHOW_PREVIEW:
    sample = random.sample(all_raw, N_PREVIEW)
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10), constrained_layout=True)
    for ax, img_path in zip(axes.flatten(), sample):
        ax.imshow(Image.open(img_path))
        ax.set_title(str(img_path.relative_to(RAW_PATH)), fontsize=8)
        ax.axis("off")
    plt.show()

# === CLEAN OUTPUT FOLDER ===

def clear_folder(folder: Path):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

clear_folder(IMAGES_PATH)

# === EFFECT FUNCTIONS ===

def add_motion_blur(img, degree=10, angle=0):
    M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
    k = np.diag(np.ones(degree))
    k = cv2.warpAffine(k, M, (degree, degree))
    k = k / degree
    return cv2.filter2D(img, -1, k)

def add_gaussian_blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 2)

def add_noise(img):
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def change_exposure(img, factor=0.5):
    enhancer = ImageEnhance.Brightness(Image.fromarray(img))
    return np.array(enhancer.enhance(factor))

# === CREATE BURST SEQUENCES ===

for idx, raw_path in enumerate(sorted(all_raw), start=1):
    subfolder = IMAGES_PATH / f"{idx:03d}"
    subfolder.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(raw_path))
    if img is None:
        print(f"⚠️  Cannot read {raw_path}, skipping")
        continue

    stem = raw_path.stem
    sharp_file = subfolder / f"{stem}_sharp.jpg"
    cv2.imwrite(str(sharp_file), img)

    for i in range(1, BURST_LENGTH):
        variant = img.copy()
        choice = random.choice(['motion', 'gauss', 'noise', 'exposure'])
        if choice == 'motion':
            variant = add_motion_blur(variant,
                                      degree=random.randint(5, 15),
                                      angle=random.randint(0, 180))
        elif choice == 'gauss':
            variant = add_gaussian_blur(variant, ksize=random.choice([3,5,7]))
        elif choice == 'noise':
            variant = add_noise(variant)
        else:
            variant = change_exposure(variant, factor=random.uniform(0.4, 1.2))

        degraded_file = subfolder / f"{stem}_degraded_{i}.jpg"
        cv2.imwrite(str(degraded_file), variant)

# === GENERATE PAIRWISE LABELS ===

pairs = []
for burst_folder in sorted(IMAGES_PATH.iterdir()):
    if not burst_folder.is_dir():
        continue

    images = sorted(burst_folder.glob('*.jpg'))
    sharp = next((f for f in images if f.name.endswith("_sharp.jpg")), None)
    degraded = [f for f in images if "_degraded_" in f.name]

    if not sharp or not degraded:
        print(f"⚠️  Skipping {burst_folder.name} (missing sharp or degraded)")
        continue

    sharp_rel = str(sharp.relative_to(IMAGES_PATH))
    for d in degraded:
        pairs.append({
            "burst": burst_folder.name,
            "frame_a": sharp_rel,
            "frame_b": str(d.relative_to(IMAGES_PATH)),
            "label": 1
        })

SYN_PATH.mkdir(parents=True, exist_ok=True)
with open(JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(pairs, f, indent=2)

print(f"✅ Generated {len(pairs)} pairs and wrote to {JSON_PATH}")
