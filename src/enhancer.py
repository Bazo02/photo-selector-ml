import cv2
import os
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def enhance_image(image_path, output_path=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Image not found: {image_path}")
        return

    sharpening_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    enhanced = cv2.filter2D(img, -1, sharpening_kernel)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    enhanced_image = Image.fromarray(enhanced_rgb)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        enhanced_image.save(output_path)
        print(f"✅ Enhanced: {output_path}")
    else:
        return enhanced_image

def process_image_pair(input_output_tuple):
    input_path, output_path = input_output_tuple
    enhance_image(input_path, output_path=output_path)

if __name__ == "__main__":
    base_dirs = [
        "/Users/alexbazo/Desktop/Project_4630/Raw_Image_Data",
        "/Users/alexbazo/Desktop/Project_4630/Synthetic_Data",
        "/Users/alexbazo/Desktop/Project_4630/Synthetic_Data/Images"
    ]
    output_base = "/Users/alexbazo/Desktop/Project_4630/Enhanced"
    image_paths = []

    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(".jpg"):
                    input_path = os.path.join(root, file)
                    relative_path = os.path.relpath(input_path, base_dir)
                    output_path = os.path.join(output_base, relative_path)
                    image_paths.append((input_path, output_path))

    # Use all available CPU cores for parallel processing
    with ProcessPoolExecutor() as executor:
        executor.map(process_image_pair, image_paths)
