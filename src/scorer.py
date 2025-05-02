import torch
import torch.nn as nn
import json
import os
import csv
from torchvision.models import resnet50, ResNet50_Weights
from preprocess import preprocess_batch

class ImageScorer:
    def __init__(self, device='cpu'):
        self.device = device
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)
        self.model.fc = nn.Identity()
        self.model.eval().to(self.device)
        self.preprocess = weights.transforms()

    def score_images(self, image_paths):
        images = preprocess_batch(image_paths, transform=self.preprocess).to(self.device)
        with torch.no_grad():
            features = self.model(images)
            scores = features.norm(dim=1)
        return scores.cpu().tolist()

if __name__ == "__main__":
    project_root = "/Users/alexbazo/Desktop/Project_4630"
    json_path = os.path.join(project_root, "/Users/alexbazo/Desktop/Project_4630/Synthetic_Data/pairwise_labels.json")
    base_dir = os.path.join(project_root, "Synthetic_Data", "Images")
    results_csv = os.path.join(project_root, "results.csv")

    scorer = ImageScorer()

    with open(json_path, "r") as f:
        data = json.load(f)

    results = []

    for i, item in enumerate(data):
        frame_a_path = os.path.join(base_dir, item["frame_a"])
        frame_b_path = os.path.join(base_dir, item["frame_b"])
        label = item["label"]

        scores = scorer.score_images([frame_a_path, frame_b_path])
        pred = int(scores[0] > scores[1])
        correct = (pred == label)

        print(f"[{i:03}] {item['frame_a']} vs {item['frame_b']} | Pred: {pred} | Label: {label} | Correct: {correct}")

        results.append({
            "index": i,
            "frame_a": item["frame_a"],
            "frame_b": item["frame_b"],
            "score_a": scores[0],
            "score_b": scores[1],
            "pred": pred,
            "label": label,
            "correct": correct
        })

    # Save to CSV
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    with open(results_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Results saved to {results_csv}")
