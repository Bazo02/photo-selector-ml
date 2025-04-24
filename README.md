
# Photo Selector or Creator ML

This project explores using machine learning and deep learning models to automatically select the best photo from a set of similar images — or generate an enhanced version using GANs.

##  Goal
- Evaluate image quality using metrics such as contrast, brightness, sharpness, and composition.
- Use ML/DL models (e.g., ResNet-50, CNNs) to score or classify image quality.
- Optionally use GANs to generate enhanced or composite images.

##  Methods
- ResNet-50 for classification
- GANs for enhancement
- Synthetic data and real-world datasets like AVA (Aesthetic Visual Analysis)

##  Project Structure
photo-selector-ml/
├── data/               # For sample image sets
├── notebooks/          # Jupyter notebooks (e.g., Colab)
├── src/                # Core Python scripts
│   ├── preprocess.py
│   ├── scorer.py
│   └── enhancer.py
├── requirements.txt    # Python dependencies
├── README.md           # Project info
└── .gitignore          # Ignored files/folders

# Install dependencies

pip install -r requirements.txt
eller 
pip3 install -r requirements.txt



## 🧑 Authors
- Alexander Bazo (@Bazo02)
- Kimberly Crimson (@kimmicode)
- Espen Fodstad (@esfod)
