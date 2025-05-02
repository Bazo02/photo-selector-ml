
# Photo Selector or Creator ML
This project explores using machine learning and deep learning models to automatically select the best photo from a set of similar images and generate an enhanced version using GANs.

##  Goal
- Evaluate image quality using metrics such as contrast, brightness, sharpness, and composition.
- Use ML/DL models (e.g., ResNet-50, CNNs) to score or classify image quality.
- Use GANs to generate enhanced or composite images.


##  Methods
- ResNet-50 for classification
- GANs for image enhancement
- Synthetic data and real-world datasets like AVA (Aesthetic Visual Analysis)

##  Project Structure
```
photo-selector-ml/
├── Project_4630/                   # Data folders
│   ├── Enhanced/                   # Output folder for enhanced images
│   ├── Raw_Image_Data/             # Folder with raw input images
│   └── Synthetic_Data/             # Auto-generated data for training/evaluation
│       ├── Images/                 # Synthetic image pairs
│       └── pairwise_labels.json    # JSON file with pairwise preferences/labels
├── notebooks/                      # Jupyter notebooks (e.g., Colab)  
├── src/                            # Core Python scripts  
│   ├── preprocess.py               # Resizes, normalizes, or transforms images before modeling
│   ├── syntetic_data_creator.py    #Generates labeled image pairs to simulate training data
│   │── image_pair_dataset.py       #Prepares image pair datasets for training or scoring
│   │── 
│   │── train.py
│   ├── scorer.py                   # Trains ML models (e.g., ResNet, CNN) using labeled image pairs
│   └── enhancer.py                 # Uses GAN or image processing to enhance image quality
├── requirements.txt                # Python dependencies  
├── README.md                       # Project info  
├── LICENSE                         # License information
└── .gitignore                      # Ignored files/folders
```

##  Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Bazo02/photo-selector-ml.git
   ```

2. **Navigate to the repository**  
   ```bash
   cd photo-selector-ml
   ```

3. **Create a virtual environment**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   ```

   **Windows:**  
   ```bash
   .venv\Scripts\activate
   ```

4. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   # or if needed:
   pip3 install -r requirements.txt
   ```

5. **(Optional) Register Jupyter kernel and run the notebook**  
   ```bash
   python3 -m ipykernel install --user --name=photo-selector-env
   ```

   Then open `notebooks/first_colab_notebook.ipynb` and select the kernel you just created.

## Authors
- Alexander Bazo ([Bazo02](https://github.com/Bazo02))
- Kimberly Crimson ([kimmicode](https://github.com/kimmicode))
- Espen Fodstad ([esfod](https://github.com/esfod))

```