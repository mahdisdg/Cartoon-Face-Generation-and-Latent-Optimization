import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Target image size for the network
IMAGE_SIZE = 64

def preprocess():
    """
    Defines the image transformation pipeline.
    Resizes images and normalizes pixel values to [-1, 1] for GAN stability.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),                          
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

def parse_metadata(csv_path):
    """
    Reads the CSV associated with an image and converts attributes 
    into a flattened one-hot encoded vector.
    """
    try:
        # Load CSV (no header): Col 1 = current value, Col 2 = total variants
        df = pd.read_csv(csv_path, header=None, index_col=None)
        
        one_hot_list = []
        
        for index, row in df.iterrows():
            current_val = int(row[1])
            total_variants = int(row[2])
            
            # Handle potential index out of bounds errors in data
            if current_val >= total_variants:
                current_val = total_variants - 1
            
            # Create one-hot vector for this specific attribute
            vec = np.zeros(total_variants, dtype=np.float32)
            vec[current_val] = 1.0
            
            one_hot_list.extend(vec)
            
        return np.array(one_hot_list, dtype=np.float32)
        
    except Exception as e:
        print(f"Error reading metadata {csv_path}: {e}")
        # Return zero vector of approx length 217 (dataset specific) if parsing fails
        return np.zeros(217, dtype=np.float32)

class CartoonDataset(Dataset):
    """
    PyTorch Dataset class for loading CartoonSet images and their metadata.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = preprocess()
        self.image_paths = []
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Data directory not found: {root_dir}")
            
        print(f"Loading dataset from {root_dir}...")
        
        # recursively find all .png files in subdirectories
        for subdir in os.listdir(root_dir):
            path = os.path.join(root_dir, subdir)
            if os.path.isdir(path):
                files = glob.glob(os.path.join(path, "*.png"))
                self.image_paths.extend(files)
        
        print(f"Found {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Construct corresponding csv path
        csv_path = img_path.replace('.png', '.csv')
        
        # Read and process image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        # Read and process metadata
        metadata = parse_metadata(csv_path)
        metadata = torch.tensor(metadata, dtype=torch.float32)
        
        return image, metadata