import os
import sys
import time
import zipfile
import shutil
import json
import random
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
def train_test_split(paths, labels, test_size, stratify=None, random_state=42):
    """
    Stratified split helper to remove sklearn dependency.
    """
    rng = random.Random(random_state)
    
    if stratify is None:
        indices = list(range(len(paths)))
        rng.shuffle(indices)
        split_idx = int(len(paths) * (1 - test_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
    else:
        classes = set(stratify)
        class_indices = {c: [] for c in classes}
        for i, label in enumerate(stratify):
            class_indices[label].append(i)
            
        train_indices = []
        test_indices = []
        
        for c, idxs in class_indices.items():
            shuffled_idxs = list(idxs)
            rng.shuffle(shuffled_idxs)
            split_idx = int(len(shuffled_idxs) * (1 - test_size))
            train_indices.extend(shuffled_idxs[:split_idx])
            test_indices.extend(shuffled_idxs[split_idx:])
            
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)
        
    train_paths = [paths[i] for i in train_indices]
    test_paths = [paths[i] for i in test_indices]
    train_labels = [labels[i] for i in train_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return train_paths, test_paths, train_labels, test_labels
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Detect if running in Google Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Dataset extraction helpers (Colab / Local)

def setup_dataset(drive_zip_path='/content/drive/MyDrive/chest_xray.zip', local_zip_path='/content/chest_xray.zip', extract_dir=None):
    """
    Prepares the dataset.
    - If in Colab: mounts Google Drive, copies the ZIP file locally to Colab's NVMe drive, and unzips.
    - If running locally: checks for a local zip file or raw folder and extracts it.
    """
    global IN_COLAB
    
    if extract_dir is None:
        extract_dir = '/content/chest_xray_raw' if IN_COLAB else 'chest_xray_raw'

    if IN_COLAB:
        # Mount drive
        if not os.path.exists('/content/drive'):
            print("Mounting Google Drive...")
            from google.colab import drive
            drive.mount('/content/drive')
        
        # Locate ZIP file
        if not os.path.exists(drive_zip_path):
            common_paths = [
                '/content/drive/MyDrive/chest_xray.zip',
                '/content/drive/MyDrive/New folder/chest_xray.zip',
                '/content/drive/MyDrive/Colab Notebooks/chest_xray.zip'
            ]
            found = False
            for path in common_paths:
                if os.path.exists(path):
                    print(f"Found dataset zip file at path: {path}")
                    drive_zip_path = path
                    found = True
                    break
            if not found:
                raise FileNotFoundError(
                    f"Could not find chest_xray.zip at {drive_zip_path}. "
                    "Please upload the zip file to your Google Drive and specify the correct path."
                )
                
        # Copy to fast local disk in Colab
        if not os.path.exists(local_zip_path):
            print(f"Copying dataset from {drive_zip_path} to local storage {local_zip_path}...")
            start_time = time.time()
            shutil.copy2(drive_zip_path, local_zip_path)
            print(f"Copy finished in {time.time() - start_time:.2f} seconds.")
        else:
            print(f"Local zip already exists at {local_zip_path}.")

        # Extract archive
        if not os.path.exists(extract_dir):
            print(f"Extracting dataset to {extract_dir}...")
            start_time = time.time()
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Extraction finished in {time.time() - start_time:.2f} seconds.")
        else:
            print(f"Dataset already extracted at {extract_dir}.")
            
    else:
        # Local environment checks
        print("Running locally. Checking local environment...")
        local_zip_candidates = [
            'chest_xray.zip',
            '../chest_xray.zip',
            os.path.join(os.getcwd(), 'chest_xray.zip')
        ]
        
        # Use existing raw folder if it exists
        if os.path.exists('chest_xray_raw'):
            extract_dir = 'chest_xray_raw'
            print(f"Found existing raw directory: {extract_dir}")
        else:
            zip_found = None
            for cand in local_zip_candidates:
                if os.path.exists(cand):
                    zip_found = cand
                    break
            if zip_found:
                print(f"Found local zip file at {zip_found}. Extracting to local 'chest_xray_raw'...")
                extract_dir = 'chest_xray_raw'
                if not os.path.exists(extract_dir):
                    with zipfile.ZipFile(zip_found, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
            else:
                # Check for an already resized version before erroring
                if os.path.exists('chest_xray_resized'):
                    print("Found existing resized dataset folder. Skipping extraction.")
                    return 'chest_xray_resized'
                raise FileNotFoundError(
                    "Could not find local chest_xray.zip or chest_xray_raw folder. "
                    "Please place chest_xray.zip in the current directory."
                )

    # Find the nested folder containing train/test splits (filtering macOS metadata)
    base_dir = None
    for root, dirs, files in os.walk(extract_dir):
        if '__MACOSX' in root:
            continue
        if 'train' in dirs and 'test' in dirs:
            base_dir = root
            break
            
    if base_dir is None:
        # Check top-level folder
        if 'train' in os.listdir(extract_dir) and 'test' in os.listdir(extract_dir):
            return extract_dir
        raise FileNotFoundError(f"Could not find 'train' and 'test' subfolders inside {extract_dir}.")
        
    print(f"Dataset root identified at: {base_dir}")
    return base_dir

# Parallel image resizing for faster training throughput

def resize_single_image(args):
    src_path, dst_path, size = args
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with Image.open(src_path) as img:
            img = img.convert('RGB')
            img_resized = img.resize(size, Image.Resampling.LANCZOS)
            img_resized.save(dst_path, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"Error resizing {src_path}: {e}")
        return False

def resize_dataset_locally(raw_base_dir, resized_dst_dir=None, size=(224, 224)):
    """
    Resizes all images in raw_base_dir to the specified size in parallel.
    """
    global IN_COLAB
    if resized_dst_dir is None:
        resized_dst_dir = '/content/chest_xray_resized' if IN_COLAB else 'chest_xray_resized'

    if os.path.exists(resized_dst_dir):
        # Verify existing resized structure
        train_path = os.path.join(resized_dst_dir, 'train')
        test_path = os.path.join(resized_dst_dir, 'test')
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"Resized dataset already exists at {resized_dst_dir}. Skipping resizing.")
            return resized_dst_dir

    print(f"Scanning raw images in: {raw_base_dir}")
    tasks = []
    
    for root, dirs, files in os.walk(raw_base_dir):
        for file in files:
            if file.startswith('.') or not file.lower().endswith(('.jpeg', '.jpg', '.png')):
                continue
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_path, raw_base_dir)
            dst_path = os.path.join(resized_dst_dir, rel_path)
            tasks.append((src_path, dst_path, size))
            
    total_images = len(tasks)
    print(f"Found {total_images} images to resize.")
    
    if total_images == 0:
        # Check if raw_base_dir is already formatted/resized
        train_path = os.path.join(raw_base_dir, 'train')
        test_path = os.path.join(raw_base_dir, 'test')
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"No images to resize, folder already resized: {raw_base_dir}")
            return raw_base_dir
        raise ValueError("No raw images found to resize.")
        
    print(f"Starting parallel resizing using 8 CPU workers...")
    start_time = time.time()
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(resize_single_image, tasks)
        for i, res in enumerate(results):
            if res:
                success_count += 1
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1} / {total_images} images...")
                
    elapsed = time.time() - start_time
    print(f"Resizing finished! Successfully resized {success_count} / {total_images} images in {elapsed:.2f}s.")
    return resized_dst_dir

# Dataset & Data Loaders

class PneumoniaDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_image_files_and_labels(base_dir):
    image_paths = []
    labels = []
    categories = {'NORMAL': 0, 'PNEUMONIA': 1}
    
    for split in ['train', 'test']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for cat_name, cat_label in categories.items():
            cat_dir = os.path.join(split_dir, cat_name)
            if not os.path.exists(cat_dir):
                continue
                
            for filename in os.listdir(cat_dir):
                if filename.startswith('.') or not filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                    continue
                file_path = os.path.join(cat_dir, filename)
                image_paths.append(file_path)
                labels.append(cat_label)
                
    return image_paths, labels

def get_data_loaders(base_dir, batch_size=32, use_balanced_sampler=False, use_augmentation=False):
    paths, labels = get_image_files_and_labels(base_dir)
    print(f"Loaded: Normal: {labels.count(0)}, Pneumonia: {labels.count(1)}, Total: {len(paths)}")
    
    # Split: 70% train, 15% val, 15% test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels, test_size=0.15, stratify=labels, random_state=42
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=0.1765, stratify=train_val_labels, random_state=42
    )
    
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
        
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    
    train_dataset = PneumoniaDataset(train_paths, train_labels, train_transform)
    val_dataset = PneumoniaDataset(val_paths, val_labels, val_test_transform)
    test_dataset = PneumoniaDataset(test_paths, test_labels, val_test_transform)
    
    sampler = None
    if use_balanced_sampler:
        num_normal = train_labels.count(0)
        num_pneumonia = train_labels.count(1)
        total_train = len(train_labels)
        class_weights = [total_train / num_normal, total_train / num_pneumonia]
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader

# Model Architectures

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.fc(x).view(b, c, 1, 1)
        return x * w

class CNNBaseline(nn.Module):
    def __init__(self):
        super(CNNBaseline, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CNNWithAttention(nn.Module):
    def __init__(self):
        super(CNNWithAttention, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SqueezeExcitation(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SqueezeExcitation(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SqueezeExcitation(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SqueezeExcitation(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Utilities & Evaluation Metrics

def calculate_metrics(y_true, y_pred):
    # Ensure inputs are list of ints
    y_true = [int(y) for y in y_true]
    y_pred = [int(y) for y in y_pred]
    
    total = len(y_true)
    if total == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'specificity': 0.0,
            'confusion_matrix': [[0, 0], [0, 0]]
        }
        
    tn = fp = fn = tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        elif yt == 1 and yp == 1:
            tp += 1
            
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
    }

def plot_curves(history, save_path, title_prefix):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-s', label='Val Loss')
    plt.title(f'{title_prefix} - Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-o', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-s', label='Val Acc')
    plt.title(f'{title_prefix} - Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved learning curves to: {save_path}")

# Training & Evaluation Functions

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = running_loss / total
    metrics['accuracy'] = correct / total
    return metrics

def run_single_experiment(model_type, data_dir, epochs=8, batch_size=32, lr=1e-3, 
                          use_weighted_loss=False, use_augmentation=False, prefix='experiment'):
    global IN_COLAB
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Starting Experiment: {prefix} on {device} ---")
    
    # Load data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        base_dir=data_dir, batch_size=batch_size, use_balanced_sampler=False, use_augmentation=use_augmentation
    )
    
    # Initialize the selected model
    if model_type == 'baseline_cnn':
        model = CNNBaseline()
    elif model_type == 'attention_cnn':
        model = CNNWithAttention()
    else:
        raise ValueError("Invalid model type.")
    model = model.to(device)
    
    # Loss function and optimizer setup
    if use_weighted_loss:
        # Weight by ~2.88x (imbalance ratio)
        weights = torch.tensor([2.88, 1.0], device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # Training Loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_val_f1 = 0.0
    checkpoint_dir = '/content/results' if IN_COLAB else 'results'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}_best_model.pth")
    
    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_score'])
        
        scheduler.step(val_metrics['loss'])
        
        print(f"Epoch [{epoch+1}/{epochs}] ({dt:.1f}s) | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']*100:.2f}% F1: {val_metrics['f1_score']*100:.2f}%")
        
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved new best checkpoint with Val F1={best_val_f1*100:.2f}%")
            
    # Evaluate model on the test set
    print("\nEvaluating on Test Set...")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print("="*40)
    print(f"TEST RESULTS - {prefix.upper()}")
    print(f"Accuracy:    {test_metrics['accuracy']*100:.2f}%")
    print(f"Precision:   {test_metrics['precision']*100:.2f}%")
    print(f"Recall:      {test_metrics['recall']*100:.2f}%")
    print(f"F1-Score:    {test_metrics['f1_score']*100:.2f}%")
    print(f"Specificity: {test_metrics['specificity']*100:.2f}%")
    print("="*40)
    
    # Save curves and log metrics
    plot_curves(history, os.path.join(checkpoint_dir, f"{prefix}_curves.png"), prefix)
    with open(os.path.join(checkpoint_dir, f"{prefix}_metrics.json"), 'w') as f:
        json.dump({'config': {'model': model_type, 'weighted': use_weighted_loss, 'augmented': use_augmentation},
                   'test_metrics': test_metrics, 'history': history}, f, indent=4)
                   
    return test_metrics, checkpoint_path

# Main Execution Pipeline

def main(drive_zip_path='/content/drive/MyDrive/chest_xray.zip', output_drive_dir='/content/drive/MyDrive/pneumonia_results'):
    """
    Main function to execute the full pipeline.
    """
    global IN_COLAB
    print("==================================================")
    print(f"Starting pipeline (Environment: {'Google Colab' if IN_COLAB else 'Local PC'})")
    print("==================================================")
    
    # Get and extract dataset
    try:
        raw_dataset_dir = setup_dataset(drive_zip_path)
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        if IN_COLAB:
            print("\n[INSTRUCTIONS]")
            print("1. Upload 'chest_xray.zip' to the root of your Google Drive.")
            print("2. Pass the path to main(), e.g. main(drive_zip_path='/content/drive/MyDrive/your_folder/chest_xray.zip')")
        else:
            print("\n[INSTRUCTIONS]")
            print("Please place 'chest_xray.zip' in the current working directory.")
        return
        
    # Run parallel resizing pipeline
    resized_data_dir = resize_dataset_locally(raw_dataset_dir)
    
    # Run the configured experiments
    experiments_config = [
        ('baseline_cnn', False, False, 'baseline_cnn'),
        ('baseline_cnn', True, True, 'baseline_cnn_augmented_weighted'),
        ('attention_cnn', True, True, 'attention_cnn_augmented_weighted')
    ]
    
    results = {}
    checkpoint_paths = []
    
    for model_type, weighted, augmented, prefix in experiments_config:
        metrics, ckpt = run_single_experiment(
            model_type=model_type,
            data_dir=resized_data_dir,
            epochs=8,
            use_weighted_loss=weighted,
            use_augmentation=augmented,
            prefix=prefix
        )
        results[prefix] = metrics
        checkpoint_paths.append(ckpt)
        
    # Build results comparison table
    report = []
    report.append("# Experiment Results Summary\n")
    report.append("| Model Paradigm | Test Accuracy | Test Precision | Test Recall (Sensitivity) | Test F1-Score | Test Specificity |\n")
    report.append("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
    for name, metrics in results.items():
        acc = metrics['accuracy'] * 100
        prec = metrics['precision'] * 100
        rec = metrics['recall'] * 100
        f1 = metrics['f1_score'] * 100
        spec = metrics['specificity'] * 100
        report.append(f"| {name} | {acc:.2f}% | {prec:.2f}% | {rec:.2f}% | {f1:.2f}% | {spec:.2f}% |\n")
        
    report_text = "".join(report)
    print("\n" + "="*40 + "\nFINAL EXPERIMENT COMPARISON\n" + "="*40)
    print(report_text)
    
    results_dir = '/content/results' if IN_COLAB else 'results'
    with open(os.path.join(results_dir, 'report.md'), 'w') as f:
        f.write(report_text)
        
    # Copy results back to Google Drive if running in Colab
    if IN_COLAB:
        os.makedirs(output_drive_dir, exist_ok=True)
        print(f"\nBacking up all results, weights, and curves to Google Drive at {output_drive_dir}...")
        for file in os.listdir(results_dir):
            src_file = os.path.join(results_dir, file)
            dst_file = os.path.join(output_drive_dir, file)
            if os.path.isdir(src_file):
                continue
            shutil.copy2(src_file, dst_file)
        print("Backup completed successfully! You can find the saved weights, report, and curves in your Google Drive.")
    else:
        print(f"\nPipeline finished locally! Results and weights are saved in the '{results_dir}/' directory.")

if __name__ == '__main__':
    main()
