import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.load_dataset import DataAugmentedESC50Dataset
from utils.models import get_model
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
TRAIN_CSV_PATH = "data/metadata/log-mel_spectrograms_no_aug.csv"  
TRAIN_SPEC_DIR = "data/log-mel_spectrograms/no_aug" 
"""VAL PATHS consistent regardless of augmentation/model since validation is always on original data"""
VAL_CSV_PATH = "data/metadata/log-mel_spectrograms_no_aug.csv"
VAL_SPEC_DIR = "data/log-mel_spectrograms/no_aug" 
MODEL_NAME = "mobilenet_v3" 
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "../saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- EARLY STOPPING CONFIG ---
EARLY_STOPPING_PATIENCE = 7   # Stop if no improvement after 7 epochs
EARLY_STOPPING_MIN_DELTA = 0.1 # Improvement must be at least 1% to count

def train_one_fold(fold_idx, train_folds, val_folds):
    print(f"\n{'='*20} Starting Fold {fold_idx} {'='*20}")
    print(f"Train Folds: {train_folds} | Val Folds: {val_folds}")

    # 1. Prepare Datasets & Loaders for THIS fold
    train_dataset = DataAugmentedESC50Dataset(
        csv_file=TRAIN_CSV_PATH,
        root_dir=TRAIN_SPEC_DIR,
        folds=train_folds
    )
    
    val_dataset = DataAugmentedESC50Dataset(
        csv_file=VAL_CSV_PATH,
        root_dir=VAL_SPEC_DIR,
        folds=val_folds
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Re-Initialize Model (CRITICAL: Must be fresh for every fold)
    model = get_model(MODEL_NAME, num_classes=50) # ESC-50 has 50 classes
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # History for this specific fold
    fold_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    best_acc = 0.0
    early_stopping_counter = 0

    # 3. Training Loop
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in tqdm(train_loader, desc=f"Fold {fold_idx} Ep {epoch+1}/{EPOCHS}", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        ep_train_loss = running_loss / len(train_loader)
        ep_train_acc = 100 * correct_train / total_train

        # --- VALIDATION ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        ep_val_loss = running_val_loss / len(val_loader)
        ep_val_acc = 100 * correct_val / total_val

        # Update Scheduler
        scheduler.step(ep_val_loss)

        # Store Metrics
        fold_history['train_loss'].append(ep_train_loss)
        fold_history['train_acc'].append(ep_train_acc)
        fold_history['val_loss'].append(ep_val_loss)
        fold_history['val_acc'].append(ep_val_acc)

        print(f"   Ep {epoch+1}: T-Acc: {ep_train_acc:.2f}% | V-Acc: {ep_val_acc:.2f}% | V-Loss: {ep_val_loss:.4f}")

        # --- CHECKPOINTING (Best Model for this Fold) ---
        if ep_val_acc > (best_acc + EARLY_STOPPING_MIN_DELTA):
            best_acc = ep_val_acc
            early_stopping_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{MODEL_NAME}_fold{fold_idx}_best.pth"))
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print(f"   Early stopping at epoch {epoch+1}")
                break # Stop this fold
    
    return fold_history, best_acc

def run_cross_validation():
    all_folds_history = []
    fold_accuracies = []

    # ESC-50 has 5 folds: [1, 2, 3, 4, 5]
    all_folds = [1, 2, 3, 4, 5]

    for i in range(5):
        val_folds = [all_folds[i]]
        # Train folds are everyone else
        train_folds = [x for x in all_folds if x not in val_folds]
        
        # Run Training for this Fold
        history, best_acc = train_one_fold(i+1, train_folds, val_folds)
        
        all_folds_history.append(history)
        fold_accuracies.append(best_acc)
    
    print("\n" + "="*30)
    print("CROSS VALIDATION RESULTS")
    print("="*30)
    for i, acc in enumerate(fold_accuracies):
        print(f"Fold {i+1}: {acc:.2f}%")
    print(f"Average Accuracy: {np.mean(fold_accuracies):.2f}% (+/- {np.std(fold_accuracies):.2f})")

    plot_average_curves(all_folds_history)

def plot_average_curves(all_histories):
    # Determine the maximum length (in case early stopping made them different lengths)
    # Strategy: Pad shorter runs with their last value to make lengths equal
    max_len = max([len(h['train_loss']) for h in all_histories])
    
    # Helper to pad list
    def pad_metric(metric_list, target_len):
        last_val = metric_list[-1]
        return metric_list + [last_val] * (target_len - len(metric_list))

    # Aggregate Data
    metrics = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
    avg_data = {m: [] for m in metrics}

    for metric in metrics:
        # Create a matrix [5 folds x max_epochs]
        matrix = []
        for h in all_histories:
            matrix.append(pad_metric(h[metric], max_len))
        
        # Calculate Mean
        avg_data[metric] = np.mean(matrix, axis=0)

    # Plot
    epochs = range(1, max_len + 1)
    
    plt.figure(figsize=(14, 6))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, avg_data['train_loss'], label='Avg Train Loss', color='blue')
    plt.plot(epochs, avg_data['val_loss'], label='Avg Val Loss', color='orange')
    plt.title(f'{MODEL_NAME} - 5-Fold Avg Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, avg_data['train_acc'], label='Avg Train Acc', color='blue')
    plt.plot(epochs, avg_data['val_acc'], label='Avg Val Acc', color='orange')
    plt.title(f'{MODEL_NAME} - 5-Fold Avg Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_5fold_average.png")
    plt.savefig(save_path)
    print(f"Average training curves saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    run_cross_validation()